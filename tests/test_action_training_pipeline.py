from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from action_training_pipeline import (
    DownloadedItem,
    build_adaptive_class_weight_multipliers,
    build_prepared_quality_rules,
    check_prepared_entry_quality,
    compute_split_counts,
    create_prepare_stats,
    decide_prepare_sample_usage,
    finalize_prepare_stats,
    infer_missing_prediction_classes_from_confusion,
    iter_jsonl_entries,
    materialize_training_manifests,
    register_prepare_input,
    register_prepare_skip,
    register_prepare_used,
    should_defer_training_for_class_coverage,
    split_dataset,
    training_class_coverage_report,
    validate_training_class_coverage,
    write_downloaded_items_manifest,
)


class ActionTrainingPipelineTests(unittest.TestCase):
    def test_iter_jsonl_entries_skips_malformed_lines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "items.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"item_id": "a"}, ensure_ascii=False),
                        "{broken",
                        json.dumps(["not", "a", "dict"], ensure_ascii=False),
                        json.dumps({"item_id": "b"}, ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )

            rows = list(iter_jsonl_entries(manifest))

        self.assertEqual([row["item_id"] for row in rows], ["a", "b"])

    def test_split_dataset_requires_train_and_validation_samples(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            paths = {
                "current_split_train": workspace / "train.jsonl",
                "current_split_val": workspace / "val.jsonl",
                "current_split_test": workspace / "test.jsonl",
            }
            item = DownloadedItem(
                item_id="one",
                source_label="source",
                target_label="normal",
                video_path=workspace / "one.mp4",
                download_url="local",
                metadata={},
            )

            with self.assertRaisesRegex(RuntimeError, "학습/검증 split"):
                split_dataset(
                    [item],
                    {"split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}},
                    paths,
                )

    def test_compute_split_counts_keeps_validation_for_two_samples(self) -> None:
        self.assertEqual(
            compute_split_counts(
                2,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            ),
            (1, 1, 0),
        )

    def test_decide_prepare_sample_usage_recovers_partial_pose_when_allowed(self) -> None:
        keep, reason, recovery_actions = decide_prepare_sample_usage(
            {"valid_frames": 1, "total_valid_keypoints": 3, "recovery_actions": []},
            sequence_length=48,
            min_frames_with_person=4,
            fallback_min_frames_with_person=1,
            min_total_keypoints=1,
            max_missing_frames_ratio=0.98,
            allow_partial_pose=True,
            allow_padding=True,
        )

        self.assertTrue(keep)
        self.assertEqual(reason, "")
        self.assertIn("partial_pose_padding", recovery_actions)

    def test_decide_prepare_sample_usage_keeps_strict_skip_when_not_recoverable(self) -> None:
        keep, reason, recovery_actions = decide_prepare_sample_usage(
            {"valid_frames": 1, "total_valid_keypoints": 3, "recovery_actions": []},
            sequence_length=48,
            min_frames_with_person=4,
            fallback_min_frames_with_person=1,
            min_total_keypoints=1,
            max_missing_frames_ratio=0.98,
            allow_partial_pose=False,
            allow_padding=True,
        )

        self.assertFalse(keep)
        self.assertEqual(reason, "min_frames_with_person")
        self.assertEqual(recovery_actions, [])

    def test_decide_prepare_sample_usage_enforces_missing_ratio_even_with_padding(self) -> None:
        keep, reason, recovery_actions = decide_prepare_sample_usage(
            {
                "valid_frames": 3,
                "confirmed_frames": 3,
                "total_valid_keypoints": 30,
                "recovery_actions": [],
            },
            sequence_length=48,
            min_frames_with_person=4,
            fallback_min_frames_with_person=1,
            min_total_keypoints=1,
            max_missing_frames_ratio=0.5,
            allow_partial_pose=True,
            allow_padding=True,
        )

        self.assertFalse(keep)
        self.assertEqual(reason, "too_many_missing_frames")
        self.assertEqual(recovery_actions, [])

    def test_prepared_quality_rules_filter_existing_low_quality_entries(self) -> None:
        rules = build_prepared_quality_rules(
            {
                "preprocess": {
                    "sequence_length": 48,
                    "min_frames_with_person": 8,
                    "fallback_min_frames_with_person": 4,
                    "min_confirmed_frames_with_person": 2,
                    "max_missing_frames_ratio": 0.85,
                    "max_fallback_frames_ratio": 0.6,
                    "min_total_keypoints": 32,
                    "allow_partial_pose": True,
                    "allow_padding": True,
                }
            }
        )

        keep, reason = check_prepared_entry_quality(
            {
                "valid_frames": 3,
                "confirmed_frames": 3,
                "fallback_frames": 0,
                "total_valid_keypoints": 30,
            },
            rules,
        )

        self.assertFalse(keep)
        self.assertEqual(reason, "pose_keypoint_insufficient")

    def test_finalize_prepare_stats_reports_class_skip_ratios(self) -> None:
        stats = create_prepare_stats(["normal", "violence"])
        register_prepare_input(stats, split_name="train", label="normal")
        register_prepare_used(stats, split_name="train", label="normal", recovery_actions=[])
        register_prepare_input(stats, split_name="val", label="violence")
        register_prepare_skip(stats, split_name="val", label="violence", reason="person_not_detected")

        summary = finalize_prepare_stats(stats)

        self.assertEqual(summary["overall"]["total"], 2)
        self.assertEqual(summary["overall"]["skipped"], 1)
        self.assertEqual(summary["by_reason"]["person_not_detected"], 1)
        self.assertEqual(summary["by_label"]["violence"]["skip_ratio"], 1.0)

    def test_materialize_training_manifests_removes_cross_split_duplicates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manifests = workspace / "manifests"
            manifests.mkdir()
            train_source = manifests / "prepared_train.jsonl"
            val_source = manifests / "prepared_val.jsonl"
            test_source = manifests / "prepared_test.jsonl"
            duplicate_entry = {
                "item_id": "same",
                "source_label": "normal",
                "target_label": "normal",
                "pose_path": str(workspace / "same.npz"),
                "label_idx": 0,
            }
            train_source.write_text(json.dumps(duplicate_entry, ensure_ascii=False) + "\n", encoding="utf-8")
            val_source.write_text(json.dumps(duplicate_entry, ensure_ascii=False) + "\n", encoding="utf-8")
            test_source.write_text("", encoding="utf-8")
            paths = {
                "active_prepared_train": manifests / "active_train.jsonl",
                "active_prepared_val": manifests / "active_val.jsonl",
                "active_prepared_test": manifests / "active_test.jsonl",
                "active_manifest_state": manifests / "active_state.json",
            }

            active_paths = materialize_training_manifests(
                {"dataset": {"target_labels": ["normal"], "label_mapping": {"normal": "normal"}}},
                paths,
                {"train": train_source, "val": val_source, "test": test_source},
            )

            train_rows = list(iter_jsonl_entries(active_paths["train"]))
            val_rows = list(iter_jsonl_entries(active_paths["val"]))

        self.assertEqual(len(train_rows), 1)
        self.assertEqual(val_rows, [])

    def test_materialize_training_manifests_removes_same_video_with_different_pose_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            manifests = workspace / "manifests"
            manifests.mkdir()
            train_source = manifests / "prepared_train.jsonl"
            val_source = manifests / "prepared_val.jsonl"
            test_source = manifests / "prepared_test.jsonl"
            base_entry = {
                "item_id": "same-video",
                "source_label": "normal",
                "target_label": "normal",
                "label_idx": 0,
                "video_path": str(workspace / "raw" / "same.mp4"),
                "metadata": {"relative_path": "same.mp4"},
            }
            train_source.write_text(
                json.dumps({**base_entry, "pose_path": str(workspace / "train" / "same.npz")}) + "\n",
                encoding="utf-8",
            )
            val_source.write_text(
                json.dumps({**base_entry, "pose_path": str(workspace / "val" / "same.npz")}) + "\n",
                encoding="utf-8",
            )
            test_source.write_text("", encoding="utf-8")
            paths = {
                "active_prepared_train": manifests / "active_train.jsonl",
                "active_prepared_val": manifests / "active_val.jsonl",
                "active_prepared_test": manifests / "active_test.jsonl",
                "active_manifest_state": manifests / "active_state.json",
            }

            active_paths = materialize_training_manifests(
                {"dataset": {"target_labels": ["normal"], "label_mapping": {"normal": "normal"}}},
                paths,
                {"train": train_source, "val": val_source, "test": test_source},
            )

            train_rows = list(iter_jsonl_entries(active_paths["train"]))
            val_rows = list(iter_jsonl_entries(active_paths["val"]))

        self.assertEqual(len(train_rows), 1)
        self.assertEqual(val_rows, [])

    def test_write_downloaded_items_manifest_uses_expected_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "raw.jsonl"
            item = DownloadedItem(
                item_id="one",
                source_label="source",
                target_label="normal",
                video_path=Path(tmpdir) / "one.mp4",
                download_url="local",
                metadata={"relative_path": "one.mp4"},
            )

            write_downloaded_items_manifest(manifest, [item])
            rows = list(iter_jsonl_entries(manifest))

        self.assertEqual(rows[0]["item_id"], "one")
        self.assertEqual(rows[0]["target_label"], "normal")
        self.assertIn("metadata", rows[0])

    def test_stage_all_can_defer_when_only_one_class_is_ready(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            train_manifest = workspace / "train.jsonl"
            val_manifest = workspace / "val.jsonl"
            for path in (train_manifest, val_manifest):
                path.write_text(
                    json.dumps({"item_id": path.stem, "target_label": "normal"}, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            config = {
                "dataset": {"target_labels": ["normal", "violence"]},
                "training": {"min_active_train_classes": 2, "min_active_val_classes": 2},
            }
            manifests = {"train": train_manifest, "val": val_manifest}

            report = training_class_coverage_report(config, manifests)

            self.assertFalse(report["ok"])
            self.assertTrue(should_defer_training_for_class_coverage(config, report, stage="all"))
            self.assertFalse(should_defer_training_for_class_coverage(config, report, stage="train"))
            with self.assertRaisesRegex(RuntimeError, "클래스 수가 부족"):
                validate_training_class_coverage(config, manifests)

    def test_infer_missing_prediction_classes_from_confusion(self) -> None:
        missing = infer_missing_prediction_classes_from_confusion(
            [
                [0, 2, 1],
                [0, 3, 0],
                [0, 1, 2],
            ],
            labels=["violence", "collapse", "abduction"],
        )

        self.assertEqual(missing, ["violence"])

    def test_adaptive_class_weighting_boosts_all_low_performing_classes(self) -> None:
        multipliers = build_adaptive_class_weight_multipliers(
            {
                "training": {
                    "class_weight_multipliers": {},
                    "adaptive_class_weighting": {
                        "enabled": True,
                        "target_recall": 0.55,
                        "target_f1": 0.45,
                        "max_multiplier": 2.5,
                    },
                }
            },
            labels=["violence", "collapse", "abduction", "loitering"],
            metric_payloads=[
                {
                    "final_validation": {
                        "per_class": [
                            {"class_index": 0, "label": "violence", "recall": 0.2, "f1": 0.2, "support": 5, "predicted": 5},
                            {"class_index": 1, "label": "collapse", "recall": 0.67, "f1": 0.57, "support": 9, "predicted": 12},
                            {"class_index": 2, "label": "abduction", "recall": 0.17, "f1": 0.22, "support": 6, "predicted": 3},
                            {"class_index": 3, "label": "loitering", "recall": 0.64, "f1": 0.64, "support": 11, "predicted": 11},
                        ]
                    },
                    "train_distribution": {
                        "counts": [
                            {"label": "violence", "count": 29},
                            {"label": "collapse", "count": 44},
                            {"label": "abduction", "count": 40},
                            {"label": "loitering", "count": 51},
                        ]
                    },
                }
            ],
        )

        self.assertGreater(multipliers["violence"], 1.0)
        self.assertGreater(multipliers["abduction"], 1.0)
        self.assertNotIn("collapse", multipliers)
        self.assertNotIn("loitering", multipliers)

    def test_adaptive_class_weighting_uses_missing_prediction_cap(self) -> None:
        multipliers = build_adaptive_class_weight_multipliers(
            {
                "training": {
                    "class_weight_multipliers": {},
                    "adaptive_class_weighting": {
                        "enabled": True,
                        "missing_prediction_multiplier": 2.5,
                        "max_multiplier": 2.5,
                    },
                }
            },
            labels=["violence", "collapse"],
            metric_payloads=[
                {
                    "final_validation": {
                        "confusion_matrix": [
                            [0, 5],
                            [0, 10],
                        ],
                        "per_class": [
                            {"class_index": 0, "label": "violence", "recall": 0.0, "f1": 0.0, "support": 5},
                            {"class_index": 1, "label": "collapse", "recall": 0.9, "f1": 0.75, "support": 10},
                        ]
                    },
                }
            ],
        )

        self.assertEqual(multipliers["violence"], 2.5)
        self.assertNotIn("collapse", multipliers)


if __name__ == "__main__":
    unittest.main()

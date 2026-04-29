from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from action_training_pipeline import (
    DownloadedItem,
    compute_split_counts,
    create_prepare_stats,
    decide_prepare_sample_usage,
    finalize_prepare_stats,
    iter_jsonl_entries,
    materialize_training_manifests,
    register_prepare_input,
    register_prepare_skip,
    register_prepare_used,
    split_dataset,
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


if __name__ == "__main__":
    unittest.main()

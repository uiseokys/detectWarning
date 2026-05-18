from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

if importlib.util.find_spec("uvicorn") is None or importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("dashboard runtime dependencies are not installed")

from training_dashboard import (
    build_auto_recommendation_policy,
    build_diagnosis_label_priorities,
    find_next_trainable_aihub_entry,
    infer_job_state_from_log,
    normalize_aihub_lookup_entries,
    restore_completed_jobs_from_logs,
)


class TrainingDashboardRecommendationTests(unittest.TestCase):
    def test_lookup_excludes_mapping_targets_outside_active_labels(self) -> None:
        entries = normalize_aihub_lookup_entries(
            [
                {"filekey": "1", "name": "normal/outsidedoor_1.zip"},
                {"filekey": "2", "name": "violence/outsidedoor_2.zip"},
            ],
            label_mapping={"normal": "normal", "violence": "violence"},
            excluded_source_labels=[],
            target_labels=["violence", "collapse", "abduction", "loitering"],
        )

        by_filekey = {entry["filekey"]: entry for entry in entries}
        self.assertEqual(by_filekey["1"]["status"], "excluded")
        self.assertFalse(by_filekey["1"]["trainable"])
        self.assertEqual(by_filekey["1"]["out_of_scope_target_label"], "normal")
        self.assertEqual(by_filekey["2"]["status"], "trainable")
        self.assertEqual(by_filekey["2"]["target_label"], "violence")

    def test_recommendation_prioritizes_underrepresented_target_label(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {"filekey": "10", "status": "trainable", "target_label": "normal", "selectable": True},
                {"filekey": "11", "status": "trainable", "target_label": "danger", "selectable": True},
            ],
            prepared_label_counts={"normal": 20, "danger": 2},
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "11")
        self.assertEqual(selected["recommendation_score"]["target_label"], "danger")

    def test_recommendation_counts_running_label_as_planned(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {"filekey": "10", "status": "running", "target_label": "danger", "selectable": False},
                {"filekey": "11", "status": "trainable", "target_label": "danger", "selectable": True},
                {"filekey": "12", "status": "trainable", "target_label": "normal", "selectable": True},
            ],
            prepared_label_counts={"normal": 0, "danger": 0},
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "12")

    def test_recommendation_does_not_reselect_prepared_filekey(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {"filekey": "10", "status": "prepared", "target_label": "danger", "selectable": False},
                {"filekey": "11", "status": "trainable", "target_label": "normal", "selectable": True},
            ],
            prepared_label_counts={"normal": 0, "danger": 1},
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "11")

    def test_recommendation_can_be_limited_to_outside_zip_group(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "20",
                    "status": "trainable",
                    "target_label": "danger",
                    "selectable": True,
                    "zip_group": "insidedoor",
                },
                {
                    "filekey": "21",
                    "status": "trainable",
                    "target_label": "normal",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"normal": 100, "danger": 0},
            allowed_zip_groups={"outsidedoor"},
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "21")
        self.assertEqual(selected["recommendation_scope"]["zip_group"], "outsidedoor")

    def test_recommendation_returns_none_when_no_outside_candidate_exists(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "20",
                    "status": "trainable",
                    "target_label": "danger",
                    "selectable": True,
                    "zip_group": "insidedoor",
                },
                {
                    "filekey": "22",
                    "status": "trainable",
                    "target_label": "warning",
                    "selectable": True,
                    "name": "inside_croki_1.zip",
                },
            ],
            allowed_zip_groups={"outsidedoor"},
        )

        self.assertIsNone(selected)

    def test_recommendation_infers_zip_group_from_filename(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "30",
                    "status": "trainable",
                    "target_label": "normal",
                    "selectable": True,
                    "name": "outsidedoor_12.zip",
                }
            ],
            allowed_zip_groups={"outsidedoor"},
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "30")

    def test_recommendation_uses_diagnosis_priority_between_balanced_classes(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "40",
                    "status": "trainable",
                    "target_label": "normal",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
                {
                    "filekey": "41",
                    "status": "trainable",
                    "target_label": "danger",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"normal": 48, "danger": 48},
            prepared_split_label_counts={
                "train": {"normal": 40, "danger": 40},
                "val": {"normal": 8, "danger": 8},
            },
            allowed_zip_groups={"outsidedoor"},
            insights={
                "class_insights": [
                    {
                        "level": "critical",
                        "title": "위험 클래스 recall 부족",
                        "details": {"label": "danger", "recall": 0.1},
                    }
                ]
            },
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "41")
        self.assertEqual(selected["recommendation_reason"], "diagnosis_guided")
        self.assertGreater(selected["recommendation_score"]["insight_priority"], 0)

    def test_recommendation_prefers_underrepresented_class_over_dominant_low_recall_class(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "46",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
                {
                    "filekey": "47",
                    "status": "trainable",
                    "target_label": "loitering",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"violence": 139, "loitering": 33},
            prepared_split_label_counts={
                "train": {"violence": 115, "loitering": 28},
                "val": {"violence": 24, "loitering": 5},
            },
            allowed_zip_groups={"outsidedoor"},
            insights={
                "class_insights": [
                    {
                        "level": "critical",
                        "title": "violence recall low",
                        "details": {"label": "violence", "recall": 0.0},
                    }
                ]
            },
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "47")
        self.assertEqual(selected["recommendation_score"]["target_label"], "loitering")

    def test_recommendation_fills_missing_class_before_repeating_diagnosed_class(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "42",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
                {
                    "filekey": "43",
                    "status": "trainable",
                    "target_label": "collapse",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"violence": 80},
            prepared_split_label_counts={
                "train": {"violence": 60},
                "val": {"violence": 20},
            },
            allowed_zip_groups={"outsidedoor"},
            insights={
                "class_insights": [
                    {
                        "level": "critical",
                        "title": "violence recall low",
                        "details": {"label": "violence", "recall": 0.0},
                    }
                ]
            },
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "43")
        self.assertEqual(selected["recommendation_score"]["target_label"], "collapse")
        self.assertEqual(selected["recommendation_score"]["coverage_state"], "missing")

    def test_recommendation_repairs_missing_validation_split_before_diagnosis(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "44",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
                {
                    "filekey": "45",
                    "status": "trainable",
                    "target_label": "collapse",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"violence": 80, "collapse": 12},
            prepared_split_label_counts={
                "train": {"violence": 60, "collapse": 12},
                "val": {"violence": 20},
            },
            allowed_zip_groups={"outsidedoor"},
            insights={
                "class_insights": [
                    {
                        "level": "critical",
                        "title": "violence recall low",
                        "details": {"label": "violence", "recall": 0.0},
                    }
                ]
            },
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "45")
        self.assertEqual(selected["recommendation_score"]["coverage_state"], "split_incomplete")

    def test_diagnosis_priorities_include_confusion_source_label(self) -> None:
        priorities = build_diagnosis_label_priorities(
            {
                "confusion_insights": [
                    {
                        "level": "warning",
                        "title": "위험 클래스를 normal로 오분류",
                        "details": {"from": "warning", "to": "normal", "count": 8},
                    }
                ]
            }
        )

        self.assertIn("warning", priorities)
        self.assertGreater(priorities["warning"]["priority"], 0)

    def test_recommendation_prioritizes_empty_class_over_dominant_low_recall_class(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "50",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
                {
                    "filekey": "51",
                    "status": "trainable",
                    "target_label": "loitering",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"violence": 180, "loitering": 0},
            allowed_zip_groups={"outsidedoor"},
            insights={
                "class_insights": [
                    {
                        "level": "critical",
                        "title": "낮은 클래스 recall",
                        "details": {"label": "violence", "recall": 0.0},
                    }
                ],
                "diagnostics": [
                    {
                        "level": "critical",
                        "title": "학습 데이터 분포",
                        "details": {
                            "dominant_label": "violence",
                            "empty_labels": ["loitering"],
                        },
                    }
                ],
            },
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "51")
        self.assertEqual(selected["recommendation_score"]["target_label"], "loitering")

    def test_auto_policy_blocks_class_above_one_point_five_ratio(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "60",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
                {
                    "filekey": "61",
                    "status": "trainable",
                    "target_label": "loitering",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"violence": 20, "loitering": 10},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "61")
        self.assertEqual(selected["recommendation_score"]["target_label"], "loitering")

    def test_auto_policy_selects_missing_violence_before_more_existing_classes(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "v1",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
                {
                    "filekey": "c1",
                    "status": "trainable",
                    "target_label": "collapse",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"collapse": 10769, "loitering": 7531, "abduction": 1166},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "v1")
        self.assertEqual(selected["recommendation_score"]["target_label"], "violence")

    def test_auto_policy_keeps_extracting_when_only_non_abduction_class_is_available(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "62",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"violence": 20, "abduction": 10},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "62")

    def test_auto_policy_excludes_abduction_from_new_auto_extraction(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "63",
                    "status": "trainable",
                    "target_label": "abduction",
                    "selectable": True,
                    "zip_group": "insidedoor",
                },
                {
                    "filekey": "64",
                    "status": "trainable",
                    "target_label": "violence",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"abduction": 4, "violence": 10},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "64")
        self.assertEqual(selected["recommendation_score"]["target_label"], "violence")

    def test_auto_policy_allows_collapse_insidedoor_after_outside_is_exhausted(self) -> None:
        entries = [
            {
                "filekey": "65",
                "status": "trainable",
                "target_label": "collapse",
                "selectable": True,
                "zip_group": "insidedoor",
            }
        ]
        without_diagnosis = find_next_trainable_aihub_entry(
            entries,
            prepared_label_counts={"collapse": 6, "abduction": 6},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
        )
        with_diagnosis = find_next_trainable_aihub_entry(
            entries,
            prepared_label_counts={"collapse": 6, "abduction": 6},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
            insights={
                "class_insights": [
                    {
                        "level": "warning",
                        "title": "collapse recall low",
                        "details": {"label": "collapse", "recall": 0.1},
                    }
                ]
            },
        )

        self.assertIsNotNone(without_diagnosis)
        self.assertEqual(without_diagnosis["filekey"], "65")
        self.assertEqual(without_diagnosis["recommendation_scope"]["scope_reason"], "always_fallback")
        self.assertIsNotNone(with_diagnosis)
        self.assertEqual(with_diagnosis["filekey"], "65")
        self.assertEqual(with_diagnosis["recommendation_scope"]["scope_reason"], "always_fallback")

    def test_auto_policy_prefers_outside_collapse_before_insidedoor(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "66",
                    "status": "trainable",
                    "target_label": "collapse",
                    "selectable": True,
                    "zip_group": "insidedoor",
                },
                {
                    "filekey": "67",
                    "status": "trainable",
                    "target_label": "collapse",
                    "selectable": True,
                    "zip_group": "outsidedoor",
                },
            ],
            prepared_label_counts={"collapse": 6, "abduction": 6},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
            insights={
                "class_insights": [
                    {
                        "level": "warning",
                        "title": "collapse recall low",
                        "details": {"label": "collapse", "recall": 0.1},
                    }
                ]
            },
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["filekey"], "67")
        self.assertEqual(selected["recommendation_scope"]["zip_group"], "outsidedoor")

    def test_auto_policy_excludes_inside_croki(self) -> None:
        selected = find_next_trainable_aihub_entry(
            [
                {
                    "filekey": "68",
                    "status": "trainable",
                    "target_label": "abduction",
                    "selectable": True,
                    "zip_group": "inside_croki",
                }
            ],
            prepared_label_counts={"abduction": 1, "violence": 3},
            allowed_zip_groups={"outsidedoor"},
            recommendation_policy=build_auto_recommendation_policy(),
        )

        self.assertIsNone(selected)

    def test_running_launcher_log_is_not_restored_as_completed_warning(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workspace = repo_root / ".tmp_test_log_restore"
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            job_logs = workspace / "job_logs"
            runtime_configs = workspace / "runtime_configs"
            job_logs.mkdir(parents=True)
            runtime_configs.mkdir(parents=True)
            job_id = "job_20260430_101537_551245_49658"
            log_path = job_logs / f"{job_id}.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[launcher] datasetkey 171 | filekey 49658 작업을 시작합니다.",
                        "[pipeline][starting][running] 0.0% 학습 파이프라인을 시작합니다.",
                        "[pipeline][download][running] 8.0% AIHub에서 분할 ZIP 데이터를 다운로드하는 중입니다.",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(infer_job_state_from_log(log_path), "running")
            self.assertEqual(restore_completed_jobs_from_logs(job_logs, runtime_configs), [])
        finally:
            if workspace.exists() and workspace.resolve().is_relative_to(repo_root.resolve()):
                shutil.rmtree(workspace, ignore_errors=True)

    def test_data_ready_log_is_restored_as_prepared_not_warning(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workspace = repo_root / ".tmp_test_log_restore_data_ready"
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            job_logs = workspace / "job_logs"
            runtime_configs = workspace / "runtime_configs"
            job_logs.mkdir(parents=True)
            runtime_configs.mkdir(parents=True)
            job_id = "job_20260430_101537_551245_49659"
            log_path = job_logs / f"{job_id}.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[launcher] datasetkey 171 | filekey 49659 작업을 시작합니다.",
                        "[train] deferred: 현재 filekey 데이터는 누적했지만 학습은 아직 시작하지 않았습니다.",
                        "[pipeline][data_ready][data_ready] 100.0% 데이터 준비 완료, 학습 대기",
                    ]
                ),
                encoding="utf-8",
            )

            restored = restore_completed_jobs_from_logs(job_logs, runtime_configs)

            self.assertEqual(infer_job_state_from_log(log_path), "data_ready")
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0]["state"], "data_ready")
        finally:
            if workspace.exists() and workspace.resolve().is_relative_to(repo_root.resolve()):
                shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

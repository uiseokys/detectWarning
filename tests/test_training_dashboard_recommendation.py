from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

if importlib.util.find_spec("uvicorn") is None or importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("dashboard runtime dependencies are not installed")

from training_dashboard import build_diagnosis_label_priorities, find_next_trainable_aihub_entry


class TrainingDashboardRecommendationTests(unittest.TestCase):
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

    def test_recommendation_uses_diagnosis_priority_before_raw_class_count(self) -> None:
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
            prepared_label_counts={"normal": 1, "danger": 200},
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


if __name__ == "__main__":
    unittest.main()

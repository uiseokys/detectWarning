from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

if importlib.util.find_spec("uvicorn") is None or importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("dashboard runtime dependencies are not installed")

from training_dashboard import find_next_trainable_aihub_entry


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


if __name__ == "__main__":
    unittest.main()

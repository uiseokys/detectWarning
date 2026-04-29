from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from training_insights import interpret_training_results


def _messages(payload: dict, bucket: str) -> str:
    return "\n".join(
        f"{item.get('title', '')} {item.get('message', '')} {item.get('key', '')}"
        for item in payload.get(bucket, [])
        if isinstance(item, dict)
    )


class TrainingInsightsTests(unittest.TestCase):
    def test_detects_overfitting_from_loss_gap(self) -> None:
        insights = interpret_training_results(
            {
                "labels": ["normal", "danger"],
                "latest": {
                    "train_loss": 0.18,
                    "val_loss": 2.67,
                    "val_accuracy": 0.47,
                    "val_macro_f1": 0.32,
                },
                "final_validation": {
                    "per_class": [
                        {"class_index": 0, "label": "normal", "recall": 0.7, "f1": 0.6, "support": 20},
                        {"class_index": 1, "label": "danger", "recall": 0.2, "f1": 0.25, "support": 10},
                    ],
                    "confusion_matrix": [[14, 6], [8, 2]],
                },
            }
        )

        self.assertIn(insights["overall_level"], {"warning", "critical"})
        self.assertIn("overfit_regularization", _messages(insights, "recommendations"))
        self.assertIn("과적합", _messages(insights, "diagnostics"))

    def test_detects_high_accuracy_low_macro_f1(self) -> None:
        insights = interpret_training_results(
            {
                "labels": ["normal", "warning", "danger"],
                "latest": {"val_accuracy": 0.74, "val_macro_f1": 0.33, "val_balanced_accuracy": 0.42},
                "final_validation": {
                    "per_class": [
                        {"class_index": 0, "label": "normal", "recall": 0.95, "f1": 0.88, "support": 80},
                        {"class_index": 1, "label": "warning", "recall": 0.05, "f1": 0.08, "support": 10},
                        {"class_index": 2, "label": "danger", "recall": 0.10, "f1": 0.12, "support": 10},
                    ]
                },
            }
        )

        self.assertIn("accuracy와 macro F1", _messages(insights, "diagnostics"))
        self.assertIn("macro_f1_class_weight", _messages(insights, "recommendations"))

    def test_detects_danger_class_misclassified_as_normal(self) -> None:
        insights = interpret_training_results(
            {
                "labels": ["normal", "danger"],
                "latest": {"val_accuracy": 0.6, "val_macro_f1": 0.42},
                "final_validation": {
                    "per_class": [
                        {"class_index": 0, "label": "normal", "recall": 0.9, "f1": 0.8, "support": 30},
                        {"class_index": 1, "label": "danger", "recall": 0.2, "f1": 0.25, "support": 10},
                    ],
                    "confusion_matrix": [[27, 3], [8, 2]],
                },
            }
        )

        self.assertIn("위험 클래스", _messages(insights, "class_insights"))
        self.assertIn("normal", _messages(insights, "confusion_insights"))
        self.assertIn("danger_recall_data", _messages(insights, "recommendations"))

    def test_detects_many_skipped_samples(self) -> None:
        insights = interpret_training_results(
            {"labels": ["normal", "danger"], "latest": {"val_accuracy": 0.55, "val_macro_f1": 0.45}},
            data_stats={
                "skip_report": {
                    "summary": {
                        "total_items": 1000,
                        "skipped_items": 320,
                        "skip_ratio": 0.32,
                        "by_reason": {"too_many_missing_frames": 220},
                    },
                    "prepare_summary": {
                        "by_label": {
                            "danger": {"total": 100, "used": 40, "skipped": 60, "skip_ratio": 0.6}
                        }
                    },
                }
            },
        )

        self.assertIn("스킵 비율", _messages(insights, "data_quality_insights"))
        self.assertIn("class_skip_bias", _messages(insights, "recommendations"))

    def test_detects_high_loss_with_improving_macro_f1(self) -> None:
        insights = interpret_training_results(
            {
                "labels": ["normal", "danger"],
                "latest": {"train_loss": 0.5, "val_loss": 2.0, "val_macro_f1": 0.42},
                "history": [
                    {"epoch": 1, "train_loss": 0.8, "val_loss": 1.8, "val_macro_f1": 0.30},
                    {"epoch": 2, "train_loss": 0.65, "val_loss": 1.9, "val_macro_f1": 0.36},
                    {"epoch": 3, "train_loss": 0.5, "val_loss": 2.0, "val_macro_f1": 0.42},
                ],
            }
        )

        self.assertIn("loss_f1_calibration", _messages(insights, "recommendations"))
        self.assertIn("F1 개선", _messages(insights, "trend_insights"))

    def test_good_metrics_remain_good(self) -> None:
        insights = interpret_training_results(
            {
                "labels": ["normal", "danger"],
                "latest": {"train_loss": 0.42, "val_loss": 0.48, "val_accuracy": 0.9, "val_macro_f1": 0.88},
                "final_validation": {
                    "per_class": [
                        {"class_index": 0, "label": "normal", "recall": 0.9, "f1": 0.9, "support": 50},
                        {"class_index": 1, "label": "danger", "recall": 0.86, "f1": 0.86, "support": 45},
                    ],
                    "confusion_matrix": [[45, 5], [6, 39]],
                },
            }
        )

        self.assertEqual(insights["overall_level"], "good")
        self.assertTrue(insights["summary"])

    def test_missing_and_invalid_values_do_not_crash(self) -> None:
        insights = interpret_training_results(
            {"latest": {"val_loss": "nan", "val_macro_f1": None}, "final_validation": {"per_class": None}},
            confusion_matrix=None,
            data_stats=None,
        )

        self.assertIn("summary", insights)


if __name__ == "__main__":
    unittest.main()

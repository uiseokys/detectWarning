from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from reporting import (
    analyze_class_balance,
    get_latest_job,
    normalize_metric_payload,
    remap_confusion_matrix,
    remap_per_class_rows,
    summarize_stage_timings,
)


class ReportingTests(unittest.TestCase):
    def test_remap_per_class_rows_fills_missing_target_labels(self) -> None:
        rows = [
            {"class_index": 0, "precision": 0.9, "recall": 0.8, "f1": 0.85},
            {"class_index": 1, "precision": 0.5, "recall": 0.4, "f1": 0.44},
        ]

        remapped = remap_per_class_rows(
            rows,
            source_labels=["normal", "violence"],
            target_labels=["normal", "violence", "collapse"],
        )

        self.assertEqual([row["label"] for row in remapped], ["normal", "violence", "collapse"])
        self.assertEqual(remapped[0]["precision"], 0.9)
        self.assertIsNone(remapped[2]["precision"])

    def test_remap_confusion_matrix_aligns_existing_labels(self) -> None:
        remapped = remap_confusion_matrix(
            [[3, 1], [0, 4]],
            source_labels=["normal", "violence"],
            target_labels=["normal", "violence", "collapse"],
        )

        self.assertEqual(remapped, [[3, 1, 0], [0, 4, 0], [0, 0, 0]])

    def test_normalize_metric_payload_uses_target_labels(self) -> None:
        payload = {
            "labels": ["normal", "violence"],
            "final_validation": {
                "per_class": [
                    {"class_index": 0, "label": "normal", "precision": 1.0, "recall": 1.0, "f1": 1.0}
                ],
                "confusion_matrix": [[1, 0], [0, 0]],
            },
        }

        normalized = normalize_metric_payload(payload, target_labels=["normal", "violence", "collapse"])
        self.assertEqual(normalized["labels"], ["normal", "violence", "collapse"])
        self.assertEqual(len(normalized["final_validation"]["per_class"]), 3)
        self.assertEqual(len(normalized["final_validation"]["confusion_matrix"]), 3)

    def test_get_latest_job_falls_back_to_recent_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            artifacts = workspace / "artifacts"
            artifacts.mkdir()
            (artifacts / "metrics.json").write_text("{}", encoding="utf-8")
            (workspace / "pipeline_status.json").write_text("{}", encoding="utf-8")

            latest = get_latest_job(
                {
                    "workspace_dir": workspace,
                    "pipeline_status": workspace / "pipeline_status.json",
                    "artifacts_dir": artifacts,
                }
            )

            self.assertEqual(latest["state"], "completed")
            self.assertIn("최근 학습 결과", latest["message"])

    def test_analyze_class_balance_marks_empty_and_skewed_classes(self) -> None:
        summary = analyze_class_balance(
            ["normal", "violence", "collapse"],
            {"normal": 40, "violence": 4, "collapse": 0},
            min_samples=5,
            ratio_warn=4.0,
        )

        self.assertEqual(summary["severity"], "critical")
        self.assertIn("collapse", summary["empty_labels"])
        self.assertEqual(summary["dominant_label"], "normal")
        self.assertEqual(summary["minority_label"], "violence")
        self.assertGreaterEqual(float(summary["imbalance_ratio"]), 10.0)

    def test_summarize_stage_timings_orders_expected_stages(self) -> None:
        summary = summarize_stage_timings(
            {
                "train": {"started_at": "a", "finished_at": "b", "duration_seconds": 12.0},
                "download": {"started_at": "c", "finished_at": "d", "duration_seconds": 3.0},
            }
        )

        self.assertEqual(
            [row["stage"] for row in summary["ordered"]],
            ["download", "prepare", "train", "total"],
        )
        self.assertEqual(summary["by_stage"]["train"]["duration_seconds"], 12.0)


if __name__ == "__main__":
    unittest.main()

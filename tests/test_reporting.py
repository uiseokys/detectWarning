from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from reporting import (
    get_latest_job,
    normalize_metric_payload,
    remap_confusion_matrix,
    remap_per_class_rows,
)


class ReportingTests(unittest.TestCase):
    def test_remap_per_class_rows_fills_missing_target_labels(self) -> None:
        rows = [
            {"class_index": 0, "label": "normal", "precision": 0.9, "recall": 0.8, "f1": 0.85},
            {"class_index": 1, "label": "violence", "precision": 0.5, "recall": 0.4, "f1": 0.44},
        ]

        remapped = remap_per_class_rows(
            rows,
            source_labels=["normal", "violence"],
            target_labels=["normal", "violence", "collapse"],
        )

        self.assertEqual([row["label"] for row in remapped], ["normal", "violence", "collapse"])
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
                "per_class": [{"class_index": 0, "label": "normal", "precision": 1.0, "recall": 1.0, "f1": 1.0}],
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


if __name__ == "__main__":
    unittest.main()

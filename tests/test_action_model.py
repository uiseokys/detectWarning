from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this test environment")

import torch

from action_model import (
    PoseSequenceDataset,
    _compute_f1,
    _build_sample_weights,
    _build_loss_function,
    _build_validation_error_analysis,
    _resolve_class_weight_multipliers,
    _resolve_num_workers,
    _summarize_class_distribution,
)


class ActionModelMetricTests(unittest.TestCase):
    def test_class_distribution_counts_zero_label_index(self) -> None:
        summary = _summarize_class_distribution(
            [{"label_idx": 0}, {"label_idx": 1}, {"label_idx": 0}],
            ["normal", "violence"],
            min_samples=1,
            ratio_warn=10.0,
        )

        counts = {row["label"]: row["count"] for row in summary["counts"]}

        self.assertEqual(counts["normal"], 2)
        self.assertEqual(counts["violence"], 1)

    def test_compute_f1_includes_labels_and_support(self) -> None:
        _macro_f1, rows = _compute_f1(
            np.array([[2, 1], [0, 3]], dtype=np.int64),
            labels=["normal", "violence"],
        )

        self.assertEqual(rows[0]["label"], "normal")
        self.assertEqual(rows[0]["support"], 3)
        self.assertAlmostEqual(rows[1]["recall"], 1.0)

    def test_validation_error_analysis_groups_false_negatives_and_confusion_pairs(self) -> None:
        analysis = _build_validation_error_analysis(
            {
                "confusion_matrix": [[3, 0], [3, 0]],
                "per_class": [
                    {"label": "normal", "precision": 0.5, "recall": 1.0, "f1": 0.666667, "support": 3},
                    {"label": "violence", "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 3},
                ],
                "misclassified_examples": [
                    {
                        "item_id": "v1",
                        "target_label": "violence",
                        "predicted_label": "normal",
                        "true_index": 1,
                        "predicted_index": 0,
                    },
                ],
            },
            labels=["normal", "violence"],
        )

        self.assertEqual(analysis["class_summary"][1]["false_negatives"], 3)
        self.assertIn("violence", analysis["false_negative_examples"])
        self.assertEqual(analysis["confusion_pair_examples"][0]["true_label"], "violence")
        self.assertEqual(analysis["summary"]["missing_prediction_classes"], ["violence"])

    def test_resolve_num_workers_respects_explicit_zero(self) -> None:
        self.assertEqual(_resolve_num_workers(0, batch_size=8), 0)
        self.assertEqual(_resolve_num_workers("0", batch_size=8), 0)

    def test_class_weight_multipliers_boost_selected_class_sample_weights(self) -> None:
        multipliers, payload = _resolve_class_weight_multipliers(
            ["violence", "collapse"],
            {"violence": 1.6, "unknown": 3.0},
        )
        weights = _build_sample_weights(
            [{"label_idx": 0}, {"label_idx": 1}, {"label_idx": 1}],
            num_classes=2,
            class_weight_multipliers=multipliers,
        )

        self.assertEqual(payload, {"violence": 1.6})
        self.assertIsNotNone(weights)
        self.assertGreater(float(weights[0]), float(weights[1]))

    def test_focal_loss_keeps_per_sample_loss_for_weighted_training(self) -> None:
        criterion, resolved_name = _build_loss_function(
            loss_name="focal",
            class_weights=torch.tensor([1.6, 1.0]),
            focal_gamma=2.0,
            label_smoothing=0.03,
        )
        loss_values = criterion(
            torch.tensor([[0.2, 1.4], [1.6, 0.1]], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        )

        self.assertEqual(resolved_name, "focal")
        self.assertEqual(tuple(loss_values.shape), (2,))
        self.assertTrue(torch.isfinite(loss_values).all())

    def test_pose_sequence_dataset_sanitizes_non_finite_pose_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            pose_path = workspace / "sample.npz"
            manifest_path = workspace / "manifest.jsonl"
            pose = np.zeros((2, 17, 3), dtype=np.float32)
            pose[0, 0, 0] = np.nan
            pose[0, 0, 1] = np.inf
            pose[0, 0, 2] = 2.5
            pose[1, 0, 0] = -5.0
            mask = np.array([1.0, np.nan], dtype=np.float32)
            np.savez(pose_path, pose=pose, mask=mask)
            manifest_path.write_text(
                json.dumps(
                    {"pose_path": str(pose_path), "label_idx": 0, "target_label": "normal"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            loaded_pose, loaded_mask, label = PoseSequenceDataset(manifest_path)[0]

        self.assertEqual(int(label.item()), 0)
        self.assertTrue(np.isfinite(loaded_pose.numpy()).all())
        self.assertTrue(np.isfinite(loaded_mask.numpy()).all())
        self.assertGreaterEqual(float(loaded_pose[1, 0, 0]), -0.5)
        self.assertLessEqual(float(loaded_pose[0, 0, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()

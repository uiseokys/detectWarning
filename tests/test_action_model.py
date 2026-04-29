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

from action_model import (
    PoseSequenceDataset,
    _compute_f1,
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

    def test_resolve_num_workers_respects_explicit_zero(self) -> None:
        self.assertEqual(_resolve_num_workers(0, batch_size=8), 0)
        self.assertEqual(_resolve_num_workers("0", batch_size=8), 0)

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

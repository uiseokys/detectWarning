from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this test environment")

from action_model import _compute_f1, _summarize_class_distribution


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


if __name__ == "__main__":
    unittest.main()

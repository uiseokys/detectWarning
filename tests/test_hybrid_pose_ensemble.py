from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from hybrid_pose_ensemble import feature_stats


class HybridPoseEnsembleTests(unittest.TestCase):
    def test_feature_stats_handles_all_nan_columns_without_warnings(self):
        values = np.asarray(
            [
                [[np.nan, 1.0], [np.nan, np.nan]],
                [[np.nan, 3.0], [np.nan, np.nan]],
            ],
            dtype=np.float32,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            stats = feature_stats(values)

        self.assertEqual(caught, [])
        self.assertTrue(np.isfinite(stats).all())
        self.assertEqual(stats.dtype, np.float32)

    def test_feature_stats_handles_empty_time_axis(self):
        stats = feature_stats(np.empty((0, 2, 3), dtype=np.float32))

        self.assertEqual(stats.shape, (24,))
        self.assertTrue(np.all(stats == 0.0))


if __name__ == "__main__":
    unittest.main()

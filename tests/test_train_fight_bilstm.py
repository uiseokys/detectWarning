from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from train_fight_bilstm import (
    CnnBiLstmAttention,
    build_optimizer_parameter_groups,
    compute_metrics,
    fuse_rgb_pose_probabilities,
    resolve_binary_labels,
    select_binary_threshold,
)


class FightBiLstmThresholdTests(unittest.TestCase):
    def test_select_binary_threshold_raises_threshold_when_normal_false_positives_are_costly(self) -> None:
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        violence_probabilities = np.array([0.55, 0.58, 0.20, 0.10, 0.90, 0.82, 0.72, 0.40])

        threshold, metrics = select_binary_threshold(
            y_true,
            violence_probabilities,
            min_normal_recall=0.70,
        )

        self.assertGreater(threshold, 0.5)
        normal_metrics = metrics.per_class[0]
        self.assertGreaterEqual(normal_metrics["recall"], 0.70)

    def test_compute_metrics_accepts_thresholded_predictions(self) -> None:
        metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], 0.25)

        self.assertEqual(metrics.confusion_matrix, [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics.accuracy, 0.75)

    def test_cnn_tail_can_be_unfrozen_while_earlier_cnn_stays_frozen(self) -> None:
        model = CnnBiLstmAttention(
            hidden_dim=32,
            num_layers=1,
            dropout=0.1,
            freeze_cnn=True,
            pretrained_cnn=False,
            unfreeze_cnn_tail=True,
        )

        cnn_params = [(name, param.requires_grad) for name, param in model.cnn.named_parameters()]
        self.assertTrue(any(name.startswith("layer4") and trainable for name, trainable in cnn_params))
        self.assertTrue(any(name.startswith("conv1") and not trainable for name, trainable in cnn_params))

    def test_optimizer_uses_lower_learning_rate_for_cnn_tail(self) -> None:
        model = CnnBiLstmAttention(
            hidden_dim=32,
            num_layers=1,
            dropout=0.1,
            freeze_cnn=True,
            pretrained_cnn=False,
            unfreeze_cnn_tail=True,
        )

        groups = build_optimizer_parameter_groups(model, base_lr=1e-3, cnn_lr_multiplier=0.1)

        learning_rates = sorted({round(float(group["lr"]), 6) for group in groups})
        self.assertEqual(learning_rates, [0.0001, 0.001])

    def test_fuse_rgb_pose_probabilities_uses_pose_when_available(self) -> None:
        rows = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
        rgb = np.array([0.8, 0.4, 0.2])
        pose = {"a": 0.2, "c": 0.6}

        fused = fuse_rgb_pose_probabilities(rows, rgb, pose, pose_weight=0.25)

        self.assertAlmostEqual(float(fused[0]), 0.65)
        self.assertAlmostEqual(float(fused[1]), 0.4)
        self.assertAlmostEqual(float(fused[2]), 0.3)

    def test_resolve_binary_labels_supports_collapse_positive_label(self) -> None:
        labels = resolve_binary_labels({"fight_bilstm": {"labels": ["normal", "collapse"]}})

        self.assertEqual(labels, ["normal", "collapse"])


if __name__ == "__main__":
    unittest.main()

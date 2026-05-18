from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from ensemble_action_models import (
    apply_class_bias,
    candidate_from_metrics_path,
    class_bias_variants,
)


LABELS = ["violence", "collapse", "abduction", "loitering"]


class EnsembleActionModelsTests(unittest.TestCase):
    def test_apply_class_bias_uses_passed_label_order(self):
        probabilities = np.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64)

        adjusted = apply_class_bias(probabilities, {"loitering": 2.0}, labels=LABELS)

        self.assertGreater(adjusted[0, 3], adjusted[0, 0])
        self.assertAlmostEqual(float(adjusted.sum()), 1.0)

    def test_class_bias_variants_include_each_label(self):
        names = {name for name, _bias in class_bias_variants(LABELS)}

        self.assertIn("violence_1.15", names)
        self.assertIn("loitering_0.85", names)

    def test_candidate_from_metrics_skips_ensemble_metrics(self):
        tmpdir = Path(__file__).resolve().parent / f"_tmp_ensemble_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=False)
        try:
            (tmpdir / "best_action_model.pt").write_bytes(b"placeholder")
            metrics_path = tmpdir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "model_type": "ensemble",
                        "labels": LABELS,
                        "final_validation": {"accuracy": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            self.assertIsNone(
                candidate_from_metrics_path(metrics_path, labels=LABELS, target_metric="accuracy")
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

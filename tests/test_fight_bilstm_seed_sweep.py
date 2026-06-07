from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from fight_bilstm_seed_sweep import metric_score, resolve_seed_list


class FightBiLstmSeedSweepTests(unittest.TestCase):
    def test_resolve_seed_list_uses_configured_values(self) -> None:
        config = {"fight_bilstm": {"seeds": [42, "43", 44]}}

        self.assertEqual(resolve_seed_list(config), [42, 43, 44])

    def test_resolve_seed_list_defaults_to_three_seeds(self) -> None:
        self.assertEqual(resolve_seed_list({}), [42, 43, 44])

    def test_metric_score_prefers_holdout_macro_f1_then_normal_f1(self) -> None:
        weak_normal = {
            "holdout_test": {
                "macro_f1": 0.70,
                "per_class": [
                    {"label": "normal", "f1": 0.50},
                    {"label": "violence", "f1": 0.90},
                ],
            }
        }
        strong_normal = {
            "holdout_test": {
                "macro_f1": 0.70,
                "per_class": [
                    {"label": "normal", "f1": 0.70},
                    {"label": "violence", "f1": 0.70},
                ],
            }
        }

        self.assertGreater(metric_score(strong_normal), metric_score(weak_normal))

    def test_metric_score_uses_recommended_fusion_when_selected(self) -> None:
        metrics = {
            "recommended_result": "fusion",
            "holdout_test": {"macro_f1": 0.50, "per_class": []},
            "recommended_holdout_test": {
                "macro_f1": 0.80,
                "per_class": [
                    {"label": "normal", "f1": 0.78},
                    {"label": "violence", "f1": 0.82},
                ],
            },
            "recommended_validation": {"macro_f1": 0.79},
        }

        self.assertEqual(metric_score(metrics), (0.8, 0.78, 0.82, 0.79))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
import sys
import json
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from auto_tune_action_training import (
    build_exploration_trial_specs,
    build_initial_trial_specs,
    build_training_update_payload,
    compact_ensemble_result,
    compact_hybrid_result,
    extract_metric_feedback,
    objective_score,
    objective_score_for_metric,
    update_source_config_with_best,
)


LABELS = ["violence", "collapse", "abduction", "loitering"]


class AutoTuneActionTrainingTests(unittest.TestCase):
    def test_extract_metric_feedback_finds_missing_predictions_and_overprediction(self):
        metrics = sample_metrics()

        feedback = extract_metric_feedback(metrics, labels=LABELS, target_recall=0.35)

        self.assertEqual(feedback["missing_prediction_labels"], ["collapse", "loitering"])
        self.assertEqual(feedback["missing_or_low_recall_labels"], ["collapse", "loitering"])
        self.assertEqual(feedback["overpredicted_label"], "abduction")

    def test_initial_trials_boost_missing_classes_and_suppress_overpredicted_class(self):
        trials = build_initial_trial_specs(
            labels=LABELS,
            base_training={"class_weight_multipliers": {"abduction": 1.25}},
            previous_metrics=sample_metrics(),
            target_recall=0.35,
        )

        mild = next(trial for trial in trials if trial.name == "recall_recovery_mild")
        multipliers = mild.params["class_weight_multipliers"]
        self.assertEqual(multipliers["collapse"], 2.0)
        self.assertEqual(multipliers["loitering"], 2.0)
        self.assertNotIn("abduction", multipliers)

    def test_objective_score_penalizes_missing_prediction_classes(self):
        clean = {
            "final_validation": {
                "macro_f1": 0.3,
                "balanced_accuracy": 0.4,
                "per_class": [{"recall": 0.2}, {"recall": 0.3}],
            },
            "validation_error_analysis": {"missing_prediction_classes": []},
        }
        missing = {
            "final_validation": {
                "macro_f1": 0.3,
                "balanced_accuracy": 0.4,
                "per_class": [{"recall": 0.0}, {"recall": 0.3}],
            },
            "validation_error_analysis": {"missing_prediction_classes": ["collapse"]},
        }

        self.assertGreater(objective_score(clean), objective_score(missing))

    def test_accuracy_objective_can_prefer_higher_accuracy(self):
        higher_accuracy = {
            "final_validation": {
                "accuracy": 0.6,
                "macro_f1": 0.3,
                "balanced_accuracy": 0.35,
                "per_class": [{"recall": 0.2}, {"recall": 0.3}],
            },
            "validation_error_analysis": {"missing_prediction_classes": []},
        }
        higher_macro_f1 = {
            "final_validation": {
                "accuracy": 0.4,
                "macro_f1": 0.5,
                "balanced_accuracy": 0.4,
                "per_class": [{"recall": 0.2}, {"recall": 0.3}],
            },
            "validation_error_analysis": {"missing_prediction_classes": []},
        }

        self.assertGreater(
            objective_score_for_metric(higher_accuracy, target_metric="accuracy"),
            objective_score_for_metric(higher_macro_f1, target_metric="accuracy"),
        )

    def test_training_update_payload_keeps_only_json_tunable_keys(self):
        payload = build_training_update_payload(
            {
                "class_weight_multipliers": {"collapse": 1.2},
                "learning_rate": 0.0003,
                "resume_from_best": False,
                "batch_size": 96,
            }
        )

        self.assertEqual(
            payload,
            {
                "class_weight_multipliers": {"collapse": 1.2},
                "learning_rate": 0.0003,
            },
        )

    def test_update_source_config_with_best_writes_training_values(self):
        tmpdir = Path(__file__).resolve().parent / f"_tmp_auto_tune_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=False)
        try:
            config_path = tmpdir / "config.json"
            best_config_path = tmpdir / "best_training_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "training": {
                            "learning_rate": 0.0005,
                            "adaptive_class_weighting": {"enabled": True},
                        },
                        "continual_learning": {"resume_from_best": True},
                        "auto_tune": {},
                    }
                ),
                encoding="utf-8",
            )
            update_source_config_with_best(
                config_path,
                best_result={
                    "trial_number": 1,
                    "name": "best",
                    "params": {
                        "learning_rate": 0.0003,
                        "class_weight_multipliers": {"collapse": 1.35},
                    },
                },
                best_config_path=best_config_path,
                target_metric="balanced",
            )

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["training"]["learning_rate"], 0.0003)
            self.assertEqual(updated["training"]["class_weight_multipliers"], {"collapse": 1.35})
            self.assertFalse(updated["training"]["adaptive_class_weighting"]["enabled"])
            self.assertFalse(updated["continual_learning"]["resume_from_best"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_exploration_trials_include_seed_sweep(self):
        trials = build_exploration_trial_specs(
            labels=LABELS,
            base_training={
                "class_weight_multipliers": {"collapse": 1.35},
                "learning_rate": 0.0005,
            },
            previous_metrics=sample_metrics(),
            auto_tune_config={"seed_sweep_enabled": True, "seeds": [7, 13]},
            target_recall=0.35,
        )

        names = {trial.name for trial in trials}
        self.assertIn("seed_sweep_7", names)
        self.assertIn("seed_sweep_low_lr_13", names)

    def test_compact_ensemble_result_keeps_dashboard_summary_fields(self):
        compact = compact_ensemble_result(
            {
                "summary_path": "summary.json",
                "output_dir": "ensembles/run",
                "promoted": True,
                "best_result": {
                    "score": 0.52,
                    "target_metric": "accuracy",
                    "weight_mode": "score",
                    "class_bias": {},
                    "weights": [0.7, 0.3],
                    "members": [{"name": "a"}, {"name": "b"}],
                    "metrics": {
                        "final_validation": {
                            "accuracy": 0.53,
                            "macro_f1": 0.52,
                            "balanced_accuracy": 0.54,
                        }
                    },
                    "test_validation": {"accuracy": 0.45},
                },
            }
        )

        self.assertEqual(compact["val_accuracy"], 0.53)
        self.assertEqual(compact["test_accuracy"], 0.45)
        self.assertTrue(compact["promoted"])

    def test_compact_hybrid_result_keeps_key_metrics(self):
        compact = compact_hybrid_result(
            {
                "summary_path": "hybrid.json",
                "output_dir": "hybrid/run",
                "promoted": True,
                "best_result": {
                    "feature_model": "extra_trees",
                    "feature_weight": 0.04,
                    "neural_weight": 0.96,
                    "class_bias": {},
                    "metrics": {
                        "final_validation": {
                            "accuracy": 0.535,
                            "macro_f1": 0.53,
                            "balanced_accuracy": 0.54,
                        }
                    },
                    "test_validation": {"accuracy": 0.466},
                },
            }
        )

        self.assertEqual(compact["feature_model"], "extra_trees")
        self.assertEqual(compact["val_accuracy"], 0.535)
        self.assertEqual(compact["test_accuracy"], 0.466)


def sample_metrics():
    return {
        "final_validation": {
            "confusion_matrix": [
                [15, 0, 19, 0],
                [12, 0, 24, 0],
                [4, 0, 15, 0],
                [11, 0, 27, 0],
            ],
            "per_class": [
                {"label": "violence", "recall": 0.441176, "support": 34},
                {"label": "collapse", "recall": 0.0, "support": 36},
                {"label": "abduction", "recall": 0.789474, "support": 19},
                {"label": "loitering", "recall": 0.0, "support": 38},
            ],
        }
    }


if __name__ == "__main__":
    unittest.main()

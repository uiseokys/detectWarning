from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from training_config import load_action_training_config, resolve_pages_sync_config


class TrainingConfigTests(unittest.TestCase):
    def test_load_action_training_config_applies_defaults(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "dataset": {
                            "label_mapping": {
                                "폭행": "violence",
                                "일반행동": "normal",
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = load_action_training_config(config_path)
            self.assertEqual(config["dataset"]["target_labels"], ["violence", "normal"])
            self.assertIn("training", config)
            self.assertEqual(config["training"]["batch_size"], 16)
            self.assertEqual(config["training"]["loss"], "focal")
            self.assertTrue(config["training"]["balanced_sampler"])
            self.assertEqual(config["training"]["class_weight_multipliers"], {})
            self.assertTrue(config["training"]["adaptive_class_weighting"]["enabled"])

    def test_resolve_pages_sync_config_resolves_relative_pages_dir(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            pages_dir = base_dir / "pages"
            pages_dir.mkdir()

            resolved = resolve_pages_sync_config(
                {
                    "pages_sync": {
                        "enabled": True,
                        "pages_dir": "pages",
                        "project_name": "demo",
                    }
                },
                base_dir,
            )

            self.assertTrue(resolved["enabled"])
            self.assertEqual(resolved["pages_dir"], pages_dir.resolve())
            self.assertEqual(resolved["project_name"], "demo")


if __name__ == "__main__":
    unittest.main()

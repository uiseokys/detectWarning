from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from action_training_pipeline import (
    DownloadedItem,
    iter_jsonl_entries,
    split_dataset,
    write_downloaded_items_manifest,
)


class ActionTrainingPipelineTests(unittest.TestCase):
    def test_iter_jsonl_entries_skips_malformed_lines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "items.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"item_id": "a"}, ensure_ascii=False),
                        "{broken",
                        json.dumps(["not", "a", "dict"], ensure_ascii=False),
                        json.dumps({"item_id": "b"}, ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )

            rows = list(iter_jsonl_entries(manifest))

        self.assertEqual([row["item_id"] for row in rows], ["a", "b"])

    def test_split_dataset_requires_train_and_validation_samples(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            paths = {
                "current_split_train": workspace / "train.jsonl",
                "current_split_val": workspace / "val.jsonl",
                "current_split_test": workspace / "test.jsonl",
            }
            item = DownloadedItem(
                item_id="one",
                source_label="source",
                target_label="normal",
                video_path=workspace / "one.mp4",
                download_url="local",
                metadata={},
            )

            with self.assertRaisesRegex(RuntimeError, "학습/검증 split"):
                split_dataset(
                    [item],
                    {"split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}},
                    paths,
                )

    def test_write_downloaded_items_manifest_uses_expected_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "raw.jsonl"
            item = DownloadedItem(
                item_id="one",
                source_label="source",
                target_label="normal",
                video_path=Path(tmpdir) / "one.mp4",
                download_url="local",
                metadata={"relative_path": "one.mp4"},
            )

            write_downloaded_items_manifest(manifest, [item])
            rows = list(iter_jsonl_entries(manifest))

        self.assertEqual(rows[0]["item_id"], "one")
        self.assertEqual(rows[0]["target_label"], "normal")
        self.assertIn("metadata", rows[0])


if __name__ == "__main__":
    unittest.main()

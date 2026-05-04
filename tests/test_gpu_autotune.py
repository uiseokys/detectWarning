from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from gpu_autotune import build_gpu_auto_tune_plan, first_available_int, parse_cuda_device_index


class GpuAutoTuneTests(unittest.TestCase):
    def test_parse_cuda_device_index(self) -> None:
        self.assertEqual(parse_cuda_device_index("cuda"), 0)
        self.assertEqual(parse_cuda_device_index("cuda:1"), 1)
        self.assertIsNone(parse_cuda_device_index("cpu"))
        self.assertEqual(first_available_int(0, 1), 0)

    def test_build_plan_increases_preprocess_and_training_when_vram_is_free(self) -> None:
        config = {
            "gpu_auto_tune": {"enabled": True},
            "preprocess": {
                "detector_batch_size": 24,
                "video_prefetch_workers": 2,
                "person_imgsz": 640,
                "max_frames_to_scan": 160,
            },
            "training": {
                "batch_size": 48,
                "eval_batch_size": 96,
            },
        }
        plan = build_gpu_auto_tune_plan(
            config,
            stage="all",
            status={
                "available": True,
                "index": 0,
                "memory_total_mb": 24576,
                "memory_used_mb": 8192,
                "memory_free_mb": 16384,
                "utilization_gpu_percent": 45,
            },
        )

        self.assertTrue(plan["enabled"])
        self.assertEqual(plan["updates"]["preprocess"]["detector_batch_size"], 64)
        self.assertEqual(plan["updates"]["preprocess"]["video_prefetch_workers"], 4)
        self.assertEqual(plan["updates"]["preprocess"]["person_imgsz"], 960)
        self.assertEqual(plan["updates"]["preprocess"]["max_frames_to_scan"], 240)
        self.assertEqual(plan["updates"]["training"]["batch_size"], 96)
        self.assertEqual(plan["updates"]["training"]["eval_batch_size"], 192)

    def test_build_plan_respects_stage(self) -> None:
        config = {
            "gpu_auto_tune": {"enabled": True},
            "preprocess": {"detector_batch_size": 24},
            "training": {"batch_size": 32, "eval_batch_size": 64},
        }
        plan = build_gpu_auto_tune_plan(
            config,
            stage="train",
            status={
                "available": True,
                "index": 0,
                "memory_total_mb": 24576,
                "memory_used_mb": 4096,
                "memory_free_mb": 20480,
                "utilization_gpu_percent": 20,
            },
        )

        self.assertNotIn("preprocess", plan["updates"])
        self.assertIn("training", plan["updates"])

    def test_build_plan_keeps_settings_when_free_memory_is_reserved(self) -> None:
        config = {
            "gpu_auto_tune": {"enabled": True, "reserve_memory_mb": 4096},
            "preprocess": {"detector_batch_size": 24},
            "training": {"batch_size": 32},
        }
        plan = build_gpu_auto_tune_plan(
            config,
            stage="all",
            status={
                "available": True,
                "index": 0,
                "memory_total_mb": 12288,
                "memory_used_mb": 9216,
                "memory_free_mb": 3072,
                "utilization_gpu_percent": 80,
            },
        )

        self.assertEqual(plan["updates"], {})


if __name__ == "__main__":
    unittest.main()

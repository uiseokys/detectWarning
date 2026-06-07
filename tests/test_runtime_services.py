from __future__ import annotations

from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from stt_service import clova_language_code, normalize_language
from system_monitoring import SystemMonitor
from runtime_health_check import summarize_inference


class RuntimeServiceTests(unittest.TestCase):
    def test_stt_language_helpers_normalize_clova_codes(self) -> None:
        self.assertEqual(normalize_language("ko-KR"), "ko")
        self.assertEqual(clova_language_code("ko-KR"), "Kor")
        self.assertEqual(clova_language_code("en-US"), "Eng")
        self.assertEqual(clova_language_code("ja-JP"), "Jpn")
        self.assertEqual(clova_language_code("zh-CN"), "Chn")

    def test_system_monitor_snapshot_is_available_without_nvidia_smi(self) -> None:
        args = SimpleNamespace(
            stt_provider="clova",
            yolo_device="cuda:0",
            stt_device="cpu",
            stt_compute_type="int8",
            stt_beam_size=3,
            stt_best_of=3,
        )
        audio_queue: queue.Queue = queue.Queue()
        with patch("system_monitoring.shutil.which", return_value=None):
            monitor = SystemMonitor(args, audio_queue)

        snapshot = monitor.get_snapshot()

        self.assertEqual(snapshot["stt_provider"], "clova")
        self.assertEqual(snapshot["yolo_device"], "cuda:0")
        self.assertEqual(snapshot["audio_queue_size"], 0)
        self.assertEqual(snapshot["gpu_status"], "nvidia-smi not found")

    def test_runtime_health_summary_includes_webrtc_error(self) -> None:
        summary = summarize_inference(
            {
                "sessions": 1,
                "webrtc_connections": 0,
                "webrtc": {"lastError": "peer_connection_failed"},
                "app_backend_post": {
                    "queue_size": 0,
                    "circuit_breaker": {"open": False, "failure_count": 0},
                },
            }
        )

        self.assertIn("webrtc_err=peer_connection_failed", summary)


if __name__ == "__main__":
    unittest.main()

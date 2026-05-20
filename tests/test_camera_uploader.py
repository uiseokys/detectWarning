from __future__ import annotations

from pathlib import Path
from collections import deque
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

with patch.dict(sys.modules, {"cv2": SimpleNamespace()}):
    from camera_uploader import AudioStreamer, list_input_devices


class CameraUploaderTests(unittest.TestCase):
    def test_list_input_devices_returns_all_input_capable_devices(self) -> None:
        fake_sounddevice = SimpleNamespace(
            query_devices=lambda: [
                {"name": "Mic A", "max_input_channels": 2, "default_samplerate": 48000},
                {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000},
                {"name": "Mic B", "max_input_channels": 1, "default_samplerate": 44100},
            ]
        )

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
            devices = list_input_devices()

        self.assertEqual(
            devices,
            [
                (0, "Mic A", 2, 48000.0),
                (2, "Mic B", 1, 44100.0),
            ],
        )

    def test_audio_filter_rejects_quiet_noise_and_accepts_voice_peak(self) -> None:
        streamer = AudioStreamer(
            session=SimpleNamespace(),
            server_url="http://127.0.0.1:8000",
            client_id="test",
            timeout_seconds=1.0,
            input_device=None,
            phrase_seconds=1.2,
            silence_seconds=0.25,
        )

        self.assertFalse(
            streamer._should_send_audio(
                bytearray(int(16000 * 0.5 * 2)),
                deque([0.005, 0.006]),
                0.004,
                0.020,
            )
        )
        self.assertTrue(
            streamer._should_send_audio(
                bytearray(int(16000 * 0.5 * 2)),
                deque([0.021]),
                0.004,
                0.020,
            )
        )
        self.assertTrue(
            streamer._should_send_audio(
                bytearray(int(16000 * 0.7 * 2)),
                deque([0.007, 0.007]),
                0.004,
                0.020,
            )
        )


if __name__ == "__main__":
    unittest.main()

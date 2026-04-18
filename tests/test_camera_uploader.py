from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

with patch.dict(sys.modules, {"cv2": SimpleNamespace()}):
    from camera_uploader import list_input_devices


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


if __name__ == "__main__":
    unittest.main()

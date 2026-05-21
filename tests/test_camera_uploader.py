from __future__ import annotations

from pathlib import Path
from collections import deque
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

with patch.dict(sys.modules, {"cv2": SimpleNamespace()}):
    from camera_uploader import (
        AudioStreamer,
        build_client_id,
        clamp_upload_hint,
        load_or_create_client_id,
        list_input_devices,
    )


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

    def test_audio_amplifier_raises_quiet_signal_without_clipping(self) -> None:
        streamer = AudioStreamer(
            session=SimpleNamespace(),
            server_url="http://127.0.0.1:8000",
            client_id="test",
            timeout_seconds=1.0,
            input_device=None,
            phrase_seconds=1.2,
            silence_seconds=0.25,
            gain=4.0,
            auto_gain_target=0.08,
        )
        quiet = (np.ones(1600, dtype=np.int16) * 500).tobytes()
        amplified = np.frombuffer(streamer._amplify_pcm(quiet), dtype=np.int16)

        self.assertGreater(int(amplified.max()), 500)
        self.assertLessEqual(int(amplified.max()), 32767)

    def test_audio_queue_preserves_chunks_until_full(self) -> None:
        streamer = AudioStreamer(
            session=SimpleNamespace(),
            server_url="http://127.0.0.1:8000",
            client_id="test",
            timeout_seconds=1.0,
            input_device=None,
            phrase_seconds=1.2,
            silence_seconds=0.25,
        )

        for index in range(3):
            streamer._queue_latest_audio(bytes([index]))

        self.assertEqual(
            [streamer._send_queue.get_nowait() for _ in range(3)],
            [b"\x00", b"\x01", b"\x02"],
        )

    def test_upload_hint_allows_higher_single_camera_fps_up_to_configured_cap(self) -> None:
        fps, width = clamp_upload_hint(
            hinted_fps=24.0,
            hinted_width=1120,
            configured_max_fps=24.0,
            configured_frame_width=1120,
        )

        self.assertEqual(fps, 24.0)
        self.assertEqual(width, 1120)

        fps, width = clamp_upload_hint(
            hinted_fps=24.0,
            hinted_width=1120,
            configured_max_fps=18.0,
            configured_frame_width=960,
        )

        self.assertEqual(fps, 18.0)
        self.assertEqual(width, 960)

    def test_auto_client_id_is_stable_for_same_device_and_source(self) -> None:
        first = build_client_id("", source="0", host="MacBook Pro")
        second = build_client_id("", source="0", host="MacBook Pro")
        other_source = build_client_id("", source="iphone", host="MacBook Pro")

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("MacBook-Pro-"))
        self.assertNotEqual(first, other_source)

    def test_persistent_auto_client_id_survives_source_changes(self) -> None:
        client_id_path = Path.cwd() / ".camera_client_id_test.txt"
        client_id_path.unlink(missing_ok=True)
        try:
            first = load_or_create_client_id("", source="0", host="MacBook Pro", client_id_file=client_id_path)
            second = load_or_create_client_id("", source="iphone", host="MacBook Pro", client_id_file=client_id_path)
        finally:
            client_id_path.unlink(missing_ok=True)

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("MacBook-Pro-"))

    def test_explicit_client_id_bypasses_persistent_file(self) -> None:
        client_id_path = Path.cwd() / ".camera_client_id_explicit_test.txt"
        client_id_path.unlink(missing_ok=True)
        try:
            client_id = load_or_create_client_id(
                "team-macbook-camera",
                source="iphone",
                host="MacBook Pro",
                client_id_file=client_id_path,
            )
        finally:
            client_id_path.unlink(missing_ok=True)

        self.assertEqual(client_id, "team-macbook-camera")


if __name__ == "__main__":
    unittest.main()

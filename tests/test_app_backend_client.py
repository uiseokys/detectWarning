from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from app_backend_client import AppBackendEventClient


class AppBackendEventClientTests(unittest.TestCase):
    def test_send_warning_event_excludes_transcript(self) -> None:
        client = AppBackendEventClient(
            base_url="https://desktop-phtp8km.tailc597eb.ts.net",
            cctv_code="7K29QB",
            token="secret",
        )
        client.session.post = Mock()
        client.session.post.return_value.raise_for_status = Mock()
        event = SimpleNamespace(
            timestamp="2026-05-19T22:20:00+09:00",
            transcript="살려주세요",
            audio_level=0.2,
        )
        assessment = SimpleNamespace(
            level="HIGH",
            score=82,
            audio_score=60,
            video_score=40,
            categories=["피해자 구조 요청"],
            context_flags=[],
            reasons=["audio danger"],
        )

        self.assertTrue(client.send_warning_event(event, assessment))

        _url, kwargs = client.session.post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["cctvCode"], "7K29QB")
        self.assertEqual(payload["riskLevel"], "danger")
        self.assertEqual(payload["riskClass"], "audio_signal")
        self.assertTrue(payload["audioRiskSignalDetected"])
        self.assertNotIn("transcript", payload)

    def test_disabled_without_url_code_or_token(self) -> None:
        client = AppBackendEventClient(base_url="", cctv_code="7K29QB", token="secret")
        self.assertFalse(client.enabled)


if __name__ == "__main__":
    unittest.main()

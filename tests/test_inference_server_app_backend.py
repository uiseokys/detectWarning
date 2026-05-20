from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from inference_server import (
    build_app_backend_url,
    build_backend_cctv_payload,
    build_backend_event_payload,
    map_backend_risk_level,
)


class InferenceServerAppBackendTests(unittest.TestCase):
    def test_backend_event_payload_uses_pairing_code_and_excludes_raw_speech(self) -> None:
        risk = SimpleNamespace(
            score=82,
            level="HIGH",
            audio_score=35,
            video_score=70,
            speech_match_quality="strong",
            categories=["help_request"],
            context_flags=["stable_action"],
            reasons=["matched private transcript should not be forwarded"],
            matched_keywords=["help"],
        )
        action = SimpleNamespace(label="collapse", confidence=0.87)

        payload = build_backend_event_payload("7k29qb", risk, action)

        self.assertEqual(payload["cctvCode"], "7K29QB")
        self.assertEqual(payload["riskLevel"], "danger")
        self.assertEqual(payload["riskClass"], "fall")
        self.assertEqual(payload["riskScore"], 82)
        self.assertTrue(payload["audioRiskSignalDetected"])
        self.assertIsNone(payload["snapshotUrl"])
        self.assertIsNone(payload["clipUrl"])
        self.assertNotIn("transcript", payload)
        self.assertNotIn("matchedKeywords", payload)

    def test_cctv_sync_payload_is_code_only_metadata(self) -> None:
        payload = build_backend_cctv_payload(
            "ab12cd",
            name="지하주차장 B2-03",
            location="서울 A 현장 / 지하 2층",
            status="정상",
            latest_risk_score=12,
        )

        self.assertEqual(payload["code"], "AB12CD")
        self.assertEqual(payload["name"], "지하주차장 B2-03")
        self.assertEqual(payload["location"], "서울 A 현장 / 지하 2층")
        self.assertEqual(payload["status"], "정상")
        self.assertEqual(payload["latestRiskScore"], 12)
        self.assertIsNone(payload["streamUrl"])
        self.assertIsNone(payload["inferenceStreamUrl"])
        self.assertNotIn("transcript", payload)
        self.assertNotIn("speechText", payload)

    def test_url_and_level_mapping(self) -> None:
        self.assertEqual(
            build_app_backend_url("https://desktop-phtp8km.tailc597eb.ts.net/", "inference/events"),
            "https://desktop-phtp8km.tailc597eb.ts.net/inference/events",
        )
        self.assertEqual(map_backend_risk_level("CRITICAL", 91), "danger")
        self.assertEqual(map_backend_risk_level("MEDIUM", 50), "warning")


if __name__ == "__main__":
    unittest.main()

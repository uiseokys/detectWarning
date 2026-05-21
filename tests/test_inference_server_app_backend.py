from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from inference_server import (
    AudioJob,
    build_app_backend_url,
    build_adaptive_upload_hint,
    build_backend_cctv_payload,
    build_backend_event_payload,
    build_client_frame_url,
    build_client_cctv_code,
    build_detection_effect_summary,
    build_user_risk_presentation,
    enqueue_latest_audio_job,
    generate_server_instance_id,
    load_or_create_server_identity,
    map_backend_risk_level,
    should_send_backend_event,
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

    def test_user_risk_presentation_groups_noisy_internal_categories(self) -> None:
        risk = SimpleNamespace(
            score=71,
            audio_score=0,
            video_score=71,
            speech_match_quality="",
            categories=["action:collapse", "장시간 쓰러짐", "AI:stable:collapse"],
            context_flags=["사람:1", "AI:collapse:0.81"],
            reasons=["action:collapse"],
        )
        action = SimpleNamespace(label="collapse", confidence=0.81)

        presentation = build_user_risk_presentation(risk, action)

        self.assertEqual(presentation["categories"], ["쓰러짐 위험"])
        self.assertEqual(presentation["reasons"], ["영상에서 쓰러짐 위험 감지"])

    def test_backend_video_reason_uses_user_facing_categories(self) -> None:
        risk = SimpleNamespace(
            score=80,
            level="HIGH",
            audio_score=30,
            video_score=55,
            speech_match_quality="strong",
            categories=["action:violence", "matched private transcript should not be forwarded"],
            context_flags=["AI:stable:violence"],
            reasons=["matched private transcript should not be forwarded"],
            matched_keywords=["help"],
        )
        action = SimpleNamespace(label="violence", confidence=0.9)

        payload = build_backend_event_payload("7k29qb", risk, action)

        self.assertIn("폭력/몸싸움 위험", payload["videoReason"])
        self.assertIn("음성 신호로 위험 판단 강화", payload["videoReason"])
        self.assertNotIn("matched private transcript", payload["videoReason"])

    def test_backend_event_payload_includes_safe_risk_detail_metadata(self) -> None:
        risk = SimpleNamespace(
            score=58,
            level="MEDIUM",
            audio_score=58,
            video_score=0,
            raw_score=58,
            video_only_score=0,
            audio_video_gain=58,
            audio_confirmed_class="",
            speech_match_quality="strong",
            categories=["구조 요청"],
            context_flags=["loud_audio"],
            reasons=["matched private transcript should not be forwarded"],
            matched_keywords=["help"],
        )
        action = SimpleNamespace(label="normal", confidence=0.0)

        payload = build_backend_event_payload("7k29qb", risk, action)

        self.assertEqual(payload["riskStage"], "위험 의심")
        self.assertEqual(payload["riskCategories"], ["음성 기반 위험 의심"])
        self.assertEqual(payload["riskReasons"], ["영상에서는 확정하지 못했지만 음성 위험 신호 감지"])
        self.assertEqual(payload["riskSignalSource"], "audio")
        self.assertTrue(payload["audioLiftedRisk"])
        self.assertEqual(payload["videoOnlyScore"], 0)
        self.assertEqual(payload["audioVideoGain"], 58)
        self.assertNotIn("transcript", payload)
        self.assertNotIn("matchedKeywords", payload)
        self.assertNotIn("matched private transcript", str(payload))

    def test_untrained_abduction_is_not_sent_as_backend_class(self) -> None:
        risk = SimpleNamespace(
            score=76,
            level="HIGH",
            audio_score=0,
            video_score=76,
            speech_match_quality="",
            categories=["action:abduction"],
            context_flags=["AI:unsupported_action:abduction"],
            reasons=["action:abduction"],
            matched_keywords=[],
        )
        action = SimpleNamespace(label="abduction", confidence=0.9)

        payload = build_backend_event_payload("7k29qb", risk, action)
        presentation = build_user_risk_presentation(risk, action)

        self.assertEqual(payload["riskClass"], "abnormal")
        self.assertNotEqual(payload["riskClass"], "abduction")
        self.assertNotIn("abduction", payload["videoReason"].lower())
        self.assertNotIn("강제", ",".join(presentation["categories"]))

    def test_supported_video_action_is_presented_as_danger(self) -> None:
        risk = SimpleNamespace(
            score=68,
            level="HIGH",
            audio_score=0,
            video_score=68,
            speech_match_quality="",
            categories=["action:collapse"],
            context_flags=["AI:collapse:0.82"],
            reasons=["action:collapse"],
            matched_keywords=[],
        )
        action = SimpleNamespace(label="collapse", confidence=0.82)

        presentation = build_user_risk_presentation(risk, action)

        self.assertEqual(presentation["stage"], "위험")
        self.assertEqual(presentation["categories"], ["쓰러짐 위험"])
        self.assertEqual(presentation["signal_source"], "video")

    def test_audio_risk_without_video_action_is_presented_as_suspicion(self) -> None:
        risk = SimpleNamespace(
            score=58,
            level="MEDIUM",
            audio_score=58,
            video_score=0,
            speech_match_quality="strong",
            categories=["구조 요청"],
            context_flags=["loud_audio"],
            reasons=["matched private transcript should not be forwarded"],
            matched_keywords=["help"],
        )
        action = SimpleNamespace(label="normal", confidence=0.0)

        presentation = build_user_risk_presentation(risk, action)

        self.assertEqual(presentation["stage"], "위험 의심")
        self.assertEqual(presentation["categories"], ["음성 기반 위험 의심"])
        self.assertEqual(presentation["signal_source"], "audio")
        self.assertNotIn("matched private transcript", ",".join(presentation["reasons"]))

    def test_loud_audio_without_risk_words_is_presented_as_caution(self) -> None:
        risk = SimpleNamespace(
            score=28,
            level="ELEVATED",
            audio_score=12,
            video_score=0,
            speech_match_quality="",
            categories=[],
            context_flags=["loud_audio"],
            reasons=[],
            matched_keywords=[],
        )
        action = SimpleNamespace(label="normal", confidence=0.0)

        presentation = build_user_risk_presentation(risk, action)

        self.assertEqual(presentation["stage"], "주의")
        self.assertEqual(presentation["categories"], ["큰 소리 감지"])

    def test_audio_suspicion_requires_stronger_audio_than_caution(self) -> None:
        risk = SimpleNamespace(
            score=32,
            level="ELEVATED",
            audio_score=24,
            video_score=0,
            speech_match_quality="weak",
            categories=["audio:risk"],
            context_flags=["loud_audio"],
            reasons=[],
            matched_keywords=[],
        )
        action = SimpleNamespace(label="normal", confidence=0.0)

        presentation = build_user_risk_presentation(risk, action)

        self.assertEqual(presentation["stage"], "\uc8fc\uc758")
        self.assertEqual(presentation["categories"], ["\ud070 \uc18c\ub9ac \uac10\uc9c0"])

    def test_detection_effect_summary_reports_audio_gain_for_demo(self) -> None:
        summary = build_detection_effect_summary(
            [
                {
                    "risk_stage": "\uc704\ud5d8",
                    "risk_score": 82,
                    "risk_video_only_score": 58,
                    "risk_audio_video_gain": 24,
                    "risk_audio_lifted": True,
                    "risk_signal_source": "audio_video",
                },
                {
                    "risk_stage": "\uc704\ud5d8 \uc758\uc2ec",
                    "risk_score": 55,
                    "risk_video_only_score": 0,
                    "risk_audio_video_gain": 55,
                    "risk_audio_lifted": True,
                    "risk_signal_source": "audio",
                },
                {
                    "risk_stage": "\uc704\ud5d8",
                    "risk_score": 68,
                    "risk_video_only_score": 68,
                    "risk_audio_video_gain": 0,
                    "risk_audio_lifted": False,
                    "risk_signal_source": "video",
                },
            ]
        )

        self.assertEqual(summary["totalEvents"], 3)
        self.assertEqual(summary["audioLiftedEvents"], 2)
        self.assertEqual(summary["audioOnlySuspicionEvents"], 1)
        self.assertEqual(summary["videoOnlyDangerEvents"], 1)
        self.assertEqual(summary["averageAudioVideoGain"], 26.3)
        self.assertEqual(summary["maxAudioVideoGain"], 55)

    def test_backend_event_cooldown_suppresses_repeated_same_class(self) -> None:
        session = SimpleNamespace(last_backend_event_at=0.0, last_backend_event_signature="")
        payload = {
            "riskLevel": "warning",
            "riskClass": "collapse",
            "riskScore": 64,
            "riskStage": "\uc704\ud5d8",
            "audioRiskSignalDetected": False,
        }

        self.assertTrue(should_send_backend_event(session, payload))
        self.assertFalse(should_send_backend_event(session, dict(payload)))

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

    def test_client_frame_url_is_built_for_backend_cctv_sync(self) -> None:
        self.assertEqual(
            build_client_frame_url("http://desktop:8001/", "mac book 01"),
            "http://desktop:8001/api/client/mac-book-01/frame.jpg",
        )
        self.assertEqual(build_client_frame_url("", "macbook"), "")

    def test_url_and_level_mapping(self) -> None:
        self.assertEqual(
            build_app_backend_url("https://desktop-phtp8km.tailc597eb.ts.net/", "inference/events"),
            "https://desktop-phtp8km.tailc597eb.ts.net/inference/events",
        )
        self.assertEqual(map_backend_risk_level("CRITICAL", 91), "danger")
        self.assertEqual(map_backend_risk_level("MEDIUM", 50), "warning")

    def test_server_audio_queue_preserves_same_client_order_until_full(self) -> None:
        import queue

        audio_queue: queue.Queue[AudioJob] = queue.Queue(maxsize=3)
        self.assertTrue(enqueue_latest_audio_job(audio_queue, AudioJob("client", b"1")))
        self.assertTrue(enqueue_latest_audio_job(audio_queue, AudioJob("client", b"2")))
        self.assertTrue(enqueue_latest_audio_job(audio_queue, AudioJob("client", b"3")))

        self.assertEqual([audio_queue.get_nowait().wav_bytes for _ in range(3)], [b"1", b"2", b"3"])

    def test_upload_hint_uses_more_video_budget_when_latency_allows(self) -> None:
        balanced = build_adaptive_upload_hint(latency_ms=120, risk_score=0, action_label="normal")
        risky = build_adaptive_upload_hint(latency_ms=120, risk_score=65, action_label="collapse")
        overloaded = build_adaptive_upload_hint(latency_ms=1300, risk_score=65, action_label="collapse")

        self.assertEqual(balanced["max_fps"], 20.0)
        self.assertEqual(balanced["frame_width"], 960)
        self.assertEqual(risky["max_fps"], 24.0)
        self.assertEqual(risky["frame_width"], 1120)
        self.assertLessEqual(overloaded["max_fps"], 4.0)

    def test_client_cctv_code_is_stable_per_camera(self) -> None:
        first = build_client_cctv_code("7K29QB", "macbook-front")
        second = build_client_cctv_code("7K29QB", "macbook-front")
        other = build_client_cctv_code("7K29QB", "lab-laptop")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue(first.isalnum())
        self.assertNotEqual(first, other)

    def test_server_instance_id_is_stable_for_same_server_machine(self) -> None:
        first = generate_server_instance_id(host="DESKTOP-PHTP8KM", node=123456)
        second = generate_server_instance_id(host="DESKTOP-PHTP8KM", node=123456)
        other = generate_server_instance_id(host="OTHER-SERVER", node=123456)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertNotEqual(first, other)

    def test_server_identity_drops_legacy_pairing_code(self) -> None:
        identity_path = (
            Path(__file__).resolve().parents[1]
            / "training_data"
            / f"server_identity_test_{uuid.uuid4().hex}.json"
        )
        try:
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            identity_path.write_text(
                '{"pairing_code":"OLD123","cctv":{"name":"legacy"}}',
                encoding="utf-8",
            )

            identity = load_or_create_server_identity(identity_path)

            self.assertNotIn("pairing_code", identity)
            self.assertIn("server_instance_id", identity)
            self.assertEqual(identity["cctv"]["name"], "legacy")
            self.assertIsInstance(identity["camera_cctvs"], dict)
        finally:
            identity_path.unlink(missing_ok=True)

    def test_upload_hint_lowers_video_budget_for_multiple_cameras(self) -> None:
        single = build_adaptive_upload_hint(
            latency_ms=120,
            risk_score=65,
            action_label="collapse",
            active_client_count=1,
        )
        multi = build_adaptive_upload_hint(
            latency_ms=120,
            risk_score=65,
            action_label="collapse",
            active_client_count=2,
        )
        crowded = build_adaptive_upload_hint(
            latency_ms=120,
            risk_score=65,
            action_label="collapse",
            active_client_count=3,
        )

        self.assertLess(multi["max_fps"], single["max_fps"])
        self.assertLessEqual(multi["frame_width"], single["frame_width"])
        self.assertLess(crowded["max_fps"], multi["max_fps"])
        self.assertEqual(crowded["reason"], "risk_active_multi_camera")


if __name__ == "__main__":
    unittest.main()

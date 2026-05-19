from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from risk_analyzer import RiskAnalyzer


class RiskAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        rules_path = Path(__file__).resolve().parents[1] / "configs" / "risk_rules.json"
        self.analyzer = RiskAnalyzer(rules_path=rules_path)

    def test_threatening_transcript_increases_risk(self) -> None:
        speech_result = SimpleNamespace(transcript="죽여버리겠다 지금 찾아간다", audio_level=0.12)
        assessment = self.analyzer.update(speech_result, tracked_people=[], face_count=0)

        self.assertGreater(assessment.score, 0)
        self.assertNotEqual(assessment.level, "safe")
        self.assertTrue(assessment.categories)

    def test_daily_overstatement_stays_low(self) -> None:
        speech_result = SimpleNamespace(transcript="배고파 죽겠다", audio_level=0.0)
        assessment = self.analyzer.update(speech_result, tracked_people=[], face_count=0)

        self.assertLess(assessment.score, 40)

    def test_rescue_speech_with_person_becomes_critical(self) -> None:
        speech_result = SimpleNamespace(transcript="\uc0b4\ub824\uc8fc\uc138\uc694", audio_level=0.14)
        assessment = self.analyzer.update(
            speech_result,
            tracked_people=[{"bbox": (0, 0, 80, 160), "movement": 5}],
            face_count=0,
        )

        self.assertEqual(assessment.level, "CRITICAL")
        self.assertGreaterEqual(assessment.audio_score, 80)

    def test_tentative_action_without_person_stays_below_alert(self) -> None:
        action = SimpleNamespace(
            available=True,
            label="collapse",
            confidence=0.5,
            abnormal_score=0.8,
        )
        speech_result = SimpleNamespace(transcript="", audio_level=0.0)
        assessment = self.analyzer.update(
            speech_result,
            tracked_people=[],
            face_count=0,
            action_result=action,
        )

        self.assertLess(assessment.score, 20)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass
class RiskAssessment:
    score: int
    level: str
    reasons: list[str]
    matched_keywords: list[str]


class RiskAnalyzer:
    def __init__(self) -> None:
        self._last_keyword_score = 0.0
        self._last_keyword_reason = ""
        self._last_keyword_matches: list[str] = []
        self._last_keyword_time = 0.0
        self._last_audio_level = 0.0
        self._last_audio_time = 0.0
        self._keyword_window_seconds = 4.0
        self._audio_window_seconds = 2.0
        self._patterns = [
            (("살려줘", "살려 줘", "사려줘", "살려조", "살려죠"), 75, "긴급 도움 요청"),
            (("도와줘", "도와 줘", "도와주세요", "도와줘요"), 60, "도움 요청"),
            (("하지마", "하지 마", "하지마라", "그만해", "그만 해"), 58, "제지 표현"),
            (
                (
                    "왜이러세요",
                    "왜 이러세요",
                    "왜이러시는거에요",
                    "왜 이러시는 거예요",
                    "왜그러세요",
                    "왜 그러세요",
                    "왜그러시는거예요",
                ),
                52,
                "침해 의심 표현",
            ),
            (
                (
                    "이러지마세요",
                    "이러지 마세요",
                    "만지지마세요",
                    "만지지 마세요",
                    "다가오지마세요",
                    "다가오지 마세요",
                    "놔주세요",
                    "놔 주세요",
                    "저리가세요",
                    "저리 가세요",
                ),
                55,
                "거부 표현",
            ),
            (
                (
                    "싫어요",
                    "싫어",
                    "안돼요",
                    "안 돼요",
                    "안돼",
                    "안 돼",
                    "하지말라고",
                    "하지 말라고",
                    "그만하시죠",
                ),
                42,
                "저항 표현",
            ),
            (("불이야", "불이야!", "불났", "불 났", "화재야"), 70, "화재 경고"),
            (("경찰", "신고해", "신고 해", "119", "112"), 45, "신고 요청"),
            (("아파", "죽겠", "죽을 것 같", "무서워"), 35, "고통/불안"),
        ]

    def update(self, speech_result, people_count: int, face_count: int) -> RiskAssessment:
        now = monotonic()
        transcript = (speech_result.transcript or "").strip()
        if transcript:
            normalized = self._normalize(transcript)
            matches, keyword_score, keyword_reason = self._score_keywords(normalized)
            if keyword_score > 0:
                self._last_keyword_score = keyword_score
                self._last_keyword_reason = keyword_reason
                self._last_keyword_matches = matches
                self._last_keyword_time = now

        if speech_result.audio_level > 0:
            self._last_audio_level = speech_result.audio_level
            self._last_audio_time = now

        score = 0.0
        reasons: list[str] = []
        matched_keywords: list[str] = []

        keyword_age = now - self._last_keyword_time
        if self._last_keyword_score > 0 and keyword_age <= self._keyword_window_seconds:
            decay = 1.0 - (keyword_age / self._keyword_window_seconds) * 0.35
            score += self._last_keyword_score * max(decay, 0.65)
            matched_keywords = list(self._last_keyword_matches)
            reasons.append(self._last_keyword_reason)

        audio_age = now - self._last_audio_time
        recent_audio_level = self._last_audio_level if audio_age <= self._audio_window_seconds else 0.0
        audio_score = self._score_audio(recent_audio_level)
        if audio_score > 0:
            score += audio_score
            reasons.append(f"큰 소리:{recent_audio_level:.2f}")

        visual_score = 0
        if people_count > 0:
            visual_score += 6
            reasons.append(f"사람:{people_count}")
        if face_count > 0:
            visual_score += 4
            reasons.append(f"얼굴:{face_count}")
        score += visual_score

        if matched_keywords and people_count > 0:
            score += 10
            reasons.append("키워드+사람")
        if matched_keywords and recent_audio_level >= 0.10:
            score += 10
            reasons.append("키워드+큰 소리")
        if not matched_keywords and recent_audio_level >= 0.18 and people_count > 0:
            score += 12
            reasons.append("큰 소리+사람")
        if matched_keywords and face_count > 0:
            score += 5
            reasons.append("키워드+얼굴")

        score = min(int(round(score)), 100)
        level = self._score_to_level(score)
        return RiskAssessment(
            score=score,
            level=level,
            reasons=reasons[:4],
            matched_keywords=matched_keywords[:3],
        )

    def _score_keywords(self, normalized_text: str) -> tuple[list[str], int, str]:
        best_score = 0
        best_reason = ""
        matches: list[str] = []
        for variants, score, label in self._patterns:
            for variant in variants:
                if self._normalize(variant) in normalized_text:
                    best_score = max(best_score, score)
                    if variant not in matches:
                        matches.append(variant)
                    best_reason = label
                    break
        return matches, best_score, best_reason

    @staticmethod
    def _score_audio(audio_level: float) -> int:
        if audio_level >= 0.22:
            return 24
        if audio_level >= 0.18:
            return 20
        if audio_level >= 0.16:
            return 18
        if audio_level >= 0.10:
            return 10
        if audio_level >= 0.06:
            return 4
        return 0

    @staticmethod
    def _score_to_level(score: int) -> str:
        if score >= 75:
            return "HIGH"
        if score >= 45:
            return "MEDIUM"
        if score >= 20:
            return "ELEVATED"
        return "LOW"

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().strip()
        for old in (" ", "\n", "\t", ".", ",", "!", "?", "'", '"'):
            lowered = lowered.replace(old, "")
        return lowered

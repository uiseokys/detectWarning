from __future__ import annotations

from dataclasses import dataclass
from math import hypot
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
        self._loud_audio_hits: list[float] = []
        self._audio_history: list[tuple[float, float]] = []
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

    def update(self, speech_result, tracked_people, face_count: int) -> RiskAssessment:
        now = monotonic()
        people_count = len(tracked_people)
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
            self._audio_history.append((now, speech_result.audio_level))
        if speech_result.audio_level >= 0.16:
            self._loud_audio_hits.append(now)
        self._loud_audio_hits = [ts for ts in self._loud_audio_hits if now - ts <= 4.0]
        self._audio_history = [(ts, level) for ts, level in self._audio_history if now - ts <= 12.0]

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

        video_categories, video_score = self._score_video(tracked_people)
        if video_score > 0:
            score += video_score
            reasons.extend(video_categories)

        audio_categories, extra_audio_score = self._score_audio_patterns(now, recent_audio_level)
        if extra_audio_score > 0:
            score += extra_audio_score
            reasons.extend(audio_categories)

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

    def _score_audio_patterns(self, now: float, audio_level: float) -> tuple[list[str], int]:
        reasons: list[str] = []
        score = 0
        if audio_level >= 0.22:
            reasons.append("비명 의심")
            score += 18
        if len(self._loud_audio_hits) >= 3:
            reasons.append("반복적 고성")
            score += 14
        surge_ratio, baseline = self._audio_surge_ratio(now, audio_level)
        if baseline > 0:
            if surge_ratio >= 2.6 and audio_level >= 0.09:
                reasons.append(f"고성 급상승:{surge_ratio:.1f}배")
                score += 16
            elif surge_ratio >= 1.9 and audio_level >= 0.07:
                reasons.append(f"음량 급상승:{surge_ratio:.1f}배")
                score += 9
        return reasons, score

    def _audio_surge_ratio(self, now: float, current_level: float) -> tuple[float, float]:
        baseline_levels = [
            level
            for ts, level in self._audio_history
            if 2.0 <= now - ts <= 10.0
        ]
        if not baseline_levels:
            return 0.0, 0.0
        baseline = sum(baseline_levels) / len(baseline_levels)
        baseline = max(baseline, 0.02)
        return current_level / baseline, baseline

    def _score_video(self, tracked_people) -> tuple[list[str], int]:
        reasons: list[str] = []
        score = 0

        for person in tracked_people:
            x, y, w, h = person["bbox"]
            movement = person.get("movement", 0.0)
            stationary_frames = person.get("stationary_frames", 0)
            if h > 0:
                aspect_ratio = w / h
                if aspect_ratio >= 1.1:
                    reasons.append("넘어짐 의심")
                    score = max(score, 16)
                if aspect_ratio >= 1.1 and stationary_frames >= 20:
                    reasons.append("장시간 쓰러짐")
                    score = max(score, 32)
            if movement >= 28:
                reasons.append("빠른 이동")
                score = max(score, 8)

        if len(tracked_people) >= 2:
            if self._detect_fight(tracked_people):
                reasons.append("몸싸움 의심")
                score = max(score, 28)
            if self._detect_chase(tracked_people):
                reasons.append("달리며 추격")
                score = max(score, 22)

        return list(dict.fromkeys(reasons))[:3], score

    @staticmethod
    def _detect_fight(tracked_people) -> bool:
        for i, first in enumerate(tracked_people):
            for second in tracked_people[i + 1 :]:
                if RiskAnalyzer._bbox_overlap_ratio(first["bbox"], second["bbox"]) >= 0.08:
                    return True
                if RiskAnalyzer._centroid_distance(first["bbox"], second["bbox"]) <= 90:
                    if first.get("movement", 0.0) >= 12 and second.get("movement", 0.0) >= 12:
                        return True
        return False

    @staticmethod
    def _detect_chase(tracked_people) -> bool:
        fast_people = [p for p in tracked_people if p.get("movement", 0.0) >= 18]
        if len(fast_people) < 2:
            return False
        for i, first in enumerate(fast_people):
            for second in fast_people[i + 1 :]:
                if RiskAnalyzer._centroid_distance(first["bbox"], second["bbox"]) <= 180:
                    return True
        return False

    @staticmethod
    def _bbox_overlap_ratio(first_box, second_box) -> float:
        fx, fy, fw, fh = first_box
        sx, sy, sw, sh = second_box
        x_left = max(fx, sx)
        y_top = max(fy, sy)
        x_right = min(fx + fw, sx + sw)
        y_bottom = min(fy + fh, sy + sh)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        first_area = fw * fh
        second_area = sw * sh
        smaller_area = max(min(first_area, second_area), 1)
        return intersection / smaller_area

    @staticmethod
    def _centroid_distance(first_box, second_box) -> float:
        fx, fy, fw, fh = first_box
        sx, sy, sw, sh = second_box
        first_center = (fx + fw / 2, fy + fh / 2)
        second_center = (sx + sw / 2, sy + sh / 2)
        return hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])

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

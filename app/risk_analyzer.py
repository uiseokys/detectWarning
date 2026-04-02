from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from time import monotonic


@dataclass(frozen=True)
class CategoryRule:
    code: str
    label: str
    base_score: int
    phrases: tuple[str, ...]


@dataclass
class RiskAssessment:
    score: int
    level: str
    reasons: list[str]
    matched_keywords: list[str]
    categories: list[str]
    context_flags: list[str]


class RiskAnalyzer:
    def __init__(self) -> None:
        self._last_text_score = 0.0
        self._last_text_categories: list[str] = []
        self._last_text_reasons: list[str] = []
        self._last_text_flags: list[str] = []
        self._last_text_matches: list[str] = []
        self._last_text_codes: list[str] = []
        self._last_text_time = 0.0

        self._last_audio_level = 0.0
        self._last_audio_time = 0.0
        self._loud_audio_hits: list[float] = []
        self._audio_history: list[tuple[float, float]] = []
        self._recent_text_events: list[tuple[float, tuple[str, ...], str]] = []

        self._text_window_seconds = 4.0
        self._audio_window_seconds = 2.0
        self._repetition_window_seconds = 15.0

        self._category_rules = [
            CategoryRule(
                "A8",
                "특정 비명",
                42,
                ("으악", "으아악", "아악", "아아악", "꺄악", "꺄아악", "비명", "악소리"),
            ),
            CategoryRule(
                "A7",
                "피해자 구조 요청",
                62,
                (
                    "살려줘",
                    "살려주세요",
                    "도와줘",
                    "도와주세요",
                    "위험해",
                    "사람살려",
                    "신고해주세요",
                    "경찰불러",
                ),
            ),
            CategoryRule(
                "A7",
                "피해자 저지 발화",
                56,
                (
                    "하지마",
                    "하지마세요",
                    "이러지마",
                    "이러지마세요",
                    "만지지마",
                    "만지지마세요",
                    "다가오지마",
                    "다가오지마세요",
                    "놔줘",
                    "놔주세요",
                    "저리가",
                    "저리가세요",
                ),
            ),
            CategoryRule(
                "A1",
                "직접적 위해 위협",
                64,
                (
                    "죽여버리겠다",
                    "죽여버릴거야",
                    "죽여버린다",
                    "칼로찌른다",
                    "칼들고간다",
                    "패버린다",
                    "패죽인다",
                    "손봐주겠다",
                    "끝내버릴거야",
                    "그냥안둔다",
                    "가만안둔다",
                    "가만안둬",
                ),
            ),
            CategoryRule(
                "A2",
                "공격 직전 발화",
                38,
                (
                    "지금당장와",
                    "밖으로나와",
                    "나와봐",
                    "내려와",
                    "한판뜨자",
                    "도망가지마",
                    "오늘보자",
                    "나와서이야기하자",
                    "정리하자우리",
                ),
            ),
            CategoryRule(
                "A3",
                "협박·통제 발화",
                44,
                (
                    "말안들으면큰일난다",
                    "신고하면더심하게",
                    "내말대로안하면",
                    "가만안있다",
                    "망하게해주겠다",
                    "후회하게해주겠다",
                    "잘못건드렸다",
                    "인생끝나게해줄게",
                    "조심해라",
                    "어디한번해봐라",
                ),
            ),
            CategoryRule(
                "A4",
                "자해·극단선택 암시",
                48,
                (
                    "죽고싶다",
                    "살기싫다",
                    "다끝내고싶다",
                    "끝내버리고싶다",
                    "없어지는게낫겠다",
                    "이제방법이없다",
                    "다의미없다",
                    "오늘이마지막일지도",
                    "살고싶지않다",
                ),
            ),
            CategoryRule(
                "A5",
                "타해 의도 암시",
                50,
                (
                    "없애버리고싶다",
                    "혼내줘야겠다",
                    "오늘보이면안넘긴다",
                    "사람하나잡을것같다",
                    "피보게해주겠다",
                    "크게당해봐야",
                    "오늘끝이다",
                ),
            ),
            CategoryRule(
                "A6",
                "폭발 직전 감정 표현",
                32,
                (
                    "미쳐버릴것같다",
                    "참는것도한계다",
                    "다부수고싶다",
                    "누구하나다쳐야",
                    "폭발할것같다",
                    "진짜못참겠다",
                    "무슨짓할지모르겠다",
                ),
            ),
        ]

        self._low_risk_overstatements = (
            "죽겠다",
            "배고파죽겠다",
            "더워죽겠다",
            "힘들어죽겠다",
            "피곤해죽겠다",
            "웃겨죽겠다",
            "재밌어죽겠다",
            "미치겠네",
            "미치겠다",
            "돌겠다",
            "열받네",
        )
        self._target_tokens = (
            "너를",
            "너한테",
            "너는",
            "니가",
            "네가",
            "너네",
            "니네",
            "저사람",
            "그사람",
            "저새끼",
            "그새끼",
            "걔",
            "쟤",
            "사장",
            "손님",
            "여자친구",
            "남자친구",
            "누구누구",
        )
        self._immediacy_tokens = (
            "지금",
            "당장",
            "오늘",
            "곧",
            "바로",
            "즉시",
            "내려와",
            "나와",
            "찾아간다",
            "찾아가겠다",
        )
        self._means_tokens = (
            "칼",
            "망치",
            "흉기",
            "찌른다",
            "찌를",
            "패버린다",
            "패준다",
            "때린다",
            "죽여",
            "불질",
            "부순다",
            "깨버린다",
        )
        self._conditional_tokens = (
            "하면",
            "안하면",
            "말안들으면",
            "신고하면",
            "그러면",
        )
        self._plan_specific_tokens = (
            "오늘이마지막",
            "어떻게",
            "언제",
            "끝내고싶다",
            "방법이없다",
            "없어지는게낫겠다",
        )
        self._threat_codes = {"A1", "A2", "A3", "A5"}

    def update(self, speech_result, tracked_people, face_count: int) -> RiskAssessment:
        now = monotonic()
        transcript = (speech_result.transcript or "").strip()

        if transcript:
            normalized = self._normalize(transcript)
            text_analysis = self._analyze_transcript(normalized, now)
            if text_analysis["score"] > 0:
                self._last_text_score = float(text_analysis["score"])
                self._last_text_categories = list(text_analysis["categories"])
                self._last_text_reasons = list(text_analysis["reasons"])
                self._last_text_flags = list(text_analysis["flags"])
                self._last_text_matches = list(text_analysis["matches"])
                self._last_text_codes = list(text_analysis["codes"])
                self._last_text_time = now
                self._recent_text_events.append(
                    (now, tuple(text_analysis["codes"]), normalized[:60])
                )

        if speech_result.audio_level > 0:
            self._last_audio_level = speech_result.audio_level
            self._last_audio_time = now
            self._audio_history.append((now, speech_result.audio_level))
        if speech_result.audio_level >= 0.16:
            self._loud_audio_hits.append(now)

        self._audio_history = [
            (ts, level) for ts, level in self._audio_history if now - ts <= 12.0
        ]
        self._loud_audio_hits = [ts for ts in self._loud_audio_hits if now - ts <= 4.0]
        self._recent_text_events = [
            event for event in self._recent_text_events if now - event[0] <= self._repetition_window_seconds
        ]

        score = 0.0
        reasons: list[str] = []
        matched_keywords: list[str] = []
        categories: list[str] = []
        context_flags: list[str] = []

        text_age = now - self._last_text_time
        active_text_codes: list[str] = []
        if self._last_text_score > 0 and text_age <= self._text_window_seconds:
            decay = 1.0 - (text_age / self._text_window_seconds) * 0.35
            score += self._last_text_score * max(decay, 0.65)
            reasons.extend(self._last_text_reasons)
            categories.extend(self._last_text_categories)
            context_flags.extend(self._last_text_flags)
            matched_keywords.extend(self._last_text_matches)
            active_text_codes = list(self._last_text_codes)

        audio_age = now - self._last_audio_time
        recent_audio_level = self._last_audio_level if audio_age <= self._audio_window_seconds else 0.0
        audio_categories, audio_reasons, audio_score, audio_flags = self._score_audio_patterns(
            now,
            recent_audio_level,
            active_text_codes,
        )
        if audio_score > 0:
            score += audio_score
            reasons.extend(audio_reasons)
            categories.extend(audio_categories)
            context_flags.extend(audio_flags)

        video_categories, video_score = self._score_video(tracked_people)
        if video_score > 0:
            score += video_score
            reasons.extend(video_categories)
            categories.extend(video_categories)

        if len(tracked_people) > 0:
            score += 6
            context_flags.append(f"사람:{len(tracked_people)}")
        if face_count > 0:
            score += 4
            context_flags.append(f"얼굴:{face_count}")

        if active_text_codes and len(tracked_people) > 0:
            score += 8
            context_flags.append("발화+사람")
        if active_text_codes and recent_audio_level >= 0.10:
            score += 10
            context_flags.append("발화+고성")
        if recent_audio_level >= 0.18 and len(tracked_people) > 0:
            score += 8
            context_flags.append("큰소리+사람")

        reasons = list(dict.fromkeys(reasons))
        categories = list(dict.fromkeys(categories))
        context_flags = list(dict.fromkeys(context_flags))
        matched_keywords = list(dict.fromkeys(matched_keywords))

        score = min(int(round(score)), 100)
        return RiskAssessment(
            score=score,
            level=self._score_to_level(score),
            reasons=(categories + context_flags)[:6],
            matched_keywords=matched_keywords[:6],
            categories=categories[:4],
            context_flags=context_flags[:4],
        )

    def _analyze_transcript(self, normalized_text: str, now: float) -> dict:
        matched_rules: list[tuple[CategoryRule, str]] = []
        for rule in self._category_rules:
            for phrase in rule.phrases:
                normalized_phrase = self._normalize(phrase)
                if normalized_phrase and normalized_phrase in normalized_text:
                    matched_rules.append((rule, phrase))
                    break

        if not matched_rules:
            return {
                "score": 0,
                "categories": [],
                "reasons": [],
                "flags": [],
                "matches": [],
                "codes": [],
            }

        categories: list[str] = []
        codes: list[str] = []
        matches: list[str] = []
        base_scores: list[int] = []
        for rule, phrase in matched_rules:
            if rule.label not in categories:
                categories.append(rule.label)
            if rule.code not in codes:
                codes.append(rule.code)
            if phrase not in matches:
                matches.append(phrase)
            base_scores.append(rule.base_score)

        score = max(base_scores)
        for extra in sorted(base_scores, reverse=True)[1:]:
            score += int(round(extra * 0.35))

        flags: list[str] = []
        threat_like = any(code in self._threat_codes for code in codes)
        target_detected = self._contains_any(normalized_text, self._target_tokens)
        immediacy_detected = self._contains_any(normalized_text, self._immediacy_tokens)
        means_detected = self._contains_any(normalized_text, self._means_tokens)
        conditional_detected = self._contains_any(normalized_text, self._conditional_tokens)
        plan_specific = "A4" in codes and self._contains_any(normalized_text, self._plan_specific_tokens)

        if target_detected and threat_like:
            score += 12
            flags.append("대상 특정")
        if immediacy_detected and ("A4" in codes or threat_like):
            score += 15
            flags.append("즉시 실행 암시")
        if means_detected and ("A1" in codes or "A3" in codes or "A4" in codes or "A5" in codes):
            score += 15
            flags.append("수단 언급")
        if conditional_detected and ("A1" in codes or "A3" in codes):
            score += 10
            flags.append("조건부 위협")
        if "A2" in codes:
            score += 10
            flags.append("대면 충돌 유도")
        if plan_specific:
            score += 18
            flags.append("자해 계획 구체성")

        repetition_count = self._count_recent_repetition(now, codes)
        if repetition_count >= 1:
            score += min(18, 6 + repetition_count * 4)
            flags.append(f"반복 위협:{repetition_count + 1}회")

        if self._contains_any(normalized_text, self._low_risk_overstatements) and not threat_like and "A4" not in codes:
            score = max(score - 18, 0)
            flags.append("일상 과장 가능성")

        return {
            "score": min(score, 100),
            "categories": categories,
            "reasons": categories + flags,
            "flags": flags,
            "matches": matches,
            "codes": codes,
        }

    def _count_recent_repetition(self, now: float, codes: list[str]) -> int:
        code_set = set(codes)
        count = 0
        for event_time, event_codes, _text in self._recent_text_events:
            if now - event_time > self._repetition_window_seconds:
                continue
            if code_set.intersection(event_codes):
                count += 1
        return count

    def _score_audio_patterns(
        self,
        now: float,
        audio_level: float,
        active_text_codes: list[str],
    ) -> tuple[list[str], list[str], int, list[str]]:
        del now
        categories: list[str] = []
        flags: list[str] = []
        score = 0

        if "A8" in active_text_codes:
            categories.append("특정 비명")
            score += 18 if audio_level >= 0.10 else 12
        elif audio_level >= 0.24:
            categories.append("특정 비명")
            flags.append("특정 비명 의심")
            score += 18

        if len(self._loud_audio_hits) >= 3:
            categories.append("반복적 고성")
            score += 18
            flags.append(f"고성 반복:{len(self._loud_audio_hits)}회")

        if any(code in {"A1", "A3", "A5"} for code in active_text_codes):
            categories.append("위협적 음성 패턴")
            score += 16 if audio_level >= 0.08 else 10

        surge_ratio, baseline = self._audio_surge_ratio(audio_level)
        if baseline > 0:
            if surge_ratio >= 2.6 and audio_level >= 0.09:
                flags.append(f"고성 급상승:{surge_ratio:.1f}배")
                score += 12
            elif surge_ratio >= 1.9 and audio_level >= 0.07:
                flags.append(f"음량 급상승:{surge_ratio:.1f}배")
                score += 7

        if audio_level >= 0.10:
            flags.append(f"오디오 레벨:{audio_level:.2f}")

        categories = list(dict.fromkeys(categories))
        flags = list(dict.fromkeys(flags))
        return categories, categories + flags, score, flags

    def _audio_surge_ratio(self, current_level: float) -> tuple[float, float]:
        now = monotonic()
        baseline_levels = [
            level for ts, level in self._audio_history if 2.0 <= now - ts <= 10.0
        ]
        if not baseline_levels:
            return 0.0, 0.0
        baseline = max(sum(baseline_levels) / len(baseline_levels), 0.02)
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
        for index, first in enumerate(tracked_people):
            for second in tracked_people[index + 1 :]:
                if RiskAnalyzer._bbox_overlap_ratio(first["bbox"], second["bbox"]) >= 0.08:
                    return True
                if RiskAnalyzer._centroid_distance(first["bbox"], second["bbox"]) <= 90:
                    if first.get("movement", 0.0) >= 12 and second.get("movement", 0.0) >= 12:
                        return True
        return False

    @staticmethod
    def _detect_chase(tracked_people) -> bool:
        fast_people = [person for person in tracked_people if person.get("movement", 0.0) >= 18]
        if len(fast_people) < 2:
            return False
        for index, first in enumerate(fast_people):
            for second in fast_people[index + 1 :]:
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

    def _contains_any(self, normalized_text: str, tokens: tuple[str, ...]) -> bool:
        return any(self._normalize(token) in normalized_text for token in tokens)

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from time import monotonic


SUPPORTED_ACTION_LABELS = {"violence", "collapse", "loitering"}


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
    video_score: int = 0
    audio_score: int = 0
    fusion_score: int = 0
    raw_score: int = 0
    video_only_score: int = 0
    audio_video_gain: int = 0
    audio_confirmed_class: str = ""
    speech_match_quality: str = ""


def _load_rule_bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


class RiskAnalyzer:
    def __init__(self, rules_path: str | Path | None = None) -> None:
        self._last_text_score = 0.0
        self._last_text_categories: list[str] = []
        self._last_text_reasons: list[str] = []
        self._last_text_flags: list[str] = []
        self._last_text_matches: list[str] = []
        self._last_text_codes: list[str] = []
        self._last_text_time = 0.0
        self._last_match_quality = ""

        self._last_audio_level = 0.0
        self._last_audio_time = 0.0
        self._loud_audio_hits: list[float] = []
        self._audio_history: list[tuple[float, float]] = []
        self._recent_text_events: list[tuple[float, tuple[str, ...], str]] = []
        self._recent_risk_hits: list[tuple[float, float]] = []
        self._recent_action_events: list[tuple[float, str, float]] = []
        self._risk_hold_until = 0.0
        self._risk_hold_score = 0.0

        self._text_window_seconds = 4.0
        self._audio_window_seconds = 2.0
        self._repetition_window_seconds = 15.0

        self._category_rules = [
            CategoryRule(
                "A8",
                "특정 비명",
                42,
                (
                    "으악",
                    "으아악",
                    "아악",
                    "아아악",
                    "꺄악",
                    "꺄아악",
                    "비명",
                    "악소리악",
                    "헉",
                    "어어",
                    "으어",
                    "아야",
                    "으윽",
                    "깜짝이야",
                    "흐악",
                    "놔",
                    "야",
                    "어",
                    "히익",
                    "으으",
                    "어머",
                    "아이씨",
                    "끄악",
                    "허어억",
                    "으엑",
                    "헉컥",
                    "으허억",
                ),
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
                    "헬프미",
                    "긴급상황입니다",
                    "사람좀불러주세요",
                    "빨리와주세요",
                    "주변에사람있나요",
                    "지금너무위험해요",
                    "여기봐주세요",
                    "도움이필요해요",
                    "도와주실분있어요",
                    "저좀도와주세요",
                    "큰일났어요",
                    "신고부탁드립니다",
                    "신고가필요해요",
                    "부탁드립니다",
                    "제발 빨리요",
                    "좀 와주세요",
                    "도움이필요해요",
                    "지금당장와주세요"
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
                    "그만해",
                    "가",
                    "오지마",
                    "떨어져",
                    "가까이오지마",
                    "가세요",
                    "그만해",
                    "그만하세요",
                    "선넘지마",
                    "그만하라고",
                    "손치워",
                    "접촉하지마",
                    "여기서그만해",
                    "행동조심해",
                    "멈추라고",
                    "멈춰",
                    "더이상하지마",
                    "여기서끝내",
                    "그만하쇼",
                    "거리둬"
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
                    "그냥안넘어간다",
                    "죽고싶냐",
                    "뒤진다",
                    "뒤지고싶냐",
                    "지금가서조진다",
                    "너진짜맞는다",
                    "지금당장찾아간다",
                    "한번만더하면바로친다",
                    "오늘그냥안끝난다",
                    "조진다",
                    "조질거야",
                    "칼로썬다",
                    "너오늘진짜큰일난다",
                    "맞고싶냐",
                    "때릴거야",
                    "때린다"
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
                    "나와",
                    "쫄았냐",
                    "나와라",
                    "밖에나와있어",
                    "피하지말고나와",
                    "당장나와라",
                    "보자니까",
                    "바로와서얘기해",
                    "내려와라",
                    "시간끌지말고나와",
                    "당장보자",
                    "바로와",
                    "나와서끝내자",
                    "밖에서기다린다",
                    "바로나와라"
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
                    "신고하면알지",
                    "이거그냥안끝난다",
                    "지금이라도말잘해",
                    "선넘지마라",
                    "이거커진다",
                    "너한테안좋을걸",
                    "잘생각해라",
                    "지금멈추는게좋을거다",
                    "괜히일키우지마",
                    "이거후회한다",
                    "지금선택잘해",
                    "상황더나빠진다",
                    "지금그만해라",
                    "이거크게간다",
                    "좋게말할때그만해라",
                    "더가면답없다",
                    "더나가면손해다",
                    "생각잘해"
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
                    "너무힘들다",
                    "진짜지쳤다",
                    "다포기하고싶다",
                    "이제모르겠다",
                    "그냥다내려놓고싶다",
                    "사는게너무힘들다",
                    "버티기힘들다",
                    "아무것도하기싫다",
                    "그만두고싶다",
                    "이제진짜한계다",
                    "더는못하겠다",
                    "그냥쉬고싶다",
                    "다의미없는것같다",
                    "왜사는지모르겠다",
                    "너무힘들어서못버티겠다",
                    "이제진짜끝인것같다",
                    "계속가기힘들다",
                    "그만하고싶다",
                    "너무지친다",
                    "다놓고싶다",
                    "이상태로는못버틴다"
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
                    "이건그냥못넘긴다",
                    "가만히있진않는다",
                    "그냥넘어갈생각없다",
                    "이대로끝낼생각없다",
                    "한번짚고가야겠다",
                    "그냥둘수는없다",
                    "이번엔그냥안넘어간다",
                    "확실하게해야겠다",
                    "한번은봐야겠다",
                    "그냥넘어가면안되겠다",
                    "이건정리하고가자",
                    "가서얘기좀해야겠다",
                    "직접가서말한다",
                    "이번엔제대로한다",
                    "그냥넘기기엔아니다",
                    "이건선넘은거다",
                    "그냥두면계속이럴거다",
                    "이번에확실히짚는다",
                    "한번은터뜨려야겠다",
                    "그냥참고넘길일아니다",
                    "이건그냥넘어가면안된다",
                    "이번에끝까지간다",
                    "그냥두면안될것같다"
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
                    "진짜개빡친다",
                    "지금열받아서미칠거같다",
                    "더이상참기힘들다",
                    "지금터질거같다",
                    "진짜한계다",
                    "지금감정컨트롤안된다",
                    "너무짜증나서못참겠다",
                    "지금상태안좋다",
                    "진짜폭발직전이다",
                    "열받아서돌아버리겠다",
                    "지금진짜위험하다",
                    "계속이러면터진다",
                    "진짜이성나갈거같다",
                    "지금화가너무난다",
                    "참는것도여기까지다",
                    "지금감정올라왔다",
                    "더건드리면터진다",
                    "진짜미쳐버릴거같다",
                    "지금분노조절안된다",
                    "이거더이상못참는다",
                    "진짜화폭발직전이다",
                    "지금완전빡돌았다",
                    "더이상못버티겠다"
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

        configured_rules_path = (
            Path(rules_path).expanduser()
            if rules_path
            else Path(
                os.environ.get(
                    "DETECTWARNING_RISK_RULES_PATH",
                    str(Path(__file__).resolve().parents[1] / "configs" / "risk_rules.json"),
                )
            ).expanduser()
        )
        loaded_bundle = _load_rule_bundle(configured_rules_path)
        if loaded_bundle:
            self._category_rules = [
                CategoryRule(
                    str(rule.get("code", "")).strip(),
                    str(rule.get("label", "")).strip(),
                    int(rule.get("base_score", 0) or 0),
                    tuple(str(phrase).strip() for phrase in rule.get("phrases", []) if str(phrase).strip()),
                )
                for rule in loaded_bundle.get("category_rules", [])
                if isinstance(rule, dict)
            ] or self._category_rules
            self._low_risk_overstatements = tuple(
                str(token).strip()
                for token in loaded_bundle.get("low_risk_overstatements", [])
                if str(token).strip()
            ) or self._low_risk_overstatements
            self._target_tokens = tuple(
                str(token).strip()
                for token in loaded_bundle.get("target_tokens", [])
                if str(token).strip()
            ) or self._target_tokens
            self._immediacy_tokens = tuple(
                str(token).strip()
                for token in loaded_bundle.get("immediacy_tokens", [])
                if str(token).strip()
            ) or self._immediacy_tokens
            self._means_tokens = tuple(
                str(token).strip()
                for token in loaded_bundle.get("means_tokens", [])
                if str(token).strip()
            ) or self._means_tokens
            self._conditional_tokens = tuple(
                str(token).strip()
                for token in loaded_bundle.get("conditional_tokens", [])
                if str(token).strip()
            ) or self._conditional_tokens
            self._plan_specific_tokens = tuple(
                str(token).strip()
                for token in loaded_bundle.get("plan_specific_tokens", [])
                if str(token).strip()
            ) or self._plan_specific_tokens
            self._threat_codes = {
                str(code).strip()
                for code in loaded_bundle.get("threat_codes", [])
                if str(code).strip()
            } or self._threat_codes

        self._append_realtime_speech_boost_rules()
        self._normalized_category_rules = [
            (
                rule,
                tuple(
                    normalized
                    for normalized in (self._normalize(phrase) for phrase in rule.phrases)
                    if normalized
                ),
            )
            for rule in self._category_rules
        ]
        self._normalized_low_risk_overstatements = tuple(
            normalized
            for normalized in (self._normalize(token) for token in self._low_risk_overstatements)
            if normalized
        )
        self._normalized_target_tokens = tuple(
            normalized
            for normalized in (self._normalize(token) for token in self._target_tokens)
            if normalized
        )
        self._normalized_immediacy_tokens = tuple(
            normalized
            for normalized in (self._normalize(token) for token in self._immediacy_tokens)
            if normalized
        )
        self._normalized_means_tokens = tuple(
            normalized
            for normalized in (self._normalize(token) for token in self._means_tokens)
            if normalized
        )
        self._normalized_conditional_tokens = tuple(
            normalized
            for normalized in (self._normalize(token) for token in self._conditional_tokens)
            if normalized
        )
        self._normalized_plan_specific_tokens = tuple(
            normalized
            for normalized in (self._normalize(token) for token in self._plan_specific_tokens)
            if normalized
        )

    def _append_realtime_speech_boost_rules(self) -> None:
        extra_rules = [
            CategoryRule(
                "A7",
                "구조 요청",
                68,
                (
                    "살려주세요",
                    "살려 주세요",
                    "살려주새요",
                    "살려주세여",
                    "사려주세요",
                    "살려줘요",
                    "살려줘",
                    "사람 살려",
                    "도와주세요",
                    "도와 주세요",
                    "도와주새요",
                    "도와주세여",
                    "도와줘요",
                    "도와줘",
                    "제발 도와주세요",
                    "제발 살려주세요",
                    "경찰 불러주세요",
                    "경찰 불러줘",
                    "신고해주세요",
                    "신고해 주세요",
                    "일일이 신고",
                    "일일이 불러",
                    "백십이 신고",
                    "백십구 신고",
                    "구급차 불러주세요",
                ),
            ),
            CategoryRule(
                "A8",
                "비명/공포 반응",
                46,
                (
                    "하지마세요",
                    "하지 마세요",
                    "하지마",
                    "하지 마",
                    "하지마새요",
                    "그만하세요",
                    "그만해",
                    "그만 헤",
                    "멈춰주세요",
                    "멈춰",
                    "오지마세요",
                    "오지 마세요",
                    "오지마",
                    "다가오지마",
                    "다가오지 마",
                    "가까이 오지마",
                    "손대지마",
                    "손대지 마",
                    "만지지마",
                    "만지지 마",
                    "놔주세요",
                    "놔줘요",
                    "놔줘",
                    "놓아줘",
                    "무서워요",
                    "무서워",
                    "위험해요",
                    "위험해",
                    "때리지마",
                    "때리지 마",
                    "밀지마",
                    "차지마",
                ),
            ),
            CategoryRule(
                "A1",
                "직접 위해 협박",
                66,
                (
                    "죽여버릴거야",
                    "죽여 버릴거야",
                    "죽여버린다",
                    "죽인다",
                    "죽일거야",
                    "죽일 거야",
                    "가만 안둔다",
                    "가만 안 둔다",
                    "가만 안둘거야",
                    "칼로 찌른다",
                    "찌른다",
                    "찔러버린다",
                    "패버린다",
                    "때려버린다",
                    "죽고싶냐",
                    "맞고싶냐",
                ),
            ),
            CategoryRule(
                "A9",
                "납치/끌려감 의심",
                60,
                (
                    "끌고가지마",
                    "끌고 가지마",
                    "끌고 가지 마",
                    "끌려가요",
                    "끌려가",
                    "잡아가지마",
                    "잡아 가지 마",
                    "차에 태우지마",
                    "차에 태우지 마",
                    "어디 데려가",
                    "어디 가는거야",
                    "팔 놔",
                    "손 놔",
                    "따라가기 싫어",
                ),
            ),
        ]
        extra_rules.extend(self._build_large_speech_rule_pack())
        existing = {(rule.code, rule.label, rule.phrases) for rule in self._category_rules}
        for rule in extra_rules:
            key = (rule.code, rule.label, rule.phrases)
            if key not in existing:
                self._category_rules.append(rule)

    @staticmethod
    def _unique_phrases(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        phrases: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for phrase in group:
                normalized = str(phrase).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                phrases.append(normalized)
        return tuple(phrases)

    @staticmethod
    def _combine_phrases(stems: tuple[str, ...], suffixes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(f"{stem}{suffix}" for stem in stems for suffix in suffixes)

    def _build_large_speech_rule_pack(self) -> list[CategoryRule]:
        polite = ("", "요", "주세요", "해줘", "해줘요", "해 주세요", "해주세여", "해주새요")
        stop_stems = (
            "하지 마",
            "그만",
            "멈춰",
            "오지 마",
            "다가오지 마",
            "가까이 오지 마",
            "손대지 마",
            "만지지 마",
            "건드리지 마",
            "때리지 마",
            "밀지 마",
            "차지 마",
            "잡지 마",
            "끌지 마",
            "따라오지 마",
            "쫓아오지 마",
            "문 열지 마",
            "들어오지 마",
        )
        rescue = self._unique_phrases(
            (
                "살려",
                "살려줘",
                "살려줘요",
                "살려주세요",
                "살려 주십시오",
                "사려주세요",
                "살려주세여",
                "살려주새요",
                "사람 살려",
                "누가 좀 살려주세요",
                "제발 살려주세요",
                "제발 살려줘",
                "도와줘",
                "도와줘요",
                "도와주세요",
                "도와 주십시오",
                "도와주세여",
                "도와주새요",
                "누가 좀 도와주세요",
                "제발 도와주세요",
                "여기 좀 봐주세요",
                "여기 좀 와주세요",
                "빨리 와주세요",
                "빨리 도와주세요",
                "도움이 필요해요",
                "위급해요",
                "위험해요",
                "위험합니다",
                "큰일 났어요",
                "응급 상황이에요",
                "구해주세요",
                "구해줘요",
                "구해줘",
                "구조해주세요",
                "구조 요청합니다",
                "긴급 상황이에요",
                "혼자 있어요",
                "무서워요",
                "무서워",
                "사람 불러주세요",
                "관리자 불러주세요",
                "경비 불러주세요",
                "경찰 불러",
                "경찰 불러줘",
                "경찰 불러주세요",
                "경찰에 신고해",
                "경찰에 신고해주세요",
                "신고해",
                "신고해줘",
                "신고해주세요",
                "112 신고",
                "일일이 신고",
                "백십이 신고",
                "119 신고",
                "백십구 신고",
                "구급차 불러",
                "구급차 불러주세요",
                "119 불러",
                "응급차 불러주세요",
                "please help",
                "help me",
                "call police",
                "call the police",
                "call emergency",
                "call 911",
                "i am in danger",
            )
        )
        fear_stop = self._unique_phrases(
            self._combine_phrases(stop_stems, polite),
            (
                "하지마",
                "하지마요",
                "하지마세요",
                "하지마새요",
                "하지마세여",
                "그만해",
                "그만하세요",
                "그만 헤",
                "멈춰요",
                "멈춰주세요",
                "오지마",
                "오지마요",
                "오지마세요",
                "다가오지마",
                "다가오지마세요",
                "가까이 오지마",
                "손대지마",
                "손대지마세요",
                "만지지마",
                "만지지마세요",
                "놔줘",
                "놔줘요",
                "놔주세요",
                "놓아줘",
                "놓아주세요",
                "손 놔",
                "팔 놔",
                "발 놔",
                "잡은 거 놔",
                "아파",
                "아파요",
                "너무 아파요",
                "비켜",
                "비켜요",
                "떨어져",
                "떨어져요",
                "무섭다고",
                "싫어요",
                "싫다고",
                "싫다니까",
                "안 돼",
                "안돼요",
                "이러지 마",
                "이러지마세요",
                "제발 그만",
                "살려줘 제발",
                "으악",
                "아악",
                "악",
                "꺄악",
                "소리 지르지 마",
                "울지 마",
            )
        )
        direct_threat = self._unique_phrases(
            (
                "죽여버릴거야",
                "죽여 버릴거야",
                "죽여버릴 거야",
                "죽여버린다",
                "죽여 줄게",
                "죽인다",
                "죽일거야",
                "죽일 거야",
                "죽여줄게",
                "너 죽었어",
                "오늘 죽었어",
                "끝장내버린다",
                "끝장낼거야",
                "가만 안 둔다",
                "가만 안둔다",
                "가만 안둘거야",
                "가만 안 둘 거야",
                "가만 안 둬",
                "패버린다",
                "패줄게",
                "때려버린다",
                "때릴거야",
                "때릴 거야",
                "맞고 싶냐",
                "죽고 싶냐",
                "한 대 맞을래",
                "박살내버린다",
                "부숴버린다",
                "칼로 찌른다",
                "칼로 찌를거야",
                "찌른다",
                "찔러버린다",
                "찔러 죽인다",
                "목 조른다",
                "목 졸라버린다",
                "불 질러버린다",
                "태워버린다",
                "너 오늘 끝이야",
                "오늘 끝내자",
                "밖으로 나와",
                "이리 와",
                "따라와",
                "도망가지 마",
                "피하지 마",
                "숨어도 소용없어",
                "찾아간다",
                "기다리고 있어",
                "칼 가져올게",
                "무기 가져올게",
                "죽여 버릴 거다",
                "넌 끝났어",
                "가만두지 않겠다",
                "복수할 거야",
                "해코지할 거야",
                "협박하는 거야",
                "후회하게 해줄게",
                "신고하면 죽는다",
                "소리 지르면 죽는다",
                "말하면 가만 안 둬",
                "입 다물어",
                "조용히 해",
                "핸드폰 내놔",
                "폰 내놔",
                "시키는 대로 해",
                "차에 타",
                "따라가기만 해",
                "말 안 들으면 죽인다",
                "말 안 들으면 때린다",
                "신고하면 찾아간다",
            )
        )
        kidnapping = self._unique_phrases(
            (
                "끌고 가지 마",
                "끌고가지마",
                "끌고 가지마요",
                "끌고 가지 마세요",
                "끌려가요",
                "끌려가고 있어요",
                "납치",
                "납치당했어요",
                "납치당하는 중이에요",
                "잡아가지 마",
                "잡아가지마",
                "강제로 데려가요",
                "강제로 끌고 가요",
                "차에 태우지 마",
                "차에 태우지마",
                "차에 타기 싫어",
                "차 타기 싫어요",
                "어디 데려가",
                "어디 데려가는 거야",
                "어디 가는 거야",
                "따라가기 싫어",
                "가고 싶지 않아",
                "집에 보내줘",
                "문 열어줘",
                "나가게 해줘",
                "감금됐어요",
                "갇혔어요",
                "문이 잠겼어요",
                "못 나가게 해요",
                "팔 잡지 마",
                "손목 놔",
                "어깨 잡지 마",
                "끌지 마",
                "잡아끌지 마",
                "풀어줘",
                "묶지 마",
                "입 막지 마",
                "차 문 열어",
                "여기서 내릴래",
            )
        )
        sexual_contact = self._unique_phrases(
            (
                "만지지 마",
                "만지지마",
                "몸 만지지 마",
                "손대지 마",
                "손대지마",
                "이상한 짓 하지 마",
                "이러지 마세요",
                "싫다고 했잖아",
                "싫어요",
                "치한",
                "변태",
                "성추행",
                "성폭행",
                "몰래 찍지 마",
                "사진 찍지 마",
                "촬영하지 마",
                "옷 잡지 마",
                "옷 만지지 마",
                "가슴 만지지 마",
                "허리 잡지 마",
                "껴안지 마",
                "키스하지 마",
                "따라오지 마",
                "스토킹하지 마",
                "계속 따라와요",
                "저 사람 따라와요",
                "집 앞에 있어요",
                "계속 연락해요",
                "무서워서 못 가겠어요",
                "떨어져 주세요",
                "가까이 오지 마세요",
            )
        )
        self_harm = self._unique_phrases(
            (
                "죽고 싶다",
                "죽고싶다",
                "죽고 싶어요",
                "살기 싫다",
                "살기싫다",
                "살고 싶지 않아",
                "끝내고 싶다",
                "그만 살고 싶다",
                "사라지고 싶다",
                "뛰어내릴 거야",
                "뛰어내리고 싶다",
                "옥상에 갈 거야",
                "약 먹을 거야",
                "약을 먹었다",
                "목 매달고 싶다",
                "자살할 거야",
                "자살하고 싶다",
                "더는 못 버티겠다",
                "이제 못 하겠다",
                "방법이 없다",
                "유서 썼어",
                "마지막이야",
                "나 없어질 거야",
                "죽으면 편하겠지",
                "세상에서 없어지고 싶다",
                "i want to die",
                "kill myself",
                "suicide",
            )
        )
        violence_context = self._unique_phrases(
            (
                "맞고 있어요",
                "때리고 있어요",
                "폭행당했어요",
                "폭행당하고 있어요",
                "싸움 났어요",
                "싸우고 있어요",
                "사람이 맞고 있어요",
                "누가 때려요",
                "머리 때리지 마",
                "발로 차지 마",
                "밀지 마세요",
                "목 조르지 마",
                "숨 못 쉬겠어",
                "숨을 못 쉬겠어요",
                "피나요",
                "다쳤어요",
                "넘어졌어요",
                "기절했어요",
                "쓰러졌어요",
                "의식이 없어요",
                "칼 들고 있어요",
                "흉기 들고 있어요",
                "망치 들고 있어요",
                "병 들고 있어요",
                "불 지르려고 해요",
                "불났어요",
                "불이야",
                "문 부수고 있어요",
                "쫓아와요",
                "쫓기고 있어요",
            )
        )
        low_risk_more = (
            "게임에서 죽었다",
            "게임하다 죽었다",
            "배고파 죽겠다",
            "더워 죽겠다",
            "추워 죽겠다",
            "피곤해 죽겠다",
            "웃겨 죽겠다",
            "귀여워 죽겠다",
            "숙제 때문에 죽겠다",
            "시험 망했다",
            "농담이야",
            "장난이야",
            "드라마에서 죽었다",
            "영화에서 죽었다",
        )
        self._low_risk_overstatements = self._unique_phrases(self._low_risk_overstatements, low_risk_more)
        self._target_tokens = self._unique_phrases(
            self._target_tokens,
            ("저 사람", "그 사람", "아저씨", "아줌마", "남자", "여자", "학생", "아이", "친구", "선배", "후배"),
        )
        self._immediacy_tokens = self._unique_phrases(
            self._immediacy_tokens,
            ("지금", "당장", "바로", "빨리", "여기", "오늘", "이제", "계속", "또", "방금"),
        )
        self._means_tokens = self._unique_phrases(
            self._means_tokens,
            ("칼", "흉기", "망치", "병", "가위", "돌", "불", "라이터", "차", "끈", "약", "옥상", "베란다"),
        )
        self._threat_codes = set(self._threat_codes) | {"A11"}
        weapon_context = self._unique_phrases(
            (
                "칼 들고 있어요",
                "칼을 들고 있어요",
                "흉기 들고 있어요",
                "흉기를 들고 있어요",
                "망치 들고 있어요",
                "병 들고 있어요",
                "가위 들고 있어요",
                "무기 들고 있어요",
                "칼 가져왔어요",
                "흉기 가져왔어요",
                "칼을 꺼냈어요",
                "흉기를 꺼냈어요",
                "칼로 위협해요",
                "흉기로 위협해요",
                "불 지르려고 해요",
                "라이터 들고 있어요",
                "기름 뿌렸어요",
                "방화하려고 해요",
                "불났어요",
                "불이야",
            )
        )
        domestic_dating = self._unique_phrases(
            (
                "남편이 때려요",
                "아내가 때려요",
                "아빠가 때려요",
                "엄마가 때려요",
                "가족이 때려요",
                "애인을 때려요",
                "남자친구가 때려요",
                "여자친구가 때려요",
                "전 남자친구가 찾아왔어요",
                "전 여자친구가 찾아왔어요",
                "계속 집 앞에 있어요",
                "문을 두드려요",
                "문을 부수려고 해요",
                "집에 못 들어가겠어요",
                "집에서 나가고 싶어요",
                "집에 가기 무서워요",
                "계속 협박해요",
                "계속 때려요",
                "매일 맞아요",
                "목을 졸랐어요",
                "물건을 던져요",
                "핸드폰을 뺏었어요",
                "감금했어요",
                "못 나가게 해요",
                "아이를 때려요",
                "아이를 위협해요",
                "애를 데려갔어요",
                "아이를 데려가려고 해요",
                "가정폭력 신고",
                "데이트폭력 신고",
                "교제폭력 신고",
                "접근금지 어겼어요",
                "보호명령 어겼어요",
                "스토킹 신고",
                "계속 따라다녀요",
                "계속 연락해요",
                "계속 기다리고 있어요",
                "몰래 지켜봐요",
                "집 앞에서 기다려요",
                "회사 앞에서 기다려요",
                "학교 앞에서 기다려요",
                "위치 추적해요",
                "몰래 촬영해요",
                "사진을 유포한다고 해요",
                "동영상을 유포한다고 해요",
            )
        )
        child_school = self._unique_phrases(
            (
                "아이를 때리지 마",
                "애를 때리지 마",
                "아이 울어요",
                "아이가 울어요",
                "아이를 데려가지 마",
                "아이가 위험해요",
                "아이가 다쳤어요",
                "아이를 방치했어요",
                "아동학대 신고",
                "선생님 도와주세요",
                "학교폭력 신고",
                "친구들이 때려요",
                "괴롭힘 당하고 있어요",
                "따돌림 당하고 있어요",
                "돈을 뺏어요",
                "협박당하고 있어요",
                "화장실에 갇혔어요",
                "교실에 갇혔어요",
                "집에 가기 무서워요",
                "학교 가기 무서워요",
                "아이를 흔들지 마",
                "아기를 흔들지 마",
                "아기를 때리지 마",
                "아기가 숨을 못 쉬어요",
            )
        )
        medical_emergency = self._unique_phrases(
            (
                "숨을 못 쉬겠어요",
                "숨 못 쉬겠어",
                "호흡이 안 돼요",
                "가슴이 아파요",
                "심장이 아파요",
                "쓰러졌어요",
                "사람이 쓰러졌어요",
                "기절했어요",
                "의식이 없어요",
                "피를 흘려요",
                "피가 많이 나요",
                "머리를 다쳤어요",
                "움직이지 않아요",
                "반응이 없어요",
                "발작해요",
                "경련해요",
                "119 불러주세요",
                "구급차 빨리 불러주세요",
                "응급실 가야 해요",
                "약을 많이 먹었어요",
                "독을 마셨어요",
                "연기가 나요",
                "가스 냄새가 나요",
                "가스가 새요",
                "문 열어 주세요",
            )
        )
        robbery_intrusion = self._unique_phrases(
            (
                "도둑이야",
                "강도야",
                "강도가 들어왔어요",
                "집에 누가 들어왔어요",
                "모르는 사람이 들어왔어요",
                "문을 따고 있어요",
                "창문으로 들어왔어요",
                "지갑을 뺏어갔어요",
                "가방을 뺏어갔어요",
                "핸드폰을 뺏어갔어요",
                "돈을 뺏어갔어요",
                "칼 들고 돈 달래요",
                "협박해서 돈을 가져갔어요",
                "차에 누가 탔어요",
                "집 안에 숨어 있어요",
                "누가 쫓아와요",
                "도망가고 있어요",
                "문 잠가",
                "문 잠가주세요",
            )
        )
        self._low_risk_overstatements = self._unique_phrases(
            self._low_risk_overstatements,
            (
                "게임에서 맞았다",
                "영화에서 맞았다",
                "드라마에서 맞았다",
                "축구에서 졌다",
                "피곤해서 죽겠다",
                "웃겨서 죽겠다",
                "맛있어 죽겠다",
                "좋아 죽겠다",
                "심심해 죽겠다",
                "졸려 죽겠다",
                "힘들어 죽겠다",
                "일 때문에 죽겠다",
                "과제 때문에 죽겠다",
                "시험 때문에 죽겠다",
                "회사 때문에 죽겠다",
                "장난친 거야",
                "농담한 거야",
                "연기하는 거야",
                "대사였어",
                "노래 가사야",
            ),
        )
        family_subjects = (
            "남편이",
            "아내가",
            "아빠가",
            "엄마가",
            "가족이",
            "오빠가",
            "형이",
            "누나가",
            "언니가",
            "애인이",
            "남자친구가",
            "여자친구가",
            "전 남자친구가",
            "전 여자친구가",
        )
        assault_actions = (
            " 때려요",
            " 때리고 있어요",
            " 계속 때려요",
            " 목을 졸라요",
            " 밀쳤어요",
            " 발로 차요",
            " 물건을 던져요",
            " 협박해요",
            " 죽인다고 해요",
            " 칼을 들었어요",
            " 못 나가게 해요",
            " 문을 막고 있어요",
            " 핸드폰을 뺏었어요",
            " 신고하지 말래요",
        )
        stalking_subjects = (
            "저 사람이",
            "그 사람이",
            "모르는 사람이",
            "전 애인이",
            "전 남자친구가",
            "전 여자친구가",
            "아는 사람이",
            "낯선 사람이",
        )
        stalking_actions = (
            " 계속 따라와요",
            " 집 앞에 있어요",
            " 회사 앞에 있어요",
            " 학교 앞에 있어요",
            " 계속 연락해요",
            " 문자를 계속 보내요",
            " 기다리고 있어요",
            " 몰래 보고 있어요",
            " 사진을 찍어요",
            " 위치를 추적해요",
            " 차까지 따라와요",
            " 엘리베이터까지 따라와요",
        )
        child_subjects = (
            "아이가",
            "애가",
            "학생이",
            "친구들이",
            "선배가",
            "동급생이",
            "어른이",
            "선생님이",
        )
        child_actions = (
            " 맞고 있어요",
            " 울고 있어요",
            " 괴롭힘 당해요",
            " 협박당해요",
            " 돈을 뺏겨요",
            " 화장실에 갇혔어요",
            " 교실에 갇혔어요",
            " 집에 가기 무서워해요",
            " 학교 가기 무서워해요",
            " 다쳤어요",
            " 숨을 못 쉬어요",
            " 데려가지 말라고 해요",
        )
        medical_subjects = (
            "사람이",
            "아이가",
            "친구가",
            "여자가",
            "남자가",
            "어르신이",
            "환자가",
            "누가",
        )
        medical_actions = (
            " 쓰러졌어요",
            " 기절했어요",
            " 숨을 못 쉬어요",
            " 피를 흘려요",
            " 반응이 없어요",
            " 의식이 없어요",
            " 발작해요",
            " 경련해요",
            " 머리를 다쳤어요",
            " 가슴이 아프대요",
            " 움직이지 않아요",
            " 많이 다쳤어요",
            " 약을 먹었어요",
            " 넘어져서 못 일어나요",
        )
        weapon_subjects = (
            "저 사람이",
            "그 사람이",
            "모르는 사람이",
            "남자가",
            "여자가",
            "강도가",
            "누가",
            "가해자가",
        )
        weapon_actions = (
            " 칼을 들고 있어요",
            " 흉기를 들고 있어요",
            " 망치를 들고 있어요",
            " 가위를 들고 있어요",
            " 병을 들고 있어요",
            " 라이터를 들고 있어요",
            " 불을 지르려고 해요",
            " 기름을 뿌렸어요",
            " 칼로 위협해요",
            " 흉기로 위협해요",
            " 칼을 꺼냈어요",
            " 무기를 꺼냈어요",
        )
        intrusion_subjects = (
            "도둑이",
            "강도가",
            "모르는 사람이",
            "낯선 사람이",
            "누가",
            "저 사람이",
            "그 사람이",
        )
        intrusion_actions = (
            " 들어왔어요",
            " 문을 따고 있어요",
            " 창문으로 들어와요",
            " 집에 숨어 있어요",
            " 돈을 달래요",
            " 지갑을 뺏어갔어요",
            " 가방을 뺏어갔어요",
            " 핸드폰을 뺏어갔어요",
            " 도망가고 있어요",
            " 따라와요",
            " 문을 부수고 있어요",
        )
        direct_victims = (
            "나를",
            "저를",
            "아이를",
            "친구를",
            "엄마를",
            "아빠를",
            "여자를",
            "남자를",
            "사람을",
        )
        direct_actions = (
            " 때리지 마",
            " 밀지 마",
            " 차지 마",
            " 잡지 마",
            " 끌고 가지 마",
            " 차에 태우지 마",
            " 만지지 마",
            " 협박하지 마",
            " 감금하지 마",
            " 목 조르지 마",
            " 칼로 위협하지 마",
            " 따라오지 마",
        )
        generated_domestic = self._combine_phrases(family_subjects, assault_actions)
        generated_stalking = self._combine_phrases(stalking_subjects, stalking_actions)
        generated_child = self._combine_phrases(child_subjects, child_actions)
        generated_medical = self._combine_phrases(medical_subjects, medical_actions)
        generated_weapon = self._combine_phrases(weapon_subjects, weapon_actions)
        generated_intrusion = self._combine_phrases(intrusion_subjects, intrusion_actions)
        generated_stop = self._combine_phrases(direct_victims, direct_actions)
        return [
            CategoryRule("A7", "구조 요청", 68, rescue),
            CategoryRule("A8", "비명/공포 반응", 46, fear_stop),
            CategoryRule("A1", "직접 위해 협박", 66, direct_threat),
            CategoryRule("A9", "납치/끌려감 의심", 60, kidnapping),
            CategoryRule("A10", "성범죄/접촉 위험", 62, sexual_contact),
            CategoryRule("A4", "자해/극단 선택 위험", 54, self_harm),
            CategoryRule("A11", "흉기/방화 위험", 64, weapon_context),
            CategoryRule("A12", "가정/교제폭력·스토킹 위험", 60, domestic_dating),
            CategoryRule("A13", "아동/학교폭력 위험", 58, child_school),
            CategoryRule("A14", "응급 의료 위험", 62, medical_emergency),
            CategoryRule("A15", "침입/강도 위험", 62, robbery_intrusion),
            CategoryRule("A12", "가정/교제폭력·스토킹 위험", 60, generated_domestic),
            CategoryRule("A12", "가정/교제폭력·스토킹 위험", 58, generated_stalking),
            CategoryRule("A13", "아동/학교폭력 위험", 58, generated_child),
            CategoryRule("A14", "응급 의료 위험", 62, generated_medical),
            CategoryRule("A11", "흉기/방화 위험", 64, generated_weapon),
            CategoryRule("A15", "침입/강도 위험", 62, generated_intrusion),
            CategoryRule("A8", "비명/공포 반응", 50, generated_stop),
            CategoryRule("A2", "폭행/위험 상황", 48, violence_context),
        ]

    def update(self, speech_result, tracked_people, face_count: int, action_result=None) -> RiskAssessment:
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
                self._last_match_quality = str(text_analysis.get("match_quality") or "")
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
        self._recent_action_events = [
            event for event in self._recent_action_events if now - event[0] <= 5.0
        ]

        score = 0.0
        audio_component_score = 0.0
        video_component_score = 0.0
        fusion_bonus_score = 0.0
        audio_confirmed_class = ""
        reasons: list[str] = []
        matched_keywords: list[str] = []
        categories: list[str] = []
        context_flags: list[str] = []

        text_age = now - self._last_text_time
        active_text_codes: list[str] = []
        if self._last_text_score > 0 and text_age <= self._text_window_seconds:
            decay = 1.0 - (text_age / self._text_window_seconds) * 0.35
            text_score = self._last_text_score * max(decay, 0.65)
            score += text_score
            audio_component_score += text_score
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
            audio_component_score += audio_score
            reasons.extend(audio_reasons)
            categories.extend(audio_categories)
            context_flags.extend(audio_flags)

        video_categories, video_score = self._score_video(tracked_people)
        if video_score > 0:
            score += video_score
            video_component_score += video_score
            reasons.extend(video_categories)
            categories.extend(video_categories)

        has_people = len(tracked_people) > 0
        if action_result is not None and bool(getattr(action_result, "available", False)):
            raw_action_label = str(getattr(action_result, "label", "") or "").lower()
            action_label = raw_action_label if raw_action_label in SUPPORTED_ACTION_LABELS else raw_action_label
            if raw_action_label and raw_action_label not in SUPPORTED_ACTION_LABELS | {"normal", "unknown", "abnormal"}:
                action_label = "abnormal"
                context_flags.append(f"AI:unsupported_action:{raw_action_label}")
            action_confidence = float(getattr(action_result, "confidence", 0.0) or 0.0)
            abnormal_score = float(getattr(action_result, "abnormal_score", 0.0) or 0.0)
            if action_label in SUPPORTED_ACTION_LABELS:
                action_score = 24.0 + min(action_confidence, 1.0) * 26.0 + min(abnormal_score, 1.0) * 18.0
                if not has_people and not active_text_codes:
                    if abnormal_score < 0.92 or action_confidence < 0.65:
                        action_score *= 0.25
                        context_flags.append(f"AI:tentative_no_person:{action_label}")
                    else:
                        action_score *= 0.55
                        context_flags.append(f"AI:high_no_person:{action_label}")
                elif not has_people:
                    action_score *= 0.70
                    context_flags.append(f"AI:no_person_audio_context:{action_label}")
                recent_same_actions = [
                    value
                    for ts, label, value in self._recent_action_events
                    if label == action_label and now - ts <= 5.0
                ]
                self._recent_action_events.append((now, action_label, action_score))
                if recent_same_actions:
                    action_score += min(10.0, max(recent_same_actions) * 0.15)
                    context_flags.append(f"AI:stable:{action_label}")
                if (
                    not has_people
                    and len(recent_same_actions) >= 2
                    and abnormal_score >= 0.90
                    and action_confidence >= 0.70
                ):
                    action_score = max(action_score, 45.0)
                    context_flags.append(f"AI:stable_no_person_warning:{action_label}")
                score += action_score
                video_component_score += action_score
                categories.append(f"action:{action_label}")
                reasons.append(f"action:{action_label}")
                context_flags.append(f"AI:{action_label}:{action_confidence:.2f}")

                audio_confirms_action = (
                    active_text_codes
                    or audio_component_score >= 20
                    or recent_audio_level >= 0.16
                )
                if audio_confirms_action and action_score >= 25:
                    class_audio_bonus = {
                        "violence": 18.0,
                        "collapse": 14.0,
                        "loitering": 10.0,
                    }.get(action_label, 10.0)
                    if not active_text_codes and audio_component_score < 20:
                        class_audio_bonus = min(class_audio_bonus, 8.0)
                    if action_label == "violence" and recent_audio_level >= 0.10:
                        class_audio_bonus += 4.0
                    score += class_audio_bonus
                    fusion_bonus_score += class_audio_bonus
                    audio_confirmed_class = action_label
                    categories.append(f"audio_confirmed:{action_label}")
                    reasons.append(f"audio_confirmed:{action_label}")
                    context_flags.append(f"audio+action:{action_label}")
            elif abnormal_score >= 0.5:
                action_score = min(abnormal_score, 1.0) * 18.0
                score += action_score
                video_component_score += action_score
                categories.append("action:abnormal")
                reasons.append("action:abnormal")
                context_flags.append(f"AI:abnormal:{abnormal_score:.2f}")

        if len(tracked_people) > 0:
            score += 6
            video_component_score += 6
            context_flags.append(f"사람:{len(tracked_people)}")
        if face_count > 0:
            score += 4
            video_component_score += 4
            context_flags.append(f"얼굴:{face_count}")

        if active_text_codes and len(tracked_people) > 0:
            score += 8
            fusion_bonus_score += 8
            context_flags.append("발화+사람")
        if active_text_codes and recent_audio_level >= 0.10:
            score += 10
            fusion_bonus_score += 10
            context_flags.append("발화+고성")
        if recent_audio_level >= 0.18 and len(tracked_people) > 0:
            score += 8
            fusion_bonus_score += 8
            context_flags.append("큰소리+사람")

        if audio_component_score >= 30 and video_component_score >= 25:
            score += 8
            fusion_bonus_score += 8
            context_flags.append("audio+video_confirmed")

        reasons = list(dict.fromkeys(reasons))
        categories = list(dict.fromkeys(categories))
        context_flags = list(dict.fromkeys(context_flags))
        matched_keywords = list(dict.fromkeys(matched_keywords))

        raw_score = min(int(round(score)), 100)
        score = self._apply_risk_hysteresis(now, raw_score)
        video_only_score = min(int(round(video_component_score)), 100)
        audio_video_gain = max(0, raw_score - video_only_score)
        return RiskAssessment(
            score=score,
            level=self._score_to_level(score),
            reasons=(categories + context_flags)[:6],
            matched_keywords=matched_keywords[:6],
            categories=categories[:4],
            context_flags=context_flags[:4],
            video_score=min(int(round(video_component_score)), 100),
            audio_score=min(int(round(audio_component_score)), 100),
            fusion_score=score,
            raw_score=raw_score,
            video_only_score=video_only_score,
            audio_video_gain=audio_video_gain,
            audio_confirmed_class=audio_confirmed_class,
            speech_match_quality=self._last_match_quality if active_text_codes else "",
        )

    def _apply_risk_hysteresis(self, now: float, raw_score: int) -> int:
        self._recent_risk_hits = [
            (ts, value) for ts, value in self._recent_risk_hits if now - ts <= 5.0
        ]
        if raw_score >= 35:
            self._recent_risk_hits.append((now, float(raw_score)))
        strong_recent_hits = [
            value for ts, value in self._recent_risk_hits if now - ts <= 5.0 and value >= 35
        ]
        if raw_score >= 70 or len(strong_recent_hits) >= 2:
            self._risk_hold_until = max(self._risk_hold_until, now + 3.0)
            self._risk_hold_score = max(float(raw_score), self._risk_hold_score * 0.88)
        elif now > self._risk_hold_until:
            self._risk_hold_score = 0.0
        if now <= self._risk_hold_until:
            return min(100, max(raw_score, int(round(self._risk_hold_score))))
        return raw_score

    def _analyze_transcript(self, normalized_text: str, now: float) -> dict:
        matched_rules: list[tuple[CategoryRule, str, float, str]] = []
        for rule, normalized_phrases in self._normalized_category_rules:
            for phrase, normalized_phrase in zip(rule.phrases, normalized_phrases):
                match_score, match_quality = self._match_phrase_quality(normalized_text, normalized_phrase)
                if match_score > 0:
                    matched_rules.append((rule, phrase, match_score, match_quality))
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
        match_qualities: list[str] = []
        for rule, phrase, match_score, match_quality in matched_rules:
            if rule.label not in categories:
                categories.append(rule.label)
            if rule.code not in codes:
                codes.append(rule.code)
            if phrase not in matches:
                matches.append(phrase)
            base_scores.append(int(round(rule.base_score * match_score)))
            match_qualities.append(match_quality)

        score = max(base_scores)
        for extra in sorted(base_scores, reverse=True)[1:]:
            score += int(round(extra * 0.35))

        flags: list[str] = []
        threat_like = any(code in self._threat_codes for code in codes)
        target_detected = self._contains_any(normalized_text, self._normalized_target_tokens)
        immediacy_detected = self._contains_any(normalized_text, self._normalized_immediacy_tokens)
        means_detected = self._contains_any(normalized_text, self._normalized_means_tokens)
        conditional_detected = self._contains_any(normalized_text, self._normalized_conditional_tokens)
        plan_specific = "A4" in codes and self._contains_any(normalized_text, self._normalized_plan_specific_tokens)

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

        if self._contains_any(normalized_text, self._normalized_low_risk_overstatements) and not threat_like and "A4" not in codes:
            score = max(score - 18, 0)
            flags.append("일상 과장 가능성")

        return {
            "score": min(score, 100),
            "categories": categories,
            "reasons": categories + flags,
            "flags": flags,
            "matches": matches,
            "codes": codes,
            "match_quality": "exact" if "exact" in match_qualities else "fuzzy" if match_qualities else "",
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
        if score >= 90:
            return "CRITICAL"
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

    @staticmethod
    def _similarity(first: str, second: str) -> float:
        if not first or not second:
            return 0.0
        return float(SequenceMatcher(None, first, second).ratio())

    def _matches_phrase(self, normalized_text: str, normalized_phrase: str) -> bool:
        return self._match_phrase_quality(normalized_text, normalized_phrase)[0] > 0.0

    def _match_phrase_quality(self, normalized_text: str, normalized_phrase: str) -> tuple[float, str]:
        if not normalized_text or not normalized_phrase:
            return 0.0, ""
        if normalized_phrase in normalized_text:
            return 1.0, "exact"
        phrase_len = len(normalized_phrase)
        if phrase_len < 3 or len(normalized_text) < max(3, phrase_len - 1):
            return 0.0, ""
        if phrase_len <= 4:
            threshold = 0.92
        elif phrase_len <= 7:
            threshold = 0.84
        else:
            threshold = 0.78
        min_len = max(3, int(round(phrase_len * 0.72)))
        max_len = min(len(normalized_text), int(round(phrase_len * 1.25)) + 1)
        for window_len in range(min_len, max_len + 1):
            for start in range(0, len(normalized_text) - window_len + 1):
                candidate = normalized_text[start : start + window_len]
                if not self._shares_substantial_fragment(candidate, normalized_phrase):
                    continue
                if self._similarity(candidate, normalized_phrase) >= threshold:
                    return (0.86 if phrase_len >= 5 else 0.72), "fuzzy"
        return 0.0, ""

    @staticmethod
    def _shares_substantial_fragment(candidate: str, phrase: str) -> bool:
        if len(phrase) <= 3:
            return False
        if len(phrase) <= 5:
            fragments = {phrase[index : index + 2] for index in range(len(phrase) - 1)}
            return any(fragment in candidate for fragment in fragments)
        fragments = {phrase[index : index + 2] for index in range(len(phrase) - 1)}
        shared = sum(1 for fragment in fragments if fragment in candidate)
        return shared >= 2

    def _contains_any(self, normalized_text: str, tokens: tuple[str, ...]) -> bool:
        return any(self._matches_phrase(normalized_text, token) for token in tokens if token)

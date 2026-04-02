from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class WarningEvent:
    event_id: str
    timestamp: str
    source: str
    score: int
    level: str
    categories: list[str]
    context_flags: list[str]
    matched_keywords: list[str]
    transcript: str
    people_count: int
    face_count: int
    audio_level: float


class WarningEventLogger:
    def __init__(
        self,
        log_path: Path,
        min_score: int = 60,
        cooldown_seconds: float = 5.0,
    ) -> None:
        self.log_path = log_path
        self.min_score = min_score
        self.cooldown_seconds = cooldown_seconds
        self._last_logged_at = 0.0
        self._last_signature = ""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def maybe_log(
        self,
        *,
        now_monotonic: float,
        source: str,
        assessment,
        speech_result,
        people_count: int,
        face_count: int,
    ) -> WarningEvent | None:
        if assessment.score < self.min_score:
            return None

        signature = self._build_signature(assessment, speech_result, people_count, face_count)
        if (
            signature == self._last_signature
            and now_monotonic - self._last_logged_at < self.cooldown_seconds
        ):
            return None

        event = WarningEvent(
            event_id=self._build_event_id(),
            timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
            source=source,
            score=assessment.score,
            level=assessment.level,
            categories=list(assessment.categories),
            context_flags=list(assessment.context_flags),
            matched_keywords=list(assessment.matched_keywords),
            transcript=(speech_result.transcript or "").strip(),
            people_count=people_count,
            face_count=face_count,
            audio_level=round(float(speech_result.audio_level), 4),
        )
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        self._last_logged_at = now_monotonic
        self._last_signature = signature
        return event

    @staticmethod
    def _build_signature(assessment, speech_result, people_count: int, face_count: int) -> str:
        return "|".join(
            [
                str(assessment.level),
                str(assessment.score // 10),
                ",".join(assessment.categories[:2]),
                ",".join(assessment.context_flags[:2]),
                ",".join(assessment.matched_keywords[:2]),
                (speech_result.transcript or "").strip()[:30],
                str(people_count),
                str(face_count),
            ]
        )

    @staticmethod
    def _build_event_id() -> str:
        return "evt_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")

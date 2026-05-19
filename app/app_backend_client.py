from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests


RISK_CLASS_MAP = {
    "쓰러짐/급격한 자세 변화": "fall",
    "collapse": "fall",
    "fall": "fall",
    "폭력 의심": "violence",
    "violence": "violence",
    "침입/배회": "intrusion",
    "intrusion": "intrusion",
}


@dataclass
class AppBackendEventClient:
    base_url: str
    cctv_code: str
    token: str
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.cctv_code = self.cctv_code.strip()
        self.token = self.token.strip()
        self.session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.cctv_code and self.token)

    def close(self) -> None:
        self.session.close()

    def send_warning_event(self, event, assessment) -> bool:
        if not self.enabled:
            return False

        payload = {
            "cctvCode": self.cctv_code,
            "riskLevel": self._risk_level(assessment.level),
            "riskClass": self._risk_class(assessment),
            "riskScore": int(assessment.score),
            "videoReason": self._video_reason(assessment),
            "audioRiskSignalDetected": bool(getattr(event, "audio_level", 0.0) > 0 or assessment.audio_score > 0),
            "occurredAt": self._occurred_at(event.timestamp),
        }
        response = self.session.post(
            f"{self.base_url}/inference/events",
            json=payload,
            headers={"X-Inference-Token": self.token},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return True

    @staticmethod
    def _risk_level(level: str) -> str:
        normalized = str(level or "").upper()
        if normalized in {"HIGH", "CRITICAL"}:
            return "danger"
        return "warning"

    @staticmethod
    def _risk_class(assessment) -> str:
        values: list[str] = []
        values.extend(str(item) for item in getattr(assessment, "context_flags", []) or [])
        values.extend(str(item) for item in getattr(assessment, "categories", []) or [])
        values.extend(str(item) for item in getattr(assessment, "reasons", []) or [])
        for value in values:
            for key, mapped in RISK_CLASS_MAP.items():
                if key in value:
                    return mapped
        if getattr(assessment, "audio_score", 0) > getattr(assessment, "video_score", 0):
            return "audio_signal"
        return "fall"

    @staticmethod
    def _video_reason(assessment) -> str:
        reasons = list(getattr(assessment, "reasons", []) or [])
        flags = list(getattr(assessment, "context_flags", []) or [])
        text = ", ".join(str(item) for item in [*reasons, *flags] if item)
        return text[:160] if text else "위험 점수 기준 초과"

    @staticmethod
    def _occurred_at(timestamp: str) -> str:
        try:
            parsed = datetime.fromisoformat(timestamp)
            return parsed.isoformat()
        except Exception:
            return datetime.now().astimezone().isoformat(timespec="seconds")

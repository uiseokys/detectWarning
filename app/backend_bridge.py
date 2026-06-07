from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass
class BackendCircuitBreaker:
    failure_threshold: int = 3
    cooldown_seconds: float = 5.0
    failure_count: int = 0
    opened_until: float = 0.0
    last_error: str = ""

    def allow(self, priority: int, now: float | None = None) -> bool:
        if int(priority) <= 0:
            return True
        current = monotonic() if now is None else float(now)
        return current >= self.opened_until

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_until = 0.0
        self.last_error = ""

    def record_failure(self, error: str = "", now: float | None = None) -> None:
        current = monotonic() if now is None else float(now)
        self.failure_count += 1
        self.last_error = str(error or "")[:180]
        if self.failure_count >= self.failure_threshold:
            self.opened_until = current + self.cooldown_seconds

    def snapshot(self, now: float | None = None) -> dict:
        current = monotonic() if now is None else float(now)
        return {
            "open": current < self.opened_until,
            "failure_count": self.failure_count,
            "opened_for_seconds": round(max(0.0, self.opened_until - current), 2),
            "last_error": self.last_error,
        }

    def reset(self) -> None:
        self.failure_count = 0
        self.opened_until = 0.0
        self.last_error = ""

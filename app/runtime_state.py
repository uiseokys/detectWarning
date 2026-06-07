from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClientRuntimeSnapshot:
    client_id: str
    latest_meta: dict[str, Any]
    has_frame: bool
    last_seen_seconds: float
    last_frame_age_seconds: float | None
    analysis_fps: float
    camera_online: bool
    inference_online: bool
    status_reason: str


def safe_meta_snapshot(meta: object) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    return dict(meta)


def safe_list(value: object) -> list:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def snapshot_client_runtime(
    client_id: str,
    session: object,
    *,
    now: float,
    default_fps: float,
    ttl_seconds: float,
    fps_from_interval,
) -> ClientRuntimeSnapshot:
    meta = safe_meta_snapshot(getattr(session, "latest_meta", None))
    last_seen = safe_float(getattr(session, "last_seen", 0.0), 0.0)
    last_frame_at = safe_float(getattr(session, "last_frame_at", 0.0), 0.0)
    has_frame = bool(getattr(session, "latest_frame_jpeg", None) is not None)
    last_seen_seconds = round(max(0.0, now - last_seen), 1) if last_seen > 0 else 999999.0
    last_frame_age_seconds = round(max(0.0, now - last_frame_at), 1) if has_frame and last_frame_at > 0 else None
    inference_online = last_seen > 0 and now - last_seen <= ttl_seconds
    camera_online = has_frame and last_frame_at > 0 and now - last_frame_at <= ttl_seconds
    analysis_fps = fps_from_interval(
        safe_float(getattr(session, "smoothed_frame_interval_seconds", 0.0), 0.0),
        fallback=safe_float(meta.get("analysis_fps", default_fps), default_fps),
    )
    if camera_online and inference_online:
        status_reason = "live"
    elif inference_online:
        status_reason = "camera frame missing"
    else:
        status_reason = "inference client offline"
    return ClientRuntimeSnapshot(
        client_id=str(client_id),
        latest_meta=meta,
        has_frame=has_frame,
        last_seen_seconds=last_seen_seconds,
        last_frame_age_seconds=last_frame_age_seconds,
        analysis_fps=analysis_fps,
        camera_online=camera_online,
        inference_online=inference_online,
        status_reason=status_reason,
    )

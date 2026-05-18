from __future__ import annotations

from urllib.parse import quote

import requests


NOTIFICATION_DEFAULT_NTFY_SERVER = "https://ntfy.sh"
NOTIFICATION_ALLOWED_EVENTS = {"started", "completed", "completed_warning", "data_ready", "error", "aborted"}
NOTIFICATION_DEFAULT_EVENTS = ["started", "completed", "completed_warning", "error", "aborted"]


def normalize_notification_settings(payload: dict | None) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    notify_on_raw = payload.get("notify_on")
    if isinstance(notify_on_raw, str):
        notify_on_values = [part.strip() for part in notify_on_raw.split(",")]
    elif isinstance(notify_on_raw, list):
        notify_on_values = [str(part).strip() for part in notify_on_raw]
    else:
        notify_on_values = NOTIFICATION_DEFAULT_EVENTS
    notify_on = [value for value in notify_on_values if value in NOTIFICATION_ALLOWED_EVENTS] or list(
        NOTIFICATION_DEFAULT_EVENTS
    )
    if "started" not in notify_on:
        notify_on.insert(0, "started")
    return {
        "enabled": bool(payload.get("enabled")),
        "provider": str(payload.get("provider") or "ntfy").strip().lower(),
        "ntfy_server": str(payload.get("ntfy_server") or NOTIFICATION_DEFAULT_NTFY_SERVER).strip().rstrip("/"),
        "ntfy_topic": str(payload.get("ntfy_topic") or "").strip(),
        "notify_on": notify_on,
        "last_error": str(payload.get("last_error") or "").strip(),
        "last_sent_at": str(payload.get("last_sent_at") or "").strip(),
    }


def notification_settings_public(settings: dict | None) -> dict:
    normalized = normalize_notification_settings(settings)
    return {
        "enabled": normalized["enabled"],
        "provider": normalized["provider"],
        "ntfy_server": normalized["ntfy_server"],
        "ntfy_topic": normalized["ntfy_topic"],
        "notify_on": normalized["notify_on"],
        "last_error": normalized["last_error"],
        "last_sent_at": normalized["last_sent_at"],
    }


def ascii_notification_header(value: str, *, default: str = "detectWarning") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        ascii_text = text.encode("ascii", errors="ignore").decode("ascii").strip()
        return ascii_text or default


def send_ntfy_notification(settings: dict, *, title: str, message: str, priority: str = "default") -> None:
    topic = str(settings.get("ntfy_topic") or "").strip()
    if not topic:
        raise RuntimeError("ntfy topic is empty.")
    server = str(settings.get("ntfy_server") or NOTIFICATION_DEFAULT_NTFY_SERVER).strip().rstrip("/")
    headers = {
        "Title": ascii_notification_header(title),
        "Priority": ascii_notification_header(priority, default="default"),
        "Tags": "warning",
    }
    response = requests.post(
        f"{server}/{quote(topic, safe='')}",
        data=str(message or "").encode("utf-8"),
        headers=headers,
        timeout=6,
    )
    response.raise_for_status()


def job_source_filekey(job: dict) -> str:
    return str(job.get("source_filekey") or job.get("filekey") or "-").strip() or "-"


def job_step_label(job: dict) -> str:
    explicit = str(job.get("notification_step") or "").strip()
    if explicit:
        return explicit
    job_kind = str(job.get("job_kind") or "aihub").strip().lower()
    stage = str(job.get("stage") or "").strip().lower()
    filekey = str(job.get("filekey") or "").strip()
    if job_kind == "aihub" and stage == "extract":
        return "filekey_pose_preprocess"
    if job_kind == "guideline" or filekey == "guideline_clips":
        return "guideline_clips"
    if job_kind == "rgb" or filekey == "rgb_i3d_features":
        return "rgb_i3d_features"
    if job_kind == "cleanup" or filekey == "cleanup_raw_after_features":
        return "cleanup_raw_after_features"
    if job_kind == "train_guideline":
        return "guideline_train"
    if job_kind == "auto_tune":
        return "best_performance_auto_tune"
    return str(job.get("display_name") or filekey or "job").strip()


def build_job_notification_payload(job: dict, state: str, message: str) -> tuple[str, str, str]:
    step = job_step_label(job)
    filekey = job_source_filekey(job)
    datasetkey = str(job.get("source_datasetkey") or job.get("datasetkey") or "-")
    if state in {"started", "running"}:
        title = f"Started: {step}"
    elif state in {"completed", "data_ready"}:
        title = f"Done: {step}"
    elif state == "completed_warning":
        title = f"Warning: {step}"
    elif state == "aborted":
        title = f"Stopped: {step}"
    else:
        title = f"Failed: {step}"
    priority = "high" if state in {"error", "aborted"} else "default"
    body = f"filekey: {filekey}\ntask: {step}\nstatus: {state}\ndatasetkey: {datasetkey}\n{message}"
    return title, body, priority

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dashboard_runtime import build_job


def normalize_performance_plan_request(payload: dict, config: dict) -> dict:
    datasetkey = str(payload.get("datasetkey") or config.get("aihub_shell", {}).get("datasetkey") or "").strip()
    rgb_model = str(payload.get("rgb_model") or "i3d_r50").strip() or "i3d_r50"
    target_metric = str(payload.get("target_metric") or "accuracy").strip() or "accuracy"
    try:
        days = float(payload.get("days") or 12)
    except (TypeError, ValueError):
        days = 12.0
    days = min(max(days, 1.0), 14.0)
    try:
        max_trials = int(payload.get("trials") or config.get("auto_tune", {}).get("max_trials") or 24)
    except (TypeError, ValueError):
        max_trials = 24
    max_trials = min(max(max_trials, 1), 96)
    excluded_training_labels = []
    abduction_mode = str(payload.get("abduction_mode") or payload.get("training_label_mode") or "").strip().lower()
    if abduction_mode in {"exclude_abduction", "without_abduction", "no_abduction"}:
        excluded_training_labels.append("abduction")
    if str(payload.get("exclude_abduction") or "").strip().lower() in {"1", "true", "on", "yes"}:
        excluded_training_labels.append("abduction")
    deadline_at = datetime.now(timezone.utc).astimezone() + timedelta(days=days)
    return {
        "datasetkey": datasetkey,
        "rgb_model": rgb_model,
        "target_metric": target_metric,
        "days": days,
        "max_trials": max_trials,
        "excluded_training_labels": sorted(set(excluded_training_labels)),
        "deadline_at": deadline_at.isoformat(),
    }


def performance_plan_deadline_reached(plan: dict | None) -> bool:
    plan = plan if isinstance(plan, dict) else {}
    deadline = str(plan.get("deadline_at") or "").strip()
    if not deadline:
        return False
    try:
        deadline_dt = datetime.fromisoformat(deadline)
        if deadline_dt.tzinfo is None:
            deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc).astimezone() >= deadline_dt.astimezone()
    except ValueError:
        return False


def build_auto_tune_dashboard_job(plan: dict, *, reason: str) -> dict:
    trials = int(plan.get("max_trials") or 24)
    job = build_job("best_performance_auto_tune", datasetkey=plan.get("datasetkey") or "prepared", stage="auto_tune")
    job.update(
        {
            "job_kind": "auto_tune",
            "display_name": "최고 성능 auto-tune/ensemble",
            "notification_step": "best_performance_auto_tune",
            "source_filekey": "all_extracted",
            "source_datasetkey": plan.get("datasetkey") or "prepared",
            "trials": max(trials, 1),
            "target_metric": str(plan.get("target_metric") or "accuracy"),
            "excluded_training_labels": list(plan.get("excluded_training_labels") or []),
            "running_message": "누적 guideline/RGB+Pose feature로 대규모 auto-tune/ensemble/hybrid 탐색을 실행 중입니다.",
            "success_message": f"최고 성능 auto-tune/ensemble 학습이 완료되었습니다. reason={reason}",
        }
    )
    return job

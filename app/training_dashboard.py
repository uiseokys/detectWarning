from __future__ import annotations

import argparse
import atexit
from collections import Counter
import hashlib
from html import escape
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from action_training_pipeline import (
    collect_aihub_file_entries,
    cleanup_transient_job_data,
    extract_json_payload,
    fetch_aihub_file_tree,
    fetch_aihub_file_tree_via_shell,
    get_target_labels,
    load_config,
    parse_aihub_file_tree_listing,
    read_jsonl_entries,
    remove_transient_path_with_retries,
    resolve_aihub_shell_path,
    resolve_paths,
    write_jsonl_entries,
)
from dashboard_aihub import (
    AIHUB_AUTO_RECOMMEND_EXCLUDED_ZIP_GROUPS,
    AIHUB_AUTO_RECOMMEND_FALLBACK_ZIP_GROUPS,
    AIHUB_AUTO_RECOMMEND_MAX_PENDING,
    AIHUB_AUTO_RECOMMEND_ZIP_GROUPS,
    AIHUB_FILEKEY_LOOKUP_CACHE_SECONDS,
    AIHUB_PREPARED_JOB_STATES,
    AIHUB_TRAINED_JOB_STATES,
    aihub_entry_zip_group,
    build_aihub_lookup_result,
    build_auto_recommendation_policy,
    build_diagnosis_label_priorities,
    diagnosis_priority_for_label,
    find_next_trainable_aihub_entry,
    normalize_aihub_lookup_entries,
    normalize_auto_recommendation_policy,
    normalize_label_count_map,
    normalize_recommendation_label,
    normalize_recommendation_zip_groups,
    optional_aihub_api_key,
    recommendation_label,
    recommendation_zip_scope_state,
)
from dashboard_gpu import query_gpu_status
from dashboard_normal_ratio import apply_auto_normal_ratio_to_config
from dashboard_runtime import (
    build_job,
    build_retry_job_from,
    classify_job_exit,
    collect_result_summary,
    current_timestamp,
    flush_pages_pushes,
    persist_launcher_history,
    read_log_preview,
    read_log_tail,
    infer_dashboard_job_kind,
    snapshot_job,
    sync_pages_live,
    sync_pages_report,
    write_dashboard_status,
)
from dashboard_notifications import (
    NOTIFICATION_DEFAULT_NTFY_SERVER,
    build_job_notification_payload,
    normalize_notification_settings,
    notification_settings_public,
    send_ntfy_notification,
)
from dashboard_performance_plan import (
    build_auto_tune_dashboard_job,
    normalize_performance_plan_request,
    performance_plan_deadline_reached as plan_deadline_reached,
)
from dashboard_quality import build_guideline_quality_summary
from reporting import (
    STATE_SCHEMA_VERSION,
    analyze_class_balance,
    build_effective_pipeline_status,
    build_file_signature,
    build_path_diagnostic,
    build_restored_launcher_summary,
    enrich_completed_job,
    normalize_metric_payload,
    read_json,
    sort_jobs_by_recency,
    summarize_manifest,
    write_json_atomic,
)
from training_config import resolve_pages_sync_config
from training_insights import interpret_training_results
from training_dashboard_view import render_dashboard_live_fragments, render_dashboard_page

FILEKEY_RANGE_PATTERN = re.compile(r"^(\d+)(?:~|[-–—])(\d+)$")
MAX_FILEKEY_RANGE_SIZE = 1000
OVERVIEW_CACHE: dict[tuple, dict] = {}
OVERVIEW_CACHE_LOCK = threading.Lock()
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
NOTICE_COOKIE_NAME = "dw_dashboard_notice"
NOTICE_LEVEL_COOKIE_NAME = "dw_dashboard_notice_level"
NOTICE_COOKIE_MAX_AGE_SECONDS = 30
PREDOWNLOAD_COMPLETED_HISTORY_LIMIT = 128
PREDOWNLOAD_FAILED_HISTORY_LIMIT = 32
LIVE_URL_ENV_NAME = "DETECTWARNING_LIVE_URL"
LOCAL_DASHBOARD_HOSTS = {"127.0.0.1", "localhost", "::1"}
PUBLIC_VIEWER_HOST_SUFFIXES = (".trycloudflare.com", ".workers.dev")
OVERVIEW_DATASET_MANIFEST_SPECS = (
    ("raw", "raw_manifest"),
    ("train", "split_train"),
    ("val", "split_val"),
    ("test", "split_test"),
    ("prepared_train", "prepared_train"),
    ("prepared_val", "prepared_val"),
    ("prepared_test", "prepared_test"),
)
OVERVIEW_CURRENT_DATASET_MANIFEST_SPECS = (
    ("raw", "current_raw_manifest"),
    ("train", "current_split_train"),
    ("val", "current_split_val"),
    ("test", "current_split_test"),
    ("prepared_train", "current_prepared_train"),
    ("prepared_val", "current_prepared_val"),
    ("prepared_test", "current_prepared_test"),
)
def wants_json_response(request: Request | None) -> bool:
    if request is None:
        return False
    if request.headers.get("x-dashboard-async") == "1":
        return True
    accept = str(request.headers.get("accept") or "").lower()
    return "application/json" in accept


def normalize_hostname(value: str | None) -> str:
    return str(value or "").strip().strip("[]").lower()


def get_live_url_hostname() -> str:
    raw_value = str(os.environ.get(LIVE_URL_ENV_NAME) or "").strip()
    if not raw_value:
        return ""
    try:
        return normalize_hostname(urlparse(raw_value).hostname)
    except Exception:
        return ""


def is_viewer_request(request: Request | None) -> bool:
    if request is None:
        return False
    try:
        viewer_flag = str(request.query_params.get("viewer") or "").strip().lower()
        if viewer_flag in {"1", "true", "yes", "on"}:
            return True
    except Exception:
        pass

    host = normalize_hostname(getattr(request.url, "hostname", None))
    if not host or host in LOCAL_DASHBOARD_HOSTS:
        return False
    if any(host.endswith(suffix) for suffix in PUBLIC_VIEWER_HOST_SUFFIXES):
        return True

    live_host = get_live_url_hostname()
    return bool(live_host and host == live_host)


def ensure_dashboard_control_access(request: Request | None) -> None:
    if is_viewer_request(request):
        raise HTTPException(
            status_code=403,
            detail="공유 viewer에서는 작업 제어를 할 수 없습니다. 로컬 대시보드에서 실행해 주세요.",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="행동 학습 진행 상황 대시보드")
    parser.add_argument("--host", default="0.0.0.0", help="대시보드 바인드 주소")
    parser.add_argument("--port", type=int, default=8010, help="대시보드 포트")
    parser.add_argument(
        "--config",
        default="configs/action_training.example.json",
        help="학습 파이프라인과 같은 설정 파일 경로",
    )
    return parser.parse_args()


def resolve_allowed_origins(config: dict) -> list[str]:
    dashboard_config = config.get("dashboard", {})
    configured = dashboard_config.get("allowed_origins")
    if isinstance(configured, str):
        configured = [item.strip() for item in configured.split(",") if item.strip()]
    elif not isinstance(configured, list):
        configured = []

    env_value = os.environ.get("DETECTWARNING_ALLOWED_ORIGINS", "").strip()
    env_origins = [item.strip() for item in env_value.split(",") if item.strip()]
    allowed = list(dict.fromkeys([*configured, *env_origins]))
    if allowed:
        return allowed

    pages_sync = config.get("pages_sync", {})
    report_url = str(pages_sync.get("report_url") or "").strip()
    if report_url.startswith("http://") or report_url.startswith("https://"):
        match = re.match(r"^https?://[^/]+", report_url)
        if match:
            return [match.group(0)]
    if pages_sync.get("enabled"):
        return ["*"]
    return ["http://127.0.0.1:8010", "http://localhost:8010"]


def apply_manifest_aware_training_overrides(runtime_config: dict, paths: dict, job: dict) -> dict:
    stage = str(job.get("stage") or "").strip().lower()
    job_kind = str(job.get("job_kind") or "").strip().lower()
    if stage != "train" and job_kind != "auto_tune":
        return {}

    train_manifest = paths.get("active_prepared_train")
    if not isinstance(train_manifest, Path) or not train_manifest.exists():
        train_manifest = paths.get("guideline_prepared_train")
    if not isinstance(train_manifest, Path) or not train_manifest.exists():
        return {}

    labels = [str(label) for label in get_target_labels(runtime_config)]
    counts: Counter[str] = Counter()
    pose_counts: Counter[str] = Counter()
    rgb_counts: Counter[str] = Counter()
    for row in read_jsonl_entries(train_manifest):
        label = str(row.get("target_label") or "").strip()
        if label not in labels:
            continue
        counts[label] += 1
        pose_path = str(row.get("pose_path") or "").strip()
        if pose_path and Path(pose_path).exists():
            pose_counts[label] += 1
        rgb_path = str(row.get("rgb_feature_path") or "").strip()
        if rgb_path and Path(rgb_path).exists():
            rgb_counts[label] += 1

    trainable_counts = {label: int(pose_counts.get(label, 0) or counts.get(label, 0)) for label in labels}
    nonzero_counts = [count for count in trainable_counts.values() if count > 0]
    if len(nonzero_counts) <= 1:
        return {}

    manifest_multipliers: dict[str, float] = {}
    if trainable_counts.get("abduction", 0) > 0:
        manifest_multipliers["abduction"] = 1.75

    training_config = runtime_config.setdefault("training", {})
    merged_multipliers: dict[str, float] = dict(manifest_multipliers)
    training_config["class_weight_multipliers"] = dict(
        sorted(merged_multipliers.items(), key=lambda item: labels.index(item[0]) if item[0] in labels else len(labels))
    )
    training_config["balanced_sampler"] = True
    training_config["class_weight"] = "balanced"
    training_config["loss"] = "focal"
    training_config["focal_gamma"] = 1.0
    training_config["label_smoothing"] = 0.0
    training_config["temporal_pooling"] = "mean"
    training_config["learning_rate"] = 0.0003
    training_config["selection_metric"] = "accuracy"
    training_config["weight_decay"] = 0.003
    training_config["dropout"] = 0.35
    training_config["early_stopping_patience"] = max(
        int(training_config.get("early_stopping_patience") or 0),
        12,
    )
    training_config["overfit_guard_enabled"] = False
    training_config["overfit_guard_min_epoch"] = 16
    training_config["overfit_guard_loss_gap"] = 8.0
    training_config["overfit_guard_patience"] = 3
    training_config["max_duplicate_pose_label_samples"] = 0
    adaptive_config = training_config.setdefault("adaptive_class_weighting", {})
    if isinstance(adaptive_config, dict):
        adaptive_config["enabled"] = False
        adaptive_config["disabled_reason"] = (
            "manifest_aware_training_uses_current_distribution_only; "
            "previous metrics can be stale after label or manifest repairs"
        )
        adaptive_config["minority_count_multiplier"] = max(
            float(adaptive_config.get("minority_count_multiplier") or 1.25),
            1.5,
        )
        adaptive_config["max_multiplier"] = max(float(adaptive_config.get("max_multiplier") or 2.5), 2.5)

    auto_tune_config = runtime_config.setdefault("auto_tune", {})
    auto_tune_config["target_metric"] = "macro_f1_supported"
    auto_tune_config["ensemble_enabled"] = True
    auto_tune_config["hybrid_enabled"] = True
    auto_tune_config["hybrid_feature_weight_max"] = max(
        float(auto_tune_config.get("hybrid_feature_weight_max") or 0.45),
        0.95,
    )
    auto_tune_config["hybrid_feature_weight_step"] = min(
        float(auto_tune_config.get("hybrid_feature_weight_step") or 0.05),
        0.025,
    )
    auto_tune_config["hybrid_temperature_values"] = sorted(
        {
            *[float(value) for value in auto_tune_config.get("hybrid_temperature_values", []) or []],
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.25,
            1.5,
        }
    )
    auto_tune_config["hybrid_class_bias_multipliers"] = sorted(
        {
            *[float(value) for value in auto_tune_config.get("hybrid_class_bias_multipliers", []) or []],
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            1.0,
            1.05,
            1.1,
            1.2,
            1.3,
            1.4,
            1.6,
            1.8,
            2.0,
        }
    )
    auto_tune_config["ensemble_temperature_values"] = sorted(
        {
            *[float(value) for value in auto_tune_config.get("ensemble_temperature_values", []) or []],
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.25,
            1.5,
        }
    )
    auto_tune_config["ensemble_probability_power_values"] = sorted(
        {
            *[float(value) for value in auto_tune_config.get("ensemble_probability_power_values", []) or []],
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.25,
            1.5,
            2.0,
        }
    )
    auto_tune_config["ensemble_class_bias_multipliers"] = sorted(
        {
            *[float(value) for value in auto_tune_config.get("ensemble_class_bias_multipliers", []) or []],
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            1.0,
            1.05,
            1.1,
            1.2,
            1.3,
            1.4,
        }
    )
    auto_tune_config["promote_ensemble"] = True
    auto_tune_config["promote_hybrid"] = True

    total = sum(counts.values())
    pose_ready = sum(pose_counts.values())
    rgb_ready = sum(rgb_counts.values())
    optimization = {
        "enabled": True,
        "source_manifest": str(train_manifest),
        "counts": dict(counts),
        "pose_ready_counts": dict(pose_counts),
        "rgb_ready_counts": dict(rgb_counts),
        "pose_ready_ratio": round(pose_ready / max(total, 1), 6),
        "rgb_ready_ratio": round(rgb_ready / max(total, 1), 6),
        "class_weight_multipliers": training_config.get("class_weight_multipliers") or {},
        "pose_profile": {
            "name": "legacy_focal_balanced_pose_plus_rgb_i3d",
            "loss": training_config.get("loss"),
            "class_weight": training_config.get("class_weight"),
            "balanced_sampler": training_config.get("balanced_sampler"),
            "temporal_pooling": training_config.get("temporal_pooling"),
            "learning_rate": training_config.get("learning_rate"),
            "max_duplicate_pose_label_samples": training_config.get("max_duplicate_pose_label_samples"),
            "overfit_guard": {
                "enabled": training_config.get("overfit_guard_enabled"),
                "min_epoch": training_config.get("overfit_guard_min_epoch"),
                "loss_gap": training_config.get("overfit_guard_loss_gap"),
                "patience": training_config.get("overfit_guard_patience"),
            },
        },
        "auto_tune_target_metric": auto_tune_config.get("target_metric"),
    }
    runtime_config["_manifest_training_optimization"] = optimization
    return optimization


def normalize_excluded_training_labels(value) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = []
    return sorted({item for item in items if item})


def excluded_training_labels_from_payload(payload: dict) -> list[str]:
    mode = str(payload.get("abduction_mode") or payload.get("training_label_mode") or "").strip().lower()
    excluded = normalize_excluded_training_labels(payload.get("excluded_training_labels"))
    if mode in {"exclude_abduction", "without_abduction", "no_abduction"}:
        excluded.append("abduction")
    if payload.get("exclude_abduction") is True or str(payload.get("exclude_abduction") or "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }:
        excluded.append("abduction")
    return normalize_excluded_training_labels(excluded)


def apply_training_label_filter_to_runtime_config(runtime_config: dict, job: dict) -> dict:
    excluded_labels = normalize_excluded_training_labels(job.get("excluded_training_labels"))
    if not excluded_labels:
        return {}
    dataset_config = runtime_config.setdefault("dataset", {})
    labels = [str(label) for label in get_target_labels(runtime_config)]
    filtered_labels = [label for label in labels if label not in set(excluded_labels)]
    if len(filtered_labels) < 2 or filtered_labels == labels:
        return {}
    dataset_config["target_labels"] = filtered_labels

    training_config = runtime_config.setdefault("training", {})
    multipliers = training_config.get("class_weight_multipliers")
    if isinstance(multipliers, dict):
        training_config["class_weight_multipliers"] = {
            str(label): value
            for label, value in multipliers.items()
            if str(label) in filtered_labels
        }

    tasks_config = runtime_config.get("training_tasks")
    if isinstance(tasks_config, dict):
        pose_multipliers = tasks_config.get("pose_classification_class_weight_multipliers")
        if isinstance(pose_multipliers, dict):
            tasks_config["pose_classification_class_weight_multipliers"] = {
                str(label): value
                for label, value in pose_multipliers.items()
                if str(label) in filtered_labels
            }

    payload = {
        "enabled": True,
        "excluded_labels": excluded_labels,
        "labels_before": labels,
        "labels_after": filtered_labels,
    }
    runtime_config["_training_label_filter"] = payload
    return payload


def compute_overview_signature(paths: dict, launcher_status: dict | None, *, lite: bool = False) -> tuple:
    launcher_status = launcher_status or {}
    watched = [
        paths["pipeline_status"],
        paths["training_progress"],
        paths["continual_state"],
        paths["current_skip_report"],
        paths["cumulative_skip_report"],
        paths["artifacts_dir"] / "metrics.json",
        paths["artifacts_dir"] / "hybrid_summary.json",
        paths["artifacts_dir"] / "hybrid_metrics.json",
        paths["artifacts_dir"] / "ensemble_summary.json",
        paths["artifacts_dir"] / "specialized_tasks" / "summary.json",
        paths["artifacts_dir"] / "specialized_tasks" / "detection" / "metrics.json",
        paths["artifacts_dir"] / "specialized_tasks" / "classification" / "metrics.json",
        paths["artifacts_dir"] / "specialized_tasks" / "pose_classification" / "metrics.json",
        paths["artifacts_dir"] / "labels.json",
        paths["raw_manifest"],
        paths["split_train"],
        paths["split_val"],
        paths["split_test"],
        paths["prepared_train"],
        paths["prepared_val"],
        paths["prepared_test"],
        paths["guideline_prepared_train"],
        paths["guideline_prepared_val"],
        paths["guideline_prepared_test"],
        paths["active_prepared_train"],
        paths["active_prepared_val"],
        paths["active_prepared_test"],
        paths["current_raw_manifest"],
        paths["current_split_train"],
        paths["current_split_val"],
        paths["current_split_test"],
        paths["current_prepared_train"],
        paths["current_prepared_val"],
        paths["current_prepared_test"],
        paths["workspace_dir"] / "launcher_history.json",
    ]
    file_signature = [build_file_signature(path) for path in watched]

    current_job = launcher_status.get("current_job") or {}
    recent_completed_jobs = [
        job for job in launcher_status.get("completed_jobs", [])[:10]
        if isinstance(job, dict)
    ]
    latest_completed_job = next(
        (job for job in recent_completed_jobs if job.get("state") in {"completed", "completed_warning"}),
        None,
    )
    latest_error_job = next(
        (job for job in recent_completed_jobs if job.get("state") == "error"),
        None,
    )
    dynamic_log_paths = []
    predownload_status = launcher_status.get("predownload") or {}
    predownload_running = predownload_status.get("running") if isinstance(predownload_status, dict) else []
    predownload_completed = predownload_status.get("completed") if isinstance(predownload_status, dict) else []
    predownload_failed = predownload_status.get("failed") if isinstance(predownload_status, dict) else []
    if not lite:
        for candidate in (
            current_job.get("log_path") if isinstance(current_job, dict) else None,
            latest_completed_job.get("log_path") if isinstance(latest_completed_job, dict) else None,
            latest_error_job.get("log_path") if isinstance(latest_error_job, dict) else None,
            *(
                item.get("log_path")
                for item in (predownload_running or [])[:4]
                if isinstance(item, dict)
            ),
            *(
                item.get("log_path")
                for item in (predownload_failed or [])[:2]
                if isinstance(item, dict)
            ),
        ):
            if not candidate:
                continue
            path = Path(str(candidate))
            dynamic_log_paths.append(build_file_signature(path))
    launcher_signature = (
        launcher_status.get("state"),
        tuple(
            (job.get("datasetkey"), job.get("filekey"), job.get("state"))
            for job in launcher_status.get("pending_jobs", [])
            if isinstance(job, dict)
        ),
        tuple(
            (job.get("datasetkey"), job.get("filekey"), job.get("state"), job.get("finished_at"))
            for job in launcher_status.get("completed_jobs", [])[:10]
            if isinstance(job, dict)
        ),
        tuple(
            (key, current_job.get(key))
            for key in ("datasetkey", "filekey", "started_at", "state", "message", "log_path")
            if key in current_job
        ),
        tuple(
            (job.get("datasetkey"), job.get("filekey"), job.get("state"), job.get("started_at"), job.get("log_path"))
            for job in (predownload_running or [])[:8]
            if isinstance(job, dict)
        ),
        tuple(
            (job.get("datasetkey"), job.get("filekey"), job.get("state"), job.get("finished_at"), job.get("exit_code"))
            for job in (predownload_completed or [])[-8:]
            if isinstance(job, dict)
        ),
        tuple(
            (job.get("datasetkey"), job.get("filekey"), job.get("state"), job.get("finished_at"), job.get("exit_code"))
            for job in (predownload_failed or [])[:8]
            if isinstance(job, dict)
        ),
        predownload_status.get("pause_reason") if isinstance(predownload_status, dict) else "",
        tuple(dynamic_log_paths),
        launcher_status.get("auto_start_enabled"),
        launcher_status.get("auto_enqueue_enabled"),
        launcher_status.get("auto_enqueue_datasetkey"),
    )
    return (tuple(file_signature), launcher_signature, lite)


def derive_stage_ratio(pipeline_status: dict | None, training_progress: dict | None) -> float:
    pipeline_status = pipeline_status or {}
    training_progress = training_progress or {}
    explicit = pipeline_status.get("stage_progress")
    if isinstance(explicit, (int, float)):
        ratio = float(explicit)
    else:
        stage = str(pipeline_status.get("stage", "")).strip().lower()
        ratio = {
            "starting": 0.0,
            "queued": 0.0,
            "download": 0.2,
            "prepare": 0.6,
            "train": 0.85,
            "completed": 1.0,
            "error": 1.0,
        }.get(stage, 0.0)

    stage = str(pipeline_status.get("stage", "")).strip().lower()
    if stage == "train":
        epochs_total = int(training_progress.get("epochs_total") or 0)
        epochs_completed = int(training_progress.get("epochs_completed") or 0)
        epoch_ratio = (epochs_completed / epochs_total) if epochs_total > 0 else 0.0
        ratio = max(ratio, 0.8 + (0.2 * epoch_ratio))
    return max(0.0, min(1.0, ratio))


def build_current_job_progress(pipeline_status: dict | None, training_progress: dict | None, launcher_status: dict | None) -> dict:
    pipeline_status = pipeline_status or {}
    training_progress = training_progress or {}
    launcher_status = launcher_status or {}

    ratio = derive_stage_ratio(pipeline_status, training_progress)
    stage = str(pipeline_status.get("stage", "")).strip().lower() or str(launcher_status.get("state", "idle"))
    stage_label_map = {
        "idle": "대기",
        "queued": "대기",
        "download": "다운로드",
        "prepare": "전처리",
        "train": "학습",
        "completed": "완료",
        "error": "오류",
        "running": "실행 중",
        "starting": "시작 중",
    }
    label = stage_label_map.get(stage, stage or "대기")

    if stage == "prepare":
        processed = int(pipeline_status.get("processed_items") or 0)
        total = int(pipeline_status.get("total_items") or 0)
        detail = f"{pipeline_status.get('current_split', '-')} split | {processed}/{total}"
    elif stage == "train":
        epochs_completed = int(training_progress.get("epochs_completed") or 0)
        epochs_total = int(training_progress.get("epochs_total") or 0)
        detail = f"epoch {epochs_completed}/{epochs_total}"
    elif stage == "download":
        found = pipeline_status.get("discovered_items")
        detail = f"발견 샘플 {found}" if found not in (None, "") else str(pipeline_status.get("message") or "-")
    else:
        detail = str(pipeline_status.get("message") or launcher_status.get("message") or "-")

    return {
        "ratio": round(ratio, 4),
        "percent": int(round(ratio * 100)),
        "stage": stage,
        "label": label,
        "detail": detail,
        "current_video": pipeline_status.get("current_video"),
        "processed_items": pipeline_status.get("processed_items"),
        "total_items": pipeline_status.get("total_items"),
    }


def estimate_eta(current_job_progress: dict | None, launcher_status: dict | None) -> dict:
    current_job_progress = current_job_progress or {}
    launcher_status = launcher_status or {}
    current_job = launcher_status.get("current_job") if isinstance(launcher_status, dict) else None
    started_at = current_job.get("started_at") if isinstance(current_job, dict) else None
    ratio = float(current_job_progress.get("ratio") or 0.0)
    if not started_at or ratio <= 0.0 or ratio >= 1.0:
        return {"seconds_remaining": None, "label": "-"}

    try:
        started_dt = datetime.fromisoformat(str(started_at))
    except ValueError:
        return {"seconds_remaining": None, "label": "-"}

    now_dt = datetime.now(timezone.utc).astimezone()
    elapsed_seconds = max(0.0, (now_dt - started_dt).total_seconds())
    if elapsed_seconds <= 0.0:
        return {"seconds_remaining": None, "label": "-"}

    total_estimated = elapsed_seconds / ratio
    remaining_seconds = max(0, int(round(total_estimated - elapsed_seconds)))
    return {
        "seconds_remaining": remaining_seconds,
        "label": format_duration(remaining_seconds),
    }


def overview_has_live_activity(overview: dict | None) -> bool:
    overview = overview or {}
    launcher = overview.get("launcher") or {}
    pipeline = overview.get("pipeline_status") or {}
    queue_progress = overview.get("queue_progress") or {}
    current_job_progress = overview.get("current_job_progress") or {}

    active_states = {"starting", "queued", "running", "prepare", "download", "train"}
    launcher_state = str(launcher.get("state") or "").strip().lower()
    pipeline_state = str(pipeline.get("state") or "").strip().lower()
    if launcher_state in active_states or pipeline_state in active_states:
        return True

    current_job = launcher.get("current_job")
    if isinstance(current_job, dict) and any(current_job.get(key) for key in ("filekey", "datasetkey", "log_path")):
        return True

    predownload = launcher.get("predownload") if isinstance(launcher, dict) else {}
    if isinstance(predownload, dict) and predownload.get("running"):
        return True

    try:
        if int(queue_progress.get("active") or 0) > 0 or int(queue_progress.get("pending") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass

    try:
        ratio = float(current_job_progress.get("ratio") or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    return 0.0 < ratio < 1.0


def format_duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "-"
    total_seconds = int(max(0, round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}시간 {minutes}분"
    if minutes > 0:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def build_predownload_log_payload(predownload: dict | None, get_log_tail) -> dict:
    predownload = predownload if isinstance(predownload, dict) else {}
    running = predownload.get("running") if isinstance(predownload.get("running"), list) else []
    completed = predownload.get("completed") if isinstance(predownload.get("completed"), list) else []
    failed = predownload.get("failed") if isinstance(predownload.get("failed"), list) else []
    max_parallel = int(predownload.get("max_parallel") or 0)
    pause_reason = str(predownload.get("pause_reason") or "").strip()
    lines = [
        "[predownload][status] "
        f"running={len(running)}/{max_parallel or '-'} "
        f"completed_recent={len(completed)} failed_recent={len(failed)} "
        f"free_disk_gb={predownload.get('free_gb', '-')}"
    ]
    if pause_reason:
        lines.append(f"[predownload][paused] {pause_reason}")
    if running:
        lines.append("[predownload][running] 전처리/학습 중에도 남는 다운로드 슬롯에서 다음 filekey를 미리 받습니다.")
        for index, item in enumerate(running[:4], start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                "[predownload][slot] "
                f"{index}/{max_parallel or len(running)} "
                f"datasetkey={item.get('datasetkey') or '-'} "
                f"filekey={item.get('filekey') or '-'} "
                f"state={item.get('state') or '-'} "
                f"started_at={item.get('started_at') or '-'}"
            )
            log_tail = get_log_tail(item.get("log_path"))
            if log_tail:
                lines.append(f"----- predownload log: filekey={item.get('filekey') or '-'} -----")
                lines.append(log_tail)
    else:
        lines.append("[predownload][idle] 현재 병렬 다운로드 작업은 없습니다.")

    for item in completed[-3:]:
        if isinstance(item, dict):
            lines.append(
                "[predownload][completed] "
                f"filekey={item.get('filekey') or '-'} exit_code={item.get('exit_code', '-')} "
                f"finished_at={item.get('finished_at') or '-'}"
            )
    for item in failed[:3]:
        if isinstance(item, dict):
            lines.append(
                "[predownload][failed] "
                f"filekey={item.get('filekey') or '-'} exit_code={item.get('exit_code', '-')} "
                f"finished_at={item.get('finished_at') or '-'}"
            )

    return {
        "running_count": len(running),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "max_parallel": max_parallel,
        "pause_reason": pause_reason,
        "tail": "\n".join(lines),
    }


def build_manifest_summary_group(paths: dict, manifest_specs: tuple[tuple[str, str], ...]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for key, path_key in manifest_specs:
        path = paths.get(path_key)
        if isinstance(path, Path):
            summary[key] = summarize_manifest(path, label_field="target_label")
        else:
            summary[key] = {"total": 0, "by_label": {}}
    return summary


def workspace_has_saved_state(workspace_dir: Path | None) -> bool:
    if not isinstance(workspace_dir, Path):
        return False
    candidates: list[Path] = [
        workspace_dir / "launcher_history.json",
        workspace_dir / "artifacts" / "best_action_model.pt",
        workspace_dir / "artifacts" / "metrics.json",
        workspace_dir / "manifests" / "cumulative_raw_items.jsonl",
        workspace_dir / "manifests" / "cumulative_prepared_train.jsonl",
    ]
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return True
        except OSError:
            continue
    for directory in (workspace_dir / "job_logs", workspace_dir / "runtime_configs"):
        try:
            if directory.exists() and any(directory.iterdir()):
                return True
        except OSError:
            continue
    return False


def find_workspace_state_candidates(project_root: Path, active_workspace_dir: Path) -> list[Path]:
    candidates = [
        project_root / "training_data" / "action_pipeline_aihub",
        project_root / "training_data" / "action_pipeline",
    ]
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen or resolved == active_workspace_dir:
            continue
        seen.add(key)
        unique_candidates.append(resolved)
    return unique_candidates


def _job_timestamp_from_id(job_id: str | None) -> str | None:
    if not job_id:
        return None
    match = re.match(r"^job_(\d{8}_\d{6}_\d{6})_", str(job_id))
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S_%f")
    except ValueError:
        return None
    local_tz = datetime.now(timezone.utc).astimezone().tzinfo
    return parsed.replace(tzinfo=local_tz).astimezone().isoformat()


def infer_job_state_from_log(log_path: Path) -> str:
    tail = read_log_tail(log_path, max_lines=120, max_chars=12000)
    lowered = tail.lower()
    success_markers = ("[train] best model:", "[train] metrics:", "[train] labels:")
    if any(marker in tail for marker in success_markers):
        return "completed"
    if "keyboardinterrupt" in lowered or "강제 중단" in tail:
        return "aborted"
    if "traceback" in lowered or "[pipeline][error][error]" in lowered or "runtimeerror:" in lowered:
        return "error"
    if "[pipeline][data_ready][data_ready]" in lowered or "[train] deferred:" in lowered:
        return "data_ready"
    if "[pipeline][completed][completed]" in lowered or "정상 완료" in tail:
        return "completed"
    if "[pipeline][" in lowered or "[launcher]" in lowered:
        return "running"
    return "unknown"


def restore_completed_jobs_from_logs(job_logs_dir: Path, runtime_config_dir: Path, limit: int = 30) -> list[dict]:
    if not job_logs_dir.exists():
        return []

    restored_jobs: list[dict] = []
    log_paths = sorted(
        (path for path in job_logs_dir.glob("job_*.log") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for log_path in log_paths[:limit]:
        inferred_state = infer_job_state_from_log(log_path)
        if inferred_state not in {"completed", "completed_warning", "data_ready", "error", "aborted"}:
            continue
        job_id = log_path.stem
        runtime_config_path = runtime_config_dir / f"{job_id}.json"
        runtime_payload = read_json(runtime_config_path) if runtime_config_path.exists() else {}
        shell_config = runtime_payload.get("aihub_shell", {}) if isinstance(runtime_payload, dict) else {}
        inferred_started_at = _job_timestamp_from_id(job_id)
        try:
            finished_at = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat()
        except OSError:
            finished_at = None
        filekey = (
            shell_config.get("filekey")
            or job_id.split("_", 4)[-1]
            or "-"
        )
        restored_jobs.append(
            {
                "job_id": job_id,
                "filekey": str(filekey),
                "datasetkey": shell_config.get("datasetkey"),
                "retry_of": None,
                "retry_count": 0,
                "queued_at": inferred_started_at,
                "started_at": inferred_started_at,
                "finished_at": finished_at,
                "state": inferred_state,
                "exit_code": None,
                "runtime_config_path": str(runtime_config_path) if runtime_config_path.exists() else None,
                "log_path": str(log_path),
                "result_summary": None,
            }
        )
    return restored_jobs


def merge_completed_jobs(existing_jobs: list[dict], restored_jobs: list[dict], *, limit: int = 30) -> list[dict]:
    merged: dict[str, dict] = {}
    for source in (restored_jobs, existing_jobs):
        for job in source:
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                job_id = "|".join(
                    str(job.get(key) or "").strip()
                    for key in ("datasetkey", "filekey", "stage", "started_at", "log_path")
                )
            current = merged.get(job_id)
            if not isinstance(current, dict):
                merged[job_id] = dict(job)
                continue
            enriched = dict(current)
            for key, value in job.items():
                if enriched.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                    enriched[key] = value
            merged[job_id] = enriched
    return sort_jobs_by_recency(list(merged.values()))[:limit]


def create_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    pages_sync = resolve_pages_sync_config(config, config_path.parent)
    project_root = Path(__file__).resolve().parent.parent
    pipeline_script = Path(__file__).resolve().with_name("action_training_pipeline.py")
    job_logs_dir = paths["workspace_dir"] / "job_logs"
    job_logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_dir = paths["workspace_dir"] / "runtime_configs"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)
    launcher_history_path = paths["workspace_dir"] / "launcher_history.json"
    notification_settings_path = paths["workspace_dir"] / "dashboard_notifications.json"

    app = FastAPI(title="detectWarning Training Dashboard")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolve_allowed_origins(config),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    state_lock = threading.RLock()
    aihub_filekey_lookup_cache: dict[str, dict] = {}
    aihub_filekey_lookup_cache_lock = threading.Lock()

    launcher_state: dict[str, object] = {
        "process": None,
        "started_at": None,
        "runtime_config_path": None,
        "current_job": None,
        "queued_jobs": [],
        "completed_jobs": [],
        "auto_start_enabled": True,
        "auto_enqueue_enabled": False,
        "auto_extract_enabled": False,
        "auto_enqueue_datasetkey": None,
        "auto_enqueue_api_key": "",
        "auto_rgb_model": "i3d_r50",
        "predownload_enabled": True,
        "predownload_max_parallel": 2,
        "predownload_min_free_gb": 100,
        "predownload_pause_reason": "",
        "predownload_processes": {},
        "predownload_completed": [],
        "predownload_failed": [],
        "performance_plan": {},
        "last_state": "idle",
        "last_exit_code": None,
        "last_message": "아직 실행 기록이 없습니다.",
        "log_path": None,
        "pages_sync_warning": None,
        "filekey_pipeline_status": {},
        "notification_settings": normalize_notification_settings(
            read_json(notification_settings_path)
            or (config.get("dashboard", {}).get("notifications") if isinstance(config.get("dashboard"), dict) else {})
            or {}
        ),
    }

    def save_notification_settings(settings: dict) -> dict:
        normalized = normalize_notification_settings(settings)
        launcher_state["notification_settings"] = normalized
        notification_settings_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(notification_settings_path, normalized)
        return normalized

    def dispatch_dashboard_notification(event: str, *, title: str, message: str, priority: str = "default") -> None:
        settings = normalize_notification_settings(launcher_state.get("notification_settings"))
        if not settings.get("enabled") or event not in set(settings.get("notify_on") or []):
            return
        if settings.get("provider") != "ntfy":
            settings["last_error"] = f"지원하지 않는 알림 provider입니다: {settings.get('provider')}"
            save_notification_settings(settings)
            return

        def worker() -> None:
            try:
                send_ntfy_notification(settings, title=title, message=message, priority=priority)
                settings["last_sent_at"] = current_timestamp()
                settings["last_error"] = ""
            except Exception as exc:
                settings["last_error"] = str(exc)
            try:
                save_notification_settings(settings)
            except Exception as exc:
                print(f"[notify] settings save failed: {exc}", flush=True)

        threading.Thread(target=worker, daemon=True).start()

    FILEKEY_PIPELINE_STEP_ORDER = (
        "pose_extract",
        "guideline_clips",
        "rgb_i3d_features",
        "cleanup_raw_after_job",
    )
    FILEKEY_PIPELINE_STEP_LABELS = {
        "pose_extract": "pose 추출",
        "guideline_clips": "guideline_clips",
        "rgb_i3d_features": "RGB/I3D feature",
        "cleanup_raw_after_job": "cleanup_raw_after_job",
    }
    FILEKEY_PIPELINE_SUCCESS_STATES = {"completed", "completed_warning", "data_ready"}

    def filekey_pipeline_step_id(job: dict | None) -> str:
        if not isinstance(job, dict):
            return ""
        job_kind = infer_dashboard_job_kind(job)
        stage = str(job.get("stage") or "").strip().lower()
        notification_step = str(job.get("notification_step") or "").strip().lower()
        filekey = str(job.get("filekey") or "").strip().lower()
        if job_kind == "aihub" and stage == "extract":
            return "pose_extract"
        if job_kind == "guideline" or notification_step == "guideline_clips" or filekey == "guideline_clips":
            return "guideline_clips"
        if job_kind == "rgb" or notification_step == "rgb_i3d_features" or filekey == "rgb_i3d_features":
            return "rgb_i3d_features"
        if job_kind == "cleanup" or notification_step in {"cleanup_raw_after_features", "cleanup_raw_after_job"}:
            return "cleanup_raw_after_job"
        return ""

    def filekey_pipeline_identity(job: dict | None) -> tuple[str, str, str]:
        if not isinstance(job, dict):
            return "", "", ""
        step_id = filekey_pipeline_step_id(job)
        if not step_id:
            return "", "", ""
        filekey = str(job.get("source_filekey") or "").strip()
        if not filekey and infer_dashboard_job_kind(job) == "aihub":
            filekey = str(job.get("filekey") or "").strip()
        datasetkey = str(job.get("source_datasetkey") or job.get("datasetkey") or "").strip()
        if not filekey:
            return "", "", ""
        return f"{datasetkey}:{filekey}", datasetkey, filekey

    def filekey_pipeline_record_for_job(job: dict, *, reset_on_start: bool = False) -> dict | None:
        identity, datasetkey, filekey = filekey_pipeline_identity(job)
        if not identity:
            return None
        records = launcher_state.setdefault("filekey_pipeline_status", {})
        if not isinstance(records, dict):
            launcher_state["filekey_pipeline_status"] = {}
            records = launcher_state["filekey_pipeline_status"]
        if reset_on_start or identity not in records:
            records[identity] = {
                "datasetkey": datasetkey,
                "filekey": filekey,
                "steps": {},
                "errors": [],
                "started_at": current_timestamp(),
                "final_sent": False,
            }
        return records.get(identity) if isinstance(records.get(identity), dict) else None

    def mark_filekey_pipeline_started(job: dict) -> None:
        step_id = filekey_pipeline_step_id(job)
        if not step_id:
            return
        identity, _, _ = filekey_pipeline_identity(job)
        if not identity:
            return
        records = launcher_state.get("filekey_pipeline_status", {})
        existing = records.get(identity) if isinstance(records, dict) else None
        reset_on_start = step_id == "pose_extract" or not isinstance(existing, dict) or bool(existing.get("final_sent"))
        record = filekey_pipeline_record_for_job(job, reset_on_start=reset_on_start)
        if not isinstance(record, dict):
            return
        steps = record.setdefault("steps", {})
        if isinstance(steps, dict):
            steps[step_id] = {
                "state": "running",
                "job_id": job.get("job_id"),
                "started_at": job.get("started_at") or current_timestamp(),
            }

    def dispatch_filekey_step_started_notification(job: dict, message: str) -> None:
        step_id = filekey_pipeline_step_id(job)
        if not step_id:
            return
        mark_filekey_pipeline_started(job)
        _, datasetkey, filekey = filekey_pipeline_identity(job)
        if not filekey:
            return
        step_label = FILEKEY_PIPELINE_STEP_LABELS.get(step_id, step_id)
        title = f"Started: {step_label}"
        body = (
            f"filekey: {filekey}\n"
            f"task: {step_label}\n"
            f"status: started\n"
            f"datasetkey: {datasetkey or '-'}\n"
            f"{message}"
        )
        dispatch_dashboard_notification("started", title=title, message=body, priority="default")

    def record_filekey_pipeline_finished(job: dict, state: str, message: str) -> tuple[bool, str, str, str]:
        step_id = filekey_pipeline_step_id(job)
        if not step_id:
            return False, "", "", ""
        record = filekey_pipeline_record_for_job(job)
        if not isinstance(record, dict):
            return False, "", "", ""
        steps = record.setdefault("steps", {})
        if isinstance(steps, dict):
            steps[step_id] = {
                "state": state,
                "job_id": job.get("job_id"),
                "finished_at": job.get("finished_at") or current_timestamp(),
                "message": message,
            }
        filekey = str(record.get("filekey") or "-")
        datasetkey = str(record.get("datasetkey") or "-")
        if state not in FILEKEY_PIPELINE_SUCCESS_STATES:
            errors = record.setdefault("errors", [])
            if isinstance(errors, list):
                error_text = f"{FILEKEY_PIPELINE_STEP_LABELS.get(step_id, step_id)}: {message}"
                if error_text not in errors:
                    errors.append(error_text)
            if bool(record.get("final_sent")):
                return True, "", "", ""
            record["final_sent"] = True
            error_items = record.get("errors") if isinstance(record.get("errors"), list) else []
            title = f"Filekey error: {filekey}"
            body = (
                f"filekey: {filekey}\n"
                "status: error\n"
                f"datasetkey: {datasetkey}\n"
                "파일키 전체 작업이 오류로 중단되었습니다.\n"
                + "\n".join(f"- {item}" for item in error_items[:6])
            )
            return True, "error", title, body
        if step_id != "cleanup_raw_after_job" or bool(record.get("final_sent")):
            return True, "", "", ""
        record["final_sent"] = True
        errors = record.get("errors") if isinstance(record.get("errors"), list) else []
        if errors:
            title = f"Filekey error: {filekey}"
            body = (
                f"filekey: {filekey}\n"
                f"status: error\n"
                f"datasetkey: {datasetkey}\n"
                "파일키 전체 작업이 끝났지만 중간 오류가 있었습니다.\n"
                + "\n".join(f"- {item}" for item in errors[:6])
            )
            return True, "error", title, body
        completed_steps = []
        record_steps = record.get("steps") if isinstance(record.get("steps"), dict) else {}
        for step in FILEKEY_PIPELINE_STEP_ORDER:
            if step in record_steps:
                completed_steps.append(FILEKEY_PIPELINE_STEP_LABELS.get(step, step))
        title = f"Filekey done: {filekey}"
        body = (
            f"filekey: {filekey}\n"
            f"status: completed\n"
            f"datasetkey: {datasetkey}\n"
            "파일키 전체 작업이 정상 완료되었습니다.\n"
            f"completed_steps: {', '.join(completed_steps) if completed_steps else '-'}"
        )
        return True, "completed", title, body

    def performance_plan_payload() -> dict:
        plan = launcher_state.get("performance_plan")
        return dict(plan) if isinstance(plan, dict) else {}

    def performance_plan_is_active() -> bool:
        return bool(performance_plan_payload().get("enabled"))

    def performance_plan_deadline_reached() -> bool:
        return plan_deadline_reached(performance_plan_payload())

    def enqueue_performance_plan_training_locked(reason: str) -> dict:
        plan = performance_plan_payload()
        if not plan.get("enabled"):
            return {"ok": False, "message": "성능 플랜이 켜져 있지 않습니다.", "job": None}
        if plan.get("final_training_queued"):
            return {"ok": True, "message": "최종 auto-tune 학습이 이미 큐에 있습니다.", "job": None}
        has_guideline_features = paths["guideline_prepared_train"].exists() and paths["guideline_prepared_val"].exists()
        if not has_guideline_features:
            launcher_state["auto_extract_enabled"] = False
            plan["enabled"] = False
            plan["finished_at"] = current_timestamp()
            plan["status"] = "stopped_no_features"
            launcher_state["performance_plan"] = plan
            message = "성능 플랜을 끝냈지만 guideline feature가 없어 최종 학습을 시작하지 못했습니다."
            launcher_state["last_message"] = message
            return {"ok": False, "message": message, "job": None}
        job = build_auto_tune_dashboard_job(plan, reason=reason)
        pending_jobs = launcher_state.setdefault("queued_jobs", [])
        if isinstance(pending_jobs, list):
            pending_jobs.append(job)
        launcher_state["auto_extract_enabled"] = False
        launcher_state["auto_enqueue_enabled"] = False
        plan["enabled"] = False
        plan["final_training_queued"] = True
        plan["final_training_reason"] = reason
        plan["finished_extract_at"] = current_timestamp()
        plan["status"] = "final_training_queued"
        launcher_state["performance_plan"] = plan
        launcher_state["last_state"] = "queued"
        launcher_state["last_message"] = f"성능 플랜 추출을 마치고 최종 auto-tune/ensemble 학습을 큐에 추가했습니다. reason={reason}"
        return {"ok": True, "message": launcher_state["last_message"], "job": job}

    def apply_restored_launcher_summary(completed_jobs_override: list[dict] | None = None) -> None:
        if isinstance(launcher_state.get("process"), subprocess.Popen):
            return
        completed_jobs = (
            completed_jobs_override
            if isinstance(completed_jobs_override, list)
            else launcher_state.get("completed_jobs", [])
        )
        summary = build_restored_launcher_summary(
            completed_jobs if isinstance(completed_jobs, list) else [],
            pipeline_status=read_json(paths["pipeline_status"]) or {},
            training_progress=read_json(paths["training_progress"]) or {},
        )
        launcher_state["last_state"] = summary.get("state") or launcher_state.get("last_state") or "idle"
        launcher_state["last_message"] = summary.get("message") or launcher_state.get("last_message") or "아직 실행 기록이 없습니다."
        if summary.get("log_path"):
            launcher_state["log_path"] = summary.get("log_path")
        if summary.get("last_exit_code") is not None:
            launcher_state["last_exit_code"] = summary.get("last_exit_code")

    def remember_pages_sync_warning(action: str, exc: Exception) -> None:
        detail = str(exc)
        hint = ""
        if isinstance(exc, PermissionError) or "Permission denied" in detail or "액세스가 거부" in detail:
            hint = " detectWarning-pages 폴더 권한 또는 동기화 중인 프로세스를 확인해 주세요."
        message = f"[pages-sync] {action} failed: {detail}{(' | ' + hint) if hint else ''}"
        print(message, file=sys.stderr)
        launcher_state["pages_sync_warning"] = message

    def safe_sync_pages_report(action: str = "report") -> None:
        try:
            sync_pages_report(
                paths,
                pages_sync,
                project_name=str(pages_sync["project_name"]),
                report_title=str(pages_sync["report_title"]),
                target_labels=get_target_labels(config),
            )
            launcher_state["pages_sync_warning"] = None
        except Exception as exc:
            remember_pages_sync_warning(action, exc)

    def safe_sync_pages_live(status: str, *, action: str) -> None:
        try:
            sync_pages_live(pages_sync, status)
            launcher_state["pages_sync_warning"] = None
        except Exception as exc:
            remember_pages_sync_warning(action, exc)

    if launcher_history_path.exists():
        try:
            with launcher_history_path.open("r", encoding="utf-8") as handle:
                history_payload = json.load(handle)
            completed_jobs = history_payload.get("completed_jobs", [])
            if isinstance(completed_jobs, list):
                restored_jobs = restore_completed_jobs_from_logs(job_logs_dir, runtime_config_dir)
                normalized_completed_jobs = merge_completed_jobs(completed_jobs, restored_jobs)
                launcher_state["completed_jobs"] = normalized_completed_jobs
                if normalized_completed_jobs != completed_jobs:
                    persist_launcher_history(launcher_history_path, launcher_state)
                apply_restored_launcher_summary(launcher_state["completed_jobs"])
        except (OSError, json.JSONDecodeError):
            launcher_state["completed_jobs"] = []

    if not launcher_state["completed_jobs"]:
        restored_jobs = restore_completed_jobs_from_logs(job_logs_dir, runtime_config_dir)
        if restored_jobs:
            launcher_state["completed_jobs"] = sort_jobs_by_recency(restored_jobs)
            persist_launcher_history(launcher_history_path, launcher_state)
            apply_restored_launcher_summary(launcher_state["completed_jobs"])

    def reload_completed_jobs_from_history() -> list[dict]:
        history_payload = read_json(launcher_history_path) or {}
        completed_jobs = history_payload.get("completed_jobs", [])
        restored_jobs = restore_completed_jobs_from_logs(job_logs_dir, runtime_config_dir)
        if isinstance(completed_jobs, list) and completed_jobs:
            normalized_completed_jobs = merge_completed_jobs(completed_jobs, restored_jobs)
            launcher_state["completed_jobs"] = normalized_completed_jobs
            if normalized_completed_jobs != completed_jobs:
                persist_launcher_history(launcher_history_path, launcher_state)
            apply_restored_launcher_summary(launcher_state["completed_jobs"])
            return launcher_state["completed_jobs"]
        if restored_jobs:
            launcher_state["completed_jobs"] = sort_jobs_by_recency(restored_jobs)
            persist_launcher_history(launcher_history_path, launcher_state)
            apply_restored_launcher_summary(launcher_state["completed_jobs"])
            return launcher_state["completed_jobs"]
        return []

    workspace_default_dir = paths.get("workspace_default_dir")
    workspace_source = str(paths.get("workspace_source") or "config")
    workspace_override_env = str(paths.get("workspace_override_env") or "").strip()
    active_workspace_dir = paths["workspace_dir"]
    if (
        workspace_source == "env"
        and isinstance(workspace_default_dir, Path)
        and workspace_default_dir != active_workspace_dir
    ):
        active_has_state = workspace_has_saved_state(active_workspace_dir)
        default_has_state = workspace_has_saved_state(workspace_default_dir)
        if not active_has_state and default_has_state:
            launcher_state["last_message"] = (
                f"현재는 환경변수 {workspace_override_env or 'DETECTWARNING_WORKSPACE_DIR'} 때문에 "
                f"{active_workspace_dir} 작업공간을 보고 있습니다. "
                f"기본 작업공간 {workspace_default_dir} 에 기존 학습 기록이 남아 있습니다."
            )
    elif not workspace_has_saved_state(active_workspace_dir):
        alternative_workspaces = [
            candidate
            for candidate in find_workspace_state_candidates(project_root, active_workspace_dir)
            if workspace_has_saved_state(candidate)
        ]
        if alternative_workspaces:
            alternative_text = ", ".join(str(path) for path in alternative_workspaces[:2])
            launcher_state["last_message"] = (
                f"현재 작업공간 {active_workspace_dir} 에는 저장된 학습 기록이 거의 없습니다. "
                f"다른 작업공간 {alternative_text} 에 기존 학습 기록이 남아 있을 가능성이 큽니다."
            )

    def terminate_process_tree(process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def hard_reset_workspace() -> None:
        predownload_processes = launcher_state.get("predownload_processes", {})
        if isinstance(predownload_processes, dict):
            for item in list(predownload_processes.values()):
                process = item.get("process") if isinstance(item, dict) else None
                if isinstance(process, subprocess.Popen) and process.poll() is None:
                    terminate_process_tree(process)
            predownload_processes.clear()
        reset_dirs = []
        for key in ("raw_dir", "import_dir", "predownload_dir", "extracted_dir", "manifests_dir", "prepared_dir", "artifacts_dir"):
            target = paths.get(key)
            if isinstance(target, Path):
                reset_dirs.append(target)
        workspace_dir = paths.get("workspace_dir")
        if isinstance(workspace_dir, Path):
            reset_dirs.extend(
                [
                    workspace_dir / "prepared_pose_guideline",
                    workspace_dir / "rgb_clip_features",
                    workspace_dir / "job_logs",
                    workspace_dir / "runtime_configs",
                ]
            )
        reset_dirs.extend([job_logs_dir, runtime_config_dir])

        seen_reset_dirs: set[Path] = set()
        for target in reset_dirs:
            resolved = target.resolve()
            if resolved in seen_reset_dirs:
                continue
            seen_reset_dirs.add(resolved)
            if target.exists():
                remove_transient_path_with_retries(target, recreate_dir=False, retries=12, delay_seconds=1.0)

        for target in (
            launcher_history_path,
            paths.get("pipeline_status"),
            paths.get("training_progress"),
        ):
            if isinstance(target, Path) and target.exists():
                remove_transient_path_with_retries(target, retries=12, delay_seconds=1.0)

        for key in (
            "workspace_dir",
            "raw_dir",
            "import_dir",
            "predownload_dir",
            "extracted_dir",
            "manifests_dir",
            "prepared_dir",
            "artifacts_dir",
        ):
            target = paths.get(key)
            if isinstance(target, Path):
                target.mkdir(parents=True, exist_ok=True)

        job_logs_dir.mkdir(parents=True, exist_ok=True)
        runtime_config_dir.mkdir(parents=True, exist_ok=True)

        launcher_state["process"] = None
        launcher_state["started_at"] = None
        launcher_state["runtime_config_path"] = None
        launcher_state["current_job"] = None
        launcher_state["queued_jobs"] = []
        launcher_state["completed_jobs"] = []
        launcher_state["auto_start_enabled"] = True
        launcher_state["auto_enqueue_enabled"] = False
        launcher_state["auto_extract_enabled"] = False
        launcher_state["auto_enqueue_datasetkey"] = None
        launcher_state["auto_enqueue_api_key"] = ""
        launcher_state["auto_rgb_model"] = "i3d_r50"
        launcher_state["predownload_processes"] = {}
        launcher_state["predownload_completed"] = []
        launcher_state["predownload_failed"] = []
        launcher_state["filekey_pipeline_status"] = {}
        launcher_state["last_state"] = "idle"
        launcher_state["last_exit_code"] = None
        launcher_state["last_message"] = "학습 워크스페이스를 초기화했습니다. 처음부터 다시 시작할 수 있습니다."
        launcher_state["log_path"] = None

        write_dashboard_status(
            paths,
            stage="idle",
            state="idle",
            message="학습 워크스페이스를 초기화했습니다.",
            stage_progress=0.0,
        )
        persist_launcher_history(launcher_history_path, launcher_state)
        safe_sync_pages_report("reset workspace report")

    def stop_process_tree(process: subprocess.Popen | None) -> int | None:
        if not isinstance(process, subprocess.Popen):
            return None

        if process.poll() is not None:
            return process.returncode

        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except Exception:
                    process.terminate()
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

        try:
            process.wait(timeout=8)
        except Exception:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=3)
            except Exception:
                pass
        return process.returncode

    def build_job_startup_message(job: dict) -> str:
        label = str(job.get("display_name") or job.get("filekey") or "job").strip()
        if str(job.get("job_kind") or "aihub").strip().lower() == "aihub":
            return (
                f"[launcher] datasetkey {job.get('datasetkey', '-')}"
                f" | filekey {job['filekey']} 작업을 시작합니다."
            )
        return f"[launcher] {label} 작업을 시작합니다."

    def build_job_command(
        job: dict,
        *,
        runtime_config_path: Path,
        pipeline_script: Path,
        project_root: Path,
    ) -> list[str]:
        job_kind = str(job.get("job_kind") or "aihub").strip().lower()
        if job_kind == "guideline":
            return [
                sys.executable,
                "-X",
                "utf8",
                str(project_root / "app" / "guideline_pose_dataset.py"),
                "--config",
                str(runtime_config_path),
                "--source",
                str(job.get("source") or "cumulative"),
            ]
        if job_kind == "rgb":
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(project_root / "app" / "extract_rgb_video_features.py"),
                "--config",
                str(runtime_config_path),
                "--model",
                str(job.get("rgb_model") or "i3d_r50"),
                "--device",
                str(job.get("device") or "cuda"),
            ]
            if payload_bool(job.get("reuse_existing_only", False)):
                command.append("--reuse-existing-only")
            return command
        if job_kind == "cleanup":
            return [
                sys.executable,
                "-X",
                "utf8",
                str(project_root / "app" / "cleanup_transient_data.py"),
                "--config",
                str(runtime_config_path),
            ]
        if job_kind == "auto_tune":
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(project_root / "app" / "auto_tune_action_training.py"),
                "--config",
                str(runtime_config_path),
                "--trials",
                str(max(int(job.get("trials") or 1), 1)),
                "--target-metric",
                str(job.get("target_metric") or "accuracy"),
            ]
            if not payload_bool(job.get("update_config", True)):
                command.append("--no-update-config")
            return command
        return [
            sys.executable,
            "-X",
            "utf8",
            str(pipeline_script),
            "--config",
            str(runtime_config_path),
            "--stage",
            str(job.get("stage") or "all"),
        ]

    def classify_dashboard_job_exit(current_job: dict | None, exit_code: int) -> tuple[str, str]:
        if isinstance(current_job, dict) and str(current_job.get("job_kind") or "aihub").strip().lower() != "aihub":
            if exit_code == 0:
                return "completed", str(current_job.get("success_message") or "대시보드 작업이 완료되었습니다.")
            return "error", f"{current_job.get('display_name') or current_job.get('filekey')} 작업이 종료 코드 {exit_code}로 중단되었습니다."
        return classify_job_exit(paths, current_job, exit_code)

    def start_pipeline_for_job(job: dict) -> None:
        job_kind = str(job.get("job_kind") or "aihub").strip().lower()
        metric_payloads_for_normal_ratio = [
            read_json(paths["artifacts_dir"] / "metrics.json"),
            read_json(paths["training_progress"]),
        ]
        if job_kind == "aihub":
            reset_training_workspace(paths)
            if payload_bool(job.get("reextract_filekey_only", False)):
                reset_filekey_training_outputs(
                    paths,
                    datasetkey=str(job.get("datasetkey") or ""),
                    filekey=str(job.get("filekey") or ""),
                )
        runtime_config_dir.mkdir(parents=True, exist_ok=True)
        job_logs_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = json.loads(json.dumps(config))
        runtime_config["dataset_source"] = "aihub_shell"
        runtime_paths = runtime_config.setdefault("paths", {})
        runtime_paths["workspace_dir"] = str(paths["workspace_dir"])
        if job_kind in {"guideline", "aihub"}:
            normal_ratio_adjustment = apply_auto_normal_ratio_to_config(
                runtime_config,
                metric_payloads=metric_payloads_for_normal_ratio,
                labels=get_target_labels(runtime_config),
            )
            if normal_ratio_adjustment.get("enabled") and normal_ratio_adjustment.get("source") == "metrics":
                launcher_state["last_normal_ratio_adjustment"] = normal_ratio_adjustment
            cached_adjustment = (
                job.get("normal_ratio_adjustment")
                if isinstance(job.get("normal_ratio_adjustment"), dict)
                else launcher_state.get("last_normal_ratio_adjustment")
            )
            if (
                normal_ratio_adjustment.get("reason") in {"no_metrics", "missing_confusion_matrix"}
                and isinstance(cached_adjustment, dict)
                and cached_adjustment.get("enabled")
                and cached_adjustment.get("target_ratio") is not None
            ):
                runtime_guideline = runtime_config.setdefault("guideline_sampling", {})
                runtime_guideline["target_normal_clip_ratio"] = cached_adjustment["target_ratio"]
                normal_ratio_adjustment = {
                    **cached_adjustment,
                    "source": "cached_metrics",
                    "applied": True,
                }
                runtime_guideline["_auto_normal_ratio_adjustment"] = normal_ratio_adjustment
            if normal_ratio_adjustment.get("enabled"):
                job["normal_ratio_adjustment"] = normal_ratio_adjustment
        runtime_shell = runtime_config.setdefault("aihub_shell", {})
        if job_kind == "aihub":
            if job.get("datasetkey") not in (None, ""):
                runtime_shell["datasetkey"] = job["datasetkey"]
            runtime_shell["filekey"] = job["filekey"]
            if job.get("api_key"):
                runtime_shell["api_key"] = str(job["api_key"])
                runtime_shell["api_key_env"] = ""
        if job_kind == "auto_tune":
            runtime_auto_tune = runtime_config.setdefault("auto_tune", {})
            runtime_auto_tune["max_trials"] = max(int(job.get("trials") or runtime_auto_tune.get("max_trials") or 1), 1)
            runtime_auto_tune["target_metric"] = str(job.get("target_metric") or runtime_auto_tune.get("target_metric") or "accuracy")
            runtime_auto_tune["ensemble_enabled"] = True
            runtime_auto_tune["hybrid_enabled"] = True
            runtime_auto_tune["promote_best"] = True
            runtime_auto_tune["promote_ensemble"] = True
            runtime_auto_tune["promote_hybrid"] = True

        training_label_filter = apply_training_label_filter_to_runtime_config(runtime_config, job)
        manifest_training_optimization = apply_manifest_aware_training_overrides(runtime_config, paths, job)
        if manifest_training_optimization:
            job["manifest_training_optimization"] = manifest_training_optimization
        if training_label_filter:
            job["training_label_filter"] = training_label_filter

        runtime_config_path = runtime_config_dir / f"{job['job_id']}.json"
        write_json_atomic(runtime_config_path, runtime_config)

        log_path = job_logs_dir / f"{job['job_id']}.log"
        startup_message = (
            f"[launcher] datasetkey {job.get('datasetkey', '-')}"
            f" | filekey {job['filekey']} 작업을 시작합니다."
        )
        startup_message = build_job_startup_message(job)
        write_dashboard_status(
            paths,
            stage=str(job.get("stage") or "queued"),
            state="running",
            message=startup_message.replace("[launcher] ", ""),
            current_filekey=job["filekey"],
            current_datasetkey=job.get("datasetkey"),
            job_kind=job_kind,
        )
        command = build_job_command(
            job,
            runtime_config_path=runtime_config_path,
            pipeline_script=pipeline_script,
            project_root=project_root,
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"{startup_message}\n")
            log_handle.write(f"[launcher] runtime config: {runtime_config_path}\n")
            if manifest_training_optimization:
                log_handle.write(
                    "[launcher] manifest training optimization: "
                    f"pose_ready={manifest_training_optimization.get('pose_ready_ratio')} "
                    f"rgb_ready={manifest_training_optimization.get('rgb_ready_ratio')} "
                    f"class_weight_multipliers={manifest_training_optimization.get('class_weight_multipliers')} "
                    f"target_metric={manifest_training_optimization.get('auto_tune_target_metric')}\n"
                )
            if training_label_filter:
                log_handle.write(
                    "[launcher] training label filter: "
                    f"excluded={training_label_filter.get('excluded_labels')} "
                    f"labels={training_label_filter.get('labels_after')}\n"
                )
            log_handle.write(f"[launcher] command: {' '.join(str(part) for part in command)}\n")
            log_handle.flush()
            child_env = os.environ.copy()
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=child_env,
                start_new_session=(os.name != "nt"),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
            )

        job["started_at"] = current_timestamp()
        job["state"] = "running"
        job["runtime_config_path"] = runtime_config_path
        job["log_path"] = log_path
        job["message"] = startup_message.replace("[launcher] ", "")
        launcher_state["process"] = process
        launcher_state["started_at"] = job["started_at"]
        launcher_state["runtime_config_path"] = runtime_config_path
        launcher_state["current_job"] = job
        launcher_state["last_state"] = "running"
        launcher_state["last_exit_code"] = None
        launcher_state["log_path"] = log_path
        launcher_state["last_message"] = (
            f"datasetkey {job.get('datasetkey', '-')} | filekey {job['filekey']} 학습을 진행 중입니다."
        )
        launcher_state["last_message"] = job.get("running_message") or startup_message.replace("[launcher] ", "")
        safe_sync_pages_live("online", action="job start live status")
        if filekey_pipeline_step_id(job):
            dispatch_filekey_step_started_notification(job, launcher_state["last_message"])

    def remove_dependent_followups_locked(failed_job: dict) -> int:
        source_filekey = str(failed_job.get("source_filekey") or failed_job.get("filekey") or "").strip()
        if not source_filekey:
            return 0
        source_datasetkey = str(failed_job.get("source_datasetkey") or failed_job.get("datasetkey") or "").strip()
        pending_jobs = launcher_state.get("queued_jobs", [])
        if not isinstance(pending_jobs, list) or not pending_jobs:
            return 0
        dependent_kinds = {"guideline", "rgb", "cleanup", "train_guideline"}
        retained: list[dict] = []
        removed_count = 0
        for pending_job in pending_jobs:
            if not isinstance(pending_job, dict):
                retained.append(pending_job)
                continue
            pending_kind = infer_dashboard_job_kind(pending_job)
            pending_filekey = str(pending_job.get("source_filekey") or pending_job.get("filekey") or "").strip()
            pending_datasetkey = str(pending_job.get("source_datasetkey") or pending_job.get("datasetkey") or "").strip()
            same_source = pending_filekey == source_filekey and (
                not source_datasetkey
                or not pending_datasetkey
                or pending_datasetkey in {source_datasetkey, "prepared"}
            )
            if pending_kind in dependent_kinds and same_source:
                removed_count += 1
                continue
            retained.append(pending_job)
        if removed_count:
            launcher_state["queued_jobs"] = retained
        return removed_count

    def update_process_state() -> None:
        update_predownload_processes_locked()
        process = launcher_state.get("process")
        if process is not None and isinstance(process, subprocess.Popen):
            exit_code = process.poll()
            if exit_code is None:
                launcher_state["last_state"] = "running"
                current_job = launcher_state.get("current_job") or {}
                filekey = current_job.get("filekey", "-") if isinstance(current_job, dict) else "-"
                datasetkey = current_job.get("datasetkey", "-") if isinstance(current_job, dict) else "-"
                if (
                    isinstance(current_job, dict)
                    and str(current_job.get("job_kind") or "aihub").strip().lower() != "aihub"
                ):
                    launcher_state["last_message"] = str(
                        current_job.get("running_message")
                        or current_job.get("message")
                        or f"{current_job.get('display_name') or filekey} 작업을 실행 중입니다."
                    )
                    return
                pending_jobs = launcher_state.get("queued_jobs", [])
                auto_start_enabled = bool(launcher_state.get("auto_start_enabled", True))
                if not auto_start_enabled and isinstance(pending_jobs, list) and pending_jobs:
                    launcher_state["last_message"] = (
                        f"datasetkey {datasetkey} | filekey {filekey} 작업이 실행 중입니다. "
                        "현재 작업이 끝나면 다음 큐 자동 시작은 멈춥니다."
                    )
                else:
                    launcher_state["last_message"] = (
                        f"datasetkey {datasetkey} | filekey {filekey} 작업이 실행 중입니다."
                    )
            else:
                current_job = launcher_state.get("current_job")
                final_message = f"현재 작업이 종료 코드 {exit_code}로 중단되었습니다."
                if isinstance(current_job, dict):
                    current_job["finished_at"] = current_timestamp()
                    current_job["exit_code"] = exit_code
                    final_state, final_message = classify_dashboard_job_exit(current_job, exit_code)
                    current_job["state"] = final_state
                    current_job["message"] = final_message
                    current_job["result_summary"] = collect_result_summary(paths)
                    current_job = enrich_completed_job(current_job)
                    completed_jobs = launcher_state.setdefault("completed_jobs", [])
                    if isinstance(completed_jobs, list):
                        completed_jobs.insert(0, snapshot_job(current_job))
                        del completed_jobs[30:]
                    persist_launcher_history(launcher_history_path, launcher_state)
                    safe_sync_pages_report("job completion report")
                launcher_state["process"] = None
                launcher_state["current_job"] = None
                launcher_state["last_exit_code"] = exit_code
                if isinstance(current_job, dict) and current_job.get("state") in {
                    "completed",
                    "completed_warning",
                    "data_ready",
                }:
                    launcher_state["last_state"] = current_job.get("state")
                    launcher_state["last_message"] = final_message
                else:
                    launcher_state["last_state"] = "error"
                    launcher_state["last_message"] = final_message
                    if isinstance(current_job, dict) and infer_dashboard_job_kind(current_job) == "aihub":
                        removed_followups = remove_dependent_followups_locked(current_job)
                        if removed_followups:
                            launcher_state["last_message"] = (
                                f"{final_message} Dependent guideline/RGB/cleanup jobs for filekey "
                                f"{current_job.get('filekey') or '-'} were removed: {removed_followups}."
                            )
                if isinstance(current_job, dict):
                    job_state = str(current_job.get("state") or launcher_state.get("last_state") or "unknown")
                    if filekey_pipeline_step_id(current_job):
                        _, final_event, final_title, final_body = record_filekey_pipeline_finished(
                            current_job,
                            job_state,
                            final_message,
                        )
                        if final_event:
                            dispatch_dashboard_notification(
                                final_event,
                                title=final_title,
                                message=final_body,
                                priority="high" if final_event == "error" else "default",
                            )
                    else:
                        notify_title, notify_body, notify_priority = build_job_notification_payload(
                            current_job,
                            job_state,
                            final_message,
                        )
                        notification_event = str(current_job.get("state") or "error")
                        if notification_event == "data_ready":
                            notification_event = "completed"
                        dispatch_dashboard_notification(
                            notification_event,
                            title=notify_title,
                            message=notify_body,
                            priority=notify_priority,
                        )

        active_process = launcher_state.get("process")
        queued_jobs = launcher_state.get("queued_jobs", [])
        maybe_start_predownloads_locked()
        if (
            active_process is None
            and bool(launcher_state.get("auto_start_enabled", True))
            and bool(launcher_state.get("auto_extract_enabled", False))
            and isinstance(queued_jobs, list)
            and not queued_jobs
        ):
            if performance_plan_is_active() and performance_plan_deadline_reached():
                enqueue_performance_plan_training_locked("deadline_reached")
            else:
                resumed = enqueue_interrupted_auto_extract_job_locked(
                    str(launcher_state.get("auto_enqueue_datasetkey") or ""),
                    str(launcher_state.get("auto_enqueue_api_key") or ""),
                    rgb_model=str(launcher_state.get("auto_rgb_model") or "i3d_r50"),
                )
                if resumed is None:
                    enqueue_next_extract_recommended_job_locked()
            queued_jobs = launcher_state.get("queued_jobs", [])
        if (
            active_process is None
            and bool(launcher_state.get("auto_start_enabled", True))
            and bool(launcher_state.get("auto_enqueue_enabled", False))
            and isinstance(queued_jobs, list)
            and not queued_jobs
        ):
            enqueue_next_recommended_job_locked()
            queued_jobs = launcher_state.get("queued_jobs", [])
        if active_process is None and isinstance(queued_jobs, list) and queued_jobs:
            if bool(launcher_state.get("auto_start_enabled", True)):
                next_job = queued_jobs.pop(0)
                start_pipeline_for_job(next_job)
            else:
                launcher_state["last_state"] = "paused"
                launcher_state["last_message"] = "다음 큐 자동 시작이 중지되었습니다. 시작 버튼을 누르면 재개합니다."

    def queue_worker() -> None:
        while True:
            with state_lock:
                update_process_state()
            time.sleep(1.0)

    def get_launcher_status() -> dict:
        with state_lock:
            update_process_state()
            active_process = launcher_state.get("process")
            active_pid = active_process.pid if isinstance(active_process, subprocess.Popen) else None
            current_job = snapshot_job(launcher_state.get("current_job"))
            queued_jobs = launcher_state.get("queued_jobs", [])
            completed_jobs = launcher_state.get("completed_jobs", [])
            predownload_processes = launcher_state.get("predownload_processes", {})
            predownload_running = []
            if isinstance(predownload_processes, dict):
                for item in predownload_processes.values():
                    if isinstance(item, dict):
                        predownload_running.append({key: value for key, value in item.items() if key != "process"})
            predownload_completed = (
                list(launcher_state.get("predownload_completed", []))
                if isinstance(launcher_state.get("predownload_completed"), list)
                else []
            )
            predownload_completed_filekeys = {
                str(item.get("filekey") or "").strip()
                for item in predownload_completed
                if isinstance(item, dict)
            }
            datasetkey_for_cache = str(
                launcher_state.get("auto_enqueue_datasetkey")
                or config.get("aihub_shell", {}).get("datasetkey")
                or ""
            ).strip()
            for item in collect_completed_predownload_cache_items_locked(datasetkey_for_cache):
                filekey = str(item.get("filekey") or "").strip()
                if filekey and filekey not in predownload_completed_filekeys:
                    predownload_completed.append(item)
                    predownload_completed_filekeys.add(filekey)
            if (not isinstance(completed_jobs, list) or not completed_jobs) and current_job is None:
                completed_jobs = reload_completed_jobs_from_history()
            elif current_job is None:
                apply_restored_launcher_summary(completed_jobs if isinstance(completed_jobs, list) else [])
        return {
            "state": launcher_state.get("last_state", "idle"),
            "message": launcher_state.get("last_message", ""),
            "pages_sync_warning": launcher_state.get("pages_sync_warning"),
            "pid": active_pid,
            "started_at": launcher_state.get("started_at"),
            "auto_start_enabled": bool(launcher_state.get("auto_start_enabled", True)),
            "auto_enqueue_enabled": bool(launcher_state.get("auto_enqueue_enabled", False)),
            "auto_enqueue_datasetkey": launcher_state.get("auto_enqueue_datasetkey"),
            "auto_extract_enabled": bool(launcher_state.get("auto_extract_enabled", False)),
            "performance_plan": performance_plan_payload(),
            "runtime_config_path": str(launcher_state["runtime_config_path"])
            if launcher_state.get("runtime_config_path")
            else None,
            "current_job": current_job,
            "pending_jobs": [snapshot_job(job) for job in queued_jobs] if isinstance(queued_jobs, list) else [],
            "completed_jobs": [snapshot_job(job) for job in completed_jobs] if isinstance(completed_jobs, list) else [],
            "predownload": {
                "enabled": bool(launcher_state.get("predownload_enabled", True)),
                "max_parallel": int(launcher_state.get("predownload_max_parallel") or 2),
                "min_free_gb": float(launcher_state.get("predownload_min_free_gb") or 100),
                "free_gb": round(predownload_free_disk_gb(), 2),
                "pause_reason": str(launcher_state.get("predownload_pause_reason") or ""),
                "running": predownload_running,
                "completed": predownload_completed,
                "failed": launcher_state.get("predownload_failed", [])
                if isinstance(launcher_state.get("predownload_failed"), list)
                else [],
            },
            "last_exit_code": launcher_state.get("last_exit_code"),
            "log_path": str(launcher_state["log_path"]) if launcher_state.get("log_path") else None,
            "notification_settings": notification_settings_public(launcher_state.get("notification_settings")),
        }

    safe_sync_pages_report("startup report")
    safe_sync_pages_live("online", action="startup live status")

    def sync_pages_shutdown() -> None:
        try:
            safe_sync_pages_report("shutdown report")
        except Exception as exc:
            print(f"[pages-sync] 종료 시 report 동기화 실패: {exc}")
        try:
            safe_sync_pages_live("offline", action="shutdown live status")
        except Exception as exc:
            print(f"[pages-sync] 종료 시 offline 동기화 실패: {exc}")
        try:
            flush_pages_pushes()
        except Exception as exc:
            print(f"[pages-sync] 종료 시 push flush 실패: {exc}")

    atexit.register(sync_pages_shutdown)

    def server_tone_class(state: str) -> str:
        normalized = str(state or "").strip().lower()
        if normalized in {"completed", "completed_warning", "online"}:
            return "tone-good"
        if normalized in {"running", "queued", "data_ready"}:
            return "tone-accent"
        if normalized in {"paused", "warning"}:
            return "tone-warn"
        if normalized in {"error", "aborted", "offline"}:
            return "tone-danger"
        return "tone-neutral"

    def build_server_snapshot_html(overview: dict | None) -> str:
        overview = overview or {}
        launcher = overview.get("launcher") or {}
        pipeline = overview.get("pipeline_status") or {}
        progress = overview.get("training_progress") or {}
        metrics = overview.get("metrics") or {}
        dataset = overview.get("dataset") or {}
        latest = progress.get("latest") or {}
        final_validation = metrics.get("final_validation") or progress.get("final_validation") or {}
        model_type = metrics.get("model_type") or "single"
        current_state = launcher.get("state") or pipeline.get("state") or "idle"
        prepared_total = sum(
            int((dataset.get(key) or {}).get("total", 0) or 0)
            for key in ("prepared_train", "prepared_val", "prepared_test")
        )
        raw_total = int((dataset.get("raw") or {}).get("total", 0) or 0)
        completed_jobs = launcher.get("completed_jobs") or []
        latest_jobs = []
        for job in completed_jobs[:5]:
            if not isinstance(job, dict):
                continue
            latest_jobs.append(
                "<li>"
                f"<strong>{escape(str(job.get('filekey') or '-'))}</strong>"
                f" <span style=\"color: var(--muted);\">{escape(str(job.get('state') or '-'))}</span>"
                "</li>"
            )
        jobs_html = "".join(latest_jobs) or "<li>완료된 작업 기록이 없습니다.</li>"
        pages_sync_warning = launcher.get("pages_sync_warning")
        warning_html = ""
        if pages_sync_warning:
            warning_html = (
                "<div style=\"margin-top:12px;padding:12px 14px;border-radius:14px;"
                "background:rgba(217,119,6,0.08);border:1px solid rgba(217,119,6,0.16);"
                "color:#9a3412;font-size:13px;line-height:1.6;\">"
                "<strong>Pages 동기화 경고</strong><br>"
                f"{escape(str(pages_sync_warning))}"
                "</div>"
            )
        latest_epoch = latest.get("epoch", "-")
        latest_acc = latest.get("val_accuracy")
        latest_f1 = latest.get("val_macro_f1")
        best_f1 = progress.get("best_val_macro_f1")
        best_epoch = progress.get("best_epoch", "-")
        latest_acc_text = f"{float(latest_acc):.3f}" if latest_acc is not None else "-"
        latest_f1_text = f"{float(latest_f1):.3f}" if latest_f1 is not None else "-"
        final_acc = final_validation.get("accuracy")
        final_f1 = final_validation.get("macro_f1")
        final_acc_text = f"{float(final_acc):.3f}" if final_acc is not None else "-"
        final_f1_text = f"{float(final_f1):.3f}" if final_f1 is not None else "-"
        message = launcher.get("message") or pipeline.get("message") or "상세 메시지가 없습니다."
        return (
            "<section class=\"card\" style=\"margin-bottom:20px;padding:24px 24px 18px;\">"
            "<div style=\"display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start;\">"
            "<div>"
            "<div class=\"hero-side-label\">Server Snapshot</div>"
            "<h2 style=\"margin:6px 0 8px;font-size:24px;letter-spacing:-0.03em;\">"
            "현재 결과 요약</h2>"
            "<div style=\"color:var(--muted);font-size:14px;line-height:1.7;\">"
            "스크립트 렌더링이 멈춰도 이 영역은 서버가 직접 채웁니다."
            "</div>"
            "</div>"
            f"<div class=\"status-pill {server_tone_class(current_state)}\">{escape(str(current_state))}</div>"
            "</div>"
            "<div class=\"hero-meta\" style=\"margin-top:16px;\">"
            f"<div class=\"hero-chip\"><strong>Final</strong> acc {final_acc_text} / f1 {final_f1_text} / {escape(str(model_type))}</div>"
            f"<div class=\"hero-chip\"><strong>Latest</strong> epoch {escape(str(latest_epoch))} / acc {latest_acc_text} / f1 {latest_f1_text}</div>"
            f"<div class=\"hero-chip\"><strong>Dataset</strong> raw {raw_total} / prepared {prepared_total}</div>"
            f"<div class=\"hero-chip\"><strong>Completed Jobs</strong> {len(completed_jobs)}</div>"
            "</div>"
            "<div style=\"margin-top:14px;padding:12px 14px;border-radius:14px;background:rgba(255,255,255,0.78);"
            "border:1px solid rgba(148,163,184,0.16);color:var(--muted);font-size:13px;line-height:1.7;\">"
            f"{escape(str(message))}"
            "</div>"
            f"{warning_html}"
            "<div style=\"margin-top:16px;display:grid;grid-template-columns:minmax(0,1.3fr) minmax(240px,0.7fr);gap:16px;\">"
            "<div style=\"padding:16px;border-radius:18px;background:rgba(248,250,252,0.88);border:1px solid rgba(148,163,184,0.14);\">"
            "<div class=\"hero-side-label\">Workspace</div>"
            f"<div style=\"margin-top:8px;font-size:13px;line-height:1.7;color:var(--muted);word-break:break-all;\">{escape(str(overview.get('workspace_dir') or '-'))}</div>"
            "</div>"
            "<div style=\"padding:16px;border-radius:18px;background:rgba(248,250,252,0.88);border:1px solid rgba(148,163,184,0.14);\">"
            "<div class=\"hero-side-label\">Recent Jobs</div>"
            f"<ol style=\"margin:10px 0 0;padding-left:18px;color:var(--ink);line-height:1.8;\">{jobs_html}</ol>"
            "</div>"
            "</div>"
            "</section>"
        )

    def build_server_fallback_sections_html(overview: dict | None) -> str:
        overview = overview or {}
        progress = overview.get("training_progress") or {}
        metrics = overview.get("metrics") or {}
        dataset = overview.get("dataset") or {}
        launcher = overview.get("launcher") or {}
        diagnostics = overview.get("diagnostics") or {}
        final_validation = metrics.get("final_validation") or progress.get("final_validation") or {}
        labels = metrics.get("labels") or progress.get("labels") or []
        history = list(progress.get("history") or metrics.get("history") or [])
        completed_jobs = launcher.get("completed_jobs") or []
        per_class_rows = final_validation.get("per_class") or []
        warning_rows = diagnostics.get("warnings") or []

        def fmt(value, digits: int = 3) -> str:
            if value in (None, ""):
                return "-"
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return escape(str(value))

        dataset_rows = []
        for key in ("raw", "train", "val", "test", "prepared_train", "prepared_val", "prepared_test"):
            info = dataset.get(key) or {}
            total = int(info.get("total", 0) or 0)
            if total <= 0:
                continue
            labels_text = ", ".join(
                f"{label} {count}" for label, count in sorted((info.get("by_label") or {}).items())
            ) or "-"
            dataset_rows.append(
                "<tr>"
                f"<td>{escape(key)}</td>"
                f"<td>{total}</td>"
                f"<td>{escape(labels_text)}</td>"
                "</tr>"
            )
        dataset_html = "".join(dataset_rows) or (
            "<tr><td colspan=\"3\" style=\"text-align:center;color:var(--muted);\">표시할 누적 데이터가 없습니다.</td></tr>"
        )
        dataset_list_items = []
        for key in ("raw", "train", "val", "test", "prepared_train", "prepared_val", "prepared_test"):
            info = dataset.get(key) or {}
            total = int(info.get("total", 0) or 0)
            if total <= 0:
                continue
            labels_text = ", ".join(
                f"{label} {count}" for label, count in sorted((info.get("by_label") or {}).items())
            ) or "-"
            dataset_list_items.append(
                f"<li><strong>{escape(key)}</strong> {total}개"
                f"<div class=\"server-fallback-copy\">{escape(labels_text)}</div></li>"
            )
        dataset_list_html = "".join(dataset_list_items) or (
            "<li><strong>누적 데이터셋</strong><div class=\"server-fallback-copy\">표시할 데이터가 없습니다.</div></li>"
        )

        history_rows = []
        for row in history[-10:]:
            if not isinstance(row, dict):
                continue
            history_rows.append(
                "<tr>"
                f"<td>{escape(str(row.get('epoch', '-')))}</td>"
                f"<td>{fmt(row.get('train_loss'), 4)}</td>"
                f"<td>{fmt(row.get('val_loss'), 4)}</td>"
                f"<td>{fmt(row.get('val_accuracy'), 4)}</td>"
                f"<td>{fmt(row.get('val_macro_f1'), 4)}</td>"
                "</tr>"
            )
        history_html = "".join(history_rows) or (
            "<tr><td colspan=\"5\" style=\"text-align:center;color:var(--muted);\">epoch history가 없습니다.</td></tr>"
        )
        history_list_items = []
        for row in history[-5:]:
            if not isinstance(row, dict):
                continue
            history_list_items.append(
                "<li>"
                f"<strong>epoch {escape(str(row.get('epoch', '-')))}</strong>"
                f"<div class=\"server-fallback-copy\">"
                f"train {fmt(row.get('train_loss'), 4)} / val {fmt(row.get('val_loss'), 4)} / "
                f"acc {fmt(row.get('val_accuracy'), 4)} / f1 {fmt(row.get('val_macro_f1'), 4)}"
                "</div>"
                "</li>"
            )
        history_list_html = "".join(history_list_items) or (
            "<li><strong>Epoch History</strong><div class=\"server-fallback-copy\">기록이 없습니다.</div></li>"
        )

        support_map: dict[int, int] = {}
        confusion = final_validation.get("confusion_matrix") or []
        if isinstance(confusion, list):
            for index, row in enumerate(confusion):
                if isinstance(row, list):
                    support_map[index] = sum(int(value or 0) for value in row)
        per_class_html_rows = []
        for index, row in enumerate(per_class_rows):
            if not isinstance(row, dict):
                continue
            class_index = int(row.get("class_index", index) or index)
            label = labels[class_index] if 0 <= class_index < len(labels) else str(row.get("label") or class_index)
            per_class_html_rows.append(
                "<tr>"
                f"<td>{escape(label)}</td>"
                f"<td>{fmt(row.get('precision'), 4)}</td>"
                f"<td>{fmt(row.get('recall'), 4)}</td>"
                f"<td>{fmt(row.get('f1'), 4)}</td>"
                f"<td>{support_map.get(class_index, 0)}</td>"
                "</tr>"
            )
        per_class_html = "".join(per_class_html_rows) or (
            "<tr><td colspan=\"5\" style=\"text-align:center;color:var(--muted);\">클래스별 지표가 없습니다.</td></tr>"
        )
        per_class_list_items = []
        for index, row in enumerate(per_class_rows):
            if not isinstance(row, dict):
                continue
            class_index = int(row.get("class_index", index) or index)
            label = labels[class_index] if 0 <= class_index < len(labels) else str(row.get("label") or class_index)
            per_class_list_items.append(
                "<li>"
                f"<strong>{escape(label)}</strong>"
                f"<div class=\"server-fallback-copy\">"
                f"precision {fmt(row.get('precision'), 4)} / "
                f"recall {fmt(row.get('recall'), 4)} / "
                f"f1 {fmt(row.get('f1'), 4)} / "
                f"support {support_map.get(class_index, 0)}"
                "</div>"
                "</li>"
            )
        per_class_list_html = "".join(per_class_list_items) or (
            "<li><strong>클래스별 지표</strong><div class=\"server-fallback-copy\">기록이 없습니다.</div></li>"
        )

        job_rows = []
        for job in completed_jobs[:8]:
            if not isinstance(job, dict):
                continue
            summary = job.get("result_summary") or {}
            prepared_total = sum(
                int(summary.get(key, 0) or 0)
                for key in ("prepared_train_total", "prepared_val_total", "prepared_test_total")
            )
            job_rows.append(
                "<tr>"
                f"<td>{escape(str(job.get('filekey') or '-'))}</td>"
                f"<td>{escape(str(job.get('state') or '-'))}</td>"
                f"<td>{prepared_total}</td>"
                f"<td>{escape(str(job.get('finished_at') or job.get('started_at') or '-'))}</td>"
                "</tr>"
            )
        jobs_fallback_html = "".join(job_rows) or (
            "<tr><td colspan=\"4\" style=\"text-align:center;color:var(--muted);\">완료 이력이 없습니다.</td></tr>"
        )
        jobs_list_items = []
        for job in completed_jobs[:6]:
            if not isinstance(job, dict):
                continue
            summary = job.get("result_summary") or {}
            prepared_total = sum(
                int(summary.get(key, 0) or 0)
                for key in ("prepared_train_total", "prepared_val_total", "prepared_test_total")
            )
            jobs_list_items.append(
                "<li>"
                f"<strong>{escape(str(job.get('filekey') or '-'))} / {escape(str(job.get('state') or '-'))}</strong>"
                f"<div class=\"server-fallback-copy\">prepared {prepared_total} / "
                f"finished {escape(str(job.get('finished_at') or job.get('started_at') or '-'))}</div>"
                "</li>"
            )
        jobs_list_html = "".join(jobs_list_items) or (
            "<li><strong>최근 작업 이력</strong><div class=\"server-fallback-copy\">기록이 없습니다.</div></li>"
        )

        warning_html = ""
        if warning_rows:
            warning_items = "".join(
                f"<li style=\"margin:4px 0;\">{escape(str(item))}</li>"
                for item in warning_rows[:6]
            )
            warning_html = (
                "<div style=\"margin-bottom:16px;padding:14px 16px;border-radius:16px;"
                "background:rgba(217,119,6,0.08);border:1px solid rgba(217,119,6,0.16);"
                "color:#9a3412;font-size:13px;line-height:1.7;\">"
                "<strong>상태 진단</strong>"
                f"<ul style=\"margin:8px 0 0 18px;padding:0;\">{warning_items}</ul>"
                "</div>"
            )

        return (
            "<section class=\"server-fallback-data\">"
            "<div class=\"section-title\">"
            "<div><h2>서버 렌더링 결과</h2><p>브라우저 스크립트가 실행되지 않아도 핵심 학습 데이터를 바로 보여줍니다.</p></div>"
            "<div class=\"section-pill\">Fallback</div>"
            "</div>"
            f"{warning_html}"
            "<div class=\"server-fallback-summary\">"
            "<article class=\"mini-card\"><div class=\"mini-title\">누적 데이터셋</div>"
            f"<ul class=\"server-fallback-list\">{dataset_list_html}</ul></article>"
            "<article class=\"mini-card\"><div class=\"mini-title\">최근 Epoch</div>"
            f"<ul class=\"server-fallback-list\">{history_list_html}</ul></article>"
            "<article class=\"mini-card\"><div class=\"mini-title\">클래스별 지표</div>"
            f"<ul class=\"server-fallback-list\">{per_class_list_html}</ul></article>"
            "<article class=\"mini-card\"><div class=\"mini-title\">최근 작업 이력</div>"
            f"<ul class=\"server-fallback-list\">{jobs_list_html}</ul></article>"
            "</div>"
            "<div class=\"server-fallback-grid\">"
            "<article class=\"panel\"><div class=\"panel-head\"><div><h2 class=\"panel-title\">누적 데이터셋</h2><div class=\"panel-copy\">cumulative manifests 기준</div></div></div><div class=\"panel-body\">"
            "<table class=\"table metric-table\"><thead><tr><th>split</th><th>total</th><th>labels</th></tr></thead>"
            f"<tbody>{dataset_html}</tbody></table></div></article>"
            "<article class=\"panel\"><div class=\"panel-head\"><div><h2 class=\"panel-title\">Epoch History</h2><div class=\"panel-copy\">최근 10 epoch</div></div></div><div class=\"panel-body\">"
            "<table class=\"table metric-table\"><thead><tr><th>epoch</th><th>train loss</th><th>val loss</th><th>val acc</th><th>val f1</th></tr></thead>"
            f"<tbody>{history_html}</tbody></table></div></article>"
            "<article class=\"panel\"><div class=\"panel-head\"><div><h2 class=\"panel-title\">클래스별 지표</h2><div class=\"panel-copy\">최종 validation 결과</div></div></div><div class=\"panel-body\">"
            "<table class=\"table metric-table\"><thead><tr><th>label</th><th>precision</th><th>recall</th><th>f1</th><th>support</th></tr></thead>"
            f"<tbody>{per_class_html}</tbody></table></div></article>"
            "<article class=\"panel\"><div class=\"panel-head\"><div><h2 class=\"panel-title\">최근 작업 이력</h2><div class=\"panel-copy\">launcher history 기준</div></div></div><div class=\"panel-body\">"
            "<table class=\"table metric-table\"><thead><tr><th>filekey</th><th>state</th><th>prepared</th><th>finished</th></tr></thead>"
            f"<tbody>{jobs_fallback_html}</tbody></table></div></article>"
            "</div>"
            "</section>"
        )

    def serialize_initial_overview(overview: dict | None) -> str:
        return json.dumps(overview or {}, ensure_ascii=False).replace("</", "<\\/")

    def render_dashboard(initial_overview: dict | None = None) -> str:
        template = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Training Dashboard</title>
  <script id="dashboardAppSource" type="text/plain">
    window.__dashboardBootErrors = [];
    window.addEventListener('error', function (event) {
      try {
        var message = 'unknown error';
        if (event && typeof event.message === 'string' && event.message) {
          message = event.message;
        } else if (event && event.error !== undefined && event.error !== null) {
          message = String(event.error);
        }
        window.__dashboardBootErrors.push(message);
        var banner = document.getElementById('bootErrorBanner');
        if (banner) {
          banner.textContent = '브라우저 스크립트 오류: ' + message;
          banner.style.display = 'block';
        }
      } catch (error) {}
    });
    window.addEventListener('unhandledrejection', function (event) {
      try {
        var reason = event && event.reason !== undefined ? event.reason : null;
        var message = 'unknown rejection';
        if (reason && typeof reason.message === 'string' && reason.message) {
          message = reason.message;
        } else if (reason !== undefined && reason !== null) {
          message = String(reason);
        }
        window.__dashboardBootErrors.push(message);
        var banner = document.getElementById('bootErrorBanner');
        if (banner) {
          banner.textContent = '브라우저 스크립트 오류: ' + message;
          banner.style.display = 'block';
        }
      } catch (error) {}
    });
    (function () {
      try {
        const params = new URLSearchParams(window.location.search);
        const host = String(window.location.hostname || '').toLowerCase();
        const isTunnelHost =
          host.includes('trycloudflare') ||
          host.endsWith('.workers.dev');
        if (params.get('viewer') === '1' || isTunnelHost) {
          document.documentElement.classList.add('viewer-mode-page');
        }
      } catch (error) {}
    })();
  </script>
  <style>
    :root {
      --bg: #eef3fb;
      --bg-deep: #e3ebf8;
      --panel: rgba(255, 255, 255, 0.92);
      --panel-strong: rgba(255, 255, 255, 0.98);
      --panel-soft: rgba(246, 249, 253, 0.92);
      --ink: #0f172a;
      --muted: #5f6f86;
      --line: rgba(148, 163, 184, 0.18);
      --line-strong: rgba(148, 163, 184, 0.28);
      --accent: #2563eb;
      --accent-strong: #1d4ed8;
      --accent-soft: rgba(37, 99, 235, 0.12);
      --good: #059669;
      --warn: #d97706;
      --danger: #dc2626;
      --shadow: 0 20px 44px rgba(15, 23, 42, 0.08);
      --shadow-soft: 0 12px 30px rgba(15, 23, 42, 0.05);
      --radius-xl: 30px;
      --radius-lg: 24px;
      --radius-md: 18px;
      --radius-sm: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "SF Pro Display", "Pretendard", "Apple SD Gothic Neo", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.13), transparent 30%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.09), transparent 24%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 48%, var(--bg-deep) 100%);
      min-height: 100vh;
      position: relative;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.04) 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.4), transparent 85%);
    }
    .wrap {
      max-width: 1560px;
      margin: 0 auto;
      padding: 30px 30px 40px;
    }
    .server-fallback-data {
      margin-bottom: 24px;
    }
    .server-fallback-summary {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .server-fallback-list {
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 10px;
      color: var(--ink);
      font-size: 13px;
      line-height: 1.6;
    }
    .server-fallback-list li {
      margin: 0;
    }
    .server-fallback-copy {
      margin-top: 4px;
      color: var(--muted);
      word-break: break-word;
    }
    .server-fallback-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .metric-table {
      width: 100%;
      table-layout: fixed;
    }
    .metric-table th,
    .metric-table td {
      word-break: break-word;
    }
    html.dashboard-hydrated .server-fallback-data {
      display: none;
    }
    .hero {
      position: relative;
      display: flex;
      justify-content: space-between;
      align-items: stretch;
      gap: 24px;
      margin-bottom: 24px;
      padding: 30px 32px;
      border-radius: var(--radius-xl);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.94), rgba(248,250,253,0.88)),
        radial-gradient(circle at top right, rgba(37,99,235,0.14), transparent 32%);
      border: 1px solid rgba(255,255,255,0.66);
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(18px);
    }
    .hero::after {
      content: "";
      position: absolute;
      right: -60px;
      top: -60px;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(37,99,235,0.18), transparent 68%);
      pointer-events: none;
    }
    .hero-copy {
      position: relative;
      z-index: 1;
      flex: 1 1 auto;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 14px;
      padding: 7px 12px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.05);
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .eyebrow::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: linear-gradient(135deg, #2563eb, #0ea5e9);
    }
    .hero h1 {
      margin: 0;
      font-size: 44px;
      line-height: 0.98;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 14px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
      max-width: 780px;
    }
    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    .hero-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.18);
      box-shadow: var(--shadow-soft);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .hero-chip strong {
      color: var(--ink);
      font-weight: 800;
    }
    .hero-side {
      position: relative;
      z-index: 1;
      min-width: 300px;
      max-width: 340px;
      padding: 22px;
      border-radius: 24px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,248,253,0.88));
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: var(--shadow-soft);
      display: grid;
      align-content: start;
      gap: 14px;
    }
    .hero-side-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hero-side-title {
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.2;
    }
    .hero-side-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid transparent;
      font-size: 13px;
      font-weight: 700;
    }
    .status-pill::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
    }
    .tone-neutral { color: #475569; background: rgba(148,163,184,0.10); border-color: rgba(148,163,184,0.18); }
    .tone-good { color: var(--good); background: rgba(5,150,105,0.10); border-color: rgba(5,150,105,0.18); }
    .tone-warn { color: var(--warn); background: rgba(217,119,6,0.10); border-color: rgba(217,119,6,0.18); }
    .tone-accent { color: var(--accent); background: rgba(37,99,235,0.10); border-color: rgba(37,99,235,0.18); }
    .tone-danger { color: var(--danger); background: rgba(220,38,38,0.10); border-color: rgba(220,38,38,0.18); }
    .card,
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(14px);
      position: relative;
    }
    .card::before,
    .panel::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 1px;
      background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255,255,255,0));
      pointer-events: none;
    }
    .control-panel {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 20px;
      padding: 24px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,252,255,0.9)),
        radial-gradient(circle at right top, rgba(37,99,235,0.08), transparent 30%);
    }
    .control-title {
      margin: 0 0 10px;
      font-size: 26px;
      letter-spacing: -0.03em;
    }
    .control-copy {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
      margin-bottom: 16px;
    }
    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }
    .meta-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 13px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(37, 99, 235, 0.09);
      color: var(--accent);
      border: 1px solid rgba(37, 99, 235, 0.14);
    }
    .form-label {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .input-area {
      width: 100%;
      min-height: 136px;
      border: 1px solid rgba(148,163,184,0.24);
      border-radius: 18px;
      padding: 16px;
      font: inherit;
      font-size: 15px;
      line-height: 1.7;
      color: var(--ink);
      background: rgba(255,255,255,0.92);
      resize: vertical;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .input-area:focus {
      outline: none;
      border-color: rgba(37,99,235,0.36);
      box-shadow: 0 0 0 5px rgba(37,99,235,0.10);
    }
    .text-input {
      width: 100%;
      height: 54px;
      border: 1px solid rgba(148,163,184,0.24);
      border-radius: 16px;
      padding: 0 16px;
      font: inherit;
      font-size: 15px;
      color: var(--ink);
      background: rgba(255,255,255,0.92);
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
      margin-bottom: 14px;
    }
    .text-input:focus {
      outline: none;
      border-color: rgba(37,99,235,0.36);
      box-shadow: 0 0 0 5px rgba(37,99,235,0.10);
    }
    .control-actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      margin-top: 14px;
    }
    .primary-button {
      border: none;
      border-radius: 16px;
      padding: 14px 18px;
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      color: white;
      font-size: 14px;
      font-weight: 800;
      letter-spacing: -0.01em;
      cursor: pointer;
      box-shadow: 0 14px 28px rgba(37, 99, 235, 0.24);
      transition: transform 0.16s ease, box-shadow 0.16s ease, opacity 0.16s ease;
    }
    .primary-button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 18px 34px rgba(37, 99, 235, 0.28);
    }
    .primary-button:disabled {
      cursor: not-allowed;
      opacity: 0.65;
      box-shadow: none;
    }
    .helper {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
      max-width: 620px;
    }
    .launch-box {
      display: grid;
      gap: 12px;
      align-content: start;
      padding: 18px;
      border-radius: 22px;
      background:
        linear-gradient(180deg, rgba(247,250,254,0.96), rgba(242,247,252,0.88));
      border: 1px solid rgba(148,163,184,0.14);
    }
    .launch-item {
      padding: 13px 15px;
      border-radius: 16px;
      background: rgba(255,255,255,0.86);
      border: 1px solid rgba(148,163,184,0.14);
    }
    .launch-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .launch-value {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.6;
      word-break: break-word;
    }
    .launch-value-compact {
      font-size: 13px;
      font-weight: 650;
      color: var(--text);
    }
    .launch-message {
      min-height: 52px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(255,255,255,0.84);
      font-size: 14px;
      line-height: 1.6;
      color: var(--muted);
      white-space: pre-line;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }
    .card {
      padding: 20px;
      min-height: 146px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.92));
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .value {
      font-size: 30px;
      font-weight: 800;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
      font-variant-numeric: tabular-nums;
    }
    .subvalue {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
      white-space: pre-line;
      word-break: break-word;
    }
    .main-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(360px, 0.85fr);
      gap: 20px;
      align-items: start;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 20px 24px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(251,253,255,0.74));
    }
    .panel-title {
      margin: 0;
      font-size: 21px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .panel-copy {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    .panel-body {
      padding: 22px 24px 24px;
    }
    .chart-wrap {
      padding: 16px;
      border-radius: 22px;
      background:
        linear-gradient(180deg, rgba(248,250,253,0.96), rgba(244,248,252,0.9));
      border: 1px solid rgba(148,163,184,0.14);
      margin-bottom: 16px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.85);
    }
    .chart {
      width: 100%;
      height: 260px;
      display: block;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .legend span::before {
      content: "";
      width: 12px;
      height: 3px;
      border-radius: 999px;
      background: currentColor;
    }
    .legend .blue { color: #2563eb; }
    .legend .green { color: #059669; }
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .mini-card {
      padding: 16px 18px;
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(248,250,253,0.94), rgba(244,248,252,0.88));
      border: 1px solid rgba(148,163,184,0.14);
    }
    .mini-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .mini-value {
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.03em;
      margin-bottom: 6px;
    }
    .mini-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      word-break: break-word;
    }
    .table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    .table tbody tr {
      transition: background 0.16s ease;
    }
    .table tbody tr:hover {
      background: rgba(37, 99, 235, 0.04);
    }
    .table th,
    .table td {
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid rgba(148,163,184,0.14);
      vertical-align: top;
    }
    .table th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .table-source {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      margin-top: 3px;
      text-transform: none;
    }
    .empty {
      padding: 20px;
      border-radius: 18px;
      background: rgba(255,255,255,0.68);
      border: 1px dashed rgba(148,163,184,0.30);
      color: var(--muted);
      text-align: center;
    }
    .mono {
      font-family: "SF Mono", "JetBrains Mono", monospace;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }
    .progress-track {
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: rgba(148,163,184,0.16);
      overflow: hidden;
      margin-top: 12px;
    }
    .progress-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #2563eb 0%, #0ea5e9 42%, #059669 100%);
      width: 0%;
      transition: width 0.25s ease;
      box-shadow: 0 0 18px rgba(37, 99, 235, 0.18);
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .mini-pill {
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(15,23,42,0.05);
      color: var(--muted);
      border: 1px solid rgba(148,163,184,0.16);
    }
    .logs-section {
      margin-top: 20px;
    }
    .scroll-panel {
      max-height: 360px;
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(148,163,184,0.12);
      background: var(--panel-soft);
    }
    .log-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }
    .log-card {
      border-radius: 18px;
      border: 1px solid rgba(148,163,184,0.12);
      background: linear-gradient(180deg, rgba(250,252,255,0.94), rgba(244,247,252,0.9));
      overflow: hidden;
      box-shadow: var(--shadow-soft);
    }
    .log-card-head {
      padding: 16px 18px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
      background: rgba(255,255,255,0.76);
    }
    .log-card-title {
      margin: 0;
      font-size: 15px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }
    .log-card-copy {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .log-pre {
      margin: 0;
      padding: 16px 18px;
      min-height: 240px;
      max-height: 340px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SF Mono", "JetBrains Mono", monospace;
      font-size: 12px;
      line-height: 1.65;
      color: #dbe7ff;
      background:
        radial-gradient(circle at top right, rgba(59,130,246,0.12), transparent 34%),
        linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin: 0 0 14px;
    }
    .section-title h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }
    .section-title p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .section-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.8);
      border: 1px solid rgba(148,163,184,0.16);
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }
    @media (max-width: 1200px) {
      .hero,
      .control-panel,
      .main-grid {
        grid-template-columns: 1fr;
      }
      .server-fallback-grid {
        grid-template-columns: 1fr;
      }
      .server-fallback-summary {
        grid-template-columns: 1fr;
      }
      .grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .log-grid {
        grid-template-columns: 1fr;
      }
      .hero {
        flex-direction: column;
      }
      .hero-side {
        max-width: none;
      }
    }
    @media (max-width: 720px) {
      .wrap { padding: 18px; }
      .hero { padding: 24px 20px; }
      .hero h1 { font-size: 34px; }
      .grid,
      .two-col {
        grid-template-columns: 1fr;
      }
      .control-actions {
        flex-direction: column;
        align-items: stretch;
      }
      .primary-button {
        width: 100%;
      }
    }

    /* Readability + layout overrides */
    :root {
      --bg: #f2f6fc;
      --bg-deep: #e7eef8;
      --panel: rgba(255, 255, 255, 0.98);
      --panel-strong: rgba(255, 255, 255, 1);
      --panel-soft: rgba(248, 250, 253, 0.98);
      --ink: #0f172a;
      --muted: #435267;
      --line: rgba(148, 163, 184, 0.18);
      --line-strong: rgba(148, 163, 184, 0.26);
      --accent: #2563eb;
      --accent-strong: #1d4ed8;
      --accent-soft: rgba(37, 99, 235, 0.12);
      --good: #059669;
      --warn: #d97706;
      --danger: #dc2626;
      --shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
      --shadow-soft: 0 10px 24px rgba(15, 23, 42, 0.05);
      --radius-xl: 24px;
      --radius-lg: 20px;
      --radius-md: 16px;
      --radius-sm: 12px;
    }
    body {
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.08), transparent 22%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 46%, var(--bg-deep) 100%);
      color: var(--ink);
    }
    body::before {
      background-image:
        linear-gradient(rgba(148, 163, 184, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148, 163, 184, 0.03) 1px, transparent 1px);
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.35), transparent 90%);
    }
    .wrap {
      max-width: 1660px;
      padding: 18px 18px 30px;
    }
    .hero {
      margin-bottom: 14px;
      padding: 20px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,0.985), rgba(247,250,253,0.965));
      border: 1px solid rgba(203, 213, 225, 0.78);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .hero::after {
      right: -34px;
      top: -36px;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(37,99,235,0.11), transparent 70%);
    }
    .hero-copy {
      display: grid;
      gap: 10px;
      min-width: 0;
    }
    .eyebrow {
      background: rgba(37, 99, 235, 0.08);
      color: #1d4ed8;
      border: 1px solid rgba(37, 99, 235, 0.12);
      width: fit-content;
      margin-bottom: 0;
    }
    .hero h1 {
      color: var(--ink);
      font-size: 34px;
      line-height: 1;
      margin: 0;
    }
    .hero p,
    .hero-side-copy,
    .section-title p,
    .panel-copy,
    .control-copy {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }
    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 0;
    }
    .hero-chip {
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.92);
      border-color: rgba(148, 163, 184, 0.16);
      color: var(--muted);
      box-shadow: none;
      font-size: 12px;
    }
    .hero-chip strong {
      color: var(--ink);
    }
    .hero-side {
      min-width: 300px;
      max-width: 350px;
      padding: 16px 18px;
      display: grid;
      gap: 10px;
      background: linear-gradient(180deg, rgba(249,251,255,0.98), rgba(244,248,252,0.94));
      border-color: rgba(203, 213, 225, 0.72);
      box-shadow: none;
    }
    .hero-side-label {
      color: #2563eb;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hero-side-title {
      color: var(--ink);
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .dashboard-shell {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      align-items: start;
    }
    .sidebar-stack {
      display: grid;
      gap: 14px;
      position: static;
    }
    .main-stack {
      min-width: 0;
      display: grid;
      gap: 16px;
    }
    .sidebar-stack .control-panel {
      grid-template-columns: 1fr;
      padding: 16px;
      margin-bottom: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.99), rgba(246,249,253,0.96));
      border-color: rgba(203, 213, 225, 0.72);
      box-shadow: var(--shadow);
    }
    .control-panel > div:first-child {
      display: grid;
      gap: 12px;
    }
    .sidebar-stack .control-title,
    .sidebar-stack .launch-value,
    .sidebar-stack .helper strong,
    .sidebar-stack .launch-message {
      color: var(--ink);
    }
    .sidebar-stack .control-copy,
    .sidebar-stack .helper,
    .sidebar-stack .launch-label,
    .sidebar-stack .form-label,
    .sidebar-stack .meta-chip {
      color: var(--muted);
    }
    .sidebar-stack .meta-chip,
    .sidebar-stack .launch-item,
    .sidebar-stack .launch-box {
      background: rgba(248, 250, 253, 0.98);
      border-color: rgba(203, 213, 225, 0.72);
      box-shadow: none;
    }
    .sidebar-stack .text-input,
    .sidebar-stack .input-area {
      background: #ffffff;
      color: var(--ink);
      border-color: rgba(148, 163, 184, 0.24);
    }
    .sidebar-stack .text-input::placeholder,
    .sidebar-stack .input-area::placeholder {
      color: #94a3b8;
    }
    .sidebar-stack .text-input:focus,
    .sidebar-stack .input-area:focus {
      border-color: rgba(37, 99, 235, 0.34);
      box-shadow: 0 0 0 5px rgba(37, 99, 235, 0.10);
    }
    .sidebar-stack .launch-message {
      background: rgba(255,255,255,0.92);
    }
    .meta-row {
      gap: 8px;
      margin-top: 0;
    }
    .meta-chip {
      padding: 8px 11px;
      font-size: 12px;
      font-weight: 700;
    }
    .queue-editor {
      display: grid;
      gap: 10px;
    }
    .form-label {
      margin-bottom: 0;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #5b6d82;
    }
    .viewer-mode .text-input,
    .viewer-mode .input-area {
      background: rgba(241, 245, 249, 0.9);
      color: #7c8aa0;
      cursor: not-allowed;
    }
    html.viewer-mode-page #controlPanel .meta-row,
    .viewer-mode .meta-row {
      display: none;
    }
    html.viewer-mode-page .hero-meta,
    .viewer-mode-page .hero-meta {
      display: none;
    }
    html.viewer-mode-page #controlPanel .queue-editor,
    html.viewer-mode-page #controlPanel .control-actions,
    html.viewer-mode-page #controlPanel .queue-manager,
    .viewer-mode .queue-editor,
    .viewer-mode .control-actions,
    .viewer-mode .queue-manager {
      display: none !important;
    }
    html.viewer-mode-page #controlPanel,
    .viewer-mode {
      padding: 16px;
    }
    html.viewer-mode-page .dashboard-shell {
      grid-template-columns: 1fr;
    }
    html.viewer-mode-page .hero {
      padding: 18px 20px;
    }
    html.viewer-mode-page .hero h1 {
      font-size: 30px;
    }
    html.viewer-mode-page .hero-side {
      min-width: 260px;
      max-width: 300px;
    }
    html.viewer-mode-page .main-stack {
      gap: 14px;
    }
    html.viewer-mode-page .section-title p {
      display: none;
    }
    html.viewer-mode-page .section-pill {
      padding: 7px 11px;
      font-size: 11px;
    }
    html.viewer-mode-page #controlPanel .launch-box,
    .viewer-mode .launch-box {
      padding: 0;
      background: transparent;
      border: none;
      gap: 8px;
    }
    html.viewer-mode-page #controlPanel .launch-item,
    .viewer-mode .launch-item {
      background: rgba(248, 250, 253, 0.98);
    }
    html.viewer-mode-page #controlPanel .launch-item {
      min-height: 76px;
    }
    html.viewer-mode-page #controlPanel .primary-button,
    .viewer-mode .primary-button {
      opacity: 0.55;
      box-shadow: none !important;
      cursor: not-allowed;
      pointer-events: none;
      filter: grayscale(0.08);
    }
    .helper {
      font-size: 12.5px;
      line-height: 1.65;
      color: var(--muted);
    }
    .launch-box {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .launch-item {
      padding: 11px 13px;
      min-height: 88px;
    }
    .launch-message {
      grid-column: 1 / -1;
      margin-top: 2px;
    }
    .launch-value {
      font-size: 14px;
      line-height: 1.45;
    }
    .queue-manager {
      display: grid;
      gap: 10px;
      margin-top: 4px;
      padding-top: 12px;
      border-top: 1px solid rgba(203, 213, 225, 0.72);
    }
    .queue-manager-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .queue-manager-title {
      margin: 0;
      font-size: 14px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: var(--ink);
    }
    .queue-manager-copy {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .queued-job-list {
      display: grid;
      gap: 8px;
    }
    .queued-job-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(248, 250, 253, 0.98);
      border: 1px solid rgba(203, 213, 225, 0.72);
    }
    .queued-job-main {
      min-width: 0;
      display: grid;
      gap: 4px;
    }
    .queued-job-key {
      color: var(--ink);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: -0.02em;
      word-break: break-word;
    }
    .queued-job-meta {
      color: var(--muted);
      font-size: 11.5px;
      line-height: 1.5;
      word-break: break-word;
    }
    .queued-remove-button {
      border: 1px solid rgba(220, 38, 38, 0.14);
      background: rgba(220, 38, 38, 0.06);
      color: #b91c1c;
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
      transition: background 0.16s ease, transform 0.16s ease;
      flex-shrink: 0;
    }
    .queued-remove-button:hover {
      background: rgba(220, 38, 38, 0.1);
      transform: translateY(-1px);
    }
    .queued-job-empty {
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(248, 250, 253, 0.98);
      border: 1px dashed rgba(203, 213, 225, 0.72);
      color: var(--muted);
      font-size: 12.5px;
      text-align: center;
    }
    .control-actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      align-items: stretch;
      margin-top: 2px;
    }
    .control-actions .helper {
      grid-column: 1 / -1;
      margin-top: 2px;
    }
    .primary-button {
      border-radius: 12px;
      padding: 12px 15px;
      box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18);
      justify-content: center;
      min-height: 46px;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: -0.01em;
    }
    .primary-button:hover:not(:disabled) {
      transform: translateY(-1px);
    }
    .section-title h2 {
      color: var(--ink);
      font-size: 18px;
    }
    .section-title p {
      margin-top: 2px;
      font-size: 12.5px;
      color: #64748b;
    }
    .section-pill {
      background: rgba(255,255,255,0.86);
      color: var(--muted);
      border-color: rgba(148, 163, 184, 0.16);
    }
    .grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      grid-auto-flow: dense;
    }
    .card,
    .panel,
    .log-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.99), rgba(246,249,253,0.96));
      border-color: rgba(203, 213, 225, 0.74);
      box-shadow: var(--shadow-soft);
    }
    .card {
      padding: 16px;
      min-height: 118px;
    }
    .card-featured {
      grid-column: span 2;
      background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 34%),
        linear-gradient(180deg, rgba(255,255,255,0.99), rgba(246,249,253,0.96));
    }
    .card-progress {
      background:
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.99), rgba(246,249,253,0.96));
    }
    .card-queue {
      grid-column: span 2;
      background:
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 36%),
        linear-gradient(180deg, rgba(255,255,255,0.99), rgba(246,249,253,0.96));
    }
    .label,
    .mini-title,
    .table th {
      color: #5b6d82;
    }
    .value {
      color: var(--ink);
      font-size: 30px;
      line-height: 1.08;
    }
    .subvalue,
    .mini-copy,
    .table td,
    .empty {
      color: var(--muted);
    }
    .subvalue {
      font-size: 13.5px;
      line-height: 1.55;
    }
    .main-grid {
      grid-template-columns: minmax(0, 1.52fr) minmax(340px, 0.92fr);
      gap: 14px;
    }
    .panel-head {
      background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(248,250,253,0.92));
      border-bottom-color: rgba(148,163,184,0.12);
    }
    .panel-title {
      font-size: 20px;
      color: var(--ink);
    }
    .panel-copy {
      font-size: 12.5px;
      line-height: 1.55;
    }
    .chart-wrap,
    .mini-card,
    .scroll-panel,
    .log-card {
      background: linear-gradient(180deg, rgba(249,251,254,0.98), rgba(244,248,252,0.95));
    }
    .chart-wrap {
      border-color: rgba(148,163,184,0.12);
      position: relative;
      overflow: hidden;
    }
    .chart-shell {
      position: relative;
    }
    .chart {
      position: relative;
      z-index: 1;
    }
    .chart-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }
    .chart-row .chart-wrap {
      margin-top: 0;
    }
    .chart-caption {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
      color: #5b6d82;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    .chart-caption strong {
      color: var(--ink);
      font-size: 12.5px;
      text-transform: none;
      letter-spacing: 0;
    }
    .chart-tooltip {
      position: absolute;
      z-index: 3;
      min-width: 150px;
      max-width: 240px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(15, 23, 42, 0.94);
      color: #e2e8f0;
      box-shadow: 0 14px 28px rgba(15, 23, 42, 0.24);
      border: 1px solid rgba(148, 163, 184, 0.18);
      backdrop-filter: blur(10px);
      pointer-events: none;
      opacity: 0;
      transform: none;
      transition: opacity 0.14s ease;
    }
    .chart-tooltip.visible {
      opacity: 1;
    }
    .chart-tooltip-title {
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #93c5fd;
      margin-bottom: 6px;
    }
    .chart-tooltip-line {
      font-size: 12px;
      line-height: 1.55;
      color: #e2e8f0;
      white-space: nowrap;
    }
    .chart-tooltip-line strong {
      color: #f8fafc;
      font-weight: 800;
      margin-right: 6px;
    }
    .chart-point {
      cursor: pointer;
      transition: opacity 0.12s ease;
    }
    .chart-point-core {
      pointer-events: none;
    }
    .chart-point-hit {
      fill: transparent;
      pointer-events: all;
    }
    .chart-point.is-active .chart-point-core {
      filter: drop-shadow(0 0 8px rgba(37, 99, 235, 0.22));
    }
    .chart-point.is-active .chart-point-core:last-child {
      opacity: 1;
    }
    .chart-hover-line {
      stroke: rgba(148,163,184,0.28);
      stroke-width: 1;
      stroke-dasharray: 4 4;
      opacity: 0;
      pointer-events: none;
    }
    .mini-card {
      border-radius: 16px;
    }
    .legend {
      color: #5b6d82;
      font-weight: 700;
    }
    .log-grid {
      grid-template-columns: 1fr;
      gap: 16px;
    }
    .log-card-head {
      background: rgba(255,255,255,0.92);
    }
    .log-card-title {
      color: var(--ink);
    }
    .log-card-copy {
      color: var(--muted);
    }
    .log-pre {
      min-height: 280px;
      color: #edf4ff;
      font-size: 12.5px;
      line-height: 1.72;
      background:
        radial-gradient(circle at top right, rgba(59,130,246,0.12), transparent 30%),
        linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }
    .metric-stack {
      display: grid;
      gap: 12px;
    }
    .diagnostic-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .insight-card {
      padding: 14px 16px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(249,251,254,0.98), rgba(244,248,252,0.95));
      border: 1px solid rgba(148,163,184,0.12);
    }
    .insight-label {
      color: #5b6d82;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .insight-value {
      color: var(--ink);
      font-size: 23px;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.05;
      margin-bottom: 5px;
    }
    .insight-copy {
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.55;
      word-break: break-word;
    }
    .heatmap-wrap {
      overflow: auto;
      border-radius: 16px;
      border: 1px solid rgba(148,163,184,0.12);
      background: rgba(249,251,254,0.98);
    }
    .heatmap-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 520px;
      font-size: 12px;
    }
    .heatmap-table th,
    .heatmap-table td {
      padding: 10px 8px;
      border-bottom: 1px solid rgba(148,163,184,0.10);
      border-right: 1px solid rgba(148,163,184,0.08);
      text-align: center;
      font-variant-numeric: tabular-nums;
    }
    .heatmap-table th:first-child,
    .heatmap-table td:first-child {
      text-align: left;
      position: sticky;
      left: 0;
      background: rgba(248,250,253,0.98);
      z-index: 1;
    }
    .heatmap-table thead th {
      position: sticky;
      top: 0;
      background: rgba(248,250,253,0.98);
      z-index: 2;
      color: #5b6d82;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .heatmap-table thead th:first-child {
      z-index: 3;
    }
    .heat-cell {
      color: #0f172a;
      font-weight: 700;
    }
    .class-bars {
      display: grid;
      gap: 12px;
      margin-bottom: 16px;
    }
    .class-bar-row {
      display: grid;
      grid-template-columns: minmax(92px, 0.22fr) minmax(0, 1fr);
      gap: 12px;
      align-items: center;
    }
    .class-bar-label {
      color: var(--ink);
      font-size: 12.5px;
      font-weight: 800;
      word-break: break-word;
    }
    .class-bar-stack {
      display: grid;
      gap: 5px;
    }
    .class-bar-track {
      position: relative;
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(226, 232, 240, 0.92);
    }
    .class-bar-fill {
      height: 100%;
      border-radius: inherit;
      transition: width 0.18s ease;
    }
    .class-bar-values {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: #64748b;
      font-size: 11.5px;
      font-variant-numeric: tabular-nums;
    }
    .class-bar-values strong {
      color: var(--ink);
      font-weight: 800;
    }
    @media (max-width: 1480px) {
      .grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
      .card-featured,
      .card-queue {
        grid-column: span 2;
      }
    }
    @media (max-width: 1260px) {
      .sidebar-stack .control-panel {
        grid-template-columns: 1fr;
      }
      .control-actions {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
      .grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .diagnostic-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 860px) {
      .hero,
      .sidebar-stack .control-panel,
      .main-grid,
      .log-grid,
      .grid,
      .two-col {
        grid-template-columns: 1fr;
      }
      .hero {
        flex-direction: column;
      }
      .hero-side {
        max-width: none;
        width: 100%;
      }
      .card-featured,
      .card-queue {
        grid-column: span 1;
      }
      .control-actions,
      .launch-box {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .diagnostic-grid {
        grid-template-columns: 1fr;
      }
      .chart-row {
        grid-template-columns: 1fr;
      }
      .value {
        font-size: 27px;
      }
    }
    @media (max-width: 640px) {
      .control-actions,
      .launch-box {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div id="bootErrorBanner" style="display:none;margin-bottom:18px;padding:14px 16px;border-radius:16px;background:rgba(220,38,38,0.08);border:1px solid rgba(220,38,38,0.18);color:#991b1b;font-size:13px;line-height:1.7;"></div>
    __SERVER_SNAPSHOT__
    <script id="initialOverviewData" type="application/json">__INITIAL_OVERVIEW_JSON__</script>
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Training Queue</div>
        <h1>detectWarning Training Dashboard</h1>
        <p>현재 학습 상태, 큐 진행률, 최신 성능을 한 화면에서 확인합니다.</p>
        <div class="hero-meta">
          <div class="hero-chip"><strong>순차 큐</strong> filekey 자동 처리</div>
          <div class="hero-chip"><strong>누적 학습</strong> prepared 데이터 유지</div>
          <div class="hero-chip"><strong>원격 확인</strong> 실시간 상태 공유</div>
        </div>
      </div>
      <aside class="hero-side">
        <div class="hero-side-label">Pipeline</div>
        <div class="hero-side-title">현재 파이프라인 상태</div>
        <div class="hero-side-copy">현재 실행 상태와 주요 지표를 바로 확인합니다.</div>
        <div id="pipelineState" class="status-pill tone-neutral">상태 확인 중</div>
      </aside>
    </section>

    <div class="dashboard-shell">
    <aside class="sidebar-stack">
    <section id="controlPanel" class="card control-panel">
      <div>
        <h2 id="controlTitle" class="control-title">작업 제어</h2>
        <div id="controlCopy" class="control-copy">큐를 구성하고 현재 실행 상태를 확인합니다.</div>
        <div class="meta-row">
          <div class="meta-chip">datasetkey <span id="datasetKeyChip">-</span></div>
          <div class="meta-chip">workspace <span id="workspaceChip">-</span></div>
        </div>
        <div class="queue-editor">
          <label class="form-label" for="datasetKeyInput">AIHub datasetkey</label>
          <input id="datasetKeyInput" class="text-input" type="text" placeholder="예: 12345" />
          <label class="form-label" for="apiKeyInput">AIHub API 키</label>
          <input id="apiKeyInput" class="text-input" type="password" placeholder="AIHub API 키를 입력하세요" />
          <label class="form-label" for="filekeysInput">분할 ZIP filekey 입력</label>
          <textarea id="filekeysInput" class="input-area" placeholder="예:&#10;123456&#10;123457&#10;49841 ~ 49843"></textarea>
        </div>
        <div class="control-actions">
          <button id="startButton" class="primary-button" type="button">시작 / 추가</button>
          <button id="reextractButton" class="primary-button" type="button" style="background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%); box-shadow: 0 14px 28px rgba(13, 148, 136, 0.20);">선택 filekey 재추출</button>
          <button id="stopButton" class="primary-button" type="button" style="background: linear-gradient(135deg, #d97706 0%, #b45309 100%); box-shadow: 0 14px 28px rgba(217, 119, 6, 0.20);">현재 작업 후 중지</button>
          <button id="forceStopButton" class="primary-button" type="button" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); box-shadow: 0 14px 28px rgba(239, 68, 68, 0.22);">지금 중단</button>
          <button id="resetButton" class="primary-button" type="button" style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); box-shadow: 0 14px 28px rgba(220, 38, 38, 0.22);">처음부터 다시 시작</button>
          <div class="helper">datasetkey는 데이터셋 키, filekey는 분할 ZIP key입니다. 범위 입력은 `49841 ~ 49843`처럼 쓸 수 있습니다.</div>
        </div>
      </div>
      <div class="launch-box">
        <div class="launch-item">
          <div class="launch-label">런처 상태</div>
          <div class="launch-value" id="launcherState">대기 중</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">현재 실행 filekey</div>
          <div class="launch-value" id="currentFilekey">-</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">Target filekeys</div>
          <div class="launch-value launch-value-compact" id="currentFilekeyDetails">-</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">현재 실행 datasetkey</div>
          <div class="launch-value" id="currentDatasetkey">-</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">대기 중 filekey</div>
          <div class="launch-value" id="pendingFilekeys">-</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">최근 완료 작업</div>
          <div class="launch-value" id="completedJobs">-</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">다음 큐 자동 시작</div>
          <div class="launch-value" id="autoStartState">켜짐</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">실행 로그</div>
          <div class="launch-value mono" id="launcherLogPath">-</div>
        </div>
        <div id="launchMessage" class="launch-message">최근 실행 메시지가 여기에 표시됩니다.</div>
      </div>
      <div class="queue-manager">
        <div class="queue-manager-head">
          <h3 class="queue-manager-title">대기 중 작업</h3>
          <div class="queue-manager-copy" id="queuedJobCount">0개</div>
        </div>
        <div id="queuedJobList" class="queued-job-list">
          <div class="queued-job-empty">대기 중인 filekey가 없습니다.</div>
        </div>
      </div>
    </section>
    </aside>
    <main class="main-stack">

    <div class="section-title">
      <div>
        <h2>핵심 지표</h2>
        <p>현재 실행 상태</p>
      </div>
      <div class="section-pill">Overview</div>
    </div>
    <section class="grid">
      <article class="card card-featured">
        <div class="label">현재 단계</div>
        <div class="value" id="currentStage">-</div>
        <div class="subvalue" id="currentMessage">-</div>
      </article>
      <article class="card card-progress">
        <div class="label">현재 filekey 진행률</div>
        <div class="value" id="currentJobProgressText">0%</div>
        <div class="subvalue" id="currentJobProgressMeta">대기 중</div>
        <div class="progress-track"><div id="currentJobProgressFill" class="progress-fill"></div></div>
      </article>
      <article class="card">
        <div class="label">예상 남은 시간</div>
        <div class="value" id="etaText">-</div>
        <div class="subvalue" id="etaMeta">진행률이 쌓이면 계산합니다.</div>
      </article>
      <article class="card">
        <div class="label">학습 진행</div>
        <div class="value" id="epochProgress">0 / 0</div>
        <div class="subvalue" id="bestF1">best macro F1: -</div>
      </article>
      <article class="card">
        <div class="label">원본 / 준비 완료</div>
        <div class="value" id="datasetTotals">0 / 0</div>
        <div class="subvalue" id="datasetSummary">raw / prepared</div>
      </article>
      <article class="card">
        <div class="label">모델 산출물</div>
        <div class="value" id="artifactState">-</div>
        <div class="subvalue" id="workspaceDir">-</div>
      </article>
      <article class="card">
        <div class="label">GPU 사용률</div>
        <div class="value" id="gpuUsageText">-</div>
        <div class="subvalue" id="gpuUsageMeta">GPU 상태를 불러오는 중입니다.</div>
      </article>
      <article class="card">
        <div class="label">학습 VRAM</div>
        <div class="value" id="gpuVramText">-</div>
        <div class="subvalue" id="gpuVramMeta">VRAM 상태를 불러오는 중입니다.</div>
      </article>
      <article class="card card-queue">
        <div class="label">큐 진행률</div>
        <div class="value" id="queueProgressText">0 / 0</div>
        <div class="subvalue" id="queueProgressMeta">완료 0 / 실패 0 / 대기 0</div>
        <div class="progress-track"><div id="queueProgressFill" class="progress-fill"></div></div>
      </article>
    </section>

    <div class="section-title">
      <div>
        <h2>성능 분석</h2>
        <p>성능, 데이터, 진행 추이</p>
      </div>
      <div class="section-pill">Analytics</div>
    </div>
    <section class="main-grid">
      <article class="panel analytics-primary">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">학습 성능 그래프</h2>
            <div class="panel-copy">Validation Accuracy, Macro F1, Loss</div>
          </div>
        </div>
        <div class="panel-body">
          <div class="metric-stack">
            <div class="chart-wrap">
              <div class="chart-shell">
                <svg id="trainingChart" class="chart" viewBox="0 0 800 260" preserveAspectRatio="none"></svg>
                <div id="trainingChartTooltip" class="chart-tooltip"></div>
              </div>
              <div class="legend">
                <span class="blue">Validation Accuracy</span>
                <span class="green">Validation Macro F1</span>
              </div>
            </div>
            <div class="diagnostic-grid">
              <div class="insight-card">
                <div class="insight-label">최종 Validation</div>
                <div class="insight-value" id="finalValMetrics">-</div>
                <div class="insight-copy" id="finalValMetricsCopy">최종 accuracy / macro F1</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Best Epoch</div>
                <div class="insight-value" id="bestEpochValue">-</div>
                <div class="insight-copy" id="bestEpochCopy">가장 높은 macro F1을 기록한 epoch</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Loss Gap</div>
                <div class="insight-value" id="lossGapValue">-</div>
                <div class="insight-copy" id="lossGapCopy">최신 val loss - train loss</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Validation Samples</div>
                <div class="insight-value" id="valSampleTotal">-</div>
                <div class="insight-copy" id="valSampleCopy">최종 validation 샘플 수</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Class Coverage</div>
                <div class="insight-value" id="classCoverageValue">-</div>
                <div class="insight-copy" id="classCoverageCopy">validation에 등장한 클래스 수</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Dominant Class</div>
                <div class="insight-value" id="dominantClassValue">-</div>
                <div class="insight-copy" id="dominantClassCopy">validation에서 가장 많은 클래스</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Early Stop</div>
                <div class="insight-value" id="earlyStopValue">-</div>
                <div class="insight-copy" id="earlyStopCopy">조기 종료 정보</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Train Imbalance</div>
                <div class="insight-value" id="trainImbalanceValue">-</div>
                <div class="insight-copy" id="trainImbalanceCopy">학습 클래스 분포</div>
              </div>
              <div class="insight-card">
                <div class="insight-label">Stage Durations</div>
                <div class="insight-value" id="stageTimingValue">-</div>
                <div class="insight-copy" id="stageTimingCopy">download / prepare / train</div>
              </div>
            </div>
          </div>
          <div class="chart-wrap" style="margin-top:18px;">
            <div class="chart-shell">
              <svg id="lossChart" class="chart" viewBox="0 0 800 260" preserveAspectRatio="none"></svg>
              <div id="lossChartTooltip" class="chart-tooltip"></div>
            </div>
            <div class="legend">
              <span class="blue">Train Loss</span>
              <span class="green">Validation Loss</span>
            </div>
          </div>
          <div class="chart-row">
            <div class="chart-wrap">
              <div class="chart-caption"><span>Learning Rate</span><strong id="learningRateChartValue">-</strong></div>
              <div class="chart-shell">
                <svg id="learningRateChart" class="chart" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
                <div id="learningRateChartTooltip" class="chart-tooltip"></div>
              </div>
            </div>
            <div class="chart-wrap">
              <div class="chart-caption"><span>Loss Gap</span><strong id="lossGapChartValue">-</strong></div>
              <div class="chart-shell">
                <svg id="lossGapChart" class="chart" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
                <div id="lossGapChartTooltip" class="chart-tooltip"></div>
              </div>
            </div>
          </div>
          <div class="two-col">
            <div class="mini-card">
              <div class="mini-title">최근 Epoch</div>
              <div class="mini-value" id="latestEpoch">-</div>
              <div class="mini-copy" id="latestMetrics">-</div>
            </div>
            <div class="mini-card">
              <div class="mini-title">업데이트 시각</div>
              <div class="mini-value" id="updatedAt">-</div>
              <div class="mini-copy" id="configPath">-</div>
            </div>
            <div class="mini-card">
              <div class="mini-title">최근 Loss / LR</div>
              <div class="mini-value" id="latestLoss">-</div>
              <div class="mini-copy" id="latestLearningRate">-</div>
            </div>
            <div class="mini-card">
              <div class="mini-title">Resume / 샘플 수</div>
              <div class="mini-value" id="resumeState">-</div>
              <div class="mini-copy" id="sampleCounts">-</div>
            </div>
          </div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">데이터셋 요약</h2>
            <div class="panel-copy">pose, guideline/RGB, active 기준을 분리한 split과 클래스 분포</div>
          </div>
        </div>
        <div class="panel-body">
          <div id="perClassMetricChart" class="class-bars"></div>
          <table class="table">
            <thead>
              <tr>
                <th>기준</th>
                <th>구간</th>
                <th>총 샘플</th>
                <th>클래스 분포</th>
              </tr>
            </thead>
            <tbody id="datasetTable"></tbody>
          </table>
          <div id="datasetEmpty" class="empty" style="display:none; margin-top:14px;">아직 수집되거나 준비된 데이터가 없습니다.</div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">현재 filekey 세부 진행</h2>
            <div class="panel-copy">현재 작업 상태</div>
          </div>
        </div>
        <div class="panel-body">
          <div class="mini-card">
            <div class="mini-title">현재 처리 정보</div>
            <div class="mini-value" id="jobStageDetail">-</div>
            <div class="mini-copy" id="jobCurrentVideo">-</div>
          </div>
          <div class="two-col" style="margin-top:16px;">
            <div class="mini-card">
              <div class="mini-title">현재 filekey 원본 / 준비</div>
              <div class="mini-value" id="currentDatasetTotals">0 / 0</div>
              <div class="mini-copy" id="currentDatasetSummary">raw / prepared</div>
            </div>
            <div class="mini-card">
              <div class="mini-title">누적 학습 상태</div>
              <div class="mini-value" id="continualStateText">-</div>
              <div class="mini-copy" id="continualStateMeta">-</div>
            </div>
          </div>
          <div class="two-col" style="margin-top:12px;">
            <div class="mini-card">
              <div class="mini-title">손상 영상</div>
              <div class="mini-value" id="brokenVideoCount">0</div>
              <div class="mini-copy" id="brokenVideoSummary">읽기 실패 없음</div>
            </div>
            <div class="mini-card">
              <div class="mini-title">건너뜀</div>
              <div class="mini-value" id="skippedVideoCount">0</div>
              <div class="mini-copy" id="skippedVideoSummary">조건 미달 없음</div>
            </div>
          </div>
          <div class="mini-card" style="margin-top:12px;">
            <div class="mini-title">GPU 상태</div>
            <div class="mini-value" id="gpuDeviceText">-</div>
            <div class="mini-copy" id="gpuMemoryText">-</div>
          </div>
          <div class="mini-card" style="margin-top:12px;">
            <div class="mini-title">학습 장치</div>
            <div class="mini-value" id="trainingDeviceText">-</div>
            <div class="mini-copy" id="trainingDeviceMeta">학습 프로세스 장치 정보가 없습니다.</div>
          </div>
          <div class="mini-card" style="margin-top:12px;">
            <div class="mini-title">단계별 소요 시간</div>
            <div class="mini-value" id="currentStageTimingText">-</div>
            <div class="mini-copy" id="currentStageTimingMeta">download / prepare / train / total</div>
          </div>
          <div class="mini-card" style="margin-top:12px;">
            <div class="mini-title">데이터 분포 경고</div>
            <div class="mini-value" id="imbalanceStatusText">-</div>
            <div class="mini-copy" id="imbalanceStatusMeta">학습/검증 클래스 분포를 확인합니다.</div>
          </div>
          <div class="scroll-panel" style="margin-top:16px; max-height: 260px;">
            <table class="table">
              <thead>
                <tr>
                  <th>구분</th>
                  <th>split</th>
                  <th>파일</th>
                  <th>사유</th>
                </tr>
              </thead>
              <tbody id="issueVideoTable"></tbody>
            </table>
          </div>
          <div id="issueVideoEmpty" class="empty" style="display:none; margin-top:14px;">현재 작업에서 기록된 문제 영상이 없습니다.</div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">클래스별 검증 지표</h2>
            <div class="panel-copy">Precision / Recall / F1 / Support</div>
          </div>
        </div>
        <div class="panel-body">
          <table class="table">
            <thead>
              <tr>
                <th>클래스</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody id="perClassMetricsTable"></tbody>
          </table>
          <div id="perClassMetricsEmpty" class="empty" style="display:none; margin-top:14px;">클래스별 지표가 아직 없습니다.</div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">Confusion Matrix</h2>
            <div class="panel-copy">최종 validation confusion matrix</div>
          </div>
        </div>
        <div class="panel-body">
          <div id="confusionMatrixWrap" class="heatmap-wrap"></div>
          <div id="confusionMatrixEmpty" class="empty" style="display:none; margin-top:14px;">confusion matrix가 아직 없습니다.</div>
        </div>
      </article>
    </section>

    <div class="section-title">
      <div>
        <h2>로그 & 이력</h2>
        <p>최근 완료 작업과 현재 로그</p>
      </div>
      <div class="section-pill">Logs</div>
    </div>
    <section class="panel logs-section">
      <div class="panel-head">
        <div>
          <h2 class="panel-title">완료 데이터 로그</h2>
          <div class="panel-copy">최근 처리 결과</div>
        </div>
      </div>
      <div class="panel-body">
        <div class="pill-row">
          <div class="mini-pill" id="completedCountPill">완료 0</div>
          <div class="mini-pill" id="failedCountPill">실패 0</div>
          <div class="mini-pill" id="pendingCountPill">대기 0</div>
        </div>
        <div class="scroll-panel" style="margin-top:14px;">
          <table class="table">
            <thead>
              <tr>
                <th>filekey</th>
                <th>상태</th>
                <th>원본 / 준비</th>
                <th>시작 / 완료</th>
                <th>로그 내용</th>
              </tr>
            </thead>
            <tbody id="completedLogsTable"></tbody>
          </table>
        </div>
        <div id="completedLogsEmpty" class="empty" style="display:none; margin-top:14px;">아직 완료된 작업 로그가 없습니다.</div>
      </div>
    </section>

    <section class="log-grid">
      <article class="log-card">
        <div class="log-card-head">
          <h3 class="log-card-title">현재 작업 로그</h3>
          <div class="log-card-copy" id="currentLogMeta">현재 로그 또는 최근 로그</div>
        </div>
        <pre id="currentLogText" class="log-pre">로그를 불러오는 중입니다.</pre>
      </article>
      <article class="log-card">
        <div class="log-card-head">
          <h3 class="log-card-title">최근 오류 로그</h3>
          <div class="log-card-copy" id="errorLogMeta">최근 실패 로그</div>
        </div>
        <pre id="errorLogText" class="log-pre">오류 로그가 아직 없습니다.</pre>
      </article>
    </section>
    </main>
    </div>
  </div>
  <script>
    function toneClass(state) {
      if (state === 'completed') return 'tone-good';
      if (state === 'completed_warning') return 'tone-warn';
      if (state === 'data_ready') return 'tone-accent';
      if (state === 'running') return 'tone-accent';
      if (state === 'paused') return 'tone-warn';
      if (state === 'aborted') return 'tone-danger';
      if (state === 'error') return 'tone-danger';
      if (state === 'download' || state === 'prepare' || state === 'train') return 'tone-warn';
      return 'tone-neutral';
    }

    function formatFilekeys(filekeys) {
      if (!filekeys || !filekeys.length) {
        return '-';
      }
      return filekeys.join(', ');
    }

    function formatDatasetkeys(jobs) {
      if (!jobs || !jobs.length) {
        return '-';
      }
      const values = jobs
        .map((job) => job?.datasetkey)
        .filter((value) => value !== null && value !== undefined && value !== '');
      if (!values.length) {
        return '-';
      }
      return [...new Set(values)].join(', ');
    }

    function formatJob(job) {
      if (!job || !job.filekey) {
        return '-';
      }
      const sourceFilekey = job.source_filekey && job.source_filekey !== job.filekey
        ? ` | source ${job.source_filekey}`
        : '';
      return `${job.filekey} (${job.state || 'unknown'})${sourceFilekey}`;
    }

    function formatCurrentJobFilekeys(summary) {
      const items = Array.isArray(summary?.filekeys) ? summary.filekeys : [];
      if (!items.length) {
        return '-';
      }
      const head = items.slice(0, 8).map((item) => {
        const splits = item?.splits || {};
        const splitText = ['train', 'val', 'test']
          .map((split) => splits[split] ? `${split[0]}${splits[split]}` : null)
          .filter(Boolean)
          .join('/');
        const rgbMissing = Number(item?.rgb_missing || 0);
        const rgbText = rgbMissing > 0 ? `, rgb pending ${rgbMissing}` : '';
        return `${item.filekey}: ${item.total}${splitText ? ` (${splitText}${rgbText})` : rgbText}`;
      });
      const extra = items.length > head.length ? ` +${items.length - head.length} more` : '';
      const source = summary?.source ? `${summary.source}: ` : '';
      return `${source}${head.join(' / ')}${extra}`;
    }

    function formatCompletedJobs(jobs) {
      if (!jobs || !jobs.length) {
        return '-';
      }
      return jobs.slice(0, 3).map((job) => {
        const state =
          job.state === 'completed' ? '완료' :
          job.state === 'completed_warning' ? '경고 종료' :
          job.state === 'data_ready' ? '데이터 준비' :
          job.state === 'aborted' ? '강제 중단' :
          '실패';
        return `${job.filekey} ${state}`;
      }).join(' / ');
    }

    function formatLauncherState(state) {
      if (state === 'running') return '실행 중';
      if (state === 'queued') return '대기열 준비';
      if (state === 'paused') return '자동 시작 중지';
      if (state === 'completed') return '완료';
      if (state === 'completed_warning') return '경고 종료';
      if (state === 'data_ready') return '데이터 준비';
      if (state === 'aborted') return '강제 중단';
      if (state === 'error') return '오류';
      return state || '대기 중';
    }

    function formatGpuUsage(gpu) {
      if (!gpu || gpu.available === false) {
        return '-';
      }
      if (gpu.utilization_gpu === null || gpu.utilization_gpu === undefined) {
        return gpu.summary || '-';
      }
      return `${gpu.utilization_gpu}%`;
    }

    function formatGpuMeta(gpu) {
      if (!gpu) {
        return 'GPU 상태를 불러오는 중입니다.';
      }
      return gpu.detail || 'GPU 상태를 읽지 못했습니다.';
    }

    function formatGpuDevice(gpu) {
      if (!gpu || gpu.available === false) {
        return '-';
      }
      const index = gpu.device_index !== null && gpu.device_index !== undefined
        ? `GPU ${gpu.device_index}`
        : (gpu.device_name ? 'GPU' : '-');
      if (gpu.device_name) {
        return `${index} · ${gpu.device_name}`;
      }
      return index;
    }

    function formatGpuMemory(gpu) {
      if (!gpu || gpu.available === false) {
        return 'GPU 상태를 읽지 못했습니다.';
      }
      return gpu.detail || '-';
    }

    function formatGpuVram(gpu) {
      if (!gpu || gpu.available === false) {
        return '-';
      }
      const used = gpu.memory_used_mb;
      const total = gpu.memory_total_mb;
      if (used === null || used === undefined || total === null || total === undefined || total === 0) {
        return '-';
      }
      return `${(used / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GB`;
    }

    function formatGpuVramMeta(gpu) {
      if (!gpu || gpu.available === false) {
        return 'VRAM 상태를 읽지 못했습니다.';
      }
      const parts = [];
      if (gpu.memory_percent !== null && gpu.memory_percent !== undefined) {
        parts.push(`${gpu.memory_percent.toFixed(1)}% 사용 중`);
      }
      if (gpu.utilization_memory !== null && gpu.utilization_memory !== undefined) {
        parts.push(`mem util ${gpu.utilization_memory}%`);
      }
      if (gpu.temperature_c !== null && gpu.temperature_c !== undefined) {
        parts.push(`${gpu.temperature_c}°C`);
      }
      return parts.length ? parts.join(' · ') : (gpu.detail || '-');
    }

    function formatTrainingDevice(progress, gpu) {
      const device = progress?.device;
      if (!device) {
        return '-';
      }
      if (device.startsWith('cuda') && gpu?.device_name) {
        return `${device} · ${gpu.device_name}`;
      }
      return device;
    }

    function formatTrainingDeviceMeta(progress) {
      if (!progress || !progress.device) {
        return '학습 프로세스 장치 정보가 없습니다.';
      }
      const ampText = progress.amp_enabled ? 'AMP on' : 'AMP off';
      const workerText = progress.num_workers !== null && progress.num_workers !== undefined
        ? `workers ${progress.num_workers}`
        : 'workers -';
      return `${ampText} · ${workerText}`;
    }

    const viewerMode = (() => {
      const params = new URLSearchParams(window.location.search);
      const host = String(window.location.hostname || '').toLowerCase();
      return (
        params.get('viewer') === '1' ||
        host.includes('trycloudflare') ||
        host.endsWith('.workers.dev')
      );
    })();

    function applyViewerMode() {
      if (!viewerMode) {
        return;
      }

      const panel = document.getElementById('controlPanel');
      if (panel) {
        panel.classList.add('viewer-mode');
      }
      const queueEditor = panel ? panel.querySelector('.queue-editor') : null;
      const controlActions = panel ? panel.querySelector('.control-actions') : null;
      const metaRow = panel ? panel.querySelector('.meta-row') : null;
      const queueManager = panel ? panel.querySelector('.queue-manager') : null;
      if (queueEditor) {
        queueEditor.style.display = 'none';
      }
      if (controlActions) {
        controlActions.style.display = 'none';
      }
      if (metaRow) {
        metaRow.style.display = 'none';
      }
      if (queueManager) {
        queueManager.style.display = 'none';
      }
      const title = document.getElementById('controlTitle');
      const copy = document.getElementById('controlCopy');
      if (title) {
        title.textContent = '실행 현황';
      }
      if (copy) {
        copy.textContent = '현재 작업과 최근 완료 상태를 확인합니다.';
      }

      [
        'datasetKeyInput',
        'apiKeyInput',
        'filekeysInput',
        'startButton',
        'reextractButton',
        'stopButton',
        'forceStopButton',
        'resetButton',
      ].forEach((id) => {
        const element = document.getElementById(id);
        if (!element) return;
        element.disabled = true;
        if (element.tagName === 'TEXTAREA' || element.tagName === 'INPUT') {
          element.setAttribute('readonly', 'readonly');
          element.setAttribute('tabindex', '-1');
        }
      });
      const eyebrow = document.querySelector('.eyebrow');
      if (eyebrow) {
        eyebrow.textContent = 'Viewer';
      }
      const heroCopy = document.querySelector('.hero-copy p');
      if (heroCopy) {
        heroCopy.textContent = '학습 상태와 최근 결과를 실시간으로 확인합니다.';
      }
    }

    function getElement(id) {
      return document.getElementById(id);
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function setText(id, value) {
      const element = getElement(id);
      if (element) {
        element.textContent = value;
      }
      return element;
    }

    function setHTML(id, value) {
      const element = getElement(id);
      if (element) {
        element.innerHTML = value;
      }
      return element;
    }

    function setWidth(id, value) {
      const element = getElement(id);
      if (element) {
        element.style.width = value;
      }
      return element;
    }

    function setLaunchMessage(message, isError) {
      const box = document.getElementById('launchMessage');
      if (!box) {
        return;
      }
      box.textContent = message || '-';
      box.style.color = isError ? '#dc2626' : '#64748b';
      box.style.borderColor = isError ? 'rgba(220,38,38,0.18)' : 'rgba(148,163,184,0.18)';
      box.style.background = isError ? 'rgba(220,38,38,0.06)' : 'rgba(255,255,255,0.84)';
    }

    function renderQueuedJobs(jobs) {
      const count = document.getElementById('queuedJobCount');
      const list = document.getElementById('queuedJobList');
      if (!count || !list) {
        return;
      }
      const items = Array.isArray(jobs) ? jobs : [];
      count.textContent = `${items.length}개`;
      if (!items.length) {
        list.innerHTML = '<div class="queued-job-empty">대기 중인 filekey가 없습니다.</div>';
        return;
      }
      list.innerHTML = items.map((job) => `
        <div class="queued-job-item">
          <div class="queued-job-main">
            <div class="queued-job-key">filekey ${escapeHtml(job.filekey || '-')}</div>
            <div class="queued-job-meta">
              datasetkey ${escapeHtml(job.datasetkey || '-')} · queued ${formatDateTime(job.queued_at)}
            </div>
          </div>
          <button
            class="queued-remove-button"
            type="button"
            data-job-id="${escapeHtml(job.job_id || '')}"
            data-filekey="${escapeHtml(job.filekey || '')}"
          >제거</button>
        </div>
      `).join('');
    }

    function updateControlButtons(launcher) {
      const startButton = document.getElementById('startButton');
      const stopButton = document.getElementById('stopButton');
      const forceStopButton = document.getElementById('forceStopButton');
      const reextractButton = document.getElementById('reextractButton');
      const resetButton = document.getElementById('resetButton');
      if (!startButton || !stopButton || !forceStopButton || !resetButton || !reextractButton) {
        return;
      }
      if (viewerMode) {
        startButton.disabled = true;
        reextractButton.disabled = true;
        stopButton.disabled = true;
        forceStopButton.disabled = true;
        resetButton.disabled = true;
        return;
      }
      const autoStartEnabled = launcher?.auto_start_enabled !== false;
      const hasCurrentJob = !!launcher?.current_job;
      const pendingCount = (launcher?.pending_jobs || []).length;

      if (!autoStartEnabled && (hasCurrentJob || pendingCount > 0)) {
        startButton.textContent = hasCurrentJob ? '큐 재개' : '대기열 시작';
      } else {
        startButton.textContent = '시작 / 추가';
      }
      reextractButton.disabled = false;

      if (hasCurrentJob || pendingCount > 0) {
        stopButton.disabled = !autoStartEnabled;
        stopButton.textContent = autoStartEnabled ? '현재 작업 후 중지' : '중지 예약됨';
      } else {
        stopButton.disabled = true;
        stopButton.textContent = '현재 작업 후 중지';
      }

      forceStopButton.disabled = !hasCurrentJob;
      forceStopButton.textContent = '지금 중단';
    }

    function safeLocalStorageGet(key) {
      try {
        return window.localStorage.getItem(key);
      } catch (error) {
        console.warn(`localStorage get failed for ${key}`, error);
        return null;
      }
    }

    function safeLocalStorageSet(key, value) {
      try {
        window.localStorage.setItem(key, value);
        return true;
      } catch (error) {
        console.warn(`localStorage set failed for ${key}`, error);
        return false;
      }
    }

    function safeLocalStorageRemove(key) {
      try {
        window.localStorage.removeItem(key);
        return true;
      } catch (error) {
        console.warn(`localStorage remove failed for ${key}`, error);
        return false;
      }
    }

    function loadSavedApiKey() {
      if (viewerMode) {
        return;
      }
      const input = document.getElementById('apiKeyInput');
      if (!input) {
        return;
      }
      const saved = safeLocalStorageGet('training_dashboard_aihub_api_key');
      if (saved) {
        input.value = saved;
      }
    }

    function saveApiKey() {
      if (viewerMode) {
        return '';
      }
      const input = document.getElementById('apiKeyInput');
      if (!input) {
        return '';
      }
      const value = input.value.trim();
      if (value) {
        safeLocalStorageSet('training_dashboard_aihub_api_key', value);
      } else {
        safeLocalStorageRemove('training_dashboard_aihub_api_key');
      }
      return value;
    }

    function loadSavedDatasetKey() {
      if (viewerMode) {
        return;
      }
      const input = document.getElementById('datasetKeyInput');
      if (!input) {
        return;
      }
      const saved = safeLocalStorageGet('training_dashboard_aihub_datasetkey');
      if (saved) {
        input.value = saved;
      }
    }

    function saveDatasetKey() {
      const input = document.getElementById('datasetKeyInput');
      if (!input) {
        return '';
      }
      if (viewerMode) {
        return input.value.trim();
      }
      const value = input.value.trim();
      if (value) {
        safeLocalStorageSet('training_dashboard_aihub_datasetkey', value);
      } else {
        safeLocalStorageRemove('training_dashboard_aihub_datasetkey');
      }
      return value;
    }

    function buildAreaPath(points, height, padBottom) {
      if (!points.length) {
        return '';
      }
      const [firstX, firstY] = points[0].split(',').map(Number);
      const [lastX] = points[points.length - 1].split(',').map(Number);
      return `M ${firstX} ${height - padBottom} L ${firstX} ${firstY} L ${points.join(' L ')} L ${lastX} ${height - padBottom} Z`;
    }

    function attachChartTooltip({
      svgId,
      tooltipId,
      lineId,
      bottomY,
    }) {
      const svg = document.getElementById(svgId);
      const tooltip = document.getElementById(tooltipId);
      if (!svg || !tooltip) {
        return;
      }
      const line = lineId ? svg.querySelector(`#${lineId}`) : null;
      const points = svg.querySelectorAll('.chart-point-hit');
      let activeGroup = null;
      const hideTooltip = () => {
        tooltip.classList.remove('visible');
        if (line) {
          line.style.opacity = '0';
        }
        if (activeGroup) {
          activeGroup.classList.remove('is-active');
          activeGroup = null;
        }
      };
      points.forEach((point) => {
        const showTooltip = (event) => {
          const group = point.closest('.chart-point');
          if (activeGroup && activeGroup !== group) {
            activeGroup.classList.remove('is-active');
          }
          if (group) {
            group.classList.add('is-active');
            activeGroup = group;
          }
          const title = point.dataset.title || '';
          const lines = (point.dataset.lines || '').split('|').filter(Boolean);
          tooltip.innerHTML = `
            <div class="chart-tooltip-title">${escapeHtml(title)}</div>
            ${lines.map((entry) => {
              const parts = entry.split(':');
              const key = parts.shift() || '';
              const value = parts.join(':');
              return `<div class="chart-tooltip-line"><strong>${escapeHtml(key)}</strong>${escapeHtml(value)}</div>`;
            }).join('')}
          `;
          const shellRect = tooltip.parentElement.getBoundingClientRect();
          const x = event.clientX - shellRect.left;
          const y = event.clientY - shellRect.top;
          tooltip.classList.add('visible');
          const tooltipWidth = tooltip.offsetWidth || 180;
          const tooltipHeight = tooltip.offsetHeight || 72;
          const shellWidth = shellRect.width;
          const shellHeight = shellRect.height;
          const margin = 10;
          const verticalGap = 14;

          let left = x - tooltipWidth / 2;
          left = Math.max(margin, Math.min(left, shellWidth - tooltipWidth - margin));

          let top = y - tooltipHeight - verticalGap;
          if (top < margin) {
            top = Math.min(shellHeight - tooltipHeight - margin, y + verticalGap);
          }
          top = Math.max(margin, top);

          tooltip.style.left = `${left}px`;
          tooltip.style.top = `${top}px`;
          tooltip.classList.add('visible');
          if (line) {
            const cx = Number(point.dataset.cx || 0);
            line.setAttribute('x1', cx);
            line.setAttribute('x2', cx);
            line.setAttribute('y1', 16);
            line.setAttribute('y2', bottomY);
            line.style.opacity = '1';
          }
        };
        point.addEventListener('mouseenter', showTooltip);
        point.addEventListener('mousemove', showTooltip);
        point.addEventListener('mouseleave', hideTooltip);
      });
      svg.addEventListener('mouseleave', hideTooltip);
    }

    function renderChart(history) {
      const svg = document.getElementById('trainingChart');
      if (!history || !history.length) {
        svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="16">아직 학습 기록이 없습니다</text>';
        const tooltip = document.getElementById('trainingChartTooltip');
        if (tooltip) {
          tooltip.classList.remove('visible');
        }
        return;
      }

      const width = 800;
      const height = 260;
      const padLeft = 46;
      const padRight = 20;
      const padTop = 16;
      const padBottom = 30;
      const innerW = width - padLeft - padRight;
      const innerH = height - padTop - padBottom;

      const accPoints = [];
      const f1Points = [];
      const accCircles = [];
      const f1Circles = [];
      const maxX = Math.max(history.length - 1, 1);

      history.forEach((row, index) => {
        const x = padLeft + (index / maxX) * innerW;
        const accY = padTop + (1 - Math.max(0, Math.min(1, row.val_accuracy ?? 0))) * innerH;
        const f1Y = padTop + (1 - Math.max(0, Math.min(1, row.val_macro_f1 ?? 0))) * innerH;
        accPoints.push(`${x},${accY}`);
        f1Points.push(`${x},${f1Y}`);
        accCircles.push(`
          <g class="chart-point" transform="translate(${x}, ${accY})">
            <circle class="chart-point-core" r="4.5" fill="#ffffff"></circle>
            <circle class="chart-point-core" r="3" fill="#2563eb"></circle>
            <circle
              class="chart-point-hit"
              r="12"
              data-cx="${x}"
              data-title="Epoch ${row.epoch}"
              data-lines="Val Acc:${Number(row.val_accuracy ?? 0).toFixed(4)}|Val Macro F1:${Number(row.val_macro_f1 ?? 0).toFixed(4)}|Train Loss:${Number(row.train_loss ?? 0).toFixed(4)}|Val Loss:${Number(row.val_loss ?? 0).toFixed(4)}"
            ></circle>
          </g>
        `);
        f1Circles.push(`
          <g class="chart-point" transform="translate(${x}, ${f1Y})">
            <circle class="chart-point-core" r="4.5" fill="#ffffff"></circle>
            <circle class="chart-point-core" r="3" fill="#059669"></circle>
            <circle
              class="chart-point-hit"
              r="12"
              data-cx="${x}"
              data-title="Epoch ${row.epoch}"
              data-lines="Val Macro F1:${Number(row.val_macro_f1 ?? 0).toFixed(4)}|Val Acc:${Number(row.val_accuracy ?? 0).toFixed(4)}|Train Loss:${Number(row.train_loss ?? 0).toFixed(4)}|Val Loss:${Number(row.val_loss ?? 0).toFixed(4)}"
            ></circle>
          </g>
        `);
      });

      const gridLines = [0, 0.25, 0.5, 0.75, 1].map(value => {
        const y = padTop + (1 - value) * innerH;
        return `
          <line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="rgba(148,163,184,0.18)" />
          <text x="8" y="${y + 4}" fill="#94a3b8" font-size="11">${value.toFixed(2)}</text>
        `;
      }).join('');

      const xLabels = history.map((row, index) => {
        const x = padLeft + (index / maxX) * innerW;
        return `<text x="${x}" y="${height - 8}" fill="#94a3b8" font-size="11" text-anchor="middle">${row.epoch}</text>`;
      }).join('');
      const accArea = buildAreaPath(accPoints, height, padBottom);
      const f1Area = buildAreaPath(f1Points, height, padBottom);

      svg.innerHTML = `
        <defs>
          <linearGradient id="trainingAccStroke" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#3b82f6" />
            <stop offset="100%" stop-color="#2563eb" />
          </linearGradient>
          <linearGradient id="trainingF1Stroke" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#34d399" />
            <stop offset="100%" stop-color="#059669" />
          </linearGradient>
          <linearGradient id="trainingAccFill" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.22" />
            <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.01" />
          </linearGradient>
          <linearGradient id="trainingF1Fill" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#059669" stop-opacity="0.20" />
            <stop offset="100%" stop-color="#059669" stop-opacity="0.01" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"></rect>
        ${gridLines}
        <path d="${accArea}" fill="url(#trainingAccFill)"></path>
        <path d="${f1Area}" fill="url(#trainingF1Fill)"></path>
        <line id="trainingChartHoverLine" class="chart-hover-line" x1="0" y1="0" x2="0" y2="0"></line>
        <polyline fill="none" stroke="url(#trainingAccStroke)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" points="${accPoints.join(' ')}"></polyline>
        <polyline fill="none" stroke="url(#trainingF1Stroke)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" points="${f1Points.join(' ')}"></polyline>
        ${accCircles.join('')}
        ${f1Circles.join('')}
        ${xLabels}
      `;
      attachChartTooltip({
        svgId: 'trainingChart',
        tooltipId: 'trainingChartTooltip',
        lineId: 'trainingChartHoverLine',
        bottomY: height - padBottom,
      });
    }

    function renderLossChart(history) {
      const svg = document.getElementById('lossChart');
      if (!history || !history.length) {
        svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="16">아직 손실 기록이 없습니다</text>';
        const tooltip = document.getElementById('lossChartTooltip');
        if (tooltip) {
          tooltip.classList.remove('visible');
        }
        return;
      }

      const width = 800;
      const height = 260;
      const padLeft = 52;
      const padRight = 20;
      const padTop = 16;
      const padBottom = 30;
      const innerW = width - padLeft - padRight;
      const innerH = height - padTop - padBottom;

      const maxLoss = Math.max(
        0.001,
        ...history.flatMap((row) => [Number(row.train_loss || 0), Number(row.val_loss || 0)])
      );
      const maxX = Math.max(history.length - 1, 1);
      const trainPoints = [];
      const valPoints = [];
      const trainCircles = [];
      const valCircles = [];

      history.forEach((row, index) => {
        const x = padLeft + (index / maxX) * innerW;
        const trainY = padTop + (1 - Math.min(1, Number(row.train_loss || 0) / maxLoss)) * innerH;
        const valY = padTop + (1 - Math.min(1, Number(row.val_loss || 0) / maxLoss)) * innerH;
        trainPoints.push(`${x},${trainY}`);
        valPoints.push(`${x},${valY}`);
        trainCircles.push(`
          <g class="chart-point" transform="translate(${x}, ${trainY})">
            <circle class="chart-point-core" r="4.5" fill="#ffffff"></circle>
            <circle class="chart-point-core" r="3" fill="#2563eb"></circle>
            <circle
              class="chart-point-hit"
              r="12"
              data-cx="${x}"
              data-title="Epoch ${row.epoch}"
              data-lines="Train Loss:${Number(row.train_loss ?? 0).toFixed(4)}|Val Loss:${Number(row.val_loss ?? 0).toFixed(4)}|Val Acc:${Number(row.val_accuracy ?? 0).toFixed(4)}|Val Macro F1:${Number(row.val_macro_f1 ?? 0).toFixed(4)}"
            ></circle>
          </g>
        `);
        valCircles.push(`
          <g class="chart-point" transform="translate(${x}, ${valY})">
            <circle class="chart-point-core" r="4.5" fill="#ffffff"></circle>
            <circle class="chart-point-core" r="3" fill="#059669"></circle>
            <circle
              class="chart-point-hit"
              r="12"
              data-cx="${x}"
              data-title="Epoch ${row.epoch}"
              data-lines="Val Loss:${Number(row.val_loss ?? 0).toFixed(4)}|Train Loss:${Number(row.train_loss ?? 0).toFixed(4)}|Val Acc:${Number(row.val_accuracy ?? 0).toFixed(4)}|Val Macro F1:${Number(row.val_macro_f1 ?? 0).toFixed(4)}"
            ></circle>
          </g>
        `);
      });

      const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = padTop + (1 - ratio) * innerH;
        const value = (maxLoss * ratio).toFixed(3);
        return `
          <line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="rgba(148,163,184,0.18)" />
          <text x="8" y="${y + 4}" fill="#94a3b8" font-size="11">${value}</text>
        `;
      }).join('');

      const xLabels = history.map((row, index) => {
        const x = padLeft + (index / maxX) * innerW;
        return `<text x="${x}" y="${height - 8}" fill="#94a3b8" font-size="11" text-anchor="middle">${row.epoch}</text>`;
      }).join('');
      const trainArea = buildAreaPath(trainPoints, height, padBottom);
      const valArea = buildAreaPath(valPoints, height, padBottom);

      svg.innerHTML = `
        <defs>
          <linearGradient id="lossTrainStroke" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#60a5fa" />
            <stop offset="100%" stop-color="#2563eb" />
          </linearGradient>
          <linearGradient id="lossValStroke" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#6ee7b7" />
            <stop offset="100%" stop-color="#059669" />
          </linearGradient>
          <linearGradient id="lossTrainFill" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#2563eb" stop-opacity="0.20" />
            <stop offset="100%" stop-color="#2563eb" stop-opacity="0.01" />
          </linearGradient>
          <linearGradient id="lossValFill" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#059669" stop-opacity="0.18" />
            <stop offset="100%" stop-color="#059669" stop-opacity="0.01" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"></rect>
        ${ticks}
        <path d="${trainArea}" fill="url(#lossTrainFill)"></path>
        <path d="${valArea}" fill="url(#lossValFill)"></path>
        <line id="lossChartHoverLine" class="chart-hover-line" x1="0" y1="0" x2="0" y2="0"></line>
        <polyline fill="none" stroke="url(#lossTrainStroke)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" points="${trainPoints.join(' ')}"></polyline>
        <polyline fill="none" stroke="url(#lossValStroke)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" points="${valPoints.join(' ')}"></polyline>
        ${trainCircles.join('')}
        ${valCircles.join('')}
        ${xLabels}
      `;
      attachChartTooltip({
        svgId: 'lossChart',
        tooltipId: 'lossChartTooltip',
        lineId: 'lossChartHoverLine',
        bottomY: height - padBottom,
      });
    }

    function renderSingleMetricChart({
      svgId,
      tooltipId,
      lineId,
      valueId,
      history,
      label,
      color,
      valueGetter,
      valueFormatter,
      emptyText,
      includeZeroLine = false,
    }) {
      const svg = document.getElementById(svgId);
      if (!svg) {
        return;
      }
      const rows = Array.isArray(history) ? history : [];
      const values = rows.map((row) => {
        const value = Number(valueGetter(row));
        return Number.isFinite(value) ? value : null;
      });
      const validValues = values.filter((value) => value !== null);
      const latestValue = validValues.length ? validValues[validValues.length - 1] : null;
      setText(valueId, latestValue === null ? '-' : valueFormatter(latestValue));
      if (!rows.length || !validValues.length) {
        svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="15">${escapeHtml(emptyText)}</text>`;
        const tooltip = document.getElementById(tooltipId);
        if (tooltip) {
          tooltip.classList.remove('visible');
        }
        return;
      }

      const width = 800;
      const height = 220;
      const padLeft = 58;
      const padRight = 20;
      const padTop = 16;
      const padBottom = 30;
      const innerW = width - padLeft - padRight;
      const innerH = height - padTop - padBottom;
      let minY = Math.min(...validValues);
      let maxY = Math.max(...validValues);
      if (includeZeroLine) {
        minY = Math.min(minY, 0);
        maxY = Math.max(maxY, 0);
      }
      if (maxY <= minY) {
        maxY = minY + 1;
      }
      const padding = (maxY - minY) * 0.12 || 0.001;
      minY -= padding;
      maxY += padding;
      const maxX = Math.max(rows.length - 1, 1);
      const xFor = (index) => padLeft + (index / maxX) * innerW;
      const yFor = (value) => padTop + (1 - ((value - minY) / (maxY - minY))) * innerH;

      const points = [];
      const circles = [];
      rows.forEach((row, index) => {
        const value = values[index];
        if (value === null) {
          return;
        }
        const x = xFor(index);
        const y = yFor(value);
        points.push(`${x},${y}`);
        circles.push(`
          <g class="chart-point" transform="translate(${x}, ${y})">
            <circle class="chart-point-core" r="4.4" fill="#ffffff"></circle>
            <circle class="chart-point-core" r="3" fill="${color}"></circle>
            <circle
              class="chart-point-hit"
              r="12"
              data-cx="${x}"
              data-title="Epoch ${escapeHtml(row.epoch ?? index + 1)}"
              data-lines="${escapeHtml(label)}:${escapeHtml(valueFormatter(value))}|Train Loss:${Number(row.train_loss ?? 0).toFixed(4)}|Val Loss:${Number(row.val_loss ?? 0).toFixed(4)}|Val Macro F1:${Number(row.val_macro_f1 ?? 0).toFixed(4)}"
            ></circle>
          </g>
        `);
      });

      const grid = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = padTop + (1 - ratio) * innerH;
        const value = minY + ((maxY - minY) * ratio);
        return `
          <line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="rgba(148,163,184,0.18)" />
          <text x="8" y="${y + 4}" fill="#94a3b8" font-size="11">${escapeHtml(valueFormatter(value))}</text>
        `;
      }).join('');
      const zeroLine = includeZeroLine && minY < 0 && maxY > 0
        ? `<line x1="${padLeft}" y1="${yFor(0)}" x2="${width - padRight}" y2="${yFor(0)}" stroke="rgba(15,23,42,0.22)" stroke-dasharray="5 5" />`
        : '';
      const xLabels = rows.map((row, index) => {
        if (rows.length > 10 && index % Math.ceil(rows.length / 6) !== 0 && index !== rows.length - 1) {
          return '';
        }
        return `<text x="${xFor(index)}" y="${height - 8}" fill="#94a3b8" font-size="11" text-anchor="middle">${escapeHtml(row.epoch ?? index + 1)}</text>`;
      }).join('');

      svg.innerHTML = `
        ${grid}
        ${zeroLine}
        <line id="${lineId}" class="chart-hover-line" x1="0" y1="0" x2="0" y2="0"></line>
        <polyline fill="none" stroke="${color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" points="${points.join(' ')}"></polyline>
        ${circles.join('')}
        ${xLabels}
      `;
      attachChartTooltip({
        svgId,
        tooltipId,
        lineId,
        bottomY: height - padBottom,
      });
    }

    function renderLearningRateChart(history) {
      renderSingleMetricChart({
        svgId: 'learningRateChart',
        tooltipId: 'learningRateChartTooltip',
        lineId: 'learningRateChartHoverLine',
        valueId: 'learningRateChartValue',
        history,
        label: 'Learning Rate',
        color: '#7c3aed',
        valueGetter: (row) => row.learning_rate,
        valueFormatter: (value) => Number(value).toExponential(2),
        emptyText: 'learning rate 기록이 없습니다',
      });
    }

    function renderLossGapChart(history) {
      renderSingleMetricChart({
        svgId: 'lossGapChart',
        tooltipId: 'lossGapChartTooltip',
        lineId: 'lossGapChartHoverLine',
        valueId: 'lossGapChartValue',
        history,
        label: 'Val - Train',
        color: '#f97316',
        valueGetter: (row) => Number(row.val_loss) - Number(row.train_loss),
        valueFormatter: (value) => `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(4)}`,
        emptyText: 'loss gap 기록이 없습니다',
        includeZeroLine: true,
      });
    }

    function renderDatasetTable(dataset, distributionSources) {
      const tbody = document.getElementById('datasetTable');
      const empty = document.getElementById('datasetEmpty');
      const rows = [];
      const splitSections = ['raw', 'train', 'val', 'test'];
      const preparedSections = ['prepared_train', 'prepared_val', 'prepared_test'];
      const sectionLabels = {
        raw: '누적 원본',
        train: '누적 train split',
        val: '누적 val split',
        test: '누적 test split',
        prepared_train: 'train',
        prepared_val: 'val',
        prepared_test: 'test',
      };
      const sourceLabels = {
        cumulative: '누적 원본/Pose',
        guideline: 'Guideline/RGB clip',
        active: '학습 active',
        current: '현재 작업',
      };

      const appendRow = (sourceLabel, key, info) => {
        if (!info || !info.total) {
          return;
        }
        const labels = Object.entries(info.by_label || {})
          .map(([label, count]) => `${escapeHtml(label)} ${count}`)
          .join(' / ');
        rows.push(`
          <tr>
            <td>${escapeHtml(sourceLabel)}</td>
            <td>${escapeHtml(sectionLabels[key] || key)}</td>
            <td>${info.total}</td>
            <td>${labels || '-'}</td>
          </tr>
        `);
      };

      if (distributionSources && Object.keys(distributionSources).length) {
        const cumulative = distributionSources.cumulative || {};
        splitSections.forEach((key) => appendRow(sourceLabels.cumulative, key, cumulative[key]));
        preparedSections.forEach((key) => appendRow(sourceLabels.cumulative, key, cumulative[key]));
        preparedSections.forEach((key) => appendRow(sourceLabels.guideline, key, distributionSources.guideline?.[key]));
        preparedSections.forEach((key) => appendRow(sourceLabels.active, key, distributionSources.active?.[key]));
        splitSections.concat(preparedSections).forEach((key) => appendRow(sourceLabels.current, key, distributionSources.current?.[key]));
      } else {
        splitSections.concat(preparedSections).forEach((key) => appendRow('데이터셋', key, dataset[key]));
      }

      if (!rows.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
      }
      empty.style.display = 'none';
      tbody.innerHTML = rows.join('');
    }

    function renderCurrentJobProgress(jobProgress, currentDataset, continualState, progress) {
      const ratio = Math.max(0, Math.min(100, Math.round((jobProgress?.ratio ?? 0) * 100)));
      document.getElementById('currentJobProgressText').textContent = `${ratio}%`;
      document.getElementById('currentJobProgressMeta').textContent =
        `${jobProgress?.label || '-'} | ${jobProgress?.detail || '-'}`;
      document.getElementById('currentJobProgressFill').style.width = `${ratio}%`;

      const currentRaw = currentDataset?.raw?.total ?? 0;
      const currentPrepared =
        (currentDataset?.prepared_train?.total ?? 0) +
        (currentDataset?.prepared_val?.total ?? 0) +
        (currentDataset?.prepared_test?.total ?? 0);
      document.getElementById('currentDatasetTotals').textContent = `${currentRaw} / ${currentPrepared}`;
      document.getElementById('currentDatasetSummary').textContent = 'current raw / current prepared';

      document.getElementById('jobStageDetail').textContent = jobProgress?.detail || '-';
      document.getElementById('jobCurrentVideo').textContent = jobProgress?.current_video || '-';

      const continualTrain = continualState?.prepared_train_total ?? (progress?.train_samples ?? 0);
      const continualVal = continualState?.prepared_val_total ?? (progress?.val_samples ?? 0);
      const resumed = progress?.resumed_from_checkpoint ? 'resume on' : 'resume off';
      document.getElementById('continualStateText').textContent = resumed;
      document.getElementById('continualStateMeta').textContent =
        `누적 prepared train ${continualTrain} / val ${continualVal}`;

      const trainImbalance = formatImbalanceSummary(progress?.train_distribution || null);
      const valImbalance = formatImbalanceSummary(progress?.val_distribution || null);
      document.getElementById('imbalanceStatusText').textContent = trainImbalance.value;
      document.getElementById('imbalanceStatusMeta').textContent =
        `train ${trainImbalance.copy} | val ${valImbalance.copy}`;
    }

    function buildPerClassSupport(labels, confusion) {
      const matrix = Array.isArray(confusion) ? confusion : [];
      const supports = [];
      for (let index = 0; index < matrix.length; index += 1) {
        const row = Array.isArray(matrix[index]) ? matrix[index] : [];
        const support = row.reduce((sum, value) => sum + Number(value || 0), 0);
        supports.push({
          class_index: index,
          label: labels?.[index] || `class_${index}`,
          support,
        });
      }
      return supports;
    }

    function formatImbalanceSummary(distribution) {
      if (!distribution) {
        return { value: '-', copy: '클래스 분포 정보가 없습니다.' };
      }
      const severity = String(distribution.severity || 'ok');
      const covered = Number(distribution.covered ?? 0);
      const total = Number(distribution.total ?? 0);
      const ratio = distribution.imbalance_ratio !== null && distribution.imbalance_ratio !== undefined
        ? `ratio ${Number(distribution.imbalance_ratio).toFixed(2)}`
        : 'ratio -';
      const label =
        severity === 'critical' ? 'critical' :
        severity === 'warning' ? 'warning' :
        'ok';
      const message = Array.isArray(distribution.messages) && distribution.messages.length
        ? distribution.messages[0]
        : '클래스 분포가 크게 치우치지 않았습니다.';
      return {
        value: `${label} · ${covered}/${total}`,
        copy: `${ratio} · ${message}`,
      };
    }

    function formatStageTimingValue(stageTimings) {
      const byStage = stageTimings?.by_stage || {};
      const parts = ['download', 'prepare', 'train']
        .map((stage) => {
          const seconds = byStage?.[stage]?.duration_seconds;
          if (seconds === null || seconds === undefined) {
            return null;
          }
          return `${stage} ${formatDuration(seconds)}`;
        })
        .filter(Boolean);
      return parts.length ? parts.join(' · ') : '-';
    }

    function formatStageTimingMeta(stageTimings, totalSeconds) {
      const byStage = stageTimings?.by_stage || {};
      const completedCount = ['download', 'prepare', 'train']
        .filter((stage) => byStage?.[stage]?.duration_seconds !== null && byStage?.[stage]?.duration_seconds !== undefined)
        .length;
      const totalLabel =
        totalSeconds !== null && totalSeconds !== undefined
          ? `total ${formatDuration(totalSeconds)}`
          : null;
      return [totalLabel, `${completedCount}개 단계 기록`]
        .filter(Boolean)
        .join(' · ') || 'download / prepare / train 소요 시간이 아직 없습니다.';
    }

    function formatActiveStageElapsed(pipeline) {
      if (!pipeline?.stage_started_at || !pipeline?.stage) {
        return null;
      }
      const start = new Date(pipeline.stage_started_at).getTime();
      if (Number.isNaN(start)) {
        return null;
      }
      const seconds = Math.max(0, Math.round((Date.now() - start) / 1000));
      return `${pipeline.stage} ${formatDuration(seconds)} 진행 중`;
    }

    function renderMetricInsights(progress, metrics) {
      const finalValidation = metrics?.final_validation || progress?.final_validation || {};
      const history = progress?.history || [];
      const latest = progress?.latest || history[history.length - 1] || null;
      const labels = metrics?.labels || progress?.labels || [];
      const earlyStopping = progress?.early_stopping || metrics?.early_stopping || {};
      const stoppedEarly = Boolean(progress?.stopped_early ?? metrics?.stopped_early);
      const trainImbalance = formatImbalanceSummary(progress?.train_distribution || metrics?.train_distribution || null);
      const supports = buildPerClassSupport(labels, finalValidation?.confusion_matrix || []);
      const totalSupport = supports.reduce((sum, row) => sum + Number(row.support || 0), 0);
      const coveredClasses = supports.filter((row) => Number(row.support || 0) > 0);
      const dominant = supports.slice().sort((a, b) => Number(b.support || 0) - Number(a.support || 0))[0];
      const lossGap =
        latest && latest.train_loss !== undefined && latest.val_loss !== undefined
          ? Number(latest.val_loss) - Number(latest.train_loss)
          : null;

      document.getElementById('finalValMetrics').textContent =
        finalValidation?.accuracy !== undefined && finalValidation?.macro_f1 !== undefined
          ? `${Number(finalValidation.accuracy).toFixed(3)} / ${Number(finalValidation.macro_f1).toFixed(3)}`
          : '-';
      document.getElementById('finalValMetricsCopy').textContent = 'accuracy / macro F1';

      document.getElementById('bestEpochValue').textContent =
        progress?.best_epoch !== undefined && progress?.best_epoch !== null
          ? `Epoch ${progress.best_epoch}`
          : '-';
      document.getElementById('bestEpochCopy').textContent =
        progress?.best_val_macro_f1 !== undefined && progress?.best_val_macro_f1 !== null
          ? `best macro F1 ${Number(progress.best_val_macro_f1).toFixed(3)}`
          : '가장 높은 macro F1을 기록한 epoch';

      document.getElementById('lossGapValue').textContent =
        lossGap !== null ? `${lossGap >= 0 ? '+' : ''}${lossGap.toFixed(4)}` : '-';
      document.getElementById('lossGapCopy').textContent =
        latest ? `latest val ${latest.val_loss} - train ${latest.train_loss}` : '최신 val loss - train loss';

      document.getElementById('valSampleTotal').textContent =
        totalSupport > 0 ? String(totalSupport) : '-';
      document.getElementById('valSampleCopy').textContent = '최종 validation 샘플 수';

      document.getElementById('classCoverageValue').textContent =
        supports.length ? `${coveredClasses.length} / ${supports.length}` : '-';
      document.getElementById('classCoverageCopy').textContent = 'validation에 등장한 클래스 수';

      document.getElementById('dominantClassValue').textContent =
        dominant && Number(dominant.support || 0) > 0 ? dominant.label : '-';
      document.getElementById('dominantClassCopy').textContent =
        dominant && Number(dominant.support || 0) > 0
          ? `support ${dominant.support}`
          : 'validation에서 가장 많은 클래스';

      document.getElementById('earlyStopValue').textContent =
        stoppedEarly
          ? `yes · epoch ${progress?.epochs_completed ?? latest?.epoch ?? '-'}`
          : (earlyStopping?.enabled ? 'armed' : 'off');
      document.getElementById('earlyStopCopy').textContent =
        stoppedEarly
          ? (progress?.stop_reason || metrics?.stop_reason || '조기 종료되었습니다.')
          : (
            earlyStopping?.enabled
              ? `patience ${earlyStopping?.patience ?? '-'} · min delta ${earlyStopping?.min_delta ?? '-'}`
              : '조기 종료가 꺼져 있습니다.'
          );

      document.getElementById('trainImbalanceValue').textContent = trainImbalance.value;
      document.getElementById('trainImbalanceCopy').textContent = trainImbalance.copy;
    }

    function renderPerClassMetricChart(labels, perClass, confusion) {
      const wrap = document.getElementById('perClassMetricChart');
      if (!wrap) {
        return;
      }
      if (!Array.isArray(perClass) || !perClass.length) {
        wrap.innerHTML = '<div class="empty">클래스별 그래프가 아직 없습니다.</div>';
        return;
      }
      const supportRows = buildPerClassSupport(labels, confusion);
      const supportByClass = new Map(supportRows.map((row) => [Number(row.class_index), Number(row.support || 0)]));
      const sortedRows = perClass
        .filter((row) => row && row.class_index !== undefined)
        .slice()
        .sort((a, b) => Number(a.class_index) - Number(b.class_index));
      wrap.innerHTML = sortedRows.map((row) => {
        const classIndex = Number(row.class_index);
        const label = labels?.[classIndex] || row.label || `class_${classIndex}`;
        const precision = Math.max(0, Math.min(1, Number(row.precision || 0)));
        const recall = Math.max(0, Math.min(1, Number(row.recall || 0)));
        const f1 = Math.max(0, Math.min(1, Number(row.f1 || 0)));
        const support = supportByClass.get(classIndex) ?? Number(row.support || 0);
        return `
          <div class="class-bar-row">
            <div class="class-bar-label">${escapeHtml(label)}</div>
            <div class="class-bar-stack">
              <div class="class-bar-track" title="F1 ${f1.toFixed(4)}">
                <div class="class-bar-fill" style="width:${(f1 * 100).toFixed(2)}%;background:linear-gradient(90deg,#22c55e,#2563eb);"></div>
              </div>
              <div class="class-bar-values">
                <span><strong>F1</strong> ${f1.toFixed(3)}</span>
                <span><strong>Recall</strong> ${recall.toFixed(3)}</span>
                <span><strong>Precision</strong> ${precision.toFixed(3)}</span>
                <span><strong>Support</strong> ${support}</span>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderPerClassMetrics(labels, perClass, confusion) {
      const tbody = document.getElementById('perClassMetricsTable');
      const empty = document.getElementById('perClassMetricsEmpty');
      if (!perClass || !perClass.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        renderPerClassMetricChart(labels, [], confusion);
        return;
      }
      empty.style.display = 'none';
      renderPerClassMetricChart(labels, perClass, confusion);
      const supportRows = buildPerClassSupport(labels, confusion);
      tbody.innerHTML = perClass.map((row) => {
        const label = labels?.[row.class_index] || `class_${row.class_index}`;
        const support = supportRows.find((item) => item.class_index === row.class_index)?.support ?? 0;
        return `
          <tr>
            <td>${escapeHtml(label)}</td>
            <td>${row.precision ?? '-'}</td>
            <td>${row.recall ?? '-'}</td>
            <td>${row.f1 ?? '-'}</td>
            <td>${support}</td>
          </tr>
        `;
      }).join('');
    }

    function renderConfusionMatrix(labels, confusion) {
      const wrap = document.getElementById('confusionMatrixWrap');
      const empty = document.getElementById('confusionMatrixEmpty');
      const matrix = Array.isArray(confusion) ? confusion : [];
      if (!wrap || !matrix.length) {
        if (wrap) {
          wrap.innerHTML = '';
        }
        if (empty) {
          empty.style.display = 'block';
        }
        return;
      }
      if (empty) {
        empty.style.display = 'none';
      }
      const maxValue = Math.max(1, ...matrix.flatMap((row) => Array.isArray(row) ? row.map((value) => Number(value || 0)) : [0]));
      const headerCells = labels.map((label) => `<th>${escapeHtml(label)}</th>`).join('');
      const bodyRows = matrix.map((row, rowIndex) => {
        const label = labels?.[rowIndex] || `class_${rowIndex}`;
        const cells = row.map((value) => {
          const numeric = Number(value || 0);
          const intensity = Math.max(0, Math.min(1, numeric / maxValue));
          const bg = `rgba(37, 99, 235, ${0.06 + intensity * 0.44})`;
          const color = intensity > 0.55 ? '#eff6ff' : '#0f172a';
          return `<td class="heat-cell" style="background:${bg};color:${color};">${numeric}</td>`;
        }).join('');
        return `<tr><th>${escapeHtml(label)}</th>${cells}</tr>`;
      }).join('');
      wrap.innerHTML = `
        <table class="heatmap-table">
          <thead>
            <tr>
              <th>True \\ Pred</th>
              ${headerCells}
            </tr>
          </thead>
          <tbody>
            ${bodyRows}
          </tbody>
        </table>
      `;
    }

    function formatDateTime(value) {
      if (!value) {
        return '-';
      }
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) {
        return value;
      }
      return date.toLocaleString('ko-KR', { hour12: false });
    }

    function renderCompletedLogs(jobs, queueProgress) {
      const tbody = document.getElementById('completedLogsTable');
      const empty = document.getElementById('completedLogsEmpty');
      const completedCount = queueProgress?.completed ?? 0;
      const failedCount = queueProgress?.failed ?? 0;
      const pendingCount = queueProgress?.pending ?? 0;

      document.getElementById('completedCountPill').textContent = `완료 ${completedCount}`;
      document.getElementById('failedCountPill').textContent = `실패 ${failedCount}`;
      document.getElementById('pendingCountPill').textContent = `대기 ${pendingCount}`;

      if (!jobs || !jobs.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
      }

      empty.style.display = 'none';
      tbody.innerHTML = jobs.map((job) => {
        const summary = job.result_summary || {};
        const rawTotal = summary.raw_total ?? 0;
        const preparedTotal =
          (summary.prepared_train_total ?? 0) +
          (summary.prepared_val_total ?? 0) +
          (summary.prepared_test_total ?? 0);
        const issueTotal = Number(summary.total_issues ?? ((summary.broken_count ?? 0) + (summary.skipped_count ?? 0)));
        const totalDuration = summary.total_duration_seconds;
        const stoppedEarly = Boolean(summary.stopped_early);
        const stateLabel =
          job.state === 'completed' ? '완료' :
          job.state === 'completed_warning' ? '경고 종료' :
          job.state === 'data_ready' ? '데이터 준비' :
          job.state === 'aborted' ? '강제 중단' :
          '실패';
        const logPreview = job.log_preview || job.log_path || '-';
        return `
          <tr>
            <td>${escapeHtml(job.filekey || '-')}</td>
            <td>${stateLabel}${job.exit_code !== null && job.exit_code !== undefined ? ` (${job.exit_code})` : ''}</td>
            <td>${rawTotal} / ${preparedTotal}${issueTotal > 0 ? `<br>issue ${issueTotal}` : ''}${totalDuration !== null && totalDuration !== undefined ? `<br>time ${formatDuration(totalDuration)}` : ''}${stoppedEarly ? '<br>early stop' : ''}</td>
            <td>${formatDateTime(job.started_at)}<br>${formatDateTime(job.finished_at)}</td>
            <td class="mono">${escapeHtml(logPreview)}</td>
          </tr>
        `;
      }).join('');
    }

    function formatIssueReason(issue) {
      if (!issue) {
        return '-';
      }
      if (issue.category === 'broken') {
        return issue.detail || issue.reason || '읽기 실패';
      }
      if (issue.reason === 'min_frames_with_person') {
        const validFrames = Number(issue.valid_frames || 0);
        const confirmedFrames = Number(issue.confirmed_frames || 0);
        return `person frame 부족 (${validFrames}, confirmed ${confirmedFrames})`;
      }
      return issue.detail || issue.reason || '조건 미달';
    }

    function renderIssueVideos(skipReport, cumulativeSkipReport) {
      const summary = skipReport?.summary || {};
      const cumulativeSummary = cumulativeSkipReport?.summary || {};
      const issues = Array.isArray(skipReport?.issues) ? skipReport.issues : [];
      const tbody = document.getElementById('issueVideoTable');
      const empty = document.getElementById('issueVideoEmpty');

      const brokenCount = Number(summary.broken_count || 0);
      const skippedCount = Number(summary.skipped_count || 0);
      const cumulativeBroken = Number(cumulativeSummary.broken_count || 0);
      const cumulativeSkipped = Number(cumulativeSummary.skipped_count || 0);

      document.getElementById('brokenVideoCount').textContent = String(brokenCount);
      document.getElementById('skippedVideoCount').textContent = String(skippedCount);
      document.getElementById('brokenVideoSummary').textContent =
        brokenCount > 0 ? `누적 ${cumulativeBroken}개` : '읽기 실패 없음';
      document.getElementById('skippedVideoSummary').textContent =
        skippedCount > 0 ? `누적 ${cumulativeSkipped}개` : '조건 미달 없음';

      if (!issues.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
      }

      empty.style.display = 'none';
      tbody.innerHTML = issues.slice(-20).reverse().map((issue) => `
        <tr>
          <td>${issue.category === 'broken' ? '손상' : 'skip'}</td>
          <td>${escapeHtml(issue.split || '-')}</td>
          <td>${escapeHtml(issue.video_name || '-')}</td>
          <td>${escapeHtml(formatIssueReason(issue))}</td>
        </tr>
      `).join('');
    }

    function renderLogPanels(logs, launcher) {
      const current = logs?.current || {};
      const latestError = logs?.latest_error || {};
      const currentMeta = current.filekey
        ? `datasetkey ${current.datasetkey || '-'} | filekey ${current.filekey}${current.path ? ' | ' + current.path : ''}`
        : (current.path || launcher?.log_path || '실행 중인 작업이 없으면 최근 완료 로그를 표시합니다.');
      const waitingMessage = current.path || launcher?.log_path
        ? [
            '로그 파일이 생성되었습니다. 첫 출력이 도착하면 여기에 표시됩니다.',
            launcher?.message || ''
          ].filter(Boolean).join('\n')
        : '표시할 로그가 없습니다.';
      document.getElementById('currentLogMeta').textContent =
        currentMeta;
      document.getElementById('currentLogText').textContent =
        current.tail || waitingMessage;

      const errorMeta = latestError.filekey
        ? `최근 실패 datasetkey: ${latestError.datasetkey || '-'} | filekey: ${latestError.filekey}${latestError.path ? ' | ' + latestError.path : ''}`
        : '최근 실패 작업이 있으면 마지막 로그를 표시합니다.';
      document.getElementById('errorLogMeta').textContent = errorMeta;
      document.getElementById('errorLogText').textContent =
        latestError.tail || '오류 로그가 아직 없습니다.';
    }

    async function startTraining() {
      if (viewerMode) {
        setLaunchMessage('읽기 전용 공유 화면에서는 작업을 시작할 수 없습니다.', true);
        return;
      }
      const input = document.getElementById('filekeysInput').value.trim();
      const datasetKey = saveDatasetKey();
      const apiKey = saveApiKey();
      const launcher = latestOverview?.launcher || {};
      const pausedQueueExists =
        launcher?.auto_start_enabled === false &&
        ((launcher?.pending_jobs || []).length > 0 || !!launcher?.current_job);
      const resumeOnly = !input && pausedQueueExists;

      if (!input && !resumeOnly) {
        setLaunchMessage('filekey를 하나 이상 입력해 주세요.', true);
        return;
      }
      if (!resumeOnly && !datasetKey) {
        setLaunchMessage('datasetkey를 입력해 주세요.', true);
        return;
      }

      const button = document.getElementById('startButton');
      button.disabled = true;
      button.textContent = '실행 시작 중...';

      try {
        const response = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ datasetkey: datasetKey, filekeys: input, api_key: apiKey, resume_only: resumeOnly }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || '학습 시작에 실패했습니다.');
        }
        setLaunchMessage(data.message || '학습을 시작했습니다.', false);
        if (!resumeOnly) {
          document.getElementById('filekeysInput').value = '';
        }
        await refresh({ forceFull: true });
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        button.disabled = false;
        updateControlButtons(latestOverview?.launcher || {});
      }
    }

    async function reextractSelectedFilekeys() {
      if (viewerMode) {
        setLaunchMessage('읽기 전용 공유 화면에서는 재추출을 시작할 수 없습니다.', true);
        return;
      }
      const input = document.getElementById('filekeysInput').value.trim();
      const datasetKey = saveDatasetKey();
      const apiKey = saveApiKey();
      if (!input) {
        setLaunchMessage('재추출할 filekey를 입력해 주세요. 예: 49665', true);
        return;
      }
      if (!datasetKey) {
        setLaunchMessage('datasetkey를 입력해 주세요.', true);
        return;
      }
      const button = document.getElementById('reextractButton');
      button.disabled = true;
      button.textContent = '재추출 큐 추가 중...';
      try {
        const response = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            datasetkey: datasetKey,
            filekeys: input,
            api_key: apiKey,
            stage: 'extract',
            feature_extract: true,
            reextract_filekey_only: true,
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || 'filekey 재추출 큐 추가에 실패했습니다.');
        }
        setLaunchMessage(data.message || '선택 filekey 재추출을 큐에 추가했습니다.', false);
        document.getElementById('filekeysInput').value = '';
        await refresh({ forceFull: true });
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        button.textContent = '선택 filekey 재추출';
        updateControlButtons(latestOverview?.launcher || {});
      }
    }

    async function pauseQueue() {
      if (viewerMode) {
        setLaunchMessage('읽기 전용 공유 화면에서는 큐를 중지할 수 없습니다.', true);
        return;
      }
      const button = document.getElementById('stopButton');
      button.disabled = true;
      button.textContent = '중지 요청 중...';

      try {
        const response = await fetch('/api/pause', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || '중지 요청에 실패했습니다.');
        }
        setLaunchMessage(data.message || '현재 작업까지만 진행하고 다음 큐 자동 시작을 멈춥니다.', false);
        await refresh({ forceFull: true });
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        updateControlButtons(latestOverview?.launcher || {});
      }
    }

    async function forceStopCurrentJob() {
      if (viewerMode) {
        setLaunchMessage('읽기 전용 공유 화면에서는 강제 중단을 할 수 없습니다.', true);
        return;
      }
      const confirmed = window.confirm('현재 진행 중인 filekey 작업을 즉시 강제 중단할까요? 현재 작업은 나중에 재시작 시 처음부터 다시 시도되며, 다음 큐 자동 시작은 멈춥니다.');
      if (!confirmed) {
        return;
      }

      const button = document.getElementById('forceStopButton');
      button.disabled = true;
      button.textContent = '강제 중단 중...';

      try {
        const response = await fetch('/api/force-stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || '강제 중단에 실패했습니다.');
        }
        setLaunchMessage(data.message || '현재 작업을 강제 중단했습니다.', false);
        await refresh({ forceFull: true });
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        updateControlButtons(latestOverview?.launcher || {});
      }
    }

    async function resetWorkspace() {
      if (viewerMode) {
        setLaunchMessage('읽기 전용 공유 화면에서는 초기화를 할 수 없습니다.', true);
        return;
      }
      const confirmed = window.confirm('현재 작업과 다운로드를 중단하고 처음부터 다시 시작할까요? 누적 prepared 데이터, 모델, 메트릭, 완료 로그, 대기열이 모두 삭제됩니다.');
      if (!confirmed) {
        return;
      }

      const button = document.getElementById('resetButton');
      button.disabled = true;
      button.textContent = '초기화 중...';

      try {
        const response = await fetch('/api/reset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || '초기화에 실패했습니다.');
        }
        setLaunchMessage(data.message || '학습 워크스페이스를 초기화했습니다.', false);
        await refresh({ forceFull: true });
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        button.disabled = false;
        button.textContent = '처음부터 다시 시작';
      }
    }

    async function removeQueuedJob(jobId, filekey) {
      if (viewerMode) {
        setLaunchMessage('읽기 전용 공유 화면에서는 대기열을 수정할 수 없습니다.', true);
        return;
      }
      const confirmed = window.confirm(`대기 중인 filekey ${filekey || ''} 작업을 큐에서 삭제할까요?`);
      if (!confirmed) {
        return;
      }

      try {
        const response = await fetch('/api/remove-queued-job', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_id: jobId }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || '대기열 삭제에 실패했습니다.');
        }
        setLaunchMessage(data.message || '선택한 대기열 작업을 삭제했습니다.', false);
        await refresh({ forceFull: true });
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      }
    }

    function runRenderStep(label, errors, fn) {
      try {
        fn();
      } catch (error) {
        console.error(`dashboard render failed: ${label}`, error, latestOverview);
        errors.push(`${label}: ${error?.message || String(error)}`);
      }
    }

    const OVERVIEW_FULL_REFRESH_INTERVAL_MS = 5000;
    let lastFullOverviewAt = 0;

    function isEmptyPayload(value) {
      if (value === null || value === undefined) {
        return true;
      }
      if (Array.isArray(value)) {
        return value.length === 0;
      }
      if (typeof value === 'object') {
        return Object.keys(value).length === 0;
      }
      return false;
    }

    function mergeLiteOverview(previous, incoming, isLite) {
      if (!isLite || !previous) {
        return incoming;
      }
      const merged = { ...previous, ...incoming };
      [
        'dataset',
        'guideline_quality',
        'prepared_pose_items',
        'logs',
        'metrics',
        'skip_report',
        'cumulative_skip_report',
        'diagnostics',
        'insights',
      ].forEach((key) => {
        if (isEmptyPayload(incoming?.[key]) && previous?.[key] !== undefined) {
          merged[key] = previous[key];
        }
      });
      merged.training_progress = {
        ...(previous.training_progress || {}),
        ...(incoming.training_progress || {}),
      };
      merged.launcher = {
        ...(previous.launcher || {}),
        ...(incoming.launcher || {}),
      };
      return merged;
    }

    async function refresh(options = {}) {
      let data;
      if (pendingInitialOverview) {
        data = pendingInitialOverview;
        pendingInitialOverview = null;
        lastFullOverviewAt = Date.now();
      } else {
        try {
          const now = Date.now();
          const forceFull = Boolean(options?.forceFull);
          const shouldFetchFull =
            forceFull ||
            !latestOverview ||
            !lastRenderSucceeded ||
            (now - lastFullOverviewAt) >= OVERVIEW_FULL_REFRESH_INTERVAL_MS;
          const response = await fetch(shouldFetchFull ? '/api/overview' : '/api/overview?lite=1', { cache: 'no-store' });
          if (!response.ok) {
            let detail = `대시보드 상태를 불러오지 못했습니다. (${response.status})`;
            try {
              const errorPayload = await response.json();
              detail = errorPayload?.detail || errorPayload?.message || detail;
            } catch (parseError) {
              // ignore response parse error
            }
            setLaunchMessage(detail, true);
            return;
          }
          const incoming = await response.json();
          if (shouldFetchFull) {
            lastFullOverviewAt = Date.now();
          }
          data = mergeLiteOverview(latestOverview, incoming, !shouldFetchFull);
        } catch (error) {
          setLaunchMessage(error?.message || '대시보드 상태 요청에 실패했습니다.', true);
          return;
        }
      }
      if (
        lastRenderSucceeded &&
        latestOverview &&
        latestOverview.overview_revision &&
        data.overview_revision &&
        latestOverview.overview_revision === data.overview_revision
      ) {
        return;
      }
      latestOverview = data;
      const pipeline = data.pipeline_status || {};
      const progress = data.training_progress || {};
      const metrics = data.metrics || {};
      const launcher = data.launcher || {};
      const queueProgress = data.queue_progress || {};
      const logs = data.logs || {};
      const currentJobProgress = data.current_job_progress || {};
      const continualState = data.continual_state || {};
      const eta = data.eta || {};
      const gpu = data.gpu || {};
      const skipReport = data.skip_report || {};
      const cumulativeSkipReport = data.cumulative_skip_report || {};
      const stageTimings = pipeline?.stage_timings ? { by_stage: pipeline.stage_timings } : { by_stage: {} };
      const totalDurationSeconds = pipeline?.total_duration_seconds ?? null;

      const renderErrors = [];

      runRenderStep('overview summary', renderErrors, () => {
        const stateEl = document.getElementById('pipelineState');
        const displayState = launcher.state || pipeline.state || 'unknown';
        if (stateEl) {
          stateEl.textContent = formatLauncherState(displayState);
          stateEl.className = `status-pill ${toneClass(displayState)}`;
        }

        const datasetKey = data.aihub?.datasetkey ?? '-';
        setText('datasetKeyChip', datasetKey);
        const datasetKeyInput = document.getElementById('datasetKeyInput');
        if (datasetKeyInput && datasetKey !== '-' && !datasetKeyInput.value.trim()) {
          datasetKeyInput.value = datasetKey;
        }
        setText('workspaceChip', data.workspace_name || '-');
        setText('launcherState', formatLauncherState(launcher.state || 'idle'));
        setText('currentFilekey', formatJob(launcher.current_job));
        setText('currentFilekeyDetails', formatCurrentJobFilekeys(data.current_job_filekeys || {}));
        setText(
          'currentDatasetkey',
          launcher.current_job?.datasetkey || formatDatasetkeys(launcher.pending_jobs || [])
        );
        setText('pendingFilekeys', formatFilekeys((launcher.pending_jobs || []).map((job) => job.filekey)));
        setText('completedJobs', formatCompletedJobs(launcher.completed_jobs || []));
        setText('autoStartState', launcher.auto_start_enabled === false ? '꺼짐' : '켜짐');
        setText('launcherLogPath', launcher.log_path || '-');
        updateControlButtons(launcher);
        renderQueuedJobs(launcher.pending_jobs || []);
      });

      runRenderStep('pipeline status', renderErrors, () => {
        setText('currentStage', pipeline.stage || '-');
        setText('currentMessage', pipeline.message || '-');
        setText('etaText', eta.label || '-');
        setText(
          'etaMeta',
          eta.seconds_remaining !== null && eta.seconds_remaining !== undefined
            ? '현재 filekey 기준 예상 남은 시간'
            : '진행률이 쌓이면 계산합니다.'
        );
        setText('epochProgress', `${progress.epochs_completed ?? 0} / ${progress.epochs_total ?? 0}`);
        setText('bestF1', `best macro F1: ${progress.best_val_macro_f1 ?? '-'}`);
      });

      runRenderStep('dataset and gpu summary', renderErrors, () => {
        const rawTotal = data.dataset?.raw?.total ?? 0;
        const preparedTotal =
          (data.dataset?.prepared_train?.total ?? 0) +
          (data.dataset?.prepared_val?.total ?? 0) +
          (data.dataset?.prepared_test?.total ?? 0);
        setText('datasetTotals', `${rawTotal} / ${preparedTotal}`);
        setText('datasetSummary', 'raw videos / prepared pose samples');
        setText('artifactState', data.artifacts?.has_model ? 'ready' : 'pending');
        setText('workspaceDir', data.workspace_dir || '-');
        setText('gpuUsageText', formatGpuUsage(gpu));
        setText('gpuUsageMeta', formatGpuMeta(gpu));
        setText('gpuVramText', formatGpuVram(gpu));
        setText('gpuVramMeta', formatGpuVramMeta(gpu));
        setText('queueProgressText', `${queueProgress.completed ?? 0} / ${queueProgress.total ?? 0}`);
        setText(
          'queueProgressMeta',
          `완료 ${queueProgress.completed ?? 0} / 실패 ${queueProgress.failed ?? 0} / 대기 ${queueProgress.pending ?? 0}`
        );
        setWidth(
          'queueProgressFill',
          `${Math.max(0, Math.min(100, Math.round((queueProgress.ratio ?? 0) * 100)))}%`
        );
        renderIssueVideos(skipReport, cumulativeSkipReport);
      });

      runRenderStep('latest training snapshot', renderErrors, () => {
        if (progress.latest) {
          setText('latestEpoch', `Epoch ${progress.latest.epoch}`);
          setText(
            'latestMetrics',
            `train loss ${progress.latest.train_loss} / val acc ${progress.latest.val_accuracy} / val f1 ${progress.latest.val_macro_f1}`
          );
          setText('latestLoss', `train ${progress.latest.train_loss} / val ${progress.latest.val_loss}`);
          setText('latestLearningRate', `lr ${progress.latest.learning_rate ?? '-'}`);
        } else {
          setText('latestEpoch', '-');
          setText('latestMetrics', '-');
          setText('latestLoss', '-');
          setText('latestLearningRate', '-');
        }
      });

      runRenderStep('training device summary', renderErrors, () => {
        setText('resumeState', progress.resumed_from_checkpoint ? '이전 모델 이어학습' : '새 학습');
        setText('sampleCounts', `train ${progress.train_samples ?? 0} / val ${progress.val_samples ?? 0}`);
        setText('gpuDeviceText', formatGpuDevice(gpu));
        setText('gpuMemoryText', formatGpuMemory(gpu));
        setText('trainingDeviceText', formatTrainingDevice(progress, gpu));
        setText('trainingDeviceMeta', formatTrainingDeviceMeta(progress));
        setText('stageTimingValue', formatStageTimingValue(stageTimings));
        setText('stageTimingCopy', formatStageTimingMeta(stageTimings, totalDurationSeconds));
        setText('currentStageTimingText', formatStageTimingValue(stageTimings));
        setText(
          'currentStageTimingMeta',
          formatActiveStageElapsed(pipeline) || formatStageTimingMeta(stageTimings, totalDurationSeconds)
        );
        setText('updatedAt', pipeline.updated_at || progress.updated_at || '-');
        setText('configPath', launcher.runtime_config_path || data.config_path || '-');
      });

      runRenderStep('charts and dataset tables', renderErrors, () => {
        renderChart(progress.history || []);
        renderLossChart(progress.history || []);
        renderLearningRateChart(progress.history || []);
        renderLossGapChart(progress.history || []);
        renderMetricInsights(progress, metrics);
        renderDatasetTable(data.dataset || {}, data.distribution_sources || {});
        renderCurrentJobProgress(currentJobProgress, data.current_dataset || {}, continualState, progress);
      });

      runRenderStep('validation metrics', renderErrors, () => {
        const metricLabels = metrics.labels || progress.labels || [];
        const finalValidation = metrics.final_validation || progress.final_validation || {};
        renderPerClassMetrics(
          metricLabels,
          finalValidation.per_class || [],
          finalValidation.confusion_matrix || []
        );
        renderConfusionMatrix(metricLabels, finalValidation.confusion_matrix || []);
      });

      runRenderStep('logs and completed jobs', renderErrors, () => {
        renderCompletedLogs(launcher.completed_jobs || [], queueProgress);
        renderLogPanels(logs, launcher);
      });

      lastRenderSucceeded = renderErrors.length === 0;
      if (renderErrors.length) {
        document.documentElement.classList.remove('dashboard-hydrated');
        setLaunchMessage(`렌더링 오류: ${renderErrors[0]}`, true);
      } else {
        document.documentElement.classList.add('dashboard-hydrated');
        setLaunchMessage(
          launcher.message || '여기에서 시작 결과와 최근 실행 메시지를 확인할 수 있습니다.',
          launcher.state === 'error'
        );
      }
    }

    let latestOverview = null;
    let lastRenderSucceeded = false;
    let pendingInitialOverview = null;
    try {
      const initialOverviewNode = document.getElementById('initialOverviewData');
      pendingInitialOverview = initialOverviewNode?.textContent
        ? JSON.parse(initialOverviewNode.textContent)
        : null;
    } catch (error) {
      console.error('failed to parse initial overview', error);
      pendingInitialOverview = null;
    }

    try {
      loadSavedDatasetKey();
    } catch (error) {
      console.error('loadSavedDatasetKey failed', error);
    }
    try {
      loadSavedApiKey();
    } catch (error) {
      console.error('loadSavedApiKey failed', error);
    }
    document.getElementById('datasetKeyInput').addEventListener('change', saveDatasetKey);
    document.getElementById('apiKeyInput').addEventListener('change', saveApiKey);
    document.getElementById('startButton').addEventListener('click', startTraining);
    document.getElementById('reextractButton').addEventListener('click', reextractSelectedFilekeys);
    document.getElementById('stopButton').addEventListener('click', pauseQueue);
    document.getElementById('forceStopButton').addEventListener('click', forceStopCurrentJob);
    document.getElementById('resetButton').addEventListener('click', resetWorkspace);
    document.getElementById('queuedJobList')?.addEventListener('click', (event) => {
      const button = event.target.closest('.queued-remove-button');
      if (!button) {
        return;
      }
      removeQueuedJob(button.dataset.jobId || '', button.dataset.filekey || '');
    });
    applyViewerMode();
    refresh({ forceFull: true });
    setInterval(() => refresh(), 1000);
  </script>
  <script>
    (function () {
      function setBootBanner(message) {
        try {
          var banner = document.getElementById('bootErrorBanner');
          if (!banner) {
            return;
          }
          banner.textContent = message;
          banner.style.display = 'block';
        } catch (error) {}
      }

      function supportsDashboardScript() {
        try {
          new Function("var probe = function (value) { return (value?.count ?? 0) + 1; }; return probe({ count: 1 });");
          return true;
        } catch (error) {
          return false;
        }
      }

      var sourceNode = document.getElementById('dashboardAppSource');
      if (!sourceNode) {
        return;
      }

      if (!supportsDashboardScript()) {
        document.documentElement.classList.remove('dashboard-hydrated');
        setBootBanner(
          '브라우저 호환 모드로 서버 렌더링 결과만 표시합니다. 이 브라우저에서는 대시보드 상호작용 스크립트를 실행할 수 없습니다.'
        );
        return;
      }

      try {
        var runtimeScript = document.createElement('script');
        runtimeScript.type = 'text/javascript';
        runtimeScript.text = sourceNode.text || sourceNode.textContent || '';
        document.body.appendChild(runtimeScript);
      } catch (error) {
        document.documentElement.classList.remove('dashboard-hydrated');
        setBootBanner(
          '브라우저 스크립트 오류: ' + (error && error.message ? error.message : String(error || 'unknown error'))
        );
      }
    })();
  </script>
</body>
</html>"""
        return (
            template
            .replace(
                "__SERVER_SNAPSHOT__",
                build_server_snapshot_html(initial_overview) + build_server_fallback_sections_html(initial_overview),
            )
            .replace("__INITIAL_OVERVIEW_JSON__", serialize_initial_overview(initial_overview))
        )

    def build_dashboard_redirect(message: str, level: str = "good") -> RedirectResponse:
        response = RedirectResponse(url="/", status_code=303, headers=NO_CACHE_HEADERS)
        response.set_cookie(
            NOTICE_COOKIE_NAME,
            quote(str(message), safe=""),
            max_age=NOTICE_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            NOTICE_LEVEL_COOKIE_NAME,
            str(level or "info"),
            max_age=NOTICE_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    def build_dashboard_action_response(
        request: Request | None,
        *,
        message: str,
        level: str,
        ok: bool = True,
        status_code: int = 200,
        payload: dict | None = None,
    ):
        if wants_json_response(request):
            body = {
                "ok": bool(ok),
                "message": str(message),
                "level": str(level or "info"),
            }
            if isinstance(payload, dict):
                body.update(payload)
            return JSONResponse(body, status_code=status_code, headers=NO_CACHE_HEADERS)
        return build_dashboard_redirect(str(message), str(level or "info"))

    async def read_form_payload(request: Request) -> dict:
        payload: dict[str, object] = {}

        try:
            form = await request.form()
        except Exception:
            form = None

        if form is not None:
            for key, value in form.multi_items():
                if hasattr(value, "filename"):
                    continue
                if key in payload:
                    existing = payload[key]
                    if isinstance(existing, list):
                        existing.append(value)
                    else:
                        payload[key] = [existing, value]
                else:
                    payload[key] = value
            if payload:
                return payload

        body_text = (await request.body()).decode("utf-8", errors="replace")
        parsed = parse_qs(body_text, keep_blank_values=True)
        for key, values in parsed.items():
            if len(values) == 1:
                payload[key] = values[0]
            else:
                payload[key] = values
        return payload

    @app.post("/actions/notifications")
    async def update_notifications(request: Request):
        payload = await read_form_payload(request)
        enabled_value = payload.get("enabled")
        if isinstance(enabled_value, list):
            enabled_value = enabled_value[0] if enabled_value else ""
        notify_on = payload.get("notify_on")
        settings = normalize_notification_settings(
            {
                "enabled": enabled_value in {"1", "true", "on", "yes", True},
                "provider": payload.get("provider") or "ntfy",
                "ntfy_server": payload.get("ntfy_server") or NOTIFICATION_DEFAULT_NTFY_SERVER,
                "ntfy_topic": payload.get("ntfy_topic") or "",
                "notify_on": notify_on if isinstance(notify_on, list) else [notify_on] if notify_on else [],
            }
        )
        mode = str(payload.get("mode") or "save").strip().lower()
        with state_lock:
            save_notification_settings(settings)
        if mode == "test":
            if not settings.get("enabled"):
                return build_dashboard_action_response(
                    request,
                    message="푸시 알림이 꺼져 있습니다. 켠 다음 테스트해 주세요.",
                    level="warn",
                    ok=False,
                    status_code=400,
                )
            try:
                send_ntfy_notification(
                    settings,
                    title="detectWarning 테스트 알림",
                    message="대시보드 푸시 알림 연결이 정상입니다.",
                    priority="default",
                )
                settings["last_sent_at"] = current_timestamp()
                settings["last_error"] = ""
                with state_lock:
                    save_notification_settings(settings)
                return build_dashboard_action_response(
                    request,
                    message="아이폰으로 테스트 푸시를 보냈습니다.",
                    level="good",
                    payload={"notification_settings": notification_settings_public(settings)},
                )
            except Exception as exc:
                settings["last_error"] = str(exc)
                with state_lock:
                    save_notification_settings(settings)
                return build_dashboard_action_response(
                    request,
                    message=f"테스트 푸시 전송 실패: {exc}",
                    level="danger",
                    ok=False,
                    status_code=502,
                    payload={"notification_settings": notification_settings_public(settings)},
                )
        return build_dashboard_action_response(
            request,
            message="푸시 알림 설정을 저장했습니다.",
            level="good",
            payload={"notification_settings": notification_settings_public(settings)},
        )

    def fetch_aihub_file_tree_for_dashboard(datasetkey: str, api_key: str = "") -> tuple[dict | list, str]:
        shell_config = config.get("aihub_shell", {})
        if not isinstance(shell_config, dict):
            shell_config = {}
        errors: list[str] = []

        try:
            return fetch_aihub_file_tree(datasetkey=datasetkey), "public_api"
        except Exception as exc:
            errors.append(f"public API: {exc}")

        shell_api_key = optional_aihub_api_key(shell_config, api_key)
        if not shell_api_key:
            errors.append("aihubshell fallback: API 키가 없어 건너뜀")
        else:
            try:
                shell_path = resolve_aihub_shell_path(shell_config)
                merged_output = fetch_aihub_file_tree_via_shell(
                    shell_path=shell_path,
                    api_key=shell_api_key,
                    datasetkey=datasetkey,
                )
                payload_text = extract_json_payload(merged_output)
                if payload_text:
                    return json.loads(payload_text), "aihubshell"
                listing_entries = parse_aihub_file_tree_listing(merged_output)
                if listing_entries:
                    return listing_entries, "aihubshell"
                errors.append("aihubshell fallback: 파일 목록 응답을 해석하지 못함")
            except Exception as exc:
                errors.append(f"aihubshell fallback: {exc}")

        raise RuntimeError(
            "AIHub 파일 목록 조회에 실패했습니다.\n"
            f"- datasetkey: {datasetkey}\n"
            + "\n".join(f"- {message}" for message in errors)
        )

    def collect_aihub_filekey_job_marks(datasetkey: str, *, refresh_process: bool = True) -> dict[str, dict]:
        normalized_datasetkey = str(datasetkey or "").strip()
        marks: dict[str, dict] = {}

        def dataset_matches(job: dict) -> bool:
            job_datasetkey = str(job.get("datasetkey") or "").strip()
            return not job_datasetkey or not normalized_datasetkey or job_datasetkey == normalized_datasetkey

        def remember(job: dict, status: str, priority: int) -> None:
            if not isinstance(job, dict) or not dataset_matches(job):
                return
            filekey = str(job.get("filekey") or "").strip()
            if not filekey:
                return
            current = marks.get(filekey)
            if current and int(current.get("priority") or 0) > priority:
                return
            marks[filekey] = {
                "status": status,
                "priority": priority,
                "finished_at": job.get("finished_at"),
                "started_at": job.get("started_at"),
                "log_path": job.get("log_path"),
                "job_id": job.get("job_id"),
            }

        with state_lock:
            if refresh_process:
                update_process_state()
            current_job = launcher_state.get("current_job")
            pending_jobs = launcher_state.get("queued_jobs", [])
            completed_jobs = launcher_state.get("completed_jobs", [])
            if isinstance(current_job, dict):
                remember(snapshot_job(current_job) or current_job, "running", 40)
            if isinstance(pending_jobs, list):
                for job in pending_jobs:
                    remember(snapshot_job(job) or job, "queued", 30)
            if isinstance(completed_jobs, list):
                for job in completed_jobs:
                    job_state = str(job.get("state") or "").strip().lower()
                    if job_state in AIHUB_TRAINED_JOB_STATES:
                        remember(snapshot_job(job) or job, "trained", 20)
                    elif job_state in AIHUB_PREPARED_JOB_STATES:
                        remember(snapshot_job(job) or job, "prepared", 15)

        history_payload = read_json(launcher_history_path) or {}
        history_jobs = history_payload.get("completed_jobs", []) if isinstance(history_payload, dict) else []
        if isinstance(history_jobs, list):
            for job in history_jobs:
                if not isinstance(job, dict):
                    continue
                job_state = str(job.get("state") or "").strip().lower()
                if job_state in AIHUB_TRAINED_JOB_STATES:
                    remember(job, "trained", 20)
                elif job_state in AIHUB_PREPARED_JOB_STATES:
                    remember(job, "prepared", 15)

        for job in restore_completed_jobs_from_logs(job_logs_dir, runtime_config_dir, limit=1000):
            job_state = str(job.get("state") or "").strip().lower()
            if job_state in AIHUB_TRAINED_JOB_STATES:
                remember(job, "trained", 10)
            elif job_state in AIHUB_PREPARED_JOB_STATES:
                remember(job, "prepared", 8)

        return marks

    def apply_aihub_filekey_job_marks(
        entries: list[dict],
        datasetkey: str,
        *,
        refresh_process: bool = True,
    ) -> list[dict]:
        marks = collect_aihub_filekey_job_marks(datasetkey, refresh_process=refresh_process)
        marked_entries: list[dict] = []
        for entry in entries:
            item = dict(entry)
            filekey = str(item.get("filekey") or "").strip()
            mark = marks.get(filekey)
            item["base_status"] = item.get("status") or "unknown"
            if mark:
                mark_status = mark.get("status") or item["base_status"]
                item["status"] = mark_status
                item["trainable"] = False
                item["selectable"] = False
                item["job_state"] = mark_status
                item["job_id"] = mark.get("job_id")
                if mark_status == "trained":
                    item["trained_at"] = mark.get("finished_at")
                elif mark_status == "prepared":
                    item["prepared_at"] = mark.get("finished_at")
                item["started_at"] = mark.get("started_at")
                item["log_path"] = mark.get("log_path")
            else:
                item["selectable"] = item.get("status") in {"trainable", "unknown"}
            marked_entries.append(item)
        return marked_entries

    def build_aihub_filekey_lookup_for_dashboard(
        *,
        datasetkey: str,
        api_key: str = "",
        refresh_process: bool = True,
    ) -> dict:
        if not datasetkey:
            raise HTTPException(status_code=400, detail="datasetkey를 입력해 주세요.")

        now = time.time()
        with aihub_filekey_lookup_cache_lock:
            cached = aihub_filekey_lookup_cache.get(datasetkey)
            if (
                isinstance(cached, dict)
                and now - float(cached.get("timestamp") or 0.0) < AIHUB_FILEKEY_LOOKUP_CACHE_SECONDS
                and isinstance(cached.get("value"), dict)
            ):
                cached_value = cached["value"]
                entries = apply_aihub_filekey_job_marks(
                    list(cached_value.get("entries") or []),
                    datasetkey,
                    refresh_process=refresh_process,
                )
                return build_aihub_lookup_result(
                    datasetkey=datasetkey,
                    source=str(cached_value.get("source") or "cache"),
                    entries=entries,
                    cache_hit=True,
                    target_labels=get_target_labels(config),
                )

        try:
            raw_payload, source = fetch_aihub_file_tree_for_dashboard(datasetkey, api_key=api_key)
            collected_entries = collect_aihub_file_entries(raw_payload)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        dataset_config = config.get("dataset", {})
        if not isinstance(dataset_config, dict):
            dataset_config = {}
        label_mapping = dataset_config.get("label_mapping", {})
        if not isinstance(label_mapping, dict):
            label_mapping = {}
        excluded_source_labels = dataset_config.get("excluded_source_labels", [])
        if not isinstance(excluded_source_labels, list):
            excluded_source_labels = []

        entries = normalize_aihub_lookup_entries(
            collected_entries,
            label_mapping=label_mapping,
            excluded_source_labels=excluded_source_labels,
            target_labels=get_target_labels(config),
        )
        result = build_aihub_lookup_result(
            datasetkey=datasetkey,
            source=source,
            entries=apply_aihub_filekey_job_marks(
                entries,
                datasetkey,
                refresh_process=refresh_process,
            ),
            cache_hit=False,
            target_labels=get_target_labels(config),
        )
        with aihub_filekey_lookup_cache_lock:
            aihub_filekey_lookup_cache[datasetkey] = {
                "timestamp": now,
                "value": {
                    "source": source,
                    "entries": entries,
                },
            }
        return result

    def lookup_aihub_filekeys_request(payload: dict) -> dict:
        datasetkey = str(payload.get("datasetkey", "")).strip()
        api_key = str(payload.get("api_key", "")).strip()
        return build_aihub_filekey_lookup_for_dashboard(
            datasetkey=datasetkey,
            api_key=api_key,
            refresh_process=True,
        )

    def payload_bool(value) -> bool:
        return value is True or str(value).strip().lower() in {"1", "true", "on", "yes"}

    def queue_key(datasetkey: str, filekey: str) -> str:
        return f"{str(datasetkey or '').strip()}:{str(filekey or '').strip()}"

    def collect_active_queue_filekeys_locked(datasetkey: str, *, include_prepared: bool = True) -> set[str]:
        normalized_datasetkey = str(datasetkey or "").strip()
        filekeys: set[str] = set()
        current_job = launcher_state.get("current_job")
        if isinstance(current_job, dict):
            current_datasetkey = str(current_job.get("datasetkey") or "").strip()
            if current_datasetkey == normalized_datasetkey:
                current_filekey = str(current_job.get("filekey") or "").strip()
                if current_filekey:
                    filekeys.add(current_filekey)
        pending_jobs = launcher_state.get("queued_jobs", [])
        if isinstance(pending_jobs, list):
            for job in pending_jobs:
                if not isinstance(job, dict):
                    continue
                job_datasetkey = str(job.get("datasetkey") or "").strip()
                if job_datasetkey != normalized_datasetkey:
                    continue
                filekey = str(job.get("filekey") or "").strip()
                if filekey:
                    filekeys.add(filekey)
        completed_jobs = launcher_state.get("completed_jobs", [])
        terminal_blocking_states = set(AIHUB_TRAINED_JOB_STATES)
        if include_prepared:
            terminal_blocking_states |= AIHUB_PREPARED_JOB_STATES
        if isinstance(completed_jobs, list):
            for job in completed_jobs:
                if not isinstance(job, dict):
                    continue
                job_datasetkey = str(job.get("datasetkey") or "").strip()
                if job_datasetkey != normalized_datasetkey:
                    continue
                if str(job.get("state") or "").strip().lower() not in terminal_blocking_states:
                    continue
                filekey = str(job.get("filekey") or "").strip()
                if filekey:
                    filekeys.add(filekey)
        if include_prepared:
            filekeys.update(collect_manifest_filekeys_locked())
        return filekeys

    def collect_manifest_filekeys_locked() -> set[str]:
        filekeys: set[str] = set()
        for path_key in (
            "prepared_train",
            "prepared_val",
            "prepared_test",
            "guideline_prepared_train",
            "guideline_prepared_val",
            "guideline_prepared_test",
        ):
            manifest_path = paths.get(path_key)
            if not isinstance(manifest_path, Path) or not manifest_path.exists():
                continue
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        filekey = str(row.get("source_filekey") or row.get("filekey") or "").strip()
                        if filekey and filekey not in {"guideline_clips", "rgb_i3d_features"}:
                            filekeys.add(filekey)
            except OSError:
                continue
        return filekeys

    def collect_predownload_filekeys_locked(datasetkey: str) -> set[str]:
        normalized_datasetkey = str(datasetkey or "").strip()
        filekeys: set[str] = set()
        running = launcher_state.get("predownload_processes", {})
        if isinstance(running, dict):
            for item in running.values():
                if not isinstance(item, dict):
                    continue
                if str(item.get("datasetkey") or "").strip() != normalized_datasetkey:
                    continue
                filekey = str(item.get("filekey") or "").strip()
                if filekey:
                    filekeys.add(filekey)
        for key in ("predownload_completed",):
            items = launcher_state.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("datasetkey") or "").strip() != normalized_datasetkey:
                    continue
                filekey = str(item.get("filekey") or "").strip()
                if filekey:
                    filekeys.add(filekey)
        predownload_dir = paths.get("predownload_dir")
        if isinstance(predownload_dir, Path) and predownload_dir.exists():
            try:
                cache_dirs = [path for path in predownload_dir.iterdir() if path.is_dir()]
            except OSError:
                cache_dirs = []
            for cache_dir in cache_dirs:
                marker_path = cache_dir / ".predownload_complete.json"
                if not marker_path.exists():
                    continue
                marker = read_json(marker_path) or {}
                if isinstance(marker, dict):
                    marker_datasetkey = str(marker.get("datasetkey") or "").strip()
                    if marker_datasetkey and marker_datasetkey != normalized_datasetkey:
                        continue
                    marker_filekey = str(marker.get("filekey") or "").strip()
                else:
                    marker_filekey = ""
                if not marker_filekey:
                    marker_filekey = cache_dir.name[4:] if cache_dir.name.startswith("job_") else cache_dir.name
                marker_filekey = marker_filekey.strip()
                if marker_filekey:
                    filekeys.add(marker_filekey)
        return filekeys

    def collect_completed_predownload_cache_items_locked(datasetkey: str) -> list[dict]:
        normalized_datasetkey = str(datasetkey or "").strip()
        predownload_dir = paths.get("predownload_dir")
        if not isinstance(predownload_dir, Path) or not predownload_dir.exists():
            return []
        try:
            cache_dirs = [path for path in predownload_dir.iterdir() if path.is_dir()]
        except OSError:
            return []
        prepared_filekeys = collect_manifest_filekeys_locked()
        items: list[dict] = []
        for cache_dir in cache_dirs:
            marker_path = cache_dir / ".predownload_complete.json"
            if not marker_path.exists():
                continue
            marker = read_json(marker_path) or {}
            marker = marker if isinstance(marker, dict) else {}
            marker_datasetkey = str(marker.get("datasetkey") or "").strip()
            if marker_datasetkey and normalized_datasetkey and marker_datasetkey != normalized_datasetkey:
                continue
            marker_filekey = str(marker.get("filekey") or "").strip()
            if not marker_filekey:
                marker_filekey = cache_dir.name[4:] if cache_dir.name.startswith("job_") else cache_dir.name
            marker_filekey = marker_filekey.strip()
            if not marker_filekey:
                continue
            if marker_filekey in prepared_filekeys:
                continue
            try:
                payload_files = [path for path in cache_dir.iterdir() if path.name != marker_path.name]
            except OSError:
                payload_files = []
            if not payload_files:
                continue
            items.append(
                {
                    "job_id": f"disk_cache_{marker_filekey}",
                    "datasetkey": marker_datasetkey or normalized_datasetkey,
                    "filekey": marker_filekey,
                    "recommendation": {
                        "filekey": marker_filekey,
                        "recommendation_reason": "predownload_disk_cache",
                    },
                    "state": "completed",
                    "finished_at": marker.get("finished_at"),
                    "started_at": marker.get("started_at"),
                    "cache_dir": str(cache_dir),
                    "file_count": marker.get("file_count") or len(payload_files),
                    "total_bytes": marker.get("total_bytes"),
                }
            )
        items.sort(key=lambda item: str(item.get("finished_at") or ""), reverse=False)
        return items

    def collect_prepared_label_counts() -> dict[str, int]:
        counts: Counter[str] = Counter()
        for by_label in collect_prepared_label_counts_by_split().values():
            counts.update(by_label)
        return dict(counts)

    def collect_prepared_label_counts_by_split() -> dict[str, dict[str, int]]:
        split_counts: dict[str, dict[str, int]] = {}
        for split_name, path_key in (
            ("train", "prepared_train"),
            ("val", "prepared_val"),
            ("test", "prepared_test"),
        ):
            manifest_path = paths.get(path_key)
            if not manifest_path:
                continue
            summary = summarize_manifest(Path(manifest_path), label_field="target_label")
            by_label = summary.get("by_label") if isinstance(summary, dict) else {}
            if not isinstance(by_label, dict):
                continue
            normalized_counts = normalize_label_count_map(by_label)
            if normalized_counts:
                split_counts[split_name] = dict(normalized_counts)
        return split_counts

    def build_auto_recommendation_insights() -> dict:
        target_labels = get_target_labels(config)
        training_progress = normalize_metric_payload(
            read_json(paths["training_progress"]),
            target_labels=target_labels,
        )
        metrics = normalize_metric_payload(
            read_json(paths["artifacts_dir"] / "metrics.json"),
            target_labels=target_labels,
        )
        history = training_progress.get("history") or metrics.get("history") or []
        latest = training_progress.get("latest") or (
            history[-1] if isinstance(history, list) and history else {}
        )
        final_validation = (
            metrics.get("final_validation")
            or training_progress.get("final_validation")
            or {}
        )
        insight_metrics = {
            **(metrics if isinstance(metrics, dict) else {}),
            "labels": target_labels,
            "history": history if isinstance(history, list) else [],
            "latest": latest if isinstance(latest, dict) else {},
            "final_validation": final_validation if isinstance(final_validation, dict) else {},
            "best_epoch": metrics.get("best_epoch") or training_progress.get("best_epoch"),
            "best_val_macro_f1": metrics.get("best_val_macro_f1") or training_progress.get("best_val_macro_f1"),
            "best_validation": metrics.get("best_validation") or training_progress.get("best_validation") or {},
        }
        return interpret_training_results(
            insight_metrics,
            history=insight_metrics["history"],
            class_report=insight_metrics["final_validation"].get("per_class") or [],
            confusion_matrix=insight_metrics["final_validation"].get("confusion_matrix") or [],
            data_stats={
                "labels": target_labels,
                "skip_report": read_json(paths["current_skip_report"]) or {},
                "cumulative_skip_report": read_json(paths["cumulative_skip_report"]) or {},
                "train_distribution": training_progress.get("train_distribution") or metrics.get("train_distribution") or {},
                "val_distribution": training_progress.get("val_distribution") or metrics.get("val_distribution") or {},
            },
        )

    def count_auto_recommended_pending_jobs_locked(datasetkey: str) -> int:
        normalized_datasetkey = str(datasetkey or "").strip()
        pending_jobs = launcher_state.get("queued_jobs", [])
        if not isinstance(pending_jobs, list):
            return 0
        return sum(
            1
            for job in pending_jobs
            if isinstance(job, dict)
            and job.get("auto_recommended")
            and str(job.get("datasetkey") or "").strip() == normalized_datasetkey
        )

    def pop_completed_predownload_locked(datasetkey: str) -> dict | None:
        normalized_datasetkey = str(datasetkey or "").strip()
        active_filekeys = collect_active_queue_filekeys_locked(normalized_datasetkey, include_prepared=False)
        completed = launcher_state.get("predownload_completed", [])
        if not isinstance(completed, list):
            completed = []
        for index, item in enumerate(list(completed)):
            if not isinstance(item, dict):
                continue
            if str(item.get("datasetkey") or "").strip() != normalized_datasetkey:
                continue
            recommendation = (
                item.get("recommendation")
                if isinstance(item.get("recommendation"), dict)
                else {}
            )
            scope = recommendation.get("recommendation_scope") if isinstance(recommendation.get("recommendation_scope"), dict) else {}
            zip_group = (
                aihub_entry_zip_group(recommendation)
                or str(scope.get("zip_group") or "").strip().lower()
            )
            allowed_cache_groups = set(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS) | set(AIHUB_AUTO_RECOMMEND_FALLBACK_ZIP_GROUPS)
            if zip_group and zip_group not in allowed_cache_groups:
                completed.pop(index)
                continue
            if str(item.get("filekey") or "").strip() in active_filekeys:
                continue
            return completed.pop(index)
        for item in collect_completed_predownload_cache_items_locked(normalized_datasetkey):
            filekey = str(item.get("filekey") or "").strip()
            if filekey and filekey not in active_filekeys:
                return item
        return None

    def pop_failed_predownload_retry_locked(datasetkey: str, *, existing_filekeys: set[str]) -> dict | None:
        normalized_datasetkey = str(datasetkey or "").strip()
        failed = launcher_state.get("predownload_failed", [])
        if not isinstance(failed, list):
            return None
        for index, item in enumerate(list(failed)):
            if not isinstance(item, dict):
                continue
            if str(item.get("datasetkey") or "").strip() != normalized_datasetkey:
                continue
            filekey = str(item.get("filekey") or "").strip()
            if not filekey or filekey in existing_filekeys:
                continue
            retry_item = failed.pop(index)
            retry_item["retry_of"] = retry_item.get("job_id")
            retry_item["retry_count"] = int(retry_item.get("retry_count", 0) or 0) + 1
            return retry_item
        return None

    def find_incomplete_predownload_cache_retry_locked(datasetkey: str, *, existing_filekeys: set[str]) -> dict | None:
        predownload_dir = paths.get("predownload_dir")
        if not isinstance(predownload_dir, Path) or not predownload_dir.exists():
            return None
        candidates: list[Path] = []
        try:
            candidates = [path for path in predownload_dir.iterdir() if path.is_dir()]
        except OSError:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
        for cache_dir in candidates:
            if (cache_dir / ".predownload_complete.json").exists():
                continue
            filekey = cache_dir.name
            if filekey.startswith("job_"):
                filekey = filekey[4:]
            filekey = filekey.strip()
            if not filekey or filekey in existing_filekeys:
                continue
            try:
                has_partial_files = any(path.name != ".predownload_complete.json" for path in cache_dir.iterdir())
            except OSError:
                has_partial_files = False
            return {
                "job_id": f"incomplete_cache_{filekey}",
                "datasetkey": str(datasetkey or "").strip(),
                "filekey": filekey,
                "recommendation": {
                    "filekey": filekey,
                    "recommendation_reason": "incomplete_predownload_cache",
                },
                "retry_of": f"incomplete_cache_{filekey}",
                "retry_count": 1,
                "partial_cache_dir": str(cache_dir),
                "partial_files_detected": has_partial_files,
            }
        return None

    def dashboard_job_resume_identity(job: dict) -> str:
        job_kind = infer_dashboard_job_kind(job)
        stage = str(job.get("stage") or "").strip()
        if not stage:
            stage = "extract" if job_kind == "aihub" else "all"
        return "|".join(
            [
                str(job.get("datasetkey") or "").strip(),
                str(job.get("filekey") or "").strip(),
                job_kind,
                stage,
                str(job.get("source_filekey") or "").strip(),
            ]
        )

    def active_dashboard_job_identities_locked() -> set[str]:
        identities: set[str] = set()
        current_job = launcher_state.get("current_job")
        if isinstance(current_job, dict):
            identities.add(dashboard_job_resume_identity(current_job))
        pending_jobs = launcher_state.get("queued_jobs", [])
        if isinstance(pending_jobs, list):
            for job in pending_jobs:
                if isinstance(job, dict):
                    identities.add(dashboard_job_resume_identity(job))
        return identities

    def enqueue_followup_jobs_for_retry_locked(retry_job: dict, *, rgb_model: str) -> None:
        job_kind = infer_dashboard_job_kind(retry_job)
        pending_jobs = launcher_state.setdefault("queued_jobs", [])
        if not isinstance(pending_jobs, list):
            launcher_state["queued_jobs"] = []
            pending_jobs = launcher_state["queued_jobs"]

        source_filekey = str(retry_job.get("source_filekey") or retry_job.get("filekey") or "").strip()
        source_datasetkey = str(retry_job.get("source_datasetkey") or retry_job.get("datasetkey") or "").strip()
        followup_payload = {
            "source": "cumulative",
            "include_rgb": True,
            "start_train": False,
            "cleanup_after": True,
            "rgb_model": retry_job.get("rgb_model") or rgb_model or "i3d_r50",
            "device": retry_job.get("device") or "cuda",
            "source_filekey": source_filekey,
            "source_datasetkey": source_datasetkey,
        }
        if job_kind == "aihub":
            followups = build_guideline_dashboard_jobs(followup_payload)
        elif job_kind == "guideline":
            followups = build_guideline_dashboard_jobs({**followup_payload, "skip_guideline": True})
        elif job_kind == "rgb":
            followups = build_guideline_dashboard_jobs(
                {
                    **followup_payload,
                    "skip_guideline": True,
                    "include_rgb": False,
                    "start_train": False,
                    "cleanup_after": True,
                }
            )
        else:
            followups = []

        existing = active_dashboard_job_identities_locked()
        for followup in followups:
            identity = dashboard_job_resume_identity(followup)
            if identity in existing:
                continue
            pending_jobs.append(followup)
            existing.add(identity)

    def enqueue_missing_completed_extract_followups_locked(*, rgb_model: str) -> dict | None:
        completed_jobs = launcher_state.get("completed_jobs", [])
        if not isinstance(completed_jobs, list):
            return None
        pending_jobs = launcher_state.setdefault("queued_jobs", [])
        if not isinstance(pending_jobs, list):
            launcher_state["queued_jobs"] = []
            pending_jobs = launcher_state["queued_jobs"]
        if any(
            isinstance(job, dict)
            and infer_dashboard_job_kind(job) in {"guideline", "rgb", "cleanup", "train_guideline"}
            for job in pending_jobs
        ):
            return None
        current_job = launcher_state.get("current_job")
        if isinstance(current_job, dict) and infer_dashboard_job_kind(current_job) in {
            "guideline",
            "rgb",
            "cleanup",
            "train_guideline",
        }:
            return None
        if not guideline_manifests_are_stale():
            return None

        completed_followup_sources = {
            str(job.get("source_filekey") or "").strip()
            for job in completed_jobs
            if isinstance(job, dict)
            and infer_dashboard_job_kind(job) in {"cleanup", "rgb"}
            and str(job.get("state") or "").strip().lower() in {"completed", "completed_warning"}
        }
        for completed_job in completed_jobs:
            if not isinstance(completed_job, dict):
                continue
            if infer_dashboard_job_kind(completed_job) != "aihub":
                continue
            if str(completed_job.get("state") or "").strip().lower() not in {"completed", "completed_warning", "data_ready"}:
                continue
            source_filekey = str(completed_job.get("source_filekey") or completed_job.get("filekey") or "").strip()
            if not source_filekey or source_filekey in completed_followup_sources:
                continue
            source_datasetkey = str(completed_job.get("source_datasetkey") or completed_job.get("datasetkey") or "").strip()
            followups = build_guideline_dashboard_jobs(
                {
                    "source": "cumulative",
                    "include_rgb": True,
                    "start_train": False,
                    "cleanup_after": True,
                    "rgb_model": rgb_model or launcher_state.get("auto_rgb_model") or "i3d_r50",
                    "device": "cuda",
                    "source_filekey": source_filekey,
                    "source_datasetkey": source_datasetkey,
                }
            )
            pending_jobs.extend(followups)
            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = (
                f"filekey {source_filekey} 추출은 끝났지만 guideline/RGB 후속 산출물이 오래되어 "
                "새 filekey보다 먼저 후속 작업을 큐에 추가했습니다."
            )
            persist_launcher_history(launcher_history_path, launcher_state)
            return {"ok": True, "message": launcher_state["last_message"], "job": followups[0] if followups else None}
        return None

    def enqueue_interrupted_auto_extract_job_locked(datasetkey: str, api_key: str, *, rgb_model: str) -> dict | None:
        followup_result = enqueue_missing_completed_extract_followups_locked(rgb_model=rgb_model)
        if followup_result is not None:
            return followup_result
        completed_jobs = launcher_state.get("completed_jobs", [])
        if not isinstance(completed_jobs, list):
            return None
        active_identities = active_dashboard_job_identities_locked()
        resolved_identities: set[str] = set()
        for completed_job in completed_jobs:
            if not isinstance(completed_job, dict):
                continue
            identity = dashboard_job_resume_identity(completed_job)
            state = str(completed_job.get("state") or "").strip().lower()
            if state in {"completed", "completed_warning", "data_ready"}:
                resolved_identities.add(identity)
                continue
            if state not in {"aborted", "error"}:
                continue
            if identity in resolved_identities or identity in active_identities:
                continue
            retry_job = build_retry_job_from(completed_job)
            if infer_dashboard_job_kind(retry_job) == "aihub":
                try:
                    retry_count = int(retry_job.get("retry_count", 0) or 0)
                except (TypeError, ValueError):
                    retry_count = 0
                if retry_count >= 3:
                    resolved_identities.add(identity)
                    launcher_state["last_message"] = (
                        f"filekey {retry_job.get('filekey')} extraction already failed {retry_count} times; "
                        "skipping retry and selecting the next filekey."
                    )
                    continue
            if api_key and infer_dashboard_job_kind(retry_job) == "aihub":
                retry_job["api_key"] = api_key
            if datasetkey and not retry_job.get("datasetkey"):
                retry_job["datasetkey"] = datasetkey
            retry_job["auto_resume_interrupted"] = True
            if infer_dashboard_job_kind(retry_job) == "rgb":
                retry_job.setdefault("rgb_model", rgb_model or "i3d_r50")
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            if not isinstance(pending_jobs, list):
                launcher_state["queued_jobs"] = []
                pending_jobs = launcher_state["queued_jobs"]
            pending_jobs.append(retry_job)
            enqueue_followup_jobs_for_retry_locked(retry_job, rgb_model=rgb_model)
            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = (
                f"중단됐던 작업을 먼저 재개합니다: filekey {retry_job.get('filekey')} "
                f"({infer_dashboard_job_kind(retry_job)})."
            )
            persist_launcher_history(launcher_history_path, launcher_state)
            return {"ok": True, "message": launcher_state["last_message"], "job": retry_job}
        return None

    def append_predownload_log_line(item: dict, line: str) -> None:
        log_path = item.get("log_path") if isinstance(item, dict) else None
        if not log_path:
            return
        try:
            with Path(str(log_path)).open("a", encoding="utf-8") as handle:
                handle.write(line.rstrip() + "\n")
        except OSError:
            pass

    def update_predownload_processes_locked() -> None:
        running = launcher_state.get("predownload_processes", {})
        if not isinstance(running, dict):
            launcher_state["predownload_processes"] = {}
            return
        completed = launcher_state.setdefault("predownload_completed", [])
        failed = launcher_state.setdefault("predownload_failed", [])
        for job_id, item in list(running.items()):
            if not isinstance(item, dict):
                running.pop(job_id, None)
                continue
            process = item.get("process")
            if not isinstance(process, subprocess.Popen):
                running.pop(job_id, None)
                continue
            exit_code = process.poll()
            if exit_code is None:
                continue
            item["finished_at"] = current_timestamp()
            item["exit_code"] = exit_code
            item.pop("process", None)
            item["state"] = "completed" if exit_code == 0 else "error"
            if exit_code == 0:
                append_predownload_log_line(
                    item,
                    f"[predownload][completed] filekey={item.get('filekey')} exit_code=0 finished_at={item['finished_at']}",
                )
            else:
                append_predownload_log_line(
                    item,
                    f"[predownload][error] filekey={item.get('filekey')} exit_code={exit_code} finished_at={item['finished_at']}",
                )
            running.pop(job_id, None)
            if exit_code == 0 and isinstance(completed, list):
                item_filekey = str(item.get("filekey") or "").strip()
                if item_filekey:
                    completed[:] = [
                        existing
                        for existing in completed
                        if not (
                            isinstance(existing, dict)
                            and str(existing.get("datasetkey") or "").strip() == str(item.get("datasetkey") or "").strip()
                            and str(existing.get("filekey") or "").strip() == item_filekey
                        )
                    ]
                completed.append(item)
                del completed[:-PREDOWNLOAD_COMPLETED_HISTORY_LIMIT]
            elif isinstance(failed, list):
                failed.insert(0, item)
                del failed[PREDOWNLOAD_FAILED_HISTORY_LIMIT:]

    def active_aihub_download_slot_locked() -> int:
        current_job = launcher_state.get("current_job")
        if not isinstance(current_job, dict):
            return 0
        if str(current_job.get("job_kind") or "aihub").strip().lower() != "aihub":
            return 0
        if str(current_job.get("stage") or "").strip().lower() not in {"", "all", "extract"}:
            return 0
        return 1

    def predownload_free_disk_gb() -> float:
        try:
            usage = shutil.disk_usage(paths["workspace_dir"])
        except OSError:
            return 0.0
        return float(usage.free) / (1024 ** 3)

    def maybe_start_predownloads_locked() -> None:
        update_predownload_processes_locked()
        if not bool(launcher_state.get("predownload_enabled", True)):
            return
        if not bool(launcher_state.get("auto_extract_enabled", False)):
            return
        datasetkey = str(
            launcher_state.get("auto_enqueue_datasetkey")
            or config.get("aihub_shell", {}).get("datasetkey")
            or ""
        ).strip()
        api_key = str(launcher_state.get("auto_enqueue_api_key") or "").strip()
        if not datasetkey:
            return
        min_free_gb = max(float(launcher_state.get("predownload_min_free_gb") or 100), 0.0)
        if predownload_free_disk_gb() <= min_free_gb:
            launcher_state["predownload_pause_reason"] = (
                f"디스크 여유 공간이 {min_free_gb:g}GB 이하라 새 predownload를 잠시 멈췄습니다."
            )
            return
        launcher_state["predownload_pause_reason"] = ""
        running = launcher_state.setdefault("predownload_processes", {})
        if not isinstance(running, dict):
            launcher_state["predownload_processes"] = {}
            running = launcher_state["predownload_processes"]
        max_parallel = max(int(launcher_state.get("predownload_max_parallel") or 2), 0)
        ready_cache_filekeys = {
            str(item.get("filekey") or "").strip()
            for item in collect_completed_predownload_cache_items_locked(datasetkey)
            if str(item.get("filekey") or "").strip()
        }
        ready_cache_filekeys.update(
            str(item.get("filekey") or "").strip()
            for item in launcher_state.get("predownload_completed", [])
            if isinstance(item, dict) and str(item.get("datasetkey") or "").strip() == datasetkey
        )
        ready_cache_filekeys.discard("")
        if len(ready_cache_filekeys) >= max_parallel:
            launcher_state["predownload_pause_reason"] = (
                f"준비된 predownload cache가 {len(ready_cache_filekeys)}개라 새 다운로드를 멈췄습니다. "
                f"목표 보관 개수는 {max_parallel}개입니다."
            )
            return
        desired_new_downloads = max_parallel - len(ready_cache_filekeys) - len(running)
        # Predownload slots are independent from the active pipeline job:
        # allow 1 processing filekey plus max_parallel background downloads.
        concurrent_capacity = max_parallel - len(running)
        available_slots = min(desired_new_downloads, concurrent_capacity)
        if available_slots <= 0:
            return
        try:
            lookup = build_aihub_filekey_lookup_for_dashboard(datasetkey=datasetkey, api_key=api_key, refresh_process=False)
        except Exception:
            return
        lookup_entries = lookup.get("entries") or []
        entry_by_filekey = {
            str(entry.get("filekey") or "").strip(): entry
            for entry in lookup_entries
            if isinstance(entry, dict) and str(entry.get("filekey") or "").strip()
        }
        prepared_split_label_counts = collect_prepared_label_counts_by_split()
        prepared_label_counts: Counter[str] = Counter()
        for by_label in prepared_split_label_counts.values():
            prepared_label_counts.update(normalize_label_count_map(by_label))
        for _ in range(available_slots):
            existing_filekeys = collect_active_queue_filekeys_locked(datasetkey, include_prepared=True)
            existing_filekeys |= collect_predownload_filekeys_locked(datasetkey)
            retry_predownload = pop_failed_predownload_retry_locked(datasetkey, existing_filekeys=existing_filekeys)
            if not isinstance(retry_predownload, dict):
                retry_predownload = find_incomplete_predownload_cache_retry_locked(
                    datasetkey,
                    existing_filekeys=existing_filekeys,
                )
            if isinstance(retry_predownload, dict):
                recommended = (
                    retry_predownload.get("recommendation")
                    if isinstance(retry_predownload.get("recommendation"), dict)
                    else {}
                )
                retry_filekey = str(retry_predownload.get("filekey") or "").strip()
                recommended = {**entry_by_filekey.get(retry_filekey, {}), **recommended, "filekey": retry_filekey}
            else:
                recommended = find_next_trainable_aihub_entry(
                    lookup_entries,
                    existing_filekeys=existing_filekeys,
                    prepared_label_counts=dict(prepared_label_counts),
                    prepared_split_label_counts=prepared_split_label_counts,
                    allowed_zip_groups=AIHUB_AUTO_RECOMMEND_ZIP_GROUPS,
                    insights=build_auto_recommendation_insights(),
                    recommendation_policy=build_auto_recommendation_policy(),
                )
                if not isinstance(recommended, dict) or not recommended.get("filekey"):
                    return
            if predownload_free_disk_gb() <= min_free_gb:
                return
            filekey = str(recommended["filekey"])
            zip_group = aihub_entry_zip_group(recommended)
            allowed_predownload_groups = set(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS) | set(AIHUB_AUTO_RECOMMEND_FALLBACK_ZIP_GROUPS)
            if zip_group not in allowed_predownload_groups:
                launcher_state["predownload_pause_reason"] = (
                    f"허용되지 않은 추천 filekey {filekey}({zip_group or 'unknown'})를 차단했습니다."
                )
                return
            job_id = f"predownload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{re.sub(r'[^0-9A-Za-z_-]+', '_', filekey)}"
            runtime_config = json.loads(json.dumps(config))
            runtime_paths = runtime_config.setdefault("paths", {})
            runtime_paths["workspace_dir"] = str(paths["workspace_dir"])
            runtime_config["dataset_source"] = "aihub_shell"
            runtime_shell = runtime_config.setdefault("aihub_shell", {})
            runtime_shell["datasetkey"] = datasetkey
            runtime_shell["filekey"] = filekey
            if api_key:
                runtime_shell["api_key"] = api_key
                runtime_shell["api_key_env"] = ""
            runtime_config_path = runtime_config_dir / f"{job_id}.json"
            write_json_atomic(runtime_config_path, runtime_config)
            log_path = job_logs_dir / f"{job_id}.log"
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(Path(__file__).resolve().with_name("predownload_aihub_filekey.py")),
                "--config",
                str(runtime_config_path),
            ]
            active_pipeline_job = launcher_state.get("current_job") if isinstance(launcher_state.get("current_job"), dict) else {}
            active_pipeline_stage = str(active_pipeline_job.get("stage") or active_pipeline_job.get("job_kind") or "-")
            active_pipeline_filekey = str(active_pipeline_job.get("filekey") or "-")
            running_after_start = len(running) + 1
            active_download_slot = active_aihub_download_slot_locked()
            recommendation_reason = str(
                recommended.get("recommendation_reason")
                or recommended.get("reason")
                or recommended.get("diagnosis_reason")
                or "-"
            )
            recommendation_scope = aihub_entry_zip_group(recommended) or "-"
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write(f"[predownload][start] datasetkey={datasetkey} filekey={filekey}\n")
                if isinstance(retry_predownload, dict):
                    log_handle.write(
                        "[predownload][resume] "
                        f"retry_of={retry_predownload.get('retry_of') or retry_predownload.get('job_id') or '-'} "
                        f"retry_count={retry_predownload.get('retry_count') or 1}\n"
                    )
                log_handle.write(
                    "[predownload][parallel] "
                    f"running_predownload={running_after_start}/{max_parallel} "
                    f"active_pipeline_download_slot={active_download_slot} "
                    "slot_policy=active_pipeline_excluded "
                    f"free_disk_gb={predownload_free_disk_gb():.1f}\n"
                )
                log_handle.write(
                    "[predownload][alongside] "
                    f"pipeline_filekey={active_pipeline_filekey} pipeline_stage={active_pipeline_stage}\n"
                )
                log_handle.write(
                    "[predownload][recommendation] "
                    f"target_label={recommended.get('target_label') or '-'} "
                    f"zip_group={recommendation_scope} reason={recommendation_reason}\n"
                )
                log_handle.write(f"[predownload][config] runtime config: {runtime_config_path}\n")
                log_handle.flush()
                child_env = os.environ.copy()
                child_env["PYTHONUTF8"] = "1"
                child_env["PYTHONIOENCODING"] = "utf-8"
                child_env["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    command,
                    cwd=str(project_root),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=child_env,
                    start_new_session=(os.name != "nt"),
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
                )
            running[job_id] = {
                "job_id": job_id,
                "datasetkey": datasetkey,
                "filekey": filekey,
                "recommendation": recommended,
                "state": "running",
                "started_at": current_timestamp(),
                "runtime_config_path": str(runtime_config_path),
                "log_path": str(log_path),
                "process": process,
                "retry_of": retry_predownload.get("retry_of") if isinstance(retry_predownload, dict) else None,
                "retry_count": retry_predownload.get("retry_count") if isinstance(retry_predownload, dict) else None,
            }

    def optimize_auto_recommended_queue_locked(
        datasetkey: str,
        entries: list[dict],
        insights: dict,
        *,
        prepared_label_counts: dict[str, int] | None = None,
        prepared_split_label_counts: dict[str, dict[str, int]] | None = None,
    ) -> dict:
        pending_jobs = launcher_state.get("queued_jobs", [])
        if not isinstance(pending_jobs, list) or not pending_jobs:
            return {"removed": [], "retained_auto_count": 0, "targeted_available": False}

        normalized_datasetkey = str(datasetkey or "").strip()
        recommendation_policy = build_auto_recommendation_policy()
        normalized_policy = normalize_auto_recommendation_policy(recommendation_policy)
        allowed_groups = normalize_recommendation_zip_groups(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS)
        diagnosis_priorities = build_diagnosis_label_priorities(insights)
        entry_by_filekey = {
            str(entry.get("filekey") or "").strip(): entry
            for entry in entries
            if isinstance(entry, dict) and str(entry.get("filekey") or "").strip()
        }
        current_best = find_next_trainable_aihub_entry(
            entries,
            existing_filekeys=set(),
            prepared_label_counts=prepared_label_counts,
            prepared_split_label_counts=prepared_split_label_counts,
            allowed_zip_groups=AIHUB_AUTO_RECOMMEND_ZIP_GROUPS,
            insights=insights,
            recommendation_policy=recommendation_policy,
        )
        current_best_label = recommendation_label(current_best) if isinstance(current_best, dict) else ""
        current_best_score = current_best.get("recommendation_score") if isinstance(current_best, dict) else {}
        try:
            targeted_available = float(
                current_best_score.get("insight_priority", 0.0) if isinstance(current_best_score, dict) else 0.0
            ) > 0
        except (TypeError, ValueError):
            targeted_available = False

        retained_jobs: list = []
        removed_jobs: list[dict] = []
        retained_auto: list[tuple[float, int, dict]] = []
        seen_auto_filekeys: set[str] = set()
        for order, job in enumerate(pending_jobs):
            if not isinstance(job, dict):
                retained_jobs.append(job)
                continue
            if not job.get("auto_recommended") or str(job.get("datasetkey") or "").strip() != normalized_datasetkey:
                retained_jobs.append(job)
                continue

            filekey = str(job.get("filekey") or "").strip()
            entry = entry_by_filekey.get(filekey) or job
            label = str(job.get("target_label") or recommendation_label(entry))
            priority_payload = diagnosis_priority_for_label(label, diagnosis_priorities)
            insight_priority = float(priority_payload.get("priority", 0.0) or 0.0)
            remove_reason = ""
            if not filekey:
                remove_reason = "missing_filekey"
            elif filekey in seen_auto_filekeys:
                remove_reason = "duplicate_auto_recommendation"
            elif current_best is None:
                remove_reason = "recommendation_unavailable"
            elif not recommendation_zip_scope_state(
                entry,
                allowed_groups=allowed_groups,
                policy=normalized_policy,
                label=label,
                primary_candidate_labels=set(),
                diagnosis_priority=priority_payload,
                label_counts=Counter(),
                strict_fallback=False,
            ).get("allowed"):
                remove_reason = "outside_scope_changed"
            elif current_best_label and normalize_recommendation_label(label) != normalize_recommendation_label(current_best_label):
                remove_reason = "recommendation_balance_changed"
            elif targeted_available and insight_priority <= 0:
                remove_reason = "diagnosis_priority_changed"

            if remove_reason:
                removed = snapshot_job(job)
                removed["remove_reason"] = remove_reason
                removed_jobs.append(removed)
                continue

            seen_auto_filekeys.add(filekey)
            job["recommendation_reason"] = "diagnosis_guided" if insight_priority > 0 else job.get("recommendation_reason", "class_balance")
            job["recommendation_scope"] = {
                "allowed_zip_groups": list(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS),
                "zip_group": aihub_entry_zip_group(entry),
                "primary_zip_groups": list(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS),
                "fallback_zip_groups": list(AIHUB_AUTO_RECOMMEND_FALLBACK_ZIP_GROUPS),
                "excluded_zip_groups": list(AIHUB_AUTO_RECOMMEND_EXCLUDED_ZIP_GROUPS),
            }
            score = job.setdefault("recommendation_score", {})
            if isinstance(score, dict):
                score["target_label"] = label
                score["insight_priority"] = round(insight_priority, 6)
                score["insight_reasons"] = list(priority_payload.get("reasons") or [])[:4]
                score["insight_level"] = priority_payload.get("level") or "good"
            retained_jobs.append(job)
            retained_auto.append((insight_priority, order, job))

        if len(retained_auto) > AIHUB_AUTO_RECOMMEND_MAX_PENDING:
            keep_ids = {
                id(job)
                for _, _, job in sorted(retained_auto, key=lambda item: (-item[0], item[1]))[:AIHUB_AUTO_RECOMMEND_MAX_PENDING]
            }
            trimmed_jobs = []
            for job in retained_jobs:
                if (
                    isinstance(job, dict)
                    and job.get("auto_recommended")
                    and str(job.get("datasetkey") or "").strip() == normalized_datasetkey
                    and id(job) not in keep_ids
                ):
                    removed = snapshot_job(job)
                    removed["remove_reason"] = "auto_recommendation_queue_limit"
                    removed_jobs.append(removed)
                    continue
                trimmed_jobs.append(job)
            retained_jobs = trimmed_jobs

        launcher_state["queued_jobs"] = retained_jobs
        return {
            "removed": removed_jobs,
            "retained_auto_count": count_auto_recommended_pending_jobs_locked(datasetkey),
            "targeted_available": targeted_available,
        }

    def enable_auto_enqueue_locked(datasetkey: str, api_key: str) -> None:
        launcher_state["auto_enqueue_enabled"] = True
        launcher_state["auto_extract_enabled"] = False
        launcher_state["auto_enqueue_datasetkey"] = str(datasetkey or "").strip()
        launcher_state["auto_enqueue_api_key"] = str(api_key or "").strip()
        launcher_state["auto_start_enabled"] = True

    def enable_auto_extract_locked(datasetkey: str, api_key: str, rgb_model: str = "i3d_r50") -> None:
        launcher_state["auto_extract_enabled"] = True
        launcher_state["auto_enqueue_enabled"] = False
        launcher_state["auto_enqueue_datasetkey"] = str(datasetkey or "").strip()
        launcher_state["auto_enqueue_api_key"] = str(api_key or "").strip()
        launcher_state["auto_rgb_model"] = str(rgb_model or "i3d_r50").strip() or "i3d_r50"
        launcher_state["auto_start_enabled"] = True

    def disable_auto_enqueue_locked(message: str) -> None:
        launcher_state["auto_enqueue_enabled"] = False
        launcher_state["auto_extract_enabled"] = False
        launcher_state["auto_enqueue_datasetkey"] = None
        launcher_state["auto_enqueue_api_key"] = ""
        launcher_state["auto_rgb_model"] = "i3d_r50"
        launcher_state["last_message"] = message

    def enqueue_next_extract_recommended_job_locked() -> dict:
        datasetkey = str(launcher_state.get("auto_enqueue_datasetkey") or config.get("aihub_shell", {}).get("datasetkey") or "").strip()
        api_key = str(launcher_state.get("auto_enqueue_api_key") or "").strip()
        if not datasetkey:
            message = "자동 추출을 위한 datasetkey가 없습니다."
            disable_auto_enqueue_locked(message)
            return {"ok": False, "message": message, "job": None}
        completed_predownload = pop_completed_predownload_locked(datasetkey)
        if isinstance(completed_predownload, dict):
            filekey = str(completed_predownload.get("filekey") or "").strip()
            recommended = completed_predownload.get("recommendation") if isinstance(completed_predownload.get("recommendation"), dict) else {}
            job = build_job(filekey, datasetkey=datasetkey, api_key=api_key, stage="extract")
            job["auto_recommended"] = True
            job["target_label"] = recommended.get("target_label")
            job["recommendation_reason"] = recommended.get("recommendation_reason") or "predownload_cache"
            job["notification_step"] = "filekey_pose_preprocess"
            job["source_filekey"] = filekey
            job["source_datasetkey"] = datasetkey
            job["predownload_cache"] = True
            launcher_state.setdefault("queued_jobs", []).append(job)
            launcher_state.setdefault("queued_jobs", []).extend(
                build_guideline_dashboard_jobs(
                    {
                        "source": "cumulative",
                        "include_rgb": True,
                        "start_train": False,
                        "cleanup_after": True,
                        "rgb_model": launcher_state.get("auto_rgb_model") or "i3d_r50",
                        "device": "cuda",
                        "source_filekey": filekey,
                        "source_datasetkey": datasetkey,
                    }
                )
            )
            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = (
                f"미리 다운로드된 filekey {filekey}를 전처리 큐에 추가했습니다."
            )
            return {"ok": True, "message": launcher_state["last_message"], "job": job}
        try:
            lookup = build_aihub_filekey_lookup_for_dashboard(datasetkey=datasetkey, api_key=api_key, refresh_process=False)
        except Exception as exc:
            message = f"분포 맞춤 자동추출 filekey 조회에 실패했습니다: {exc}"
            disable_auto_enqueue_locked(message)
            launcher_state["last_state"] = "error"
            return {"ok": False, "message": message, "job": None}
        prepared_split_label_counts = collect_prepared_label_counts_by_split()
        prepared_label_counts: Counter[str] = Counter()
        for by_label in prepared_split_label_counts.values():
            prepared_label_counts.update(normalize_label_count_map(by_label))
        recommended = find_next_trainable_aihub_entry(
            lookup.get("entries") or [],
            existing_filekeys=collect_active_queue_filekeys_locked(datasetkey, include_prepared=True),
            prepared_label_counts=dict(prepared_label_counts),
            prepared_split_label_counts=prepared_split_label_counts,
            allowed_zip_groups=AIHUB_AUTO_RECOMMEND_ZIP_GROUPS,
            insights=build_auto_recommendation_insights(),
            recommendation_policy=build_auto_recommendation_policy(),
        )
        if not isinstance(recommended, dict) or not recommended.get("filekey"):
            message = "분포 기준에 맞는 자동 추출 추천 filekey가 없습니다."
            if performance_plan_is_active():
                return enqueue_performance_plan_training_locked("no_more_recommended_filekeys")
            disable_auto_enqueue_locked(message)
            return {"ok": False, "message": message, "job": None}
        filekey = str(recommended["filekey"])
        job = build_job(filekey, datasetkey=datasetkey, api_key=api_key, stage="extract")
        job["auto_recommended"] = True
        job["target_label"] = recommended.get("target_label")
        job["recommendation_reason"] = recommended.get("recommendation_reason") or "class_balance"
        job["notification_step"] = "filekey_pose_preprocess"
        job["source_filekey"] = filekey
        job["source_datasetkey"] = datasetkey
        launcher_state.setdefault("queued_jobs", []).append(job)
        launcher_state["last_state"] = "queued"
        launcher_state["last_message"] = f"분포 기준 자동 추출 filekey {filekey}를 큐에 추가했습니다."
        launcher_state.setdefault("queued_jobs", []).extend(
            build_guideline_dashboard_jobs(
                {
                    "source": "cumulative",
                    "include_rgb": True,
                    "start_train": False,
                    "cleanup_after": True,
                    "rgb_model": launcher_state.get("auto_rgb_model") or "i3d_r50",
                    "device": "cuda",
                    "source_filekey": filekey,
                    "source_datasetkey": datasetkey,
                }
            )
        )
        launcher_state["last_message"] = (
            f"분포 기준 자동 추출 filekey {filekey}를 큐에 추가했고, RGB+Pose feature 추출 후 원본 정리까지 이어서 실행합니다."
        )
        return {"ok": True, "message": launcher_state["last_message"], "job": job}

    def enqueue_next_recommended_job_locked() -> dict:
        datasetkey = str(
            launcher_state.get("auto_enqueue_datasetkey")
            or config.get("aihub_shell", {}).get("datasetkey")
            or ""
        ).strip()
        api_key = str(launcher_state.get("auto_enqueue_api_key") or "").strip()
        has_guideline_features = paths["guideline_prepared_train"].exists() and paths["guideline_prepared_val"].exists()
        has_prepared_pose = paths["prepared_train"].exists() and paths["prepared_val"].exists()
        if not has_guideline_features and not has_prepared_pose:
            message = "추출된 prepared 데이터가 없어 학습을 시작할 수 없습니다."
            disable_auto_enqueue_locked(message)
            launcher_state["last_state"] = launcher_state.get("last_state") or "completed"
            return {"ok": False, "message": message, "job": None}
        pending_jobs = launcher_state.setdefault("queued_jobs", [])
        job = build_job(
            "guideline_features" if has_guideline_features else "prepared_pose",
            datasetkey=datasetkey or "prepared",
            api_key=api_key,
            stage="train",
        )
        job["auto_recommended"] = True
        job["recommendation_reason"] = "prepared_data_training"
        if has_guideline_features:
            job["job_kind"] = "train_guideline"
            job["display_name"] = "추출된 RGB+Pose feature 학습"
            job["running_message"] = "누적 guideline/RGB+Pose feature로 학습/auto-tune/ensemble을 실행 중입니다."
            job["success_message"] = "추출된 feature 기반 학습이 완료되었습니다."
        if isinstance(pending_jobs, list):
            pending_jobs.append(job)
        launcher_state["auto_enqueue_enabled"] = False
        launcher_state["last_state"] = "queued"
        launcher_state["last_message"] = "추출된 prepared 데이터 학습을 큐에 추가했습니다."
        return {"ok": True, "message": launcher_state["last_message"], "job": job, "queue_plan": {}}
        if not datasetkey:
            message = "추천 자동 시작을 할 datasetkey가 없습니다."
            disable_auto_enqueue_locked(message)
            return {"ok": False, "message": message, "job": None}

        try:
            lookup = build_aihub_filekey_lookup_for_dashboard(
                datasetkey=datasetkey,
                api_key=api_key,
                refresh_process=False,
            )
        except Exception as exc:
            message = f"추천 filekey 조회에 실패해 자동 큐 추가를 멈췄습니다: {exc}"
            disable_auto_enqueue_locked(message)
            launcher_state["last_state"] = "error"
            return {"ok": False, "message": message, "job": None}

        prepared_split_label_counts = collect_prepared_label_counts_by_split()
        prepared_label_counts: Counter[str] = Counter()
        for by_label in prepared_split_label_counts.values():
            prepared_label_counts.update(normalize_label_count_map(by_label))
        insights = build_auto_recommendation_insights()
        queue_plan = optimize_auto_recommended_queue_locked(
            datasetkey,
            lookup.get("entries") or [],
            insights,
            prepared_label_counts=dict(prepared_label_counts),
            prepared_split_label_counts=prepared_split_label_counts,
        )
        if count_auto_recommended_pending_jobs_locked(datasetkey) >= AIHUB_AUTO_RECOMMEND_MAX_PENDING:
            message = "진단 기준으로 기존 자동 추천 큐를 유지합니다. 현재 대기 중인 자동 추천 작업이 먼저 실행됩니다."
            if queue_plan.get("removed"):
                message = f"{len(queue_plan.get('removed') or [])}개 자동 추천 큐를 진단 기준에 맞게 정리했고, 기존 자동 추천 큐를 유지합니다."
            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = message
            pending_jobs = launcher_state.get("queued_jobs", [])
            kept_job = next(
                (
                    snapshot_job(job)
                    for job in pending_jobs
                    if isinstance(job, dict)
                    and job.get("auto_recommended")
                    and str(job.get("datasetkey") or "").strip() == datasetkey
                ),
                None,
            )
            return {"ok": True, "message": message, "job": kept_job, "queue_plan": queue_plan}

        existing_filekeys = collect_active_queue_filekeys_locked(datasetkey, include_prepared=False)
        recommended = find_next_trainable_aihub_entry(
            lookup.get("entries") or [],
            existing_filekeys=existing_filekeys,
            prepared_label_counts=dict(prepared_label_counts),
            prepared_split_label_counts=prepared_split_label_counts,
            allowed_zip_groups=AIHUB_AUTO_RECOMMEND_ZIP_GROUPS,
            insights=insights,
            recommendation_policy=build_auto_recommendation_policy(),
        )
        if not recommended:
            message = "outside(outsidedoor)에서 더 이상 학습 가능한 추천 filekey가 없어 자동 큐 추가를 멈췄습니다."
            message = (
                "Auto recommendation stopped because no filekey matched the current policy. "
                "Policy: outside first, no inside_croki, and no class above 1.5x the smallest class."
            )
            disable_auto_enqueue_locked(message)
            launcher_state["last_state"] = launcher_state.get("last_state") or "completed"
            return {"ok": False, "message": message, "job": None}

        filekey = str(recommended.get("filekey") or "").strip()
        pending_jobs = launcher_state.setdefault("queued_jobs", [])
        job = build_job(filekey, datasetkey=datasetkey, api_key=api_key)
        job["auto_recommended"] = True
        job["source_label"] = recommended.get("source_label") or recommended.get("matched_source_label")
        job["target_label"] = recommended.get("target_label")
        job["recommendation_reason"] = recommended.get("recommendation_reason") or "class_balance"
        job["recommendation_scope"] = recommended.get("recommendation_scope") or {
            "allowed_zip_groups": list(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS),
            "zip_group": aihub_entry_zip_group(recommended),
        }
        job["recommendation_score"] = recommended.get("recommendation_score") or {}
        if isinstance(pending_jobs, list):
            pending_jobs.append(job)
        launcher_state["last_state"] = "queued"
        score = job["recommendation_score"] if isinstance(job.get("recommendation_score"), dict) else {}
        target_label = score.get("target_label") or job.get("target_label") or "-"
        insight_priority = float(score.get("insight_priority", 0.0) or 0.0)
        insight_reasons = [str(reason) for reason in score.get("insight_reasons") or [] if str(reason).strip()]
        reason_text = ", ".join(insight_reasons[:2]) if insight_reasons else f"부족한 클래스 {target_label}"
        if queue_plan.get("removed"):
            reason_text = f"{len(queue_plan.get('removed') or [])}개 자동 추천 큐 정리 후 {reason_text}"
        launcher_state["last_message"] = (
            f"outside 추천 filekey {filekey}를 자동으로 큐에 추가했습니다. "
            f"기준: outsidedoor 범위 / 부족한 클래스 {target_label}"
        )
        launcher_state["last_message"] = (
            f"outside 추천 filekey {filekey}를 자동으로 큐에 추가했습니다. "
            f"기준: {'진단 우선순위' if insight_priority > 0 else '클래스 균형'} / {reason_text}"
        )
        scope = job["recommendation_scope"] if isinstance(job.get("recommendation_scope"), dict) else {}
        zip_group = scope.get("zip_group") or aihub_entry_zip_group(recommended) or "-"
        reason_text = ", ".join(insight_reasons[:2]) if insight_reasons else f"부족한 클래스 {target_label}"
        if queue_plan.get("removed"):
            reason_text = f"{len(queue_plan.get('removed') or [])}개 자동 추천 큐 정리 후 {reason_text}"
        launcher_state["last_message"] = (
            f"추천 filekey {filekey}를 자동으로 큐에 추가했습니다. "
            f"범위: {zip_group} / 기준: {'진단 우선순위' if insight_priority > 0 else '클래스 균형'} / {reason_text}"
        )
        launcher_state["last_message"] = (
            f"Auto recommendation queued filekey {filekey}. "
            f"scope={zip_group}, basis={'diagnosis' if insight_priority > 0 else 'class_balance'}, target={target_label}"
        )
        return {"ok": True, "message": launcher_state["last_message"], "job": job, "queue_plan": queue_plan}

    def start_training_request(payload: dict) -> dict:
        if str(config.get("dataset_source", "")).strip().lower() != "aihub_shell":
            raise HTTPException(
                status_code=400,
                detail="이 대시보드에서 직접 filekey 실행은 dataset_source가 aihub_shell일 때만 지원합니다.",
            )

        try:
            filekeys = parse_filekeys(payload.get("filekeys", ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        datasetkey = str(payload.get("datasetkey", "")).strip()
        api_key = str(payload.get("api_key", "")).strip()
        resume_only_raw = payload.get("resume_only", False)
        resume_only = payload_bool(resume_only_raw)
        auto_enqueue_next = payload_bool(payload.get("auto_enqueue_next", False))
        auto_extract_next = payload_bool(payload.get("auto_extract_next", False))
        feature_extract = payload_bool(payload.get("feature_extract", False))
        reextract_filekey_only = payload_bool(payload.get("reextract_filekey_only", False))
        excluded_training_labels = excluded_training_labels_from_payload(payload)
        requested_stage = str(payload.get("stage") or "all").strip().lower()
        if requested_stage not in {"all", "extract"}:
            requested_stage = "all"
        if auto_enqueue_next:
            filekeys = []
            requested_stage = "train"
        if auto_extract_next:
            filekeys = []
            requested_stage = "extract"
        if feature_extract:
            requested_stage = "extract"
        if reextract_filekey_only:
            requested_stage = "extract"
            feature_extract = True
        if not filekeys and not resume_only and not auto_enqueue_next and not auto_extract_next:
            raise HTTPException(status_code=400, detail="filekey를 하나 이상 입력해 주세요.")

        if not datasetkey:
            datasetkey = str(config.get("aihub_shell", {}).get("datasetkey", "")).strip()
        if not resume_only and not auto_enqueue_next and datasetkey in (None, ""):
            raise HTTPException(
                status_code=400,
                detail="datasetkey를 입력해 주세요. datasetkey는 filekey가 아니라 AIHub 데이터셋 키입니다.",
            )

        with state_lock:
            update_process_state()
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            current_job = launcher_state.get("current_job")
            if resume_only:
                if auto_enqueue_next:
                    enable_auto_enqueue_locked(datasetkey, api_key)
                    enqueue_next_recommended_job_locked()
                    pending_jobs = launcher_state.setdefault("queued_jobs", [])
                has_pending = isinstance(pending_jobs, list) and len(pending_jobs) > 0
                if not current_job and not has_pending:
                    raise HTTPException(status_code=409, detail="재개하거나 추천할 학습 가능 filekey가 없습니다.")
                launcher_state["auto_start_enabled"] = True
                if current_job:
                    launcher_state["last_state"] = "running"
                    launcher_state["last_message"] = (
                        "현재 작업 완료 후 다음 큐 자동 시작을 다시 허용합니다."
                    )
                else:
                    launcher_state["last_state"] = "queued"
                    launcher_state["last_message"] = "대기열 자동 시작을 다시 켰습니다. 다음 큐를 이어서 시작합니다."
                    if isinstance(pending_jobs, list) and pending_jobs:
                        next_job = pending_jobs.pop(0)
                        start_pipeline_for_job(next_job)
                return {
                    "ok": True,
                    "message": (
                        "추천 자동 시작을 켰습니다."
                        if auto_enqueue_next
                        else "대기열 자동 시작을 재개했습니다."
                    ),
                    "launcher": get_launcher_status(),
                }

            existing_keys = set()
            if isinstance(current_job, dict) and current_job.get("filekey"):
                existing_keys.add(f"{current_job.get('datasetkey', '')}:{current_job['filekey']}:{current_job.get('stage', 'all')}")
            if isinstance(pending_jobs, list):
                existing_keys.update(
                    f"{job.get('datasetkey', '')}:{job.get('filekey')}:{job.get('stage', 'all')}"
                    for job in pending_jobs
                    if isinstance(job, dict) and job.get("filekey")
                )

            appended = []
            skipped = []
            if auto_enqueue_next:
                enable_auto_enqueue_locked(datasetkey, api_key)
            if auto_extract_next:
                enable_auto_extract_locked(datasetkey, api_key, str(payload.get("rgb_model") or "i3d_r50"))

            if filekeys:
                for filekey in filekeys:
                    unique_key = f"{queue_key(datasetkey, filekey)}:{requested_stage}"
                    if unique_key in existing_keys:
                        skipped.append(filekey)
                        continue
                    job = build_job(filekey, datasetkey=datasetkey, api_key=api_key, stage=requested_stage)
                    if excluded_training_labels and requested_stage in {"all", "train"}:
                        job["excluded_training_labels"] = excluded_training_labels
                    if requested_stage == "extract":
                        job["notification_step"] = "filekey_pose_preprocess"
                        job["source_filekey"] = filekey
                        job["source_datasetkey"] = datasetkey
                    if reextract_filekey_only:
                        job["reextract_filekey_only"] = True
                        job["recommendation_reason"] = "manual_filekey_reextract"
                        job["running_message"] = f"filekey {filekey} 기존 산출물을 정리한 뒤 단일 재추출을 실행 중입니다."
                    if isinstance(pending_jobs, list):
                        pending_jobs.append(job)
                        if feature_extract and requested_stage == "extract":
                            pending_jobs.extend(
                                build_guideline_dashboard_jobs(
                                    {
                                        "source": "cumulative",
                                        "include_rgb": True,
                                        "start_train": False,
                                        "cleanup_after": True,
                                        "rgb_model": payload.get("rgb_model") or "i3d_r50",
                                        "device": payload.get("device") or "cuda",
                                        "source_filekey": filekey,
                                        "source_datasetkey": datasetkey,
                                        "excluded_training_labels": excluded_training_labels,
                                    }
                                )
                            )
                    appended.append(filekey)
                    existing_keys.add(unique_key)
            elif auto_extract_next:
                recommendation = enqueue_interrupted_auto_extract_job_locked(
                    datasetkey,
                    api_key,
                    rgb_model=str(payload.get("rgb_model") or "i3d_r50"),
                )
                if recommendation is None:
                    recommendation = enqueue_next_extract_recommended_job_locked()
                if not recommendation.get("ok"):
                    raise HTTPException(status_code=409, detail=str(recommendation.get("message") or "자동추출 추천 filekey를 찾지 못했습니다."))
                job = recommendation.get("job")
                if isinstance(job, dict) and job.get("filekey"):
                    appended.append(str(job["filekey"]))
            elif auto_enqueue_next:
                recommendation = enqueue_next_recommended_job_locked()
                job = recommendation.get("job")
                if isinstance(job, dict) and job.get("filekey"):
                    appended.append(str(job["filekey"]))

            if not appended:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "추천할 학습 가능 filekey가 없습니다."
                        if auto_enqueue_next
                        else "입력한 filekey가 모두 현재 작업 또는 대기열에 이미 있습니다."
                    ),
                )

            launcher_state["auto_start_enabled"] = True
            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = (
                f"datasetkey {datasetkey} 에 대해 {len(appended)}개 filekey를 대기열에 추가했습니다."
            )
            if not current_job and isinstance(pending_jobs, list) and pending_jobs:
                next_job = pending_jobs.pop(0)
                start_pipeline_for_job(next_job)

        return {
            "ok": True,
            "message": (
                f"datasetkey {datasetkey} | filekey {', '.join(appended)} 를 대기열에 추가했습니다."
                + (f" 중복으로 건너뜀: {', '.join(skipped)}" if skipped else "")
            ),
            "launcher": get_launcher_status(),
        }

    def build_guideline_dashboard_jobs(payload: dict) -> list[dict]:
        include_rgb = payload_bool(payload.get("include_rgb", True))
        start_train = payload_bool(payload.get("start_train", True))
        cleanup_after = payload_bool(payload.get("cleanup_after", False))
        skip_guideline = payload_bool(payload.get("skip_guideline", False))
        excluded_training_labels = excluded_training_labels_from_payload(payload)
        source = str(payload.get("source") or "cumulative").strip() or "cumulative"
        rgb_model = str(payload.get("rgb_model") or "i3d_r50").strip() or "i3d_r50"
        source_filekey = str(payload.get("source_filekey") or "").strip()
        source_datasetkey = str(payload.get("source_datasetkey") or payload.get("datasetkey") or "prepared").strip()
        if not source_filekey:
            source_filekey = infer_guideline_pipeline_source_filekeys(
                paths,
                use_guideline=bool(skip_guideline),
            )
        jobs: list[dict] = []

        if not skip_guideline:
            guideline_job = build_job("guideline_clips", datasetkey="prepared", stage="guideline")
            guideline_job.update(
                {
                    "job_kind": "guideline",
                    "notification_step": "guideline_clips",
                    "source_filekey": source_filekey,
                    "source_datasetkey": source_datasetkey,
                    "display_name": "가이드라인 clip/normal 생성",
                    "source": source,
                    "running_message": "XML/event/context 기준 clip manifest와 normal 샘플을 생성 중입니다.",
                    "success_message": "가이드라인 clip/normal manifest 생성이 완료되었습니다.",
                }
            )
            jobs.append(guideline_job)

        if include_rgb:
            rgb_job = build_job("rgb_i3d_features", datasetkey="prepared", stage="rgb")
            rgb_job.update(
                {
                    "job_kind": "rgb",
                    "notification_step": "rgb_i3d_features",
                    "source_filekey": source_filekey,
                    "source_datasetkey": source_datasetkey,
                    "display_name": "RGB/I3D feature 추출",
                    "rgb_model": rgb_model,
                    "device": str(payload.get("device") or "cuda"),
                    "running_message": f"{rgb_model} RGB feature를 clip별로 추출 중입니다.",
                    "success_message": "RGB/I3D feature 추출이 완료되었습니다.",
                }
            )
            jobs.append(rgb_job)

        if start_train:
            train_job = build_job("guideline_train", datasetkey="prepared", stage="train")
            train_job.update(
                {
                    "job_kind": "train_guideline",
                    "notification_step": "guideline_train",
                    "source_filekey": source_filekey,
                    "source_datasetkey": source_datasetkey,
                    "display_name": "가이드라인 fusion 학습",
                    "running_message": "가이드라인 clip manifest로 학습/auto-tune/ensemble을 실행 중입니다.",
                    "success_message": "가이드라인 기반 학습이 완료되었습니다.",
                }
            )
            if excluded_training_labels:
                train_job["excluded_training_labels"] = excluded_training_labels
            jobs.append(train_job)

        if cleanup_after:
            cleanup_job = build_job("cleanup_raw_after_features", datasetkey="prepared", stage="cleanup")
            cleanup_job.update(
                {
                    "job_kind": "cleanup",
                    "notification_step": "cleanup_raw_after_job",
                    "source_filekey": source_filekey,
                    "source_datasetkey": source_datasetkey,
                    "display_name": "원본/임시 파일 정리",
                    "running_message": "누적 feature/manifest는 유지하고 원본/임시 다운로드 데이터를 정리 중입니다.",
                    "success_message": "원본/임시 다운로드 데이터 정리가 완료되었습니다.",
                }
            )
            jobs.append(cleanup_job)

        return jobs

    def start_guideline_pipeline_request(payload: dict) -> dict:
        jobs = build_guideline_dashboard_jobs(payload)
        if not jobs:
            raise HTTPException(status_code=400, detail="실행할 작업이 없습니다.")
        with state_lock:
            update_process_state()
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            if not isinstance(pending_jobs, list):
                launcher_state["queued_jobs"] = []
                pending_jobs = launcher_state["queued_jobs"]
            pending_jobs.extend(jobs)
            launcher_state["auto_start_enabled"] = True
            launcher_state["auto_enqueue_enabled"] = False
            launcher_state["auto_extract_enabled"] = False
            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = f"가이드라인 개선 파이프라인 {len(jobs)}단계를 대기열에 추가했습니다."
            if launcher_state.get("process") is None and pending_jobs:
                next_job = pending_jobs.pop(0)
                start_pipeline_for_job(next_job)
        return {
            "ok": True,
            "message": f"가이드라인 개선 파이프라인 {len(jobs)}단계를 시작했습니다.",
            "launcher": get_launcher_status(),
        }

    def start_performance_plan_request(payload: dict) -> dict:
        plan_request = normalize_performance_plan_request(payload, config)
        datasetkey = plan_request["datasetkey"]
        api_key = str(payload.get("api_key") or "").strip()
        if not datasetkey:
            raise HTTPException(status_code=400, detail="성능 플랜을 시작하려면 datasetkey가 필요합니다.")

        with state_lock:
            update_process_state()
            enable_auto_extract_locked(datasetkey, api_key, plan_request["rgb_model"])
            launcher_state["performance_plan"] = {
                "enabled": True,
                "status": "extracting",
                "started_at": current_timestamp(),
                **plan_request,
                "final_training_queued": False,
            }
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = (
                f"최고 성능 자동 플랜을 시작했습니다. {plan_request['days']:g}일 동안 분포 기준 추출을 누적하고 "
                f"마지막에 {plan_request['max_trials']} trial auto-tune/ensemble 학습을 실행합니다."
            )
            if launcher_state.get("process") is None and isinstance(pending_jobs, list) and not pending_jobs:
                resumed = enqueue_interrupted_auto_extract_job_locked(
                    datasetkey,
                    api_key,
                    rgb_model=plan_request["rgb_model"],
                )
                if resumed is None:
                    enqueue_next_extract_recommended_job_locked()
                pending_jobs = launcher_state.setdefault("queued_jobs", [])
                if launcher_state.get("process") is None and isinstance(pending_jobs, list) and pending_jobs:
                    next_job = pending_jobs.pop(0)
                    start_pipeline_for_job(next_job)
        return {"ok": True, "message": launcher_state["last_message"], "launcher": get_launcher_status()}

    def clear_guideline_outputs() -> None:
        workspace_dir = paths.get("workspace_dir")
        manifest_dir = paths.get("manifests_dir")
        targets = [
            paths.get("guideline_prepared_train"),
            paths.get("guideline_prepared_val"),
            paths.get("guideline_prepared_test"),
            manifest_dir / "guideline_prepare_summary.json" if isinstance(manifest_dir, Path) else None,
            workspace_dir / "prepared_pose_guideline" if isinstance(workspace_dir, Path) else None,
            workspace_dir / "rgb_clip_features" if isinstance(workspace_dir, Path) else None,
        ]
        for target in targets:
            if isinstance(target, Path) and target.exists():
                remove_transient_path_with_retries(target, recreate_dir=False, retries=12, delay_seconds=1.0)

    def restart_guideline_pipeline_request(payload: dict) -> dict:
        with state_lock:
            update_process_state()
            current_job = launcher_state.get("current_job")
            if isinstance(current_job, dict):
                job_kind = str(current_job.get("job_kind") or "").strip().lower()
                if job_kind in {"guideline", "rgb", "cleanup", "train_guideline"}:
                    raise HTTPException(
                        status_code=409,
                        detail="현재 guideline/RGB/cleanup 작업이 실행 중입니다. 먼저 강제종료한 뒤 다시 눌러 주세요.",
                    )
            pending_jobs = launcher_state.get("queued_jobs", [])
            if isinstance(pending_jobs, list):
                launcher_state["queued_jobs"] = [
                    job
                    for job in pending_jobs
                    if not (
                        isinstance(job, dict)
                        and str(job.get("job_kind") or "").strip().lower()
                        in {"guideline", "rgb", "cleanup", "train_guideline"}
                    )
                ]
            clear_guideline_outputs()

        restart_payload = {
            **payload,
            "source": str(payload.get("source") or "cumulative").strip() or "cumulative",
            "include_rgb": payload.get("include_rgb", "1"),
            "start_train": payload.get("start_train", "0"),
            "cleanup_after": payload.get("cleanup_after", "1"),
            "skip_guideline": "0",
        }
        result = start_guideline_pipeline_request(restart_payload)
        return {
            **result,
            "message": "guideline 산출물을 지우고 새 설정으로 guideline_clips부터 다시 시작했습니다.",
        }

    def clear_rgb_feature_outputs() -> None:
        workspace_dir = paths.get("workspace_dir")
        target = workspace_dir / "rgb_clip_features" if isinstance(workspace_dir, Path) else None
        if isinstance(target, Path) and target.exists():
            remove_transient_path_with_retries(target, recreate_dir=False, retries=12, delay_seconds=1.0)

    def guideline_manifests_are_stale() -> bool:
        source_paths = [
            paths["active_prepared_train"],
            paths["active_prepared_val"],
            paths["active_prepared_test"],
            paths["prepared_train"],
            paths["prepared_val"],
            paths["prepared_test"],
        ]
        guideline_paths = [
            paths["guideline_prepared_train"],
            paths["guideline_prepared_val"],
            paths["guideline_prepared_test"],
        ]
        existing_sources = [path for path in source_paths if isinstance(path, Path) and path.exists()]
        existing_guidelines = [path for path in guideline_paths if isinstance(path, Path) and path.exists()]
        if not existing_sources:
            return False
        if len(existing_guidelines) < len(guideline_paths):
            return True
        try:
            newest_source = max(path.stat().st_mtime for path in existing_sources)
            oldest_guideline = min(path.stat().st_mtime for path in existing_guidelines)
        except OSError:
            return True
        return newest_source > oldest_guideline + 0.5

    def restart_rgb_pipeline_request(payload: dict) -> dict:
        with state_lock:
            update_process_state()
            current_job = launcher_state.get("current_job")
            if isinstance(current_job, dict):
                job_kind = str(current_job.get("job_kind") or "").strip().lower()
                if job_kind in {"guideline", "rgb", "cleanup", "train_guideline"}:
                    raise HTTPException(
                        status_code=409,
                        detail="현재 guideline/RGB/cleanup 작업이 실행 중입니다. 먼저 강제종료하거나 완료 후 다시 눌러 주세요.",
                    )
            if not paths["guideline_prepared_train"].exists() or not paths["guideline_prepared_val"].exists():
                raise HTTPException(
                    status_code=409,
                    detail="guideline clip manifest가 아직 없습니다. guideline_clips 완료 후 RGB/I3D부터 재시작할 수 있습니다.",
                )
            pending_jobs = launcher_state.get("queued_jobs", [])
            if isinstance(pending_jobs, list):
                launcher_state["queued_jobs"] = [
                    job
                    for job in pending_jobs
                    if not (
                        isinstance(job, dict)
                        and str(job.get("job_kind") or "").strip().lower()
                        in {"rgb", "cleanup", "train_guideline"}
                    )
                ]
            if str(payload.get("clear_rgb_features") or "").strip().lower() in {"1", "true", "yes", "on"}:
                clear_rgb_feature_outputs()
            refresh_guideline_first = guideline_manifests_are_stale()

        restart_payload = {
            **payload,
            "skip_guideline": "0" if refresh_guideline_first else "1",
            "include_rgb": "1",
            "start_train": payload.get("start_train", "0"),
            "cleanup_after": payload.get("cleanup_after", "1"),
            "rgb_model": payload.get("rgb_model") or "i3d_r50",
        }
        result = start_guideline_pipeline_request(restart_payload)
        message = (
            "guideline manifest가 최신 active prepared보다 오래되어 guideline_clips부터 다시 만든 뒤 RGB/I3D feature 추출을 시작했습니다."
            if refresh_guideline_first
            else "RGB/I3D feature 산출물을 지우고 RGB/I3D feature 추출부터 다시 시작했습니다."
        )
        return {
            **result,
            "message": message,
        }

    def pause_after_current_request() -> dict:
        with state_lock:
            update_process_state()
            current_job = launcher_state.get("current_job")
            pending_jobs = launcher_state.get("queued_jobs", [])
            has_pending = isinstance(pending_jobs, list) and len(pending_jobs) > 0
            if not current_job and not has_pending:
                raise HTTPException(status_code=409, detail="중지할 작업이 없습니다.")

            launcher_state["auto_start_enabled"] = False
            launcher_state["auto_enqueue_enabled"] = False
            launcher_state["auto_extract_enabled"] = False
            if current_job:
                launcher_state["last_state"] = "running"
                launcher_state["last_message"] = "현재 작업까지만 진행하고, 완료 후 다음 큐 자동 시작을 멈춥니다."
            else:
                launcher_state["last_state"] = "paused"
                launcher_state["last_message"] = "다음 큐 자동 시작을 중지했습니다. 시작 버튼을 누르면 재개합니다."

        return {
            "ok": True,
            "message": "현재 작업까지만 진행하고, 다음 큐 자동 시작을 멈춥니다.",
            "launcher": get_launcher_status(),
        }

    def stop_after_queue_request() -> dict:
        with state_lock:
            launcher_state["auto_start_enabled"] = True
            launcher_state["auto_enqueue_enabled"] = False
            launcher_state["auto_extract_enabled"] = False
            launcher_state["auto_enqueue_datasetkey"] = None
            launcher_state["auto_enqueue_api_key"] = ""
            update_process_state()
            current_job = launcher_state.get("current_job")
            pending_jobs = launcher_state.get("queued_jobs", [])
            pending_count = len(pending_jobs) if isinstance(pending_jobs, list) else 0
            if current_job or pending_count > 0:
                launcher_state["last_state"] = "queued" if pending_count > 0 else "running"
                launcher_state["last_message"] = (
                    f"자동 전처리 추가를 중단했습니다. 현재 큐 {pending_count}개까지 처리한 뒤 멈춥니다."
                )
            else:
                launcher_state["last_state"] = "idle"
                launcher_state["last_message"] = "자동 전처리 추가를 중단했습니다. 대기 중인 큐가 없습니다."

        return {
            "ok": True,
            "message": "이번 큐 이후에는 새 filekey 전처리를 자동으로 추가하지 않습니다.",
            "launcher": get_launcher_status(),
        }

    def remove_queued_job_request(job_id: str) -> dict:
        if not job_id:
            raise HTTPException(status_code=400, detail="삭제할 job_id가 필요합니다.")

        with state_lock:
            update_process_state()
            pending_jobs = launcher_state.get("queued_jobs", [])
            if not isinstance(pending_jobs, list) or not pending_jobs:
                raise HTTPException(status_code=404, detail="삭제할 대기열 작업이 없습니다.")

            removed_job = None
            remaining_jobs = []
            for job in pending_jobs:
                if (
                    removed_job is None
                    and isinstance(job, dict)
                    and str(job.get("job_id", "")).strip() == job_id
                ):
                    removed_job = snapshot_job(job)
                    continue
                remaining_jobs.append(job)

            if removed_job is None:
                raise HTTPException(status_code=404, detail="선택한 대기열 작업을 찾을 수 없습니다.")

            launcher_state["queued_jobs"] = remaining_jobs
            launcher_state["last_state"] = "queued" if remaining_jobs else (launcher_state.get("last_state") or "idle")
            launcher_state["last_message"] = (
                f"datasetkey {removed_job.get('datasetkey', '-')}"
                f" | filekey {removed_job.get('filekey', '-')} 를 대기열에서 삭제했습니다."
            )

        return {
            "ok": True,
            "message": (
                f"datasetkey {removed_job.get('datasetkey', '-')}"
                f" | filekey {removed_job.get('filekey', '-')} 를 대기열에서 삭제했습니다."
            ),
            "launcher": get_launcher_status(),
        }

    def force_stop_current_job_request() -> dict:
        with state_lock:
            update_process_state()
            active_process = launcher_state.get("process")
            current_job = launcher_state.get("current_job")
            if not isinstance(active_process, subprocess.Popen) or not isinstance(current_job, dict):
                raise HTTPException(status_code=409, detail="현재 강제 중단할 작업이 없습니다.")

            launcher_state["auto_start_enabled"] = False
            launcher_state["auto_enqueue_enabled"] = False
            launcher_state["auto_extract_enabled"] = False

            stop_process_tree(active_process)

            current_job["finished_at"] = current_timestamp()
            current_job["exit_code"] = active_process.returncode if active_process.returncode is not None else -1
            current_job["state"] = "aborted"
            current_job["message"] = (
                f"filekey {current_job.get('filekey')} 작업을 강제 중단했습니다. "
                "같은 작업을 대기열 맨 앞으로 다시 넣었고, 시작 버튼을 눌러야 재개됩니다."
            )
            current_job["result_summary"] = collect_result_summary(paths)
            current_job = enrich_completed_job(current_job)

            completed_jobs = launcher_state.setdefault("completed_jobs", [])
            if isinstance(completed_jobs, list):
                completed_jobs.insert(0, snapshot_job(current_job))
                del completed_jobs[30:]
            persist_launcher_history(launcher_history_path, launcher_state)

            retry_job = build_retry_job_from(current_job)
            try:
                retry_count = int(retry_job.get("retry_count", 0) or 0)
            except (TypeError, ValueError):
                retry_count = 0
            scope = retry_job.get("recommendation_scope") if isinstance(retry_job.get("recommendation_scope"), dict) else {}
            zip_group = str(scope.get("zip_group") or "").strip().lower()
            should_requeue_retry = retry_count < 3 and (
                not zip_group or zip_group in set(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS)
            )
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            if should_requeue_retry and isinstance(pending_jobs, list):
                pending_jobs.insert(0, retry_job)

            launcher_state["process"] = None
            launcher_state["started_at"] = None
            launcher_state["runtime_config_path"] = None
            launcher_state["current_job"] = None
            launcher_state["last_state"] = "paused"
            launcher_state["last_exit_code"] = current_job["exit_code"]
            launcher_state["log_path"] = current_job.get("log_path")
            launcher_state["last_message"] = (
                f"filekey {current_job.get('filekey')} 작업을 강제 중단했습니다. "
                "같은 작업을 대기열 맨 앞으로 다시 넣었고, 시작 버튼을 눌러야 재개됩니다."
            )

            reset_training_workspace(paths)
            write_dashboard_status(
                paths,
                stage="paused",
                state="paused",
                message=(
                    f"filekey {current_job.get('filekey')} 작업을 강제 중단했습니다. "
                    "다시 시작하면 현재 filekey를 처음부터 재시도합니다."
                ),
                stage_progress=0.0,
                current_filekey=current_job.get("filekey"),
                current_datasetkey=current_job.get("datasetkey"),
            )
            abort_message = str(current_job.get("message") or launcher_state.get("last_message") or "")
            if filekey_pipeline_step_id(current_job):
                record_filekey_pipeline_finished(current_job, "aborted", abort_message)
                _, datasetkey, filekey = filekey_pipeline_identity(current_job)
                dispatch_dashboard_notification(
                    "aborted",
                    title=f"Filekey stopped: {filekey or '-'}",
                    message=(
                        f"filekey: {filekey or '-'}\n"
                        "status: aborted\n"
                        f"datasetkey: {datasetkey or '-'}\n"
                        "파일키 전체 작업이 중단되었습니다.\n"
                        f"{abort_message}"
                    ),
                    priority="high",
                )
            else:
                notify_title, notify_body, notify_priority = build_job_notification_payload(
                    current_job,
                    "aborted",
                    abort_message,
                )
                dispatch_dashboard_notification("aborted", title=notify_title, message=notify_body, priority=notify_priority)

        return {
            "ok": True,
            "message": (
                f"filekey {current_job.get('filekey')} 작업을 강제 중단했습니다. "
                "같은 filekey를 큐 맨 앞으로 다시 넣었고, 시작 버튼을 누르면 처음부터 다시 시작합니다."
            ),
            "launcher": get_launcher_status(),
        }

    def reset_training_data_request() -> dict:
        with state_lock:
            update_process_state()
            active_process = launcher_state.get("process")
            if isinstance(active_process, subprocess.Popen) and active_process.poll() is None:
                launcher_state["auto_start_enabled"] = False
                launcher_state["auto_enqueue_enabled"] = False
                launcher_state["auto_extract_enabled"] = False
                stop_process_tree(active_process)
                launcher_state["process"] = None
                launcher_state["started_at"] = None
                launcher_state["runtime_config_path"] = None
                launcher_state["current_job"] = None
            hard_reset_workspace()

        return {
            "ok": True,
            "message": "현재 작업과 대기열을 중단하고 학습 워크스페이스를 초기화했습니다. datasetkey와 filekey를 다시 넣어 처음부터 시작할 수 있습니다.",
            "launcher": get_launcher_status(),
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, notice: str | None = None, notice_level: str = "info") -> HTMLResponse:
        cookie_notice = request.cookies.get(NOTICE_COOKIE_NAME)
        cookie_notice_level = request.cookies.get(NOTICE_LEVEL_COOKIE_NAME)
        effective_notice = notice
        effective_notice_level = notice_level
        if not effective_notice and cookie_notice:
            try:
                effective_notice = unquote(cookie_notice)
            except Exception:
                effective_notice = cookie_notice
            effective_notice_level = cookie_notice_level or notice_level or "info"

        viewer_mode = is_viewer_request(request)
        initial_overview = build_overview(
            paths,
            config_path,
            config=config,
            launcher_status=get_launcher_status(),
        )
        if effective_notice:
            launcher_payload = initial_overview.setdefault("launcher", {})
            if isinstance(launcher_payload, dict):
                launcher_payload["message"] = effective_notice
                if effective_notice_level == "danger":
                    launcher_payload["state"] = "error"
                elif effective_notice_level == "warn":
                    launcher_payload["state"] = launcher_payload.get("state") or "warning"
        html = render_dashboard_page(
            initial_overview,
            config_path=str(config_path),
            default_datasetkey=str(
                (
                    (initial_overview.get("aihub") or {}).get("datasetkey")
                    or config.get("aihub_shell", {}).get("datasetkey")
                    or ""
                )
            ),
            controls_enabled=(
                str(config.get("dataset_source") or "").strip().lower() == "aihub_shell"
                and not viewer_mode
            ),
            controls_notice=(
                "공유 viewer에서는 작업 제어가 비활성화되어 있습니다. "
                "시작/중지/초기화는 로컬 대시보드에서만 할 수 있습니다."
                if viewer_mode
                else None
            ),
            notice=effective_notice,
            notice_level=effective_notice_level,
            refresh_seconds=0,
        )
        response = HTMLResponse(html, headers=NO_CACHE_HEADERS)
        if cookie_notice or cookie_notice_level:
            response.delete_cookie(NOTICE_COOKIE_NAME, path="/")
            response.delete_cookie(NOTICE_LEVEL_COOKIE_NAME, path="/")
        return response

    @app.get("/api/overview")
    def overview(lite: bool = False) -> JSONResponse:
        return JSONResponse(
            build_overview(
                paths,
                config_path,
                config=config,
                launcher_status=get_launcher_status(),
                lite=lite,
            ),
            headers=NO_CACHE_HEADERS,
        )

    @app.get("/api/live-fragments")
    def live_fragments(request: Request) -> JSONResponse:
        overview_payload = build_overview(
            paths,
            config_path,
            config=config,
            launcher_status=get_launcher_status(),
        )
        active = overview_has_live_activity(overview_payload)
        overview_revision = overview_payload.get("overview_revision")
        client_revision = str(request.query_params.get("revision") or "").strip()
        if client_revision and overview_revision and client_revision == overview_revision:
            return JSONResponse(
                {
                    "ok": True,
                    "overview_revision": overview_revision,
                    "active": active,
                    "poll_interval_ms": 1200 if active else 8000,
                    "fragments": {},
                },
                headers=NO_CACHE_HEADERS,
            )
        return JSONResponse(
            {
                "ok": True,
                "overview_revision": overview_revision,
                "active": active,
                "poll_interval_ms": 1200 if active else 8000,
                "fragments": render_dashboard_live_fragments(overview_payload),
            },
            headers=NO_CACHE_HEADERS,
        )

    @app.get("/api/live-ping")
    def live_ping() -> JSONResponse:
        launcher = get_launcher_status()
        return JSONResponse(
            {
                "ok": True,
                "state": launcher.get("state"),
                "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            },
            headers=NO_CACHE_HEADERS,
        )

    @app.get("/api/aihub/filekeys")
    def lookup_aihub_filekeys_get(request: Request, datasetkey: str = "") -> JSONResponse:
        ensure_dashboard_control_access(request)
        return JSONResponse(
            lookup_aihub_filekeys_request({"datasetkey": datasetkey}),
            headers=NO_CACHE_HEADERS,
        )

    @app.post("/api/aihub/filekeys")
    async def lookup_aihub_filekeys(request: Request) -> JSONResponse:
        ensure_dashboard_control_access(request)
        try:
            payload = await request.json()
        except Exception:
            payload = await read_form_payload(request)
        if not isinstance(payload, dict):
            payload = {}
        return JSONResponse(lookup_aihub_filekeys_request(payload), headers=NO_CACHE_HEADERS)

    @app.post("/api/start")
    async def start_training(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        payload = await request.json()
        return start_training_request(payload)

    @app.post("/api/guideline-pipeline")
    async def start_guideline_pipeline(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        payload = await request.json()
        return start_guideline_pipeline_request(payload)

    @app.post("/api/performance-plan")
    async def start_performance_plan(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        payload = await request.json()
        return start_performance_plan_request(payload)

    @app.post("/api/restart-guideline")
    async def restart_guideline_pipeline(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        payload = await request.json()
        return restart_guideline_pipeline_request(payload)

    @app.post("/api/restart-rgb")
    async def restart_rgb_pipeline(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        payload = await request.json()
        return restart_rgb_pipeline_request(payload)

    @app.post("/api/pause")
    def pause_after_current(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        return pause_after_current_request()

    @app.post("/api/stop-after-queue")
    def stop_after_queue(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        return stop_after_queue_request()

    @app.post("/api/remove-queued-job")
    async def remove_queued_job(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        payload = await request.json()
        return remove_queued_job_request(str(payload.get("job_id", "")).strip())

    @app.post("/api/force-stop")
    def force_stop_current_job(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        return force_stop_current_job_request()

    @app.post("/api/reset")
    def reset_training_data(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        return reset_training_data_request()

    @app.post("/actions/start")
    async def start_training_action(request: Request):
        ensure_dashboard_control_access(request)
        payload = await read_form_payload(request)
        try:
            result = start_training_request(payload)
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "작업을 시작했습니다."),
                level="good",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )

    @app.post("/actions/guideline-pipeline")
    async def start_guideline_pipeline_action(request: Request):
        ensure_dashboard_control_access(request)
        payload = await read_form_payload(request)
        try:
            result = start_guideline_pipeline_request(payload)
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "가이드라인 개선 파이프라인을 시작했습니다."),
                level="good",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )
        except Exception as exc:
            return build_dashboard_action_response(
                request,
                message=f"작업 시작 중 오류가 발생했습니다: {exc}",
                level="danger",
                ok=False,
                status_code=500,
            )

    @app.post("/actions/performance-plan")
    async def start_performance_plan_action(request: Request):
        ensure_dashboard_control_access(request)
        payload = await read_form_payload(request)
        try:
            result = start_performance_plan_request(payload)
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "최고 성능 자동 플랜을 시작했습니다."),
                level="good",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )
        except Exception as exc:
            return build_dashboard_action_response(
                request,
                message=f"성능 플랜 시작 중 오류가 발생했습니다: {exc}",
                level="danger",
                ok=False,
                status_code=500,
            )

    @app.post("/actions/restart-guideline")
    async def restart_guideline_pipeline_action(request: Request):
        ensure_dashboard_control_access(request)
        payload = await read_form_payload(request)
        try:
            result = restart_guideline_pipeline_request(payload)
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "guideline_clips를 다시 시작했습니다."),
                level="warn",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )
        except Exception as exc:
            return build_dashboard_action_response(
                request,
                message=f"guideline 재시작 중 오류가 발생했습니다: {exc}",
                level="danger",
                ok=False,
                status_code=500,
            )

    @app.post("/actions/restart-rgb")
    async def restart_rgb_pipeline_action(request: Request):
        ensure_dashboard_control_access(request)
        payload = await read_form_payload(request)
        try:
            result = restart_rgb_pipeline_request(payload)
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "RGB/I3D feature 추출을 다시 시작했습니다."),
                level="warn",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )
        except Exception as exc:
            return build_dashboard_action_response(
                request,
                message=f"RGB/I3D 재시작 중 오류가 발생했습니다: {exc}",
                level="danger",
                ok=False,
                status_code=500,
            )

    @app.post("/actions/pause")
    def pause_after_current_action(request: Request):
        ensure_dashboard_control_access(request)
        try:
            result = pause_after_current_request()
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "자동 시작을 멈췄습니다."),
                level="warn",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )
        except Exception as exc:
            return build_dashboard_action_response(
                request,
                message=f"학습 파이프라인 시작 중 오류가 발생했습니다: {exc}",
                level="danger",
                ok=False,
                status_code=500,
            )

    @app.post("/actions/stop-after-queue")
    def stop_after_queue_action(request: Request):
        ensure_dashboard_control_access(request)
        try:
            result = stop_after_queue_request()
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "이번 큐 이후 자동 전처리를 중단합니다."),
                level="warn",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )
        except Exception as exc:
            return build_dashboard_action_response(
                request,
                message=f"전처리 중단 예약 중 오류가 발생했습니다: {exc}",
                level="danger",
                ok=False,
                status_code=500,
            )

    @app.post("/actions/remove-queued")
    async def remove_queued_job_action(request: Request):
        ensure_dashboard_control_access(request)
        payload = await read_form_payload(request)
        try:
            result = remove_queued_job_request(str(payload.get("job_id", "")).strip())
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "대기열에서 제거했습니다."),
                level="good",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )

    @app.post("/actions/force-stop")
    def force_stop_current_job_action(request: Request):
        ensure_dashboard_control_access(request)
        try:
            result = force_stop_current_job_request()
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "현재 작업을 중단했습니다."),
                level="warn",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )

    @app.post("/actions/reset")
    def reset_training_data_action(request: Request):
        ensure_dashboard_control_access(request)
        try:
            result = reset_training_data_request()
            return build_dashboard_action_response(
                request,
                message=str(result.get("message") or "워크스페이스를 초기화했습니다."),
                level="warn",
            )
        except HTTPException as exc:
            return build_dashboard_action_response(
                request,
                message=str(exc.detail),
                level="danger",
                ok=False,
                status_code=exc.status_code,
            )
        except Exception as exc:
            return build_dashboard_action_response(
                request,
                message=f"초기화 중 오류가 발생했습니다: {exc}",
                level="danger",
                ok=False,
                status_code=500,
            )

    worker = threading.Thread(target=queue_worker, daemon=True)
    worker.start()
    return app


def build_overview_file_diagnostics(paths: dict) -> dict:
    return {
        "pipeline_status": build_path_diagnostic(paths["pipeline_status"]),
        "training_progress": build_path_diagnostic(paths["training_progress"]),
        "metrics": build_path_diagnostic(paths["artifacts_dir"] / "metrics.json"),
        "labels": build_path_diagnostic(paths["artifacts_dir"] / "labels.json"),
        "launcher_history": build_path_diagnostic(paths["workspace_dir"] / "launcher_history.json"),
        "current_raw_manifest": build_path_diagnostic(paths["current_raw_manifest"]),
        "current_prepared_train": build_path_diagnostic(paths["current_prepared_train"]),
        "current_prepared_val": build_path_diagnostic(paths["current_prepared_val"]),
        "current_prepared_test": build_path_diagnostic(paths["current_prepared_test"]),
        "guideline_prepared_train": build_path_diagnostic(paths["guideline_prepared_train"]),
        "guideline_prepared_val": build_path_diagnostic(paths["guideline_prepared_val"]),
        "guideline_prepared_test": build_path_diagnostic(paths["guideline_prepared_test"]),
    }


def build_overview_diagnostics(
    *,
    paths: dict,
    pipeline_diagnostics: dict,
    training_progress: dict,
    metrics: dict,
    dataset_summary: dict,
    current_dataset_summary: dict,
    completed_jobs: list[dict],
) -> dict:
    warnings = list(pipeline_diagnostics.get("warnings") or [])
    if dataset_summary:
        prepared_total = sum(
            int((dataset_summary.get(key) or {}).get("total", 0) or 0)
            for key in ("prepared_train", "prepared_val", "prepared_test")
        )
        current_prepared_total = sum(
            int((current_dataset_summary.get(key) or {}).get("total", 0) or 0)
            for key in ("prepared_train", "prepared_val", "prepared_test")
        )
        if prepared_total > 0 and current_prepared_total == 0:
            warnings.append(
                "current_* manifest는 비어 있지만 cumulative prepared 데이터는 남아 있습니다. "
                "완료 후 정리(cleanup_raw_after_job)된 정상 상태일 수 있습니다."
            )
    if not (training_progress.get("history") or metrics.get("history")):
        warnings.append("history가 비어 있어 epoch 추이를 표시할 수 없습니다.")
    has_success_job = any(
        str(job.get("state") or "").strip().lower() in {"completed", "completed_warning"}
        for job in completed_jobs
        if isinstance(job, dict)
    )
    has_prepared_job = any(
        str(job.get("state") or "").strip().lower() in AIHUB_PREPARED_JOB_STATES
        for job in completed_jobs
        if isinstance(job, dict)
    )
    has_failed_job = any(
        str(job.get("state") or "").strip().lower() in {"error", "aborted"}
        for job in completed_jobs
        if isinstance(job, dict)
    )
    if completed_jobs and not has_success_job:
        if has_prepared_job and not has_failed_job:
            warnings.append("데이터 준비 작업은 완료됐지만 아직 모델 학습 완료 작업은 없습니다.")
        elif has_failed_job:
            warnings.append("완료 이력에는 성공 학습 작업이 없고 중단/실패 작업이 포함되어 있습니다.")
    return {
        "warnings": warnings,
        "pipeline": pipeline_diagnostics,
        "files": build_overview_file_diagnostics(paths),
    }


def build_prepared_pose_items(paths: dict, *, limit: int = 100) -> list[dict]:
    grouped: dict[str, dict] = {}
    for split_name, path_key in (
        ("train", "prepared_train"),
        ("val", "prepared_val"),
        ("test", "prepared_test"),
    ):
        manifest_path = paths.get(path_key)
        if not isinstance(manifest_path, Path) or not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_filekey = entry.get("job_filekey") or entry.get("filekey") or "-"
                filekey = ",".join(str(value) for value in raw_filekey) if isinstance(raw_filekey, list) else str(raw_filekey)
                item = grouped.setdefault(filekey, {"filekey": filekey, "total": 0, "splits": Counter(), "labels": Counter()})
                item["total"] += 1
                item["splits"][split_name] += 1
                item["labels"][str(entry.get("target_label") or "-")] += 1
    return [
        {
            "filekey": key,
            "total": value["total"],
            "splits": dict(value["splits"]),
            "labels": dict(value["labels"]),
        }
        for key, value in list(grouped.items())[:limit]
    ]


def build_current_job_filekey_summary(paths: dict, current_job: dict | None, *, limit: int = 16) -> dict:
    if not isinstance(current_job, dict):
        return {"source": "", "total_items": 0, "filekey_count": 0, "filekeys": []}

    job_kind = infer_dashboard_job_kind(current_job)
    if job_kind in {"rgb", "train_guideline"}:
        manifest_specs = (
            ("train", "guideline_prepared_train"),
            ("val", "guideline_prepared_val"),
            ("test", "guideline_prepared_test"),
        )
        source = "guideline_prepared"
    elif job_kind == "guideline":
        manifest_specs = (
            ("train", "split_train"),
            ("val", "split_val"),
            ("test", "split_test"),
        )
        source = "cumulative_split"
    else:
        manifest_specs = (
            ("train", "current_split_train"),
            ("val", "current_split_val"),
            ("test", "current_split_test"),
        )
        source = "current_split"

    grouped: dict[str, dict] = {}
    for split_name, path_key in manifest_specs:
        manifest_path = paths.get(path_key)
        if not isinstance(manifest_path, Path) or not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    filekey = manifest_row_filekey(row)
                    item = grouped.setdefault(
                        filekey,
                        {
                            "filekey": filekey,
                            "total": 0,
                            "splits": Counter(),
                            "labels": Counter(),
                            "rgb_ready": 0,
                            "rgb_missing": 0,
                        },
                    )
                    item["total"] += 1
                    item["splits"][split_name] += 1
                    item["labels"][str(row.get("target_label") or row.get("source_label") or "unknown")] += 1
                    if str(row.get("rgb_feature_path") or "").strip():
                        item["rgb_ready"] += 1
                    else:
                        item["rgb_missing"] += 1
        except OSError:
            continue

    items = [
        {
            "filekey": value["filekey"],
            "total": value["total"],
            "splits": dict(value["splits"]),
            "labels": dict(value["labels"].most_common(5)),
            "rgb_ready": value["rgb_ready"],
            "rgb_missing": value["rgb_missing"],
        }
        for value in sorted(grouped.values(), key=lambda entry: (-int(entry["total"]), str(entry["filekey"])))[:limit]
    ]
    return {
        "source": source,
        "total_items": sum(int(value["total"]) for value in grouped.values()),
        "filekey_count": len(grouped),
        "filekeys": items,
    }


def manifest_row_filekey(row: dict) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for container in (row, metadata):
        if not isinstance(container, dict):
            continue
        for key in (
            "source_filekey",
            "filekey",
            "file_key",
            "fileKey",
            "aihub_filekey",
            "job_filekey",
            "dataset_filekey",
            "rgb_feature_filekey",
        ):
            value = container.get(key)
            if isinstance(value, list):
                value = ",".join(str(item).strip() for item in value if str(item).strip())
            text = str(value or "").strip()
            if text:
                return text
    for key in ("rgb_feature_path", "video_path", "source_video_path", "pose_path"):
        value = str(row.get(key) or metadata.get(key) or "").strip()
        for part in Path(value).parts:
            lower = part.lower()
            if lower.startswith("job_"):
                parts = part.split("_")
                return parts[1] if len(parts) > 1 and parts[1].isdigit() else part
            if part.isdigit() and len(part) >= 4:
                return part
    return "unknown"


def infer_guideline_pipeline_source_filekeys(paths: dict, *, use_guideline: bool) -> str:
    specs = (
        (
            ("guideline_prepared_train", "guideline_prepared_val", "guideline_prepared_test")
            if use_guideline
            else ("split_train", "split_val", "split_test")
        )
    )
    counts: Counter[str] = Counter()
    for path_key in specs:
        manifest_path = paths.get(path_key)
        if not isinstance(manifest_path, Path) or not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    filekey = manifest_row_filekey(row)
                    if filekey and filekey != "unknown":
                        counts[filekey] += 1
        except OSError:
            continue
    if not counts:
        return ""
    ordered = [filekey for filekey, _ in counts.most_common()]
    if len(ordered) <= 6:
        return ",".join(ordered)
    return ",".join([*ordered[:6], f"+{len(ordered) - 6}more"])


def build_overview(
    paths: dict,
    config_path: Path,
    *,
    config: dict,
    launcher_status: dict | None = None,
    lite: bool = False,
) -> dict:
    launcher_status = launcher_status or {}
    signature = compute_overview_signature(paths, launcher_status, lite=lite)
    overview_revision = hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:16]
    with OVERVIEW_CACHE_LOCK:
        cached_value = OVERVIEW_CACHE.get(signature)
        if isinstance(cached_value, dict):
            return cached_value

    gpu_status = query_gpu_status()
    raw_pipeline_status = read_json(paths["pipeline_status"]) or {}
    target_labels = get_target_labels(config)
    training_progress = normalize_metric_payload(
        read_json(paths["training_progress"]),
        target_labels=target_labels,
    )
    metrics = normalize_metric_payload(
        read_json(paths["artifacts_dir"] / "metrics.json"),
        target_labels=target_labels,
    )
    current_skip_report = read_json(paths["current_skip_report"]) or {
        "summary": {"total_issues": 0, "broken_count": 0, "skipped_count": 0},
        "issues": [],
    }
    cumulative_skip_report = read_json(paths["cumulative_skip_report"]) or {
        "summary": {"total_issues": 0, "broken_count": 0, "skipped_count": 0},
        "issues": [],
    }
    launcher_history = read_json(paths["workspace_dir"] / "launcher_history.json") or {}
    history_completed_jobs = launcher_history.get("completed_jobs", []) if isinstance(launcher_history, dict) else []
    completed_jobs = launcher_status.get("completed_jobs", []) if isinstance(launcher_status, dict) else []
    if not isinstance(completed_jobs, list) or not completed_jobs:
        completed_jobs = history_completed_jobs if isinstance(history_completed_jobs, list) else []
    if (not isinstance(completed_jobs, list) or not completed_jobs) and not lite:
        completed_jobs = restore_completed_jobs_from_logs(
            paths["workspace_dir"] / "job_logs",
            paths["workspace_dir"] / "runtime_configs",
        )
    completed_jobs = sort_jobs_by_recency(completed_jobs if isinstance(completed_jobs, list) else [])
    pending_jobs = launcher_status.get("pending_jobs", []) if isinstance(launcher_status, dict) else []
    current_job = launcher_status.get("current_job") if isinstance(launcher_status, dict) else None
    pipeline_status, pipeline_diagnostics = build_effective_pipeline_status(
        raw_pipeline_status,
        training_progress=training_progress,
        completed_jobs=completed_jobs,
        workspace_dir=paths["workspace_dir"],
    )
    effective_launcher_status = dict(launcher_status) if isinstance(launcher_status, dict) else {}
    restored_launcher = build_restored_launcher_summary(
        completed_jobs,
        pipeline_status=pipeline_status,
        training_progress=training_progress,
    )
    if not effective_launcher_status.get("state"):
        effective_launcher_status["state"] = restored_launcher.get("state")
    if not effective_launcher_status.get("message"):
        effective_launcher_status["message"] = restored_launcher.get("message")
    if not effective_launcher_status.get("log_path") and restored_launcher.get("log_path"):
        effective_launcher_status["log_path"] = restored_launcher.get("log_path")
    if effective_launcher_status.get("last_exit_code") is None and restored_launcher.get("last_exit_code") is not None:
        effective_launcher_status["last_exit_code"] = restored_launcher.get("last_exit_code")
    latest_completed_job = next(
        (
            job for job in completed_jobs
            if isinstance(job, dict) and job.get("state") in {"completed", "completed_warning"}
        ),
        None,
    )
    log_tail_cache: dict[str, str] = {}
    log_preview_cache: dict[str, str] = {}

    def get_cached_log_tail(path_value) -> str:
        if lite or not path_value:
            return ""
        cache_key = str(path_value)
        if cache_key not in log_tail_cache:
            log_tail_cache[cache_key] = read_log_tail(path_value)
        return log_tail_cache[cache_key]

    def get_cached_log_preview(path_value) -> str:
        if lite or not path_value:
            return ""
        cache_key = str(path_value)
        if cache_key not in log_preview_cache:
            log_preview_cache[cache_key] = read_log_preview(path_value)
        return log_preview_cache[cache_key]

    current_log_source = current_job if isinstance(current_job, dict) else latest_completed_job
    current_log_path = current_log_source.get("log_path") if isinstance(current_log_source, dict) else None
    current_log_tail = get_cached_log_tail(current_log_path)

    latest_error_job = next(
        (
            job for job in completed_jobs
            if isinstance(job, dict) and job.get("state") == "error"
        ),
        None,
    )
    latest_error_log_path = latest_error_job.get("log_path") if isinstance(latest_error_job, dict) else None
    latest_error_log_tail = get_cached_log_tail(latest_error_log_path)
    enriched_completed_jobs = []
    for job in completed_jobs:
        if not isinstance(job, dict):
            continue
        enriched_job = enrich_completed_job(job)
        if not lite:
            enriched_job["log_preview"] = get_cached_log_preview(enriched_job.get("log_path"))
        enriched_completed_jobs.append(enriched_job)

    completed_count = len([
        job for job in completed_jobs
        if isinstance(job, dict) and job.get("state") in {"completed", "completed_warning"}
    ])
    prepared_count = len([
        job for job in completed_jobs
        if isinstance(job, dict) and str(job.get("state") or "").strip().lower() in AIHUB_PREPARED_JOB_STATES
    ])
    failed_count = len([job for job in completed_jobs if isinstance(job, dict) and job.get("state") == "error"])
    pending_count = len(pending_jobs) if isinstance(pending_jobs, list) else 0
    active_count = 1 if current_job else 0
    total_count = completed_count + prepared_count + failed_count + pending_count + active_count
    progress_ratio = ((completed_count + prepared_count + failed_count) / total_count) if total_count > 0 else 0.0
    current_job_progress = build_current_job_progress(pipeline_status, training_progress, effective_launcher_status)
    eta = estimate_eta(current_job_progress, effective_launcher_status)
    cumulative_dataset_summary = (
        build_manifest_summary_group(paths, OVERVIEW_DATASET_MANIFEST_SPECS)
        if not lite
        else {}
    )
    dataset_summary = dict(cumulative_dataset_summary)
    guideline_dataset_summary = (
        build_manifest_summary_group(
            paths,
            (
                ("prepared_train", "guideline_prepared_train"),
                ("prepared_val", "guideline_prepared_val"),
                ("prepared_test", "guideline_prepared_test"),
            ),
        )
        if not lite
        else {}
    )
    if guideline_dataset_summary and int((guideline_dataset_summary.get("prepared_train") or {}).get("total", 0) or 0) > 0:
        dataset_summary.update(guideline_dataset_summary)
    active_dataset_summary = (
        build_manifest_summary_group(
            paths,
            (
                ("prepared_train", "active_prepared_train"),
                ("prepared_val", "active_prepared_val"),
                ("prepared_test", "active_prepared_test"),
            ),
        )
        if not lite
        else {}
    )
    if active_dataset_summary and int((active_dataset_summary.get("prepared_train") or {}).get("total", 0) or 0) > 0:
        dataset_summary.update(active_dataset_summary)
    guideline_quality = build_guideline_quality_summary(paths, config=config) if not lite else {}
    current_dataset_summary = build_manifest_summary_group(paths, OVERVIEW_CURRENT_DATASET_MANIFEST_SPECS)
    distribution_sources = (
        {
            "cumulative": cumulative_dataset_summary,
            "guideline": guideline_dataset_summary,
            "active": active_dataset_summary,
            "current": current_dataset_summary,
        }
        if not lite
        else {}
    )
    if "train_distribution" not in training_progress and dataset_summary:
        training_progress["train_distribution"] = analyze_class_balance(
            target_labels,
            dataset_summary["prepared_train"].get("by_label"),
        )
    if "val_distribution" not in training_progress and dataset_summary:
        training_progress["val_distribution"] = analyze_class_balance(
            target_labels,
            dataset_summary["prepared_val"].get("by_label"),
        )
    if "train_distribution" not in metrics and dataset_summary:
        metrics["train_distribution"] = analyze_class_balance(
            target_labels,
            dataset_summary["prepared_train"].get("by_label"),
        )
    if "val_distribution" not in metrics and dataset_summary:
        metrics["val_distribution"] = analyze_class_balance(
            target_labels,
            dataset_summary["prepared_val"].get("by_label"),
        )
    diagnostics = build_overview_diagnostics(
        paths=paths,
        pipeline_diagnostics=pipeline_diagnostics,
        training_progress=training_progress,
        metrics=metrics,
        dataset_summary=dataset_summary,
        current_dataset_summary=current_dataset_summary,
        completed_jobs=enriched_completed_jobs,
    )
    metric_history = training_progress.get("history") or metrics.get("history") or []
    latest_metrics = training_progress.get("latest") or (
        metric_history[-1] if isinstance(metric_history, list) and metric_history else {}
    )
    final_validation = (
        metrics.get("final_validation")
        or training_progress.get("final_validation")
        or {}
    )
    insight_metrics = {
        **(metrics if isinstance(metrics, dict) else {}),
        "labels": target_labels,
        "history": metric_history if isinstance(metric_history, list) else [],
        "latest": latest_metrics if isinstance(latest_metrics, dict) else {},
        "final_validation": final_validation if isinstance(final_validation, dict) else {},
        "best_epoch": metrics.get("best_epoch") or training_progress.get("best_epoch"),
        "best_val_macro_f1": metrics.get("best_val_macro_f1") or training_progress.get("best_val_macro_f1"),
        "best_validation": metrics.get("best_validation") or training_progress.get("best_validation") or {},
    }
    insights = {} if lite else interpret_training_results(
        insight_metrics,
        history=insight_metrics["history"],
        class_report=insight_metrics["final_validation"].get("per_class") or [],
        confusion_matrix=insight_metrics["final_validation"].get("confusion_matrix") or [],
        data_stats={
            "labels": target_labels,
            "dataset": dataset_summary,
            "current_dataset": current_dataset_summary,
            "skip_report": current_skip_report,
            "cumulative_skip_report": cumulative_skip_report,
            "train_distribution": training_progress.get("train_distribution") or metrics.get("train_distribution") or {},
            "val_distribution": training_progress.get("val_distribution") or metrics.get("val_distribution") or {},
        },
    )
    predownload_log_payload = build_predownload_log_payload(
        effective_launcher_status.get("predownload") if isinstance(effective_launcher_status, dict) else {},
        get_cached_log_tail,
    )
    current_job_filekeys = build_current_job_filekey_summary(paths, current_job)

    final_model_summary = build_final_model_summary(paths, metrics, training_progress)
    task_performance_summary = build_task_performance_summary(paths, metrics)

    overview = {
        "overview_revision": overview_revision,
        "schema_version": STATE_SCHEMA_VERSION,
        "workspace_dir": str(paths["workspace_dir"]),
        "workspace_name": paths["workspace_dir"].name,
        "workspace": {
            "source": paths.get("workspace_source", "config"),
            "default_dir": str(paths.get("workspace_default_dir") or paths["workspace_dir"]),
            "override_env": paths.get("workspace_override_env"),
            "override_value": paths.get("workspace_override_value"),
        },
        "config_path": str(config_path),
        "target_labels": target_labels,
        "pipeline_status": pipeline_status,
        "training_progress": training_progress,
        "launcher": {
            **effective_launcher_status,
            "completed_jobs": enriched_completed_jobs[: (8 if lite else 30)],
        },
        "aihub": {
            "datasetkey": (
                (current_job.get("datasetkey") if isinstance(current_job, dict) else None)
                or (pending_jobs[0].get("datasetkey") if isinstance(pending_jobs, list) and pending_jobs and isinstance(pending_jobs[0], dict) else None)
                or config.get("aihub_shell", {}).get("datasetkey")
            ),
        },
        "queue_progress": {
            "total": total_count,
            "completed": completed_count,
            "prepared": prepared_count,
            "failed": failed_count,
            "pending": pending_count,
            "active": active_count,
            "ratio": round(progress_ratio, 4),
        },
        "current_job_progress": current_job_progress,
        "current_job_filekeys": current_job_filekeys,
        "eta": eta,
        "gpu": gpu_status,
        "logs": {
            "current": {
                "filekey": current_log_source.get("filekey") if isinstance(current_log_source, dict) else None,
                "datasetkey": current_log_source.get("datasetkey") if isinstance(current_log_source, dict) else None,
                "path": current_log_path,
                "tail": current_log_tail,
            },
            "latest_completed": {
                "filekey": latest_completed_job.get("filekey") if isinstance(latest_completed_job, dict) else None,
                "datasetkey": latest_completed_job.get("datasetkey") if isinstance(latest_completed_job, dict) else None,
                "path": latest_completed_job.get("log_path") if isinstance(latest_completed_job, dict) else None,
                "tail": get_cached_log_tail(
                    latest_completed_job.get("log_path") if isinstance(latest_completed_job, dict) else None
                ),
            },
            "latest_error": {
                "filekey": latest_error_job.get("filekey") if isinstance(latest_error_job, dict) else None,
                "datasetkey": latest_error_job.get("datasetkey") if isinstance(latest_error_job, dict) else None,
                "path": latest_error_log_path,
                "tail": latest_error_log_tail,
            },
            "predownload": predownload_log_payload,
        } if not lite else {},
        "dataset": dataset_summary,
        "distribution_sources": distribution_sources,
        "current_dataset": current_dataset_summary,
        "guideline_quality": guideline_quality,
        "prepared_pose_items": build_prepared_pose_items(paths, limit=100) if not lite else [],
        "continual_state": read_json(paths["continual_state"]),
        "artifacts": {
            "has_model": (paths["artifacts_dir"] / "best_action_model.pt").exists(),
            "has_metrics": (paths["artifacts_dir"] / "metrics.json").exists(),
            "has_labels": (paths["artifacts_dir"] / "labels.json").exists(),
        },
        "diagnostics": diagnostics,
        "final_model_summary": final_model_summary,
        "task_performance_summary": task_performance_summary,
        "insights": insights,
        "skip_report": current_skip_report,
        "cumulative_skip_report": cumulative_skip_report if not lite else {},
        "metrics": metrics if not lite else {},
    }
    with OVERVIEW_CACHE_LOCK:
        OVERVIEW_CACHE[signature] = overview
        while len(OVERVIEW_CACHE) > 4:
            oldest_key = next(iter(OVERVIEW_CACHE))
            if oldest_key == signature and len(OVERVIEW_CACHE) == 1:
                break
            OVERVIEW_CACHE.pop(oldest_key, None)
    return overview


def build_final_model_summary(paths: dict, metrics: dict, training_progress: dict) -> dict:
    artifacts_dir = paths.get("artifacts_dir")
    if not isinstance(artifacts_dir, Path):
        return {}

    pose_validation = {}
    if isinstance(metrics, dict):
        pose_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    if not pose_validation and isinstance(training_progress, dict):
        pose_validation = (
            training_progress.get("final_validation")
            if isinstance(training_progress.get("final_validation"), dict)
            else {}
        )

    hybrid_summary_path = artifacts_dir / "hybrid_summary.json"
    hybrid_summary = read_json(hybrid_summary_path) or {}
    hybrid_best = hybrid_summary.get("best_result") if isinstance(hybrid_summary.get("best_result"), dict) else {}
    promoted_hybrid_validation = {}
    promoted_hybrid_test = {}
    if str(metrics.get("model_type") or "").strip().lower() == "hybrid_ensemble":
        promoted_hybrid_validation = (
            metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
        )
        promoted_hybrid_test = metrics.get("holdout_test") if isinstance(metrics.get("holdout_test"), dict) else {}

    ensemble_summary_path = artifacts_dir / "ensemble_summary.json"
    ensemble_summary = read_json(ensemble_summary_path) or {}
    ensemble_best = ensemble_summary.get("best_result") if isinstance(ensemble_summary.get("best_result"), dict) else {}
    ensemble_metrics = (
        ensemble_best.get("metrics", {}).get("final_validation")
        if isinstance(ensemble_best.get("metrics"), dict)
        else {}
    )
    if not isinstance(ensemble_metrics, dict):
        ensemble_metrics = {}

    pose_entry = {
        "name": "Pose only",
        "kind": "pose",
        "available": bool(pose_validation),
        "accuracy": coerce_optional_float(pose_validation.get("accuracy")),
        "macro_f1": coerce_optional_float(pose_validation.get("macro_f1")),
        "macro_f1_supported": coerce_optional_float(pose_validation.get("macro_f1_supported")),
        "balanced_accuracy": coerce_optional_float(pose_validation.get("balanced_accuracy")),
        "source_path": str(artifacts_dir / "metrics.json"),
    }
    ensemble_entry = {
        "name": "Pose ensemble",
        "kind": "ensemble",
        "available": bool(ensemble_metrics),
        "accuracy": coerce_optional_float(ensemble_metrics.get("accuracy")),
        "macro_f1": coerce_optional_float(ensemble_metrics.get("macro_f1")),
        "macro_f1_supported": coerce_optional_float(ensemble_metrics.get("macro_f1_supported")),
        "balanced_accuracy": coerce_optional_float(ensemble_metrics.get("balanced_accuracy")),
        "source_path": str(ensemble_summary_path),
    }
    hybrid_entry = {
        "name": "RGB/I3D + Pose hybrid",
        "kind": "hybrid",
        "available": bool(hybrid_best or promoted_hybrid_validation),
        "accuracy": coerce_optional_float(hybrid_best.get("val_accuracy"))
        or coerce_optional_float(promoted_hybrid_validation.get("accuracy")),
        "macro_f1": coerce_optional_float(hybrid_best.get("macro_f1"))
        or coerce_optional_float(promoted_hybrid_validation.get("macro_f1")),
        "macro_f1_supported": coerce_optional_float(hybrid_best.get("macro_f1_supported"))
        or coerce_optional_float(promoted_hybrid_validation.get("macro_f1_supported")),
        "balanced_accuracy": coerce_optional_float(hybrid_best.get("balanced_accuracy"))
        or coerce_optional_float(promoted_hybrid_validation.get("balanced_accuracy")),
        "test_accuracy": coerce_optional_float(hybrid_best.get("test_accuracy"))
        or coerce_optional_float(promoted_hybrid_test.get("accuracy")),
        "test_macro_f1": coerce_optional_float(hybrid_best.get("test_macro_f1"))
        or coerce_optional_float(promoted_hybrid_test.get("macro_f1")),
        "test_macro_f1_supported": coerce_optional_float(hybrid_best.get("test_macro_f1_supported"))
        or coerce_optional_float(promoted_hybrid_test.get("macro_f1_supported")),
        "feature_model": hybrid_best.get("feature_model"),
        "feature_weight": coerce_optional_float(hybrid_best.get("feature_weight")),
        "neural_weight": coerce_optional_float(hybrid_best.get("neural_weight")),
        "class_bias_name": hybrid_best.get("class_bias_name"),
        "promoted": bool(hybrid_summary.get("promoted")),
        "source_path": str(hybrid_summary_path),
    }
    recommended = hybrid_entry if hybrid_entry["available"] else (ensemble_entry if ensemble_entry["available"] else pose_entry)
    return {
        "recommended_kind": recommended.get("kind"),
        "recommended_name": recommended.get("name"),
        "recommended_accuracy": recommended.get("test_accuracy") or recommended.get("accuracy"),
        "recommended_macro_f1": recommended.get("test_macro_f1") or recommended.get("macro_f1"),
        "recommended_macro_f1_supported": recommended.get("test_macro_f1_supported")
        or recommended.get("macro_f1_supported"),
        "pose": pose_entry,
        "ensemble": ensemble_entry,
        "hybrid": hybrid_entry,
        "note": "RGB/I3D feature까지 적용한 최종 판단 기준은 hybrid 항목입니다.",
    }


def build_task_performance_summary(paths: dict, metrics: dict) -> dict:
    artifacts_dir = paths.get("artifacts_dir")
    if not isinstance(artifacts_dir, Path):
        return {}
    labels = [str(label) for label in metrics.get("labels", [])] if isinstance(metrics, dict) else []
    final_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    holdout_test = metrics.get("holdout_test") if isinstance(metrics.get("holdout_test"), dict) else {}
    specialized_dir = artifacts_dir / "specialized_tasks"
    detection_metrics = read_json(specialized_dir / "detection" / "metrics.json") or {}
    classification_metrics = read_json(specialized_dir / "classification" / "metrics.json") or {}
    pose_classification_metrics = read_json(specialized_dir / "pose_classification" / "metrics.json") or {}
    specialized_summary = read_json(specialized_dir / "summary.json") or {}
    derived_validation = build_derived_task_metrics(final_validation, labels=labels)
    derived_test = build_derived_task_metrics(holdout_test, labels=labels) if holdout_test else {}
    return {
        "source": "specialized_feature_models",
        "summary_path": str(specialized_dir / "summary.json"),
        "updated_at": specialized_summary.get("updated_at") or specialized_summary.get("created_at"),
        "detection": compact_specialized_task_metrics(detection_metrics, "detection"),
        "classification": compact_specialized_task_metrics(
            classification_metrics,
            "classification",
            active_labels=labels,
        ),
        "pose_classification": compact_specialized_task_metrics(
            pose_classification_metrics,
            "pose_classification",
            active_labels=labels,
        ),
        "derived_from_final_model": {
            "validation": derived_validation,
            "holdout_test": derived_test,
            "source_path": str(artifacts_dir / "metrics.json"),
        },
        "note": (
            "감지 전용은 normal/abnormal 기준, 분류 전용은 abnormal 4클래스 기준입니다. "
            "specialized 결과가 없으면 현재 최종 모델의 confusion matrix에서 파생 지표를 함께 보여줍니다."
        ),
    }


def compact_specialized_task_metrics(payload: dict, task_name: str, *, active_labels: list[str] | None = None) -> dict:
    if not isinstance(payload, dict) or not payload.get("available"):
        return {
            "available": False,
            "task": task_name,
            "reason": payload.get("reason") if isinstance(payload, dict) else None,
        }
    best = payload.get("best_result") if isinstance(payload.get("best_result"), dict) else {}
    validation = best.get("validation") if isinstance(best.get("validation"), dict) else {}
    holdout = best.get("holdout_test") if isinstance(best.get("holdout_test"), dict) else {}
    labels = payload.get("labels") if isinstance(payload.get("labels"), list) else []
    if task_name in {"classification", "pose_classification"} and active_labels:
        expected_labels = [str(label) for label in active_labels if str(label) != "normal"]
        payload_labels = [str(label) for label in labels]
        if expected_labels and payload_labels != expected_labels:
            return {
                "available": False,
                "task": task_name,
                "reason": "stale_label_set",
                "labels": payload_labels,
                "expected_labels": expected_labels,
                "source_path": str(Path("specialized_tasks") / task_name / "metrics.json"),
            }
    if task_name in {"classification", "pose_classification"}:
        if isinstance(validation, dict) and "classification" not in validation:
            validation = {**validation, "classification": derive_abnormal_classification_metrics(validation, labels=labels)}
        if isinstance(holdout, dict) and holdout and "classification" not in holdout:
            holdout = {**holdout, "classification": derive_abnormal_classification_metrics(holdout, labels=labels)}
    return {
        "available": True,
        "task": task_name,
        "model": best.get("model"),
        "threshold": coerce_optional_float(best.get("threshold")),
        "score": coerce_optional_float(best.get("score")),
        "labels": labels,
        "train_samples": payload.get("train_samples"),
        "val_samples": payload.get("val_samples"),
        "test_samples": payload.get("test_samples"),
        "validation": validation,
        "holdout_test": holdout,
        "source_path": str(Path("specialized_tasks") / task_name / "metrics.json"),
    }


def build_derived_task_metrics(metrics_payload: dict, *, labels: list[str]) -> dict:
    if not isinstance(metrics_payload, dict) or not labels:
        return {}
    return {
        "detection": derive_detection_metrics(metrics_payload, labels=labels),
        "classification": derive_abnormal_classification_metrics(metrics_payload, labels=labels),
    }


def derive_detection_metrics(metrics_payload: dict, *, labels: list[str]) -> dict:
    confusion = metrics_payload.get("confusion_matrix")
    if not isinstance(confusion, list) or not confusion:
        return {}
    try:
        normal_index = labels.index("normal")
    except ValueError:
        normal_index = 0
    total = 0
    tn = fp = fn = tp = 0
    for row_index, row in enumerate(confusion):
        if not isinstance(row, list):
            continue
        for column_index, value in enumerate(row):
            count = int(coerce_float_default(value, 0.0))
            total += count
            actual_normal = row_index == normal_index
            predicted_normal = column_index == normal_index
            if actual_normal and predicted_normal:
                tn += count
            elif actual_normal and not predicted_normal:
                fp += count
            elif not actual_normal and predicted_normal:
                fn += count
            else:
                tp += count
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall / max(precision + recall, 1e-12)) if total else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": fp / max(fp + tn, 1),
        "miss_rate": fn / max(fn + tp, 1),
    }


def derive_abnormal_classification_metrics(metrics_payload: dict, *, labels: list[str]) -> dict:
    per_class = metrics_payload.get("per_class") if isinstance(metrics_payload.get("per_class"), list) else []
    danger_rows = [
        row
        for row in per_class
        if isinstance(row, dict) and str(row.get("label") or "") != "normal" and int(coerce_float_default(row.get("support"), 0.0)) > 0
    ]
    if not danger_rows:
        return {}
    supports = [int(coerce_float_default(row.get("support"), 0.0)) for row in danger_rows]
    f1_values = [float(coerce_float_default(row.get("f1"), 0.0)) for row in danger_rows]
    recall_values = [float(coerce_float_default(row.get("recall"), 0.0)) for row in danger_rows]
    precision_values = [float(coerce_float_default(row.get("precision"), 0.0)) for row in danger_rows]
    total_support = sum(supports)
    return {
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "macro_recall": sum(recall_values) / max(len(recall_values), 1),
        "macro_precision": sum(precision_values) / max(len(precision_values), 1),
        "weighted_f1": sum(f1 * support for f1, support in zip(f1_values, supports)) / max(total_support, 1),
        "support": total_support,
        "classes": [str(row.get("label") or "") for row in danger_rows],
    }


def coerce_optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_float_default(value, default: float = 0.0) -> float:
    result = coerce_optional_float(value)
    return default if result is None else result


def parse_filekeys(raw_value) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        tokens = raw_value
    else:
        text = str(raw_value).replace("{", " ").replace("}", " ")
        text = re.sub(r"(\d)\s*(~|[-–—])\s*(\d)", r"\1\2\3", text)
        tokens = re.split(r"[\s,;]+", text)

    cleaned: list[str] = []
    seen = set()
    for token in tokens:
        value = str(token).strip()
        if not value:
            continue

        match = FILEKEY_RANGE_PATTERN.fullmatch(value)
        expanded_values = [value]
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            step = 1 if end >= start else -1
            count = abs(end - start) + 1
            if count > MAX_FILEKEY_RANGE_SIZE:
                raise ValueError(
                    f"filekey 범위가 너무 큽니다: {value} (최대 {MAX_FILEKEY_RANGE_SIZE}개까지 허용)"
                )
            expanded_values = [str(number) for number in range(start, end + step, step)]

        for expanded in expanded_values:
            if not expanded or expanded in seen:
                continue
            seen.add(expanded)
            cleaned.append(expanded)
    return cleaned


FILEKEY_MANIFEST_KEYS = (
    "raw_manifest",
    "split_train",
    "split_val",
    "split_test",
    "prepared_train",
    "prepared_val",
    "prepared_test",
    "guideline_prepared_train",
    "guideline_prepared_val",
    "guideline_prepared_test",
    "active_prepared_train",
    "active_prepared_val",
    "active_prepared_test",
    "current_raw_manifest",
    "current_split_train",
    "current_split_val",
    "current_split_test",
    "current_prepared_train",
    "current_prepared_val",
    "current_prepared_test",
)


FILEKEY_VALUE_FIELDS = (
    "filekey",
    "file_key",
    "source_filekey",
    "job_filekey",
    "dataset_filekey",
    "rgb_feature_filekey",
    "annotation_cache_filekey",
)


FILEKEY_PATH_FIELDS = (
    "video_path",
    "source_path",
    "xml_path",
    "annotation_path",
    "cached_xml_path",
    "pose_path",
    "feature_path",
    "rgb_feature_path",
    "relative_path",
)


def _iter_filekey_candidate_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        candidates: list[str] = []
        for item in value:
            candidates.extend(_iter_filekey_candidate_values(item))
        return candidates
    if isinstance(value, dict):
        candidates = []
        for nested in value.values():
            candidates.extend(_iter_filekey_candidate_values(nested))
        return candidates
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[\s,;]+", text) if part.strip()]
    return parts or [text]


def manifest_row_matches_filekey(row: dict, filekey: str) -> bool:
    target = str(filekey or "").strip()
    if not target or not isinstance(row, dict):
        return False

    containers = [row]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        containers.append(metadata)

    for container in containers:
        for field in FILEKEY_VALUE_FIELDS:
            for candidate in _iter_filekey_candidate_values(container.get(field)):
                if candidate == target:
                    return True

    path_tokens = (
        f"/job_{target}/",
        f"job_{target}",
        f"__{target}/",
        f"/{target}/",
        f"\\{target}\\",
        f"\\job_{target}\\",
    )
    for container in containers:
        for field in FILEKEY_PATH_FIELDS:
            raw_value = container.get(field)
            if raw_value is None:
                continue
            normalized = str(raw_value).replace("\\", "/")
            if any(token.replace("\\", "/") in normalized for token in path_tokens):
                return True
    return False


def reset_filekey_training_outputs(paths: dict, *, datasetkey: str, filekey: str) -> dict:
    target_filekey = str(filekey or "").strip()
    if not target_filekey:
        return {"filekey": target_filekey, "removed_rows": {}, "removed_paths": []}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = paths.get("manifests_dir")
    backup_dir = None
    removed_rows: dict[str, int] = {}
    removed_paths: list[str] = []

    if isinstance(manifest_dir, Path):
        backup_dir = manifest_dir / f"backup_before_reextract_{target_filekey}_{timestamp}"
        for key in FILEKEY_MANIFEST_KEYS:
            manifest_path = paths.get(key)
            if not isinstance(manifest_path, Path) or not manifest_path.exists():
                continue
            entries = read_jsonl_entries(manifest_path)
            kept = [entry for entry in entries if not manifest_row_matches_filekey(entry, target_filekey)]
            removed_count = len(entries) - len(kept)
            if removed_count <= 0:
                continue
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, backup_dir / manifest_path.name)
            write_jsonl_entries(manifest_path, kept)
            removed_rows[key] = removed_count

        active_state = paths.get("active_manifest_state")
        if isinstance(active_state, Path) and active_state.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(active_state, backup_dir / active_state.name)
            remove_transient_path_with_retries(active_state)

    datasetkey_text = str(datasetkey or "").strip()
    prepared_dir = paths.get("prepared_dir")
    if isinstance(prepared_dir, Path):
        prepared_targets = [prepared_dir / target_filekey]
        if datasetkey_text:
            prepared_targets.append(prepared_dir / f"{datasetkey_text}__{target_filekey}")
        for target in prepared_targets:
            if target.exists() and remove_transient_path_with_retries(target):
                removed_paths.append(str(target))

    xml_cache_dir = paths.get("xml_cache_dir")
    if isinstance(xml_cache_dir, Path):
        target = xml_cache_dir / f"job_{target_filekey}"
        if target.exists() and remove_transient_path_with_retries(target):
            removed_paths.append(str(target))

    workspace_dir = paths.get("workspace_dir")
    rgb_root = workspace_dir / "rgb_clip_features" if isinstance(workspace_dir, Path) else None
    if isinstance(rgb_root, Path) and rgb_root.exists():
        for model_dir in rgb_root.iterdir():
            if not model_dir.is_dir():
                continue
            for split_dir in model_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                target = split_dir / target_filekey
                if target.exists() and remove_transient_path_with_retries(target):
                    removed_paths.append(str(target))

    summary = {
        "filekey": target_filekey,
        "datasetkey": datasetkey_text,
        "reset_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "removed_rows": removed_rows,
        "removed_paths": removed_paths,
        "backup_dir": str(backup_dir) if backup_dir and backup_dir.exists() else None,
    }
    if isinstance(manifest_dir, Path):
        write_json_atomic(
            manifest_dir / f"reextract_reset_{target_filekey}_{timestamp}.json",
            summary,
        )
    print(
        f"[dashboard] filekey {target_filekey} reextract reset: "
        f"rows={sum(removed_rows.values())}, paths={len(removed_paths)}"
    )
    return summary


def reset_training_workspace(paths: dict) -> None:
    cleanup_transient_job_data(paths)
    current_skip_report = paths.get("current_skip_report")
    if isinstance(current_skip_report, Path) and current_skip_report.exists():
        try:
            current_skip_report.unlink()
        except OSError:
            pass

    for key in ("manifests_dir", "prepared_dir", "artifacts_dir"):
        target = paths.get(key)
        if isinstance(target, Path):
            target.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    app = create_app(config_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

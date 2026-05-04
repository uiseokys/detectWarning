from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

from gpu_autotune import apply_gpu_auto_tune
from reporting import write_json_atomic
from training_config import load_action_training_config


@dataclass
class DownloadedItem:
    item_id: str
    source_label: str
    target_label: str
    video_path: Path
    download_url: str
    metadata: dict


DOWNLOAD_PROGRESS_PATTERN = re.compile(
    r"(?:^|\s)(?P<percent>\d{1,3})(?:%|\s+\S+\s+\d+\s+\S+)"
)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SIZE_TOKEN_PATTERN = re.compile(r"^\d+(?:\.\d+)?(?:[KMGTP]i?B?|B)$", re.IGNORECASE)
DOWNLOAD_STATUS_MIN_INTERVAL_SECONDS = 2.0
DOWNLOAD_STATUS_MIN_BYTES_DELTA = 128 * 1024 * 1024
AIHUB_FILE_TREE_REQUEST_TIMEOUT_SECONDS = 60
AIHUB_FILE_TREE_MAX_RETRIES = 3
AIHUB_FILE_TREE_RETRY_BACKOFF_SECONDS = 1.5
AIHUB_FILE_TREE_HEADERS = {
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    "User-Agent": "detectWarning/1.0 (+https://api.aihub.or.kr)",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
AIHUB_FILE_TREE_CATEGORY_PATTERN = re.compile(
    r"[├└]\s*─?\s*(?P<index>\d{2})\.(?P<label>[^()|\r\n]+?)\((?P<alias>[A-Za-z0-9_ -]+)\)"
)
AIHUB_FILE_TREE_FILE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]+\.zip)\s*\|\s*[^|]+\|\s*(?P<filekey>\d{4,})\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="API에서 행동 영상을 받아 pose 기반 행동 분류 학습까지 자동으로 수행합니다."
    )
    parser.add_argument(
        "--config",
        default="configs/action_training.example.json",
        help="학습 파이프라인 설정 JSON 경로",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "download", "prepare", "train"),
        help="실행할 단계",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    return load_action_training_config(config_path)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    validate_source_config(config, config_path)
    config = apply_gpu_auto_tune(config, stage=args.stage, logger=print)
    paths = resolve_paths(config, config_path.parent)
    continual_config = get_continual_config(config)
    pipeline_started_at = current_timestamp_iso()
    stage_timings: dict[str, dict] = {}
    active_stage_started_at = pipeline_started_at
    training_deferred = False
    training_defer_message = ""
    training_defer_coverage: dict | None = None
    write_pipeline_status(
        paths,
        stage="starting",
        state="running",
        message="학습 파이프라인을 시작합니다.",
        config_path=str(config_path),
        stage_progress=0.0,
        pipeline_started_at=pipeline_started_at,
        stage_started_at=active_stage_started_at,
        stage_timings=stage_timings,
    )

    try:
        downloaded: list[DownloadedItem] = []
        split_manifests: dict[str, Path] | None = None
        training_manifests: dict[str, Path] | None = None

        if args.stage in {"all", "download"}:
            active_stage_started_at = current_timestamp_iso()
            write_pipeline_status(
                paths,
                stage="download",
                state="running",
                message="API에서 영상을 다운로드하는 중입니다.",
                stage_progress=0.05,
                pipeline_started_at=pipeline_started_at,
                stage_started_at=active_stage_started_at,
                stage_timings=stage_timings,
            )
            downloaded = download_dataset(config, paths)
            stage_timings["download"] = build_stage_timing_entry(active_stage_started_at, current_timestamp_iso())
            split_manifests = split_dataset(downloaded, config, paths)
        elif args.stage == "prepare":
            split_manifests = load_existing_current_split_manifests(paths)

        if args.stage in {"all", "prepare"}:
            if split_manifests is None:
                split_manifests = load_existing_current_split_manifests(paths)
            active_stage_started_at = current_timestamp_iso()
            write_pipeline_status(
                paths,
                stage="prepare",
                state="running",
                message="영상에서 pose 시퀀스를 추출하는 중입니다.",
                stage_progress=0.55,
                pipeline_started_at=pipeline_started_at,
                stage_started_at=active_stage_started_at,
                stage_timings=stage_timings,
            )
            prepared_manifests = prepare_pose_dataset(config, paths, split_manifests)
            stage_timings["prepare"] = build_stage_timing_entry(active_stage_started_at, current_timestamp_iso())
            training_manifests = update_cumulative_manifests(config, paths, split_manifests, prepared_manifests)
        elif args.stage == "train":
            training_manifests = load_training_manifests(
                config,
                paths,
                continual_enabled=continual_config["enabled"],
            )

        if args.stage in {"all", "train"}:
            if training_manifests is None:
                training_manifests = load_training_manifests(
                    config,
                    paths,
                    continual_enabled=continual_config["enabled"],
                )
            labels = get_target_labels(config)
            train_manifest = training_manifests["train"]
            val_manifest = training_manifests["val"]
            coverage_report = training_class_coverage_report(config, training_manifests, labels=labels)

            if should_defer_training_for_class_coverage(config, coverage_report, stage=args.stage):
                active_stage_started_at = current_timestamp_iso()
                stage_timings["data_ready"] = build_stage_timing_entry(
                    active_stage_started_at,
                    current_timestamp_iso(),
                )
                training_deferred = True
                training_defer_message = build_training_class_coverage_message(
                    coverage_report,
                    prefix="현재 filekey 데이터는 누적했지만 학습은 아직 시작하지 않았습니다.",
                    suffix=(
                        "다음 outside filekey를 계속 누적하고, train/val에 최소 클래스 수가 채워지면 "
                        "자동으로 모델 학습을 시작합니다."
                    ),
                )
                print(f"[train] deferred: {training_defer_message}")
                training_defer_coverage = coverage_report
                if args.stage == "all" and continual_config["cleanup_raw_after_job"]:
                    cleanup_transient_job_data(paths)
            else:
                active_stage_started_at = current_timestamp_iso()
                write_pipeline_status(
                    paths,
                    stage="train",
                    state="running",
                    message="행동 분류 모델을 학습하는 중입니다.",
                    stage_progress=0.8,
                    pipeline_started_at=pipeline_started_at,
                    stage_started_at=active_stage_started_at,
                    stage_timings=stage_timings,
                    class_coverage=coverage_report,
                )
                validate_training_class_coverage(config, training_manifests, labels=labels)
                adaptive_class_weight_multipliers = resolve_adaptive_class_weight_multipliers(
                    config,
                    paths,
                    labels=labels,
                )
                config.setdefault("training", {})["class_weight_multipliers"] = adaptive_class_weight_multipliers
                if adaptive_class_weight_multipliers:
                    print(f"[train] adaptive class weight multipliers: {adaptive_class_weight_multipliers}")
                resume_from = None
                if continual_config["enabled"] and continual_config["resume_from_best"]:
                    candidate_checkpoint = paths["artifacts_dir"] / "best_action_model.pt"
                    if candidate_checkpoint.exists() and checkpoint_is_safe_for_resume(config, paths, labels=labels):
                        resume_from = candidate_checkpoint
                    elif candidate_checkpoint.exists():
                        print("[train] 기존 best checkpoint는 단일/부족 클래스 학습 결과라 resume을 건너뜁니다.")
                from action_model import train_action_classifier

                artifacts = train_action_classifier(
                    train_manifest=train_manifest,
                    val_manifest=val_manifest,
                    output_dir=paths["artifacts_dir"],
                    labels=labels,
                    epochs=int(config.get("training", {}).get("epochs", 20)),
                    batch_size=int(config.get("training", {}).get("batch_size", 16)),
                    eval_batch_size=int(config.get("training", {}).get("eval_batch_size", 0) or 0),
                    learning_rate=float(config.get("training", {}).get("learning_rate", 1e-3)),
                    weight_decay=float(config.get("training", {}).get("weight_decay", 1e-3)),
                    hidden_dim=int(config.get("training", {}).get("hidden_dim", 128)),
                    num_layers=int(config.get("training", {}).get("num_layers", 2)),
                    dropout=float(config.get("training", {}).get("dropout", 0.2)),
                    label_smoothing=float(config.get("training", {}).get("label_smoothing", 0.05)),
                    loss_name=str(config.get("training", {}).get("loss", "cross_entropy")),
                    focal_gamma=float(config.get("training", {}).get("focal_gamma", 2.0)),
                    class_weight=config.get("training", {}).get("class_weight", "balanced"),
                    class_weight_multipliers=adaptive_class_weight_multipliers,
                    balanced_sampler=config.get("training", {}).get("balanced_sampler", "auto"),
                    grad_clip_norm=float(config.get("training", {}).get("grad_clip_norm", 1.0)),
                    seed=config.get("training", {}).get("seed", config.get("split", {}).get("seed", 42)),
                    deterministic=bool(config.get("training", {}).get("deterministic", False)),
                    num_workers=config.get("training", {}).get("num_workers", "auto"),
                    device=str(config.get("training", {}).get("device", "cuda")),
                    amp=bool(config.get("training", {}).get("amp", True)),
                    amp_dtype=str(config.get("training", {}).get("amp_dtype", "auto")),
                    compile_model=bool(config.get("training", {}).get("compile_model", True)),
                    compile_backend=str(config.get("training", {}).get("compile_backend", "auto")),
                    dataset_cache_size=int(config.get("training", {}).get("dataset_cache_size", 2048)),
                    prefetch_factor=int(config.get("training", {}).get("prefetch_factor", 2)),
                    persistent_workers=bool(config.get("training", {}).get("persistent_workers", True)),
                    pin_memory=config.get("training", {}).get("pin_memory", "auto"),
                    early_stopping_patience=int(config.get("training", {}).get("early_stopping_patience", 5)),
                    early_stopping_min_delta=float(config.get("training", {}).get("early_stopping_min_delta", 0.001)),
                    imbalance_warn_min_samples=int(config.get("training", {}).get("imbalance_warn_min_samples", 8)),
                    imbalance_warn_ratio=float(config.get("training", {}).get("imbalance_warn_ratio", 5.0)),
                    progress_path=paths["training_progress"],
                    resume_from=resume_from,
                )
                stage_timings["train"] = build_stage_timing_entry(active_stage_started_at, current_timestamp_iso())
                print(f"[train] best model: {artifacts.best_model_path}")
                print(f"[train] metrics: {artifacts.metrics_path}")
                print(f"[train] labels: {artifacts.labels_path}")

                if args.stage == "all" and continual_config["cleanup_raw_after_job"]:
                    cleanup_transient_job_data(paths)

        stage_timings["total"] = build_stage_timing_entry(pipeline_started_at, current_timestamp_iso())
        if training_deferred:
            write_pipeline_status(
                paths,
                stage="data_ready",
                state="data_ready",
                message=training_defer_message,
                stage_progress=1.0,
                pipeline_started_at=pipeline_started_at,
                stage_started_at=active_stage_started_at,
                stage_timings=stage_timings,
                total_duration_seconds=stage_timings["total"]["duration_seconds"],
                class_coverage=training_defer_coverage,
            )
        else:
            write_pipeline_status(
                paths,
                stage="completed",
                state="completed",
                message="학습 파이프라인이 완료되었습니다.",
                stage_progress=1.0,
                pipeline_started_at=pipeline_started_at,
                stage_started_at=active_stage_started_at,
                stage_timings=stage_timings,
                total_duration_seconds=stage_timings["total"]["duration_seconds"],
            )
    except Exception as exc:
        stage_timings["total"] = build_stage_timing_entry(pipeline_started_at, current_timestamp_iso())
        write_pipeline_status(
            paths,
            stage="error",
            state="error",
            message=str(exc),
            stage_progress=1.0,
            pipeline_started_at=pipeline_started_at,
            stage_started_at=active_stage_started_at,
            stage_timings=stage_timings,
            total_duration_seconds=stage_timings["total"]["duration_seconds"],
        )
        raise


def resolve_paths(config: dict, base_dir: Path) -> dict:
    paths_config = config.get("paths", {})
    workspace_dir_setting = paths_config.get("workspace_dir", "training_data/action_pipeline")
    workspace_dir_env = str(paths_config.get("workspace_dir_env", "DETECTWARNING_WORKSPACE_DIR") or "").strip()
    workspace_dir_override = os.environ.get(workspace_dir_env, "").strip() if workspace_dir_env else ""
    workspace_path = Path(workspace_dir_setting).expanduser()
    workspace_dir_default = (
        workspace_path.resolve()
        if workspace_path.is_absolute()
        else (base_dir / workspace_path).resolve()
    )

    if workspace_dir_override:
        workspace_dir = Path(workspace_dir_override).expanduser().resolve()
        workspace_source = "env"
    else:
        workspace_dir = workspace_dir_default
        workspace_source = "config"
    raw_dir = workspace_dir / "raw_videos"
    import_dir = workspace_dir / "imported_dataset"
    extracted_dir = workspace_dir / "extracted_dataset"
    manifests_dir = workspace_dir / "manifests"
    prepared_dir = workspace_dir / "prepared_pose"
    artifacts_dir = workspace_dir / "artifacts"
    for path in (workspace_dir, raw_dir, import_dir, extracted_dir, manifests_dir, prepared_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "workspace_dir": workspace_dir,
        "workspace_default_dir": workspace_dir_default,
        "workspace_source": workspace_source,
        "workspace_override_env": workspace_dir_env,
        "workspace_override_value": workspace_dir_override or None,
        "raw_dir": raw_dir,
        "import_dir": import_dir,
        "extracted_dir": extracted_dir,
        "manifests_dir": manifests_dir,
        "prepared_dir": prepared_dir,
        "artifacts_dir": artifacts_dir,
        "pipeline_status": workspace_dir / "pipeline_status.json",
        "training_progress": artifacts_dir / "training_progress.json",
        "raw_manifest": manifests_dir / "cumulative_raw_items.jsonl",
        "split_train": manifests_dir / "cumulative_split_train.jsonl",
        "split_val": manifests_dir / "cumulative_split_val.jsonl",
        "split_test": manifests_dir / "cumulative_split_test.jsonl",
        "prepared_train": manifests_dir / "cumulative_prepared_train.jsonl",
        "prepared_val": manifests_dir / "cumulative_prepared_val.jsonl",
        "prepared_test": manifests_dir / "cumulative_prepared_test.jsonl",
        "active_prepared_train": manifests_dir / "active_prepared_train.jsonl",
        "active_prepared_val": manifests_dir / "active_prepared_val.jsonl",
        "active_prepared_test": manifests_dir / "active_prepared_test.jsonl",
        "current_raw_manifest": manifests_dir / "current_raw_items.jsonl",
        "current_split_train": manifests_dir / "current_split_train.jsonl",
        "current_split_val": manifests_dir / "current_split_val.jsonl",
        "current_split_test": manifests_dir / "current_split_test.jsonl",
        "current_prepared_train": manifests_dir / "current_prepared_train.jsonl",
        "current_prepared_val": manifests_dir / "current_prepared_val.jsonl",
        "current_prepared_test": manifests_dir / "current_prepared_test.jsonl",
        "current_skip_report": manifests_dir / "current_skipped_videos.json",
        "cumulative_skip_report": manifests_dir / "cumulative_skipped_videos.json",
        "continual_state": manifests_dir / "continual_state.json",
        "active_manifest_state": manifests_dir / "active_manifest_state.json",
    }


def write_pipeline_status(paths: dict, *, stage: str, state: str, message: str, **extra) -> None:
    existing = read_json_file(paths["pipeline_status"]) or {}
    payload = {
        "stage": stage,
        "state": state,
        "message": message,
        "workspace_dir": str(paths["workspace_dir"]),
        "updated_at": datetime.now(UTC).astimezone().isoformat(),
        "pipeline_started_at": extra.pop("pipeline_started_at", existing.get("pipeline_started_at")),
        "stage_started_at": extra.pop("stage_started_at", existing.get("stage_started_at")),
        "stage_timings": extra.pop("stage_timings", existing.get("stage_timings") or {}),
        "total_duration_seconds": extra.pop("total_duration_seconds", existing.get("total_duration_seconds")),
        **extra,
    }
    write_json_atomic(paths["pipeline_status"], payload)
    print(format_pipeline_status_log(payload))


def current_timestamp_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat()


def build_stage_timing_entry(started_at: str | None, finished_at: str | None) -> dict:
    duration_seconds = None
    if started_at and finished_at:
        try:
            start_dt = datetime.fromisoformat(str(started_at))
            finish_dt = datetime.fromisoformat(str(finished_at))
            duration_seconds = max(0.0, (finish_dt - start_dt).total_seconds())
        except ValueError:
            duration_seconds = None
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
    }


def create_skip_report() -> dict:
    return {
        "updated_at": datetime.now(UTC).astimezone().isoformat(),
        "summary": {
            "total_issues": 0,
            "broken_count": 0,
            "skipped_count": 0,
            "by_reason": {},
            "by_split": {},
            "by_label": {},
        },
        "issues": [],
    }


def append_skip_issue(
    report: dict,
    *,
    category: str,
    split_name: str,
    video_path: Path,
    reason: str,
    target_label: str | None = None,
    detail: str | None = None,
    valid_frames: int | None = None,
    confirmed_frames: int | None = None,
    total_valid_keypoints: int | None = None,
    recovery_actions: list[str] | None = None,
) -> None:
    normalized_label = str(target_label or "unknown").strip() or "unknown"
    issue = {
        "category": category,
        "split": split_name,
        "target_label": normalized_label,
        "video_name": video_path.name,
        "video_path": str(video_path),
        "reason": reason,
        "detail": detail,
        "valid_frames": valid_frames,
        "confirmed_frames": confirmed_frames,
        "total_valid_keypoints": total_valid_keypoints,
        "recovery_actions": recovery_actions or [],
        "created_at": datetime.now(UTC).astimezone().isoformat(),
    }
    issues = report.setdefault("issues", [])
    issues.append(issue)
    summary = report.setdefault("summary", {})
    summary["total_issues"] = int(summary.get("total_issues", 0) or 0) + 1
    if category == "broken":
        summary["broken_count"] = int(summary.get("broken_count", 0) or 0) + 1
    else:
        summary["skipped_count"] = int(summary.get("skipped_count", 0) or 0) + 1
    increment_nested_counter(summary, "by_reason", reason)
    increment_nested_counter(summary, "by_split", split_name)
    increment_nested_counter(summary, "by_label", normalized_label)


def increment_nested_counter(payload: dict, key: str, item: str) -> None:
    counters = payload.setdefault(key, {})
    item_key = str(item or "unknown")
    counters[item_key] = int(counters.get(item_key, 0) or 0) + 1


def build_skip_issue_key(issue: dict) -> str:
    return "|".join(
        [
            str(issue.get("category") or ""),
            str(issue.get("video_path") or ""),
            str(issue.get("reason") or ""),
        ]
    )


def read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_adaptive_class_weight_multipliers(config: dict, paths: dict, *, labels: list[str]) -> dict[str, float]:
    training_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    base_multipliers = normalize_class_weight_multiplier_map(
        training_config.get("class_weight_multipliers"),
        labels=labels,
    )
    adaptive_config = training_config.get("adaptive_class_weighting", {})
    if isinstance(adaptive_config, bool):
        adaptive_config = {"enabled": adaptive_config}
    if not isinstance(adaptive_config, dict) or not bool(adaptive_config.get("enabled", False)):
        return base_multipliers

    payloads = []
    artifacts_dir = paths.get("artifacts_dir")
    metric_paths = [paths.get("training_progress")]
    if isinstance(artifacts_dir, Path):
        metric_paths.append(artifacts_dir / "metrics.json")
    for path in metric_paths:
        payload = read_json_file(path) if isinstance(path, Path) else None
        if isinstance(payload, dict):
            payloads.append(payload)

    return build_adaptive_class_weight_multipliers(
        config,
        labels=labels,
        metric_payloads=payloads,
        base_multipliers=base_multipliers,
    )


def build_adaptive_class_weight_multipliers(
    config: dict,
    *,
    labels: list[str],
    metric_payloads: list[dict] | tuple[dict, ...] | None = None,
    base_multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    training_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    adaptive_config = training_config.get("adaptive_class_weighting", {})
    if isinstance(adaptive_config, bool):
        adaptive_config = {"enabled": adaptive_config}
    if not isinstance(adaptive_config, dict) or not bool(adaptive_config.get("enabled", False)):
        return normalize_class_weight_multiplier_map(base_multipliers or training_config.get("class_weight_multipliers"), labels=labels)

    multipliers = normalize_class_weight_multiplier_map(
        base_multipliers or training_config.get("class_weight_multipliers"),
        labels=labels,
    )
    payload = select_metric_payload_for_adaptive_weights(metric_payloads or [])
    if not payload:
        return multipliers

    min_multiplier = safe_float(adaptive_config.get("min_multiplier"), 1.0)
    max_multiplier = safe_float(adaptive_config.get("max_multiplier"), 2.5)
    if max_multiplier < min_multiplier:
        min_multiplier, max_multiplier = max_multiplier, min_multiplier
    min_multiplier = max(min_multiplier, 0.25)
    max_multiplier = min(max(max_multiplier, min_multiplier), 8.0)
    target_recall = max(safe_float(adaptive_config.get("target_recall"), 0.55), 1e-6)
    target_f1 = max(safe_float(adaptive_config.get("target_f1"), 0.45), 1e-6)
    low_recall_multiplier = clamp_float(
        safe_float(adaptive_config.get("low_recall_multiplier"), 2.0),
        min_multiplier,
        max_multiplier,
    )
    low_f1_multiplier = clamp_float(
        safe_float(adaptive_config.get("low_f1_multiplier"), 1.6),
        min_multiplier,
        max_multiplier,
    )
    missing_prediction_multiplier = clamp_float(
        safe_float(adaptive_config.get("missing_prediction_multiplier"), max_multiplier),
        min_multiplier,
        max_multiplier,
    )
    minority_count_multiplier = clamp_float(
        safe_float(adaptive_config.get("minority_count_multiplier"), 1.25),
        min_multiplier,
        max_multiplier,
    )
    min_validation_support = max(int(safe_float(adaptive_config.get("min_validation_support"), 1)), 0)

    final_validation = payload.get("final_validation") if isinstance(payload.get("final_validation"), dict) else {}
    rows_by_label = class_report_by_label(final_validation.get("per_class"), labels=labels)
    analysis = payload.get("validation_error_analysis") if isinstance(payload.get("validation_error_analysis"), dict) else {}
    missing_prediction_classes = {str(label) for label in analysis.get("missing_prediction_classes") or []}
    confusion_supports, confusion_predictions = confusion_support_and_prediction_counts(
        final_validation.get("confusion_matrix"),
        labels=labels,
    )
    missing_prediction_classes.update(
        infer_missing_prediction_classes_from_confusion(
            final_validation.get("confusion_matrix"),
            labels=labels,
        )
    )
    train_counts = distribution_counts_by_label(payload.get("train_distribution"))
    max_train_count = max(train_counts.values()) if train_counts else 0

    for label in labels:
        label_key = str(label)
        current_multiplier = multipliers.get(label_key, 1.0)
        row = rows_by_label.get(label_key, {})
        support = int(safe_float(row.get("support"), float(confusion_supports.get(label_key, 0))))
        predicted = int(safe_float(row.get("predicted"), float(confusion_predictions.get(label_key, 1))))
        if support >= min_validation_support:
            recall = safe_optional_float(row.get("recall"))
            f1 = safe_optional_float(row.get("f1"))
            if label_key in missing_prediction_classes or predicted <= 0:
                current_multiplier = max(current_multiplier, missing_prediction_multiplier)
            if recall is not None and recall < target_recall:
                severity = clamp_float((target_recall - recall) / target_recall, 0.0, 1.0)
                current_multiplier = max(
                    current_multiplier,
                    1.0 + severity * (low_recall_multiplier - 1.0),
                )
            if f1 is not None and f1 < target_f1:
                severity = clamp_float((target_f1 - f1) / target_f1, 0.0, 1.0)
                current_multiplier = max(
                    current_multiplier,
                    1.0 + severity * (low_f1_multiplier - 1.0),
                )

        train_count = int(train_counts.get(label_key, 0) or 0)
        if max_train_count > 0 and train_count > 0 and train_count < max_train_count:
            count_ratio = max_train_count / max(train_count, 1)
            if count_ratio >= 1.25:
                current_multiplier = max(
                    current_multiplier,
                    min(minority_count_multiplier, math.sqrt(count_ratio)),
                )

        current_multiplier = clamp_float(current_multiplier, min_multiplier, max_multiplier)
        if abs(current_multiplier - 1.0) > 1e-6:
            multipliers[label_key] = round(current_multiplier, 6)
        else:
            multipliers.pop(label_key, None)

    return dict(sorted(multipliers.items(), key=lambda item: labels.index(item[0]) if item[0] in labels else len(labels)))


def select_metric_payload_for_adaptive_weights(metric_payloads: list[dict] | tuple[dict, ...]) -> dict | None:
    for payload in metric_payloads:
        if not isinstance(payload, dict):
            continue
        final_validation = payload.get("final_validation")
        if isinstance(final_validation, dict) and final_validation.get("per_class"):
            return payload
    return None


def normalize_class_weight_multiplier_map(value, *, labels: list[str]) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    label_set = {str(label) for label in labels}
    normalized: dict[str, float] = {}
    for label, raw_multiplier in value.items():
        label_key = str(label or "").strip()
        if label_key not in label_set:
            continue
        multiplier = safe_optional_float(raw_multiplier)
        if multiplier is None:
            continue
        multiplier = clamp_float(multiplier, 0.25, 8.0)
        if abs(multiplier - 1.0) > 1e-6:
            normalized[label_key] = round(multiplier, 6)
    return normalized


def class_report_by_label(class_report, *, labels: list[str]) -> dict[str, dict]:
    if not isinstance(class_report, list):
        return {}
    rows: dict[str, dict] = {}
    for row in class_report:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        if label in (None, ""):
            class_index = int(safe_float(row.get("class_index"), -1.0))
            if 0 <= class_index < len(labels):
                label = labels[class_index]
        if label in (None, ""):
            continue
        rows[str(label)] = row
    return rows


def distribution_counts_by_label(distribution) -> dict[str, int]:
    if not isinstance(distribution, dict):
        return {}
    counts = distribution.get("counts")
    if isinstance(counts, list):
        result = {}
        for row in counts:
            if isinstance(row, dict) and row.get("label") not in (None, ""):
                result[str(row.get("label"))] = int(safe_float(row.get("count"), 0.0))
        return result
    by_label = distribution.get("by_label")
    if isinstance(by_label, dict):
        return {str(label): int(safe_float(count, 0.0)) for label, count in by_label.items()}
    return {}


def confusion_support_and_prediction_counts(confusion_matrix, *, labels: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    supports = {str(label): 0 for label in labels}
    predictions = {str(label): 0 for label in labels}
    if not isinstance(confusion_matrix, list):
        return supports, predictions
    for row_index, row in enumerate(confusion_matrix):
        if not isinstance(row, list):
            continue
        if row_index < len(labels):
            supports[str(labels[row_index])] = sum(int(safe_float(value, 0.0)) for value in row)
        for column_index, value in enumerate(row):
            if column_index < len(labels):
                predictions[str(labels[column_index])] += int(safe_float(value, 0.0))
    return supports, predictions


def safe_optional_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def safe_float(value, default: float) -> float:
    result = safe_optional_float(value)
    return default if result is None else result


def clamp_float(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def write_skip_reports(paths: dict, report: dict) -> None:
    report["updated_at"] = datetime.now(UTC).astimezone().isoformat()
    write_json_atomic(paths["current_skip_report"], report)

    cumulative = read_json_file(paths["cumulative_skip_report"]) or create_skip_report()
    cumulative_issues = cumulative.setdefault("issues", [])
    seen = {build_skip_issue_key(issue) for issue in cumulative_issues if isinstance(issue, dict)}
    for issue in report.get("issues", []):
        if not isinstance(issue, dict):
            continue
        issue_key = build_skip_issue_key(issue)
        if issue_key in seen:
            continue
        cumulative_issues.append(issue)
        seen.add(issue_key)

    cumulative_issues[:] = cumulative_issues[-1000:]
    cumulative["updated_at"] = report["updated_at"]
    cumulative["summary"] = build_skip_issue_summary(cumulative_issues)
    write_json_atomic(paths["cumulative_skip_report"], cumulative)


def build_skip_issue_summary(issues: list[dict]) -> dict:
    by_reason: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    for issue in issues:
        by_reason[str(issue.get("reason") or "unknown")] += 1
        by_split[str(issue.get("split") or "unknown")] += 1
        by_label[str(issue.get("target_label") or "unknown")] += 1
    return {
        "total_issues": len(issues),
        "broken_count": sum(1 for issue in issues if issue.get("category") == "broken"),
        "skipped_count": sum(1 for issue in issues if issue.get("category") != "broken"),
        "by_reason": dict(sorted(by_reason.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_label": dict(sorted(by_label.items())),
    }


def format_pipeline_status_log(payload: dict) -> str:
    stage = str(payload.get("stage", "-")).strip() or "-"
    state = str(payload.get("state", "-")).strip() or "-"
    message = str(payload.get("message", "")).strip()

    segments: list[str] = [f"[pipeline][{stage}][{state}]"]

    stage_progress = payload.get("stage_progress")
    if isinstance(stage_progress, (int, float)):
        segments.append(f"{float(stage_progress) * 100:.1f}%")

    processed_items = payload.get("processed_items")
    total_items = payload.get("total_items")
    if isinstance(processed_items, int) and isinstance(total_items, int) and total_items > 0:
        segments.append(f"{processed_items}/{total_items}")

    current_split = payload.get("current_split")
    split_index = payload.get("split_index")
    split_total = payload.get("split_total")
    if current_split and isinstance(split_index, int) and isinstance(split_total, int) and split_total > 0:
        segments.append(f"{current_split} {split_index}/{split_total}")

    epochs_completed = payload.get("epochs_completed")
    epochs_total = payload.get("epochs_total")
    if isinstance(epochs_completed, int) and isinstance(epochs_total, int) and epochs_total > 0:
        segments.append(f"epoch {epochs_completed}/{epochs_total}")

    current_video = payload.get("current_video")
    if current_video:
        segments.append(str(current_video))

    broken_videos = payload.get("broken_videos")
    skipped_videos = payload.get("skipped_videos")
    if isinstance(broken_videos, int) or isinstance(skipped_videos, int):
        segments.append(
            "issues "
            f"broken={int(broken_videos or 0)} "
            f"skipped={int(skipped_videos or 0)}"
        )

    if message:
        segments.append(message)
    return " ".join(segments)


def decode_stream_text(data: bytes) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_progress_line(line: str) -> str:
    cleaned = ANSI_ESCAPE_PATTERN.sub("", line or "")
    cleaned = cleaned.replace("\b", "").replace("\x00", "")
    return cleaned.strip()


def iter_process_output_lines(stream):
    buffer = b""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buffer += chunk
        while True:
            newline_positions = [pos for pos in (buffer.find(b"\n"), buffer.find(b"\r")) if pos >= 0]
            if not newline_positions:
                break
            split_at = min(newline_positions)
            raw_line = buffer[:split_at]
            buffer = buffer[split_at + 1 :]
            if buffer.startswith(b"\n"):
                buffer = buffer[1:]
            yield decode_stream_text(raw_line)
    if buffer:
        yield decode_stream_text(buffer)


def extract_download_percent(line: str) -> int | None:
    normalized = clean_progress_line(line)
    match = DOWNLOAD_PROGRESS_PATTERN.search(normalized)
    if not match:
        return None
    try:
        percent = int(match.group("percent"))
    except ValueError:
        return None
    return max(0, min(100, percent))


def size_token_to_bytes(token: str) -> float:
    normalized = token.strip().upper().replace("IB", "I").replace("B", "")
    match = re.match(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[KMGTP]?)$", normalized)
    if not match:
        return 0.0
    value = float(match.group("value"))
    unit = match.group("unit")
    multiplier = {
        "": 1.0,
        "K": 1024.0,
        "M": 1024.0 ** 2,
        "G": 1024.0 ** 3,
        "T": 1024.0 ** 4,
        "P": 1024.0 ** 5,
    }.get(unit, 1.0)
    return value * multiplier


def parse_download_snapshot(line: str) -> dict | None:
    normalized = clean_progress_line(line)
    if not normalized:
        return None
    tokens = normalized.split()
    size_tokens = [token for token in tokens if SIZE_TOKEN_PATTERN.match(token)]
    if not size_tokens:
        return None

    transferred = None
    if len(size_tokens) >= 2:
        transferred = max(size_tokens[:-1], key=size_token_to_bytes)
    else:
        transferred = size_tokens[0]
    speed = size_tokens[-1]
    return {
        "transferred": transferred,
        "speed": speed,
        "raw_line": normalized,
    }


def run_download_command_with_progress(command: list[str], *, cwd: Path, paths: dict) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )

    last_reported_percent = -1
    last_status_at = 0.0
    last_message_at = 0.0
    last_snapshot_signature = ""
    last_snapshot_bytes = 0.0
    inferred_stage_progress = 0.08
    assert process.stdout is not None
    for raw_line in iter_process_output_lines(process.stdout):
        line = clean_progress_line(raw_line)
        if line:
            print(line, flush=True)

        percent = extract_download_percent(line)
        if percent == 100 and "%" not in line:
            percent = None

        snapshot = parse_download_snapshot(line)
        if percent is None:
            if "Download successful" in line or "Request successful with HTTP status 200" in line:
                percent = 100
            else:
                now = time.monotonic()
                if snapshot:
                    snapshot_signature = f"{snapshot['transferred']}|{snapshot['speed']}"
                    snapshot_bytes = size_token_to_bytes(snapshot["transferred"])
                    bytes_delta = max(0.0, snapshot_bytes - last_snapshot_bytes)
                    should_emit_snapshot = (
                        snapshot_signature != last_snapshot_signature
                        and (
                            (now - last_message_at) >= DOWNLOAD_STATUS_MIN_INTERVAL_SECONDS
                            or bytes_delta >= DOWNLOAD_STATUS_MIN_BYTES_DELTA
                        )
                    )
                    if should_emit_snapshot:
                        inferred_stage_progress = min(0.215, inferred_stage_progress + 0.0025)
                        write_pipeline_status(
                            paths,
                            stage="download",
                            state="running",
                            message=(
                                "AIHub에서 분할 ZIP 데이터를 다운로드하는 중입니다. "
                                f"{snapshot['transferred']} 수신 · {snapshot['speed']}/s"
                            ),
                            stage_progress=inferred_stage_progress,
                            download_transferred=snapshot["transferred"],
                            download_speed=snapshot["speed"],
                        )
                        last_snapshot_signature = snapshot_signature
                        last_snapshot_bytes = snapshot_bytes
                        last_message_at = now
                elif line and (now - last_message_at) >= DOWNLOAD_STATUS_MIN_INTERVAL_SECONDS:
                    write_pipeline_status(
                        paths,
                        stage="download",
                        state="running",
                        message=f"AIHub에서 분할 ZIP 데이터를 다운로드하는 중입니다. {line[:120]}",
                        stage_progress=inferred_stage_progress,
                        download_percent=last_reported_percent if last_reported_percent >= 0 else None,
                    )
                    last_message_at = now
                continue

        now = time.monotonic()
        if percent == last_reported_percent and (now - last_status_at) < DOWNLOAD_STATUS_MIN_INTERVAL_SECONDS:
            continue
        last_reported_percent = percent
        last_status_at = now
        stage_progress = 0.08 + ((percent / 100.0) * 0.14)
        write_pipeline_status(
            paths,
            stage="download",
            state="running",
            message=f"AIHub에서 분할 ZIP 데이터를 다운로드하는 중입니다. ({percent}%)",
            stage_progress=stage_progress,
            download_percent=percent,
        )

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def download_dataset(config: dict, paths: dict) -> list[DownloadedItem]:
    source_mode = str(config.get("dataset_source", "json_api")).strip().lower()
    if source_mode == "aihub_shell":
        return download_dataset_via_aihub_shell(config, paths)

    api_config = config["api"]
    dataset_config = config["dataset"]
    raw_manifest_path = paths["current_raw_manifest"]

    session = requests.Session()
    headers = build_headers(api_config)
    session.headers.update(headers)

    label_mapping = dataset_config.get("label_mapping", {})
    max_items_per_class = int(dataset_config.get("max_items_per_class", 0))
    per_class_counts: dict[str, int] = defaultdict(int)

    items = fetch_api_items(session, api_config)
    downloaded: list[DownloadedItem] = []

    for item in items:
        source_label = str(extract_field(item, api_config["fields"]["label"])).strip()
        target_label = label_mapping.get(source_label)
        if not target_label:
            continue

        if max_items_per_class > 0 and per_class_counts[target_label] >= max_items_per_class:
            continue

        download_url = build_download_url(item, api_config)
        if not download_url:
            continue

        item_id = str(extract_field(item, api_config["fields"]["id"]))
        filename = build_filename(item, api_config, download_url, item_id)
        safe_target_label = slugify(target_label)
        target_dir = paths["raw_dir"] / safe_target_label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        download_to_file(session, download_url, target_path, timeout=float(api_config.get("timeout_seconds", 60.0)))

        downloaded_item = DownloadedItem(
            item_id=item_id,
            source_label=source_label,
            target_label=target_label,
            video_path=target_path.resolve(),
            download_url=download_url,
            metadata=item,
        )
        downloaded.append(downloaded_item)
        per_class_counts[target_label] += 1

    if not downloaded:
        raise RuntimeError("다운로드된 학습 가능 영상이 없습니다. label_mapping과 API 응답 필드를 확인해 주세요.")

    write_downloaded_items_manifest(raw_manifest_path, downloaded)

    print(f"[download] saved {len(downloaded)} items -> {raw_manifest_path}")
    return downloaded


def download_dataset_via_aihub_shell(config: dict, paths: dict) -> list[DownloadedItem]:
    shell_config = config.get("aihub_shell", {})
    dataset_config = config.get("dataset", {})
    shell_path = resolve_aihub_shell_path(shell_config)
    api_key = resolve_aihub_api_key(shell_config)
    mode = str(shell_config.get("mode", "d")).strip()
    datasetkey = shell_config.get("datasetkey")
    datapackagekey = shell_config.get("datapackagekey")
    filekey = shell_config.get("filekey")
    import_dir = paths["import_dir"]
    raw_manifest_path = paths["current_raw_manifest"]

    command = build_aihub_shell_command(shell_path, api_key, mode=mode)
    if datasetkey is not None and filekey:
        validate_aihub_filekeys(
            shell_path,
            api_key,
            datasetkey=datasetkey,
            requested_filekeys=filekey,
            excluded_source_labels=dataset_config.get("excluded_source_labels", []),
        )
    if datasetkey is not None:
        command.extend(["-datasetkey", str(datasetkey)])
    if datapackagekey is not None:
        command.extend(["-datapckagekey", str(datapackagekey)])
    if filekey:
        if isinstance(filekey, list):
            command.extend(["-filekey", "{" + ",".join(str(item) for item in filekey) + "}"])
        else:
            command.extend(["-filekey", str(filekey)])

    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="AIHub에서 분할 ZIP 데이터를 다운로드하는 중입니다.",
        stage_progress=0.08,
    )
    run_download_command_with_progress(command, cwd=import_dir, paths=paths)
    import_files = list_indexed_files(import_dir)
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="다운로드한 분할 ZIP 조각을 병합하는 중입니다.",
        stage_progress=0.22,
    )
    merged_archives = merge_split_archives(import_dir, indexed_files=import_files)
    zip_candidates = collect_zip_candidates(import_files, merged_archives)
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="병합된 ZIP 파일을 압축 해제하는 중입니다.",
        stage_progress=0.38,
    )
    source_root = extract_archives(import_dir, paths["extracted_dir"], zip_files=zip_candidates)
    source_files = import_files if source_root == import_dir else list_indexed_files(source_root)
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message="압축 해제된 영상 파일을 스캔하고 라벨을 정리하는 중입니다.",
        stage_progress=0.5,
    )
    downloaded = scan_local_video_dataset(
        config,
        paths,
        source_root=source_root,
        indexed_files=source_files,
    )
    write_pipeline_status(
        paths,
        stage="download",
        state="running",
        message=f"압축 해제 영상 스캔이 완료되었습니다. {len(downloaded)}개 샘플을 찾았습니다.",
        stage_progress=0.55,
        discovered_items=len(downloaded),
    )
    if not downloaded:
        raise RuntimeError(
            "다운로드 후 학습용 영상 파일을 찾지 못했습니다.\n"
            "확인할 것:\n"
            "1. 입력한 filekey가 실제 승인된 분할 파일인지\n"
            "2. AIHub에서 해당 데이터셋 다운로드 승인이 완료되었는지\n"
            "3. 분할 압축 파일이 .zip.part* 형태로 정상 저장되었는지\n"
            "4. label_mapping의 한글 라벨명이 압축 해제 폴더명과 일치하는지"
        )

    write_downloaded_items_manifest(raw_manifest_path, downloaded)

    print(f"[download] aihubshell imported {len(downloaded)} videos -> {raw_manifest_path}")
    return downloaded


def list_indexed_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def collect_zip_candidates(indexed_files: list[Path], merged_archives: list[Path]) -> list[Path]:
    merged_lookup = {path.resolve() for path in merged_archives}
    zip_candidates = [
        path
        for path in indexed_files
        if path.suffix.lower() == ".zip"
    ]
    for merged_path in merged_archives:
        resolved = merged_path.resolve()
        if resolved not in merged_lookup:
            continue
        if all(existing.resolve() != resolved for existing in zip_candidates):
            zip_candidates.append(merged_path)
    return sorted(zip_candidates)


def merge_split_archives(import_dir: Path, *, indexed_files: list[Path] | None = None) -> list[Path]:
    part_groups: dict[Path, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^(?P<base>.+\.zip)\.part(?P<part>.+)$", re.IGNORECASE)

    for path in indexed_files or list_indexed_files(import_dir):
        match = pattern.match(path.name)
        if not match:
            continue
        base_name = match.group("base")
        base_path = path.with_name(base_name)
        part_groups[base_path].append(path)

    merged_archives: list[Path] = []
    for base_path, parts in sorted(part_groups.items(), key=lambda item: str(item[0])):
        sorted_parts = sorted(parts, key=split_part_sort_key)
        if not sorted_parts:
            continue

        needs_merge = True
        if base_path.exists() and base_path.stat().st_size > 0:
            latest_part_mtime = max(part.stat().st_mtime for part in sorted_parts)
            if base_path.stat().st_mtime >= latest_part_mtime:
                needs_merge = False

        if needs_merge:
            with base_path.open("wb") as merged_handle:
                for part_path in sorted_parts:
                    with part_path.open("rb") as part_handle:
                        shutil.copyfileobj(part_handle, merged_handle, length=16 * 1024 * 1024)
            if base_path.stat().st_size == 0:
                raise RuntimeError(
                    f"분할 압축 병합 결과가 0바이트입니다: {base_path}\n"
                    "AIHub 안내처럼 filekey와 폴더 경로가 맞는지 다시 확인해 주세요."
                )

        merged_archives.append(base_path)

    return merged_archives


def split_part_sort_key(path: Path):
    match = re.search(r"\.part(.+)$", path.name, re.IGNORECASE)
    part_token = match.group(1) if match else path.name
    if part_token.isdigit():
        return (0, int(part_token))
    numeric = re.sub(r"[^0-9]", "", part_token)
    if numeric.isdigit():
        return (1, int(numeric), part_token)
    return (2, part_token)


def build_aihub_shell_command(shell_path: str, api_key: str, *, mode: str) -> list[str]:
    shell_candidate = Path(shell_path)
    if os.name == "nt" and is_probably_shell_script(shell_candidate):
        bash_path = resolve_windows_bash()
        if not bash_path:
            raise RuntimeError(
                "현재 aihubshell 파일이 Windows 실행 파일이 아니라 bash 스크립트입니다.\n"
                "확인할 것:\n"
                "1. Git Bash를 설치해서 bash.exe 를 사용할 수 있는지\n"
                "2. 또는 Windows용 aihubshell.exe 가 있는지\n"
                "3. 프로젝트 루트의 aihubshell 파일이 macOS/Linux용 스크립트가 아닌지"
            )
        return [bash_path, str(shell_candidate), "-mode", mode, "-aihubapikey", api_key]

    return [shell_path, "-mode", mode, "-aihubapikey", api_key]


def is_probably_shell_script(path: Path) -> bool:
    if path.suffix.lower() in {".exe", ".bat", ".cmd", ".com"}:
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(128)
    except OSError:
        return False
    return header.startswith(b"#!") or b"/bin/bash" in header or b"/bin/sh" in header


def resolve_windows_bash() -> str | None:
    candidates = [
        shutil.which("bash"),
        shutil.which("bash.exe"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
    return None


def validate_aihub_filekeys(
    shell_path: str,
    api_key: str,
    *,
    datasetkey,
    requested_filekeys,
    excluded_source_labels=None,
) -> None:
    requested = normalize_requested_filekeys(requested_filekeys)
    if not requested:
        return

    try:
        payload = fetch_aihub_file_tree(
            datasetkey=datasetkey,
            shell_path=shell_path,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"[aihubshell] filekey 목록 검증을 건너뜁니다: {exc}")
        return

    available_entries = collect_aihub_file_entries(payload)
    if not available_entries:
        print("[aihubshell] filekey 목록이 비어 있어 검증을 건너뜁니다.")
        return

    available_keys = {entry["filekey"] for entry in available_entries}
    missing = [filekey for filekey in requested if filekey not in available_keys]
    if not missing:
        excluded_requested = find_requested_aihub_entries(
            requested,
            available_entries,
            source_labels=excluded_source_labels or [],
        )
        if not excluded_requested:
            return

        excluded_keys = ", ".join(entry["filekey"] for entry in excluded_requested)
        excluded_labels = ", ".join(
            sorted(
                {
                    entry.get("source_label")
                    or entry.get("source_alias")
                    or entry.get("name")
                    or "-"
                    for entry in excluded_requested
                }
            )
        )
        excluded_examples = "\n".join(
            f"- {entry['filekey']}: {entry.get('name', '-')}"
            for entry in excluded_requested[:10]
        )
        raise RuntimeError(
            "입력한 filekey는 현재 학습 대상에서 제외한 클래스입니다.\n"
            f"- datasetkey: {datasetkey}\n"
            f"- 제외한 클래스: {excluded_labels}\n"
            f"- 제외 filekey: {excluded_keys}\n"
            "현재 설정은 주취행동(drunken)과 투기(dump)를 학습에서 제외하도록 되어 있습니다.\n"
            "아래 filekey를 확인해 주세요:\n"
            f"{excluded_examples}"
        )

    examples = ", ".join(entry["filekey"] for entry in available_entries[:10])
    matched_names = "\n".join(
        f"- {entry['filekey']}: {entry.get('name', '-')}"
        for entry in available_entries[:10]
    )
    raise RuntimeError(
        "입력한 filekey가 AIHub 파일 목록에 없습니다.\n"
        f"- datasetkey: {datasetkey}\n"
        f"- 요청 filekey: {', '.join(missing)}\n"
        f"- 예시 filekey: {examples}\n"
        "아래 목록을 먼저 확인해 주세요:\n"
        f"{matched_names}"
    )


def normalize_requested_filekeys(requested_filekeys) -> list[str]:
    if requested_filekeys is None:
        return []
    if isinstance(requested_filekeys, list):
        values = requested_filekeys
    else:
        values = [requested_filekeys]
    normalized = []
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "all":
            normalized.append(text)
    return normalized


def normalize_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", "", str(text or "").lower())


def build_source_label_matchers(source_labels) -> list[tuple[str, str, str]]:
    matchers: list[tuple[str, str, str]] = []
    for source_label in source_labels or []:
        source_text = str(source_label).strip()
        if not source_text:
            continue
        source_text_lower = source_text.lower()
        matchers.append((source_text, source_text_lower, normalize_match_text(source_text_lower)))
    return matchers


def build_label_mapping_matchers(label_mapping: dict) -> list[tuple[str, str, str, str]]:
    matchers: list[tuple[str, str, str, str]] = []
    for source_label, target_label in label_mapping.items():
        source_text = str(source_label).strip()
        if not source_text:
            continue
        source_text_lower = source_text.lower()
        matchers.append(
            (
                source_text,
                str(target_label),
                source_text_lower,
                normalize_match_text(source_text_lower),
            )
        )
    return matchers


def match_source_label_text(text: str, source_labels=None, *, matchers=None) -> str:
    raw_text = str(text or "")
    raw_text_lower = raw_text.lower()
    normalized_text = normalize_match_text(raw_text_lower)
    active_matchers = matchers if matchers is not None else build_source_label_matchers(source_labels or [])
    for source_text, source_text_lower, normalized_source in active_matchers:
        if (
            source_text in raw_text
            or source_text_lower in raw_text_lower
            or (normalized_source and normalized_source in normalized_text)
        ):
            return source_text
    return ""


def find_requested_aihub_entries(requested_filekeys: list[str], available_entries: list[dict], *, source_labels) -> list[dict]:
    requested_set = {str(filekey).strip() for filekey in requested_filekeys if str(filekey).strip()}
    matched: list[dict] = []
    for entry in available_entries:
        if entry.get("filekey") not in requested_set:
            continue
        entry_text = " ".join(
            str(entry.get(key) or "")
            for key in ("source_label", "source_alias", "name")
        )
        if match_source_label_text(entry_text, source_labels):
            matched.append(entry)
    return matched


def fetch_aihub_file_tree(*, datasetkey, shell_path: str | None = None, api_key: str | None = None) -> dict | list:
    filetree_url = f"https://api.aihub.or.kr/info/{datasetkey}.do"
    errors: list[str] = []

    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(1, AIHUB_FILE_TREE_MAX_RETRIES + 1):
            try:
                response = session.get(
                    filetree_url,
                    headers=AIHUB_FILE_TREE_HEADERS,
                    timeout=AIHUB_FILE_TREE_REQUEST_TIMEOUT_SECONDS,
                )
                merged_output = response.text.strip()
                payload_text = extract_json_payload(merged_output)
                if payload_text:
                    return json.loads(payload_text)

                listing_entries = parse_aihub_file_tree_listing(merged_output)
                if listing_entries:
                    return listing_entries

                response.raise_for_status()
                raise RuntimeError(
                    "AIHub 파일 목록 조회 결과를 해석하지 못했습니다.\n"
                    f"- datasetkey: {datasetkey}\n"
                    f"- raw output: {merged_output[:1000] if merged_output else '(empty)'}\n"
                    "AIHub 파일 목록 응답 형식이 예상과 다를 수 있습니다."
                )
            except Exception as exc:
                errors.append(f"HTTP attempt {attempt}: {exc}")
                if attempt < AIHUB_FILE_TREE_MAX_RETRIES:
                    time.sleep(AIHUB_FILE_TREE_RETRY_BACKOFF_SECONDS * attempt)

    if shell_path:
        try:
            merged_output = fetch_aihub_file_tree_via_shell(
                shell_path=shell_path,
                api_key=str(api_key or ""),
                datasetkey=datasetkey,
            )
            payload_text = extract_json_payload(merged_output)
            if payload_text:
                return json.loads(payload_text)
            listing_entries = parse_aihub_file_tree_listing(merged_output)
            if listing_entries:
                return listing_entries
            errors.append(
                "shell fallback: AIHub shell 출력에서 파일 목록 JSON을 찾지 못했습니다."
            )
        except Exception as exc:
            errors.append(f"shell fallback: {exc}")

    raise RuntimeError(
        "AIHub 파일 목록 조회에 실패했습니다.\n"
        f"- datasetkey: {datasetkey}\n"
        + "\n".join(f"- {message}" for message in errors)
    )


def fetch_aihub_file_tree_via_shell(*, shell_path: str, api_key: str, datasetkey) -> str:
    command = build_aihub_shell_command(shell_path, api_key, mode="l")
    command.extend(["-datasetkey", str(datasetkey)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=AIHUB_FILE_TREE_REQUEST_TIMEOUT_SECONDS,
        check=False,
    )
    merged_output = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    ).strip()
    if result.returncode != 0 and not merged_output:
        raise RuntimeError(
            f"AIHub shell 파일 목록 조회가 실패했습니다. exit code={result.returncode}"
        )
    return merged_output


def extract_json_payload(text: str) -> str:
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(text[start:end + 1])
    if not candidates:
        return ""
    return max(candidates, key=len)


def parse_aihub_file_tree_listing(text: str) -> list[dict]:
    entries: list[dict] = []
    current_label = ""
    current_alias = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        category_match = AIHUB_FILE_TREE_CATEGORY_PATTERN.search(line)
        if category_match and ".zip" not in line.lower():
            current_label = category_match.group("label").strip()
            current_alias = category_match.group("alias").strip()
            continue

        file_match = AIHUB_FILE_TREE_FILE_PATTERN.search(line)
        if not file_match:
            continue

        file_name = file_match.group("name").strip()
        filekey = file_match.group("filekey").strip()
        display_name = file_name
        if current_label:
            display_name = f"{current_label} / {file_name}"
        elif current_alias:
            display_name = f"{current_alias} / {file_name}"

        entries.append(
            {
                "filekey": filekey,
                "fileName": file_name,
                "name": display_name,
                "sourceLabel": current_label,
                "sourceAlias": current_alias,
            }
        )
    return entries


def collect_aihub_file_entries(payload) -> list[dict]:
    entries: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            filekey = None
            for key in ("fileSn", "filesn", "fileKey", "filekey"):
                if key in node and node[key] not in (None, ""):
                    filekey = str(node[key]).strip()
                    break
            if filekey:
                entry = {
                    "filekey": filekey,
                    "name": str(
                        node.get("fileNm")
                        or node.get("name")
                        or node.get("fileName")
                        or node.get("filePath")
                        or node.get("path")
                        or ""
                    ).strip(),
                }
                source_label = str(
                    node.get("sourceLabel")
                    or node.get("source_label")
                    or node.get("label")
                    or ""
                ).strip()
                source_alias = str(
                    node.get("sourceAlias")
                    or node.get("source_alias")
                    or ""
                ).strip()
                if source_label:
                    entry["source_label"] = source_label
                if source_alias:
                    entry["source_alias"] = source_alias
                entries.append(entry)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)

    unique: dict[str, dict] = {}
    for entry in entries:
        unique.setdefault(entry["filekey"], entry)
    return sorted(unique.values(), key=lambda item: item["filekey"])


def scan_local_video_dataset(
    config: dict,
    paths: dict,
    source_root: Path | None = None,
    *,
    indexed_files: list[Path] | None = None,
) -> list[DownloadedItem]:
    dataset_config = config["dataset"]
    import_dir = source_root or paths["import_dir"]
    raw_dir = paths["raw_dir"]
    label_mapping = dataset_config.get("label_mapping", {})
    excluded_source_labels = dataset_config.get("excluded_source_labels", [])
    label_matchers = build_label_mapping_matchers(label_mapping)
    excluded_matchers = build_source_label_matchers(excluded_source_labels)
    extensions = tuple(
        ext.lower()
        for ext in dataset_config.get("video_extensions", [".mp4", ".avi", ".mov", ".mkv", ".wmv"])
    )

    scanned: list[DownloadedItem] = []
    unlabeled_examples: list[str] = []
    excluded_examples: list[str] = []
    candidate_video_count = 0
    excluded_video_count = 0
    for video_path in sorted(indexed_files or list_indexed_files(import_dir)):
        if video_path.suffix.lower() not in extensions:
            continue
        candidate_video_count += 1

        source_label, target_label = infer_label_from_path(
            video_path,
            label_mapping,
            label_matchers=label_matchers,
        )
        if not target_label:
            excluded_label = match_source_label_text(str(video_path), matchers=excluded_matchers)
            if excluded_label:
                excluded_video_count += 1
                if len(excluded_examples) < 12:
                    excluded_examples.append(f"{video_path} [{excluded_label}]")
                continue
            if len(unlabeled_examples) < 12:
                unlabeled_examples.append(str(video_path))
            continue

        destination_dir = raw_dir / slugify(target_label)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / sanitize_filename(video_path.name)
        materialize_local_video_asset(video_path, destination_path)

        relative_id = str(video_path.relative_to(import_dir)).replace("\\", "/")
        scanned.append(
            DownloadedItem(
                item_id=slugify(relative_id),
                source_label=source_label,
                target_label=target_label,
                video_path=destination_path.resolve(),
                download_url="aihubshell://local-import",
                metadata={
                    "source_path": str(video_path.resolve()),
                    "relative_path": relative_id,
                },
            )
        )
    if excluded_video_count:
        print(f"[scan] excluded videos by config: {excluded_video_count}")
        if excluded_examples:
            print("[scan] excluded examples:")
            for sample_path in excluded_examples:
                print(f"  - {sample_path}")
    if candidate_video_count and not scanned and not unlabeled_examples and excluded_video_count:
        print("[scan] 모든 후보 영상이 excluded_source_labels에 해당해 학습 대상에서 제외되었습니다.")
    elif candidate_video_count and not scanned:
        print("[scan] 영상 파일은 찾았지만 label_mapping과 경로가 맞지 않아 학습 데이터로 분류되지 않았습니다.")
        print(f"[scan] candidate videos: {candidate_video_count}")
        if unlabeled_examples:
            print("[scan] unlabeled examples:")
            for sample_path in unlabeled_examples:
                print(f"  - {sample_path}")
    elif scanned:
        print(f"[scan] labeled videos: {len(scanned)} / candidates: {candidate_video_count}")
    return scanned


def materialize_local_video_asset(source_path: Path, destination_path: Path) -> None:
    if destination_path.exists():
        try:
            source_stat = source_path.stat()
            destination_stat = destination_path.stat()
            if (
                source_stat.st_size == destination_stat.st_size
                and int(source_stat.st_mtime) == int(destination_stat.st_mtime)
            ):
                return
        except OSError:
            pass
        try:
            destination_path.unlink()
        except OSError:
            pass

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, destination_path)
        return
    except OSError:
        pass

    shutil.copy2(source_path, destination_path)


def extract_archives(import_dir: Path, extracted_dir: Path, *, zip_files: list[Path] | None = None) -> Path:
    zip_files = sorted(zip_files or [])
    if not zip_files:
        return import_dir

    extracted_any = False
    for zip_path in zip_files:
        target_dir = extracted_dir / sanitize_filename(zip_path.stem)
        marker = target_dir / ".extracted_ok"
        if marker.exists():
            extracted_any = True
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target_dir)
        marker.write_text("ok", encoding="utf-8")
        extracted_any = True

    return extracted_dir if extracted_any else import_dir


def downloaded_item_to_manifest_entry(item: DownloadedItem) -> dict:
    return {
        "item_id": item.item_id,
        "source_label": item.source_label,
        "target_label": item.target_label,
        "video_path": str(item.video_path),
        "download_url": item.download_url,
        "metadata": item.metadata,
    }


def write_downloaded_items_manifest(path: Path, items: list[DownloadedItem]) -> None:
    write_jsonl_entries(path, [downloaded_item_to_manifest_entry(item) for item in items])


def split_dataset(downloaded: list[DownloadedItem], config: dict, paths: dict) -> dict[str, Path]:
    if not downloaded:
        raise RuntimeError("split을 만들 학습 영상이 없습니다. download 단계의 라벨 매핑과 원본 데이터를 확인해 주세요.")
    split_config = config.get("split", {})
    train_ratio = float(split_config.get("train_ratio", 0.7))
    val_ratio = float(split_config.get("val_ratio", 0.15))
    test_ratio = float(split_config.get("test_ratio", 0.15))
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise RuntimeError("split 비율 합계는 1.0 이어야 합니다.")

    rng = random.Random(int(split_config.get("seed", 42)))
    by_label: dict[str, list[DownloadedItem]] = defaultdict(list)
    for item in downloaded:
        by_label[item.target_label].append(item)

    split_items = {"train": [], "val": [], "test": []}
    for _label, items in by_label.items():
        rng.shuffle(items)
        total = len(items)
        train_count, val_count, _test_count = compute_split_counts(
            total,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_end = train_count
        val_end = train_count + val_count
        split_items["train"].extend(items[:train_end])
        split_items["val"].extend(items[train_end:val_end])
        split_items["test"].extend(items[val_end:])

    if not split_items["val"] and len(split_items["train"]) > 1:
        moved = split_items["train"].pop()
        split_items["val"].append(moved)
        print("[split] validation 샘플이 없어 train에서 1개를 val로 이동했습니다.")

    if not split_items["train"] or not split_items["val"]:
        raise RuntimeError(
            "학습/검증 split을 만들 샘플이 부족합니다. 최소 2개 이상의 학습 가능 영상을 준비해 주세요."
        )

    split_paths = {
        "train": paths["current_split_train"],
        "val": paths["current_split_val"],
        "test": paths["current_split_test"],
    }
    for split_name, target_path in split_paths.items():
        write_downloaded_items_manifest(target_path, split_items[split_name])
        print(f"[split] {split_name}: {len(split_items[split_name])} -> {target_path}")

    return split_paths


def compute_split_counts(
    total: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    total = max(int(total), 0)
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0

    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio)) if val_ratio > 0 else 0
    if train_count + val_count > total:
        overflow = train_count + val_count - total
        train_count = max(1, train_count - overflow)
    test_count = max(total - train_count - val_count, 0)
    if test_ratio > 0 and test_count == 0 and train_count > 1:
        train_count -= 1
        test_count = 1
    return train_count, val_count, test_count


def load_existing_current_split_manifests(paths: dict) -> dict[str, Path]:
    split_paths = {
        "train": paths["current_split_train"],
        "val": paths["current_split_val"],
        "test": paths["current_split_test"],
    }
    for split_name, path in split_paths.items():
        if not path.exists():
            raise RuntimeError(f"기존 split manifest를 찾지 못했습니다: {split_name} -> {path}")
    return split_paths


def prepare_pose_dataset(config: dict, paths: dict, split_manifests: dict[str, Path]) -> dict[str, Path]:
    from detector import FaceDetector, PersonDetector
    from pipeline_prepare import extract_pose_sequence_from_payload, load_video_sequence_payload

    preprocess_config = config.get("preprocess", {})
    device = str(preprocess_config.get("device", "cuda:0"))
    compress_prepared_pose = bool(preprocess_config.get("compress_prepared_pose", False))
    video_prefetch_workers = max(int(preprocess_config.get("video_prefetch_workers", 2)), 0)
    person_detector = PersonDetector(
        score_threshold=float(preprocess_config.get("person_score_threshold", 0.25)),
        resize_width=int(preprocess_config.get("person_imgsz", 640)),
        device=device,
    )
    face_detector = FaceDetector()

    target_labels = get_target_labels(config)
    label_to_idx = {label: index for index, label in enumerate(target_labels)}
    sequence_length = max(int(preprocess_config.get("sequence_length", 48)), 1)
    max_frames_to_scan = max(int(preprocess_config.get("max_frames_to_scan", 160)), sequence_length)
    detector_batch_size = max(int(preprocess_config.get("detector_batch_size", 8)), 1)
    strict_data_validation = bool(preprocess_config.get("strict_data_validation", False))
    min_frames_with_person = int(preprocess_config.get("min_frames_with_person", 4))
    fallback_min_frames_with_person = int(preprocess_config.get("fallback_min_frames_with_person", 1))
    min_confirmed_frames_with_person = int(preprocess_config.get("min_confirmed_frames_with_person", 0))
    min_total_keypoints = int(preprocess_config.get("min_total_keypoints", 1))
    max_missing_frames_ratio = max(
        0.0,
        min(float(preprocess_config.get("max_missing_frames_ratio", 0.98)), 1.0),
    )
    max_fallback_frames_ratio = max(
        0.0,
        min(float(preprocess_config.get("max_fallback_frames_ratio", 1.0)), 1.0),
    )
    allow_partial_pose = (
        bool(preprocess_config.get("allow_partial_pose", True))
        and not strict_data_validation
    )
    allow_padding = bool(preprocess_config.get("allow_padding", True)) and not strict_data_validation
    allow_rejected_pose_fallback = (
        bool(preprocess_config.get("allow_rejected_pose_fallback", True))
        and not strict_data_validation
    )
    fallback_min_keypoints = int(preprocess_config.get("fallback_min_keypoints", 3))
    fallback_min_detection_confidence = float(preprocess_config.get("fallback_min_detection_confidence", 0.15))
    fallback_min_person_score = int(preprocess_config.get("fallback_min_person_score", 20))
    skip_invalid_labels = bool(preprocess_config.get("skip_invalid_labels", True))

    prepared_paths = {
        "train": paths["current_prepared_train"],
        "val": paths["current_prepared_val"],
        "test": paths["current_prepared_test"],
    }

    manifest_rows: dict[str, list[dict]] = {}
    overall_total = 0
    skip_report = create_skip_report()
    prepare_stats = create_prepare_stats(target_labels)
    for split_name, manifest_path in split_manifests.items():
        rows = read_jsonl_entries(manifest_path)
        manifest_rows[split_name] = rows
        overall_total += len(rows)
    if overall_total <= 0:
        raise RuntimeError("prepare할 split manifest 항목이 없습니다. download/split 단계를 먼저 확인해 주세요.")

    processed_total = 0
    kept_by_split: dict[str, int] = {}

    for split_name, manifest_path in split_manifests.items():
        target_manifest_path = prepared_paths[split_name]
        rows = manifest_rows.get(split_name, [])
        payload_futures: deque[tuple[dict, Future]] = deque()
        payload_executor = (
            ThreadPoolExecutor(max_workers=video_prefetch_workers, thread_name_prefix="pose-prefetch")
            if video_prefetch_workers > 0
            else None
        )

        def submit_payload(sample_row: dict) -> Future | None:
            if payload_executor is None:
                return None
            return payload_executor.submit(
                load_video_sequence_payload,
                video_path=Path(sample_row["video_path"]),
                sequence_length=sequence_length,
                max_frames_to_scan=max_frames_to_scan,
            )

        if payload_executor is not None:
            prefetch_window = max(video_prefetch_workers * 2, 1)
            for sample_row in rows[:prefetch_window]:
                future = submit_payload(sample_row)
                if future is not None:
                    payload_futures.append((sample_row, future))
        else:
            prefetch_window = 0

        split_total = len(rows)
        with target_manifest_path.open("w", encoding="utf-8") as target_handle:
            kept = 0
            skipped = 0
            try:
                for split_index in range(1, split_total + 1):
                    if payload_executor is not None:
                        sample, payload_future = payload_futures.popleft()
                        next_prefetch_index = split_index - 1 + prefetch_window
                        if next_prefetch_index < split_total:
                            next_sample = rows[next_prefetch_index]
                            next_future = submit_payload(next_sample)
                            if next_future is not None:
                                payload_futures.append((next_sample, next_future))
                    else:
                        sample = rows[split_index - 1]
                        payload_future = None

                    video_path = Path(sample["video_path"])
                    target_label = str(sample.get("target_label") or "").strip()
                    register_prepare_input(prepare_stats, split_name=split_name, label=target_label)
                    processed_total += 1
                    if (
                        processed_total == 1
                        or processed_total == overall_total
                        or processed_total % 5 == 0
                    ):
                        prepare_ratio = processed_total / max(overall_total, 1)
                        write_pipeline_status(
                            paths,
                            stage="prepare",
                            state="running",
                            message=f"{split_name} split에서 pose 시퀀스를 추출하는 중입니다.",
                            stage_progress=round(0.55 + (0.25 * prepare_ratio), 4),
                            processed_items=processed_total,
                            total_items=overall_total,
                            current_split=split_name,
                            split_index=split_index,
                            split_total=split_total,
                            current_video=video_path.name,
                            kept_items=kept,
                            skipped_items=skipped,
                            prefetch_workers=video_prefetch_workers,
                        )
                    if target_label not in label_to_idx:
                        reason = "class_mapping_failed"
                        detail = f"target_label={target_label or '-'}"
                        if not skip_invalid_labels:
                            raise RuntimeError(f"클래스 매핑 실패: {detail}")
                        skipped += 1
                        register_prepare_skip(
                            prepare_stats,
                            split_name=split_name,
                            label=target_label,
                            reason=reason,
                        )
                        append_skip_issue(
                            skip_report,
                            category="skipped",
                            split_name=split_name,
                            video_path=video_path,
                            target_label=target_label,
                            reason=reason,
                            detail=detail,
                        )
                        continue
                    if not video_path.exists():
                        reason = "file_missing"
                        skipped += 1
                        register_prepare_skip(
                            prepare_stats,
                            split_name=split_name,
                            label=target_label,
                            reason=reason,
                        )
                        append_skip_issue(
                            skip_report,
                            category="broken",
                            split_name=split_name,
                            video_path=video_path,
                            target_label=target_label,
                            reason=reason,
                        )
                        continue
                    try:
                        if payload_future is not None:
                            payload = payload_future.result()
                        else:
                            payload = load_video_sequence_payload(
                                video_path=video_path,
                                sequence_length=sequence_length,
                                max_frames_to_scan=max_frames_to_scan,
                            )

                        sequence = extract_pose_sequence_from_payload(
                            payload=payload,
                            person_detector=person_detector,
                            face_detector=face_detector,
                            detector_batch_size=detector_batch_size,
                            allow_rejected_pose_fallback=allow_rejected_pose_fallback,
                            fallback_min_keypoints=fallback_min_keypoints,
                            fallback_min_detection_confidence=fallback_min_detection_confidence,
                            fallback_min_person_score=fallback_min_person_score,
                        )
                    except Exception as exc:
                        reason = "unreadable_video"
                        skipped += 1
                        register_prepare_skip(
                            prepare_stats,
                            split_name=split_name,
                            label=target_label,
                            reason=reason,
                        )
                        append_skip_issue(
                            skip_report,
                            category="broken",
                            split_name=split_name,
                            video_path=video_path,
                            target_label=target_label,
                            reason=reason,
                            detail=str(exc),
                        )
                        print(f"[prepare] skip unreadable video: {video_path} ({exc})")
                        continue
                    keep_sample, skip_reason, recovery_actions = decide_prepare_sample_usage(
                        sequence,
                        sequence_length=sequence_length,
                        min_frames_with_person=min_frames_with_person,
                        fallback_min_frames_with_person=fallback_min_frames_with_person,
                        min_confirmed_frames_with_person=min_confirmed_frames_with_person,
                        min_total_keypoints=min_total_keypoints,
                        max_missing_frames_ratio=max_missing_frames_ratio,
                        max_fallback_frames_ratio=max_fallback_frames_ratio,
                        allow_partial_pose=allow_partial_pose,
                        allow_padding=allow_padding,
                    )
                    if not keep_sample:
                        skipped += 1
                        register_prepare_skip(
                            prepare_stats,
                            split_name=split_name,
                            label=target_label,
                            reason=skip_reason,
                        )
                        append_skip_issue(
                            skip_report,
                            category="skipped",
                            split_name=split_name,
                            video_path=video_path,
                            target_label=target_label,
                            reason=skip_reason,
                            valid_frames=int(sequence["valid_frames"]),
                            confirmed_frames=int(sequence["confirmed_frames"]),
                            total_valid_keypoints=int(sequence.get("total_valid_keypoints", 0) or 0),
                        )
                        continue
                    register_prepare_used(
                        prepare_stats,
                        split_name=split_name,
                        label=target_label,
                        recovery_actions=recovery_actions,
                    )

                    pose_output_dir = paths["prepared_dir"] / split_name / slugify(target_label)
                    pose_output_dir.mkdir(parents=True, exist_ok=True)
                    pose_path = pose_output_dir / f"{video_path.stem}_{sample['item_id']}.npz"
                    save_npz = np.savez_compressed if compress_prepared_pose else np.savez
                    save_npz(
                        pose_path,
                        pose=sequence["pose"],
                        mask=sequence["mask"],
                        label_idx=np.int64(label_to_idx[target_label]),
                    )

                    target_handle.write(
                        json.dumps(
                            {
                                **sample,
                                "pose_path": str(pose_path.resolve()),
                                "label_idx": label_to_idx[target_label],
                                "valid_frames": sequence["valid_frames"],
                                "confirmed_frames": sequence["confirmed_frames"],
                                "fallback_frames": sequence.get("fallback_frames", 0),
                                "total_valid_keypoints": sequence.get("total_valid_keypoints", 0),
                                "avg_pose_confidence": round(
                                    float(sequence.get("avg_pose_confidence", 0.0) or 0.0),
                                    6,
                                ),
                                "chosen_track_id": sequence["chosen_track_id"],
                                "recovery_actions": recovery_actions,
                                "frame_stats": sequence.get("frame_stats", {}),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    kept += 1
            finally:
                if payload_executor is not None:
                    payload_executor.shutdown(wait=True, cancel_futures=True)
            write_pipeline_status(
                paths,
                stage="prepare",
                state="running",
                message=f"{split_name} split pose 추출이 완료되었습니다.",
                stage_progress=round(0.55 + (0.25 * (processed_total / max(overall_total, 1))), 4) if overall_total else 0.8,
                processed_items=processed_total,
                total_items=overall_total,
                current_split=split_name,
                split_index=split_total,
                split_total=split_total,
                kept_items=kept,
                skipped_items=skipped,
                broken_videos=skip_report.get("summary", {}).get("broken_count", 0),
                skipped_videos=skip_report.get("summary", {}).get("skipped_count", 0),
            )
        print(f"[prepare] {split_name}: {kept} samples ({skipped} skipped) -> {target_manifest_path}")
        kept_by_split[split_name] = kept

    prepare_summary = finalize_prepare_stats(prepare_stats)
    skip_report["prepare_summary"] = prepare_summary
    skip_report["summary"].update(
        {
            "total_items": prepare_summary["overall"]["total"],
            "used_items": prepare_summary["overall"]["used"],
            "skipped_items": prepare_summary["overall"]["skipped"],
            "recovered_items": prepare_summary["overall"]["recovered"],
            "skip_ratio": prepare_summary["overall"]["skip_ratio"],
            "by_reason": prepare_summary["by_reason"],
            "by_split": {
                split_name: split_payload["skipped"]
                for split_name, split_payload in prepare_summary["by_split"].items()
            },
            "by_label": {
                label: label_payload["skipped"]
                for label, label_payload in prepare_summary["by_label"].items()
            },
        }
    )
    print_prepare_summary(prepare_summary)
    write_skip_reports(paths, skip_report)
    summary = skip_report.get("summary", {})
    print(
        "[prepare] issue summary: "
        f"broken={summary.get('broken_count', 0)}, "
        f"skipped={summary.get('skipped_count', 0)}"
    )

    if kept_by_split.get("train", 0) <= 0 or kept_by_split.get("val", 0) <= 0:
        raise RuntimeError(
            "전처리 후 학습/검증 샘플이 부족합니다. min_frames_with_person, 라벨 매핑, 원본 영상을 확인해 주세요."
        )

    return prepared_paths


def create_prepare_stats(labels: list[str]) -> dict:
    return {
        "labels": list(labels),
        "overall": Counter(),
        "by_reason": Counter(),
        "by_split": defaultdict(Counter),
        "by_split_reason": defaultdict(Counter),
        "by_label": defaultdict(Counter),
        "by_label_reason": defaultdict(Counter),
        "recovery_actions": Counter(),
    }


def register_prepare_input(stats: dict, *, split_name: str, label: str) -> None:
    normalized_label = normalize_stats_label(label)
    stats["overall"]["total"] += 1
    stats["by_split"][split_name]["total"] += 1
    stats["by_label"][normalized_label]["total"] += 1


def register_prepare_used(
    stats: dict,
    *,
    split_name: str,
    label: str,
    recovery_actions: list[str],
) -> None:
    normalized_label = normalize_stats_label(label)
    stats["overall"]["used"] += 1
    stats["by_split"][split_name]["used"] += 1
    stats["by_label"][normalized_label]["used"] += 1
    if recovery_actions:
        stats["overall"]["recovered"] += 1
        stats["by_split"][split_name]["recovered"] += 1
        stats["by_label"][normalized_label]["recovered"] += 1
        for action in recovery_actions:
            stats["recovery_actions"][str(action)] += 1


def register_prepare_skip(stats: dict, *, split_name: str, label: str, reason: str) -> None:
    normalized_label = normalize_stats_label(label)
    normalized_reason = str(reason or "unknown")
    stats["overall"]["skipped"] += 1
    stats["by_reason"][normalized_reason] += 1
    stats["by_split"][split_name]["skipped"] += 1
    stats["by_split_reason"][split_name][normalized_reason] += 1
    stats["by_label"][normalized_label]["skipped"] += 1
    stats["by_label_reason"][normalized_label][normalized_reason] += 1


def normalize_stats_label(label: str) -> str:
    return str(label or "unknown").strip() or "unknown"


def decide_prepare_sample_usage(
    sequence: dict,
    *,
    sequence_length: int,
    min_frames_with_person: int,
    fallback_min_frames_with_person: int,
    min_total_keypoints: int,
    max_missing_frames_ratio: float,
    allow_partial_pose: bool,
    allow_padding: bool,
    min_confirmed_frames_with_person: int = 0,
    max_fallback_frames_ratio: float = 1.0,
) -> tuple[bool, str, list[str]]:
    valid_frames = int(sequence.get("valid_frames", 0) or 0)
    confirmed_frames = int(sequence.get("confirmed_frames", 0) or 0)
    fallback_frames = int(sequence.get("fallback_frames", 0) or 0)
    total_keypoints = int(sequence.get("total_valid_keypoints", 0) or 0)
    skip_reason = str(sequence.get("skip_reason") or "")
    if valid_frames <= 0:
        return False, skip_reason or "person_not_detected", []

    if total_keypoints < max(int(min_total_keypoints), 0):
        return False, "pose_keypoint_insufficient", []

    missing_ratio = 1.0 - (valid_frames / max(int(sequence_length), 1))
    if missing_ratio > float(max_missing_frames_ratio):
        return False, "too_many_missing_frames", []
    if confirmed_frames < max(int(min_confirmed_frames_with_person), 0):
        return False, "confirmed_frames_insufficient", []
    fallback_ratio = fallback_frames / max(valid_frames, 1)
    if fallback_ratio > float(max_fallback_frames_ratio):
        return False, "too_many_fallback_frames", []

    recovery_actions = list(sequence.get("recovery_actions") or [])
    if valid_frames >= max(int(min_frames_with_person), 1):
        if missing_ratio > 0:
            recovery_actions.append("mask_padding")
        return True, "", sorted(set(recovery_actions))

    can_recover_partial = (
        allow_partial_pose
        and allow_padding
        and valid_frames >= max(int(fallback_min_frames_with_person), 1)
    )
    if can_recover_partial:
        recovery_actions.extend(["partial_pose_padding", "below_min_frames_recovered"])
        return True, "", sorted(set(recovery_actions))

    return False, "min_frames_with_person", []


def finalize_prepare_stats(stats: dict) -> dict:
    labels = list(stats.get("labels") or [])
    overall = counter_to_plain_dict(stats["overall"])
    normalize_prepare_counter(overall)

    by_split = {
        split_name: build_prepare_counter_payload(counter)
        for split_name, counter in sorted(stats["by_split"].items())
    }
    by_label = {
        label: build_prepare_counter_payload(stats["by_label"].get(label, Counter()))
        for label in labels
    }
    for label, counter in sorted(stats["by_label"].items()):
        if label not in by_label:
            by_label[label] = build_prepare_counter_payload(counter)

    by_split_reason = {
        split_name: dict(sorted(counter.items()))
        for split_name, counter in sorted(stats["by_split_reason"].items())
    }
    by_label_reason = {
        label: dict(sorted(counter.items()))
        for label, counter in sorted(stats["by_label_reason"].items())
    }
    most_lost_class = max(
        by_label.items(),
        key=lambda item: (item[1]["skip_ratio"], item[1]["skipped"]),
        default=(None, None),
    )
    used_counts = {label: payload["used"] for label, payload in by_label.items()}
    original_counts = {label: payload["total"] for label, payload in by_label.items()}
    return {
        "overall": overall,
        "by_reason": dict(sorted(stats["by_reason"].items())),
        "by_split": by_split,
        "by_split_reason": by_split_reason,
        "by_label": by_label,
        "by_label_reason": by_label_reason,
        "recovery_actions": dict(sorted(stats["recovery_actions"].items())),
        "original_class_counts": original_counts,
        "used_class_counts": used_counts,
        "most_lost_class": {
            "label": most_lost_class[0],
            "skip_ratio": most_lost_class[1]["skip_ratio"] if most_lost_class[1] else 0.0,
            "skipped": most_lost_class[1]["skipped"] if most_lost_class[1] else 0,
        },
        "imbalance": build_prepare_imbalance_summary(original_counts, used_counts),
    }


def build_prepare_counter_payload(counter: Counter) -> dict:
    payload = counter_to_plain_dict(counter)
    normalize_prepare_counter(payload)
    return payload


def counter_to_plain_dict(counter: Counter | dict) -> dict:
    return {
        "total": int(counter.get("total", 0) or 0),
        "used": int(counter.get("used", 0) or 0),
        "skipped": int(counter.get("skipped", 0) or 0),
        "recovered": int(counter.get("recovered", 0) or 0),
    }


def normalize_prepare_counter(payload: dict) -> None:
    total = int(payload.get("total", 0) or 0)
    skipped = int(payload.get("skipped", 0) or 0)
    used = int(payload.get("used", 0) or 0)
    payload["skip_ratio"] = round(skipped / total, 6) if total else 0.0
    payload["use_ratio"] = round(used / total, 6) if total else 0.0


def build_prepare_imbalance_summary(original_counts: dict[str, int], used_counts: dict[str, int]) -> dict:
    original_ratio = compute_count_ratio(original_counts)
    used_ratio = compute_count_ratio(used_counts)
    return {
        "original_ratio": original_ratio,
        "used_ratio": used_ratio,
        "worsened_after_skip": (
            used_ratio is not None
            and original_ratio is not None
            and used_ratio > original_ratio
        ),
    }


def compute_count_ratio(counts: dict[str, int]) -> float | None:
    nonzero_counts = [int(value) for value in counts.values() if int(value or 0) > 0]
    if len(nonzero_counts) < 2:
        return None
    return round(max(nonzero_counts) / max(min(nonzero_counts), 1), 6)


def print_prepare_summary(summary: dict) -> None:
    overall = summary.get("overall", {})
    print(
        "[prepare] data usage "
        f"total={overall.get('total', 0)} "
        f"used={overall.get('used', 0)} "
        f"skipped={overall.get('skipped', 0)} "
        f"recovered={overall.get('recovered', 0)} "
        f"skip_ratio={float(overall.get('skip_ratio', 0.0)):.2%}"
    )
    by_reason = summary.get("by_reason") or {}
    if by_reason:
        print(
            "[prepare] skip reasons "
            + ", ".join(f"{reason}={count}" for reason, count in by_reason.items())
        )
    if summary.get("recovery_actions"):
        print(
            "[prepare] recovery actions "
            + ", ".join(
                f"{action}={count}"
                for action, count in summary["recovery_actions"].items()
            )
        )
    for split_name, payload in summary.get("by_split", {}).items():
        print(
            "[prepare] split usage "
            f"{split_name}: total={payload.get('total', 0)} "
            f"used={payload.get('used', 0)} "
            f"skipped={payload.get('skipped', 0)} "
            f"recovered={payload.get('recovered', 0)} "
            f"skip_ratio={float(payload.get('skip_ratio', 0.0)):.2%}"
        )
    for label, payload in summary.get("by_label", {}).items():
        print(
            "[prepare] class usage "
            f"{label}: total={payload.get('total', 0)} "
            f"used={payload.get('used', 0)} "
            f"skipped={payload.get('skipped', 0)} "
            f"recovered={payload.get('recovered', 0)} "
            f"skip_ratio={float(payload.get('skip_ratio', 0.0)):.2%}"
        )
    most_lost = summary.get("most_lost_class") or {}
    if most_lost.get("label"):
        print(
            "[prepare] most lost class "
            f"{most_lost['label']} skipped={most_lost.get('skipped', 0)} "
            f"skip_ratio={float(most_lost.get('skip_ratio', 0.0)):.2%}"
        )
    imbalance = summary.get("imbalance") or {}
    print(
        "[prepare] imbalance "
        f"original_ratio={imbalance.get('original_ratio')} "
        f"used_ratio={imbalance.get('used_ratio')} "
        f"worsened_after_skip={imbalance.get('worsened_after_skip')}"
    )


def load_existing_current_prepared_manifests(paths: dict) -> dict[str, Path]:
    prepared_paths = {
        "train": paths["current_prepared_train"],
        "val": paths["current_prepared_val"],
        "test": paths["current_prepared_test"],
    }
    for split_name, path in prepared_paths.items():
        if split_name in {"train", "val"} and not path.exists():
            raise RuntimeError(f"기존 prepared manifest를 찾지 못했습니다: {split_name} -> {path}")
    return prepared_paths


def get_continual_config(config: dict) -> dict:
    continual = config.get("continual_learning", {})
    return {
        "enabled": bool(continual.get("enabled", True)),
        "resume_from_best": bool(continual.get("resume_from_best", True)),
        "cleanup_raw_after_job": bool(continual.get("cleanup_raw_after_job", True)),
    }


def load_training_manifests(config: dict, paths: dict, *, continual_enabled: bool) -> dict[str, Path]:
    if continual_enabled and paths["prepared_train"].exists() and paths["prepared_val"].exists():
        base_manifests = {
            "train": paths["prepared_train"],
            "val": paths["prepared_val"],
            "test": paths["prepared_test"],
        }
    else:
        base_manifests = {
            "train": paths["current_prepared_train"],
            "val": paths["current_prepared_val"],
            "test": paths["current_prepared_test"],
        }
    return materialize_training_manifests(config, paths, base_manifests)


def count_manifest_labels(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in iter_jsonl_entries(path):
        label = str(entry.get("target_label") or "").strip()
        if label:
            counts[label] += 1
    return counts


def training_class_coverage_report(
    config: dict,
    training_manifests: dict[str, Path],
    *,
    labels: list[str] | None = None,
) -> dict:
    target_labels = [str(label) for label in (labels or get_target_labels(config))]
    train_counts = count_manifest_labels(training_manifests["train"])
    val_counts = count_manifest_labels(training_manifests["val"])
    train_labels = [label for label in target_labels if train_counts.get(label, 0) > 0]
    val_labels = [label for label in target_labels if val_counts.get(label, 0) > 0]
    min_train_classes, min_val_classes = required_training_class_counts(config, target_labels)
    training_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    allow_single_class_training = bool(training_config.get("allow_single_class_training", False))
    ok = (
        len(target_labels) <= 1
        or allow_single_class_training
        or (len(train_labels) >= min_train_classes and len(val_labels) >= min_val_classes)
    )
    return {
        "ok": ok,
        "target_labels": target_labels,
        "target_class_count": len(target_labels),
        "allow_single_class_training": allow_single_class_training,
        "min_train_classes": min_train_classes,
        "min_val_classes": min_val_classes,
        "train_class_count": len(train_labels),
        "val_class_count": len(val_labels),
        "train_labels": train_labels,
        "val_labels": val_labels,
        "missing_train_labels": [label for label in target_labels if label not in train_labels],
        "missing_val_labels": [label for label in target_labels if label not in val_labels],
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
    }


def build_training_class_coverage_message(
    report: dict,
    *,
    prefix: str = "학습 manifest의 클래스 수가 부족해 모델 학습을 중단했습니다.",
    suffix: str = "",
) -> str:
    target_count = int(report.get("target_class_count", 0) or 0)
    train_labels = [str(label) for label in report.get("train_labels") or []]
    val_labels = [str(label) for label in report.get("val_labels") or []]
    min_train = int(report.get("min_train_classes", 0) or 0)
    min_val = int(report.get("min_val_classes", 0) or 0)
    segments = [
        prefix,
        (
            f"train 클래스={len(train_labels)}/{target_count} {train_labels} "
            f"(필요 최소 {min_train}), "
            f"val 클래스={len(val_labels)}/{target_count} {val_labels} "
            f"(필요 최소 {min_val})."
        ),
        "단일 클래스만으로 fine-tuning하면 기존 모델이 특정 클래스로 무너질 수 있습니다.",
    ]
    if suffix:
        segments.append(suffix)
    return " ".join(segment for segment in segments if segment)


def should_defer_training_for_class_coverage(config: dict, report: dict, *, stage: str) -> bool:
    if bool(report.get("ok")):
        return False
    if str(stage or "").strip().lower() != "all":
        return False
    training_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    return bool(training_config.get("defer_until_min_classes", True))


def validate_training_class_coverage(
    config: dict,
    training_manifests: dict[str, Path],
    *,
    labels: list[str] | None = None,
) -> None:
    report = training_class_coverage_report(config, training_manifests, labels=labels)
    if bool(report.get("ok")):
        return

    raise RuntimeError(
        build_training_class_coverage_message(
            report,
            suffix=(
                "outside 자동 추천이 비어 있는 다른 클래스 filekey를 먼저 누적하도록 한 뒤 다시 학습하세요. "
                "정말 단일 클래스 실험이 필요하면 training.allow_single_class_training=true로 명시하세요."
            ),
        )
    )


def required_training_class_counts(config: dict, labels: list[str]) -> tuple[int, int]:
    if len(labels) <= 1:
        return 1, 1
    training_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    min_train_classes = max(int(training_config.get("min_active_train_classes", 2)), 1)
    min_val_classes = max(int(training_config.get("min_active_val_classes", min_train_classes)), 1)
    return min_train_classes, min_val_classes


def distribution_covered_count(payload) -> int | None:
    if not isinstance(payload, dict):
        return None
    covered = payload.get("covered")
    try:
        if covered is not None:
            return int(covered)
    except (TypeError, ValueError):
        return None
    counts = payload.get("counts")
    if isinstance(counts, list):
        return sum(
            1
            for row in counts
            if isinstance(row, dict) and int(row.get("count", 0) or 0) > 0
        )
    by_label = payload.get("by_label")
    if isinstance(by_label, dict):
        return sum(1 for count in by_label.values() if int(count or 0) > 0)
    return None


def checkpoint_is_safe_for_resume(config: dict, paths: dict, *, labels: list[str]) -> bool:
    training_config = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    if bool(training_config.get("allow_single_class_training", False)):
        return True

    block_missing_predictions = bool(training_config.get("block_resume_on_missing_predictions", True))
    target_label_set = {str(label) for label in labels}
    min_train_classes, min_val_classes = required_training_class_counts(config, labels)
    artifacts_dir = paths.get("artifacts_dir")
    metrics_path = (artifacts_dir / "metrics.json") if isinstance(artifacts_dir, Path) else None
    for path in (paths.get("training_progress"), metrics_path):
        payload = read_json_file(path) if isinstance(path, Path) else None
        if not isinstance(payload, dict):
            continue
        if block_missing_predictions:
            final_validation = payload.get("final_validation") if isinstance(payload.get("final_validation"), dict) else {}
            missing_prediction_classes = []
            analysis = payload.get("validation_error_analysis")
            if isinstance(analysis, dict):
                missing_prediction_classes = [
                    str(label)
                    for label in (analysis.get("missing_prediction_classes") or [])
                    if str(label) in target_label_set
                ]
            if not missing_prediction_classes:
                missing_prediction_classes = infer_missing_prediction_classes_from_confusion(
                    final_validation.get("confusion_matrix"),
                    labels=labels,
                )
            if missing_prediction_classes:
                print(
                    "[train] previous checkpoint is skipped because it never predicted "
                    f"these validation classes: {missing_prediction_classes}"
                )
                return False
        train_covered = distribution_covered_count(payload.get("train_distribution"))
        val_covered = distribution_covered_count(payload.get("val_distribution"))
        if train_covered is None and val_covered is None:
            continue
        return (
            (train_covered is None or train_covered >= min_train_classes)
            and (val_covered is None or val_covered >= min_val_classes)
        )
    return True


def infer_missing_prediction_classes_from_confusion(confusion_matrix, *, labels: list[str]) -> list[str]:
    if not isinstance(confusion_matrix, list):
        return []
    missing: list[str] = []
    for class_index, label in enumerate(labels):
        support = 0
        predicted = 0
        for row_index, row in enumerate(confusion_matrix):
            if not isinstance(row, list):
                continue
            if row_index == class_index:
                support = sum(int(value or 0) for value in row)
            if class_index < len(row):
                predicted += int(row[class_index] or 0)
        if support > 0 and predicted <= 0:
            missing.append(str(label))
    return missing


def update_cumulative_manifests(
    config: dict,
    paths: dict,
    split_manifests: dict[str, Path],
    prepared_manifests: dict[str, Path],
) -> dict[str, Path]:
    continual_config = get_continual_config(config)
    if not continual_config["enabled"]:
        return materialize_training_manifests(config, paths, prepared_manifests)

    shell_config = config.get("aihub_shell", {})
    job_meta = {
        "job_datasetkey": str(shell_config.get("datasetkey", "")).strip() or None,
        "job_filekey": normalize_requested_filekeys(shell_config.get("filekey")),
        "job_added_at": datetime.now(UTC).astimezone().isoformat(),
    }

    merge_jsonl_entries(
        paths["current_raw_manifest"],
        paths["raw_manifest"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        split_manifests["train"],
        paths["split_train"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        split_manifests["val"],
        paths["split_val"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        split_manifests["test"],
        paths["split_test"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        prepared_manifests["train"],
        paths["prepared_train"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        prepared_manifests["val"],
        paths["prepared_val"],
        extra_fields=job_meta,
    )
    merge_jsonl_entries(
        prepared_manifests["test"],
        paths["prepared_test"],
        extra_fields=job_meta,
    )

    write_json_atomic(
        paths["continual_state"],
        {
            "updated_at": job_meta["job_added_at"],
            "datasetkey": job_meta["job_datasetkey"],
            "filekeys": job_meta["job_filekey"],
            "raw_total": count_manifest_lines(paths["raw_manifest"]),
            "prepared_train_total": count_manifest_lines(paths["prepared_train"]),
            "prepared_val_total": count_manifest_lines(paths["prepared_val"]),
            "prepared_test_total": count_manifest_lines(paths["prepared_test"]),
        },
    )

    print(
        "[continual] cumulative prepared samples "
        f"train={count_manifest_lines(paths['prepared_train'])}, "
        f"val={count_manifest_lines(paths['prepared_val'])}, "
        f"test={count_manifest_lines(paths['prepared_test'])}"
    )
    return materialize_training_manifests(
        config,
        paths,
        {
            "train": paths["prepared_train"],
            "val": paths["prepared_val"],
            "test": paths["prepared_test"],
        },
    )


def materialize_training_manifests(
    config: dict,
    paths: dict,
    prepared_manifests: dict[str, Path],
) -> dict[str, Path]:
    target_labels = get_target_labels(config)
    label_to_idx = {label: index for index, label in enumerate(target_labels)}
    label_mapping = config.get("dataset", {}).get("label_mapping", {}) or {}
    quality_rules = build_prepared_quality_rules(config)
    active_paths = {
        "train": paths["active_prepared_train"],
        "val": paths["active_prepared_val"],
        "test": paths["active_prepared_test"],
    }
    desired_state = build_active_manifest_state(
        target_labels=target_labels,
        label_mapping=label_mapping,
        quality_rules=quality_rules,
        source_manifests=prepared_manifests,
    )
    if active_manifests_are_current(
        state_path=paths["active_manifest_state"],
        active_paths=active_paths,
        desired_state=desired_state,
    ):
        print("[train-manifest] 기존 active manifest를 재사용합니다.")
        return active_paths

    seen_keys: set[str] = set()
    split_order = [split_name for split_name in ("train", "val", "test") if split_name in prepared_manifests]
    split_order.extend(split_name for split_name in prepared_manifests if split_name not in split_order)
    for split_name in split_order:
        source_path = prepared_manifests[split_name]
        target_path = active_paths[split_name]
        kept = 0
        skipped = 0
        quality_skipped = 0
        duplicate_skipped = 0
        quality_skip_reasons: Counter[str] = Counter()
        with target_path.open("w", encoding="utf-8") as handle:
            for entry in read_jsonl_entries(source_path):
                remapped = remap_prepared_entry(entry, label_to_idx=label_to_idx, label_mapping=label_mapping)
                if remapped is None:
                    skipped += 1
                    continue
                quality_keep, quality_reason = check_prepared_entry_quality(remapped, quality_rules)
                if not quality_keep:
                    quality_skipped += 1
                    quality_skip_reasons[quality_reason] += 1
                    continue
                unique_key = build_manifest_unique_key(remapped)
                if unique_key in seen_keys:
                    duplicate_skipped += 1
                    continue
                handle.write(json.dumps(remapped, ensure_ascii=False) + "\n")
                seen_keys.add(unique_key)
                kept += 1
        quality_summary = ""
        if quality_skipped:
            quality_summary = ", quality reasons: " + ", ".join(
                f"{reason}={count}" for reason, count in sorted(quality_skip_reasons.items())
            )
        print(
            f"[train-manifest] {split_name}: {kept} kept "
            f"({skipped} label filtered, {quality_skipped} quality filtered, "
            f"{duplicate_skipped} duplicate/leakage skipped{quality_summary}) -> {target_path}"
        )

    write_active_manifest_state(paths["active_manifest_state"], desired_state)
    return active_paths


def remap_prepared_entry(entry: dict, *, label_to_idx: dict[str, int], label_mapping: dict) -> dict | None:
    mapped_entry = dict(entry)
    target_label = str(mapped_entry.get("target_label") or "").strip()
    source_label = str(mapped_entry.get("source_label") or "").strip()

    if target_label not in label_to_idx and source_label:
        remapped_target = label_mapping.get(source_label)
        if remapped_target:
            target_label = str(remapped_target)

    if target_label not in label_to_idx:
        return None

    mapped_entry["target_label"] = target_label
    mapped_entry["label_idx"] = int(label_to_idx[target_label])
    return mapped_entry


def build_prepared_quality_rules(config: dict) -> dict:
    preprocess_config = config.get("preprocess", {}) or {}
    if not bool(preprocess_config.get("filter_existing_prepared_by_quality", True)):
        return {"enabled": False}
    strict_data_validation = bool(preprocess_config.get("strict_data_validation", False))
    return {
        "enabled": True,
        "sequence_length": max(int(preprocess_config.get("sequence_length", 48)), 1),
        "min_frames_with_person": int(preprocess_config.get("min_frames_with_person", 4)),
        "fallback_min_frames_with_person": int(preprocess_config.get("fallback_min_frames_with_person", 1)),
        "min_confirmed_frames_with_person": int(preprocess_config.get("min_confirmed_frames_with_person", 0)),
        "min_total_keypoints": int(preprocess_config.get("min_total_keypoints", 1)),
        "max_missing_frames_ratio": max(
            0.0,
            min(float(preprocess_config.get("max_missing_frames_ratio", 0.98)), 1.0),
        ),
        "max_fallback_frames_ratio": max(
            0.0,
            min(float(preprocess_config.get("max_fallback_frames_ratio", 1.0)), 1.0),
        ),
        "allow_partial_pose": (
            bool(preprocess_config.get("allow_partial_pose", True))
            and not strict_data_validation
        ),
        "allow_padding": bool(preprocess_config.get("allow_padding", True)) and not strict_data_validation,
    }


def check_prepared_entry_quality(entry: dict, rules: dict) -> tuple[bool, str]:
    if not rules.get("enabled"):
        return True, ""
    valid_frames = optional_int(entry.get("valid_frames"))
    if valid_frames is None:
        return True, ""
    sequence = {
        "valid_frames": valid_frames,
        "confirmed_frames": optional_int(entry.get("confirmed_frames")) or 0,
        "fallback_frames": optional_int(entry.get("fallback_frames")) or 0,
        "total_valid_keypoints": optional_int(entry.get("total_valid_keypoints")) or 0,
    }
    keep, reason, _recovery_actions = decide_prepare_sample_usage(
        sequence,
        sequence_length=int(rules["sequence_length"]),
        min_frames_with_person=int(rules["min_frames_with_person"]),
        fallback_min_frames_with_person=int(rules["fallback_min_frames_with_person"]),
        min_total_keypoints=int(rules["min_total_keypoints"]),
        max_missing_frames_ratio=float(rules["max_missing_frames_ratio"]),
        allow_partial_pose=bool(rules["allow_partial_pose"]),
        allow_padding=bool(rules["allow_padding"]),
        min_confirmed_frames_with_person=int(rules["min_confirmed_frames_with_person"]),
        max_fallback_frames_ratio=float(rules["max_fallback_frames_ratio"]),
    )
    return keep, reason


def optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def merge_jsonl_entries(source_path: Path, target_path: Path, *, extra_fields: dict | None = None) -> int:
    source_entries = read_jsonl_entries(source_path)
    if not source_entries:
        return 0

    seen = {build_manifest_unique_key(entry) for entry in iter_jsonl_entries(target_path)}
    new_entries: list[dict] = []

    for entry in source_entries:
        merged_entry = dict(entry)
        if extra_fields:
            merged_entry.update(extra_fields)
        unique_key = build_manifest_unique_key(merged_entry)
        if unique_key in seen:
            continue
        new_entries.append(merged_entry)
        seen.add(unique_key)
    append_jsonl_entries(target_path, new_entries)
    return len(new_entries)

def iter_jsonl_entries(path: Path):
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[manifest] malformed JSONL skipped: {path}:{line_number}")
                    continue
                if isinstance(payload, dict):
                    yield payload
    except OSError as exc:
        print(f"[manifest] manifest read skipped: {path} ({exc})")


def read_jsonl_entries(path: Path) -> list[dict]:
    return list(iter_jsonl_entries(path) or [])


def write_jsonl_entries(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        temp_path.replace(path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def append_jsonl_entries(path: Path, entries: list[dict]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_manifest_unique_key(entry: dict) -> str:
    metadata = entry.get("metadata") or {}
    relative_path = metadata.get("relative_path")
    if relative_path:
        return f"relative::{normalize_manifest_identity(relative_path)}"
    source_path = metadata.get("source_path")
    if source_path:
        return f"source::{normalize_manifest_identity(source_path)}"
    video_path = entry.get("video_path")
    if video_path:
        return f"video::{normalize_manifest_identity(video_path)}"
    item_id = entry.get("item_id")
    if item_id:
        return f"item::{normalize_manifest_identity(item_id)}"
    pose_path = entry.get("pose_path")
    if pose_path:
        return f"pose::{normalize_manifest_identity(pose_path)}"
    return json.dumps(entry, sort_keys=True, ensure_ascii=False)


def normalize_manifest_identity(value) -> str:
    return str(value or "").strip().replace("\\", "/")


def count_manifest_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def build_active_manifest_state(
    *,
    target_labels: list[str],
    label_mapping: dict,
    source_manifests: dict[str, Path],
    quality_rules: dict | None = None,
) -> dict:
    schema_payload = {
        "materializer_version": 3,
        "target_labels": list(target_labels),
        "label_mapping": label_mapping,
        "quality_rules": quality_rules or {},
    }
    schema_fingerprint = hashlib.sha1(
        json.dumps(schema_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    source_state = {
        split_name: build_manifest_signature(source_path)
        for split_name, source_path in source_manifests.items()
    }
    return {
        "schema_fingerprint": schema_fingerprint,
        "sources": source_state,
    }


def build_manifest_signature(path: Path) -> dict:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": str(path),
            "exists": False,
        }
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def active_manifests_are_current(
    *,
    state_path: Path,
    active_paths: dict[str, Path],
    desired_state: dict,
) -> bool:
    if not all(path.exists() for path in active_paths.values()):
        return False
    if not state_path.exists():
        return False
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            current_state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        current_state.get("schema_fingerprint") == desired_state.get("schema_fingerprint")
        and current_state.get("sources") == desired_state.get("sources")
    )


def write_active_manifest_state(state_path: Path, desired_state: dict) -> None:
    write_json_atomic(
        state_path,
        {
            **desired_state,
            "materializer_version": 3,
            "materialized_at": datetime.now(UTC).astimezone().isoformat(),
        },
    )


def cleanup_transient_job_data(paths: dict) -> None:
    for key in ("raw_dir", "import_dir", "extracted_dir"):
        target = paths.get(key)
        if isinstance(target, Path) and target.exists():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    for key in (
        "current_raw_manifest",
        "current_split_train",
        "current_split_val",
        "current_split_test",
        "current_prepared_train",
        "current_prepared_val",
        "current_prepared_test",
    ):
        target = paths.get(key)
        if isinstance(target, Path) and target.exists():
            target.unlink()


def fetch_api_items(session: requests.Session, api_config: dict) -> list[dict]:
    list_url = api_config["list_url"]
    page_param = api_config.get("page_param")
    page_start = int(api_config.get("page_start", 1))
    static_params = dict(api_config.get("params", {}))
    max_pages = int(api_config.get("max_pages", 0))

    results = []
    page = page_start
    fetched_pages = 0

    while True:
        params = dict(static_params)
        if page_param:
            params[page_param] = page
        try:
            response = session.get(
                list_url,
                params=params,
                timeout=float(api_config.get("timeout_seconds", 60.0)),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "API 목록 조회에 실패했습니다.\n"
                f"- 요청 주소: {list_url}\n"
                f"- 에러: {exc}\n\n"
                "확인할 것:\n"
                "1. config의 api.list_url 이 실제 주소로 바뀌었는지\n"
                "2. items_path / next_path / fields 설정이 API 응답 구조와 맞는지\n"
                "3. 비공개 API라면 auth_required / auth_token_env 설정이 맞는지"
            ) from exc
        payload = response.json()
        items = extract_field(payload, api_config.get("items_path", "")) if api_config.get("items_path") else payload
        if not isinstance(items, list):
            raise RuntimeError("API items_path 결과가 리스트가 아닙니다.")
        results.extend(items)

        fetched_pages += 1
        if max_pages > 0 and fetched_pages >= max_pages:
            break

        next_value = extract_field(payload, api_config.get("next_path", "")) if api_config.get("next_path") else None
        if next_value:
            if isinstance(next_value, str) and next_value.startswith("http"):
                list_url = next_value
                page_param = None
            else:
                page += 1
            continue

        if page_param and len(items) > 0:
            page += 1
            continue
        break

    return results


def build_headers(api_config: dict) -> dict:
    headers = dict(api_config.get("headers", {}))
    auth_env = api_config.get("auth_token_env")
    auth_header = api_config.get("auth_header", "Authorization")
    auth_required = bool(api_config.get("auth_required", False))
    if auth_env:
        value = os.environ.get(auth_env)
        if not value:
            if auth_required:
                raise RuntimeError(
                    f"환경변수 {auth_env} 가 설정되지 않았습니다. "
                    "비공개 API라면 토큰을 export 하거나, 공개 API라면 config에서 "
                    "`auth_required: false` 또는 `auth_token_env: \"\"` 로 설정해 주세요."
                )
            return headers
        if auth_header.lower() == "authorization" and not value.lower().startswith("bearer "):
            value = f"Bearer {value}"
        headers[auth_header] = value
    return headers


def validate_source_config(config: dict, config_path: Path) -> None:
    source_mode = str(config.get("dataset_source", "json_api")).strip().lower()
    if source_mode == "aihub_shell":
        validate_aihub_shell_config(config.get("aihub_shell", {}), config_path)
        return
    validate_api_config(config.get("api", {}), config_path)


def validate_api_config(api_config: dict, config_path: Path) -> None:
    list_url = str(api_config.get("list_url", "")).strip()
    if not list_url:
        raise RuntimeError(
            f"API 설정이 비어 있습니다: {config_path}\n"
            "config의 api.list_url 에 실제 데이터 목록 API 주소를 넣어 주세요."
        )

    parsed = urlparse(list_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            f"api.list_url 형식이 올바르지 않습니다: {list_url}\n"
            "예: https://example.com/api/videos"
        )

    placeholder_hosts = {
        "your-dataset-api.example.com",
        "example.com",
    }
    if parsed.netloc in placeholder_hosts or "your-dataset-api" in parsed.netloc:
        raise RuntimeError(
            "예제용 placeholder API 주소가 그대로 들어 있습니다.\n"
            f"- 현재 주소: {list_url}\n"
            f"- 설정 파일: {config_path}\n\n"
            "해야 할 일:\n"
            "1. configs/action_training.example.json 의 api.list_url 을 실제 API 주소로 변경\n"
            "2. API 응답에 맞게 items_path / fields.id / fields.label / fields.download_url 수정\n"
            "3. 비공개 API라면 auth_required 와 auth_token_env 도 함께 설정"
        )


def validate_aihub_shell_config(shell_config: dict, config_path: Path) -> None:
    if shell_config.get("datasetkey") in (None, "") and shell_config.get("datapackagekey") in (None, ""):
        raise RuntimeError(
            f"AIHub shell 설정이 부족합니다: {config_path}\n"
            "aihub_shell.datasetkey 또는 aihub_shell.datapackagekey 중 하나는 필요합니다."
        )
    resolve_aihub_shell_path(shell_config)
    resolve_aihub_api_key(shell_config)


def resolve_aihub_shell_path(shell_config: dict) -> str:
    configured = str(shell_config.get("path", "")).strip()
    if configured:
        raw_candidate = Path(configured).expanduser()
        candidates = [raw_candidate]
        if raw_candidate.suffix == "":
            candidates.extend(
                [
                    raw_candidate.with_suffix(".exe"),
                    raw_candidate.with_suffix(".bat"),
                    raw_candidate.with_suffix(".cmd"),
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        raise RuntimeError(
            f"aihubshell 경로를 찾지 못했습니다: {configured}\n"
            "확인할 것:\n"
            "1. config의 aihub_shell.path 가 현재 PC 기준 경로인지\n"
            "2. Windows라면 aihubshell.exe 인지\n"
            "3. 프로젝트 루트에 있다면 path를 'aihubshell' 로 둘 수 있는지"
        )

    discovered = shutil.which("aihubshell")
    if discovered:
        return discovered

    raise RuntimeError(
        "aihubshell 실행 파일을 찾지 못했습니다.\n"
        "AIHub 공식 안내처럼 aihubshell을 설치한 뒤,\n"
        "1. PATH에 등록하거나\n"
        "2. config의 aihub_shell.path 에 실행 파일 경로를 넣어 주세요."
    )


def resolve_aihub_api_key(shell_config: dict) -> str:
    direct_key = str(shell_config.get("api_key", "")).strip()
    if direct_key:
        return direct_key

    env_name = str(shell_config.get("api_key_env", "AIHUB_API_KEY")).strip()
    value = os.environ.get(env_name)
    if value:
        return value

    raise RuntimeError(
        f"AIHub API 키를 찾지 못했습니다.\n"
        f"- 환경변수 {env_name} 를 export 하거나\n"
        "- config의 aihub_shell.api_key 에 직접 넣어 주세요.\n"
        "또한 AIHub 데이터셋은 승인 완료 후 다운로드 가능합니다."
    )


def infer_label_from_path(
    video_path: Path,
    label_mapping: dict,
    *,
    label_matchers: list[tuple[str, str, str, str]] | None = None,
) -> tuple[str, str | None]:
    relative_text = str(video_path).replace("\\", "/")
    active_matchers = label_matchers if label_matchers is not None else build_label_mapping_matchers(label_mapping)
    raw_text_lower = relative_text.lower()
    normalized_text = normalize_match_text(raw_text_lower)
    for source_text, target_label, source_text_lower, normalized_source in active_matchers:
        if (
            source_text in relative_text
            or source_text_lower in raw_text_lower
            or (normalized_source and normalized_source in normalized_text)
        ):
            return source_text, target_label
    return "", None


def build_download_url(item: dict, api_config: dict) -> str:
    direct_key = api_config["fields"].get("download_url")
    if direct_key:
        direct_url = extract_field(item, direct_key)
        if direct_url:
            return str(direct_url)

    template = api_config.get("download_url_template")
    if template:
        item_id = extract_field(item, api_config["fields"]["id"])
        return template.format(item_id=item_id)
    return ""


def build_filename(item: dict, api_config: dict, download_url: str, item_id: str) -> str:
    filename_key = api_config["fields"].get("filename")
    if filename_key:
        filename = extract_field(item, filename_key)
        if filename:
            return sanitize_filename(str(filename))

    parsed = urlparse(download_url)
    name = Path(parsed.path).name
    if name:
        return sanitize_filename(name)
    return f"{slugify(item_id)}.mp4"


def download_to_file(session: requests.Session, url: str, target_path: Path, timeout: float) -> None:
    if target_path.exists() and target_path.stat().st_size > 0:
        return

    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_field(data, path: str):
    if not path:
        return data
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z가-힣._-]+", "_", value)
    return value.strip("._-") or "item"


def sanitize_filename(filename: str) -> str:
    filename = filename.replace("\\", "_").replace("/", "_")
    if "." not in filename:
        filename += ".mp4"
    return slugify(filename.rsplit(".", 1)[0]) + "." + filename.rsplit(".", 1)[1]


def get_target_labels(config: dict) -> list[str]:
    labels = config.get("dataset", {}).get("target_labels")
    if labels:
        return list(labels)

    mapped = set(config.get("dataset", {}).get("label_mapping", {}).values())
    if not mapped:
        raise RuntimeError("dataset.target_labels 또는 dataset.label_mapping 이 필요합니다.")
    return sorted(mapped)


if __name__ == "__main__":
    main()

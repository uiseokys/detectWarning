from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path


DEFAULT_ACTION_TRAINING_CONFIG: dict = {
    "paths": {
        "workspace_dir": "../training_data/action_pipeline",
        "workspace_dir_env": "DETECTWARNING_WORKSPACE_DIR",
    },
    "dataset_source": "json_api",
    "api": {
        "list_url": "",
        "items_path": "results",
        "next_path": "next",
        "page_param": "page",
        "page_start": 1,
        "max_pages": 0,
        "timeout_seconds": 120.0,
        "auth_required": False,
        "auth_header": "Authorization",
        "auth_token_env": "",
        "headers": {},
        "params": {},
        "fields": {
            "id": "id",
            "label": "label",
            "filename": "file_name",
            "download_url": "download_url",
        },
    },
    "aihub_shell": {
        "path": "aihubshell",
        "mode": "d",
        "datasetkey": None,
        "datapackagekey": None,
        "filekey": None,
        "api_key_env": "AIHUB_API_KEY",
        "api_key": "",
        "note": "",
    },
    "dataset": {
        "target_labels": [],
        "label_mapping": {},
        "video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".wmv"],
        "max_items_per_class": 0,
    },
    "split": {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42,
    },
    "preprocess": {
        "device": "cuda:0",
        "person_score_threshold": 0.25,
        "person_imgsz": 640,
        "detector_batch_size": 8,
        "video_prefetch_workers": 1,
        "compress_prepared_pose": False,
        "sequence_length": 48,
        "max_frames_to_scan": 160,
        "min_frames_with_person": 8,
        "fallback_min_frames_with_person": 4,
        "min_confirmed_frames_with_person": 2,
        "max_missing_frames_ratio": 0.85,
        "max_fallback_frames_ratio": 0.6,
        "min_total_keypoints": 32,
        "allow_partial_pose": True,
        "allow_padding": True,
        "filter_existing_prepared_by_quality": True,
        "allow_rejected_pose_fallback": True,
        "fallback_min_keypoints": 5,
        "fallback_min_detection_confidence": 0.2,
        "fallback_min_person_score": 30,
        "skip_invalid_labels": True,
        "strict_data_validation": False,
    },
    "training": {
        "device": "cuda",
        "amp": True,
        "amp_dtype": "auto",
        "compile_model": True,
        "epochs": 30,
        "batch_size": 16,
        "eval_batch_size": 0,
        "dataset_cache_size": 2048,
        "learning_rate": 0.0007,
        "weight_decay": 0.003,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.35,
        "label_smoothing": 0.03,
        "loss": "focal",
        "focal_gamma": 2.0,
        "class_weight": "balanced",
        "balanced_sampler": True,
        "class_weight_multipliers": {},
        "adaptive_class_weighting": {
            "enabled": True,
            "target_recall": 0.55,
            "target_f1": 0.45,
            "low_recall_multiplier": 2.0,
            "low_f1_multiplier": 1.6,
            "missing_prediction_multiplier": 2.5,
            "minority_count_multiplier": 1.25,
            "min_multiplier": 1.0,
            "max_multiplier": 2.5,
            "min_validation_support": 1,
        },
        "block_resume_on_missing_predictions": True,
        "grad_clip_norm": 1.0,
        "seed": 42,
        "deterministic": False,
        "num_workers": "auto",
        "pin_memory": "auto",
        "prefetch_factor": 2,
        "persistent_workers": True,
        "early_stopping_patience": 7,
        "early_stopping_min_delta": 0.001,
        "imbalance_warn_min_samples": 8,
        "imbalance_warn_ratio": 3.0,
    },
    "gpu_auto_tune": {
        "enabled": False,
        "reserve_memory_mb": 1536,
        "preprocess": {
            "enabled": True,
            "max_detector_batch_size": 64,
            "max_video_prefetch_workers": 4,
            "max_person_imgsz": 960,
            "max_frames_to_scan": 240,
        },
        "training": {
            "enabled": True,
            "max_batch_size": 96,
            "max_eval_batch_size": 192,
        },
    },
    "continual_learning": {
        "enabled": True,
        "resume_from_best": True,
        "cleanup_raw_after_job": True,
    },
    "dashboard": {
        "allowed_origins": [],
    },
    "pages_sync": {
        "enabled": False,
        "pages_dir": "",
        "project_name": "detectWarning",
        "report_title": "행동 학습 결과 리포트",
        "report_url": "",
        "live_url": "",
        "live_url_env": "DETECTWARNING_LIVE_URL",
        "redirect_delay_seconds": 3,
        "git_auto_push": False,
        "git_commit_prefix": "Update dashboard",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_label_list(labels) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels or []:
        value = str(label).strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _normalize_label_mapping(mapping) -> dict[str, str]:
    if not isinstance(mapping, dict):
        return {}
    normalized: dict[str, str] = {}
    for source, target in mapping.items():
        source_value = str(source).strip()
        target_value = str(target).strip()
        if source_value and target_value:
            normalized[source_value] = target_value
    return normalized


def _normalize_allowed_origins(value) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def load_action_training_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise RuntimeError(f"설정 파일 최상위 구조는 객체여야 합니다: {config_path}")

    config = _deep_merge(DEFAULT_ACTION_TRAINING_CONFIG, raw)
    dataset_config = config.setdefault("dataset", {})
    dataset_config["target_labels"] = _normalize_label_list(dataset_config.get("target_labels"))
    dataset_config["label_mapping"] = _normalize_label_mapping(dataset_config.get("label_mapping"))
    dataset_config["video_extensions"] = _normalize_label_list(dataset_config.get("video_extensions"))

    dashboard_config = config.setdefault("dashboard", {})
    dashboard_config["allowed_origins"] = _normalize_allowed_origins(dashboard_config.get("allowed_origins"))

    if not dataset_config["target_labels"]:
        mapped_targets = _normalize_label_list(dataset_config["label_mapping"].values())
        dataset_config["target_labels"] = mapped_targets

    if not dataset_config["target_labels"]:
        raise RuntimeError(
            f"dataset.target_labels 또는 dataset.label_mapping 설정이 필요합니다: {config_path}"
        )

    if float(config["split"]["train_ratio"]) + float(config["split"]["val_ratio"]) + float(config["split"]["test_ratio"]) <= 0:
        raise RuntimeError(f"split 비율 합계가 0 이하입니다: {config_path}")

    return config


def resolve_pages_sync_config(config: dict, base_dir: Path) -> dict:
    raw = config.get("pages_sync") or {}
    pages_dir_raw = str(raw.get("pages_dir") or "").strip()
    live_url = str(raw.get("live_url") or "").strip()
    live_url_env = str(raw.get("live_url_env") or "").strip()
    if not live_url and live_url_env:
        live_url = str(os.getenv(live_url_env, "")).strip()

    pages_dir = None
    if pages_dir_raw:
        candidate = Path(pages_dir_raw).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        pages_dir = candidate

    enabled = bool(raw.get("enabled")) and pages_dir is not None
    return {
        "enabled": enabled,
        "pages_dir": pages_dir,
        "project_name": str(raw.get("project_name") or "detectWarning"),
        "report_title": str(raw.get("report_title") or "행동 학습 결과 리포트"),
        "report_url": str(raw.get("report_url") or "").strip(),
        "live_url": live_url,
        "redirect_delay_seconds": int(raw.get("redirect_delay_seconds") or 3),
        "git_auto_push": bool(raw.get("git_auto_push", False)),
        "git_commit_prefix": str(raw.get("git_commit_prefix") or "Update dashboard"),
    }

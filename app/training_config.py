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
        "min_frames_with_person": 4,
    },
    "training": {
        "device": "cuda",
        "amp": True,
        "amp_dtype": "auto",
        "compile_model": True,
        "epochs": 20,
        "batch_size": 16,
        "eval_batch_size": 0,
        "dataset_cache_size": 2048,
        "learning_rate": 0.001,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "num_workers": "auto",
        "pin_memory": "auto",
        "prefetch_factor": 2,
        "persistent_workers": True,
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

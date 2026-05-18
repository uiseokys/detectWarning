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
    "guideline_sampling": {
        "enabled": False,
        "normal_label": "normal",
        "include_normal": True,
        "context_before_seconds": 12.0,
        "context_after_seconds": 12.0,
        "clip_seconds": 6.0,
        "clip_stride_seconds": 6.0,
        "normal_clip_seconds": 6.0,
        "normal_stride_seconds": 30.0,
        "max_event_context_clips_per_video": 16,
        "max_normal_clips_per_video": 8,
        "max_total_clips_per_video": 24,
        "target_normal_clip_ratio": 0.25,
        "auto_normal_ratio": {
            "enabled": True,
            "min_ratio": 0.20,
            "max_ratio": 0.35,
            "step": 0.05,
            "danger_to_normal_threshold": 0.20,
            "normal_to_danger_threshold": 0.25,
            "predicted_normal_bias_threshold": 0.15,
        },
        "pose_mode": "reuse_existing",
        "fallback_event_start_ratio": 0.25,
        "fallback_event_end_ratio": 0.75,
        "event_sample_weight": 1.0,
        "normal_sample_weight": 1.0,
        "hard_negative_sample_weight": 1.5,
        "keep_pose_failed_clips": True,
        "include_pose_skipped_videos": True,
        "min_pose_valid_frames": 1,
        "fallback_min_pose_valid_frames": 1,
        "min_confirmed_pose_frames": 0,
        "min_pose_total_keypoints": 1,
        "max_pose_missing_frames_ratio": 1.0,
        "max_pose_fallback_frames_ratio": 1.0,
        "low_pose_valid_frame_ratio": 0.25,
        "low_pose_confidence": 0.12,
        "low_pose_sample_weight_multiplier": 0.35,
        "rgb_only_sample_weight_multiplier": 0.15,
        "rgb_ready_fallback_sample_weight_multiplier": 0.45,
        "xml_missing_sample_weight_multiplier": 0.35,
        "xml_roots": [],
        "rgb_feature_dir": "",
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
        "person_not_detected_retry": {
            "enabled": True,
            "reasons": ["person_not_detected"],
            "attempts": [
                {"person_score_threshold": 0.15, "person_imgsz": 960},
                {"person_score_threshold": 0.10, "person_imgsz": 1280},
            ],
        },
        "detector_batch_size": 8,
        "video_prefetch_workers": 1,
        "compress_prepared_pose": False,
        "sequence_length": 32,
        "max_frames_to_scan": 120,
        "min_frames_with_person": 8,
        "fallback_min_frames_with_person": 4,
        "min_confirmed_frames_with_person": 2,
        "max_missing_frames_ratio": 0.85,
        "max_fallback_frames_ratio": 0.6,
        "min_total_keypoints": 32,
        "allow_partial_pose": True,
        "allow_padding": True,
        "filter_existing_prepared_by_quality": True,
        "allow_insufficient_pose_samples_for_rgb_fallback": True,
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
        "selection_metric": "accuracy",
        "weight_decay": 0.003,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.35,
        "label_smoothing": 0.0,
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
            "normal_ratio_multiplier_enabled": True,
            "normal_ratio_target": 0.25,
            "normal_ratio_min": 0.20,
            "normal_ratio_multiplier": 1.35,
            "normal_ratio_max_multiplier": 1.6,
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
        "early_stopping_patience": 12,
        "early_stopping_min_delta": 0.001,
        "overfit_guard_enabled": False,
        "overfit_guard_min_epoch": 16,
        "overfit_guard_loss_gap": 8.0,
        "overfit_guard_patience": 3,
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
    "training_tasks": {
        "enabled": True,
        "normal_label": "normal",
        "detection_enabled": True,
        "classification_enabled": True,
        "pose_classification_enabled": True,
        "detection_target_metric": "detection_practical",
        "detection_min_recall": 0.85,
        "detection_max_false_alarm_rate": 0.55,
        "classification_target_metric": "macro_f1_supported",
        "classification_class_bias_multipliers": [
            0.65,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1.0,
            1.1,
            1.15,
            1.25,
            1.35,
            1.4,
            1.6,
            1.75,
            1.9,
            2.1,
            2.3,
        ],
        "pose_classification_epochs": 30,
        "pose_classification_learning_rate": 0.0007,
        "pose_classification_selection_metric": "accuracy",
        "pose_classification_weight_decay": 0.003,
        "pose_classification_dropout": 0.35,
        "pose_classification_label_smoothing": 0.0,
        "pose_classification_loss": "focal",
        "pose_classification_focal_gamma": 1.5,
        "pose_classification_early_stopping_patience": 12,
        "pose_classification_early_stopping_min_delta": 0.0015,
        "pose_classification_overfit_guard_enabled": False,
        "pose_classification_overfit_guard_min_epoch": 16,
        "pose_classification_overfit_guard_loss_gap": 8.0,
        "pose_classification_overfit_guard_patience": 3,
        "pose_classification_max_duplicate_pose_label_samples": 0,
        "pose_classification_class_weight_multipliers": {
            "abduction": 2.5
        },
        "feature_models": [
            "extra_trees_fast",
            "extra_trees_leaf2_fast",
            "extra_trees_accuracy",
            "random_forest_leaf2",
            "hist_gradient",
            "logreg",
        ],
    },
    "auto_tune": {
        "max_trials": 6,
        "target_recall": 0.35,
        "target_metric": "macro_f1_supported",
        "exploration_enabled": True,
        "seed_sweep_enabled": True,
        "seeds": [7, 13, 21, 42, 77, 123, 2026],
        "promote_best": True,
        "update_config_with_best": False,
        "ensemble_enabled": True,
        "hybrid_enabled": True,
        "hybrid_feature_weight_max": 0.95,
        "hybrid_feature_weight_step": 0.05,
        "hybrid_auto_feature_weighting": {
            "enabled": True,
            "min_rgb_ready_ratio": 0.8,
            "low_fallback_ratio": 0.05,
            "medium_fallback_ratio": 0.15,
            "high_fallback_ratio": 0.3,
            "low_feature_weight_max": 0.8,
            "medium_feature_weight_max": 0.9,
            "high_feature_weight_max": 0.95,
        },
        "output_dir": "",
    },
    "dashboard": {
        "allowed_origins": [],
        "notifications": {
            "enabled": False,
            "provider": "ntfy",
            "ntfy_server": "https://ntfy.sh",
            "ntfy_topic": "",
            "notify_on": ["completed", "completed_warning", "error", "aborted"],
        },
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

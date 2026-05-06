from __future__ import annotations

import argparse
import atexit
from collections import Counter
import hashlib
from html import escape
import json
import os
import re
import signal
import shutil
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
    build_source_label_matchers,
    collect_aihub_file_entries,
    extract_json_payload,
    fetch_aihub_file_tree,
    fetch_aihub_file_tree_via_shell,
    get_target_labels,
    load_config,
    match_source_label_text,
    parse_aihub_file_tree_listing,
    resolve_aihub_shell_path,
    resolve_paths,
)
from dashboard_runtime import (
    build_job,
    build_retry_job_from,
    classify_job_exit,
    collect_result_summary,
    current_timestamp,
    decode_process_output,
    flush_pages_pushes,
    persist_launcher_history,
    read_log_preview,
    read_log_tail,
    snapshot_job,
    sync_pages_live,
    sync_pages_report,
    write_dashboard_status,
)
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
GPU_STATUS_CACHE: dict[str, object] = {"timestamp": 0.0, "value": None}
GPU_STATUS_CACHE_LOCK = threading.Lock()
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
AIHUB_FILEKEY_LOOKUP_CACHE_SECONDS = 15 * 60
AIHUB_FILEKEY_LOOKUP_TEXT_KEYS = (
    "source_label",
    "source_alias",
    "sourceLabel",
    "sourceAlias",
    "label",
    "name",
    "fileName",
    "fileNm",
    "filePath",
    "path",
)
AIHUB_FILEKEY_LOOKUP_STATUS_KEYS = (
    "trainable",
    "trained",
    "prepared",
    "excluded",
    "queued",
    "running",
    "unknown",
)
AIHUB_TRAINED_JOB_STATES = {"completed", "completed_warning"}
AIHUB_PREPARED_JOB_STATES = {"data_ready", "deferred", "waiting_for_data"}
AIHUB_ZIP_GROUP_ORDER = ("outsidedoor", "insidedoor", "inside_croki")
AIHUB_AUTO_RECOMMEND_ZIP_GROUPS = ("outsidedoor",)
AIHUB_AUTO_RECOMMEND_FALLBACK_ZIP_GROUPS = ("insidedoor",)
AIHUB_AUTO_RECOMMEND_EXCLUDED_ZIP_GROUPS = ("inside_croki",)
AIHUB_AUTO_RECOMMEND_INSIDE_FALLBACK_LABELS = {
    "abduction": "minority_or_diagnosed",
    "collapse": "diagnosed",
}
AIHUB_AUTO_RECOMMEND_MAX_CLASS_RATIO = 1.5
AIHUB_AUTO_RECOMMEND_MAX_PENDING = 1
AIHUB_ZIP_GROUP_PATTERN = re.compile(
    r"^(?P<group>outsidedoor|insidedoor|inside_croki)(?:[_-](?P<number>\d+))?",
    re.IGNORECASE,
)
DIAGNOSIS_RECOMMENDATION_LEVEL_WEIGHTS = {
    "critical": 160.0,
    "warning": 100.0,
    "warn": 100.0,
    "watch": 45.0,
    "good": 0.0,
    "normal": 0.0,
    "improving": 0.0,
}
DIAGNOSIS_EMPTY_CLASS_PRIORITY = 5000.0


def optional_aihub_api_key(shell_config: dict, override: str = "") -> str:
    explicit = str(override or "").strip()
    if explicit:
        return explicit
    direct_key = str(shell_config.get("api_key", "")).strip()
    if direct_key:
        return direct_key
    env_name = str(shell_config.get("api_key_env", "AIHUB_API_KEY")).strip()
    return str(os.environ.get(env_name) or "").strip()


def aihub_entry_text(entry: dict) -> str:
    return " ".join(
        str(entry.get(key) or "")
        for key in AIHUB_FILEKEY_LOOKUP_TEXT_KEYS
        if entry.get(key) not in (None, "")
    )


def aihub_filekey_sort_key(entry: dict) -> tuple[int, int | str]:
    value = str(entry.get("filekey") or "").strip()
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def classify_aihub_zip_name(value: str) -> dict:
    raw_value = str(value or "").strip()
    filename = re.split(r"[/\\]", raw_value)[-1].strip()
    match = AIHUB_ZIP_GROUP_PATTERN.match(filename.lower())
    if not match:
        return {
            "zip_group": "",
            "zip_number": None,
            "zip_filename": filename,
        }
    number_text = match.group("number")
    return {
        "zip_group": match.group("group").lower(),
        "zip_number": int(number_text) if number_text else None,
        "zip_filename": filename,
    }


def aihub_entry_zip_group(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    explicit_group = str(entry.get("zip_group") or "").strip().lower()
    if explicit_group:
        return explicit_group
    for key in ("zip_filename", "name", "fileName", "fileNm", "filePath", "path"):
        value = str(entry.get(key) or "").strip()
        if not value:
            continue
        zip_group = classify_aihub_zip_name(value).get("zip_group")
        if zip_group:
            return str(zip_group)
    return ""


def aihub_entry_matches_zip_groups(entry: dict, allowed_zip_groups: set[str]) -> bool:
    if not allowed_zip_groups:
        return True
    return aihub_entry_zip_group(entry) in allowed_zip_groups


def normalize_recommendation_zip_groups(values) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def build_auto_recommendation_policy() -> dict:
    return {
        "primary_zip_groups": list(AIHUB_AUTO_RECOMMEND_ZIP_GROUPS),
        "fallback_zip_groups": list(AIHUB_AUTO_RECOMMEND_FALLBACK_ZIP_GROUPS),
        "excluded_zip_groups": list(AIHUB_AUTO_RECOMMEND_EXCLUDED_ZIP_GROUPS),
        "inside_fallback_labels": dict(AIHUB_AUTO_RECOMMEND_INSIDE_FALLBACK_LABELS),
        "max_class_ratio": AIHUB_AUTO_RECOMMEND_MAX_CLASS_RATIO,
    }


def normalize_auto_recommendation_policy(policy: dict | None) -> dict:
    if not isinstance(policy, dict):
        return {}
    max_ratio = policy.get("max_class_ratio", 0)
    try:
        max_ratio = float(max_ratio or 0)
    except (TypeError, ValueError):
        max_ratio = 0.0
    fallback_labels = policy.get("inside_fallback_labels")
    if not isinstance(fallback_labels, dict):
        fallback_labels = {}
    return {
        "primary_zip_groups": normalize_recommendation_zip_groups(policy.get("primary_zip_groups")),
        "fallback_zip_groups": normalize_recommendation_zip_groups(policy.get("fallback_zip_groups")),
        "excluded_zip_groups": normalize_recommendation_zip_groups(policy.get("excluded_zip_groups")),
        "inside_fallback_labels": {
            normalize_recommendation_label(label): str(mode or "").strip().lower()
            for label, mode in fallback_labels.items()
            if normalize_recommendation_label(label)
        },
        "max_class_ratio": max_ratio,
    }


def normalize_recommendation_label(value) -> str:
    return str(value or "").strip().lower()


def insight_level_weight(level) -> float:
    return DIAGNOSIS_RECOMMENDATION_LEVEL_WEIGHTS.get(
        str(level or "").strip().lower(),
        DIAGNOSIS_RECOMMENDATION_LEVEL_WEIGHTS["watch"],
    )


def add_diagnosis_label_priority(
    priorities: dict[str, dict],
    label,
    *,
    weight: float,
    reason: str,
    level: str = "watch",
) -> None:
    normalized = normalize_recommendation_label(label)
    if not normalized or weight <= 0:
        return
    payload = priorities.setdefault(
        normalized,
        {
            "label": str(label).strip(),
            "priority": 0.0,
            "reasons": [],
            "level": "good",
        },
    )
    payload["priority"] = round(float(payload.get("priority", 0.0) or 0.0) + float(weight), 6)
    if reason and reason not in payload["reasons"]:
        payload["reasons"].append(reason)
    if insight_level_weight(level) > insight_level_weight(payload.get("level")):
        payload["level"] = str(level or "watch").strip().lower()


def build_diagnosis_label_priorities(insights: dict | None) -> dict[str, dict]:
    insights = insights if isinstance(insights, dict) else {}
    priorities: dict[str, dict] = {}

    for item in insights.get("class_insights") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level") or "watch").strip().lower()
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        title = str(item.get("title") or "")
        base_weight = insight_level_weight(level)
        title_bonus = 40.0 if ("위험" in title or "danger" in title.lower()) else 0.0
        label = details.get("label")
        if label:
            add_diagnosis_label_priority(
                priorities,
                label,
                weight=base_weight + title_bonus,
                reason=title or "class insight",
                level=level,
            )
        for label in details.get("labels") or []:
            add_diagnosis_label_priority(
                priorities,
                label,
                weight=base_weight,
                reason=title or "class insight",
                level=level,
            )

    for item in insights.get("confusion_insights") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level") or "watch").strip().lower()
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        title = str(item.get("title") or "")
        base_weight = insight_level_weight(level)
        source_label = details.get("from")
        if source_label:
            add_diagnosis_label_priority(
                priorities,
                source_label,
                weight=base_weight + 25.0,
                reason=title or "confusion insight",
                level=level,
            )
        for label in details.get("labels") or []:
            add_diagnosis_label_priority(
                priorities,
                label,
                weight=base_weight,
                reason=title or "confusion insight",
                level=level,
            )

    for item in insights.get("diagnostics") or []:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level") or "watch").strip().lower()
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        title = str(item.get("title") or "")
        base_weight = insight_level_weight(level)
        for key in ("minority_label", "dominant_label"):
            label = details.get(key)
            if key == "dominant_label":
                continue
            if label:
                add_diagnosis_label_priority(
                    priorities,
                    label,
                    weight=base_weight * 0.7,
                    reason=title or "distribution insight",
                    level=level,
                )
        for label in details.get("empty_labels") or []:
            add_diagnosis_label_priority(
                priorities,
                label,
                weight=DIAGNOSIS_EMPTY_CLASS_PRIORITY + base_weight,
                reason=title or "empty class",
                level="critical",
            )
        for row in details.get("low_sample_labels") or []:
            if isinstance(row, dict) and row.get("label"):
                add_diagnosis_label_priority(
                    priorities,
                    row.get("label"),
                    weight=base_weight * 0.8,
                    reason=title or "low sample class",
                    level=level,
                )

    for item in insights.get("data_quality_insights") or []:
        if not isinstance(item, dict):
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        label = details.get("label")
        if label:
            level = str(item.get("level") or "watch").strip().lower()
            add_diagnosis_label_priority(
                priorities,
                label,
                weight=insight_level_weight(level) * 0.45,
                reason=str(item.get("title") or "data quality insight"),
                level=level,
            )

    return priorities


def diagnosis_priority_for_label(label, priorities: dict[str, dict]) -> dict:
    return priorities.get(
        normalize_recommendation_label(label),
        {"label": str(label or "").strip(), "priority": 0.0, "reasons": [], "level": "good"},
    )


def normalize_label_count_map(counts: dict | None) -> Counter[str]:
    normalized: Counter[str] = Counter()
    if not isinstance(counts, dict):
        return normalized
    for label, count in counts.items():
        label_key = str(label or "").strip()
        if not label_key:
            continue
        try:
            normalized[label_key] += max(int(count or 0), 0)
        except (TypeError, ValueError):
            continue
    return normalized


def normalize_split_label_counts(
    prepared_label_counts: dict[str, int] | None,
    prepared_split_label_counts: dict[str, dict[str, int]] | None,
) -> dict[str, Counter[str]]:
    if isinstance(prepared_split_label_counts, dict) and prepared_split_label_counts:
        return {
            str(split_name): normalize_label_count_map(counts if isinstance(counts, dict) else {})
            for split_name, counts in prepared_split_label_counts.items()
        }
    fallback_counts = normalize_label_count_map(prepared_label_counts)
    if not fallback_counts:
        return {}
    return {"train": Counter(fallback_counts), "val": Counter(fallback_counts)}


def recommendation_coverage_score(
    label: str,
    *,
    prepared_counts: Counter[str],
    split_counts: dict[str, Counter[str]],
    planned_count: int = 0,
) -> dict:
    train_count = int(split_counts.get("train", Counter()).get(label, 0) or 0)
    val_count = int(split_counts.get("val", Counter()).get(label, 0) or 0)
    total_count = int(prepared_counts.get(label, 0) or 0)
    effective_train_count = train_count + max(int(planned_count or 0), 0)
    effective_val_count = val_count + max(int(planned_count or 0), 0)
    if effective_train_count <= 0 and effective_val_count <= 0:
        coverage_rank = 0
        coverage_state = "missing"
    elif effective_train_count <= 0 or effective_val_count <= 0:
        coverage_rank = 1
        coverage_state = "split_incomplete"
    else:
        coverage_rank = 2
        coverage_state = "covered"
    return {
        "coverage_rank": coverage_rank,
        "coverage_state": coverage_state,
        "train_count": train_count,
        "val_count": val_count,
        "prepared_count": total_count,
        "train_balance_bucket": recommendation_count_bucket(train_count, 16),
        "val_balance_bucket": recommendation_count_bucket(val_count, 4),
        "prepared_balance_bucket": recommendation_count_bucket(total_count, 20),
        "effective_train_count": effective_train_count,
        "effective_val_count": effective_val_count,
    }


def recommendation_count_bucket(count: int, bucket_size: int) -> int:
    return max(int(count or 0), 0) // max(int(bucket_size or 1), 1)


def recommendation_effective_label_counts(
    *,
    prepared_counts: Counter[str],
    planned_counts: Counter[str],
    candidate_labels: set[str],
) -> Counter[str]:
    labels = {
        str(label or "").strip()
        for label in (*prepared_counts.keys(), *planned_counts.keys(), *candidate_labels)
        if str(label or "").strip()
    }
    return Counter(
        {
            label: max(int(prepared_counts.get(label, 0) or 0), 0)
            + max(int(planned_counts.get(label, 0) or 0), 0)
            for label in labels
        }
    )


def recommendation_minority_state(label: str, label_counts: Counter[str]) -> dict:
    normalized_label = str(label or "").strip()
    if not normalized_label or not label_counts:
        return {"is_minority": False, "min_count": 0, "label_count": 0}
    label_count = int(label_counts.get(normalized_label, 0) or 0)
    min_count = min(int(count or 0) for count in label_counts.values())
    return {
        "is_minority": label_count <= min_count,
        "min_count": min_count,
        "label_count": label_count,
    }


def recommendation_balance_limit_state(
    label: str,
    label_counts: Counter[str],
    *,
    max_ratio: float,
) -> dict:
    normalized_label = str(label or "").strip()
    if not normalized_label or max_ratio <= 0 or len(label_counts) < 2:
        return {
            "allowed": True,
            "max_ratio": max_ratio,
            "min_count": 0,
            "label_count": int(label_counts.get(normalized_label, 0) or 0),
            "projected_count": int(label_counts.get(normalized_label, 0) or 0) + 1,
            "limit": None,
            "reason": "",
        }
    label_count = int(label_counts.get(normalized_label, 0) or 0)
    min_count = min(int(count or 0) for count in label_counts.values())
    projected_count = label_count + 1
    if label_count <= min_count:
        allowed = True
    elif min_count <= 0:
        allowed = False
    else:
        allowed = label_count <= (float(min_count) * float(max_ratio))
    return {
        "allowed": allowed,
        "max_ratio": max_ratio,
        "min_count": min_count,
        "label_count": label_count,
        "projected_count": projected_count,
        "limit": round(float(min_count) * float(max_ratio), 6) if min_count > 0 else 0,
        "reason": "" if allowed else "class_ratio_limit",
    }


def recommendation_zip_scope_state(
    entry: dict,
    *,
    allowed_groups: set[str],
    policy: dict,
    label: str,
    primary_candidate_labels: set[str],
    diagnosis_priority: dict,
    label_counts: Counter[str],
    strict_fallback: bool,
) -> dict:
    zip_group = aihub_entry_zip_group(entry) or str(entry.get("zip_group") or "").strip().lower()
    if not policy:
        return {
            "allowed": aihub_entry_matches_zip_groups(entry, allowed_groups),
            "zip_group": zip_group,
            "scope_reason": "allowed_zip_group",
        }

    primary_groups = policy.get("primary_zip_groups") or allowed_groups
    fallback_groups = policy.get("fallback_zip_groups") or set()
    excluded_groups = policy.get("excluded_zip_groups") or set()
    fallback_labels = policy.get("inside_fallback_labels") or {}
    normalized_label = normalize_recommendation_label(label)
    insight_priority = float(diagnosis_priority.get("priority", 0.0) or 0.0)
    minority_state = recommendation_minority_state(label, label_counts)

    if zip_group in excluded_groups:
        return {"allowed": False, "zip_group": zip_group, "scope_reason": "excluded_zip_group"}
    if not zip_group:
        return {"allowed": False, "zip_group": zip_group, "scope_reason": "missing_zip_group"}
    if not primary_groups or zip_group in primary_groups:
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "primary_zip_group"}
    if zip_group not in fallback_groups:
        return {"allowed": False, "zip_group": zip_group, "scope_reason": "unsupported_zip_group"}
    fallback_mode = str(fallback_labels.get(normalized_label) or "").strip().lower()
    if not fallback_mode:
        return {"allowed": False, "zip_group": zip_group, "scope_reason": "fallback_label_not_allowed"}
    if normalized_label in {normalize_recommendation_label(item) for item in primary_candidate_labels}:
        return {"allowed": False, "zip_group": zip_group, "scope_reason": "primary_candidate_available"}
    if not strict_fallback:
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "fallback_zip_group"}
    if fallback_mode == "diagnosed" and insight_priority > 0:
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "diagnosed_fallback"}
    if fallback_mode == "minority_or_diagnosed" and (
        insight_priority > 0 or minority_state.get("is_minority")
    ):
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "minority_or_diagnosed_fallback"}
    return {"allowed": False, "zip_group": zip_group, "scope_reason": "fallback_condition_not_met"}


def normalize_aihub_lookup_entries(
    entries: list[dict],
    *,
    label_mapping: dict,
    excluded_source_labels: list,
    target_labels: list[str] | None = None,
) -> list[dict]:
    label_matchers = build_source_label_matchers(label_mapping.keys())
    excluded_matchers = build_source_label_matchers(excluded_source_labels)
    target_label_set = {
        str(label).strip()
        for label in (target_labels or [])
        if str(label).strip()
    }
    normalized_entries: list[dict] = []

    for entry in sorted(entries, key=aihub_filekey_sort_key):
        if not isinstance(entry, dict):
            continue
        filekey = str(entry.get("filekey") or "").strip()
        if not filekey:
            continue

        entry_text = aihub_entry_text(entry)
        matched_source_label = match_source_label_text(entry_text, matchers=label_matchers)
        excluded_source_label = match_source_label_text(entry_text, matchers=excluded_matchers)
        mapped_target_label = str(label_mapping.get(matched_source_label) or "").strip()
        out_of_scope_target_label = ""
        if mapped_target_label and target_label_set and mapped_target_label not in target_label_set:
            out_of_scope_target_label = mapped_target_label
            target_label = ""
        else:
            target_label = mapped_target_label
        source_label = str(entry.get("source_label") or entry.get("sourceLabel") or "").strip()
        source_alias = str(entry.get("source_alias") or entry.get("sourceAlias") or "").strip()
        name = str(
            entry.get("name")
            or entry.get("fileName")
            or entry.get("fileNm")
            or entry.get("filePath")
            or entry.get("path")
            or ""
        ).strip()
        zip_info = classify_aihub_zip_name(name)
        if excluded_source_label or out_of_scope_target_label:
            status = "excluded"
        elif target_label:
            status = "trainable"
        else:
            status = "unknown"

        normalized_entries.append(
            {
                "filekey": filekey,
                "name": name,
                "source_label": source_label,
                "source_alias": source_alias,
                "matched_source_label": matched_source_label,
                "target_label": target_label,
                "excluded_source_label": excluded_source_label,
                "out_of_scope_target_label": out_of_scope_target_label,
                **zip_info,
                "status": status,
                "trainable": status == "trainable",
                "selectable": status in {"trainable", "unknown"},
            }
        )

    return normalized_entries


def build_aihub_lookup_result(
    *,
    datasetkey: str,
    source: str,
    entries: list[dict],
    cache_hit: bool,
    target_labels: list[str] | None = None,
) -> dict:
    summary = {"total": len(entries)}
    for key in AIHUB_FILEKEY_LOOKUP_STATUS_KEYS:
        summary[key] = len([entry for entry in entries if entry.get("status") == key])
    summary["selectable"] = len([entry for entry in entries if entry.get("selectable")])

    return {
        "ok": True,
        "datasetkey": datasetkey,
        "source": source,
        "summary": summary,
        "groups": summarize_aihub_lookup_groups(entries),
        "zip_groups": summarize_aihub_zip_groups(entries),
        "target_labels": [str(label) for label in (target_labels or [])],
        "entries": entries,
        "filekeys": [entry["filekey"] for entry in entries],
        "trainable_filekeys": [entry["filekey"] for entry in entries if entry.get("status") == "trainable"],
        "selectable_filekeys": [entry["filekey"] for entry in entries if entry.get("selectable")],
        "trained_filekeys": [entry["filekey"] for entry in entries if entry.get("status") == "trained"],
        "prepared_filekeys": [entry["filekey"] for entry in entries if entry.get("status") == "prepared"],
        "excluded_filekeys": [entry["filekey"] for entry in entries if entry.get("status") == "excluded"],
        "unknown_filekeys": [entry["filekey"] for entry in entries if entry.get("status") == "unknown"],
        "cache": {
            "hit": bool(cache_hit),
            "ttl_seconds": AIHUB_FILEKEY_LOOKUP_CACHE_SECONDS,
        },
    }


def find_next_trainable_aihub_entry(
    entries: list[dict],
    *,
    existing_filekeys: set[str] | None = None,
    prepared_label_counts: dict[str, int] | None = None,
    prepared_split_label_counts: dict[str, dict[str, int]] | None = None,
    allowed_zip_groups: list[str] | tuple[str, ...] | set[str] | None = None,
    insights: dict | None = None,
    recommendation_policy: dict | None = None,
) -> dict | None:
    blocked = {str(filekey).strip() for filekey in (existing_filekeys or set()) if str(filekey).strip()}
    allowed_groups = normalize_recommendation_zip_groups(allowed_zip_groups)
    policy = normalize_auto_recommendation_policy(recommendation_policy)
    prepared_counts = normalize_label_count_map(prepared_label_counts)
    split_counts = normalize_split_label_counts(prepared_label_counts, prepared_split_label_counts)
    diagnosis_priorities = build_diagnosis_label_priorities(insights)
    primary_groups = policy.get("primary_zip_groups") or allowed_groups
    excluded_groups = policy.get("excluded_zip_groups") or set()
    primary_candidate_labels: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filekey = str(entry.get("filekey") or "").strip()
        if not filekey or filekey in blocked:
            continue
        if entry.get("status") != "trainable" or entry.get("selectable") is False:
            continue
        zip_group = aihub_entry_zip_group(entry)
        if zip_group in excluded_groups:
            continue
        if aihub_entry_matches_zip_groups(entry, primary_groups):
            primary_candidate_labels.add(recommendation_label(entry))

    planned_counts: Counter[str] = Counter()
    trained_counts: Counter[str] = Counter()
    active_zip_counts: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = recommendation_label(entry)
        priority_payload = diagnosis_priority_for_label(label, diagnosis_priorities)
        zip_scope = recommendation_zip_scope_state(
            entry,
            allowed_groups=allowed_groups,
            policy=policy,
            label=label,
            primary_candidate_labels=primary_candidate_labels,
            diagnosis_priority=priority_payload,
            label_counts=Counter(),
            strict_fallback=False,
        )
        if not zip_scope.get("allowed"):
            continue
        zip_group = str(zip_scope.get("zip_group") or entry.get("zip_group") or "기타")
        status = str(entry.get("status") or "unknown")
        if status in {"queued", "running"}:
            planned_counts[label] += 1
            active_zip_counts[zip_group] += 1
        elif status == "trained":
            trained_counts[label] += 1
            active_zip_counts[zip_group] += 1
        elif status == "prepared":
            active_zip_counts[zip_group] += 1

    candidate_labels: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filekey = str(entry.get("filekey") or "").strip()
        if not filekey or filekey in blocked:
            continue
        if entry.get("status") != "trainable" or entry.get("selectable") is False:
            continue
        label = recommendation_label(entry)
        priority_payload = diagnosis_priority_for_label(label, diagnosis_priorities)
        broad_scope = recommendation_zip_scope_state(
            entry,
            allowed_groups=allowed_groups,
            policy=policy,
            label=label,
            primary_candidate_labels=primary_candidate_labels,
            diagnosis_priority=priority_payload,
            label_counts=Counter(),
            strict_fallback=False,
        )
        if broad_scope.get("allowed"):
            candidate_labels.add(label)
    effective_label_counts = recommendation_effective_label_counts(
        prepared_counts=prepared_counts,
        planned_counts=planned_counts,
        candidate_labels=candidate_labels,
    )
    max_class_ratio = float(policy.get("max_class_ratio", 0.0) or 0.0)

    best: tuple[tuple, dict] | None = None
    for order, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        filekey = str(entry.get("filekey") or "").strip()
        if not filekey or filekey in blocked:
            continue
        if entry.get("status") == "trainable" and entry.get("selectable") is not False:
            label = recommendation_label(entry)
            diagnosis_priority = diagnosis_priority_for_label(label, diagnosis_priorities)
            insight_priority = float(diagnosis_priority.get("priority", 0.0) or 0.0)
            zip_scope = recommendation_zip_scope_state(
                entry,
                allowed_groups=allowed_groups,
                policy=policy,
                label=label,
                primary_candidate_labels=primary_candidate_labels,
                diagnosis_priority=diagnosis_priority,
                label_counts=effective_label_counts,
                strict_fallback=True,
            )
            if not zip_scope.get("allowed"):
                continue
            balance_limit = recommendation_balance_limit_state(
                label,
                effective_label_counts,
                max_ratio=max_class_ratio,
            )
            if not balance_limit.get("allowed"):
                continue
            zip_group = str(zip_scope.get("zip_group") or entry.get("zip_group") or "기타")
            planned_count = int(planned_counts[label])
            trained_count = int(trained_counts[label])
            active_zip_count = int(active_zip_counts[zip_group])
            coverage_score = recommendation_coverage_score(
                label,
                prepared_counts=prepared_counts,
                split_counts=split_counts,
                planned_count=planned_count,
            )
            prepared_count = int(coverage_score["prepared_count"])
            score = (
                coverage_score["coverage_rank"],
                planned_count,
                coverage_score["val_balance_bucket"],
                coverage_score["train_balance_bucket"],
                coverage_score["prepared_balance_bucket"],
                -insight_priority,
                prepared_count + planned_count,
                prepared_count,
                trained_count,
                active_zip_count,
                aihub_filekey_sort_key(entry),
                order,
            )
            candidate = {
                **entry,
                "recommendation_reason": "diagnosis_guided" if insight_priority > 0 else "class_balance",
                "recommendation_scope": {
                    "allowed_zip_groups": sorted(allowed_groups),
                    "zip_group": zip_group,
                    "scope_reason": zip_scope.get("scope_reason") or "",
                    "primary_zip_groups": sorted(policy.get("primary_zip_groups") or allowed_groups),
                    "fallback_zip_groups": sorted(policy.get("fallback_zip_groups") or []),
                    "excluded_zip_groups": sorted(policy.get("excluded_zip_groups") or []),
                },
                "recommendation_score": {
                    "target_label": label,
                    "insight_priority": round(insight_priority, 6),
                    "insight_reasons": list(diagnosis_priority.get("reasons") or [])[:4],
                    "insight_level": diagnosis_priority.get("level") or "good",
                    "prepared_count": prepared_count,
                    "planned_count": planned_count,
                    "trained_count": trained_count,
                    "active_zip_count": active_zip_count,
                    "coverage_state": coverage_score["coverage_state"],
                    "train_count": coverage_score["train_count"],
                    "val_count": coverage_score["val_count"],
                    "train_balance_bucket": coverage_score["train_balance_bucket"],
                    "val_balance_bucket": coverage_score["val_balance_bucket"],
                    "prepared_balance_bucket": coverage_score["prepared_balance_bucket"],
                    "balance_max_ratio": balance_limit.get("max_ratio"),
                    "balance_min_count": balance_limit.get("min_count"),
                    "balance_label_count": balance_limit.get("label_count"),
                    "balance_projected_count": balance_limit.get("projected_count"),
                    "balance_limit": balance_limit.get("limit"),
                    "sort_order": order,
                },
            }
            if best is None or score < best[0]:
                best = (score, candidate)
    return best[1] if best is not None else None


def recommendation_label(entry: dict) -> str:
    return str(
        entry.get("target_label")
        or entry.get("matched_source_label")
        or entry.get("source_label")
        or "unknown"
    ).strip() or "unknown"


def summarize_aihub_zip_groups(entries: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {
        label: {
            "label": label,
            "total": 0,
            "selectable": 0,
            "trained": 0,
            "prepared": 0,
            "queued": 0,
            "running": 0,
            "excluded": 0,
            "unknown": 0,
        }
        for label in AIHUB_ZIP_GROUP_ORDER
    }
    for entry in entries:
        label = str(entry.get("zip_group") or "기타")
        group = groups.setdefault(
            label,
            {
                "label": label,
                "total": 0,
                "selectable": 0,
                "trained": 0,
                "prepared": 0,
                "queued": 0,
                "running": 0,
                "excluded": 0,
                "unknown": 0,
            },
        )
        group["total"] += 1
        if entry.get("selectable"):
            group["selectable"] += 1
        status = str(entry.get("status") or "unknown")
        if status in {"trained", "prepared", "queued", "running", "excluded", "unknown"}:
            group[status] += 1

    ordered_labels = [*AIHUB_ZIP_GROUP_ORDER, *sorted(label for label in groups if label not in AIHUB_ZIP_GROUP_ORDER)]
    return [groups[label] for label in ordered_labels if groups[label]["total"] > 0]


def summarize_aihub_lookup_groups(entries: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for entry in entries:
        label = (
            entry.get("source_label")
            or entry.get("source_alias")
            or entry.get("matched_source_label")
            or "미분류"
        )
        group = groups.setdefault(
            str(label),
            {
                "label": str(label),
                "total": 0,
                "trainable": 0,
                "trained": 0,
                "prepared": 0,
                "queued": 0,
                "running": 0,
                "excluded": 0,
                "unknown": 0,
            },
        )
        group["total"] += 1
        status = str(entry.get("status") or "unknown")
        if status in {"trainable", "trained", "prepared", "queued", "running", "excluded", "unknown"}:
            group[status] += 1
    return sorted(groups.values(), key=lambda item: item["label"])


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


def query_gpu_status(*, cache_ttl_seconds: float = 5.0) -> dict:
    now = time.time()
    with GPU_STATUS_CACHE_LOCK:
        cached_timestamp = float(GPU_STATUS_CACHE.get("timestamp") or 0.0)
        cached_value = GPU_STATUS_CACHE.get("value")
        if cached_value is not None and (now - cached_timestamp) <= cache_ttl_seconds:
            return cached_value

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        result = {
            "available": False,
            "summary": "-",
            "detail": "nvidia-smi를 찾지 못했습니다.",
            "device_name": None,
            "utilization_gpu": None,
            "utilization_memory": None,
            "memory_used_mb": None,
            "memory_total_mb": None,
            "memory_percent": None,
            "temperature_c": None,
            "devices": [],
        }
    else:
        command = [
            nvidia_smi,
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=1.5,
                check=True,
            )
            stdout = decode_process_output(completed.stdout)
            devices = []
            for raw_line in stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 7:
                    continue
                gpu_util = _parse_int_or_none(parts[2])
                memory_util = _parse_int_or_none(parts[3])
                memory_used = _parse_float_or_none(parts[4])
                memory_total = _parse_float_or_none(parts[5])
                temperature = _parse_int_or_none(parts[6])
                memory_percent = None
                if memory_used is not None and memory_total not in (None, 0):
                    memory_percent = round((memory_used / memory_total) * 100.0, 1)
                devices.append(
                    {
                        "index": _parse_int_or_none(parts[0]),
                        "name": parts[1],
                        "utilization_gpu": gpu_util,
                        "utilization_memory": memory_util,
                        "memory_used_mb": memory_used,
                        "memory_total_mb": memory_total,
                        "memory_percent": memory_percent,
                        "temperature_c": temperature,
                    }
                )

            primary = None
            if devices:
                primary = max(
                    devices,
                    key=lambda item: (
                        item.get("utilization_gpu") or 0,
                        item.get("memory_percent") or 0.0,
                    ),
                )

            result = {
                "available": bool(primary),
                "summary": _build_gpu_summary(primary),
                "detail": _build_gpu_detail(primary, device_count=len(devices)),
                "device_index": primary.get("index") if primary else None,
                "device_name": primary.get("name") if primary else None,
                "utilization_gpu": primary.get("utilization_gpu") if primary else None,
                "utilization_memory": primary.get("utilization_memory") if primary else None,
                "memory_used_mb": primary.get("memory_used_mb") if primary else None,
                "memory_total_mb": primary.get("memory_total_mb") if primary else None,
                "memory_percent": primary.get("memory_percent") if primary else None,
                "temperature_c": primary.get("temperature_c") if primary else None,
                "devices": devices,
            }
        except (subprocess.SubprocessError, OSError):
            result = {
                "available": False,
                "summary": "-",
                "detail": "GPU 상태를 읽지 못했습니다.",
                "device_index": None,
                "device_name": None,
                "utilization_gpu": None,
                "utilization_memory": None,
                "memory_used_mb": None,
                "memory_total_mb": None,
                "memory_percent": None,
                "temperature_c": None,
                "devices": [],
            }

    with GPU_STATUS_CACHE_LOCK:
        GPU_STATUS_CACHE["timestamp"] = now
        GPU_STATUS_CACHE["value"] = result
    return result


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


def compute_overview_signature(paths: dict, launcher_status: dict | None, *, lite: bool = False) -> tuple:
    launcher_status = launcher_status or {}
    watched = [
        paths["pipeline_status"],
        paths["training_progress"],
        paths["continual_state"],
        paths["current_skip_report"],
        paths["cumulative_skip_report"],
        paths["artifacts_dir"] / "metrics.json",
        paths["artifacts_dir"] / "labels.json",
        paths["raw_manifest"],
        paths["split_train"],
        paths["split_val"],
        paths["split_test"],
        paths["prepared_train"],
        paths["prepared_val"],
        paths["prepared_test"],
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
    for candidate in (
        current_job.get("log_path") if isinstance(current_job, dict) else None,
        latest_completed_job.get("log_path") if isinstance(latest_completed_job, dict) else None,
        latest_error_job.get("log_path") if isinstance(latest_error_job, dict) else None,
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
        tuple(dynamic_log_paths),
        launcher_status.get("auto_start_enabled"),
        launcher_status.get("auto_enqueue_enabled"),
        launcher_status.get("auto_enqueue_datasetkey"),
    )
    return (tuple(file_signature), launcher_signature, lite)


def _parse_int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"n/a", "[not supported]"}:
        return None
    try:
        return int(float(normalized))
    except ValueError:
        return None


def _parse_float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"n/a", "[not supported]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _format_gib_from_mb(value_mb: float | None) -> str:
    if value_mb is None:
        return "-"
    return f"{value_mb / 1024.0:.1f} GB"


def _build_gpu_summary(primary: dict | None) -> str:
    if not primary:
        return "-"
    gpu_util = primary.get("utilization_gpu")
    if gpu_util is None:
        return "-"
    return f"{gpu_util}%"


def _build_gpu_detail(primary: dict | None, *, device_count: int) -> str:
    if not primary:
        return "GPU 상태를 읽지 못했습니다."
    memory_used = _format_gib_from_mb(primary.get("memory_used_mb"))
    memory_total = _format_gib_from_mb(primary.get("memory_total_mb"))
    memory_percent = primary.get("memory_percent")
    temp = primary.get("temperature_c")
    util_mem = primary.get("utilization_memory")
    index = primary.get("index")
    prefix = f"GPU {index}" if index is not None else "GPU"
    suffix_parts = [f"VRAM {memory_used} / {memory_total}"]
    if memory_percent is not None:
        suffix_parts.append(f"{memory_percent:.1f}%")
    if util_mem is not None:
        suffix_parts.append(f"mem {util_mem}%")
    if temp is not None:
        suffix_parts.append(f"{temp}°C")
    if device_count > 1:
        suffix_parts.append(f"{device_count} GPUs")
    return f"{prefix} | " + " | ".join(suffix_parts)


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
        "last_state": "idle",
        "last_exit_code": None,
        "last_message": "아직 실행 기록이 없습니다.",
        "log_path": None,
        "pages_sync_warning": None,
    }

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
        message = f"[pages-sync] {action} failed: {exc}"
        print(message, file=sys.stderr)
        launcher_state["pages_sync_warning"] = str(exc)

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
                normalized_completed_jobs = sort_jobs_by_recency(completed_jobs)
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
        if isinstance(completed_jobs, list) and completed_jobs:
            normalized_completed_jobs = sort_jobs_by_recency(completed_jobs)
            launcher_state["completed_jobs"] = normalized_completed_jobs
            if normalized_completed_jobs != completed_jobs:
                persist_launcher_history(launcher_history_path, launcher_state)
            apply_restored_launcher_summary(launcher_state["completed_jobs"])
            return launcher_state["completed_jobs"]
        restored_jobs = restore_completed_jobs_from_logs(job_logs_dir, runtime_config_dir)
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

    def hard_reset_workspace() -> None:
        for key in (
            "raw_dir",
            "import_dir",
            "extracted_dir",
            "manifests_dir",
            "prepared_dir",
            "artifacts_dir",
        ):
            target = paths.get(key)
            if isinstance(target, Path) and target.exists():
                shutil.rmtree(target)

        for target in (job_logs_dir, runtime_config_dir):
            if target.exists():
                shutil.rmtree(target)

        if launcher_history_path.exists():
            launcher_history_path.unlink()

        for key in (
            "workspace_dir",
            "raw_dir",
            "import_dir",
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

    def start_pipeline_for_job(job: dict) -> None:
        reset_training_workspace(paths)
        runtime_config_dir.mkdir(parents=True, exist_ok=True)
        job_logs_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = json.loads(json.dumps(config))
        runtime_config["dataset_source"] = "aihub_shell"
        runtime_paths = runtime_config.setdefault("paths", {})
        runtime_paths["workspace_dir"] = str(paths["workspace_dir"])
        runtime_shell = runtime_config.setdefault("aihub_shell", {})
        if job.get("datasetkey") not in (None, ""):
            runtime_shell["datasetkey"] = job["datasetkey"]
        runtime_shell["filekey"] = job["filekey"]
        if job.get("api_key"):
            runtime_shell["api_key"] = str(job["api_key"])
            runtime_shell["api_key_env"] = ""

        runtime_config_path = runtime_config_dir / f"{job['job_id']}.json"
        write_json_atomic(runtime_config_path, runtime_config)

        log_path = job_logs_dir / f"{job['job_id']}.log"
        startup_message = (
            f"[launcher] datasetkey {job.get('datasetkey', '-')}"
            f" | filekey {job['filekey']} 작업을 시작합니다."
        )
        write_dashboard_status(
            paths,
            stage="queued",
            state="running",
            message=startup_message.replace("[launcher] ", ""),
            current_filekey=job["filekey"],
            current_datasetkey=job.get("datasetkey"),
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"{startup_message}\n")
            log_handle.write(f"[launcher] runtime config: {runtime_config_path}\n")
            log_handle.flush()
            child_env = os.environ.copy()
            child_env["PYTHONUTF8"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    str(pipeline_script),
                    "--config",
                    str(runtime_config_path),
                    "--stage",
                    str(job.get("stage") or "all"),
                ],
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
        safe_sync_pages_live("online", action="job start live status")

    def update_process_state() -> None:
        process = launcher_state.get("process")
        if process is not None and isinstance(process, subprocess.Popen):
            exit_code = process.poll()
            if exit_code is None:
                launcher_state["last_state"] = "running"
                current_job = launcher_state.get("current_job") or {}
                filekey = current_job.get("filekey", "-") if isinstance(current_job, dict) else "-"
                datasetkey = current_job.get("datasetkey", "-") if isinstance(current_job, dict) else "-"
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
                    final_state, final_message = classify_job_exit(paths, current_job, exit_code)
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

        active_process = launcher_state.get("process")
        queued_jobs = launcher_state.get("queued_jobs", [])
        if (
            active_process is None
            and bool(launcher_state.get("auto_start_enabled", True))
            and bool(launcher_state.get("auto_extract_enabled", False))
            and isinstance(queued_jobs, list)
            and not queued_jobs
        ):
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
            "runtime_config_path": str(launcher_state["runtime_config_path"])
            if launcher_state.get("runtime_config_path")
            else None,
            "current_job": current_job,
            "pending_jobs": [snapshot_job(job) for job in queued_jobs] if isinstance(queued_jobs, list) else [],
            "completed_jobs": [snapshot_job(job) for job in completed_jobs] if isinstance(completed_jobs, list) else [],
            "last_exit_code": launcher_state.get("last_exit_code"),
            "log_path": str(launcher_state["log_path"]) if launcher_state.get("log_path") else None,
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

    worker = threading.Thread(target=queue_worker, daemon=True)
    worker.start()

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
        dataset = overview.get("dataset") or {}
        latest = progress.get("latest") or {}
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
        best_f1_text = f"{float(best_f1):.3f}" if best_f1 is not None else "-"
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
            f"<div class=\"hero-chip\"><strong>Best F1</strong> {best_f1_text} @ epoch {escape(str(best_epoch))}</div>"
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
        final_validation = progress.get("final_validation") or metrics.get("final_validation") or {}
        labels = progress.get("labels") or metrics.get("labels") or []
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
            <div class="panel-copy">split과 클래스 분포</div>
          </div>
        </div>
        <div class="panel-body">
          <table class="table">
            <thead>
              <tr>
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
      return `${job.filekey} (${job.state || 'unknown'})`;
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
      const resetButton = document.getElementById('resetButton');
      if (!startButton || !stopButton || !forceStopButton || !resetButton) {
        return;
      }
      if (viewerMode) {
        startButton.disabled = true;
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

    function renderDatasetTable(dataset) {
      const tbody = document.getElementById('datasetTable');
      const empty = document.getElementById('datasetEmpty');
      const rows = [];
      const sections = ['raw', 'train', 'val', 'test', 'prepared_train', 'prepared_val', 'prepared_test'];

      sections.forEach((key) => {
        const info = dataset[key];
        if (!info || !info.total) {
          return;
        }
        const labels = Object.entries(info.by_label || {})
          .map(([label, count]) => `${escapeHtml(label)} ${count}`)
          .join(' / ');
        rows.push(`
          <tr>
            <td>${escapeHtml(key)}</td>
            <td>${info.total}</td>
            <td>${labels || '-'}</td>
          </tr>
        `);
      });

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
      const finalValidation = progress?.final_validation || metrics?.final_validation || {};
      const history = progress?.history || [];
      const latest = progress?.latest || history[history.length - 1] || null;
      const labels = progress?.labels || metrics?.labels || [];
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

    function renderPerClassMetrics(labels, perClass, confusion) {
      const tbody = document.getElementById('perClassMetricsTable');
      const empty = document.getElementById('perClassMetricsEmpty');
      if (!perClass || !perClass.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
      }
      empty.style.display = 'none';
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
        await refresh();
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        button.disabled = false;
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
        await refresh();
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
        await refresh();
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
        await refresh();
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
        await refresh();
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

    async function refresh() {
      let data;
      if (pendingInitialOverview) {
        data = pendingInitialOverview;
        pendingInitialOverview = null;
      } else {
        try {
          const response = await fetch('/api/overview', { cache: 'no-store' });
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
          data = await response.json();
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
        renderMetricInsights(progress, metrics);
        renderDatasetTable(data.dataset || {});
        renderCurrentJobProgress(currentJobProgress, data.current_dataset || {}, continualState, progress);
      });

      runRenderStep('validation metrics', renderErrors, () => {
        const metricLabels = progress.labels || metrics.labels || [];
        const finalValidation = progress.final_validation || metrics.final_validation || {};
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
    refresh();
    setInterval(refresh, 1000);
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
        return filekeys

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
            training_progress.get("final_validation")
            or metrics.get("final_validation")
            or {}
        )
        insight_metrics = {
            **(metrics if isinstance(metrics, dict) else {}),
            "labels": target_labels,
            "history": history if isinstance(history, list) else [],
            "latest": latest if isinstance(latest, dict) else {},
            "final_validation": final_validation if isinstance(final_validation, dict) else {},
            "best_epoch": training_progress.get("best_epoch") or metrics.get("best_epoch"),
            "best_val_macro_f1": training_progress.get("best_val_macro_f1") or metrics.get("best_val_macro_f1"),
            "best_validation": training_progress.get("best_validation") or metrics.get("best_validation") or {},
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

    def enable_auto_extract_locked(datasetkey: str, api_key: str) -> None:
        launcher_state["auto_extract_enabled"] = True
        launcher_state["auto_enqueue_enabled"] = False
        launcher_state["auto_enqueue_datasetkey"] = str(datasetkey or "").strip()
        launcher_state["auto_enqueue_api_key"] = str(api_key or "").strip()
        launcher_state["auto_start_enabled"] = True

    def disable_auto_enqueue_locked(message: str) -> None:
        launcher_state["auto_enqueue_enabled"] = False
        launcher_state["auto_extract_enabled"] = False
        launcher_state["auto_enqueue_datasetkey"] = None
        launcher_state["auto_enqueue_api_key"] = ""
        launcher_state["last_message"] = message

    def enqueue_next_extract_recommended_job_locked() -> dict:
        datasetkey = str(launcher_state.get("auto_enqueue_datasetkey") or config.get("aihub_shell", {}).get("datasetkey") or "").strip()
        api_key = str(launcher_state.get("auto_enqueue_api_key") or "").strip()
        if not datasetkey:
            message = "자동 추출을 위한 datasetkey가 없습니다."
            disable_auto_enqueue_locked(message)
            return {"ok": False, "message": message, "job": None}
        lookup = build_aihub_filekey_lookup_for_dashboard(datasetkey=datasetkey, api_key=api_key, refresh_process=False)
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
            disable_auto_enqueue_locked(message)
            return {"ok": False, "message": message, "job": None}
        filekey = str(recommended["filekey"])
        job = build_job(filekey, datasetkey=datasetkey, api_key=api_key, stage="extract")
        job["auto_recommended"] = True
        job["target_label"] = recommended.get("target_label")
        job["recommendation_reason"] = recommended.get("recommendation_reason") or "class_balance"
        launcher_state.setdefault("queued_jobs", []).append(job)
        launcher_state["last_state"] = "queued"
        launcher_state["last_message"] = f"분포 기준 자동 추출 filekey {filekey}를 큐에 추가했습니다."
        return {"ok": True, "message": launcher_state["last_message"], "job": job}

    def enqueue_next_recommended_job_locked() -> dict:
        datasetkey = str(
            launcher_state.get("auto_enqueue_datasetkey")
            or config.get("aihub_shell", {}).get("datasetkey")
            or ""
        ).strip()
        api_key = str(launcher_state.get("auto_enqueue_api_key") or "").strip()
        if not paths["prepared_train"].exists() or not paths["prepared_val"].exists():
            message = "추출된 prepared 데이터가 없어 학습을 시작할 수 없습니다."
            disable_auto_enqueue_locked(message)
            launcher_state["last_state"] = launcher_state.get("last_state") or "completed"
            return {"ok": False, "message": message, "job": None}
        pending_jobs = launcher_state.setdefault("queued_jobs", [])
        job = build_job("prepared_pose", datasetkey=datasetkey or "prepared", api_key=api_key, stage="train")
        job["auto_recommended"] = True
        job["recommendation_reason"] = "prepared_data_training"
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
        requested_stage = str(payload.get("stage") or "all").strip().lower()
        if requested_stage not in {"all", "extract"}:
            requested_stage = "all"
        if auto_enqueue_next:
            filekeys = []
            requested_stage = "train"
        if auto_extract_next:
            filekeys = []
            requested_stage = "extract"
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
                enable_auto_extract_locked(datasetkey, api_key)

            if filekeys:
                for filekey in filekeys:
                    unique_key = f"{queue_key(datasetkey, filekey)}:{requested_stage}"
                    if unique_key in existing_keys:
                        skipped.append(filekey)
                        continue
                    job = build_job(filekey, datasetkey=datasetkey, api_key=api_key, stage=requested_stage)
                    if isinstance(pending_jobs, list):
                        pending_jobs.append(job)
                    appended.append(filekey)
                    existing_keys.add(unique_key)
            elif auto_extract_next:
                recommendation = enqueue_next_extract_recommended_job_locked()
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
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            if isinstance(pending_jobs, list):
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

    @app.post("/api/pause")
    def pause_after_current(request: Request) -> dict:
        ensure_dashboard_control_access(request)
        return pause_after_current_request()

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
    dataset_summary = (
        build_manifest_summary_group(paths, OVERVIEW_DATASET_MANIFEST_SPECS)
        if not lite
        else {}
    )
    current_dataset_summary = build_manifest_summary_group(paths, OVERVIEW_CURRENT_DATASET_MANIFEST_SPECS)
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
        training_progress.get("final_validation")
        or metrics.get("final_validation")
        or {}
    )
    insight_metrics = {
        **(metrics if isinstance(metrics, dict) else {}),
        "labels": target_labels,
        "history": metric_history if isinstance(metric_history, list) else [],
        "latest": latest_metrics if isinstance(latest_metrics, dict) else {},
        "final_validation": final_validation if isinstance(final_validation, dict) else {},
        "best_epoch": training_progress.get("best_epoch") or metrics.get("best_epoch"),
        "best_val_macro_f1": training_progress.get("best_val_macro_f1") or metrics.get("best_val_macro_f1"),
        "best_validation": training_progress.get("best_validation") or metrics.get("best_validation") or {},
    }
    insights = interpret_training_results(
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
        } if not lite else {},
        "dataset": dataset_summary,
        "current_dataset": current_dataset_summary,
        "prepared_pose_items": build_prepared_pose_items(paths, limit=100) if not lite else [],
        "continual_state": read_json(paths["continual_state"]),
        "artifacts": {
            "has_model": (paths["artifacts_dir"] / "best_action_model.pt").exists(),
            "has_metrics": (paths["artifacts_dir"] / "metrics.json").exists(),
            "has_labels": (paths["artifacts_dir"] / "labels.json").exists(),
        },
        "diagnostics": diagnostics,
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


def reset_training_workspace(paths: dict) -> None:
    transient_dirs = ("raw_dir", "import_dir", "extracted_dir")
    transient_files = (
        "current_raw_manifest",
        "current_split_train",
        "current_split_val",
        "current_split_test",
        "current_prepared_train",
        "current_prepared_val",
        "current_prepared_test",
        "current_skip_report",
    )

    for key in transient_dirs:
        target = paths.get(key)
        if not isinstance(target, Path):
            continue
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    for key in transient_files:
        target = paths.get(key)
        if not isinstance(target, Path):
            continue
        if target.exists():
            target.unlink()

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

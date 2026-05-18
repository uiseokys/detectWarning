from __future__ import annotations

import os
import re
from collections import Counter

from action_training_pipeline import build_source_label_matchers, match_source_label_text
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
    "violence": "always",
    "collapse": "always",
    "loitering": "always",
}
AIHUB_AUTO_RECOMMEND_BALANCE_LABELS = {"violence", "collapse", "loitering"}
AIHUB_AUTO_RECOMMEND_EXCLUDED_LABELS = {"abduction"}
AIHUB_AUTO_RECOMMEND_ANCHOR_LABEL = ""
AIHUB_AUTO_RECOMMEND_FULL_USE_LABELS: set[str] = set()
AIHUB_AUTO_RECOMMEND_DEFERRED_LABELS: set[str] = set()
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
        "balance_labels": sorted(AIHUB_AUTO_RECOMMEND_BALANCE_LABELS),
        "excluded_labels": sorted(AIHUB_AUTO_RECOMMEND_EXCLUDED_LABELS),
        "anchor_label": AIHUB_AUTO_RECOMMEND_ANCHOR_LABEL,
        "full_use_labels": sorted(AIHUB_AUTO_RECOMMEND_FULL_USE_LABELS),
        "deferred_labels": sorted(AIHUB_AUTO_RECOMMEND_DEFERRED_LABELS),
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
        "balance_labels": {
            normalize_recommendation_label(label)
            for label in (policy.get("balance_labels") or [])
            if normalize_recommendation_label(label)
        },
        "excluded_labels": {
            normalize_recommendation_label(label)
            for label in (policy.get("excluded_labels") or [])
            if normalize_recommendation_label(label)
        },
        "anchor_label": normalize_recommendation_label(policy.get("anchor_label")),
        "full_use_labels": {
            normalize_recommendation_label(label)
            for label in (policy.get("full_use_labels") or [])
            if normalize_recommendation_label(label)
        },
        "deferred_labels": {
            normalize_recommendation_label(label)
            for label in (policy.get("deferred_labels") or [])
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
    balance_labels: set[str] | None = None,
) -> Counter[str]:
    normalized_balance_labels = {
        normalize_recommendation_label(label)
        for label in (balance_labels or set())
        if normalize_recommendation_label(label)
    }
    labels = {
        str(label or "").strip()
        for label in (*prepared_counts.keys(), *planned_counts.keys(), *candidate_labels)
        if str(label or "").strip()
    }
    if normalized_balance_labels:
        labels = {
            label
            for label in labels
            if normalize_recommendation_label(label) in normalized_balance_labels
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
    anchor_label: str = "",
    full_use_labels: set[str] | None = None,
) -> dict:
    normalized_label = str(label or "").strip()
    normalized_anchor = normalize_recommendation_label(anchor_label)
    full_use_labels = full_use_labels or set()
    if not normalized_label or max_ratio <= 0 or len(label_counts) < 2:
        return {
            "allowed": True,
            "anchor_label": normalized_anchor,
            "anchor_count": int(label_counts.get(normalized_anchor, 0) or 0) if normalized_anchor else 0,
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
    anchor_count = int(label_counts.get(normalized_anchor, 0) or 0) if normalized_anchor else 0
    if normalized_label in full_use_labels:
        allowed = True
        limit = None
        reason = ""
    elif normalized_anchor:
        limit = float(anchor_count) * float(max_ratio)
        allowed = projected_count <= limit
        reason = "" if allowed else "anchor_class_ratio_limit"
    elif label_count <= min_count:
        allowed = True
        limit = float(min_count)
        reason = ""
    elif projected_count <= min_count:
        allowed = True
        limit = float(min_count)
        reason = ""
    elif min_count <= 0:
        allowed = False
        limit = 0.0
        reason = "class_ratio_limit"
    else:
        limit = float(min_count) * float(max_ratio)
        allowed = projected_count <= limit
        reason = "" if allowed else "class_ratio_limit"
    return {
        "allowed": allowed,
        "anchor_label": normalized_anchor,
        "anchor_count": anchor_count,
        "max_ratio": max_ratio,
        "min_count": min_count,
        "label_count": label_count,
        "projected_count": projected_count,
        "limit": round(limit, 6) if limit is not None else None,
        "reason": reason,
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
    fallback_mode = str(fallback_labels.get(normalized_label) or fallback_labels.get("*") or "").strip().lower()
    if not fallback_mode:
        return {"allowed": False, "zip_group": zip_group, "scope_reason": "fallback_label_not_allowed"}
    if normalized_label in {normalize_recommendation_label(item) for item in primary_candidate_labels}:
        return {"allowed": False, "zip_group": zip_group, "scope_reason": "primary_candidate_available"}
    if not strict_fallback:
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "fallback_zip_group"}
    if fallback_mode == "diagnosed" and insight_priority > 0:
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "diagnosed_fallback"}
    if fallback_mode == "always":
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "always_fallback"}
    if fallback_mode == "minority" and minority_state.get("is_minority"):
        return {"allowed": True, "zip_group": zip_group, "scope_reason": "minority_fallback"}
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
    excluded_labels = policy.get("excluded_labels") if isinstance(policy.get("excluded_labels"), set) else set()
    balance_labels = policy.get("balance_labels") if isinstance(policy.get("balance_labels"), set) else set()
    primary_candidate_labels: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        filekey = str(entry.get("filekey") or "").strip()
        if not filekey or filekey in blocked:
            continue
        if entry.get("status") != "trainable" or entry.get("selectable") is False:
            continue
        entry_label = recommendation_label(entry)
        if normalize_recommendation_label(entry_label) in excluded_labels:
            continue
        zip_group = aihub_entry_zip_group(entry)
        if zip_group in excluded_groups:
            continue
        if aihub_entry_matches_zip_groups(entry, primary_groups):
            primary_candidate_labels.add(entry_label)

    planned_counts: Counter[str] = Counter()
    trained_counts: Counter[str] = Counter()
    active_zip_counts: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = recommendation_label(entry)
        if normalize_recommendation_label(label) in excluded_labels:
            continue
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
        if normalize_recommendation_label(label) in excluded_labels:
            continue
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
        balance_labels=balance_labels,
    )
    max_class_ratio = float(policy.get("max_class_ratio", 0.0) or 0.0)
    anchor_label = str(policy.get("anchor_label") or "").strip()
    full_use_labels = policy.get("full_use_labels") if isinstance(policy.get("full_use_labels"), set) else set()
    deferred_labels = policy.get("deferred_labels") if isinstance(policy.get("deferred_labels"), set) else set()
    has_non_deferred_candidate = any(
        normalize_recommendation_label(label) not in deferred_labels
        for label in candidate_labels
    )

    best: tuple[tuple, dict] | None = None
    for order, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        filekey = str(entry.get("filekey") or "").strip()
        if not filekey or filekey in blocked:
            continue
        if entry.get("status") == "trainable" and entry.get("selectable") is not False:
            label = recommendation_label(entry)
            normalized_label = normalize_recommendation_label(label)
            if normalized_label in excluded_labels:
                continue
            if normalized_label in deferred_labels and has_non_deferred_candidate:
                continue
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
                anchor_label=anchor_label,
                full_use_labels=full_use_labels,
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
                    "balance_anchor_label": balance_limit.get("anchor_label"),
                    "balance_anchor_count": balance_limit.get("anchor_count"),
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


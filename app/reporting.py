from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


STATE_SCHEMA_VERSION = 1
MANIFEST_SUMMARY_CACHE: dict[tuple[str, str], dict] = {}
MANIFEST_SUMMARY_CACHE_LOCK = threading.Lock()
DEFAULT_IMBALANCE_WARN_MIN_SAMPLES = 8
DEFAULT_IMBALANCE_WARN_RATIO = 5.0


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def summarize_manifest_total(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def summarize_manifest(path: Path, label_field: str) -> dict:
    if not path.exists():
        return {"total": 0, "by_label": {}}

    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = None

    cache_key = (str(path), label_field)
    if signature is not None:
        with MANIFEST_SUMMARY_CACHE_LOCK:
            cached = MANIFEST_SUMMARY_CACHE.get(cache_key)
            if cached and cached.get("signature") == signature:
                return dict(cached["value"])

    total = 0
    by_label: Counter[str] = Counter()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                label = str(payload.get(label_field, "unknown"))
                by_label[label] += 1
    except OSError:
        return {"total": 0, "by_label": {}}
    summary = {"total": total, "by_label": dict(sorted(by_label.items()))}
    if signature is not None:
        with MANIFEST_SUMMARY_CACHE_LOCK:
            MANIFEST_SUMMARY_CACHE[cache_key] = {
                "signature": signature,
                "value": summary,
            }
    return dict(summary)


def build_label_counts(labels: list[str], by_label: dict | None) -> list[dict]:
    counts = by_label or {}
    normalized_labels = normalize_label_list(labels)
    rows: list[dict] = []
    for label in normalized_labels:
        rows.append(
            {
                "label": label,
                "count": int(counts.get(label, 0) or 0),
            }
        )
    return rows


def analyze_class_balance(
    labels: list[str],
    by_label: dict | None,
    *,
    min_samples: int = DEFAULT_IMBALANCE_WARN_MIN_SAMPLES,
    ratio_warn: float = DEFAULT_IMBALANCE_WARN_RATIO,
) -> dict:
    rows = build_label_counts(labels, by_label)
    nonzero_rows = [row for row in rows if int(row.get("count", 0) or 0) > 0]
    empty_labels = [row["label"] for row in rows if int(row.get("count", 0) or 0) <= 0]
    low_sample_rows = [
        {"label": row["label"], "count": int(row.get("count", 0) or 0)}
        for row in nonzero_rows
        if int(row.get("count", 0) or 0) < max(int(min_samples), 1)
    ]

    dominant = max(nonzero_rows, key=lambda row: int(row.get("count", 0) or 0), default=None)
    minority = min(nonzero_rows, key=lambda row: int(row.get("count", 0) or 0), default=None)
    dominant_count = int(dominant.get("count", 0) or 0) if dominant else 0
    minority_count = int(minority.get("count", 0) or 0) if minority else 0
    imbalance_ratio = (
        round(dominant_count / minority_count, 3)
        if dominant_count > 0 and minority_count > 0
        else None
    )

    messages: list[str] = []
    severity = "ok"
    if empty_labels:
        severity = "critical"
        messages.append(f"비어 있는 클래스: {', '.join(empty_labels)}")
    if low_sample_rows:
        severity = "warning" if severity == "ok" else severity
        summary = ", ".join(f"{row['label']} {row['count']}" for row in low_sample_rows[:5])
        if len(low_sample_rows) > 5:
            summary = f"{summary} 외 {len(low_sample_rows) - 5}개"
        messages.append(f"샘플이 적은 클래스: {summary}")
    if imbalance_ratio is not None and imbalance_ratio >= float(ratio_warn):
        severity = "warning" if severity == "ok" else severity
        messages.append(
            "클래스 편중이 큽니다: "
            f"{dominant.get('label') if dominant else '-'} {dominant_count} / "
            f"{minority.get('label') if minority else '-'} {minority_count}"
        )

    if not messages:
        messages.append("클래스 분포가 크게 치우치지 않았습니다.")

    return {
        "severity": severity,
        "counts": rows,
        "covered": len(nonzero_rows),
        "total": len(rows),
        "empty_labels": empty_labels,
        "low_sample_labels": low_sample_rows,
        "dominant_label": dominant.get("label") if dominant else None,
        "dominant_count": dominant_count,
        "minority_label": minority.get("label") if minority else None,
        "minority_count": minority_count,
        "imbalance_ratio": imbalance_ratio,
        "messages": messages,
        "min_samples": int(min_samples),
        "ratio_warn": float(ratio_warn),
    }


def summarize_stage_timings(stage_timings: dict | None) -> dict:
    timings = stage_timings if isinstance(stage_timings, dict) else {}
    ordered = []
    for key in ("download", "prepare", "train", "total"):
        info = timings.get(key) if isinstance(timings.get(key), dict) else {}
        ordered.append(
            {
                "stage": key,
                "started_at": info.get("started_at"),
                "finished_at": info.get("finished_at"),
                "duration_seconds": info.get("duration_seconds"),
            }
        )
    return {
        "ordered": ordered,
        "by_stage": {row["stage"]: row for row in ordered},
    }


def normalize_label_list(labels) -> list[str]:
    normalized: list[str] = []
    for label in labels or []:
        value = str(label).strip()
        if value:
            normalized.append(value)
    return normalized


def remap_per_class_rows(rows: list, *, source_labels: list[str], target_labels: list[str]) -> list[dict]:
    labels = normalize_label_list(target_labels) or normalize_label_list(source_labels)
    if not labels:
        return []

    label_to_index = {label: index for index, label in enumerate(labels)}
    remapped: dict[str, dict] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        source_index = int(row.get("class_index", -1) or -1)
        source_label = (
            source_labels[source_index]
            if 0 <= source_index < len(source_labels)
            else str(row.get("label") or "").strip()
        )
        if source_label not in label_to_index:
            continue
        remapped[source_label] = {
            **row,
            "class_index": label_to_index[source_label],
            "label": source_label,
        }

    normalized_rows: list[dict] = []
    for class_index, label in enumerate(labels):
        existing = remapped.get(label)
        if existing is None:
            normalized_rows.append(
                {
                    "class_index": class_index,
                    "label": label,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                }
            )
        else:
            normalized_rows.append(existing)
    return normalized_rows


def remap_confusion_matrix(confusion_matrix: list, *, source_labels: list[str], target_labels: list[str]) -> list[list[int]]:
    labels = normalize_label_list(target_labels) or normalize_label_list(source_labels)
    if not labels:
        return []

    index_map = {label: index for index, label in enumerate(labels)}
    normalized = [[0 for _ in labels] for _ in labels]
    matrix = confusion_matrix if isinstance(confusion_matrix, list) else []
    for source_row_index, row in enumerate(matrix):
        if not isinstance(row, list) or source_row_index >= len(source_labels):
            continue
        true_label = source_labels[source_row_index]
        true_target_index = index_map.get(true_label)
        if true_target_index is None:
            continue
        for source_col_index, value in enumerate(row):
            if source_col_index >= len(source_labels):
                continue
            pred_label = source_labels[source_col_index]
            pred_target_index = index_map.get(pred_label)
            if pred_target_index is None:
                continue
            normalized[true_target_index][pred_target_index] += int(value or 0)
    return normalized


def normalize_metric_payload(payload: dict | None, *, target_labels: list[str]) -> dict:
    if not isinstance(payload, dict):
        return {}

    normalized = dict(payload)
    source_labels = normalize_label_list(payload.get("labels") or [])
    display_labels = normalize_label_list(target_labels) or source_labels
    final_validation = dict(payload.get("final_validation") or {})
    final_validation["per_class"] = remap_per_class_rows(
        final_validation.get("per_class") or [],
        source_labels=source_labels,
        target_labels=display_labels,
    )
    final_validation["confusion_matrix"] = remap_confusion_matrix(
        final_validation.get("confusion_matrix") or [],
        source_labels=source_labels,
        target_labels=display_labels,
    )
    normalized["source_labels"] = source_labels
    normalized["labels"] = display_labels
    normalized["final_validation"] = final_validation
    return normalized


def build_per_class_support(labels: list[str], confusion_matrix: list) -> list[dict]:
    supports: list[dict] = []
    matrix = confusion_matrix if isinstance(confusion_matrix, list) else []
    for index, label in enumerate(labels):
        row = matrix[index] if index < len(matrix) and isinstance(matrix[index], list) else []
        support = sum(int(value or 0) for value in row)
        supports.append(
            {
                "class_index": index,
                "label": label,
                "support": support,
            }
        )
    return supports


def build_recent_jobs(launcher_history: dict, limit: int = 5) -> list[dict]:
    completed_jobs = launcher_history.get("completed_jobs") or []
    if not isinstance(completed_jobs, list):
        return []

    recent: list[dict] = []
    for job in completed_jobs[:limit]:
        if not isinstance(job, dict):
            continue
        recent.append(
            {
                "datasetkey": job.get("datasetkey"),
                "filekey": job.get("filekey"),
                "state": job.get("state"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "duration_minutes": job.get("duration_minutes"),
                "message": job.get("message"),
            }
        )
    return recent


def get_latest_job(paths: dict) -> dict:
    launcher_history = read_json(paths["workspace_dir"] / "launcher_history.json") or {}
    completed_jobs = launcher_history.get("completed_jobs") or []
    if not isinstance(completed_jobs, list) or not completed_jobs:
        pipeline_status = read_json(paths["pipeline_status"]) or {}
        metrics_path = paths["artifacts_dir"] / "metrics.json"
        model_path = paths["artifacts_dir"] / "best_action_model.pt"
        has_recent_artifacts = metrics_path.exists() or model_path.exists()
        fallback_finished_at = None
        if metrics_path.exists():
            fallback_finished_at = datetime.fromtimestamp(
                metrics_path.stat().st_mtime,
                tz=timezone.utc,
            ).astimezone().isoformat()
        elif model_path.exists():
            fallback_finished_at = datetime.fromtimestamp(
                model_path.stat().st_mtime,
                tz=timezone.utc,
            ).astimezone().isoformat()

        if has_recent_artifacts:
            return {
                "datasetkey": None,
                "filekey": None,
                "state": "completed",
                "started_at": None,
                "finished_at": fallback_finished_at or pipeline_status.get("updated_at"),
                "duration_minutes": None,
                "message": pipeline_status.get("message") or "최근 학습 결과를 불러왔습니다.",
            }
        return {
            "datasetkey": None,
            "filekey": None,
            "state": pipeline_status.get("state") or "offline",
            "started_at": None,
            "finished_at": pipeline_status.get("updated_at"),
            "duration_minutes": None,
            "message": pipeline_status.get("message") or "최근 학습 기록이 아직 없습니다.",
        }

    def sort_key(job: dict) -> str:
        return str(job.get("finished_at") or job.get("started_at") or "")

    latest_job = max(completed_jobs, key=sort_key)
    return {
        "datasetkey": latest_job.get("datasetkey"),
        "filekey": latest_job.get("filekey"),
        "state": latest_job.get("state"),
        "started_at": latest_job.get("started_at"),
        "finished_at": latest_job.get("finished_at"),
        "duration_minutes": latest_job.get("duration_minutes"),
        "message": latest_job.get("message"),
    }


def get_best_macro_f1(history: list[dict]) -> float | None:
    if not history:
        return None
    values = [safe_number(item.get("val_macro_f1")) for item in history]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def safe_number(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

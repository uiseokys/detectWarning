from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


STATE_SCHEMA_VERSION = 1
MANIFEST_SUMMARY_CACHE: dict[tuple[str, str], dict] = {}
MANIFEST_SUMMARY_CACHE_LOCK = threading.Lock()
MANIFEST_SUMMARY_CACHE_MAX_ENTRIES = 64
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


def parse_iso_datetime(value) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def build_path_diagnostic(path: Path) -> dict:
    payload = {
        "path": str(path),
        "exists": False,
        "size": None,
        "updated_at": None,
    }
    try:
        if not path.exists():
            return payload
        stat = path.stat()
    except OSError:
        return payload
    payload["exists"] = True
    payload["size"] = int(stat.st_size)
    payload["updated_at"] = datetime.fromtimestamp(
        stat.st_mtime,
        tz=timezone.utc,
    ).astimezone().isoformat()
    return payload


def compute_job_duration_minutes(job: dict | None) -> float | None:
    if not isinstance(job, dict):
        return None
    existing = safe_number(job.get("duration_minutes"))
    if existing is not None:
        return round(existing, 2)
    started_at = parse_iso_datetime(job.get("started_at"))
    finished_at = parse_iso_datetime(job.get("finished_at"))
    if started_at is None or finished_at is None:
        return None
    seconds = max(0.0, (finished_at - started_at).total_seconds())
    return round(seconds / 60.0, 2)


def infer_job_message(job: dict | None) -> str:
    if not isinstance(job, dict):
        return "최근 작업 정보가 없습니다."
    state = str(job.get("state") or "").strip().lower()
    filekey = str(job.get("filekey") or "-")
    datasetkey = str(job.get("datasetkey") or "-")
    exit_code = job.get("exit_code")
    if state == "completed":
        return f"datasetkey {datasetkey} | filekey {filekey} 학습이 완료되었습니다."
    if state == "completed_warning":
        return (
            f"datasetkey {datasetkey} | filekey {filekey} 학습 결과는 생성됐지만 "
            f"종료 코드 {exit_code} 경고가 남았습니다."
        )
    if state == "aborted":
        return f"datasetkey {datasetkey} | filekey {filekey} 작업이 강제 중단되었습니다."
    if state == "error":
        return (
            f"datasetkey {datasetkey} | filekey {filekey} 작업이 실패했습니다."
            + (f" (exit {exit_code})" if exit_code not in (None, "") else "")
        )
    if state == "running":
        return f"datasetkey {datasetkey} | filekey {filekey} 작업이 실행 중입니다."
    if state == "queued":
        return f"datasetkey {datasetkey} | filekey {filekey} 작업이 대기 중입니다."
    return f"datasetkey {datasetkey} | filekey {filekey} 작업 상태를 확인하세요."


def enrich_completed_job(job: dict | None) -> dict:
    if not isinstance(job, dict):
        return {}
    enriched = dict(job)
    enriched["duration_minutes"] = compute_job_duration_minutes(enriched)
    enriched["message"] = str(enriched.get("message") or infer_job_message(enriched))
    return enriched


def sort_jobs_by_recency(jobs: list[dict] | None) -> list[dict]:
    enriched_jobs = [enrich_completed_job(job) for job in (jobs or []) if isinstance(job, dict)]

    def sort_key(job: dict) -> tuple[float, str]:
        latest = (
            parse_iso_datetime(job.get("finished_at"))
            or parse_iso_datetime(job.get("started_at"))
            or datetime.fromtimestamp(0, tz=timezone.utc)
        )
        return (latest.timestamp(), str(job.get("job_id") or ""))

    return sorted(enriched_jobs, key=sort_key, reverse=True)


def build_restored_launcher_summary(
    completed_jobs: list[dict] | None,
    *,
    pipeline_status: dict | None = None,
    training_progress: dict | None = None,
) -> dict:
    jobs = sort_jobs_by_recency(completed_jobs)
    latest_job = jobs[0] if jobs else None
    pipeline_status = pipeline_status or {}
    training_progress = training_progress or {}
    if latest_job:
        return {
            "state": str(latest_job.get("state") or "idle"),
            "message": str(latest_job.get("message") or infer_job_message(latest_job)),
            "log_path": latest_job.get("log_path"),
            "last_exit_code": latest_job.get("exit_code"),
            "latest_job": latest_job,
        }
    progress_state = str(training_progress.get("state") or "").strip().lower()
    if progress_state == "completed":
        return {
            "state": "completed",
            "message": "최근 학습 결과를 불러왔습니다.",
            "log_path": None,
            "last_exit_code": 0,
            "latest_job": None,
        }
    pipeline_state = str(pipeline_status.get("state") or "").strip().lower()
    if pipeline_state:
        return {
            "state": pipeline_state,
            "message": str(pipeline_status.get("message") or "최근 상태를 불러왔습니다."),
            "log_path": None,
            "last_exit_code": None,
            "latest_job": None,
        }
    return {
        "state": "idle",
        "message": "아직 실행 기록이 없습니다.",
        "log_path": None,
        "last_exit_code": None,
        "latest_job": None,
    }


def build_effective_pipeline_status(
    pipeline_status: dict | None,
    *,
    training_progress: dict | None = None,
    completed_jobs: list[dict] | None = None,
    workspace_dir: Path | None = None,
) -> tuple[dict, dict]:
    raw = dict(pipeline_status or {})
    progress = training_progress or {}
    jobs = sort_jobs_by_recency(completed_jobs)
    latest_job = jobs[0] if jobs else None
    latest_success_job = next(
        (job for job in jobs if str(job.get("state") or "").strip().lower() in {"completed", "completed_warning"}),
        None,
    )

    raw_updated = parse_iso_datetime(raw.get("updated_at"))
    progress_updated = parse_iso_datetime(progress.get("updated_at"))
    latest_job_updated = parse_iso_datetime(
        latest_job.get("finished_at") if isinstance(latest_job, dict) else None
    ) or parse_iso_datetime(latest_job.get("started_at") if isinstance(latest_job, dict) else None)

    warnings: list[str] = []
    effective = dict(raw)
    effective_source = "pipeline_status"

    expected_workspace = None
    if workspace_dir is not None:
        try:
            expected_workspace = str(workspace_dir.resolve())
        except OSError:
            expected_workspace = str(workspace_dir)

    raw_workspace = str(raw.get("workspace_dir") or "").strip()
    workspace_mismatch = False
    if expected_workspace and raw_workspace:
        workspace_mismatch = raw_workspace.replace("\\", "/").rstrip("/") != expected_workspace.replace("\\", "/").rstrip("/")
        if workspace_mismatch:
            warnings.append(
                "pipeline_status.json의 workspace_dir가 현재 대시보드 workspace와 달라 오래된 상태 파일일 가능성이 큽니다."
            )

    freshest_nonraw = None
    for candidate in (progress_updated, latest_job_updated):
        if candidate is None:
            continue
        if freshest_nonraw is None or candidate > freshest_nonraw:
            freshest_nonraw = candidate

    stale_pipeline_status = False
    if freshest_nonraw is not None:
        if raw_updated is None or raw_updated < freshest_nonraw:
            stale_pipeline_status = True
            warnings.append("pipeline_status.json보다 최신 학습 결과 또는 실행 이력이 발견되어 상태를 보정했습니다.")
    if workspace_mismatch:
        stale_pipeline_status = True

    if stale_pipeline_status:
        if latest_job is not None and latest_job_updated is not None and (
            progress_updated is None or latest_job_updated >= progress_updated
        ):
            latest_job_state = str(latest_job.get("state") or "").strip().lower() or "completed"
            effective.update(
                {
                    "stage": (
                        "completed"
                        if latest_job_state in {"completed", "completed_warning"}
                        else ("error" if latest_job_state in {"error", "aborted"} else latest_job_state)
                    ),
                    "state": latest_job_state,
                    "message": str(latest_job.get("message") or infer_job_message(latest_job)),
                    "workspace_dir": expected_workspace or raw_workspace or None,
                    "updated_at": latest_job.get("finished_at") or latest_job.get("started_at"),
                }
            )
            effective_source = "launcher_history"
        elif progress_updated is not None:
            progress_state = str(progress.get("state") or "completed").strip().lower() or "completed"
            effective.update(
                {
                    "stage": "completed" if progress_state == "completed" else str(raw.get("stage") or progress_state),
                    "state": progress_state,
                    "message": (
                        str(raw.get("message") or "").strip()
                        if progress_state != "completed" and str(raw.get("message") or "").strip()
                        else "최근 학습 결과를 표시합니다."
                    ),
                    "workspace_dir": expected_workspace or raw_workspace or None,
                    "updated_at": progress.get("updated_at"),
                }
            )
            effective_source = "training_progress"

    if expected_workspace and not effective.get("workspace_dir"):
        effective["workspace_dir"] = expected_workspace

    diagnostics = {
        "warnings": warnings,
        "pipeline_status_source": effective_source,
        "pipeline_status_stale": stale_pipeline_status,
        "sources": {
            "pipeline_status": {
                "updated_at": raw.get("updated_at"),
                "state": raw.get("state"),
                "workspace_dir": raw.get("workspace_dir"),
            },
            "training_progress": {
                "updated_at": progress.get("updated_at"),
                "state": progress.get("state"),
                "history_len": len(progress.get("history") or []) if isinstance(progress, dict) else 0,
            },
            "latest_job": latest_job,
            "latest_success_job": latest_success_job,
        },
    }
    return effective, diagnostics


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
            while len(MANIFEST_SUMMARY_CACHE) > MANIFEST_SUMMARY_CACHE_MAX_ENTRIES:
                oldest_key = next(iter(MANIFEST_SUMMARY_CACHE))
                if oldest_key == cache_key and len(MANIFEST_SUMMARY_CACHE) == 1:
                    break
                MANIFEST_SUMMARY_CACHE.pop(oldest_key, None)
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
    for job in sort_jobs_by_recency(completed_jobs)[:limit]:
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
        training_progress = read_json(paths["training_progress"]) or {}
        effective_pipeline_status, _diagnostics = build_effective_pipeline_status(
            pipeline_status,
            training_progress=training_progress,
            completed_jobs=[],
            workspace_dir=paths.get("workspace_dir"),
        )
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
                "finished_at": fallback_finished_at or effective_pipeline_status.get("updated_at"),
                "duration_minutes": None,
                "message": effective_pipeline_status.get("message") or "최근 학습 결과를 불러왔습니다.",
            }
        return {
            "datasetkey": None,
            "filekey": None,
            "state": effective_pipeline_status.get("state") or "offline",
            "started_at": None,
            "finished_at": effective_pipeline_status.get("updated_at"),
            "duration_minutes": None,
            "message": effective_pipeline_status.get("message") or "최근 학습 기록이 아직 없습니다.",
        }

    latest_job = sort_jobs_by_recency(completed_jobs)[0]
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

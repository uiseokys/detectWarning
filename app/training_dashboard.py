from __future__ import annotations

import argparse
import atexit
import hashlib
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

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from action_training_pipeline import get_target_labels, load_config, resolve_paths
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
    normalize_metric_payload,
    read_json,
    summarize_manifest,
)
from training_config import resolve_pages_sync_config

FILEKEY_RANGE_PATTERN = re.compile(r"^(\d+)(?:~|[-–—])(\d+)$")
MAX_FILEKEY_RANGE_SIZE = 1000
GPU_STATUS_CACHE: dict[str, object] = {"timestamp": 0.0, "value": None}
GPU_STATUS_CACHE_LOCK = threading.Lock()
OVERVIEW_CACHE: dict[tuple, dict] = {}
OVERVIEW_CACHE_LOCK = threading.Lock()


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
    file_signature = []
    for path in watched:
        if not path.exists():
            file_signature.append((str(path), None, None))
            continue
        stat = path.stat()
        file_signature.append((str(path), stat.st_mtime_ns, stat.st_size))

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
        if path.exists():
            stat = path.stat()
            dynamic_log_paths.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            dynamic_log_paths.append((str(path), None, None))
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
    if "[pipeline][completed][completed]" in lowered or "정상 완료" in tail:
        return "completed"
    return "completed_warning"


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
                "state": infer_job_state_from_log(log_path),
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
    state_lock = threading.Lock()

    launcher_state: dict[str, object] = {
        "process": None,
        "started_at": None,
        "runtime_config_path": None,
        "current_job": None,
        "queued_jobs": [],
        "completed_jobs": [],
        "auto_start_enabled": True,
        "last_state": "idle",
        "last_exit_code": None,
        "last_message": "아직 실행 기록이 없습니다.",
        "log_path": None,
    }

    if launcher_history_path.exists():
        try:
            with launcher_history_path.open("r", encoding="utf-8") as handle:
                history_payload = json.load(handle)
            completed_jobs = history_payload.get("completed_jobs", [])
            if isinstance(completed_jobs, list):
                launcher_state["completed_jobs"] = completed_jobs
        except (OSError, json.JSONDecodeError):
            launcher_state["completed_jobs"] = []

    if not launcher_state["completed_jobs"]:
        restored_jobs = restore_completed_jobs_from_logs(job_logs_dir, runtime_config_dir)
        if restored_jobs:
            launcher_state["completed_jobs"] = restored_jobs
            persist_launcher_history(launcher_history_path, launcher_state)
            launcher_state["last_message"] = "기존 job 로그를 바탕으로 학습 완료 이력을 복구했습니다."

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
        sync_pages_report(
            paths,
            pages_sync,
            project_name=str(pages_sync["project_name"]),
            report_title=str(pages_sync["report_title"]),
            target_labels=get_target_labels(config),
        )

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
        with runtime_config_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_config, handle, ensure_ascii=False, indent=2)

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
                    "all",
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
        sync_pages_live(pages_sync, "online")

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
                    current_job["result_summary"] = collect_result_summary(paths)
                    completed_jobs = launcher_state.setdefault("completed_jobs", [])
                    if isinstance(completed_jobs, list):
                        completed_jobs.insert(0, snapshot_job(current_job))
                        del completed_jobs[30:]
                    persist_launcher_history(launcher_history_path, launcher_state)
                    sync_pages_report(
                        paths,
                        pages_sync,
                        project_name=str(pages_sync["project_name"]),
                        report_title=str(pages_sync["report_title"]),
                        target_labels=get_target_labels(config),
                    )
                launcher_state["process"] = None
                launcher_state["current_job"] = None
                launcher_state["last_exit_code"] = exit_code
                if isinstance(current_job, dict) and current_job.get("state") == "completed":
                    launcher_state["last_state"] = "completed"
                    launcher_state["last_message"] = final_message
                elif isinstance(current_job, dict) and current_job.get("state") == "completed_warning":
                    launcher_state["last_state"] = "completed_warning"
                    launcher_state["last_message"] = final_message
                else:
                    launcher_state["last_state"] = "error"
                    launcher_state["last_message"] = final_message

        active_process = launcher_state.get("process")
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
        return {
            "state": launcher_state.get("last_state", "idle"),
            "message": launcher_state.get("last_message", ""),
            "pid": active_pid,
            "started_at": launcher_state.get("started_at"),
            "auto_start_enabled": bool(launcher_state.get("auto_start_enabled", True)),
            "runtime_config_path": str(launcher_state["runtime_config_path"])
            if launcher_state.get("runtime_config_path")
            else None,
            "current_job": current_job,
            "pending_jobs": [snapshot_job(job) for job in queued_jobs] if isinstance(queued_jobs, list) else [],
            "completed_jobs": [snapshot_job(job) for job in completed_jobs] if isinstance(completed_jobs, list) else [],
            "last_exit_code": launcher_state.get("last_exit_code"),
            "log_path": str(launcher_state["log_path"]) if launcher_state.get("log_path") else None,
        }

    sync_pages_report(
        paths,
        pages_sync,
        project_name=str(pages_sync["project_name"]),
        report_title=str(pages_sync["report_title"]),
        target_labels=get_target_labels(config),
    )
    sync_pages_live(pages_sync, "online")

    def sync_pages_shutdown() -> None:
        try:
            sync_pages_report(
                paths,
                pages_sync,
                project_name=str(pages_sync["project_name"]),
                report_title=str(pages_sync["report_title"]),
                target_labels=get_target_labels(config),
            )
        except Exception as exc:
            print(f"[pages-sync] 종료 시 report 동기화 실패: {exc}")
        try:
            sync_pages_live(pages_sync, "offline")
        except Exception as exc:
            print(f"[pages-sync] 종료 시 offline 동기화 실패: {exc}")
        try:
            flush_pages_pushes()
        except Exception as exc:
            print(f"[pages-sync] 종료 시 push flush 실패: {exc}")

    atexit.register(sync_pages_shutdown)

    worker = threading.Thread(target=queue_worker, daemon=True)
    worker.start()

    def render_dashboard() -> str:
        return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Training Dashboard</title>
  <script>
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
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
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

    function loadSavedApiKey() {
      if (viewerMode) {
        return;
      }
      const input = document.getElementById('apiKeyInput');
      if (!input) {
        return;
      }
      const saved = window.localStorage.getItem('training_dashboard_aihub_api_key');
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
        window.localStorage.setItem('training_dashboard_aihub_api_key', value);
      } else {
        window.localStorage.removeItem('training_dashboard_aihub_api_key');
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
      const saved = window.localStorage.getItem('training_dashboard_aihub_datasetkey');
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
        window.localStorage.setItem('training_dashboard_aihub_datasetkey', value);
      } else {
        window.localStorage.removeItem('training_dashboard_aihub_datasetkey');
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

    async function refresh() {
      if (document.hidden) {
        return;
      }
      const response = await fetch('/api/overview');
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      if (
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

      const stateEl = document.getElementById('pipelineState');
      const displayState = launcher.state || pipeline.state || 'unknown';
      stateEl.textContent = formatLauncherState(displayState);
      stateEl.className = `status-pill ${toneClass(displayState)}`;

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
      setLaunchMessage(launcher.message || '여기에서 시작 결과와 최근 실행 메시지를 확인할 수 있습니다.', launcher.state === 'error');
      updateControlButtons(launcher);
      renderQueuedJobs(launcher.pending_jobs || []);

      document.getElementById('currentStage').textContent = pipeline.stage || '-';
      document.getElementById('currentMessage').textContent = pipeline.message || '-';
      document.getElementById('etaText').textContent = eta.label || '-';
      document.getElementById('etaMeta').textContent =
        eta.seconds_remaining !== null && eta.seconds_remaining !== undefined
          ? `현재 filekey 기준 예상 남은 시간`
          : '진행률이 쌓이면 계산합니다.';

      document.getElementById('epochProgress').textContent =
        `${progress.epochs_completed ?? 0} / ${progress.epochs_total ?? 0}`;
      document.getElementById('bestF1').textContent =
        `best macro F1: ${progress.best_val_macro_f1 ?? '-'}`;

      const rawTotal = data.dataset?.raw?.total ?? 0;
      const preparedTotal =
        (data.dataset?.prepared_train?.total ?? 0) +
        (data.dataset?.prepared_val?.total ?? 0) +
        (data.dataset?.prepared_test?.total ?? 0);
      document.getElementById('datasetTotals').textContent = `${rawTotal} / ${preparedTotal}`;
      document.getElementById('datasetSummary').textContent = 'raw videos / prepared pose samples';

      document.getElementById('artifactState').textContent = data.artifacts?.has_model ? 'ready' : 'pending';
      document.getElementById('workspaceDir').textContent = data.workspace_dir || '-';
      document.getElementById('gpuUsageText').textContent = formatGpuUsage(gpu);
      document.getElementById('gpuUsageMeta').textContent = formatGpuMeta(gpu);
      document.getElementById('gpuVramText').textContent = formatGpuVram(gpu);
      document.getElementById('gpuVramMeta').textContent = formatGpuVramMeta(gpu);
      document.getElementById('queueProgressText').textContent =
        `${queueProgress.completed ?? 0} / ${queueProgress.total ?? 0}`;
      document.getElementById('queueProgressMeta').textContent =
        `완료 ${queueProgress.completed ?? 0} / 실패 ${queueProgress.failed ?? 0} / 대기 ${queueProgress.pending ?? 0}`;
      document.getElementById('queueProgressFill').style.width =
        `${Math.max(0, Math.min(100, Math.round((queueProgress.ratio ?? 0) * 100)))}%`;
      renderIssueVideos(skipReport, cumulativeSkipReport);

      if (progress.latest) {
        document.getElementById('latestEpoch').textContent = `Epoch ${progress.latest.epoch}`;
        document.getElementById('latestMetrics').textContent =
          `train loss ${progress.latest.train_loss} / val acc ${progress.latest.val_accuracy} / val f1 ${progress.latest.val_macro_f1}`;
        document.getElementById('latestLoss').textContent =
          `train ${progress.latest.train_loss} / val ${progress.latest.val_loss}`;
        document.getElementById('latestLearningRate').textContent =
          `lr ${progress.latest.learning_rate ?? '-'}`;
      } else {
        document.getElementById('latestEpoch').textContent = '-';
        document.getElementById('latestMetrics').textContent = '-';
        document.getElementById('latestLoss').textContent = '-';
        document.getElementById('latestLearningRate').textContent = '-';
      }

      document.getElementById('resumeState').textContent =
        progress.resumed_from_checkpoint ? '이전 모델 이어학습' : '새 학습';
      document.getElementById('sampleCounts').textContent =
        `train ${progress.train_samples ?? 0} / val ${progress.val_samples ?? 0}`;
      document.getElementById('gpuDeviceText').textContent = formatGpuDevice(gpu);
      document.getElementById('gpuMemoryText').textContent = formatGpuMemory(gpu);
      document.getElementById('trainingDeviceText').textContent = formatTrainingDevice(progress, gpu);
      document.getElementById('trainingDeviceMeta').textContent = formatTrainingDeviceMeta(progress);
      document.getElementById('stageTimingValue').textContent = formatStageTimingValue(stageTimings);
      document.getElementById('stageTimingCopy').textContent = formatStageTimingMeta(stageTimings, totalDurationSeconds);
      document.getElementById('currentStageTimingText').textContent = formatStageTimingValue(stageTimings);
      document.getElementById('currentStageTimingMeta').textContent =
        formatActiveStageElapsed(pipeline) || formatStageTimingMeta(stageTimings, totalDurationSeconds);

      document.getElementById('updatedAt').textContent = pipeline.updated_at || progress.updated_at || '-';
      document.getElementById('configPath').textContent = launcher.runtime_config_path || data.config_path || '-';

      renderChart(progress.history || []);
      renderLossChart(progress.history || []);
      renderMetricInsights(progress, metrics);
      renderDatasetTable(data.dataset || {});
      renderCurrentJobProgress(currentJobProgress, data.current_dataset || {}, continualState, progress);
      const metricLabels = progress.labels || metrics.labels || [];
      const finalValidation = progress.final_validation || metrics.final_validation || {};
      renderPerClassMetrics(
        metricLabels,
        finalValidation.per_class || [],
        finalValidation.confusion_matrix || []
      );
      renderConfusionMatrix(metricLabels, finalValidation.confusion_matrix || []);
      renderCompletedLogs(launcher.completed_jobs || [], queueProgress);
      renderLogPanels(logs, launcher);
    }

    let latestOverview = null;

    loadSavedDatasetKey();
    loadSavedApiKey();
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
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard()

    @app.get("/api/overview")
    def overview(lite: bool = False) -> dict:
        return build_overview(
            paths,
            config_path,
            config=config,
            launcher_status=get_launcher_status(),
            lite=lite,
        )

    @app.get("/api/live-ping")
    def live_ping() -> dict:
        launcher = get_launcher_status()
        return {
            "ok": True,
            "state": launcher.get("state"),
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        }

    @app.post("/api/start")
    async def start_training(request: Request) -> dict:
        if str(config.get("dataset_source", "")).strip().lower() != "aihub_shell":
            raise HTTPException(
                status_code=400,
                detail="이 대시보드에서 직접 filekey 실행은 dataset_source가 aihub_shell일 때만 지원합니다.",
            )

        payload = await request.json()
        try:
            filekeys = parse_filekeys(payload.get("filekeys", ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        datasetkey = str(payload.get("datasetkey", "")).strip()
        api_key = str(payload.get("api_key", "")).strip()
        resume_only = bool(payload.get("resume_only", False))
        if not filekeys and not resume_only:
            raise HTTPException(status_code=400, detail="filekey를 하나 이상 입력해 주세요.")

        if not datasetkey:
            datasetkey = str(config.get("aihub_shell", {}).get("datasetkey", "")).strip()
        if not resume_only and datasetkey in (None, ""):
            raise HTTPException(
                status_code=400,
                detail="datasetkey를 입력해 주세요. datasetkey는 filekey가 아니라 AIHub 데이터셋 키입니다.",
            )

        with state_lock:
            update_process_state()
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            current_job = launcher_state.get("current_job")
            if resume_only:
                has_pending = isinstance(pending_jobs, list) and len(pending_jobs) > 0
                if not current_job and not has_pending:
                    raise HTTPException(status_code=409, detail="재개할 작업이 없습니다.")
                launcher_state["auto_start_enabled"] = True
                if current_job:
                    launcher_state["last_state"] = "running"
                    launcher_state["last_message"] = "현재 작업 완료 후 다음 큐 자동 시작을 다시 허용합니다."
                else:
                    launcher_state["last_state"] = "queued"
                    launcher_state["last_message"] = "대기열 자동 시작을 다시 켰습니다. 다음 큐를 이어서 시작합니다."
                return {
                    "ok": True,
                    "message": "대기열 자동 시작을 재개했습니다.",
                    "launcher": get_launcher_status(),
                }

            existing_keys = set()
            if isinstance(current_job, dict) and current_job.get("filekey"):
                existing_keys.add(f"{current_job.get('datasetkey', '')}:{current_job['filekey']}")
            if isinstance(pending_jobs, list):
                existing_keys.update(
                    f"{job.get('datasetkey', '')}:{job.get('filekey')}"
                    for job in pending_jobs
                    if isinstance(job, dict) and job.get("filekey")
                )

            appended = []
            skipped = []
            for filekey in filekeys:
                unique_key = f"{datasetkey}:{filekey}"
                if unique_key in existing_keys:
                    skipped.append(filekey)
                    continue
                job = build_job(filekey, datasetkey=datasetkey, api_key=api_key)
                if isinstance(pending_jobs, list):
                    pending_jobs.append(job)
                appended.append(filekey)
                existing_keys.add(unique_key)

            if not appended:
                raise HTTPException(
                    status_code=409,
                    detail="입력한 filekey가 모두 현재 작업 또는 대기열에 이미 있습니다.",
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

    @app.post("/api/pause")
    def pause_after_current() -> dict:
        with state_lock:
            update_process_state()
            current_job = launcher_state.get("current_job")
            pending_jobs = launcher_state.get("queued_jobs", [])
            has_pending = isinstance(pending_jobs, list) and len(pending_jobs) > 0
            if not current_job and not has_pending:
                raise HTTPException(status_code=409, detail="중지할 작업이 없습니다.")

            launcher_state["auto_start_enabled"] = False
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

    @app.post("/api/remove-queued-job")
    async def remove_queued_job(request: Request) -> dict:
        payload = await request.json()
        job_id = str(payload.get("job_id", "")).strip()
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

    @app.post("/api/force-stop")
    def force_stop_current_job() -> dict:
        with state_lock:
            update_process_state()
            active_process = launcher_state.get("process")
            current_job = launcher_state.get("current_job")
            if not isinstance(active_process, subprocess.Popen) or not isinstance(current_job, dict):
                raise HTTPException(status_code=409, detail="현재 강제 중단할 작업이 없습니다.")

            launcher_state["auto_start_enabled"] = False

            stop_process_tree(active_process)

            current_job["finished_at"] = current_timestamp()
            current_job["exit_code"] = active_process.returncode if active_process.returncode is not None else -1
            current_job["state"] = "aborted"
            current_job["result_summary"] = collect_result_summary(paths)

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

    @app.post("/api/reset")
    def reset_training_data() -> dict:
        with state_lock:
            update_process_state()
            active_process = launcher_state.get("process")
            if isinstance(active_process, subprocess.Popen) and active_process.poll() is None:
                launcher_state["auto_start_enabled"] = False
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

    return app


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
    pipeline_status = read_json(paths["pipeline_status"])
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
    completed_jobs = launcher_status.get("completed_jobs", []) if isinstance(launcher_status, dict) else []
    pending_jobs = launcher_status.get("pending_jobs", []) if isinstance(launcher_status, dict) else []
    current_job = launcher_status.get("current_job") if isinstance(launcher_status, dict) else None
    latest_completed_job = next(
        (
            job for job in completed_jobs
            if isinstance(job, dict) and job.get("state") in {"completed", "completed_warning"}
        ),
        None,
    )
    current_log_source = current_job if isinstance(current_job, dict) else latest_completed_job
    current_log_path = current_log_source.get("log_path") if isinstance(current_log_source, dict) else None
    current_log_tail = read_log_tail(current_log_path) if not lite else ""

    latest_error_job = next(
        (
            job for job in completed_jobs
            if isinstance(job, dict) and job.get("state") == "error"
        ),
        None,
    )
    latest_error_log_path = latest_error_job.get("log_path") if isinstance(latest_error_job, dict) else None
    latest_error_log_tail = read_log_tail(latest_error_log_path) if not lite else ""
    enriched_completed_jobs = []
    for job in completed_jobs:
        if not isinstance(job, dict):
            continue
        enriched_job = dict(job)
        if not lite:
            enriched_job["log_preview"] = read_log_preview(enriched_job.get("log_path"))
        enriched_completed_jobs.append(enriched_job)

    completed_count = len([
        job for job in completed_jobs
        if isinstance(job, dict) and job.get("state") in {"completed", "completed_warning"}
    ])
    failed_count = len([job for job in completed_jobs if isinstance(job, dict) and job.get("state") == "error"])
    pending_count = len(pending_jobs) if isinstance(pending_jobs, list) else 0
    active_count = 1 if current_job else 0
    total_count = completed_count + failed_count + pending_count + active_count
    progress_ratio = ((completed_count + failed_count) / total_count) if total_count > 0 else 0.0
    current_job_progress = build_current_job_progress(pipeline_status, training_progress, launcher_status)
    eta = estimate_eta(current_job_progress, launcher_status)
    dataset_summary = (
        {
            "raw": summarize_manifest(paths["raw_manifest"], label_field="target_label"),
            "train": summarize_manifest(paths["split_train"], label_field="target_label"),
            "val": summarize_manifest(paths["split_val"], label_field="target_label"),
            "test": summarize_manifest(paths["split_test"], label_field="target_label"),
            "prepared_train": summarize_manifest(paths["prepared_train"], label_field="target_label"),
            "prepared_val": summarize_manifest(paths["prepared_val"], label_field="target_label"),
            "prepared_test": summarize_manifest(paths["prepared_test"], label_field="target_label"),
        }
        if not lite
        else {}
    )
    current_dataset_summary = {
        "raw": summarize_manifest(paths["current_raw_manifest"], label_field="target_label"),
        "train": summarize_manifest(paths["current_split_train"], label_field="target_label"),
        "val": summarize_manifest(paths["current_split_val"], label_field="target_label"),
        "test": summarize_manifest(paths["current_split_test"], label_field="target_label"),
        "prepared_train": summarize_manifest(paths["current_prepared_train"], label_field="target_label"),
        "prepared_val": summarize_manifest(paths["current_prepared_val"], label_field="target_label"),
        "prepared_test": summarize_manifest(paths["current_prepared_test"], label_field="target_label"),
    }
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

    overview = {
        "overview_revision": overview_revision,
        "schema_version": STATE_SCHEMA_VERSION,
        "workspace_dir": str(paths["workspace_dir"]),
        "workspace_name": paths["workspace_dir"].name,
        "config_path": str(config_path),
        "target_labels": target_labels,
        "pipeline_status": pipeline_status,
        "training_progress": training_progress,
        "launcher": {
            **launcher_status,
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
                "tail": (
                    read_log_tail(latest_completed_job.get("log_path"))
                    if (not lite and isinstance(latest_completed_job, dict))
                    else ""
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
        "continual_state": read_json(paths["continual_state"]),
        "artifacts": {
            "has_model": (paths["artifacts_dir"] / "best_action_model.pt").exists(),
            "has_metrics": (paths["artifacts_dir"] / "metrics.json").exists(),
            "has_labels": (paths["artifacts_dir"] / "labels.json").exists(),
        },
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

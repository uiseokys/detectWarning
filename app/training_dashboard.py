from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from action_training_pipeline import load_config, resolve_paths
from update_pages_site import build_latest_result_payload, build_live_status_payload, write_json

FILEKEY_RANGE_PATTERN = re.compile(r"^(\d+)(?:~|[-–—])(\d+)$")
MAX_FILEKEY_RANGE_SIZE = 1000


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


def read_log_tail(path_value: str | Path | None, *, max_lines: int = 80, max_chars: int = 12000) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def read_log_preview(path_value: str | Path | None, *, max_lines: int = 6, max_chars: int = 900) -> str:
    return read_log_tail(path_value, max_lines=max_lines, max_chars=max_chars)


def decode_process_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


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


def classify_job_exit(paths: dict, current_job: dict | None, exit_code: int) -> tuple[str, str]:
    if exit_code == 0:
        return "completed", "현재 작업이 정상 완료되었습니다."

    pipeline_status = read_json(paths["pipeline_status"]) or {}
    pipeline_state = str(pipeline_status.get("state", "")).strip().lower()
    pipeline_stage = str(pipeline_status.get("stage", "")).strip().lower()
    log_tail = read_log_tail(current_job.get("log_path") if isinstance(current_job, dict) else None, max_lines=60, max_chars=6000)
    success_markers = (
        "[train] best model:",
        "[train] metrics:",
        "[train] labels:",
    )
    has_success_markers = any(marker in log_tail for marker in success_markers)

    if pipeline_state == "completed" or pipeline_stage == "completed" or has_success_markers:
        return (
            "completed_warning",
            f"현재 작업은 산출물 저장까지 완료됐지만 종료 코드 {exit_code}로 경고 종료되었습니다.",
        )

    return "error", f"현재 작업이 종료 코드 {exit_code}로 중단되었습니다."


def resolve_pages_sync_config(config: dict, base_dir: Path) -> dict:
    raw = config.get("pages_sync") or {}
    pages_dir_raw = str(raw.get("pages_dir") or "").strip()
    live_url = str(raw.get("live_url") or "").strip()
    live_url_env = str(raw.get("live_url_env") or "").strip()
    if not live_url and live_url_env:
        live_url = str(os.getenv(live_url_env, "")).strip()

    pages_dir = None
    if pages_dir_raw:
        candidate = Path(pages_dir_raw)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        pages_dir = candidate

    enabled = bool(raw.get("enabled")) and pages_dir is not None
    return {
        "enabled": enabled,
        "pages_dir": pages_dir,
        "project_name": str(raw.get("project_name") or "detectWarning"),
        "report_title": str(raw.get("report_title") or "행동 학습 결과 리포트"),
        "live_url": live_url,
        "redirect_delay_seconds": int(raw.get("redirect_delay_seconds") or 3),
        "git_auto_push": bool(raw.get("git_auto_push", False)),
        "git_commit_prefix": str(raw.get("git_commit_prefix") or "Update dashboard"),
    }


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

    def push_pages_repo(*relative_paths: str, reason: str) -> None:
        if not pages_sync["enabled"] or not pages_sync.get("git_auto_push") or pages_sync["pages_dir"] is None:
            return
        pages_dir = Path(pages_sync["pages_dir"])
        if not (pages_dir / ".git").exists():
            return
        existing_targets = [path for path in relative_paths if (pages_dir / path).exists()]
        if not existing_targets:
            return
        try:
            branch_result = subprocess.run(
                ["git", "-C", str(pages_dir), "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
            )
            branch = decode_process_output(branch_result.stdout).strip() or "main"

            unmerged_result = subprocess.run(
                ["git", "-C", str(pages_dir), "diff", "--name-only", "--diff-filter=U"],
                capture_output=True,
            )
            if unmerged_result.returncode == 0:
                unmerged_files = [
                    line.strip()
                    for line in decode_process_output(unmerged_result.stdout).splitlines()
                    if line.strip()
                ]
                if unmerged_files:
                    raise RuntimeError(
                        "detectWarning-pages 저장소에 미해결 충돌 파일이 남아 있습니다: "
                        + ", ".join(unmerged_files[:5])
                    )

            pull_result = subprocess.run(
                ["git", "-C", str(pages_dir), "pull", "--rebase", "--autostash", "origin", branch],
                capture_output=True,
            )
            if pull_result.returncode != 0:
                raise RuntimeError(
                    decode_process_output(pull_result.stderr or pull_result.stdout).strip()
                    or "git pull --rebase 에 실패했습니다."
                )

            add_result = subprocess.run(
                ["git", "-C", str(pages_dir), "add", *existing_targets],
                capture_output=True,
            )
            if add_result.returncode != 0:
                raise RuntimeError(
                    decode_process_output(add_result.stderr or add_result.stdout).strip()
                    or "git add 에 실패했습니다."
                )

            status = subprocess.run(
                ["git", "-C", str(pages_dir), "status", "--porcelain", "--", *existing_targets],
                capture_output=True,
            )
            if status.returncode != 0:
                raise RuntimeError(
                    decode_process_output(status.stderr or status.stdout).strip()
                    or "git status 확인에 실패했습니다."
                )
            if not decode_process_output(status.stdout).strip():
                return
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"{pages_sync['git_commit_prefix']} ({reason}) {timestamp}"
            commit_result = subprocess.run(
                ["git", "-C", str(pages_dir), "commit", "-m", commit_message],
                capture_output=True,
            )
            if commit_result.returncode != 0:
                raise RuntimeError(
                    decode_process_output(commit_result.stderr or commit_result.stdout).strip()
                    or "git commit 에 실패했습니다."
                )

            push_result = subprocess.run(
                ["git", "-C", str(pages_dir), "push"],
                capture_output=True,
            )
            if push_result.returncode != 0:
                fallback_push = subprocess.run(
                    ["git", "-C", str(pages_dir), "push", "-u", "origin", branch],
                    capture_output=True,
                )
                if fallback_push.returncode != 0:
                    raise RuntimeError(
                        decode_process_output(
                            fallback_push.stderr
                            or fallback_push.stdout
                            or push_result.stderr
                            or push_result.stdout
                        ).strip()
                        or "git push 에 실패했습니다."
                    )
        except Exception as exc:
            print(f"[pages-sync] 자동 push 실패: {exc}")

    def sync_pages_report() -> None:
        if not pages_sync["enabled"] or pages_sync["pages_dir"] is None:
            return
        payload = build_latest_result_payload(
            paths,
            project_name=str(pages_sync["project_name"]),
            report_title=str(pages_sync["report_title"]),
        )
        write_json(Path(pages_sync["pages_dir"]) / "latest-result.json", payload)
        push_pages_repo("latest-result.json", reason="report")

    def sync_pages_live(status: str) -> None:
        if not pages_sync["enabled"] or pages_sync["pages_dir"] is None:
            return
        live_url = str(pages_sync.get("live_url") or "")
        if status == "online" and not live_url:
            status = "offline"
        message = (
            "실시간 대시보드를 사용할 수 있습니다."
            if status == "online"
            else "현재 실시간 학습 대시보드가 꺼져 있습니다. 최신 결과 리포트를 표시합니다."
        )
        payload = build_live_status_payload(
            pages_dir=Path(pages_sync["pages_dir"]),
            status=status,
            live_url=live_url if status == "online" else "",
            message=message,
            redirect_delay_seconds=int(pages_sync.get("redirect_delay_seconds") or 3),
        )
        write_json(Path(pages_sync["pages_dir"]) / "live-status.json", payload)
        push_pages_repo("live-status.json", reason=f"live-{status}")

    def current_timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def build_job(filekey: str, datasetkey: str | int | None = None, api_key: str = "") -> dict:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_key = re.sub(r"[^0-9A-Za-z_-]+", "_", filekey).strip("_") or "filekey"
        return {
            "job_id": f"job_{stamp}_{safe_key}",
            "filekey": filekey,
            "datasetkey": str(datasetkey).strip() if datasetkey not in (None, "") else None,
            "api_key": api_key,
            "queued_at": current_timestamp(),
            "started_at": None,
            "finished_at": None,
            "state": "queued",
            "exit_code": None,
            "runtime_config_path": None,
            "log_path": None,
        }

    def build_retry_job_from(job: dict) -> dict:
        retry_job = build_job(
            str(job.get("filekey", "")),
            datasetkey=job.get("datasetkey"),
            api_key=str(job.get("api_key", "") or ""),
        )
        retry_job["retry_of"] = job.get("job_id")
        retry_job["retry_count"] = int(job.get("retry_count", 0) or 0) + 1
        return retry_job

    def write_dashboard_status(stage: str, state: str, message: str, **extra) -> None:
        payload = {
            "stage": stage,
            "state": state,
            "message": message,
            "workspace_dir": str(paths["workspace_dir"]),
            "updated_at": current_timestamp(),
            **extra,
        }
        with paths["pipeline_status"].open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def persist_launcher_history() -> None:
        completed_jobs = launcher_state.get("completed_jobs", [])
        payload = {
            "updated_at": current_timestamp(),
            "completed_jobs": [snapshot_job(job) for job in completed_jobs] if isinstance(completed_jobs, list) else [],
        }
        with launcher_history_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

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
            stage="idle",
            state="idle",
            message="학습 워크스페이스를 초기화했습니다.",
            stage_progress=0.0,
        )
        persist_launcher_history()
        sync_pages_report()

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

    def snapshot_job(job: dict | None) -> dict | None:
        if not job:
            return None
        return {
            "job_id": job.get("job_id"),
            "filekey": job.get("filekey"),
            "datasetkey": job.get("datasetkey"),
            "retry_of": job.get("retry_of"),
            "retry_count": job.get("retry_count"),
            "queued_at": job.get("queued_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "state": job.get("state"),
            "exit_code": job.get("exit_code"),
            "runtime_config_path": str(job["runtime_config_path"]) if job.get("runtime_config_path") else None,
            "log_path": str(job["log_path"]) if job.get("log_path") else None,
            "result_summary": job.get("result_summary"),
        }

    def collect_result_summary() -> dict:
        return {
            "raw_total": summarize_manifest(paths["raw_manifest"], label_field="target_label").get("total", 0),
            "train_total": summarize_manifest(paths["split_train"], label_field="target_label").get("total", 0),
            "val_total": summarize_manifest(paths["split_val"], label_field="target_label").get("total", 0),
            "test_total": summarize_manifest(paths["split_test"], label_field="target_label").get("total", 0),
            "prepared_train_total": summarize_manifest(paths["prepared_train"], label_field="target_label").get("total", 0),
            "prepared_val_total": summarize_manifest(paths["prepared_val"], label_field="target_label").get("total", 0),
            "prepared_test_total": summarize_manifest(paths["prepared_test"], label_field="target_label").get("total", 0),
        }

    def read_log_tail(path_value: str | Path | None, *, max_lines: int = 80, max_chars: int = 12000) -> str:
        if not path_value:
            return ""
        path = Path(path_value)
        if not path.exists() or not path.is_file():
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        lines = text.splitlines()
        tail = "\n".join(lines[-max_lines:])
        if len(tail) > max_chars:
            tail = tail[-max_chars:]
        return tail

    def read_log_preview(path_value: str | Path | None, *, max_lines: int = 6, max_chars: int = 900) -> str:
        return read_log_tail(path_value, max_lines=max_lines, max_chars=max_chars)

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
        write_dashboard_status(
            stage="queued",
            state="running",
            message=(
                f"datasetkey {job.get('datasetkey', '-')}"
                f" | filekey {job['filekey']} 작업을 시작합니다."
            ),
            current_filekey=job["filekey"],
            current_datasetkey=job.get("datasetkey"),
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
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
        sync_pages_live("online")

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
                    current_job["result_summary"] = collect_result_summary()
                    completed_jobs = launcher_state.setdefault("completed_jobs", [])
                    if isinstance(completed_jobs, list):
                        completed_jobs.insert(0, snapshot_job(current_job))
                        del completed_jobs[30:]
                    persist_launcher_history()
                    sync_pages_report()
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

    sync_pages_report()
    sync_pages_live("online")
    atexit.register(lambda: sync_pages_live("offline"))

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
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .sidebar-stack {
      display: grid;
      gap: 14px;
      position: sticky;
      top: 22px;
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
    .viewer-mode .queue-editor,
    .viewer-mode .control-actions {
      display: none !important;
    }
    html.viewer-mode-page #controlPanel,
    .viewer-mode {
      padding: 16px;
    }
    html.viewer-mode-page .dashboard-shell {
      grid-template-columns: 320px minmax(0, 1fr);
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
      grid-template-columns: repeat(2, minmax(0, 1fr));
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
    }
    .mini-card {
      border-radius: 16px;
    }
    .legend {
      color: #5b6d82;
      font-weight: 700;
    }
    .log-grid {
      grid-template-columns: 1fr 1fr;
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
    @media (max-width: 1480px) {
      .dashboard-shell {
        grid-template-columns: 350px minmax(0, 1fr);
      }
      .grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
      .card-featured,
      .card-queue {
        grid-column: span 2;
      }
    }
    @media (max-width: 1260px) {
      .dashboard-shell {
        grid-template-columns: 1fr;
      }
      .sidebar-stack {
        position: static;
      }
      .sidebar-stack .control-panel {
        grid-template-columns: 1.15fr 0.85fr;
      }
      .control-actions {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
      .grid {
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
        grid-template-columns: 1fr;
      }
      .value {
        font-size: 27px;
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
          <div class="chart-wrap">
            <svg id="trainingChart" class="chart" viewBox="0 0 800 260" preserveAspectRatio="none"></svg>
            <div class="legend">
              <span class="blue">Validation Accuracy</span>
              <span class="green">Validation Macro F1</span>
            </div>
          </div>
          <div class="chart-wrap" style="margin-top:18px;">
            <svg id="lossChart" class="chart" viewBox="0 0 800 260" preserveAspectRatio="none"></svg>
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
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">클래스별 검증 지표</h2>
            <div class="panel-copy">Precision / Recall / F1</div>
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
              </tr>
            </thead>
            <tbody id="perClassMetricsTable"></tbody>
          </table>
          <div id="perClassMetricsEmpty" class="empty" style="display:none; margin-top:14px;">클래스별 지표가 아직 없습니다.</div>
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
      if (queueEditor) {
        queueEditor.remove();
      }
      if (controlActions) {
        controlActions.remove();
      }
      if (metaRow) {
        metaRow.remove();
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

    function setLaunchMessage(message, isError) {
      const box = document.getElementById('launchMessage');
      box.textContent = message || '-';
      box.style.color = isError ? '#dc2626' : '#64748b';
      box.style.borderColor = isError ? 'rgba(220,38,38,0.18)' : 'rgba(148,163,184,0.18)';
      box.style.background = isError ? 'rgba(220,38,38,0.06)' : 'rgba(255,255,255,0.84)';
    }

    function updateControlButtons(launcher) {
      const startButton = document.getElementById('startButton');
      const stopButton = document.getElementById('stopButton');
      const forceStopButton = document.getElementById('forceStopButton');
      if (viewerMode) {
        startButton.disabled = true;
        stopButton.disabled = true;
        forceStopButton.disabled = true;
        document.getElementById('resetButton').disabled = true;
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
      const saved = window.localStorage.getItem('training_dashboard_aihub_api_key');
      if (saved) {
        document.getElementById('apiKeyInput').value = saved;
      }
    }

    function saveApiKey() {
      if (viewerMode) {
        return '';
      }
      const value = document.getElementById('apiKeyInput').value.trim();
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
      const saved = window.localStorage.getItem('training_dashboard_aihub_datasetkey');
      if (saved) {
        document.getElementById('datasetKeyInput').value = saved;
      }
    }

    function saveDatasetKey() {
      if (viewerMode) {
        return document.getElementById('datasetKeyInput').value.trim();
      }
      const value = document.getElementById('datasetKeyInput').value.trim();
      if (value) {
        window.localStorage.setItem('training_dashboard_aihub_datasetkey', value);
      } else {
        window.localStorage.removeItem('training_dashboard_aihub_datasetkey');
      }
      return value;
    }

    function renderChart(history) {
      const svg = document.getElementById('trainingChart');
      if (!history || !history.length) {
        svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="16">아직 학습 기록이 없습니다</text>';
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
      const maxX = Math.max(history.length - 1, 1);

      history.forEach((row, index) => {
        const x = padLeft + (index / maxX) * innerW;
        const accY = padTop + (1 - Math.max(0, Math.min(1, row.val_accuracy ?? 0))) * innerH;
        const f1Y = padTop + (1 - Math.max(0, Math.min(1, row.val_macro_f1 ?? 0))) * innerH;
        accPoints.push(`${x},${accY}`);
        f1Points.push(`${x},${f1Y}`);
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

      svg.innerHTML = `
        <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"></rect>
        ${gridLines}
        <polyline fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${accPoints.join(' ')}"></polyline>
        <polyline fill="none" stroke="#059669" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${f1Points.join(' ')}"></polyline>
        ${xLabels}
      `;
    }

    function renderLossChart(history) {
      const svg = document.getElementById('lossChart');
      if (!history || !history.length) {
        svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-size="16">아직 손실 기록이 없습니다</text>';
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

      history.forEach((row, index) => {
        const x = padLeft + (index / maxX) * innerW;
        const trainY = padTop + (1 - Math.min(1, Number(row.train_loss || 0) / maxLoss)) * innerH;
        const valY = padTop + (1 - Math.min(1, Number(row.val_loss || 0) / maxLoss)) * innerH;
        trainPoints.push(`${x},${trainY}`);
        valPoints.push(`${x},${valY}`);
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

      svg.innerHTML = `
        <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"></rect>
        ${ticks}
        <polyline fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${trainPoints.join(' ')}"></polyline>
        <polyline fill="none" stroke="#059669" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${valPoints.join(' ')}"></polyline>
        ${xLabels}
      `;
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
          .map(([label, count]) => `${label} ${count}`)
          .join(' / ');
        rows.push(`
          <tr>
            <td>${key}</td>
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
    }

    function renderPerClassMetrics(labels, perClass) {
      const tbody = document.getElementById('perClassMetricsTable');
      const empty = document.getElementById('perClassMetricsEmpty');
      if (!perClass || !perClass.length) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
      }
      empty.style.display = 'none';
      tbody.innerHTML = perClass.map((row) => {
        const label = labels?.[row.class_index] || `class_${row.class_index}`;
        return `
          <tr>
            <td>${label}</td>
            <td>${row.precision ?? '-'}</td>
            <td>${row.recall ?? '-'}</td>
            <td>${row.f1 ?? '-'}</td>
          </tr>
        `;
      }).join('');
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
        const stateLabel =
          job.state === 'completed' ? '완료' :
          job.state === 'completed_warning' ? '경고 종료' :
          job.state === 'aborted' ? '강제 중단' :
          '실패';
        const logPreview = job.log_preview || job.log_path || '-';
        return `
          <tr>
            <td>${job.filekey || '-'}</td>
            <td>${stateLabel}${job.exit_code !== null && job.exit_code !== undefined ? ` (${job.exit_code})` : ''}</td>
            <td>${rawTotal} / ${preparedTotal}</td>
            <td>${formatDateTime(job.started_at)}<br>${formatDateTime(job.finished_at)}</td>
            <td class="mono">${logPreview}</td>
          </tr>
        `;
      }).join('');
    }

    function renderLogPanels(logs, launcher) {
      const current = logs?.current || {};
      const latestError = logs?.latest_error || {};
      const currentMeta = current.filekey
        ? `datasetkey ${current.datasetkey || '-'} | filekey ${current.filekey}${current.path ? ' | ' + current.path : ''}`
        : (current.path || launcher?.log_path || '실행 중인 작업이 없으면 최근 완료 로그를 표시합니다.');
      document.getElementById('currentLogMeta').textContent =
        currentMeta;
      document.getElementById('currentLogText').textContent =
        current.tail || '표시할 로그가 없습니다.';

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

    async function refresh() {
      const response = await fetch('/api/overview');
      if (!response.ok) {
        return;
      }
      const data = await response.json();
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

      const stateEl = document.getElementById('pipelineState');
      const displayState = launcher.state || pipeline.state || 'unknown';
      stateEl.textContent = formatLauncherState(displayState);
      stateEl.className = `status-pill ${toneClass(displayState)}`;

      const datasetKey = data.aihub?.datasetkey ?? '-';
      document.getElementById('datasetKeyChip').textContent = datasetKey;
      if (datasetKey !== '-' && !document.getElementById('datasetKeyInput').value.trim()) {
        document.getElementById('datasetKeyInput').value = datasetKey;
      }
      document.getElementById('workspaceChip').textContent = data.workspace_name || '-';
      document.getElementById('launcherState').textContent = formatLauncherState(launcher.state || 'idle');
      document.getElementById('currentFilekey').textContent = formatJob(launcher.current_job);
      document.getElementById('currentDatasetkey').textContent =
        launcher.current_job?.datasetkey || formatDatasetkeys(launcher.pending_jobs || []);
      document.getElementById('pendingFilekeys').textContent = formatFilekeys((launcher.pending_jobs || []).map((job) => job.filekey));
      document.getElementById('completedJobs').textContent = formatCompletedJobs(launcher.completed_jobs || []);
      document.getElementById('autoStartState').textContent =
        launcher.auto_start_enabled === false ? '꺼짐' : '켜짐';
      document.getElementById('launcherLogPath').textContent = launcher.log_path || '-';
      setLaunchMessage(launcher.message || '여기에서 시작 결과와 최근 실행 메시지를 확인할 수 있습니다.', launcher.state === 'error');
      updateControlButtons(launcher);

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
      document.getElementById('queueProgressText').textContent =
        `${queueProgress.completed ?? 0} / ${queueProgress.total ?? 0}`;
      document.getElementById('queueProgressMeta').textContent =
        `완료 ${queueProgress.completed ?? 0} / 실패 ${queueProgress.failed ?? 0} / 대기 ${queueProgress.pending ?? 0}`;
      document.getElementById('queueProgressFill').style.width =
        `${Math.max(0, Math.min(100, Math.round((queueProgress.ratio ?? 0) * 100)))}%`;

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

      document.getElementById('updatedAt').textContent = pipeline.updated_at || progress.updated_at || '-';
      document.getElementById('configPath').textContent = launcher.runtime_config_path || data.config_path || '-';

      renderChart(progress.history || []);
      renderLossChart(progress.history || []);
      renderDatasetTable(data.dataset || {});
      renderCurrentJobProgress(currentJobProgress, data.current_dataset || {}, continualState, progress);
      renderPerClassMetrics(
        progress.labels || metrics.labels || [],
        progress.final_validation?.per_class || metrics.final_validation?.per_class || []
      );
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
    applyViewerMode();
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>"""

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard()

    @app.get("/api/overview")
    def overview() -> dict:
        return build_overview(paths, config_path, config=config, launcher_status=get_launcher_status())

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
            current_job["result_summary"] = collect_result_summary()

            completed_jobs = launcher_state.setdefault("completed_jobs", [])
            if isinstance(completed_jobs, list):
                completed_jobs.insert(0, snapshot_job(current_job))
                del completed_jobs[30:]
            persist_launcher_history()

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


def build_overview(paths: dict, config_path: Path, *, config: dict, launcher_status: dict | None = None) -> dict:
    launcher_status = launcher_status or {}
    pipeline_status = read_json(paths["pipeline_status"])
    training_progress = read_json(paths["training_progress"])
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
    current_log_tail = read_log_tail(current_log_path)

    latest_error_job = next(
        (
            job for job in completed_jobs
            if isinstance(job, dict) and job.get("state") == "error"
        ),
        None,
    )
    latest_error_log_path = latest_error_job.get("log_path") if isinstance(latest_error_job, dict) else None
    latest_error_log_tail = read_log_tail(latest_error_log_path)
    enriched_completed_jobs = []
    for job in completed_jobs:
        if not isinstance(job, dict):
            continue
        enriched_job = dict(job)
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

    return {
        "workspace_dir": str(paths["workspace_dir"]),
        "workspace_name": paths["workspace_dir"].name,
        "config_path": str(config_path),
        "pipeline_status": pipeline_status,
        "training_progress": training_progress,
        "launcher": {
            **launcher_status,
            "completed_jobs": enriched_completed_jobs,
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
                "tail": read_log_tail(latest_completed_job.get("log_path")) if isinstance(latest_completed_job, dict) else "",
            },
            "latest_error": {
                "filekey": latest_error_job.get("filekey") if isinstance(latest_error_job, dict) else None,
                "datasetkey": latest_error_job.get("datasetkey") if isinstance(latest_error_job, dict) else None,
                "path": latest_error_log_path,
                "tail": latest_error_log_tail,
            },
        },
        "dataset": {
            "raw": summarize_manifest(paths["raw_manifest"], label_field="target_label"),
            "train": summarize_manifest(paths["split_train"], label_field="target_label"),
            "val": summarize_manifest(paths["split_val"], label_field="target_label"),
            "test": summarize_manifest(paths["split_test"], label_field="target_label"),
            "prepared_train": summarize_manifest(paths["prepared_train"], label_field="target_label"),
            "prepared_val": summarize_manifest(paths["prepared_val"], label_field="target_label"),
            "prepared_test": summarize_manifest(paths["prepared_test"], label_field="target_label"),
        },
        "current_dataset": {
            "raw": summarize_manifest(paths["current_raw_manifest"], label_field="target_label"),
            "train": summarize_manifest(paths["current_split_train"], label_field="target_label"),
            "val": summarize_manifest(paths["current_split_val"], label_field="target_label"),
            "test": summarize_manifest(paths["current_split_test"], label_field="target_label"),
            "prepared_train": summarize_manifest(paths["current_prepared_train"], label_field="target_label"),
            "prepared_val": summarize_manifest(paths["current_prepared_val"], label_field="target_label"),
            "prepared_test": summarize_manifest(paths["current_prepared_test"], label_field="target_label"),
        },
        "continual_state": read_json(paths["continual_state"]),
        "artifacts": {
            "has_model": (paths["artifacts_dir"] / "best_action_model.pt").exists(),
            "has_metrics": (paths["artifacts_dir"] / "metrics.json").exists(),
            "has_labels": (paths["artifacts_dir"] / "labels.json").exists(),
        },
        "metrics": read_json(paths["artifacts_dir"] / "metrics.json"),
    }


def summarize_manifest(path: Path, label_field: str) -> dict:
    if not path.exists():
        return {"total": 0, "by_label": {}}
    total = 0
    by_label: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            payload = json.loads(line)
            label = str(payload.get(label_field, "unknown"))
            by_label[label] += 1
    return {"total": total, "by_label": dict(sorted(by_label.items()))}


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


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

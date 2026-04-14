from __future__ import annotations

import argparse
import json
import re
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


def create_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
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

    def current_timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def build_job(filekey: str, api_key: str = "") -> dict:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_key = re.sub(r"[^0-9A-Za-z_-]+", "_", filekey).strip("_") or "filekey"
        return {
            "job_id": f"job_{stamp}_{safe_key}",
            "filekey": filekey,
            "api_key": api_key,
            "queued_at": current_timestamp(),
            "started_at": None,
            "finished_at": None,
            "state": "queued",
            "exit_code": None,
            "runtime_config_path": None,
            "log_path": None,
        }

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

    def snapshot_job(job: dict | None) -> dict | None:
        if not job:
            return None
        return {
            "job_id": job.get("job_id"),
            "filekey": job.get("filekey"),
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
            message=f"filekey {job['filekey']} 작업을 시작합니다.",
            current_filekey=job["filekey"],
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
        launcher_state["last_message"] = f"filekey {job['filekey']} 학습을 진행 중입니다."

    def update_process_state() -> None:
        process = launcher_state.get("process")
        if process is not None and isinstance(process, subprocess.Popen):
            exit_code = process.poll()
            if exit_code is None:
                launcher_state["last_state"] = "running"
                current_job = launcher_state.get("current_job") or {}
                filekey = current_job.get("filekey", "-") if isinstance(current_job, dict) else "-"
                launcher_state["last_message"] = f"filekey {filekey} 작업이 실행 중입니다."
            else:
                current_job = launcher_state.get("current_job")
                if isinstance(current_job, dict):
                    current_job["finished_at"] = current_timestamp()
                    current_job["exit_code"] = exit_code
                    current_job["state"] = "completed" if exit_code == 0 else "error"
                    current_job["result_summary"] = collect_result_summary()
                    completed_jobs = launcher_state.setdefault("completed_jobs", [])
                    if isinstance(completed_jobs, list):
                        completed_jobs.insert(0, snapshot_job(current_job))
                        del completed_jobs[30:]
                    persist_launcher_history()
                launcher_state["process"] = None
                launcher_state["current_job"] = None
                launcher_state["last_exit_code"] = exit_code
                if exit_code == 0:
                    launcher_state["last_state"] = "completed"
                    launcher_state["last_message"] = "현재 작업이 정상 완료되었습니다."
                else:
                    launcher_state["last_state"] = "error"
                    launcher_state["last_message"] = f"현재 작업이 종료 코드 {exit_code}로 중단되었습니다."

        active_process = launcher_state.get("process")
        queued_jobs = launcher_state.get("queued_jobs", [])
        if active_process is None and isinstance(queued_jobs, list) and queued_jobs:
            next_job = queued_jobs.pop(0)
            start_pipeline_for_job(next_job)

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
            "runtime_config_path": str(launcher_state["runtime_config_path"])
            if launcher_state.get("runtime_config_path")
            else None,
            "current_job": current_job,
            "pending_jobs": [snapshot_job(job) for job in queued_jobs] if isinstance(queued_jobs, list) else [],
            "completed_jobs": [snapshot_job(job) for job in completed_jobs] if isinstance(completed_jobs, list) else [],
            "last_exit_code": launcher_state.get("last_exit_code"),
            "log_path": str(launcher_state["log_path"]) if launcher_state.get("log_path") else None,
        }

    worker = threading.Thread(target=queue_worker, daemon=True)
    worker.start()

    def render_dashboard() -> str:
        return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Training Dashboard</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: rgba(255, 255, 255, 0.94);
      --panel-soft: rgba(248, 250, 252, 0.98);
      --ink: #0f172a;
      --muted: #64748b;
      --line: rgba(148, 163, 184, 0.22);
      --accent: #2563eb;
      --good: #059669;
      --warn: #d97706;
      --danger: #dc2626;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      --radius-lg: 24px;
      --radius-md: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "SF Pro Display", "Pretendard", "Apple SD Gothic Neo", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 25%),
        linear-gradient(180deg, #fbfdff 0%, var(--bg) 100%);
    }
    .wrap {
      max-width: 1460px;
      margin: 0 auto;
      padding: 28px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 20px;
      margin-bottom: 24px;
    }
    .hero h1 {
      margin: 0;
      font-size: 42px;
      line-height: 1.02;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
      max-width: 760px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
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
    }
    .control-panel {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 18px;
      padding: 22px;
    }
    .control-title {
      margin: 0 0 10px;
      font-size: 24px;
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
      margin-bottom: 12px;
    }
    .meta-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(37, 99, 235, 0.08);
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
      min-height: 120px;
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
      height: 52px;
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
    }
    .launch-box {
      display: grid;
      gap: 12px;
      align-content: start;
      padding: 16px;
      border-radius: 20px;
      background: var(--panel-soft);
      border: 1px solid rgba(148,163,184,0.14);
    }
    .launch-item {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.84);
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
      min-height: 46px;
      padding: 12px 14px;
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
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .card {
      padding: 18px;
      min-height: 134px;
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
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
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
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      align-items: start;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.80), rgba(255,255,255,0.58));
    }
    .panel-title {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .panel-copy {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    .panel-body {
      padding: 20px 22px 22px;
    }
    .chart-wrap {
      padding: 14px;
      border-radius: var(--radius-md);
      background: var(--panel-soft);
      border: 1px solid rgba(148,163,184,0.14);
      margin-bottom: 14px;
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
      padding: 14px 16px;
      border-radius: 18px;
      background: var(--panel-soft);
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
      padding: 18px;
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
      height: 10px;
      border-radius: 999px;
      background: rgba(148,163,184,0.18);
      overflow: hidden;
      margin-top: 10px;
    }
    .progress-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #2563eb, #059669);
      width: 0%;
      transition: width 0.25s ease;
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
      margin-top: 18px;
    }
    .scroll-panel {
      max-height: 340px;
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
      background: var(--panel-soft);
      overflow: hidden;
    }
    .log-card-head {
      padding: 14px 16px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
      background: rgba(255,255,255,0.7);
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
      padding: 14px 16px;
      min-height: 220px;
      max-height: 320px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SF Mono", "JetBrains Mono", monospace;
      font-size: 12px;
      line-height: 1.65;
      color: #0f172a;
      background: rgba(255,255,255,0.74);
    }
    @media (max-width: 1200px) {
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
    }
    @media (max-width: 720px) {
      .wrap { padding: 18px; }
      .hero { flex-direction: column; align-items: stretch; }
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
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div>
        <h1>행동 학습 진행 대시보드</h1>
        <p>AIHub 분할 ZIP의 filekey를 여러 개 넣고, 하나가 끝나면 다음 작업이 자동으로 이어지도록 순차 학습 큐를 운영할 수 있습니다.</p>
      </div>
      <div id="pipelineState" class="status-pill tone-neutral">상태 확인 중</div>
    </section>

    <section class="card control-panel">
      <div>
        <h2 class="control-title">AIHub filekey 순차 학습 큐</h2>
        <div class="control-copy">
          여러 개의 분할 ZIP 파일 키를 쉼표 또는 줄바꿈으로 넣으면 됩니다.
          대시보드가 한 번에 하나씩 작업을 꺼내서, 다운로드와 압축 해제, pose 전처리, 학습까지 순서대로 처리합니다.
        </div>
        <div class="meta-row">
          <div class="meta-chip">datasetkey <span id="datasetKeyChip">-</span></div>
          <div class="meta-chip">workspace <span id="workspaceChip">-</span></div>
        </div>
        <label class="form-label" for="apiKeyInput">AIHub API 키</label>
        <input id="apiKeyInput" class="text-input" type="password" placeholder="AIHub API 키를 입력하세요" />
        <label class="form-label" for="filekeysInput">분할 ZIP filekey 입력</label>
        <textarea id="filekeysInput" class="input-area" placeholder="예:&#10;123456&#10;123457&#10;123458"></textarea>
        <div class="control-actions">
          <button id="startButton" class="primary-button" type="button">대기열에 추가</button>
          <div class="helper">현재 실행 중이어도 새 filekey를 추가하면 자동으로 다음 순서에 이어서 학습합니다.</div>
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
          <div class="launch-label">대기 중 filekey</div>
          <div class="launch-value" id="pendingFilekeys">-</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">최근 완료 작업</div>
          <div class="launch-value" id="completedJobs">-</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">실행 로그</div>
          <div class="launch-value mono" id="launcherLogPath">-</div>
        </div>
        <div id="launchMessage" class="launch-message">여기에서 시작 결과와 최근 실행 메시지를 확인할 수 있습니다.</div>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <div class="label">현재 단계</div>
        <div class="value" id="currentStage">-</div>
        <div class="subvalue" id="currentMessage">-</div>
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
        <div class="label">큐 진행률</div>
        <div class="value" id="queueProgressText">0 / 0</div>
        <div class="subvalue" id="queueProgressMeta">완료 0 / 실패 0 / 대기 0</div>
        <div class="progress-track"><div id="queueProgressFill" class="progress-fill"></div></div>
      </article>
    </section>

    <section class="main-grid">
      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">Epoch 진행 그래프</h2>
            <div class="panel-copy">validation accuracy와 macro F1 변화를 함께 봅니다.</div>
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
          </div>
        </div>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">데이터셋 요약</h2>
            <div class="panel-copy">split별 샘플 수와 클래스 분포를 확인합니다.</div>
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
    </section>

    <section class="panel logs-section">
      <div class="panel-head">
        <div>
          <h2 class="panel-title">완료 데이터 로그</h2>
          <div class="panel-copy">지금까지 처리된 filekey별 결과와 생성된 데이터 수를 확인합니다.</div>
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
          <div class="log-card-copy" id="currentLogMeta">실행 중인 작업이 없으면 가장 최근 로그를 표시합니다.</div>
        </div>
        <pre id="currentLogText" class="log-pre">로그를 불러오는 중입니다.</pre>
      </article>
      <article class="log-card">
        <div class="log-card-head">
          <h3 class="log-card-title">최근 오류 로그</h3>
          <div class="log-card-copy" id="errorLogMeta">최근 실패 작업이 있으면 마지막 로그를 표시합니다.</div>
        </div>
        <pre id="errorLogText" class="log-pre">오류 로그가 아직 없습니다.</pre>
      </article>
    </section>
  </div>
  <script>
    function toneClass(state) {
      if (state === 'completed') return 'tone-good';
      if (state === 'running') return 'tone-accent';
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
        const state = job.state === 'completed' ? '완료' : '실패';
        return `${job.filekey} ${state}`;
      }).join(' / ');
    }

    function setLaunchMessage(message, isError) {
      const box = document.getElementById('launchMessage');
      box.textContent = message || '-';
      box.style.color = isError ? '#dc2626' : '#64748b';
      box.style.borderColor = isError ? 'rgba(220,38,38,0.18)' : 'rgba(148,163,184,0.18)';
      box.style.background = isError ? 'rgba(220,38,38,0.06)' : 'rgba(255,255,255,0.84)';
    }

    function loadSavedApiKey() {
      const saved = window.localStorage.getItem('training_dashboard_aihub_api_key');
      if (saved) {
        document.getElementById('apiKeyInput').value = saved;
      }
    }

    function saveApiKey() {
      const value = document.getElementById('apiKeyInput').value.trim();
      if (value) {
        window.localStorage.setItem('training_dashboard_aihub_api_key', value);
      } else {
        window.localStorage.removeItem('training_dashboard_aihub_api_key');
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
        const stateLabel = job.state === 'completed' ? '완료' : '실패';
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
        ? `filekey ${current.filekey}${current.path ? ' | ' + current.path : ''}`
        : (current.path || launcher?.log_path || '실행 중인 작업이 없으면 최근 완료 로그를 표시합니다.');
      document.getElementById('currentLogMeta').textContent =
        currentMeta;
      document.getElementById('currentLogText').textContent =
        current.tail || '표시할 로그가 없습니다.';

      const errorMeta = latestError.filekey
        ? `최근 실패 filekey: ${latestError.filekey}${latestError.path ? ' | ' + latestError.path : ''}`
        : '최근 실패 작업이 있으면 마지막 로그를 표시합니다.';
      document.getElementById('errorLogMeta').textContent = errorMeta;
      document.getElementById('errorLogText').textContent =
        latestError.tail || '오류 로그가 아직 없습니다.';
    }

    async function startTraining() {
      const input = document.getElementById('filekeysInput').value.trim();
      const apiKey = saveApiKey();
      if (!input) {
        setLaunchMessage('filekey를 하나 이상 입력해 주세요.', true);
        return;
      }

      const button = document.getElementById('startButton');
      button.disabled = true;
      button.textContent = '실행 시작 중...';

      try {
        const response = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filekeys: input, api_key: apiKey }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || '학습 시작에 실패했습니다.');
        }
        setLaunchMessage(data.message || '학습을 시작했습니다.', false);
        document.getElementById('filekeysInput').value = '';
        await refresh();
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        button.disabled = false;
        button.textContent = '대기열에 추가';
      }
    }

    async function refresh() {
      const response = await fetch('/api/overview');
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      const pipeline = data.pipeline_status || {};
      const progress = data.training_progress || {};
      const launcher = data.launcher || {};
      const queueProgress = data.queue_progress || {};
      const logs = data.logs || {};

      const stateEl = document.getElementById('pipelineState');
      stateEl.textContent = pipeline.state || 'unknown';
      stateEl.className = `status-pill ${toneClass(pipeline.state)}`;

      document.getElementById('datasetKeyChip').textContent = data.aihub?.datasetkey ?? '-';
      document.getElementById('workspaceChip').textContent = data.workspace_name || '-';
      document.getElementById('launcherState').textContent = launcher.state || 'idle';
      document.getElementById('currentFilekey').textContent = formatJob(launcher.current_job);
      document.getElementById('pendingFilekeys').textContent = formatFilekeys((launcher.pending_jobs || []).map((job) => job.filekey));
      document.getElementById('completedJobs').textContent = formatCompletedJobs(launcher.completed_jobs || []);
      document.getElementById('launcherLogPath').textContent = launcher.log_path || '-';
      setLaunchMessage(launcher.message || '여기에서 시작 결과와 최근 실행 메시지를 확인할 수 있습니다.', launcher.state === 'error');

      document.getElementById('currentStage').textContent = pipeline.stage || '-';
      document.getElementById('currentMessage').textContent = pipeline.message || '-';

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
      } else {
        document.getElementById('latestEpoch').textContent = '-';
        document.getElementById('latestMetrics').textContent = '-';
      }

      document.getElementById('updatedAt').textContent = pipeline.updated_at || progress.updated_at || '-';
      document.getElementById('configPath').textContent = launcher.runtime_config_path || data.config_path || '-';

      renderChart(progress.history || []);
      renderDatasetTable(data.dataset || {});
      renderCompletedLogs(launcher.completed_jobs || [], queueProgress);
      renderLogPanels(logs, launcher);
    }

    loadSavedApiKey();
    document.getElementById('apiKeyInput').addEventListener('change', saveApiKey);
    document.getElementById('startButton').addEventListener('click', startTraining);
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
        filekeys = parse_filekeys(payload.get("filekeys", ""))
        api_key = str(payload.get("api_key", "")).strip()
        if not filekeys:
            raise HTTPException(status_code=400, detail="filekey를 하나 이상 입력해 주세요.")

        datasetkey = config.get("aihub_shell", {}).get("datasetkey")
        if datasetkey in (None, ""):
            raise HTTPException(status_code=400, detail="설정 파일에 aihub_shell.datasetkey 가 필요합니다.")

        with state_lock:
            update_process_state()
            pending_jobs = launcher_state.setdefault("queued_jobs", [])
            current_job = launcher_state.get("current_job")
            existing_keys = set()
            if isinstance(current_job, dict) and current_job.get("filekey"):
                existing_keys.add(str(current_job["filekey"]))
            if isinstance(pending_jobs, list):
                existing_keys.update(str(job.get("filekey")) for job in pending_jobs if isinstance(job, dict))

            appended = []
            skipped = []
            for filekey in filekeys:
                if filekey in existing_keys:
                    skipped.append(filekey)
                    continue
                job = build_job(filekey, api_key=api_key)
                if isinstance(pending_jobs, list):
                    pending_jobs.append(job)
                appended.append(filekey)
                existing_keys.add(filekey)

            if not appended:
                raise HTTPException(status_code=409, detail="입력한 filekey가 모두 현재 작업 또는 대기열에 이미 있습니다.")

            launcher_state["last_state"] = "queued"
            launcher_state["last_message"] = f"{len(appended)}개 filekey를 대기열에 추가했습니다."

        return {
            "ok": True,
            "message": (
                f"filekey {', '.join(appended)} 를 대기열에 추가했습니다."
                + (f" 중복으로 건너뜀: {', '.join(skipped)}" if skipped else "")
            ),
            "launcher": get_launcher_status(),
        }

    return app


def build_overview(paths: dict, config_path: Path, *, config: dict, launcher_status: dict | None = None) -> dict:
    launcher_status = launcher_status or {}
    completed_jobs = launcher_status.get("completed_jobs", []) if isinstance(launcher_status, dict) else []
    pending_jobs = launcher_status.get("pending_jobs", []) if isinstance(launcher_status, dict) else []
    current_job = launcher_status.get("current_job") if isinstance(launcher_status, dict) else None
    latest_completed_job = next(
        (
            job for job in completed_jobs
            if isinstance(job, dict) and job.get("state") == "completed"
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

    completed_count = len([job for job in completed_jobs if isinstance(job, dict) and job.get("state") == "completed"])
    failed_count = len([job for job in completed_jobs if isinstance(job, dict) and job.get("state") == "error"])
    pending_count = len(pending_jobs) if isinstance(pending_jobs, list) else 0
    active_count = 1 if current_job else 0
    total_count = completed_count + failed_count + pending_count + active_count
    progress_ratio = ((completed_count + failed_count) / total_count) if total_count > 0 else 0.0
    return {
        "workspace_dir": str(paths["workspace_dir"]),
        "workspace_name": paths["workspace_dir"].name,
        "config_path": str(config_path),
        "pipeline_status": read_json(paths["pipeline_status"]),
        "training_progress": read_json(paths["training_progress"]),
        "launcher": {
            **launcher_status,
            "completed_jobs": enriched_completed_jobs,
        },
        "aihub": {
            "datasetkey": config.get("aihub_shell", {}).get("datasetkey"),
        },
        "queue_progress": {
            "total": total_count,
            "completed": completed_count,
            "failed": failed_count,
            "pending": pending_count,
            "active": active_count,
            "ratio": round(progress_ratio, 4),
        },
        "logs": {
            "current": {
                "filekey": current_log_source.get("filekey") if isinstance(current_log_source, dict) else None,
                "path": current_log_path,
                "tail": current_log_tail,
            },
            "latest_completed": {
                "filekey": latest_completed_job.get("filekey") if isinstance(latest_completed_job, dict) else None,
                "path": latest_completed_job.get("log_path") if isinstance(latest_completed_job, dict) else None,
                "tail": read_log_tail(latest_completed_job.get("log_path")) if isinstance(latest_completed_job, dict) else "",
            },
            "latest_error": {
                "filekey": latest_error_job.get("filekey") if isinstance(latest_error_job, dict) else None,
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
        "artifacts": {
            "has_model": (paths["artifacts_dir"] / "best_action_model.pt").exists(),
            "has_metrics": (paths["artifacts_dir"] / "metrics.json").exists(),
            "has_labels": (paths["artifacts_dir"] / "labels.json").exists(),
        },
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_filekeys(raw_value) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        tokens = raw_value
    else:
        text = str(raw_value).replace("{", " ").replace("}", " ")
        tokens = re.split(r"[\s,;]+", text)

    cleaned: list[str] = []
    seen = set()
    for token in tokens:
        value = str(token).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def reset_training_workspace(paths: dict) -> None:
    reset_keys = (
        "raw_dir",
        "import_dir",
        "extracted_dir",
        "manifests_dir",
        "prepared_dir",
        "artifacts_dir",
    )
    for key in reset_keys:
        target = paths.get(key)
        if not isinstance(target, Path):
            continue
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    app = create_app(config_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

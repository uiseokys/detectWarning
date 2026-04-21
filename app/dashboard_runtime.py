from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from reporting import enrich_completed_job, read_json, summarize_manifest
from update_pages_site import build_latest_result_payload, build_live_status_payload, write_json


PAGES_PUSH_QUEUE: queue.Queue[dict] = queue.Queue()
PAGES_PUSH_WORKER_LOCK = threading.Lock()
PAGES_PUSH_WORKER_STARTED = False
PAGES_PUSH_DEBOUNCE_SECONDS = 1.2


def current_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def decode_process_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_log_tail(path_value: str | Path | None, *, max_lines: int = 80, max_chars: int = 12000) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    try:
        max_bytes = max(max_chars * 4, 16 * 1024)
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            raw = handle.read()
    except OSError:
        return ""
    for encoding in ("utf-8", "cp949", "euc-kr"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def read_log_preview(path_value: str | Path | None, *, max_lines: int = 6, max_chars: int = 900) -> str:
    return read_log_tail(path_value, max_lines=max_lines, max_chars=max_chars)


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


def snapshot_job(job: dict | None) -> dict | None:
    if not job:
        return None
    job = enrich_completed_job(job)
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
        "duration_minutes": job.get("duration_minutes"),
        "message": job.get("message"),
        "runtime_config_path": str(job["runtime_config_path"]) if job.get("runtime_config_path") else None,
        "log_path": str(job["log_path"]) if job.get("log_path") else None,
        "result_summary": job.get("result_summary"),
    }


def write_dashboard_status(paths: dict, *, stage: str, state: str, message: str, **extra) -> None:
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


def persist_launcher_history(launcher_history_path: Path, launcher_state: dict) -> None:
    completed_jobs = launcher_state.get("completed_jobs", [])
    payload = {
        "updated_at": current_timestamp(),
        "completed_jobs": [snapshot_job(job) for job in completed_jobs] if isinstance(completed_jobs, list) else [],
    }
    with launcher_history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def collect_result_summary(paths: dict) -> dict:
    pipeline_status = read_json(paths["pipeline_status"]) or {}
    training_progress = read_json(paths["training_progress"]) or {}
    current_skip_report = read_json(paths["current_skip_report"]) or {}
    current_skip_summary = current_skip_report.get("summary", {}) if isinstance(current_skip_report, dict) else {}
    return {
        "raw_total": summarize_manifest(paths["raw_manifest"], label_field="target_label").get("total", 0),
        "train_total": summarize_manifest(paths["split_train"], label_field="target_label").get("total", 0),
        "val_total": summarize_manifest(paths["split_val"], label_field="target_label").get("total", 0),
        "test_total": summarize_manifest(paths["split_test"], label_field="target_label").get("total", 0),
        "prepared_train_total": summarize_manifest(paths["prepared_train"], label_field="target_label").get("total", 0),
        "prepared_val_total": summarize_manifest(paths["prepared_val"], label_field="target_label").get("total", 0),
        "prepared_test_total": summarize_manifest(paths["prepared_test"], label_field="target_label").get("total", 0),
        "broken_count": int(current_skip_summary.get("broken_count", 0) or 0),
        "skipped_count": int(current_skip_summary.get("skipped_count", 0) or 0),
        "total_issues": int(current_skip_summary.get("total_issues", 0) or 0),
        "stage_timings": pipeline_status.get("stage_timings") or {},
        "total_duration_seconds": pipeline_status.get("total_duration_seconds"),
        "stopped_early": bool(training_progress.get("stopped_early")),
        "stop_reason": training_progress.get("stop_reason"),
        "train_distribution": training_progress.get("train_distribution") or {},
        "val_distribution": training_progress.get("val_distribution") or {},
    }


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


def _run_pages_push(pages_sync: dict, relative_paths: list[str], *, reason: str) -> None:
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


def _pages_push_worker() -> None:
    while True:
        first_job = PAGES_PUSH_QUEUE.get()
        jobs = [first_job]
        deadline = time.monotonic() + PAGES_PUSH_DEBOUNCE_SECONDS

        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                jobs.append(PAGES_PUSH_QUEUE.get(timeout=timeout))
            except queue.Empty:
                break

        grouped: dict[str, dict] = {}
        for job in jobs:
            pages_dir = str(job["pages_dir"])
            group = grouped.setdefault(
                pages_dir,
                {
                    "pages_sync": job["pages_sync"],
                    "targets": set(),
                    "reasons": [],
                },
            )
            group["targets"].update(job["targets"])
            group["reasons"].append(job["reason"])

        for group in grouped.values():
            reason = ", ".join(group["reasons"][-3:])
            _run_pages_push(
                group["pages_sync"],
                sorted(group["targets"]),
                reason=reason,
            )

        for _job in jobs:
            PAGES_PUSH_QUEUE.task_done()


def _ensure_pages_push_worker() -> None:
    global PAGES_PUSH_WORKER_STARTED
    with PAGES_PUSH_WORKER_LOCK:
        if PAGES_PUSH_WORKER_STARTED:
            return
        worker = threading.Thread(target=_pages_push_worker, name="pages-push-worker", daemon=True)
        worker.start()
        PAGES_PUSH_WORKER_STARTED = True


def push_pages_repo(pages_sync: dict, *relative_paths: str, reason: str) -> None:
    if not pages_sync["enabled"] or not pages_sync.get("git_auto_push") or pages_sync["pages_dir"] is None:
        return
    pages_dir = Path(pages_sync["pages_dir"])
    if not (pages_dir / ".git").exists():
        return
    existing_targets = [path for path in relative_paths if (pages_dir / path).exists()]
    if not existing_targets:
        return

    _ensure_pages_push_worker()
    PAGES_PUSH_QUEUE.put(
        {
            "pages_dir": str(pages_dir),
            "pages_sync": dict(pages_sync),
            "targets": existing_targets,
            "reason": reason,
        }
    )


def flush_pages_pushes(timeout_seconds: float = 10.0) -> None:
    end_time = time.monotonic() + max(timeout_seconds, 0.0)
    while PAGES_PUSH_QUEUE.unfinished_tasks:
        if time.monotonic() >= end_time:
            break
        time.sleep(0.1)


def sync_pages_report(paths: dict, pages_sync: dict, *, project_name: str, report_title: str, target_labels: list[str]) -> None:
    if not pages_sync["enabled"] or pages_sync["pages_dir"] is None:
        return
    payload = build_latest_result_payload(
        paths,
        project_name=project_name,
        report_title=report_title,
        target_labels=target_labels,
    )
    write_json(Path(pages_sync["pages_dir"]) / "latest-result.json", payload)
    push_pages_repo(pages_sync, "latest-result.json", reason="report")


def sync_pages_live(pages_sync: dict, status: str) -> None:
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
    push_pages_repo(pages_sync, "live-status.json", reason=f"live-{status}")

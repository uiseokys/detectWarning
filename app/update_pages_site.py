from __future__ import annotations

import argparse
import json
from pathlib import Path

from action_training_pipeline import get_target_labels, load_config, resolve_paths
from reporting import (
    STATE_SCHEMA_VERSION,
    build_per_class_support,
    build_recent_jobs,
    get_best_macro_f1,
    get_latest_job,
    normalize_label_list,
    normalize_metric_payload,
    now_iso,
    read_json,
    safe_number,
    summarize_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="detectWarning 학습 결과를 Pages 저장소 JSON으로 내보냅니다."
    )
    parser.add_argument(
        "--config",
        default="configs/action_training.aihub_shell.example.json",
        help="행동 학습 파이프라인 설정 JSON 경로",
    )
    parser.add_argument(
        "--pages-dir",
        required=True,
        help="Cloudflare Pages 저장소 루트 경로",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="latest-result.json을 최신 학습 결과로 갱신합니다.",
    )
    report_parser.add_argument(
        "--project-name",
        default="detectWarning",
        help="결과 페이지에 표시할 프로젝트 이름",
    )
    report_parser.add_argument(
        "--report-title",
        default="행동 학습 결과 리포트",
        help="결과 페이지 제목",
    )

    online_parser = subparsers.add_parser(
        "live-online",
        help="live-status.json을 online 상태로 바꿉니다.",
    )
    online_parser.add_argument("--live-url", required=True, help="현재 live 대시보드 URL")
    online_parser.add_argument(
        "--message",
        default="실시간 대시보드를 사용할 수 있습니다.",
        help="online 상태 메시지",
    )
    online_parser.add_argument(
        "--redirect-delay-seconds",
        type=int,
        default=3,
        help="자동 이동까지의 대기 시간(초)",
    )

    offline_parser = subparsers.add_parser(
        "live-offline",
        help="live-status.json을 offline 상태로 바꿉니다.",
    )
    offline_parser.add_argument(
        "--message",
        default="현재 실시간 학습 대시보드가 꺼져 있습니다. 최신 결과 리포트를 표시합니다.",
        help="offline 상태 메시지",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    pages_dir = Path(args.pages_dir).expanduser().resolve()
    pages_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "report":
        payload = build_latest_result_payload(
            paths,
            project_name=args.project_name,
            report_title=args.report_title,
            target_labels=get_target_labels(config),
        )
        write_json(pages_dir / "latest-result.json", payload)
        print(f"[pages] updated latest-result.json -> {pages_dir / 'latest-result.json'}")
        return

    if args.command == "live-online":
        payload = build_live_status_payload(
            pages_dir=pages_dir,
            status="online",
            live_url=args.live_url,
            message=args.message,
            redirect_delay_seconds=args.redirect_delay_seconds,
        )
        write_json(pages_dir / "live-status.json", payload)
        print(f"[pages] updated live-status.json -> online ({args.live_url})")
        return

    if args.command == "live-offline":
        payload = build_live_status_payload(
            pages_dir=pages_dir,
            status="offline",
            live_url="",
            message=args.message,
        )
        write_json(pages_dir / "live-status.json", payload)
        print("[pages] updated live-status.json -> offline")
        return

    raise RuntimeError(f"지원하지 않는 명령입니다: {args.command}")


def build_latest_result_payload(
    paths: dict,
    *,
    project_name: str,
    report_title: str,
    target_labels: list[str] | None = None,
) -> dict:
    metrics = normalize_metric_payload(
        read_json(paths["artifacts_dir"] / "metrics.json") or {},
        target_labels=target_labels or [],
    )
    labels_payload = read_json(paths["artifacts_dir"] / "labels.json") or {}
    history = metrics.get("history") or []
    final_validation = metrics.get("final_validation") or {}
    labels = (
        metrics.get("labels")
        or normalize_label_list(target_labels or [])
        or normalize_label_list(labels_payload.get("labels") or [])
    )
    best_epoch = metrics.get("best_epoch")
    resumed_from_checkpoint = bool(metrics.get("resumed_from_checkpoint"))
    latest_job = get_latest_job(paths)
    launcher_history = read_json(paths["workspace_dir"] / "launcher_history.json") or {}
    continual_state = read_json(paths["continual_state"]) or {}
    cumulative_skip_report = read_json(paths["cumulative_skip_report"]) or {
        "summary": {"total_issues": 0, "broken_count": 0, "skipped_count": 0},
        "issues": [],
    }

    dataset = {
        "raw": summarize_manifest(paths["raw_manifest"], label_field="target_label"),
        "train": summarize_manifest(paths["split_train"], label_field="target_label"),
        "val": summarize_manifest(paths["split_val"], label_field="target_label"),
        "test": summarize_manifest(paths["split_test"], label_field="target_label"),
        "prepared_train": summarize_manifest(paths["prepared_train"], label_field="target_label"),
        "prepared_val": summarize_manifest(paths["prepared_val"], label_field="target_label"),
        "prepared_test": summarize_manifest(paths["prepared_test"], label_field="target_label"),
    }
    train_total = dataset["prepared_train"]["total"]
    val_total = dataset["prepared_val"]["total"]
    test_total = dataset["prepared_test"]["total"]
    confusion_matrix = final_validation.get("confusion_matrix") or []
    supports = build_per_class_support(labels, confusion_matrix)
    covered_classes = [row for row in supports if int(row.get("support", 0)) > 0]
    dominant_class = max(supports, key=lambda row: int(row.get("support", 0)), default=None)
    total_val_samples = sum(int(row.get("support", 0)) for row in supports)
    issue_summary = cumulative_skip_report.get("summary") or {}
    recent_jobs = build_recent_jobs(launcher_history)

    per_class_rows = []
    for row in final_validation.get("per_class", []) or []:
        class_index = int(row.get("class_index", len(per_class_rows)))
        label = labels[class_index] if 0 <= class_index < len(labels) else str(class_index)
        support = next(
            (int(item.get("support", 0)) for item in supports if int(item.get("class_index", -1)) == class_index),
            0,
        )
        per_class_rows.append(
            {
                "label": label,
                "precision": safe_number(row.get("precision")),
                "recall": safe_number(row.get("recall")),
                "f1": safe_number(row.get("f1")),
                "support": support,
            }
        )

    latest_job_state = str(latest_job.get("state") or "offline")
    latest_job_message = str(latest_job.get("message") or "최근 학습 기록을 불러오지 못했습니다.")

    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "project_name": project_name,
        "report_title": report_title,
        "updated_at": now_iso(),
        "summary": {
            "current_state": latest_job_state,
            "latest_datasetkey": latest_job.get("datasetkey"),
            "latest_filekey": latest_job.get("filekey"),
            "latest_job_state": latest_job_state,
            "train_samples": train_total,
            "val_samples": val_total,
            "test_samples": test_total,
            "best_epoch": best_epoch,
            "best_val_macro_f1": safe_number(get_best_macro_f1(history)),
            "final_accuracy": safe_number(final_validation.get("accuracy")),
            "final_macro_f1": safe_number(final_validation.get("macro_f1")),
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "val_sample_total": total_val_samples,
            "class_coverage": {
                "covered": len(covered_classes),
                "total": len(labels),
            },
            "dominant_class": dominant_class.get("label") if dominant_class else None,
            "dominant_class_support": dominant_class.get("support") if dominant_class else 0,
            "issues": {
                "total": int(issue_summary.get("total_issues", 0) or 0),
                "broken": int(issue_summary.get("broken_count", 0) or 0),
                "skipped": int(issue_summary.get("skipped_count", 0) or 0),
            },
        },
        "latest_job": {
            "datasetkey": latest_job.get("datasetkey"),
            "filekey": latest_job.get("filekey"),
            "state": latest_job_state,
            "started_at": latest_job.get("started_at"),
            "finished_at": latest_job.get("finished_at"),
            "duration_minutes": latest_job.get("duration_minutes"),
            "message": latest_job_message,
        },
        "artifacts": {
            "best_model": str(paths["artifacts_dir"] / "best_action_model.pt"),
            "metrics": str(paths["artifacts_dir"] / "metrics.json"),
            "labels": str(paths["artifacts_dir"] / "labels.json"),
            "has_model": (paths["artifacts_dir"] / "best_action_model.pt").exists(),
            "has_metrics": (paths["artifacts_dir"] / "metrics.json").exists(),
            "has_labels": (paths["artifacts_dir"] / "labels.json").exists(),
        },
        "labels": labels,
        "history": history,
        "per_class": per_class_rows,
        "confusion_matrix": confusion_matrix,
        "dataset": dataset,
        "continual_state": continual_state,
        "issues": cumulative_skip_report,
        "recent_jobs": recent_jobs,
        "notes": [
            "실시간 대시보드가 켜져 있으면 이 페이지는 live 화면으로 자동 전환됩니다.",
            "경고 종료는 산출물 저장까지 완료되었지만 종료 단계에서만 경고 코드가 남은 상태입니다.",
            "이 파일은 detectWarning 학습 결과에서 자동으로 생성된 최신 공유용 리포트입니다.",
        ],
    }
    return payload


def build_live_status_payload(
    *,
    pages_dir: Path,
    status: str,
    live_url: str,
    message: str,
    redirect_delay_seconds: int = 3,
) -> dict:
    current = read_json(pages_dir / "live-status.json") or {}
    effective_live_url = str(live_url or current.get("live_url") or "").strip()
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": status,
        "project_name": current.get("project_name") or "detectWarning Training Dashboard",
        "live_title": current.get("live_title") or "실시간 연결 가능",
        "offline_title": current.get("offline_title") or "오프라인 리포트",
        "live_url": effective_live_url,
        "report_url": current.get("report_url") or "/",
        "redirect_delay_seconds": redirect_delay_seconds,
        "updated_at": now_iso(),
        "message": message,
    }
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

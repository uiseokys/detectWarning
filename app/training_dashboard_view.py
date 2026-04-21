from __future__ import annotations

from html import escape


def _text(value, default: str = "-") -> str:
    if value in (None, ""):
        return default
    return escape(str(value))


def _float_text(value, digits: int = 4, default: str = "-") -> str:
    if value in (None, ""):
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _text(value, default)


def _int_text(value, default: str = "0") -> str:
    if value in (None, ""):
        return default
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return _text(value, default)


def _join_label_counts(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return "-"
    items = payload.get("by_label") or {}
    if not isinstance(items, dict) or not items:
        return "-"
    return ", ".join(f"{label} {count}" for label, count in sorted(items.items()))


def _state_tone(state: str | None) -> str:
    normalized = str(state or "").strip().lower()
    if normalized in {"completed", "online"}:
        return "good"
    if normalized in {"running", "queued"}:
        return "accent"
    if normalized in {"completed_warning", "paused", "prepare", "download", "train", "warning"}:
        return "warn"
    if normalized in {"error", "aborted", "offline"}:
        return "danger"
    return "neutral"


def _state_label(state: str | None) -> str:
    normalized = str(state or "").strip().lower()
    labels = {
        "completed": "완료",
        "completed_warning": "경고 포함 완료",
        "online": "온라인",
        "running": "실행 중",
        "queued": "대기 중",
        "paused": "일시 중지",
        "prepare": "전처리",
        "download": "다운로드",
        "train": "학습 중",
        "warning": "주의",
        "error": "오류",
        "aborted": "중단됨",
        "offline": "오프라인",
        "idle": "대기",
        "starting": "시작 중",
    }
    if normalized in labels:
        return labels[normalized]
    return _text(state)


def _split_label(name: str | None) -> str:
    normalized = str(name or "").strip().lower()
    labels = {
        "raw": "원본",
        "train": "학습",
        "val": "검증",
        "test": "테스트",
        "prepared_train": "전처리 학습",
        "prepared_val": "전처리 검증",
        "prepared_test": "전처리 테스트",
    }
    if normalized in labels:
        return labels[normalized]
    return _text(name)


def _resume_mode_label(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    labels = {
        "full": "이전 체크포인트 이어서",
        "new": "새로 시작",
        "none": "새로 시작",
    }
    if normalized in labels:
        return labels[normalized]
    return _text(value)


def _render_table(headers: list[str], rows: list[list[str]], empty_message: str, *, compact: bool = False) -> str:
    header_html = "".join(f"<th>{_text(header)}</th>" for header in headers)
    if rows:
        body_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
    else:
        body_html = (
            f"<tr><td colspan=\"{len(headers)}\" class=\"empty-cell\">{_text(empty_message)}</td></tr>"
        )
    compact_class = " data-table-compact" if compact else ""
    return (
        "<div class=\"table-wrap\">"
        f"<table class=\"data-table{compact_class}\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
    )


def _render_summary_cards(cards: list[tuple[str, str, str, str]]) -> str:
    parts = []
    for tone, label, value, copy in cards:
        parts.append(
            f"<article class=\"summary-card summary-card-{_text(tone, 'neutral')}\">"
            f"<div class=\"summary-label\">{_text(label)}</div>"
            f"<div class=\"summary-value\">{_text(value)}</div>"
            f"<div class=\"summary-copy\">{_text(copy)}</div>"
            "</article>"
        )
    return "".join(parts)


def _render_banner(message: str, tone: str, title: str) -> str:
    return (
        f"<section class=\"banner banner-{_text(tone, 'neutral')}\">"
        f"<div class=\"banner-title\">{_text(title)}</div>"
        f"<div class=\"banner-copy\">{_text(message)}</div>"
        "</section>"
    )


def _render_warning_banner(diagnostics: dict) -> str:
    warnings = diagnostics.get("warnings") or []
    if not warnings:
        return ""
    items = "".join(f"<li>{_text(item)}</li>" for item in warnings)
    return (
        "<section class=\"banner banner-warn\">"
        "<div class=\"banner-title\">상태 진단</div>"
        f"<ul class=\"banner-list\">{items}</ul>"
        "</section>"
    )


def _render_diagnostics_details(diagnostics: dict) -> str:
    files = diagnostics.get("files") or {}
    pipeline = diagnostics.get("pipeline") or {}
    file_rows = []
    for key, payload in files.items():
        if not isinstance(payload, dict):
            continue
        file_rows.append(
            [
                _text(key),
                _text("yes" if payload.get("exists") else "no"),
                _text(payload.get("updated_at")),
                _int_text(payload.get("size"), "-"),
                _text(payload.get("path")),
            ]
        )
    pipeline_rows = [[_text("source"), _text(pipeline.get("source"))]]
    for warning in pipeline.get("warnings") or []:
        pipeline_rows.append([_text("warning"), _text(warning)])
    return (
        "<details class=\"detail-panel\">"
        "<summary>진단 상세 보기</summary>"
        "<div class=\"detail-grid\">"
        "<section class=\"mini-panel\">"
        "<h3>Pipeline 진단</h3>"
        + _render_table(["kind", "value"], pipeline_rows, "pipeline 진단 정보가 없습니다.", compact=True)
        + "</section>"
        "<section class=\"mini-panel\">"
        "<h3>파일 진단</h3>"
        + _render_table(["file", "exists", "updated_at", "size", "path"], file_rows, "진단 파일 정보가 없습니다.", compact=True)
        + "</section>"
        "</div>"
        "</details>"
    )


def _render_actions_panel(default_datasetkey: str, controls_enabled: bool) -> str:
    if not controls_enabled:
        return (
            "<section class=\"panel sidebar-panel\">"
            "<div class=\"panel-head\"><div><h2>작업 제어</h2><p>이 설정에서는 브라우저에서 직접 실행을 지원하지 않습니다.</p></div></div>"
            "<div class=\"panel-body\">"
            "<p class=\"muted-block\">dataset_source가 <code>aihub_shell</code>일 때만 filekey 큐 제어를 사용할 수 있습니다.</p>"
            "</div>"
            "</section>"
        )
    return (
        "<section class=\"panel sidebar-panel\">"
        "<div class=\"panel-head\"><div><h2>작업 제어</h2><p>스크립트 없이도 동작하는 기본 입력 폼입니다.</p></div></div>"
        "<div class=\"panel-body stack\">"
        "<form method=\"post\" action=\"/actions/start\" class=\"stack-form\">"
        "<label class=\"field-label\">datasetkey"
        f"<input type=\"text\" name=\"datasetkey\" value=\"{_text(default_datasetkey, '')}\" placeholder=\"예: 171\" />"
        "</label>"
        "<label class=\"field-label\">AIHub API 키"
        "<input type=\"password\" name=\"api_key\" value=\"\" placeholder=\"필요할 때만 입력\" />"
        "</label>"
        "<label class=\"field-label\">filekeys"
        "<textarea name=\"filekeys\" rows=\"7\" placeholder=\"예:&#10;49841&#10;49842&#10;49843\"></textarea>"
        "</label>"
        "<button type=\"submit\" class=\"primary-button\">큐 시작 / 추가</button>"
        "</form>"
        "<div class=\"action-grid\">"
        "<form method=\"post\" action=\"/actions/start\"><input type=\"hidden\" name=\"resume_only\" value=\"1\" /><button type=\"submit\" class=\"secondary-button\">자동 시작 재개</button></form>"
        "<form method=\"post\" action=\"/actions/pause\"><button type=\"submit\" class=\"secondary-button secondary-button-warn\">현재 작업 후 중지</button></form>"
        "<form method=\"post\" action=\"/actions/force-stop\"><button type=\"submit\" class=\"secondary-button secondary-button-danger\">지금 중단</button></form>"
        "<form method=\"post\" action=\"/actions/reset\"><button type=\"submit\" class=\"secondary-button secondary-button-danger\">처음부터 다시 시작</button></form>"
        "</div>"
        "</div>"
        "</section>"
    )


def _render_queue_panel(launcher: dict, current_job_progress: dict) -> str:
    pending_jobs = launcher.get("pending_jobs") or []
    current_job = launcher.get("current_job") or {}
    queue_items = []
    for job in pending_jobs:
        if not isinstance(job, dict):
            continue
        queue_items.append(
            "<li class=\"queue-item\">"
            "<div class=\"queue-main\">"
            f"<strong>{_text(job.get('filekey'))}</strong>"
            f"<div class=\"queue-copy\">datasetkey {_text(job.get('datasetkey'))} / 대기 등록 {_text(job.get('queued_at'))}</div>"
            "</div>"
            "<form method=\"post\" action=\"/actions/remove-queued\">"
            f"<input type=\"hidden\" name=\"job_id\" value=\"{_text(job.get('job_id'), '')}\" />"
            "<button type=\"submit\" class=\"secondary-button\">제거</button>"
            "</form>"
            "</li>"
        )
    queue_html = "".join(queue_items) or "<li class=\"queue-empty\">대기 중인 작업이 없습니다.</li>"
    return (
        "<section class=\"panel sidebar-panel\">"
        "<div class=\"panel-head\"><div><h2>현재 작업과 대기열</h2><p>현재 상태와 다음 실행 대상을 빠르게 확인합니다.</p></div></div>"
        "<div class=\"panel-body stack\">"
        "<div class=\"key-metric-grid\">"
        f"<div><span>current filekey</span><strong>{_text(current_job.get('filekey'))}</strong></div>"
        f"<div><span>현재 datasetkey</span><strong>{_text(current_job.get('datasetkey'))}</strong></div>"
        f"<div><span>현재 단계</span><strong>{_text(current_job_progress.get('detail'))}</strong></div>"
        f"<div><span>실행 설정</span><strong>{_text(launcher.get('runtime_config_path'))}</strong></div>"
        "</div>"
        f"<ul class=\"queue-list\">{queue_html}</ul>"
        "</div>"
        "</section>"
    )


def _render_system_panel(
    overview: dict,
    config_path: str,
    artifacts: dict,
    gpu: dict,
    progress: dict,
    queue_progress: dict,
) -> str:
    return (
        "<section class=\"panel sidebar-panel\">"
        "<div class=\"panel-head\"><div><h2>시스템 요약</h2><p>경로, 장치, 아티팩트 상태를 한 눈에 봅니다.</p></div></div>"
        "<div class=\"panel-body stack\">"
        "<div class=\"key-metric-grid\">"
        f"<div><span>워크스페이스</span><strong>{_text(overview.get('workspace_name'))}</strong></div>"
        f"<div><span>GPU</span><strong>{_text(gpu.get('summary'))}</strong></div>"
        f"<div><span>최고 모델</span><strong>{_text('준비됨' if artifacts.get('has_model') else '없음')}</strong></div>"
        f"<div><span>지표 파일</span><strong>{_text('준비됨' if artifacts.get('has_metrics') else '없음')}</strong></div>"
        f"<div><span>재개 방식</span><strong>{_resume_mode_label(progress.get('resume_mode') or ('full' if progress.get('resumed_from_checkpoint') else 'new'))}</strong></div>"
        f"<div><span>완료 진행률</span><strong>{_int_text(queue_progress.get('completed'))} / {_int_text(queue_progress.get('total'))}</strong></div>"
        "</div>"
        f"<div class=\"path-box\"><div class=\"path-label\">설정 파일</div><div class=\"path-value\">{_text(config_path)}</div></div>"
        f"<div class=\"path-box\"><div class=\"path-label\">워크스페이스 경로</div><div class=\"path-value\">{_text(overview.get('workspace_dir'))}</div></div>"
        "<div class=\"link-row\">"
        "<a href=\"/\">새로고침</a>"
        "<a href=\"/api/overview\">원본 JSON</a>"
        "<a href=\"/api/overview?lite=1\">요약 JSON</a>"
        "</div>"
        "</div>"
        "</section>"
    )


def _render_logs(logs: dict) -> str:
    sections = []
    for key, title, opened in (
        ("current", "현재 로그", True),
        ("latest_completed", "최근 완료 로그", False),
        ("latest_error", "최근 오류 로그", False),
    ):
        payload = logs.get(key) or {}
        open_attr = " open" if opened else ""
        sections.append(
            f"<details class=\"detail-panel\"{open_attr}>"
            f"<summary>{_text(title)}</summary>"
            "<div class=\"detail-body stack\">"
            f"<div class=\"path-box\"><div class=\"path-label\">path</div><div class=\"path-value\">{_text(payload.get('path'))}</div></div>"
            f"<pre class=\"log-box\">{_text(payload.get('tail'))}</pre>"
            "</div>"
            "</details>"
        )
    return "".join(sections)


def _render_section_nav() -> str:
    items = [
        ("overview", "개요"),
        ("training", "학습"),
        ("dataset", "데이터셋"),
        ("validation", "검증"),
        ("jobs", "작업"),
        ("logs", "로그"),
    ]
    return (
        "<nav class=\"section-nav\">"
        + "".join(f"<a href=\"#{_text(anchor)}\">{_text(label)}</a>" for anchor, label in items)
        + "</nav>"
    )


def render_dashboard_page(
    overview: dict,
    *,
    config_path: str,
    default_datasetkey: str = "",
    controls_enabled: bool = True,
    notice: str | None = None,
    notice_level: str = "info",
    refresh_seconds: int = 15,
) -> str:
    overview = overview or {}
    pipeline = overview.get("pipeline_status") or {}
    progress = overview.get("training_progress") or {}
    metrics = overview.get("metrics") or {}
    launcher = overview.get("launcher") or {}
    queue_progress = overview.get("queue_progress") or {}
    current_job_progress = overview.get("current_job_progress") or {}
    eta = overview.get("eta") or {}
    gpu = overview.get("gpu") or {}
    dataset = overview.get("dataset") or {}
    current_dataset = overview.get("current_dataset") or {}
    diagnostics = overview.get("diagnostics") or {}
    logs = overview.get("logs") or {}
    continual_state = overview.get("continual_state") or {}
    artifacts = overview.get("artifacts") or {}
    skip_report = overview.get("skip_report") or {}
    cumulative_skip_report = overview.get("cumulative_skip_report") or {}
    labels = progress.get("labels") or metrics.get("labels") or []
    final_validation = progress.get("final_validation") or metrics.get("final_validation") or {}
    completed_jobs = launcher.get("completed_jobs") or []

    raw_total = int(((dataset.get("raw") or {}).get("total") or 0))
    prepared_total = sum(
        int(((dataset.get(key) or {}).get("total") or 0))
        for key in ("prepared_train", "prepared_val", "prepared_test")
    )
    latest = progress.get("latest") or {}
    updated_at = progress.get("updated_at") or pipeline.get("updated_at") or "-"
    refresh_seconds = max(0, min(int(refresh_seconds), 300))
    meta_refresh = (
        f"<meta http-equiv=\"refresh\" content=\"{refresh_seconds};url=/\" />"
        if refresh_seconds > 0
        else ""
    )

    banners = []
    if notice:
        notice_tone = _state_tone(notice_level)
        banners.append(_render_banner(notice, notice_tone, "작업 결과" if notice_tone != "danger" else "오류"))
    warning_banner = _render_warning_banner(diagnostics)
    if warning_banner:
        banners.append(warning_banner)
    banners_html = "".join(banners)

    summary_cards = _render_summary_cards(
        [
            (_state_tone(pipeline.get("state")), "파이프라인", _state_label(pipeline.get("state")), _text(pipeline.get("message"))),
            (_state_tone(launcher.get("state")), "런처", _state_label(launcher.get("state")), _text(launcher.get("message"))),
            ("accent", "최고 F1", _float_text(progress.get("best_val_macro_f1")), f"최고 epoch {_text(progress.get('best_epoch'))}"),
            ("accent", "최근 epoch", _text(latest.get("epoch")), f"정확도 {_float_text(latest.get('val_accuracy'))} / F1 {_float_text(latest.get('val_macro_f1'))}"),
            ("neutral", "데이터셋", f"원본 {raw_total} / 전처리 {prepared_total}", f"workspace {_text(overview.get('workspace_name'))}"),
            ("neutral", "대기열", f"{_int_text(queue_progress.get('completed'))} / {_int_text(queue_progress.get('total'))}", f"대기 {_int_text(queue_progress.get('pending'))} / 실행 {_int_text(queue_progress.get('active'))}"),
            ("warn", "현재 작업", _text(current_job_progress.get("label")), f"{_int_text(current_job_progress.get('percent'))}% / 예상 {_text(eta.get('label'))}"),
            ("accent", "GPU", _text(gpu.get("summary")), _text(gpu.get("detail"))),
        ]
    )

    hero_kpis = (
        "<div class=\"hero-kpis\">"
        f"<div class=\"hero-kpi\"><span>최근 갱신</span><strong>{_text(updated_at)}</strong></div>"
        f"<div class=\"hero-kpi\"><span>학습 / 검증 샘플</span><strong>{_int_text(progress.get('train_samples'))} / {_int_text(progress.get('val_samples'))}</strong></div>"
        f"<div class=\"hero-kpi\"><span>누적 스킵</span><strong>{_int_text((skip_report.get('summary') or {}).get('total_issues'))}</strong></div>"
        f"<div class=\"hero-kpi\"><span>누적 전처리 샘플</span><strong>{_int_text(continual_state.get('prepared_train_total'))} / {_int_text(continual_state.get('prepared_val_total'))} / {_int_text(continual_state.get('prepared_test_total'))}</strong></div>"
        "</div>"
    )

    dataset_rows = []
    for key in ("raw", "train", "val", "test", "prepared_train", "prepared_val", "prepared_test"):
        info = dataset.get(key) or {}
        dataset_rows.append(
            [
                _split_label(key),
                _int_text(info.get("total"), "0"),
                _text(_join_label_counts(info), "-"),
            ]
        )

    current_dataset_rows = []
    for key in ("raw", "train", "val", "test", "prepared_train", "prepared_val", "prepared_test"):
        info = current_dataset.get(key) or {}
        current_dataset_rows.append(
            [
                _split_label(key),
                _int_text(info.get("total"), "0"),
                _text(_join_label_counts(info), "-"),
            ]
        )

    history_rows = []
    for row in list(progress.get("history") or metrics.get("history") or [])[-20:]:
        if not isinstance(row, dict):
            continue
        history_rows.append(
            [
                _text(row.get("epoch")),
                _float_text(row.get("train_loss")),
                _float_text(row.get("val_loss")),
                _float_text(row.get("val_accuracy")),
                _float_text(row.get("val_macro_f1")),
                _float_text(row.get("learning_rate"), 6),
            ]
        )

    support_map: dict[int, int] = {}
    confusion = final_validation.get("confusion_matrix") or []
    if isinstance(confusion, list):
        for index, matrix_row in enumerate(confusion):
            if isinstance(matrix_row, list):
                support_map[index] = sum(int(value or 0) for value in matrix_row)

    per_class_rows = []
    for index, row in enumerate(final_validation.get("per_class") or []):
        if not isinstance(row, dict):
            continue
        class_index = int(row.get("class_index", index) or index)
        label = labels[class_index] if 0 <= class_index < len(labels) else row.get("label") or class_index
        per_class_rows.append(
            [
                _text(label),
                _float_text(row.get("precision")),
                _float_text(row.get("recall")),
                _float_text(row.get("f1")),
                _int_text(support_map.get(class_index, 0)),
            ]
        )

    confusion_rows = []
    if isinstance(confusion, list) and confusion:
        for row_index, matrix_row in enumerate(confusion):
            if not isinstance(matrix_row, list):
                continue
            label = labels[row_index] if row_index < len(labels) else f"class_{row_index}"
            confusion_rows.append([_text(label)] + [_int_text(value, "0") for value in matrix_row])
    confusion_headers = ["actual \\ predicted"] + [_text(label) for label in labels] if labels else ["matrix"]

    recent_job_rows = []
    for job in completed_jobs[:20]:
        if not isinstance(job, dict):
            continue
        summary = job.get("result_summary") or {}
        prepared_count = sum(
            int(summary.get(key, 0) or 0)
            for key in ("prepared_train_total", "prepared_val_total", "prepared_test_total")
        )
        recent_job_rows.append(
            [
                _text(job.get("filekey")),
                _text(job.get("datasetkey")),
                _state_label(job.get("state")),
                _int_text(prepared_count, "0"),
                _text(job.get("finished_at") or job.get("started_at")),
                _text(job.get("message")),
            ]
        )

    issue_rows = []
    for issue in list((skip_report.get("issues") or []))[-20:]:
        if not isinstance(issue, dict):
            continue
        issue_rows.append(
            [
                _text(issue.get("category")),
                _text(issue.get("split")),
                _text(issue.get("video_name")),
                _text(issue.get("reason")),
                _text(issue.get("created_at")),
            ]
        )

    latest_loss = (
        f"train {_float_text(latest.get('train_loss'))} / val {_float_text(latest.get('val_loss'))}"
        if latest
        else "-"
    )
    train_distribution = progress.get("train_distribution") or metrics.get("train_distribution") or {}
    val_distribution = progress.get("val_distribution") or metrics.get("val_distribution") or {}

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {meta_refresh}
  <title>detectWarning 학습 대시보드</title>
  <style>
    :root {{
      --bg: #edf3fb;
      --bg-deep: #e4edf8;
      --panel: rgba(255, 255, 255, 0.96);
      --panel-soft: rgba(248, 251, 255, 0.94);
      --ink: #11233d;
      --muted: #5c6b80;
      --line: rgba(148, 163, 184, 0.22);
      --accent: #0f6fff;
      --accent-soft: rgba(15, 111, 255, 0.12);
      --good: #059669;
      --warn: #d97706;
      --danger: #dc2626;
      --shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
      --radius-xl: 26px;
      --radius-lg: 20px;
      --radius-md: 16px;
      --radius-sm: 12px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "Pretendard", "Apple SD Gothic Neo", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 111, 255, 0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.08), transparent 20%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 52%, var(--bg-deep) 100%);
      min-height: 100vh;
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px 24px 36px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
      gap: 18px;
      padding: 24px;
      border-radius: var(--radius-xl);
      background:
        radial-gradient(circle at top right, rgba(15, 111, 255, 0.14), transparent 32%),
        linear-gradient(135deg, rgba(255,255,255,0.98), rgba(245,249,255,0.95));
      border: 1px solid rgba(255,255,255,0.78);
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 8px 0 10px;
      font-size: 38px;
      line-height: 1;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.65;
      max-width: 760px;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .eyebrow::before {{
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: linear-gradient(135deg, #0f6fff, #0ea5e9);
    }}
    .hero-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .hero-links a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .hero-kpis {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .hero-kpi {{
      padding: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
    }}
    .hero-kpi span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .hero-kpi strong {{
      display: block;
      font-size: 14px;
      line-height: 1.5;
      word-break: break-word;
    }}
    .status-card {{
      display: grid;
      gap: 14px;
      padding: 18px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(249,252,255,0.98), rgba(244,248,252,0.95));
      border: 1px solid var(--line);
    }}
    .status-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .status-badge.good {{ background: rgba(5,150,105,0.10); color: var(--good); }}
    .status-badge.warn {{ background: rgba(217,119,6,0.10); color: var(--warn); }}
    .status-badge.danger {{ background: rgba(220,38,38,0.10); color: var(--danger); }}
    .status-badge.accent {{ background: rgba(15,111,255,0.10); color: var(--accent); }}
    .status-badge.neutral {{ background: rgba(148,163,184,0.12); color: #475569; }}
    .status-main {{
      font-size: 15px;
      line-height: 1.7;
      color: var(--muted);
    }}
    .status-main strong {{
      display: block;
      color: var(--ink);
      font-size: 28px;
      line-height: 1.1;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
    }}
    .mini-stack {{
      display: grid;
      gap: 10px;
    }}
    .mini-stack .mini-row {{
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
    }}
    .mini-row span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .mini-row strong {{
      display: block;
      font-size: 14px;
      line-height: 1.5;
      word-break: break-word;
    }}
    .banner {{
      margin-bottom: 14px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: var(--panel);
    }}
    .banner-title {{
      font-size: 13px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .banner-copy,
    .banner-list {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
    }}
    .banner-list {{
      padding-left: 18px;
    }}
    .banner-good {{ background: rgba(5,150,105,0.08); border-color: rgba(5,150,105,0.18); }}
    .banner-warn {{ background: rgba(217,119,6,0.08); border-color: rgba(217,119,6,0.18); }}
    .banner-danger {{ background: rgba(220,38,38,0.08); border-color: rgba(220,38,38,0.18); }}
    .banner-accent {{ background: rgba(15,111,255,0.08); border-color: rgba(15,111,255,0.18); }}
    .section-nav {{
      position: sticky;
      top: 12px;
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 0 0 16px;
      padding: 12px;
      border-radius: 18px;
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(148,163,184,0.16);
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 26px rgba(15,23,42,0.06);
    }}
    .section-nav a {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15,111,255,0.08);
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
      font-weight: 800;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 0.82fr) minmax(0, 1.68fr);
      gap: 18px;
      align-items: start;
    }}
    .sidebar {{
      display: grid;
      gap: 16px;
      position: sticky;
      top: 72px;
    }}
    .content {{
      display: grid;
      gap: 16px;
      min-width: 0;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .summary-card {{
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 14px 28px rgba(15,23,42,0.04);
    }}
    .summary-card-good {{ background: linear-gradient(180deg, rgba(236,253,245,0.92), rgba(255,255,255,0.96)); }}
    .summary-card-warn {{ background: linear-gradient(180deg, rgba(255,247,237,0.92), rgba(255,255,255,0.96)); }}
    .summary-card-danger {{ background: linear-gradient(180deg, rgba(254,242,242,0.92), rgba(255,255,255,0.96)); }}
    .summary-card-accent {{ background: linear-gradient(180deg, rgba(239,246,255,0.92), rgba(255,255,255,0.96)); }}
    .summary-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 10px;
    }}
    .summary-value {{
      font-size: 28px;
      font-weight: 900;
      line-height: 1.05;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
      word-break: break-word;
    }}
    .summary-copy {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      word-break: break-word;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 14px 28px rgba(15,23,42,0.05);
      overflow: hidden;
    }}
    .panel-head {{
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,251,255,0.92));
    }}
    .panel-head h2 {{
      margin: 0 0 4px;
      font-size: 20px;
      line-height: 1.15;
      letter-spacing: -0.03em;
    }}
    .panel-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .panel-body {{
      padding: 18px 20px 20px;
    }}
    .sidebar-panel .panel-body {{
      padding-top: 16px;
    }}
    .stack {{
      display: grid;
      gap: 12px;
    }}
    .stack-form {{
      display: grid;
      gap: 12px;
    }}
    .field-label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    input[type="text"],
    input[type="password"],
    textarea {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.28);
      background: #ffffff;
      color: var(--ink);
      font: inherit;
    }}
    textarea {{
      resize: vertical;
      min-height: 130px;
      font-family: "Consolas", "SFMono-Regular", monospace;
      line-height: 1.6;
    }}
    .primary-button,
    .secondary-button {{
      width: 100%;
      border: none;
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      font-weight: 900;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .primary-button {{
      background: linear-gradient(135deg, #0f6fff, #1d4ed8);
      color: white;
      box-shadow: 0 12px 24px rgba(15,111,255,0.22);
    }}
    .secondary-button {{
      background: #eef4fb;
      color: var(--ink);
      border: 1px solid rgba(148,163,184,0.2);
    }}
    .secondary-button-warn {{
      background: #fff2de;
      color: #a25a00;
    }}
    .secondary-button-danger {{
      background: #ffe6e6;
      color: #9b1c1c;
    }}
    .primary-button:hover,
    .secondary-button:hover {{
      transform: translateY(-1px);
    }}
    .action-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .key-metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .key-metric-grid > div {{
      padding: 12px 14px;
      border-radius: 14px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
    }}
    .key-metric-grid span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .key-metric-grid strong {{
      display: block;
      font-size: 14px;
      line-height: 1.55;
      word-break: break-word;
    }}
    .queue-list {{
      display: grid;
      gap: 10px;
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .queue-item,
    .queue-empty {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-soft);
    }}
    .queue-main {{
      min-width: 0;
    }}
    .queue-main strong {{
      display: block;
      margin-bottom: 4px;
      word-break: break-word;
    }}
    .queue-copy,
    .muted-block {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      word-break: break-word;
    }}
    .path-box {{
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--panel-soft);
    }}
    .path-label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .path-value {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 12px;
      line-height: 1.65;
      word-break: break-all;
    }}
    .content-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .table-wrap {{
      overflow: auto;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(255,255,255,0.7);
    }}
    .data-table {{
      width: 100%;
      min-width: 680px;
      border-collapse: collapse;
    }}
    .data-table-compact {{
      min-width: 560px;
    }}
    .data-table th,
    .data-table td {{
      padding: 11px 10px;
      border-bottom: 1px solid rgba(148,163,184,0.14);
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    .data-table thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fbff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .data-table tbody tr:nth-child(even) {{
      background: rgba(248,251,255,0.6);
    }}
    .empty-cell {{
      text-align: center;
      color: var(--muted);
      padding: 18px 10px;
    }}
    .detail-panel {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      overflow: hidden;
    }}
    .detail-panel + .detail-panel {{
      margin-top: 12px;
    }}
    .detail-panel summary {{
      cursor: pointer;
      list-style: none;
      padding: 16px 18px;
      font-size: 15px;
      font-weight: 900;
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,251,255,0.92));
    }}
    .detail-body {{
      padding: 0 18px 18px;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .mini-panel {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel-soft);
      padding: 14px;
    }}
    .mini-panel h3 {{
      margin: 0 0 10px;
      font-size: 15px;
    }}
    .log-box {{
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      background: linear-gradient(180deg, #0f172a, #111827);
      color: #e2e8f0;
      min-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 12px;
      line-height: 1.7;
    }}
    code {{
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 12px;
    }}
    @media (max-width: 1280px) {{
      .summary-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
      }}
    }}
    @media (max-width: 860px) {{
      .page {{
        padding: 16px 16px 28px;
      }}
      .hero {{
        grid-template-columns: 1fr;
        padding: 18px;
      }}
      .hero h1 {{
        font-size: 30px;
      }}
      .hero-kpis,
      .summary-grid,
      .content-row,
      .action-grid,
      .key-metric-grid,
      .detail-grid {{
        grid-template-columns: 1fr;
      }}
      .section-nav {{
        top: 8px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero" id="overview">
      <div>
        <div class="eyebrow">안정 모드 대시보드</div>
        <h1>detectWarning 학습 대시보드</h1>
        <p>브라우저 스크립트에 의존하지 않고 최신 학습 결과를 바로 읽을 수 있게 다시 구성했습니다. 위쪽은 현재 상태를 빠르게 판단하는 영역이고, 아래쪽은 학습 결과와 로그를 차례대로 읽도록 정리했습니다.</p>
        <div class="hero-links">
          <a href="/">새로고침</a>
          <a href="/api/overview">원본 JSON</a>
          <a href="/api/overview?lite=1">요약 JSON</a>
        </div>
        {hero_kpis}
      </div>
      <aside class="status-card">
        <div class="status-badge {_state_tone(pipeline.get('state'))}">{_state_label(pipeline.get('state'))}</div>
        <div class="status-main">
          <strong>{_text(pipeline.get('message'))}</strong>
          현재 파이프라인 상태를 가장 먼저 보여줍니다. 아래 카드들은 성능, 큐, 데이터셋 상태를 빠르게 훑어볼 수 있도록 배치했습니다.
        </div>
        <div class="mini-stack">
          <div class="mini-row"><span>최근 손실값</span><strong>{_text(latest_loss)}</strong></div>
          <div class="mini-row"><span>런처 상태</span><strong>{_state_label(launcher.get('state'))} / {_text(launcher.get('message'))}</strong></div>
          <div class="mini-row"><span>GPU 상세</span><strong>{_text(gpu.get('detail'))}</strong></div>
        </div>
      </aside>
    </section>

    {banners_html}

    {_render_section_nav()}

    <div class="layout">
      <aside class="sidebar">
        {_render_actions_panel(default_datasetkey, controls_enabled)}
        {_render_queue_panel(launcher, current_job_progress)}
        {_render_system_panel(overview, config_path, artifacts, gpu, progress, queue_progress)}
      </aside>

      <main class="content">
        <section class="summary-grid">
          {summary_cards}
        </section>

        <section class="panel" id="training">
          <div class="panel-head"><div><h2>학습 개요</h2><p>지금 확인해야 할 핵심 학습 정보를 먼저 모았습니다.</p></div></div>
          <div class="panel-body">
            <div class="content-row">
              <div class="mini-panel">
                <h3>학습 상태</h3>
                <div class="key-metric-grid">
                  <div><span>epochs</span><strong>{_int_text(progress.get('epochs_completed'))} / {_int_text(progress.get('epochs_total'))}</strong></div>
                  <div><span>최고 epoch</span><strong>{_text(progress.get('best_epoch'))}</strong></div>
                  <div><span>최고 macro F1</span><strong>{_float_text(progress.get('best_val_macro_f1'))}</strong></div>
                  <div><span>최근 epoch</span><strong>{_text(latest.get('epoch'))}</strong></div>
                  <div><span>최근 정확도</span><strong>{_float_text(latest.get('val_accuracy'))}</strong></div>
                  <div><span>최근 F1</span><strong>{_float_text(latest.get('val_macro_f1'))}</strong></div>
                </div>
              </div>
              <div class="mini-panel">
                <h3>데이터 / 분포</h3>
                <div class="key-metric-grid">
                  <div><span>원본 / 전처리</span><strong>{raw_total} / {prepared_total}</strong></div>
                  <div><span>학습 / 검증 샘플</span><strong>{_int_text(progress.get('train_samples'))} / {_int_text(progress.get('val_samples'))}</strong></div>
                  <div><span>학습 분포</span><strong>{_text(train_distribution.get('severity'))}</strong></div>
                  <div><span>검증 분포</span><strong>{_text(val_distribution.get('severity'))}</strong></div>
                  <div><span>현재 스킵</span><strong>{_int_text((skip_report.get('summary') or {}).get('total_issues'))}</strong></div>
                  <div><span>누적 스킵</span><strong>{_int_text((cumulative_skip_report.get('summary') or {}).get('total_issues'))}</strong></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="content-row" id="dataset">
          <article class="panel">
            <div class="panel-head"><div><h2>누적 데이터셋</h2><p>cumulative manifests 기준입니다.</p></div></div>
            <div class="panel-body">
              {_render_table(["split", "total", "labels"], dataset_rows, "누적 데이터셋이 없습니다.")}
            </div>
          </article>
          <article class="panel">
            <div class="panel-head"><div><h2>현재 작업 데이터셋</h2><p>current manifests 기준입니다.</p></div></div>
            <div class="panel-body">
              {_render_table(["split", "total", "labels"], current_dataset_rows, "현재 작업 데이터셋이 없습니다.")}
            </div>
          </article>
        </section>

        <section class="panel">
          <div class="panel-head"><div><h2>에폭 기록</h2><p>최근 20개 epoch를 빠르게 훑을 수 있게 정리했습니다.</p></div></div>
          <div class="panel-body">
            {_render_table(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "learning_rate"], history_rows, "학습 history가 없습니다.")}
          </div>
        </section>

        <section class="content-row" id="validation">
          <article class="panel">
            <div class="panel-head"><div><h2>클래스별 검증 지표</h2><p>precision / recall / f1 / support를 바로 비교할 수 있습니다.</p></div></div>
            <div class="panel-body">
              {_render_table(["label", "precision", "recall", "f1", "support"], per_class_rows, "클래스별 지표가 없습니다.")}
            </div>
          </article>
          <article class="panel">
            <div class="panel-head"><div><h2>혼동 행렬</h2><p>최종 검증 결과를 실제 행렬 형태로 보여줍니다.</p></div></div>
            <div class="panel-body">
              {_render_table(confusion_headers, confusion_rows, "confusion matrix가 없습니다.")}
            </div>
          </article>
        </section>

        <section class="content-row" id="jobs">
          <article class="panel">
            <div class="panel-head"><div><h2>최근 작업 이력</h2><p>무슨 filekey가 어떤 상태로 끝났는지 바로 확인할 수 있습니다.</p></div></div>
            <div class="panel-body">
              {_render_table(["filekey", "datasetkey", "state", "prepared", "finished", "message"], recent_job_rows, "최근 작업 이력이 없습니다.")}
            </div>
          </article>
          <article class="panel">
            <div class="panel-head"><div><h2>문제 영상 이력</h2><p>skip / broken 원인을 최근 순서대로 봅니다.</p></div></div>
            <div class="panel-body">
              {_render_table(["category", "split", "video", "reason", "created_at"], issue_rows, "최근 문제 영상 기록이 없습니다.")}
            </div>
          </article>
        </section>

        <section class="panel" id="logs">
          <div class="panel-head"><div><h2>로그</h2><p>기본은 현재 로그를 펼쳐 놓고, 나머지는 필요할 때만 펼쳐 보도록 정리했습니다.</p></div></div>
          <div class="panel-body stack">
            {_render_logs(logs)}
          </div>
        </section>

        {_render_diagnostics_details(diagnostics)}
      </main>
    </div>
  </div>
</body>
</html>"""

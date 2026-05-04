from __future__ import annotations

from functools import lru_cache
from html import escape

STATE_LABELS = {
    "completed": "완료",
    "completed_warning": "경고 포함 완료",
    "data_ready": "데이터 준비",
    "deferred": "학습 대기",
    "waiting_for_data": "데이터 누적 대기",
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

SPLIT_LABELS = {
    "raw": "원본",
    "train": "학습",
    "val": "검증",
    "test": "테스트",
    "prepared_train": "전처리 학습",
    "prepared_val": "전처리 검증",
    "prepared_test": "전처리 테스트",
}

RESUME_MODE_LABELS = {
    "full": "이전 체크포인트 이어서",
    "new": "새로 시작",
    "none": "새로 시작",
}

SPLIT_SEQUENCE = (
    "raw",
    "train",
    "val",
    "test",
    "prepared_train",
    "prepared_val",
    "prepared_test",
)

SECTION_NAV_ITEMS = (
    ("overview", "개요"),
    ("diagnosis", "Diagnosis"),
    ("training", "학습"),
    ("dataset", "데이터셋"),
    ("validation", "검증"),
    ("jobs", "작업"),
    ("logs", "로그"),
)


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
    if normalized in {"running", "queued", "data_ready", "deferred", "waiting_for_data"}:
        return "accent"
    if normalized in {"completed_warning", "paused", "prepare", "download", "train", "warning"}:
        return "warn"
    if normalized in {"error", "aborted", "offline"}:
        return "danger"
    return "neutral"


def _state_label(state: str | None) -> str:
    normalized = str(state or "").strip().lower()
    if normalized in STATE_LABELS:
        return STATE_LABELS[normalized]
    return _text(state)


def _split_label(name: str | None) -> str:
    normalized = str(name or "").strip().lower()
    if normalized in SPLIT_LABELS:
        return SPLIT_LABELS[normalized]
    return _text(name)


def _resume_mode_label(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in RESUME_MODE_LABELS:
        return RESUME_MODE_LABELS[normalized]
    return _text(value)


def _render_table(headers: list[str], rows: list[list[str]], empty_message: str, *, compact: bool = False) -> str:
    header_html = "".join(f"<th>{_text(header)}</th>" for header in headers)
    if rows:
        body_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
    else:
        body_html = f"<tr><td colspan=\"{len(headers)}\" class=\"empty-cell\">{_text(empty_message)}</td></tr>"
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


def _build_summary_card_items(
    *,
    overview: dict,
    pipeline: dict,
    launcher: dict,
    progress: dict,
    queue_progress: dict,
    current_job_progress: dict,
    eta: dict,
    gpu: dict,
    raw_total: int,
    prepared_total: int,
) -> list[tuple[str, str, str, str]]:
    latest = progress.get("latest") or {}
    finished_jobs = int(queue_progress.get("completed") or 0) + int(queue_progress.get("prepared") or 0)
    return [
        (_state_tone(pipeline.get("state")), "파이프라인", _state_label(pipeline.get("state")), _text(pipeline.get("message"))),
        (_state_tone(launcher.get("state")), "런처", _state_label(launcher.get("state")), _text(launcher.get("message"))),
        ("accent", "최고 F1", _float_text(progress.get("best_val_macro_f1")), f"최고 epoch {_text(progress.get('best_epoch'))}"),
        ("accent", "최근 epoch", _text(latest.get("epoch")), f"정확도 {_float_text(latest.get('val_accuracy'))} / F1 {_float_text(latest.get('val_macro_f1'))}"),
        ("neutral", "데이터셋", f"원본 {raw_total} / 전처리 {prepared_total}", f"workspace {_text(overview.get('workspace_name'))}"),
        ("neutral", "대기열", f"{_int_text(finished_jobs)} / {_int_text(queue_progress.get('total'))}", f"준비 {_int_text(queue_progress.get('prepared'))} / 대기 {_int_text(queue_progress.get('pending'))} / 실행 {_int_text(queue_progress.get('active'))}"),
        ("warn", "현재 작업", _text(current_job_progress.get("label")), f"{_int_text(current_job_progress.get('percent'))}% / 예상 {_text(eta.get('label'))}"),
        ("accent", "GPU", _text(gpu.get("summary")), _text(gpu.get("detail"))),
    ]


def _render_summary_grid(cards: list[tuple[str, str, str, str]], *, element_id: str | None = None) -> str:
    panel_id = element_id or "queue-panel"
    id_attr = f" id=\"{_text(panel_id, '')}\""
    return f"<section{id_attr} class=\"summary-grid\">{_render_summary_cards(cards)}</section>"


def _render_banner(message: str, tone: str, title: str) -> str:
    return (
        f"<section class=\"banner banner-{_text(tone, 'neutral')}\">"
        f"<div class=\"banner-title\">{_text(title)}</div>"
        f"<div class=\"banner-copy\">{_text(message)}</div>"
        "</section>"
    )


def _render_warning_banner(diagnostics: dict) -> str:
    warnings = list(diagnostics.get("warnings") or [])
    if not warnings:
        return ""
    items = "".join(f"<li>{_text(item)}</li>" for item in warnings)
    return (
        "<section class=\"banner banner-warn\">"
        "<div class=\"banner-title\">상태 진단</div>"
        "<div class=\"banner-copy\">"
        "<ul class=\"banner-list\">"
        f"{items}"
        "</ul>"
        "</div>"
        "</section>"
    )


def _render_diagnostics_details(diagnostics: dict) -> str:
    if not diagnostics:
        return ""
    pipeline = diagnostics.get("pipeline") or {}
    files = diagnostics.get("files") or {}

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
        "<summary class=\"detail-summary\">진단 상세 보기</summary>"
        "<div class=\"detail-body\">"
        "<div class=\"detail-grid\">"
        "<section class=\"mini-panel\">"
        "<h3>파이프라인 진단</h3>"
        + _render_table(["kind", "value"], pipeline_rows, "파이프라인 진단 정보가 없습니다.", compact=True)
        + "</section>"
        "<section class=\"mini-panel\">"
        "<h3>파일 상태</h3>"
        + _render_table(["file", "exists", "updated_at", "size", "path"], file_rows, "진단 파일 정보가 없습니다.", compact=True)
        + "</section>"
        "</div>"
        "</div>"
        "</details>"
    )


def _render_actions_panel(
    default_datasetkey: str,
    controls_enabled: bool,
    controls_notice: str | None = None,
) -> str:
    if not controls_enabled:
        notice = controls_notice or "이 설정에서는 브라우저에서 직접 실행을 지원하지 않습니다."
        detail = (
            "공유 링크에서는 진행 상황만 확인할 수 있습니다."
            if controls_notice
            else "dataset_source가 <code>aihub_shell</code>일 때만 filekey 큐 제어를 사용할 수 있습니다."
        )
        return (
            "<section class=\"panel sidebar-panel\">"
            f"<div class=\"panel-head\"><div><h2>작업 제어</h2><p>{_text(notice)}</p></div></div>"
            "<div class=\"panel-body\">"
            f"<p class=\"muted-block\">{detail}</p>"
            "</div>"
            "</section>"
        )

    return (
        "<section class=\"panel sidebar-panel\">"
        "<div class=\"panel-head\"><div><h2>작업 제어</h2><p>datasetkey로 파일 목록을 불러온 뒤 선택한 filekey만 큐에 추가합니다.</p></div></div>"
        "<div class=\"panel-body stack\">"
        "<form method=\"post\" action=\"/actions/start\" class=\"stack-form\" id=\"start-job-form\">"
        "<label class=\"field-label\">datasetkey"
        f"<input type=\"text\" id=\"datasetkey-input\" name=\"datasetkey\" value=\"{_text(default_datasetkey, '')}\" placeholder=\"예: 171\" data-persist-key=\"dashboard.datasetkey\" />"
        "</label>"
        "<label class=\"field-label\">AIHub API 키"
        "<input type=\"password\" id=\"api-key-input\" name=\"api_key\" value=\"\" placeholder=\"필요할 때만 입력\" data-persist-key=\"dashboard.api_key\" />"
        "</label>"
        "<div class=\"filekey-lookup\" id=\"filekey-lookup-panel\">"
        "<div class=\"filekey-lookup-head\">"
        "<div><strong>AIHub 파일 목록</strong><span id=\"filekey-lookup-status\">datasetkey 대기 중</span></div>"
        "<button type=\"button\" class=\"secondary-button secondary-button-compact\" id=\"filekey-lookup-button\">목록 조회</button>"
        "</div>"
        "<div class=\"filekey-lookup-summary\" id=\"filekey-lookup-summary\"></div>"
        "<div class=\"filekey-lookup-zip-groups\" id=\"filekey-lookup-zip-groups\"></div>"
        "<div class=\"filekey-lookup-filters\" id=\"filekey-lookup-filters\"></div>"
        "<div class=\"filekey-lookup-groups\" id=\"filekey-lookup-groups\"></div>"
        "<div class=\"filekey-lookup-actions\">"
        "<button type=\"button\" class=\"secondary-button secondary-button-compact\" id=\"filekey-select-trainable-button\">학습 가능 전체 선택</button>"
        "<button type=\"button\" class=\"secondary-button secondary-button-compact\" id=\"filekey-clear-selection-button\">선택 해제</button>"
        "</div>"
        "<div class=\"filekey-lookup-list\" id=\"filekey-lookup-list\"></div>"
        "</div>"
        "<input type=\"hidden\" id=\"filekeys-input\" name=\"filekeys\" value=\"\" />"
        "<button type=\"submit\" class=\"primary-button\">선택한 filekey 큐 시작 / 추가</button>"
        "<button type=\"submit\" class=\"secondary-button\" name=\"auto_enqueue_next\" value=\"1\">outside 추천 자동 시작</button>"
        "</form>"
        "<div class=\"action-grid\">"
        "<form method=\"post\" action=\"/actions/start\"><input type=\"hidden\" name=\"resume_only\" value=\"1\" /><input type=\"hidden\" name=\"auto_enqueue_next\" value=\"1\" /><button type=\"submit\" class=\"secondary-button\">outside 추천 자동 시작 재개</button></form>"
        "<form method=\"post\" action=\"/actions/pause\"><button type=\"submit\" class=\"secondary-button secondary-button-warn\">현재 작업 후 중지</button></form>"
        "<form method=\"post\" action=\"/actions/force-stop\"><button type=\"submit\" class=\"secondary-button secondary-button-danger\">지금 중단</button></form>"
        "<form method=\"post\" action=\"/actions/reset\"><button type=\"submit\" class=\"secondary-button secondary-button-danger\">처음부터 다시 시작</button></form>"
        "</div>"
        "</div>"
        "</section>"
    )


def _render_hero_kpis(
    *,
    updated_at: str,
    progress: dict,
    skip_report: dict,
    continual_state: dict,
    element_id: str | None = None,
) -> str:
    id_attr = f" id=\"{_text(element_id, '')}\"" if element_id else ""
    return (
        f"<div{id_attr} class=\"hero-kpis\">"
        f"<div class=\"hero-kpi\"><span>최종 갱신</span><strong>{_text(updated_at)}</strong></div>"
        f"<div class=\"hero-kpi\"><span>학습 / 검증 샘플</span><strong>{_int_text(progress.get('train_samples'))} / {_int_text(progress.get('val_samples'))}</strong></div>"
        f"<div class=\"hero-kpi\"><span>현재 스킵</span><strong>{_int_text((skip_report.get('summary') or {}).get('total_issues'))}</strong></div>"
        f"<div class=\"hero-kpi\"><span>누적 전처리 샘플</span><strong>{_int_text(continual_state.get('prepared_train_total'))} / {_int_text(continual_state.get('prepared_val_total'))} / {_int_text(continual_state.get('prepared_test_total'))}</strong></div>"
        "</div>"
    )


def _render_hero_status_card(
    *,
    pipeline: dict,
    launcher: dict,
    gpu: dict,
    latest_loss: str,
    element_id: str | None = None,
) -> str:
    id_attr = f" id=\"{_text(element_id, '')}\"" if element_id else ""
    return (
        f"<aside{id_attr} class=\"status-card\">"
        f"<div class=\"status-badge {_state_tone(pipeline.get('state'))}\">{_state_label(pipeline.get('state'))}</div>"
        "<div class=\"status-main\">"
        f"<strong>{_text(pipeline.get('message'))}</strong>"
        "현재 파이프라인의 상태를 가장 먼저 보여줍니다. 아래 카드는 성능, 큐, 데이터셋 상태를 빠르게 훑어볼 수 있도록 배치했습니다."
        "</div>"
        "<div class=\"mini-stack\">"
        f"<div class=\"mini-row\"><span>최신 검증 loss</span><strong>{_text(latest_loss)}</strong></div>"
        f"<div class=\"mini-row\"><span>런처 상태</span><strong>{_state_label(launcher.get('state'))} / {_text(launcher.get('message'))}</strong></div>"
        f"<div class=\"mini-row\"><span>GPU 상세</span><strong>{_text(gpu.get('detail'))}</strong></div>"
        "</div>"
        "</aside>"
    )


def _render_queue_panel(
    launcher: dict,
    current_job_progress: dict,
    queue_progress: dict | None = None,
    *,
    element_id: str | None = None,
) -> str:
    pending_jobs = launcher.get("pending_jobs") or []
    current_job = launcher.get("current_job") or {}
    queue_progress = queue_progress or {}
    try:
        percent = max(0.0, min(100.0, float(current_job_progress.get("percent") or 0.0)))
    except (TypeError, ValueError):
        percent = 0.0
    progress_label = _text(current_job_progress.get("label"), "대기 중")
    progress_detail = _text(current_job_progress.get("detail"), "대기 중인 작업이 없으면 여기에 현재 작업 단계가 표시됩니다.")
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
    id_attr = f" id=\"{_text(element_id, '')}\"" if element_id else ""

    return (
        f"<section{id_attr} class=\"panel sidebar-panel\">"
        "<div class=\"panel-head\"><div><h2>현재 작업과 대기열</h2><p>지금 상태와 다음 실행 대상을 빠르게 확인합니다.</p></div></div>"
        "<div class=\"panel-body stack\">"
        "<div class=\"key-metric-grid\">"
        f"<div><span>현재 datasetkey</span><strong>{_text(current_job.get('datasetkey'))}</strong></div>"
        f"<div><span>현재 단계</span><strong>{_text(current_job_progress.get('detail'))}</strong></div>"
        f"<div class=\"key-metric-wide\"><span>실행 설정</span><strong class=\"path-value-inline\">{_text(launcher.get('runtime_config_path'))}</strong></div>"
        "</div>"
        "<div class=\"queue-summary-grid\">"
        f"<div class=\"queue-summary-tile\"><span>실행 중</span><strong>{_int_text(queue_progress.get('active'))}</strong></div>"
        f"<div class=\"queue-summary-tile\"><span>대기</span><strong>{_int_text(queue_progress.get('pending'))}</strong></div>"
        f"<div class=\"queue-summary-tile\"><span>완료</span><strong>{_int_text(queue_progress.get('completed'))}</strong></div>"
        f"<div class=\"queue-summary-tile\"><span>실패</span><strong>{_int_text(queue_progress.get('failed'))}</strong></div>"
        "</div>"
        "<div class=\"queue-progress-card\">"
        "<div class=\"queue-progress-head\">"
        "<div class=\"queue-progress-copy\">"
        f"<span>{progress_label}</span>"
        f"<strong>{_text(current_job.get('filekey'), '현재 filekey 없음')}</strong>"
        "</div>"
        f"<div class=\"queue-progress-percent\">{percent:.0f}%</div>"
        "</div>"
        "<div class=\"queue-progress-bar\">"
        f"<span class=\"queue-progress-fill\" style=\"width:{percent:.1f}%\"></span>"
        "</div>"
        f"<div class=\"queue-copy\">{progress_detail}</div>"
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
    *,
    element_id: str | None = None,
) -> str:
    id_attr = f" id=\"{_text(element_id, '')}\"" if element_id else ""
    return (
        f"<section{id_attr} class=\"panel sidebar-panel\">"
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


def _insight_tone(level: str | None) -> str:
    normalized = str(level or "").strip().lower()
    if normalized in {"good", "normal", "improving"}:
        return "good"
    if normalized in {"critical", "danger"}:
        return "danger"
    if normalized in {"watch", "warning", "warn"}:
        return "warn"
    return "neutral"


def _insight_level_label(level: str | None) -> str:
    normalized = str(level or "").strip().lower()
    labels = {
        "good": "정상",
        "normal": "정상",
        "improving": "개선",
        "watch": "주의",
        "warning": "경고",
        "warn": "경고",
        "critical": "심각",
        "danger": "심각",
    }
    return labels.get(normalized, _text(level, "정보"))


def _render_insight_badges(badges: list[dict]) -> str:
    if not badges:
        return ""
    return (
        "<div class=\"insight-badge-row\">"
        + "".join(
            (
                f"<span class=\"insight-badge {_insight_tone(badge.get('level'))}\">"
                f"<strong>{_text(badge.get('label'))}</strong>"
                f"<em>{_text(_insight_level_label(badge.get('level')))}</em>"
                f"<small>{_text(badge.get('detail'), '')}</small>"
                "</span>"
            )
            for badge in badges[:8]
            if isinstance(badge, dict)
        )
        + "</div>"
    )


def _render_insight_items(items: list[dict], empty_text: str, *, limit: int = 6) -> str:
    normalized_items = [item for item in items if isinstance(item, dict)]
    if not normalized_items:
        return f"<div class=\"chart-empty\">{_text(empty_text)}</div>"
    rows = []
    for item in normalized_items[:limit]:
        tone = _insight_tone(item.get("level"))
        rows.append(
            (
                f"<li class=\"insight-item {tone}\">"
                "<div>"
                f"<strong>{_text(item.get('title') or item.get('label'))}</strong>"
                f"<p>{_text(item.get('message') or item.get('detail'))}</p>"
                "</div>"
                f"<span class=\"status-badge {tone}\">{_text(_insight_level_label(item.get('level')))}</span>"
                "</li>"
            )
        )
    return "<ul class=\"insight-list\">" + "".join(rows) + "</ul>"


def _render_insight_group(title: str, items: list[dict], empty_text: str, *, limit: int = 6) -> str:
    return (
        "<article class=\"insight-card\">"
        f"<h3>{_text(title)}</h3>"
        f"{_render_insight_items(items, empty_text, limit=limit)}"
        "</article>"
    )


def _render_insights_panel(insights: dict | None) -> str:
    insights = insights if isinstance(insights, dict) else {}
    has_any = any(
        insights.get(key)
        for key in (
            "summary",
            "risk_badges",
            "diagnostics",
            "metric_notes",
            "class_insights",
            "confusion_insights",
            "data_quality_insights",
            "trend_insights",
            "recommendations",
        )
    )
    overall_tone = _insight_tone(insights.get("overall_level"))
    if not has_any:
        body = "<div class=\"chart-empty\">아직 해석할 학습 지표가 없습니다. 학습 history 또는 validation 결과가 생성되면 자동 진단이 표시됩니다.</div>"
    else:
        body = f"""
          {_render_insight_badges(insights.get('risk_badges') or [])}
          <div class="insight-grid">
            {_render_insight_group("모델 상태 요약", insights.get('summary') or [], "요약할 진단 결과가 아직 없습니다.", limit=5)}
            {_render_insight_group("지표별 짧은 해석", insights.get('metric_notes') or [], "표시할 핵심 지표 해석이 없습니다.", limit=5)}
            {_render_insight_group("과적합 / 일반화 / 불균형", insights.get('diagnostics') or [], "감지된 구조적 위험 신호가 없습니다.", limit=6)}
            {_render_insight_group("클래스별 성능 해석", insights.get('class_insights') or [], "class report가 없거나 감지된 class별 위험 신호가 없습니다.", limit=6)}
            {_render_insight_group("Confusion Matrix 해석", insights.get('confusion_insights') or [], "confusion matrix가 없거나 뚜렷한 혼동 패턴이 없습니다.", limit=6)}
            {_render_insight_group("데이터 품질 / 스킵 해석", insights.get('data_quality_insights') or [], "스킵/품질 통계에서 큰 위험 신호가 감지되지 않았습니다.", limit=6)}
            {_render_insight_group("학습 추세 해석", insights.get('trend_insights') or [], "epoch history가 부족하거나 뚜렷한 추세 신호가 없습니다.", limit=5)}
            {_render_insight_group("다음 개선 권장사항", insights.get('recommendations') or [], "현재 지표 기준으로 즉시 권장할 추가 조치가 없습니다.", limit=8)}
          </div>
        """
    return f"""
      <section class="panel insight-panel insight-panel-{overall_tone}" id="diagnosis">
        <div class="panel-head">
          <div>
            <h2>AI Training Diagnosis Summary</h2>
            <p>accuracy, macro F1, loss, class report, confusion matrix, skip 통계를 rule-based로 해석합니다.</p>
          </div>
          <span class="status-badge {overall_tone}">{_text(_insight_level_label(insights.get('overall_level')))}</span>
        </div>
        <div class="panel-body">
          {body}
        </div>
      </section>
    """


def _render_main_live_sections(
    *,
    summary_cards: str,
    insights: dict,
    performance_chart_html: str,
    loss_chart_html: str,
    distribution_chart_html: str,
    progress: dict,
    latest: dict,
    raw_total: int,
    prepared_total: int,
    train_distribution: dict,
    val_distribution: dict,
    skip_report: dict,
    cumulative_skip_report: dict,
    dataset_rows: list[dict[str, str]],
    current_dataset_rows: list[dict[str, str]],
    history_rows: list[dict[str, str]],
    per_class_rows: list[dict[str, str]],
    confusion_matrix_html: str,
    recent_job_rows: list[dict[str, str]],
    issue_rows: list[dict[str, str]],
    element_id: str | None = None,
) -> str:
    id_attr = f" id=\"{_text(element_id, '')}\"" if element_id else ""
    return f"""
        <div{id_attr}>
          <section class="summary-grid" id="summary-grid">
            {summary_cards}
          </section>

          {_render_insights_panel(insights)}

          <section class="chart-grid" id="training-charts">
            {performance_chart_html}
            {loss_chart_html}
            {distribution_chart_html}
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

          <section class="panel" id="epoch-history">
            <div class="panel-head"><div><h2>에폭 기록</h2><p>최근 20개 epoch를 빠르게 훑을 수 있게 정리했습니다.</p></div></div>
            <div class="panel-body">
              {_render_table(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "learning_rate"], history_rows, "학습 history가 없습니다.")}
            </div>
          </section>

          <section class="content-row" id="validation">
            <article class="panel panel-span-full">
              <div class="panel-head"><div><h2>클래스별 지표</h2><p>최종 검증 결과를 클래스별로 바로 읽을 수 있게 정리했습니다.</p></div></div>
              <div class="panel-body">
                {_render_table(["label", "precision", "recall", "f1", "support"], per_class_rows, "클래스별 지표가 없습니다.")}
              </div>
            </article>
          </section>

          <section class="chart-grid" id="validation-matrix">
            {confusion_matrix_html}
          </section>

          <section class="content-row" id="jobs">
            <article class="panel">
              <div class="panel-head"><div><h2>최근 작업 이력</h2><p>최근 완료 또는 중단된 작업을 위에서부터 보여줍니다.</p></div></div>
              <div class="panel-body">
                {_render_table(["filekey", "datasetkey", "state", "prepared", "finished", "message"], recent_job_rows, "최근 작업 이력이 없습니다.")}
              </div>
            </article>
            <article class="panel">
              <div class="panel-head"><div><h2>현재 스킵 이슈</h2><p>이번 작업에서 건너뛴 항목만 따로 모아 보여줍니다.</p></div></div>
              <div class="panel-body">
                {_render_table(["type", "name", "count"], _build_skip_summary_rows(skip_report), "표시할 현재 스킵 요약이 없습니다.", compact=True)}
                {_render_table(["split", "label", "video", "reason", "valid_frames", "keypoints", "recovery"], issue_rows, "표시할 현재 스킵 이슈가 없습니다.")}
              </div>
            </article>
          </section>
        </div>
    """


def _render_logs(logs: dict) -> str:
    sections = []
    for key, title, opened in (
        ("current", "현재 로그", True),
        ("latest_completed", "최근 완료 로그", False),
        ("latest_error", "최근 오류 로그", False),
    ):
        payload = logs.get(key) or {}
        body = _text(payload.get("tail"), "표시할 로그가 없습니다.")
        open_attr = " open" if opened else ""
        sections.append(
            f"<details class=\"log-card\" data-log-key=\"{_text(key)}\"{open_attr}>"
            f"<summary>{_text(title)}</summary>"
            "<div class=\"detail-body\">"
            f"<div class=\"muted-block\">filekey {_text(payload.get('filekey'))} / datasetkey {_text(payload.get('datasetkey'))}</div>"
            f"<pre class=\"log-box\">{body}</pre>"
            "</div>"
            "</details>"
        )
    return "".join(sections)


def _render_logs_stack(logs: dict, *, element_id: str | None = None) -> str:
    id_attr = f" id=\"{_text(element_id, '')}\"" if element_id else ""
    return f"<div{id_attr} class=\"log-stack\">{_render_logs(logs)}</div>"


@lru_cache(maxsize=1)
def _render_section_nav() -> str:
    return (
        "<nav class=\"section-nav\">"
        + "".join(f"<a href=\"#{_text(anchor)}\">{_text(label)}</a>" for anchor, label in SECTION_NAV_ITEMS)
        + "</nav>"
    )


@lru_cache(maxsize=1)
def _render_live_refresh_script() -> str:
    return """
<script>
(function () {
  if (!window.fetch) {
    return;
  }

  var timer = null;
  var inflight = false;
  var delayMs = 1200;
  var disposed = false;
  var noticeTimer = null;
  var filekeyLookupTimer = null;
  var filekeyLookupAbort = null;
  var filekeyLookupRequestId = 0;
  var lastFilekeyLookupSignature = '';
  var latestFilekeyLookupPayload = null;
  var filekeyViewFilters = { outside: true, inside: true, other: true };
  var lastOverviewRevision = document.documentElement.getAttribute('data-overview-revision') || null;

  function safeStorage() {
    try {
      if (!window.localStorage) {
        return null;
      }
      var probeKey = '__dw_dashboard_probe__';
      window.localStorage.setItem(probeKey, '1');
      window.localStorage.removeItem(probeKey);
      return window.localStorage;
    } catch (error) {
      return null;
    }
  }

  function bindPersistedInputs() {
    var storage = safeStorage();
    if (!storage) {
      return;
    }
    var inputs = document.querySelectorAll('[data-persist-key]');
    if (!inputs || !inputs.length) {
      return;
    }
    inputs.forEach(function (input) {
      var key = input.getAttribute('data-persist-key');
      if (!key) {
        return;
      }
      try {
        var saved = storage.getItem(key);
        if (saved !== null && saved !== '' && !input.value) {
          input.value = saved;
        }
      } catch (error) {
        return;
      }

      var persist = function () {
        try {
          if (input.value) {
            storage.setItem(key, input.value);
          } else {
            storage.removeItem(key);
          }
        } catch (error) {
          return;
        }
      };

      input.addEventListener('input', persist);
      input.addEventListener('change', persist);
    });
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function setFilekeyLookupStatus(message, tone) {
    var status = document.getElementById('filekey-lookup-status');
    if (!status) {
      return;
    }
    status.textContent = message || '';
    status.dataset.tone = tone || 'neutral';
  }

  function clearFilekeyLookup() {
    ['filekey-lookup-summary', 'filekey-lookup-zip-groups', 'filekey-lookup-filters', 'filekey-lookup-groups', 'filekey-lookup-list'].forEach(function (id) {
      var node = document.getElementById(id);
      if (node) {
        node.innerHTML = '';
      }
    });
  }

  function statusLabel(status) {
    if (status === 'trainable') {
      return '학습 가능';
    }
    if (status === 'trained') {
      return '학습됨';
    }
    if (status === 'prepared') {
      return '준비됨';
    }
    if (status === 'excluded') {
      return '제외됨';
    }
    if (status === 'queued') {
      return '대기중';
    }
    if (status === 'running') {
      return '학습중';
    }
    return '미확인';
  }

  function viewBucketForZipGroup(group) {
    var normalized = String(group || '').toLowerCase();
    if (normalized === 'outsidedoor') {
      return 'outside';
    }
    if (normalized === 'insidedoor' || normalized === 'inside_croki') {
      return 'inside';
    }
    return 'other';
  }

  function isVisibleByViewFilter(item) {
    var bucket = viewBucketForZipGroup(item && item.zip_group);
    return filekeyViewFilters[bucket] !== false;
  }

  function filteredLookupEntries() {
    var entries = (latestFilekeyLookupPayload && latestFilekeyLookupPayload.entries) || [];
    return entries.filter(isVisibleByViewFilter);
  }

  function lookupTargetLabelOrder() {
    var order = {};
    var labels = latestFilekeyLookupPayload && latestFilekeyLookupPayload.target_labels;
    if (!Array.isArray(labels)) {
      return order;
    }
    labels.forEach(function (label, index) {
      var key = String(label || '').trim();
      if (key && !Object.prototype.hasOwnProperty.call(order, key)) {
        order[key] = index;
      }
    });
    return order;
  }

  function lookupClassSortValue(label) {
    var key = String(label || '').trim();
    var order = lookupTargetLabelOrder();
    if (Object.prototype.hasOwnProperty.call(order, key)) {
      return order[key];
    }
    if (key === 'unmapped') {
      return 9000;
    }
    if (key === 'excluded') {
      return 9001;
    }
    return 8000;
  }

  function lookupClassLabel(item) {
    var value = item && item.target_label;
    value = String(value || '').trim();
    if (value) {
      return value;
    }
    if (item && item.excluded_source_label) {
      return 'excluded';
    }
    return 'unmapped';
  }

  function isSelectableTrainableLookupItem(item) {
    return !!(
      item &&
      item.filekey &&
      item.status === 'trainable' &&
      item.selectable !== false
    );
  }

  function renderFilekeyRow(item, selectedMap) {
    var filekey = String(item.filekey || '');
    var selectable = item.selectable !== false && item.status !== 'excluded' && item.status !== 'trained' && item.status !== 'prepared' && item.status !== 'queued' && item.status !== 'running';
    var disabled = selectable ? '' : ' disabled';
    var checked = selectable && selectedMap && selectedMap[filekey] ? ' checked' : '';
    var title = item.name || item.source_label || item.source_alias || 'AIHub file';
    var metaParts = [];
    if (item.source_label) {
      metaParts.push(item.source_label);
    }
    if (item.zip_group) {
      metaParts.push(item.zip_group + (item.zip_number != null ? '_' + item.zip_number : ''));
    }
    if (item.target_label) {
      metaParts.push('target ' + item.target_label);
    }
    if (item.excluded_source_label) {
      metaParts.push('excluded ' + item.excluded_source_label);
    }
    if (item.trained_at) {
      metaParts.push('trained ' + item.trained_at);
    }
    if (item.prepared_at) {
      metaParts.push('prepared ' + item.prepared_at);
    }
    if (item.job_state === 'queued') {
      metaParts.push('queue pending');
    }
    if (item.job_state === 'running') {
      metaParts.push('currently running');
    }
    return '<label class="filekey-row">' +
      '<input type="checkbox" data-filekey="' + escapeHtml(filekey) + '"' + checked + disabled + ' />' +
      '<span class="filekey-main">' +
      '<strong>' + escapeHtml(filekey) + ' · ' + escapeHtml(title) + '</strong>' +
      '<span>' + escapeHtml(metaParts.join(' / ') || 'no label metadata') + '</span>' +
      '</span>' +
      '<span class="filekey-badges">' +
      (item.zip_group ? '<span class="filekey-zip-badge">' + escapeHtml(item.zip_group) + '</span>' : '') +
      '<span class="filekey-status ' + escapeHtml(item.status || 'unknown') + '">' + escapeHtml(statusLabel(item.status)) + '</span>' +
      '</span>' +
      '</label>';
  }

  function renderFilekeyClassList(entries, selectedMap) {
    var groups = {};
    entries.forEach(function (item) {
      var label = lookupClassLabel(item);
      if (!groups[label]) {
        groups[label] = {
          label: label,
          total: 0,
          trainable: 0,
          trained: 0,
          prepared: 0,
          queued: 0,
          running: 0,
          excluded: 0,
          unknown: 0,
          items: []
        };
      }
      groups[label].total += 1;
      var status = String(item.status || 'unknown');
      if (Object.prototype.hasOwnProperty.call(groups[label], status)) {
        groups[label][status] += 1;
      }
      groups[label].items.push(item);
    });

    var labels = Object.keys(groups).sort(function (left, right) {
      var a = groups[left];
      var b = groups[right];
      return (lookupClassSortValue(left) - lookupClassSortValue(right)) ||
        (b.trainable - a.trainable) ||
        (b.total - a.total) ||
        left.localeCompare(right);
    });
    if (!labels.length) {
      return '<div class="chart-empty">No filekeys to display.</div>';
    }
    return labels.map(function (label) {
      var group = groups[label];
      group.items.sort(function (left, right) {
        return String(left.filekey || '').localeCompare(String(right.filekey || ''), undefined, { numeric: true });
      });
      return '<section class="filekey-class-section" data-filekey-class-section="' + escapeHtml(label) + '">' +
        '<div class="filekey-class-header">' +
        '<button type="button" class="filekey-class-select" data-filekey-class="' + escapeHtml(label) + '">' +
        escapeHtml(label) +
        '</button>' +
        '<span class="filekey-class-meta">' +
        'total ' + escapeHtml(group.total) +
        ' / trainable ' + escapeHtml(group.trainable) +
        ' / trained ' + escapeHtml(group.trained) +
        ' / prepared ' + escapeHtml(group.prepared) +
        ' / queued ' + escapeHtml(group.queued) +
        ' / running ' + escapeHtml(group.running) +
        '</span>' +
        '</div>' +
        '<div class="filekey-class-body">' +
        group.items.map(function (item) { return renderFilekeyRow(item, selectedMap); }).join('') +
        '</div>' +
        '</section>';
    }).join('');
  }

  function renderFilekeyViewFilters(payload) {
    var container = document.getElementById('filekey-lookup-filters');
    if (!container) {
      return;
    }
    var entries = (payload && payload.entries) || [];
    var counts = { outside: 0, inside: 0, other: 0 };
    entries.forEach(function (item) {
      counts[viewBucketForZipGroup(item && item.zip_group)] += 1;
    });
    var options = [
      { key: 'outside', label: 'outside', count: counts.outside },
      { key: 'inside', label: 'inside', count: counts.inside },
      { key: 'other', label: '기타', count: counts.other }
    ].filter(function (item) { return item.count > 0; });
    container.innerHTML = options.map(function (item) {
      var checked = filekeyViewFilters[item.key] !== false ? ' checked' : '';
      return '<label class="filekey-filter-option">' +
        '<input type="checkbox" data-filekey-view-filter="' + escapeHtml(item.key) + '"' + checked + ' />' +
        '<span>' + escapeHtml(item.label) + '</span>' +
        '<strong>' + escapeHtml(item.count) + '</strong>' +
        '</label>';
    }).join('');
  }

  function renderFilekeyLookup(payload) {
    latestFilekeyLookupPayload = payload || null;
    var summary = document.getElementById('filekey-lookup-summary');
    var zipGroups = document.getElementById('filekey-lookup-zip-groups');
    var groups = document.getElementById('filekey-lookup-groups');
    var list = document.getElementById('filekey-lookup-list');
    if (!summary || !zipGroups || !groups || !list) {
      return;
    }

    var stats = (payload && payload.summary) || {};
    var entries = (payload && payload.entries) || [];
    var visibleEntries = entries.filter(isVisibleByViewFilter);
    var zipGroupItems = (payload && payload.zip_groups) || [];
    var groupItems = (payload && payload.groups) || [];
    summary.innerHTML =
      '<div class="filekey-chip"><span>전체 filekey</span><strong>' + escapeHtml(stats.total || 0) + '</strong></div>' +
      '<div class="filekey-chip filekey-chip-trainable"><span>학습 가능</span><strong>' + escapeHtml(stats.selectable || 0) + '</strong></div>' +
      '<div class="filekey-chip filekey-chip-trained"><span>학습됨</span><strong>' + escapeHtml(stats.trained || 0) + '</strong></div>' +
      '<div class="filekey-chip filekey-chip-prepared"><span>준비됨</span><strong>' + escapeHtml(stats.prepared || 0) + '</strong></div>' +
      '<div class="filekey-chip"><span>학습중</span><strong>' + escapeHtml(stats.running || 0) + '</strong></div>' +
      '<div class="filekey-chip"><span>대기중</span><strong>' + escapeHtml(stats.queued || 0) + '</strong></div>' +
      '<div class="filekey-chip"><span>제외</span><strong>' + escapeHtml(stats.excluded || 0) + '</strong></div>' +
      '<div class="filekey-chip"><span>미확인</span><strong>' + escapeHtml(stats.unknown || 0) + '</strong></div>';

    zipGroups.innerHTML = zipGroupItems.map(function (item) {
      return '<div class="filekey-group filekey-zip-group">' +
        '<span>' + escapeHtml(item.label || '기타') + '</span>' +
        '<strong>' + escapeHtml(item.total || 0) + '개</strong>' +
        '<em>선택 ' + escapeHtml(item.selectable || 0) +
        ' / 학습됨 ' + escapeHtml(item.trained || 0) +
        ' / 준비됨 ' + escapeHtml(item.prepared || 0) +
        ' / 학습중 ' + escapeHtml(item.running || 0) +
        ' / 대기 ' + escapeHtml(item.queued || 0) +
        ' / 제외 ' + escapeHtml(item.excluded || 0) + '</em>' +
        '</div>';
    }).join('');

    renderFilekeyViewFilters(payload);

    var classGroupCounts = {};
    visibleEntries.forEach(function (item) {
      var label = lookupClassLabel(item);
      classGroupCounts[label] = (classGroupCounts[label] || 0) + 1;
    });
    groups.innerHTML = Object.keys(classGroupCounts).sort(function (left, right) {
      return (lookupClassSortValue(left) - lookupClassSortValue(right)) || left.localeCompare(right);
    }).map(function (label) {
      return '<div class="filekey-group filekey-class-group">' +
        '<span>class: ' + escapeHtml(label) + '</span>' +
        '<strong>' + escapeHtml(classGroupCounts[label]) + '개</strong>' +
        '</div>';
    }).join('');

    var selectedMap = {};
    selectedLookupFilekeys().forEach(function (filekey) {
      selectedMap[filekey] = true;
    });

    list.innerHTML = renderFilekeyClassList(visibleEntries, selectedMap);

    var sourceLabel = payload && payload.source === 'aihubshell' ? 'aihubshell' : 'AIHub API';
    var cacheLabel = payload && payload.cache && payload.cache.hit ? ' / cache' : '';
    setFilekeyLookupStatus(sourceLabel + cacheLabel + ' 조회 완료', 'good');
    writeFilekeys(selectedLookupFilekeys());
  }

  function selectedLookupFilekeys() {
    var list = document.getElementById('filekey-lookup-list');
    if (!list) {
      return [];
    }
    return Array.prototype.slice.call(list.querySelectorAll('input[data-filekey]:checked'))
      .map(function (input) { return input.getAttribute('data-filekey') || ''; })
      .filter(Boolean);
  }

  function selectedVisibleLookupFilekeys() {
    var visible = {};
    filteredLookupEntries().forEach(function (item) {
      if (item && item.filekey) {
        visible[String(item.filekey)] = true;
      }
    });
    return selectedLookupFilekeys().filter(function (filekey) {
      return !!visible[filekey];
    });
  }

  function writeFilekeys(values) {
    var input = document.getElementById('filekeys-input');
    if (!input) {
      return;
    }
    var unique = [];
    var seen = {};
    values.forEach(function (value) {
      var filekey = String(value || '').trim();
      if (!filekey || seen[filekey]) {
        return;
      }
      seen[filekey] = true;
      unique.push(filekey);
    });
    input.value = unique.join('\\n');
    input.dispatchEvent(new Event('input', { bubbles: true }));
    if (input.type !== 'hidden') {
      input.focus();
    }
  }

  function setLookupCheckedByFilekeys(values) {
    var list = document.getElementById('filekey-lookup-list');
    if (!list) {
      return;
    }
    var selected = {};
    (values || []).forEach(function (value) {
      var filekey = String(value || '').trim();
      if (filekey) {
        selected[filekey] = true;
      }
    });
    Array.prototype.slice.call(list.querySelectorAll('input[data-filekey]')).forEach(function (input) {
      if (input.disabled) {
        input.checked = false;
        return;
      }
      input.checked = !!selected[input.getAttribute('data-filekey') || ''];
    });
    writeFilekeys(selectedLookupFilekeys());
  }

  function clearLookupSelection() {
    setLookupCheckedByFilekeys([]);
  }

  function syncSelectedFilekeysToForm() {
    var selected = selectedVisibleLookupFilekeys();
    writeFilekeys(selected);
    return selected;
  }

  function selectVisibleClassFilekeys(classLabel) {
    var selected = {};
    selectedLookupFilekeys().forEach(function (filekey) {
      selected[filekey] = true;
    });
    filteredLookupEntries().forEach(function (item) {
      if (lookupClassLabel(item) !== classLabel || !isSelectableTrainableLookupItem(item)) {
        return;
      }
      selected[String(item.filekey)] = true;
    });
    setLookupCheckedByFilekeys(Object.keys(selected));
    var values = syncSelectedFilekeysToForm();
    setFilekeyLookupStatus(classLabel + ' 클래스의 학습 가능 filekey ' + values.length + '개 선택됨', values.length ? 'good' : 'warn');
  }

  async function lookupAiHubFilekeys(force) {
    var datasetInput = document.getElementById('datasetkey-input');
    var apiKeyInput = document.getElementById('api-key-input');
    if (!datasetInput) {
      return;
    }
    var datasetkey = String(datasetInput.value || '').trim();
    var apiKey = apiKeyInput ? String(apiKeyInput.value || '').trim() : '';
    if (!datasetkey) {
      latestFilekeyLookupPayload = null;
      lastFilekeyLookupSignature = '';
      clearFilekeyLookup();
      setFilekeyLookupStatus('datasetkey 대기 중', 'neutral');
      return;
    }
    var signature = datasetkey + '|' + apiKey;
    if (!force && signature === lastFilekeyLookupSignature && latestFilekeyLookupPayload) {
      renderFilekeyLookup(latestFilekeyLookupPayload);
      return;
    }
    lastFilekeyLookupSignature = signature;

    if (filekeyLookupAbort && typeof filekeyLookupAbort.abort === 'function') {
      filekeyLookupAbort.abort();
    }
    filekeyLookupAbort = window.AbortController ? new AbortController() : null;
    var requestId = ++filekeyLookupRequestId;
    setFilekeyLookupStatus('파일 목록 조회 중...', 'accent');
    try {
      var response = await fetch('/api/aihub/filekeys', {
        method: 'POST',
        cache: 'no-store',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ datasetkey: datasetkey, api_key: apiKey }),
        signal: filekeyLookupAbort ? filekeyLookupAbort.signal : undefined
      });
      var payload = null;
      try {
        payload = await response.json();
      } catch (error) {
        payload = null;
      }
      if (!response.ok || !payload || payload.ok === false) {
        throw new Error((payload && (payload.detail || payload.message)) || '파일 목록 조회 실패');
      }
      if (requestId !== filekeyLookupRequestId) {
        return;
      }
      renderFilekeyLookup(payload);
    } catch (error) {
      if (error && error.name === 'AbortError') {
        return;
      }
      if (requestId !== filekeyLookupRequestId) {
        return;
      }
      latestFilekeyLookupPayload = null;
      clearFilekeyLookup();
      setFilekeyLookupStatus((error && error.message) || '파일 목록 조회 실패', 'danger');
    }
  }

  function scheduleFilekeyLookup(force, delay) {
    if (filekeyLookupTimer) {
      window.clearTimeout(filekeyLookupTimer);
    }
    filekeyLookupTimer = window.setTimeout(function () {
      lookupAiHubFilekeys(!!force);
    }, typeof delay === 'number' ? delay : 650);
  }

  function bindFilekeyLookup() {
    var datasetInput = document.getElementById('datasetkey-input');
    var apiKeyInput = document.getElementById('api-key-input');
    var lookupButton = document.getElementById('filekey-lookup-button');
    var selectTrainableButton = document.getElementById('filekey-select-trainable-button');
    var clearSelectionButton = document.getElementById('filekey-clear-selection-button');
    var lookupList = document.getElementById('filekey-lookup-list');
    var filterContainer = document.getElementById('filekey-lookup-filters');
    if (!datasetInput) {
      return;
    }
    datasetInput.addEventListener('input', function () {
      var value = String(datasetInput.value || '').trim();
      if (!value) {
        latestFilekeyLookupPayload = null;
        lastFilekeyLookupSignature = '';
        clearFilekeyLookup();
        setFilekeyLookupStatus('datasetkey 대기 중', 'neutral');
        return;
      }
      scheduleFilekeyLookup(false);
    });
    datasetInput.addEventListener('change', function () {
      scheduleFilekeyLookup(true, 0);
    });
    if (apiKeyInput) {
      apiKeyInput.addEventListener('change', function () {
        scheduleFilekeyLookup(true, 0);
      });
    }
    if (lookupButton) {
      lookupButton.addEventListener('click', function () {
        scheduleFilekeyLookup(true, 0);
      });
    }
    if (selectTrainableButton) {
      selectTrainableButton.addEventListener('click', function () {
        var values = filteredLookupEntries()
          .filter(isSelectableTrainableLookupItem)
          .map(function (item) { return String(item.filekey || ''); })
          .filter(Boolean);
        if (!values.length) {
          setFilekeyLookupStatus('선택할 학습 가능 filekey가 없습니다.', 'warn');
          return;
        }
        setLookupCheckedByFilekeys(values);
        setFilekeyLookupStatus(values.length + '개 filekey 선택됨', 'good');
      });
    }
    if (clearSelectionButton) {
      clearSelectionButton.addEventListener('click', function () {
        clearLookupSelection();
        setFilekeyLookupStatus('선택을 해제했습니다.', 'neutral');
      });
    }
    if (lookupList) {
      lookupList.addEventListener('click', function (event) {
        var target = event.target && event.target.closest ? event.target.closest('button[data-filekey-class]') : null;
        if (!target) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        selectVisibleClassFilekeys(target.getAttribute('data-filekey-class') || '');
      });
      lookupList.addEventListener('change', function (event) {
        if (!event.target || !event.target.matches('input[data-filekey]')) {
          return;
        }
        var values = syncSelectedFilekeysToForm();
        setFilekeyLookupStatus(values.length + '개 filekey 선택됨', values.length ? 'good' : 'neutral');
      });
    }
    if (filterContainer) {
      filterContainer.addEventListener('change', function (event) {
        var target = event.target;
        if (!target || !target.matches('input[data-filekey-view-filter]')) {
          return;
        }
        var key = target.getAttribute('data-filekey-view-filter');
        if (key) {
          filekeyViewFilters[key] = !!target.checked;
          if (latestFilekeyLookupPayload) {
            renderFilekeyLookup(latestFilekeyLookupPayload);
          }
          var values = syncSelectedFilekeysToForm();
          setFilekeyLookupStatus(values.length + '개 filekey 선택됨', values.length ? 'good' : 'neutral');
        }
      });
    }
    if (String(datasetInput.value || '').trim()) {
      scheduleFilekeyLookup(false, 350);
    }
  }

  function resolveNoticeTone(level) {
    var normalized = String(level || '').toLowerCase();
    if (normalized === 'good' || normalized === 'success') {
      return 'good';
    }
    if (normalized === 'warn' || normalized === 'warning') {
      return 'warn';
    }
    if (normalized === 'danger' || normalized === 'error') {
      return 'danger';
    }
    if (normalized === 'accent' || normalized === 'running' || normalized === 'queued') {
      return 'accent';
    }
    return 'neutral';
  }

  function showActionNotice(message, level) {
    var region = document.getElementById('action-notice-region');
    if (!region) {
      return;
    }
    if (!message) {
      region.innerHTML = '';
      return;
    }
    var tone = resolveNoticeTone(level);
    var title = tone === 'danger' ? '오류' : '작업 결과';
    region.innerHTML =
      '<section class="banner banner-' + tone + '">' +
      '<div class="banner-title">' + escapeHtml(title) + '</div>' +
      '<div class="banner-copy">' + escapeHtml(message) + '</div>' +
      '</section>';
    if (noticeTimer) {
      window.clearTimeout(noticeTimer);
    }
    noticeTimer = window.setTimeout(function () {
      region.innerHTML = '';
      noticeTimer = null;
    }, 7000);
  }

  function replaceRegion(id, html) {
    if (!html) {
      return;
    }
    var current = document.getElementById(id);
    if (!current) {
      return;
    }
    current.outerHTML = html;
  }

  function captureDetailOpenState(root) {
    var state = {};
    if (!root) {
      return state;
    }
    root.querySelectorAll('details[data-log-key]').forEach(function (node) {
      var key = node.getAttribute('data-log-key');
      if (key) {
        state[key] = !!node.open;
      }
    });
    return state;
  }

  function restoreDetailOpenState(root, state) {
    if (!root || !state) {
      return;
    }
    root.querySelectorAll('details[data-log-key]').forEach(function (node) {
      var key = node.getAttribute('data-log-key');
      if (key && Object.prototype.hasOwnProperty.call(state, key)) {
        node.open = !!state[key];
      }
    });
  }

  function replaceLogRegion(id, html) {
    if (!html) {
      return;
    }
    var current = document.getElementById(id);
    if (!current) {
      return;
    }
    var openState = captureDetailOpenState(current);
    current.outerHTML = html;
    var replaced = document.getElementById(id);
    restoreDetailOpenState(replaced, openState);
  }

  function nextDelay() {
    return document.hidden ? Math.max(delayMs, 10000) : delayMs;
  }

  function schedule(ms) {
    if (disposed) {
      return;
    }
    if (timer) {
      window.clearTimeout(timer);
    }
    timer = window.setTimeout(poll, ms);
  }

  function requestRefreshSoon(ms) {
    schedule(typeof ms === 'number' ? ms : 250);
  }

  async function poll() {
    if (disposed) {
      return;
    }
    if (inflight) {
      schedule(nextDelay());
      return;
    }
    inflight = true;
    try {
      var url = '/api/live-fragments';
      if (lastOverviewRevision) {
        url += '?revision=' + encodeURIComponent(lastOverviewRevision);
      }
      var response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error('status ' + response.status);
      }
      var payload = await response.json();
      if (payload && payload.fragments) {
        var nextRevision = payload.overview_revision || null;
        if (nextRevision && lastOverviewRevision && nextRevision === lastOverviewRevision) {
          if (payload && payload.poll_interval_ms) {
            delayMs = payload.poll_interval_ms;
          } else if (payload && payload.active === false) {
            delayMs = 8000;
          } else {
            delayMs = 1200;
          }
          return;
        }
        replaceRegion('hero-kpis', payload.fragments.hero_kpis);
        replaceRegion('hero-status-card', payload.fragments.hero_status);
        if (payload.fragments.main_live_sections) {
          replaceRegion('main-live-sections', payload.fragments.main_live_sections);
        } else {
          replaceRegion('summary-grid', payload.fragments.summary_cards);
        }
        replaceRegion('queue-panel', payload.fragments.queue_panel);
        replaceRegion('system-panel', payload.fragments.system_panel);
        replaceLogRegion('logs-stack', payload.fragments.logs);
        if (nextRevision) {
          document.documentElement.setAttribute('data-overview-revision', nextRevision);
        }
        lastOverviewRevision = nextRevision;
      }
      if (payload && payload.poll_interval_ms) {
        delayMs = payload.poll_interval_ms;
      } else if (payload && payload.active === false) {
        delayMs = 8000;
      } else {
        delayMs = 1200;
      }
    } catch (error) {
      delayMs = Math.min(Math.max(delayMs * 2, 3000), 20000);
    } finally {
      inflight = false;
      schedule(nextDelay());
    }
  }

  function setFormPending(form, pending) {
    if (!form) {
      return;
    }
    form.dataset.pending = pending ? '1' : '0';
    var controls = form.querySelectorAll('button, input[type="submit"]');
    controls.forEach(function (control) {
      control.disabled = !!pending;
    });
  }

  async function submitActionForm(form, submitter) {
    if (!form || form.dataset.pending === '1') {
      return;
    }
    var autoRecommendedStart = !!(submitter && submitter.name === 'auto_enqueue_next');
    if (form.id === 'start-job-form') {
      var selectedFilekeys = syncSelectedFilekeysToForm();
      if (!selectedFilekeys.length && !autoRecommendedStart) {
        showActionNotice('큐에 넣을 filekey를 먼저 선택해 주세요.', 'warn');
        return;
      }
    }
    setFormPending(form, true);
    try {
      var formData = new FormData(form);
      var encodedBody = new URLSearchParams();
      formData.forEach(function (value, key) {
        encodedBody.append(key, value);
      });
      if (submitter && submitter.name) {
        encodedBody.append(submitter.name, submitter.value || '1');
      }
      var response = await fetch(form.action, {
        method: String(form.method || 'POST').toUpperCase(),
        body: encodedBody,
        cache: 'no-store',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
          'X-Dashboard-Async': '1'
        }
      });

      var payload = null;
      try {
        payload = await response.json();
      } catch (error) {
        payload = null;
      }

      if (!response.ok || !payload || payload.ok === false) {
        showActionNotice((payload && payload.message) || '요청 처리 중 오류가 발생했습니다.', 'danger');
        requestRefreshSoon(300);
        return;
      }

      showActionNotice(payload.message || '요청을 처리했습니다.', payload.level || 'good');
      requestRefreshSoon(150);
    } catch (error) {
      showActionNotice('네트워크 오류로 요청을 처리하지 못했습니다.', 'danger');
    } finally {
      setFormPending(form, false);
    }
  }

  document.addEventListener('submit', function (event) {
    var form = event.target;
    if (!(form instanceof HTMLFormElement)) {
      return;
    }
    var action = form.getAttribute('action') || '';
    if (action.indexOf('/actions/') !== 0) {
      return;
    }
    event.preventDefault();
    submitActionForm(form, event.submitter);
  });

  document.addEventListener('visibilitychange', function () {
    schedule(document.hidden ? Math.max(delayMs, 10000) : 800);
  });

  window.addEventListener('beforeunload', function () {
    disposed = true;
    if (timer) {
      window.clearTimeout(timer);
    }
  });

  bindPersistedInputs();
  bindFilekeyLookup();
  schedule(1200);
}());
</script>
"""


def _prepared_total_from_job(job: dict) -> int:
    summary = job.get("result_summary") or {}
    if not isinstance(summary, dict):
        return 0
    return sum(
        int(summary.get(key) or 0)
        for key in ("prepared_train_total", "prepared_val_total", "prepared_test_total")
    )


def _coerce_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_dataset_rows(dataset_summary: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    for key in SPLIT_SEQUENCE:
        info = dataset_summary.get(key) or {}
        rows.append(
            [
                _split_label(key),
                _int_text(info.get("total"), "0"),
                _text(_join_label_counts(info), "-"),
            ]
        )
    return rows


def _build_history_bundle(progress: dict, metrics: dict) -> dict:
    source_rows = progress.get("history") or metrics.get("history") or []
    history_rows: list[list[str]] = []
    epoch_labels: list[str] = []
    val_accuracy_values: list[float | None] = []
    val_f1_values: list[float | None] = []
    train_loss_values: list[float | None] = []
    val_loss_values: list[float | None] = []

    normalized_rows: list[dict] = []
    for row in list(source_rows)[-20:]:
        if not isinstance(row, dict):
            continue
        normalized_rows.append(row)
        epoch_label = _text(row.get("epoch"))
        train_loss = _coerce_float(row.get("train_loss"))
        val_loss = _coerce_float(row.get("val_loss"))
        val_accuracy = _coerce_float(row.get("val_accuracy"))
        val_macro_f1 = _coerce_float(row.get("val_macro_f1"))

        epoch_labels.append(epoch_label)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)
        val_accuracy_values.append(val_accuracy)
        val_f1_values.append(val_macro_f1)
        history_rows.append(
            [
                epoch_label,
                _float_text(train_loss),
                _float_text(val_loss),
                _float_text(val_accuracy),
                _float_text(val_macro_f1),
                _float_text(_coerce_float(row.get("learning_rate")), 6),
            ]
        )

    return {
        "rows": history_rows,
        "source_rows": normalized_rows,
        "epoch_labels": epoch_labels,
        "train_loss_values": train_loss_values,
        "val_loss_values": val_loss_values,
        "val_accuracy_values": val_accuracy_values,
        "val_f1_values": val_f1_values,
    }


def _build_support_map(confusion: list[list[int]] | None) -> dict[int, int]:
    support_map: dict[int, int] = {}
    if isinstance(confusion, list):
        for index, matrix_row in enumerate(confusion):
            if isinstance(matrix_row, list):
                support_map[index] = sum(int(value or 0) for value in matrix_row)
    return support_map


def _build_per_class_rows(final_validation: dict, labels: list[str], support_map: dict[int, int]) -> list[list[str]]:
    rows: list[list[str]] = []
    for index, row in enumerate(final_validation.get("per_class") or []):
        if not isinstance(row, dict):
            continue
        class_index = int(row.get("class_index", index) or index)
        label = labels[class_index] if 0 <= class_index < len(labels) else row.get("label") or class_index
        rows.append(
            [
                _text(label),
                _float_text(row.get("precision"), 4, "-"),
                _float_text(row.get("recall"), 4, "-"),
                _float_text(row.get("f1"), 4, "-"),
                _int_text(support_map.get(class_index), "0"),
            ]
        )
    return rows


def _build_recent_job_rows(completed_jobs: list[dict], limit: int = 20) -> list[list[str]]:
    rows: list[list[str]] = []
    for job in completed_jobs[:limit]:
        if not isinstance(job, dict):
            continue
        rows.append(
            [
                _text(job.get("filekey")),
                _text(job.get("datasetkey")),
                _state_label(job.get("state")),
                _int_text(_prepared_total_from_job(job), "0"),
                _text(job.get("finished_at") or job.get("started_at")),
                _text(job.get("message")),
            ]
        )
    return rows


def _build_issue_rows(issues: list[dict], limit: int = 20) -> list[list[str]]:
    rows: list[list[str]] = []
    for issue in issues[:limit]:
        if not isinstance(issue, dict):
            continue
        rows.append(
            [
                _split_label(issue.get("split")),
                _text(issue.get("target_label")),
                _text(issue.get("video_name")),
                _text(issue.get("reason")),
                _int_text(issue.get("valid_frames"), "0"),
                _int_text(issue.get("total_valid_keypoints"), "0"),
                _text(", ".join(str(action) for action in issue.get("recovery_actions") or []), "-"),
            ]
        )
    return rows


def _build_skip_summary_rows(skip_report: dict, limit: int = 30) -> list[list[str]]:
    summary = skip_report.get("summary") or {}
    rows: list[list[str]] = []
    for group_name, values in (
        ("reason", summary.get("by_reason") or {}),
        ("split", summary.get("by_split") or {}),
        ("label", summary.get("by_label") or {}),
    ):
        if not isinstance(values, dict):
            continue
        sorted_items = sorted(
            values.items(),
            key=lambda item: (-int(item[1] or 0), str(item[0])),
        )
        for name, count in sorted_items:
            rows.append([_text(group_name), _text(name), _int_text(count)])
            if len(rows) >= limit:
                return rows
    return rows


def _build_distribution_entries(dataset: dict, labels: list[str]) -> list[tuple[str, int]]:
    prepared_train_counts = ((dataset.get("prepared_train") or {}).get("by_label") or {})
    distribution_labels = labels or sorted(prepared_train_counts.keys())
    return [
        (str(label), int(prepared_train_counts.get(label) or 0))
        for label in distribution_labels
    ]


def _render_line_chart(
    title: str,
    subtitle: str,
    x_labels: list[str],
    series: list[tuple[str, str, list[float | None]]],
    *,
    fixed_min: float | None = None,
    fixed_max: float | None = None,
    decimals: int = 4,
    span_all: bool = False,
) -> str:
    valid_values = [value for _, _, values in series for value in values if value is not None]
    panel_class = "panel chart-panel chart-panel-span" if span_all else "panel chart-panel"
    if not x_labels or not valid_values:
        return (
            f"<article class=\"{panel_class}\">"
            f"<div class=\"panel-head\"><div><h2>{_text(title)}</h2><p>{_text(subtitle)}</p></div></div>"
            "<div class=\"panel-body\">"
            "<div class=\"chart-empty\">그래프를 그릴 학습 기록이 아직 없습니다.</div>"
            "</div></article>"
        )

    width = 720
    height = 280
    pad_left = 52
    pad_right = 18
    pad_top = 20
    pad_bottom = 38
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom

    y_min = fixed_min if fixed_min is not None else min(valid_values)
    y_max = fixed_max if fixed_max is not None else max(valid_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    if fixed_min is None or fixed_max is None:
        padding = (y_max - y_min) * 0.12 or 0.1
        if fixed_min is None:
            y_min -= padding
        if fixed_max is None:
            y_max += padding

    def x_pos(index: int) -> float:
        if len(x_labels) == 1:
            return pad_left + inner_w / 2
        return pad_left + (inner_w * index / (len(x_labels) - 1))

    def y_pos(value: float) -> float:
        ratio = (value - y_min) / (y_max - y_min)
        ratio = max(0.0, min(1.0, ratio))
        return pad_top + (1.0 - ratio) * inner_h

    grid_lines = []
    for step in range(5):
        ratio = step / 4
        y = pad_top + ratio * inner_h
        tick_value = y_max - ((y_max - y_min) * ratio)
        grid_lines.append(
            f"<line x1=\"{pad_left}\" y1=\"{y:.2f}\" x2=\"{width - pad_right}\" y2=\"{y:.2f}\" class=\"chart-grid-line\" />"
            f"<text x=\"{pad_left - 10}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" class=\"chart-axis-label\">{tick_value:.{decimals}f}</text>"
        )

    tick_indexes = sorted(set([0, len(x_labels) - 1] + list(range(0, len(x_labels), max(1, len(x_labels) // 5 or 1)))))
    x_ticks = []
    for index in tick_indexes:
        x = x_pos(index)
        x_ticks.append(
            f"<text x=\"{x:.2f}\" y=\"{height - 12}\" text-anchor=\"middle\" class=\"chart-axis-label\">{_text(x_labels[index])}</text>"
        )

    series_paths: list[str] = []
    legend_items: list[str] = []
    for name, color, values in series:
        segments: list[list[tuple[float, float]]] = []
        current_segment: list[tuple[float, float]] = []
        points_markup: list[str] = []
        last_value = None
        for index, value in enumerate(values):
            if value is None:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue
            x = x_pos(index)
            y = y_pos(value)
            current_segment.append((x, y))
            points_markup.append(
                f"<circle cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"4.2\" fill=\"{color}\" class=\"chart-point\" />"
            )
            last_value = value
        if current_segment:
            segments.append(current_segment)
        for segment in segments:
            if len(segment) < 2:
                continue
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in segment)
            series_paths.append(
                f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"3.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" points=\"{points}\" />"
            )
        series_paths.extend(points_markup)
        legend_items.append(
            "<div class=\"legend-item\">"
            f"<span class=\"legend-swatch\" style=\"background:{color};\"></span>"
            f"<span>{_text(name)}</span>"
            f"<strong>{'-' if last_value is None else f'{last_value:.{decimals}f}'}</strong>"
            "</div>"
        )

    svg = (
        f"<svg class=\"metric-chart\" viewBox=\"0 0 {width} {height}\" preserveAspectRatio=\"none\">"
        + "".join(grid_lines)
        + f"<line x1=\"{pad_left}\" y1=\"{height - pad_bottom}\" x2=\"{width - pad_right}\" y2=\"{height - pad_bottom}\" class=\"chart-axis\" />"
        + f"<line x1=\"{pad_left}\" y1=\"{pad_top}\" x2=\"{pad_left}\" y2=\"{height - pad_bottom}\" class=\"chart-axis\" />"
        + "".join(series_paths)
        + "".join(x_ticks)
        + "</svg>"
    )

    return (
        f"<article class=\"{panel_class}\">"
        f"<div class=\"panel-head\"><div><h2>{_text(title)}</h2><p>{_text(subtitle)}</p></div></div>"
        "<div class=\"panel-body\">"
        "<div class=\"chart-wrap\">"
        f"{svg}"
        "<div class=\"chart-legend\">"
        + "".join(legend_items)
        + "</div></div></div></article>"
    )


def _render_distribution_chart(title: str, subtitle: str, entries: list[tuple[str, int]]) -> str:
    non_empty = [(label, count) for label, count in entries if count > 0]
    if not non_empty:
        return (
            "<article class=\"panel chart-panel chart-panel-span\">"
            f"<div class=\"panel-head\"><div><h2>{_text(title)}</h2><p>{_text(subtitle)}</p></div></div>"
            "<div class=\"panel-body\">"
            "<div class=\"chart-empty\">분포를 그릴 클래스 샘플이 아직 없습니다.</div>"
            "</div></article>"
        )

    max_count = max(count for _, count in non_empty) or 1
    rows = []
    for label, count in non_empty:
        width = (count / max_count) * 100
        rows.append(
            "<div class=\"bar-row\">"
            "<div class=\"bar-meta\">"
            f"<span>{_text(label)}</span>"
            f"<strong>{count}</strong>"
            "</div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar-fill\" style=\"width:{width:.2f}%\"></div>"
            "</div>"
            "</div>"
        )

    return (
        "<article class=\"panel chart-panel chart-panel-span\">"
        f"<div class=\"panel-head\"><div><h2>{_text(title)}</h2><p>{_text(subtitle)}</p></div></div>"
        "<div class=\"panel-body\">"
        "<div class=\"bar-chart\">"
        + "".join(rows)
        + "</div></div></article>"
    )


def _render_confusion_matrix(labels: list[str], confusion: list[list[int]] | None) -> str:
    if not isinstance(confusion, list) or not confusion:
        return (
            "<article class=\"panel chart-panel chart-panel-span\">"
            "<div class=\"panel-head\"><div><h2>혼동 행렬</h2><p>최종 검증 결과를 실제 행렬 형태로 보여줍니다.</p></div></div>"
            "<div class=\"panel-body\">"
            "<div class=\"chart-empty\">혼동 행렬이 없습니다.</div>"
            "</div></article>"
        )

    size = max(len(labels), len(confusion), max((len(row) for row in confusion if isinstance(row, list)), default=0))
    if size <= 0:
        return (
            "<article class=\"panel chart-panel chart-panel-span\">"
            "<div class=\"panel-head\"><div><h2>혼동 행렬</h2><p>최종 검증 결과를 실제 행렬 형태로 보여줍니다.</p></div></div>"
            "<div class=\"panel-body\">"
            "<div class=\"chart-empty\">혼동 행렬이 없습니다.</div>"
            "</div></article>"
        )

    resolved_labels = [str(labels[index]) if index < len(labels) else f"class {index}" for index in range(size)]
    normalized_rows: list[list[int]] = []
    row_totals: list[int] = []
    col_totals = [0 for _ in range(size)]
    diagonal_total = 0
    max_value = 0
    top_errors: list[tuple[int, str, str]] = []

    for row_index in range(size):
        raw_row = confusion[row_index] if row_index < len(confusion) and isinstance(confusion[row_index], list) else []
        normalized_row: list[int] = []
        row_total = 0
        for col_index in range(size):
            raw_value = raw_row[col_index] if col_index < len(raw_row) else 0
            try:
                value = int(raw_value or 0)
            except (TypeError, ValueError):
                value = 0
            normalized_row.append(value)
            row_total += value
            col_totals[col_index] += value
            if row_index == col_index:
                diagonal_total += value
            elif value > 0:
                top_errors.append((value, resolved_labels[row_index], resolved_labels[col_index]))
            max_value = max(max_value, value)
        normalized_rows.append(normalized_row)
        row_totals.append(row_total)

    overall_total = sum(row_totals)
    overall_accuracy = (diagonal_total / overall_total) if overall_total else None
    top_errors.sort(key=lambda item: item[0], reverse=True)
    top_error_text = (
        f"{top_errors[0][1]} → {top_errors[0][2]} ({top_errors[0][0]})"
        if top_errors
        else "눈에 띄는 오분류가 없습니다"
    )

    header_cells = "".join(
        "<th class=\"matrix-col-head\">"
        f"<div class=\"matrix-col-label\">{_text(label)}</div>"
        "</th>"
        for label in resolved_labels
    )

    body_rows = []
    denominator = max(max_value, 1)
    for row_index, row_values in enumerate(normalized_rows):
        label = resolved_labels[row_index]
        row_total = row_totals[row_index]
        diagonal_value = row_values[row_index] if row_index < len(row_values) else 0
        recall = (diagonal_value / row_total) if row_total else None
        cell_markup = []
        for col_index, value in enumerate(row_values):
            intensity = value / denominator if denominator else 0.0
            opacity = 0.08 + (intensity * 0.60)
            is_diagonal = row_index == col_index
            color = f"rgba(5, 150, 105, {opacity:.3f})" if is_diagonal else f"rgba(220, 38, 38, {opacity:.3f})"
            share = (value / row_total) if row_total else None
            cell_markup.append(
                "<td class=\"matrix-cell{diag}{zero}\" style=\"background:{bg};\" title=\"{title}\">"
                "<span class=\"matrix-count\">{count}</span>"
                "<span class=\"matrix-share\">{share}</span>"
                "</td>".format(
                    diag=" is-diagonal" if is_diagonal else "",
                    zero=" is-zero" if value == 0 else "",
                    bg=color if value > 0 else "rgba(148, 163, 184, 0.06)",
                    title=escape(
                        f"실제 {label} / 예측 {resolved_labels[col_index]} / count {value} / share "
                        + ("-" if share is None else f"{share:.1%}")
                    ),
                    count=value,
                    share="-" if share is None or value == 0 else f"{share:.0%}",
                )
            )

        body_rows.append(
            "<tr>"
            "<th class=\"matrix-row-head\">"
            f"<div class=\"matrix-row-label\">{_text(label)}</div>"
            f"<div class=\"matrix-row-meta\">실제 {row_total} / 정답 {'-' if recall is None else f'{recall:.0%}'}</div>"
            "</th>"
            + "".join(cell_markup)
            + f"<td class=\"matrix-total-cell\"><span class=\"matrix-total-count\">{row_total}</span><span class=\"matrix-total-meta\">row total</span></td>"
            "</tr>"
        )

    footer_cells = "".join(
        f"<td class=\"matrix-total-cell\"><span class=\"matrix-total-count\">{value}</span><span class=\"matrix-total-meta\">pred total</span></td>"
        for value in col_totals
    )

    summary_html = (
        "<div class=\"confusion-summary\">"
        "<div class=\"confusion-kpi\">"
        "<span>전체 정확도</span>"
        f"<strong>{'-' if overall_accuracy is None else f'{overall_accuracy:.1%}'}</strong>"
        "</div>"
        "<div class=\"confusion-kpi\">"
        "<span>정답 / 전체</span>"
        f"<strong>{diagonal_total} / {overall_total}</strong>"
        "</div>"
        "<div class=\"confusion-kpi confusion-kpi-wide\">"
        "<span>가장 큰 혼동</span>"
        f"<strong>{_text(top_error_text)}</strong>"
        "</div>"
        "</div>"
    )

    legend_html = (
        "<div class=\"confusion-legend\">"
        "<span><i class=\"legend-box legend-box-diag\"></i>대각선: 정답 예측</span>"
        "<span><i class=\"legend-box legend-box-error\"></i>비대각선: 오분류</span>"
        "<span><i class=\"legend-box legend-box-total\"></i>행/열 합계</span>"
        "</div>"
    )

    matrix_html = (
        "<div class=\"matrix-wrap\">"
        "<table class=\"matrix-table\">"
        "<thead><tr>"
        "<th class=\"matrix-corner\">실제 \\ 예측</th>"
        f"{header_cells}"
        "<th class=\"matrix-col-head matrix-total-head\">행 합계</th>"
        "</tr></thead>"
        "<tbody>"
        + "".join(body_rows)
        + "<tr>"
        "<th class=\"matrix-row-head matrix-total-head\">열 합계</th>"
        + footer_cells
        + f"<td class=\"matrix-total-cell matrix-grand-total\"><span class=\"matrix-total-count\">{overall_total}</span><span class=\"matrix-total-meta\">overall</span></td>"
        + "</tr>"
        "</tbody></table></div>"
    )

    return (
        "<article class=\"panel chart-panel chart-panel-span confusion-panel\">"
        "<div class=\"panel-head\"><div><h2>혼동 행렬</h2><p>대각선은 정답, 비대각선은 오분류입니다. 색이 진할수록 빈도가 큽니다.</p></div></div>"
        "<div class=\"panel-body\">"
        f"{summary_html}"
        f"{legend_html}"
        f"{matrix_html}"
        "</div></article>"
    )


@lru_cache(maxsize=1)
def _styles() -> str:
    return """
  <style>
    :root {
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
      --shadow-soft: 0 10px 22px rgba(15, 23, 42, 0.05);
      --radius-xl: 26px;
      --radius-lg: 22px;
      --radius-md: 16px;
      --max-width: 1560px;
    }
    * {
      box-sizing: border-box;
    }
    html {
      scroll-behavior: smooth;
    }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15,111,255,0.10), transparent 28%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-deep) 100%);
      color: var(--ink);
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
      line-height: 1.55;
    }
    a {
      color: inherit;
    }
    .page {
      max-width: var(--max-width);
      margin: 0 auto;
      padding: 22px 20px 44px;
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.7fr);
      gap: 18px;
      padding: 26px;
      border-radius: 30px;
      border: 1px solid rgba(15,111,255,0.16);
      background:
        linear-gradient(135deg, rgba(255,255,255,0.94), rgba(246,250,255,0.92)),
        radial-gradient(circle at top right, rgba(15,111,255,0.12), transparent 40%);
      box-shadow: var(--shadow);
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(15,111,255,0.18);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      box-shadow: var(--shadow-soft);
    }
    .hero h1 {
      margin: 12px 0 8px;
      font-size: 38px;
      line-height: 1.12;
      letter-spacing: -0.02em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 860px;
    }
    .hero-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    .hero-links a {
      text-decoration: none;
      border-radius: 999px;
      border: 1px solid rgba(15,111,255,0.16);
      background: rgba(255,255,255,0.75);
      padding: 9px 14px;
      font-size: 13px;
      font-weight: 700;
      color: var(--accent);
    }
    .hero-kpis {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .hero-kpi {
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(148,163,184,0.18);
      box-shadow: var(--shadow-soft);
    }
    .hero-kpi span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .hero-kpi strong {
      display: block;
      margin-top: 6px;
      font-size: 16px;
      font-weight: 800;
      color: var(--ink);
    }
    .status-card {
      padding: 22px;
      border-radius: 24px;
      background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(246,248,253,0.88));
      border: 1px solid rgba(15,111,255,0.14);
      box-shadow: var(--shadow-soft);
      display: grid;
      align-content: start;
      gap: 14px;
    }
    .status-badge {
      display: inline-flex;
      width: fit-content;
      padding: 7px 11px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: #fff;
    }
    .status-badge.good { background: var(--good); }
    .status-badge.warn { background: var(--warn); }
    .status-badge.danger { background: var(--danger); }
    .status-badge.accent,
    .status-badge.neutral { background: var(--accent); }
    .status-main {
      color: var(--muted);
      font-size: 14px;
    }
    .status-main strong {
      display: block;
      margin-bottom: 8px;
      color: var(--ink);
      font-size: 18px;
      line-height: 1.35;
    }
    .mini-stack {
      display: grid;
      gap: 10px;
    }
    .mini-row {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(248,251,255,0.92);
      border: 1px solid var(--line);
      font-size: 13px;
    }
    .mini-row span {
      color: var(--muted);
      font-weight: 700;
    }
    .mini-row strong {
      text-align: right;
      font-weight: 800;
      color: var(--ink);
    }
    .banner {
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid transparent;
      box-shadow: var(--shadow-soft);
    }
    .banner-good {
      background: rgba(5,150,105,0.10);
      border-color: rgba(5,150,105,0.18);
    }
    .banner-warn {
      background: rgba(217,119,6,0.10);
      border-color: rgba(217,119,6,0.18);
    }
    .banner-danger {
      background: rgba(220,38,38,0.10);
      border-color: rgba(220,38,38,0.18);
    }
    .banner-neutral,
    .banner-accent {
      background: rgba(15,111,255,0.10);
      border-color: rgba(15,111,255,0.18);
    }
    .banner-title {
      font-weight: 900;
      margin-bottom: 8px;
    }
    .banner-copy {
      color: var(--ink);
    }
    .banner-list {
      margin: 0;
      padding-left: 18px;
    }
    .section-nav {
      position: sticky;
      top: 12px;
      z-index: 20;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 18px 0;
      padding: 12px;
      border-radius: 18px;
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(148,163,184,0.18);
      backdrop-filter: blur(14px);
      box-shadow: var(--shadow-soft);
    }
    .section-nav a {
      text-decoration: none;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15,111,255,0.08);
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
    }
    .layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
      align-items: start;
    }
    .sidebar {
      display: grid;
      grid-template-columns: minmax(420px, 1.15fr) minmax(340px, 0.85fr);
      gap: 16px;
      align-items: start;
    }
    .sidebar > .sidebar-panel:first-child {
      grid-row: span 2;
    }
    .content {
      display: grid;
      gap: 18px;
      min-width: 0;
    }
    .panel {
      border-radius: var(--radius-xl);
      background: var(--panel);
      border: 1px solid rgba(148,163,184,0.16);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-span-full {
      grid-column: 1 / -1;
    }
    .panel-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 20px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.92));
    }
    .panel-head h2 {
      margin: 0 0 6px;
      font-size: 20px;
      line-height: 1.18;
      letter-spacing: -0.01em;
    }
    .panel-head p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }
    .panel-body {
      padding: 18px 20px 20px;
    }
    .sidebar-panel .panel-body {
      padding-top: 16px;
    }
    .sidebar-panel {
      min-width: 0;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }
    .summary-card {
      padding: 18px;
      border-radius: 22px;
      border: 1px solid rgba(148,163,184,0.14);
      background: var(--panel);
      box-shadow: var(--shadow-soft);
      min-height: 140px;
      display: grid;
      align-content: start;
      gap: 10px;
    }
    .summary-card-good { border-color: rgba(5,150,105,0.18); }
    .summary-card-warn { border-color: rgba(217,119,6,0.18); }
    .summary-card-danger { border-color: rgba(220,38,38,0.18); }
    .summary-card-accent { border-color: rgba(15,111,255,0.20); }
    .summary-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .summary-value {
      font-size: 26px;
      line-height: 1.15;
      font-weight: 900;
      letter-spacing: -0.02em;
      color: var(--ink);
    }
    .summary-copy {
      color: var(--muted);
      font-size: 13px;
    }
    .insight-panel {
      border-color: rgba(15,111,255,0.18);
    }
    .insight-panel-good {
      border-color: rgba(5,150,105,0.22);
    }
    .insight-panel-warn {
      border-color: rgba(217,119,6,0.24);
    }
    .insight-panel-danger {
      border-color: rgba(220,38,38,0.24);
    }
    .insight-badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }
    .insight-badge {
      display: inline-grid;
      grid-template-columns: auto auto;
      align-items: center;
      gap: 4px 8px;
      min-width: 160px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(248,251,255,0.92);
      box-shadow: var(--shadow-soft);
    }
    .insight-badge.good {
      border-color: rgba(5,150,105,0.20);
      background: rgba(5,150,105,0.08);
    }
    .insight-badge.warn {
      border-color: rgba(217,119,6,0.20);
      background: rgba(217,119,6,0.08);
    }
    .insight-badge.danger {
      border-color: rgba(220,38,38,0.20);
      background: rgba(220,38,38,0.08);
    }
    .insight-badge strong,
    .insight-badge em,
    .insight-badge small {
      min-width: 0;
    }
    .insight-badge strong {
      font-size: 12px;
      font-weight: 900;
      color: var(--ink);
    }
    .insight-badge em {
      justify-self: end;
      font-size: 11px;
      font-style: normal;
      font-weight: 900;
      color: var(--accent);
      text-transform: uppercase;
    }
    .insight-badge.good em { color: var(--good); }
    .insight-badge.warn em { color: var(--warn); }
    .insight-badge.danger em { color: var(--danger); }
    .insight-badge small {
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .insight-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .insight-card {
      min-width: 0;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(248,251,255,0.72);
    }
    .insight-card h3 {
      margin: 0 0 10px;
      font-size: 14px;
      line-height: 1.3;
      font-weight: 900;
    }
    .insight-list {
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .insight-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
      padding: 11px 12px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(255,255,255,0.78);
    }
    .insight-item.good {
      border-color: rgba(5,150,105,0.18);
      background: rgba(5,150,105,0.06);
    }
    .insight-item.warn {
      border-color: rgba(217,119,6,0.18);
      background: rgba(217,119,6,0.06);
    }
    .insight-item.danger {
      border-color: rgba(220,38,38,0.18);
      background: rgba(220,38,38,0.06);
    }
    .insight-item strong {
      display: block;
      color: var(--ink);
      font-size: 13px;
      font-weight: 900;
      line-height: 1.35;
    }
    .insight-item p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .insight-item .status-badge {
      padding: 5px 8px;
      font-size: 10px;
      white-space: nowrap;
    }
    .stack,
    .stack-form {
      display: grid;
      gap: 12px;
    }
    .field-label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
    }
    .field-label input,
    .field-label textarea {
      width: 100%;
      padding: 12px 13px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.24);
      background: rgba(248,251,255,0.95);
      font: inherit;
      color: var(--ink);
    }
    .field-label textarea {
      resize: vertical;
      min-height: 150px;
    }
    .primary-button,
    .secondary-button {
      appearance: none;
      border: none;
      cursor: pointer;
      font: inherit;
      font-weight: 800;
      transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
    }
    .primary-button:hover,
    .secondary-button:hover {
      transform: translateY(-1px);
    }
    .primary-button {
      padding: 13px 16px;
      border-radius: 14px;
      color: #fff;
      background: linear-gradient(135deg, #0f6fff, #245dff);
      box-shadow: 0 12px 22px rgba(15,111,255,0.26);
    }
    .action-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .action-grid form {
      margin: 0;
    }
    .secondary-button {
      width: 100%;
      padding: 11px 12px;
      border-radius: 14px;
      background: rgba(15,111,255,0.10);
      color: var(--accent);
      border: 1px solid rgba(15,111,255,0.14);
    }
    .secondary-button-compact {
      width: auto;
      padding: 9px 11px;
      border-radius: 12px;
      font-size: 12px;
      white-space: nowrap;
    }
    .secondary-button-warn {
      background: rgba(217,119,6,0.10);
      color: var(--warn);
      border-color: rgba(217,119,6,0.14);
    }
    .secondary-button-danger {
      background: rgba(220,38,38,0.10);
      color: var(--danger);
      border-color: rgba(220,38,38,0.14);
    }
    .filekey-lookup {
      display: grid;
      gap: 10px;
      padding: 12px;
      border-radius: 16px;
      border: 1px solid rgba(15,111,255,0.14);
      background: rgba(248,251,255,0.72);
    }
    .filekey-lookup-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
    }
    .filekey-lookup-head strong,
    .filekey-lookup-head span {
      display: block;
    }
    .filekey-lookup-head strong {
      font-size: 13px;
      line-height: 1.35;
    }
    .filekey-lookup-head span {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .filekey-lookup-head span[data-tone="good"] {
      color: var(--good);
    }
    .filekey-lookup-head span[data-tone="accent"] {
      color: var(--accent);
    }
    .filekey-lookup-head span[data-tone="warn"] {
      color: var(--warn);
    }
    .filekey-lookup-head span[data-tone="danger"] {
      color: var(--danger);
    }
    .filekey-lookup-summary,
    .filekey-lookup-zip-groups,
    .filekey-lookup-filters,
    .filekey-lookup-groups {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .filekey-lookup-summary:empty,
    .filekey-lookup-zip-groups:empty,
    .filekey-lookup-filters:empty,
    .filekey-lookup-groups:empty,
    .filekey-lookup-list:empty {
      display: none;
    }
    .filekey-chip,
    .filekey-group {
      min-width: 0;
      padding: 9px 10px;
      border-radius: 12px;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(255,255,255,0.72);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .filekey-chip span,
    .filekey-group span {
      display: block;
      color: var(--muted);
      font-weight: 700;
    }
    .filekey-chip strong,
    .filekey-group strong {
      display: block;
      margin-top: 2px;
      color: var(--ink);
      font-size: 14px;
    }
    .filekey-chip-trainable {
      border-color: rgba(5,150,105,0.24);
      background: rgba(5,150,105,0.08);
    }
    .filekey-chip-trainable span,
    .filekey-chip-trainable strong {
      color: var(--good);
    }
    .filekey-chip-trained {
      border-color: rgba(67,56,202,0.24);
      background: rgba(67,56,202,0.08);
    }
    .filekey-chip-trained span,
    .filekey-chip-trained strong {
      color: #4338ca;
    }
    .filekey-chip-prepared {
      border-color: rgba(15,111,255,0.22);
      background: rgba(15,111,255,0.08);
    }
    .filekey-chip-prepared span,
    .filekey-chip-prepared strong {
      color: var(--accent);
    }
    .filekey-group em {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-style: normal;
      font-size: 11px;
      line-height: 1.35;
    }
    .filekey-zip-group {
      border-color: rgba(15,111,255,0.20);
      background: rgba(15,111,255,0.07);
    }
    .filekey-class-group {
      border-color: rgba(5,150,105,0.20);
      background: rgba(5,150,105,0.06);
    }
    .filekey-lookup-filters {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .filekey-filter-option {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      align-items: center;
      gap: 8px;
      min-width: 0;
      padding: 9px 10px;
      border-radius: 12px;
      border: 1px solid rgba(15,111,255,0.16);
      background: rgba(255,255,255,0.78);
      font-size: 12px;
      font-weight: 800;
      color: var(--ink);
    }
    .filekey-filter-option span {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .filekey-filter-option strong {
      color: var(--accent);
      font-size: 12px;
    }
    .filekey-lookup-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .filekey-lookup-actions:empty {
      display: none;
    }
    .filekey-lookup-list {
      display: grid;
      gap: 8px;
      max-height: 420px;
      overflow: auto;
      padding-right: 2px;
    }
    .filekey-class-section {
      min-width: 0;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(248,251,255,0.78);
      overflow: visible;
    }
    .filekey-class-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 11px 12px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
      background: rgba(255,255,255,0.72);
    }
    .filekey-class-select {
      appearance: none;
      min-width: 0;
      padding: 0;
      border: 0;
      background: transparent;
      color: var(--ink);
      font-size: 13px;
      font-weight: 900;
      text-align: left;
      overflow-wrap: anywhere;
      cursor: pointer;
    }
    .filekey-class-select:hover,
    .filekey-class-select:focus-visible {
      color: var(--accent);
      text-decoration: underline;
    }
    .filekey-class-meta {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }
    .filekey-class-body {
      display: grid;
      gap: 8px;
      padding: 8px;
    }
    .filekey-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 9px;
      align-items: start;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(255,255,255,0.78);
    }
    .filekey-row input {
      margin-top: 3px;
    }
    .filekey-main {
      min-width: 0;
    }
    .filekey-main strong {
      display: block;
      font-size: 13px;
      line-height: 1.35;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .filekey-main span {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .filekey-status {
      padding: 5px 7px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      line-height: 1.2;
      color: var(--accent);
      background: rgba(15,111,255,0.10);
      white-space: nowrap;
    }
    .filekey-badges {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 5px;
    }
    .filekey-zip-badge {
      padding: 5px 7px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      line-height: 1.2;
      color: #334155;
      background: rgba(148,163,184,0.16);
      white-space: nowrap;
    }
    .filekey-status.trainable {
      color: var(--good);
      background: rgba(5,150,105,0.10);
    }
    .filekey-status.excluded {
      color: var(--danger);
      background: rgba(220,38,38,0.10);
    }
    .filekey-status.trained {
      color: #4338ca;
      background: rgba(67,56,202,0.12);
    }
    .filekey-status.prepared {
      color: var(--accent);
      background: rgba(15,111,255,0.12);
    }
    .filekey-status.queued,
    .filekey-status.running {
      color: var(--accent);
      background: rgba(15,111,255,0.12);
    }
    .filekey-status.unknown {
      color: var(--warn);
      background: rgba(217,119,6,0.10);
    }
    .key-metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 12px;
    }
    .key-metric-grid div {
      padding: 12px 13px;
      border-radius: 14px;
      background: rgba(248,251,255,0.9);
      border: 1px solid rgba(148,163,184,0.16);
      min-width: 0;
    }
    .key-metric-grid .key-metric-wide {
      grid-column: 1 / -1;
    }
    .key-metric-grid span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .key-metric-grid strong {
      display: block;
      color: var(--ink);
      font-size: 14px;
      line-height: 1.35;
    }
    .path-value-inline {
      max-width: 100%;
      font-size: 12px;
      line-height: 1.5;
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    .path-box {
      padding: 14px;
      border-radius: 14px;
      background: rgba(248,251,255,0.92);
      border: 1px solid rgba(148,163,184,0.16);
    }
    .path-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 6px;
    }
    .path-value {
      color: var(--ink);
      font-size: 13px;
      line-height: 1.5;
      word-break: break-word;
    }
    .link-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .link-row a {
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
      font-weight: 800;
    }
    .content-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }
    .chart-panel-span {
      grid-column: 1 / -1;
    }
    .chart-wrap {
      display: grid;
      gap: 14px;
    }
    .metric-chart {
      width: 100%;
      height: 290px;
      display: block;
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(248,251,255,0.98), rgba(241,246,252,0.96));
      border: 1px solid rgba(148,163,184,0.14);
    }
    .chart-grid-line {
      stroke: rgba(148,163,184,0.18);
      stroke-width: 1;
    }
    .chart-axis {
      stroke: rgba(100,116,139,0.42);
      stroke-width: 1.2;
    }
    .chart-axis-label {
      fill: #64748b;
      font-size: 11px;
      font-weight: 700;
    }
    .chart-point {
      filter: drop-shadow(0 2px 4px rgba(15, 23, 42, 0.12));
    }
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 11px;
      border-radius: 999px;
      background: rgba(248,251,255,0.92);
      border: 1px solid rgba(148,163,184,0.16);
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }
    .legend-item strong {
      color: var(--ink);
      font-weight: 800;
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: none;
      box-shadow: 0 0 0 3px rgba(255,255,255,0.72);
    }
    .chart-empty {
      padding: 18px;
      border-radius: 16px;
      border: 1px dashed rgba(148,163,184,0.3);
      background: rgba(248,251,255,0.9);
      color: var(--muted);
      text-align: center;
      font-weight: 700;
    }
    .bar-chart {
      display: grid;
      gap: 12px;
    }
    .bar-row {
      display: grid;
      gap: 6px;
    }
    .bar-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
    }
    .bar-meta span {
      color: var(--ink);
      font-weight: 700;
    }
    .bar-meta strong {
      color: var(--muted);
      font-weight: 800;
    }
    .bar-track {
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: rgba(203,213,225,0.34);
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #0f6fff, #34c3ff);
      box-shadow: 0 8px 18px rgba(15,111,255,0.18);
    }
    .confusion-panel .panel-body {
      display: grid;
      gap: 14px;
    }
    .confusion-summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .confusion-kpi {
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(248,251,255,0.92);
      border: 1px solid rgba(148,163,184,0.16);
    }
    .confusion-kpi span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 6px;
    }
    .confusion-kpi strong {
      display: block;
      color: var(--ink);
      font-size: 18px;
      line-height: 1.3;
    }
    .confusion-kpi-wide {
      grid-column: span 1;
    }
    .confusion-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .confusion-legend span {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .legend-box {
      width: 12px;
      height: 12px;
      border-radius: 4px;
      display: inline-block;
      border: 1px solid rgba(148,163,184,0.16);
    }
    .legend-box-diag {
      background: rgba(5, 150, 105, 0.5);
    }
    .legend-box-error {
      background: rgba(220, 38, 38, 0.4);
    }
    .legend-box-total {
      background: rgba(15, 23, 42, 0.08);
    }
    .matrix-wrap {
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(252,253,255,0.96);
    }
    .matrix-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 760px;
      table-layout: fixed;
    }
    .matrix-table th,
    .matrix-table td {
      border-right: 1px solid rgba(148,163,184,0.10);
      border-bottom: 1px solid rgba(148,163,184,0.10);
      vertical-align: middle;
    }
    .matrix-corner,
    .matrix-col-head,
    .matrix-row-head,
    .matrix-total-head {
      background: rgba(246,249,253,0.98);
    }
    .matrix-corner {
      position: sticky;
      top: 0;
      left: 0;
      z-index: 4;
      min-width: 160px;
      padding: 14px 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      text-align: center;
    }
    .matrix-col-head {
      position: sticky;
      top: 0;
      z-index: 3;
      padding: 12px 10px;
      min-width: 86px;
      text-align: center;
    }
    .matrix-col-label {
      font-size: 12px;
      font-weight: 800;
      color: var(--ink);
      word-break: break-word;
    }
    .matrix-row-head {
      position: sticky;
      left: 0;
      z-index: 2;
      min-width: 170px;
      padding: 12px 14px;
      text-align: left;
    }
    .matrix-row-label {
      font-size: 13px;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 4px;
    }
    .matrix-row-meta {
      font-size: 11px;
      font-weight: 700;
      color: var(--muted);
    }
    .matrix-cell {
      min-width: 86px;
      padding: 10px 8px;
      text-align: center;
      transition: background 0.16s ease;
    }
    .matrix-cell.is-diagonal {
      box-shadow: inset 0 0 0 1px rgba(5,150,105,0.12);
    }
    .matrix-cell.is-zero {
      color: rgba(100,116,139,0.9);
    }
    .matrix-count {
      display: block;
      color: var(--ink);
      font-size: 15px;
      font-weight: 900;
      line-height: 1.1;
    }
    .matrix-share {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }
    .matrix-total-head {
      position: sticky;
      top: 0;
      z-index: 3;
      min-width: 92px;
      padding: 12px 10px;
      text-align: center;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
    }
    .matrix-total-cell {
      min-width: 92px;
      padding: 10px 8px;
      text-align: center;
      background: rgba(15,23,42,0.05);
    }
    .matrix-total-count {
      display: block;
      color: var(--ink);
      font-size: 14px;
      font-weight: 900;
      line-height: 1.15;
    }
    .matrix-total-meta {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .matrix-grand-total {
      background: rgba(15,111,255,0.10);
    }
    .mini-panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel-soft);
      padding: 16px;
    }
    .mini-panel h3 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: -0.01em;
    }
    .table-wrap {
      overflow: auto;
      border-radius: 16px;
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(252,253,255,0.94);
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 480px;
    }
    .data-table th,
    .data-table td {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(148,163,184,0.12);
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    .data-table th {
      position: sticky;
      top: 0;
      background: rgba(246,249,253,0.98);
      color: var(--muted);
      font-weight: 800;
      z-index: 1;
    }
    .data-table tbody tr:last-child td {
      border-bottom: none;
    }
    .data-table-compact {
      min-width: 0;
    }
    .data-table-compact th,
    .data-table-compact td {
      padding: 10px 12px;
      font-size: 12px;
    }
    .empty-cell {
      color: var(--muted);
    }
    .queue-list {
      display: grid;
      gap: 10px;
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .queue-summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .queue-summary-tile {
      padding: 12px 13px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.16);
      background: linear-gradient(180deg, rgba(248,251,255,0.98), rgba(241,247,255,0.92));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }
    .queue-summary-tile span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      margin-bottom: 4px;
    }
    .queue-summary-tile strong {
      display: block;
      color: var(--ink);
      font-size: 16px;
      font-weight: 900;
      letter-spacing: -0.01em;
    }
    .queue-progress-card {
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(148,163,184,0.18);
      background: linear-gradient(180deg, rgba(248,251,255,0.98), rgba(237,244,255,0.92));
    }
    .queue-progress-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .queue-progress-copy {
      min-width: 0;
      display: grid;
      gap: 4px;
    }
    .queue-progress-copy span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .queue-progress-copy strong {
      color: var(--ink);
      font-size: 15px;
      font-weight: 900;
      line-height: 1.35;
      word-break: break-word;
    }
    .queue-progress-percent {
      flex: 0 0 auto;
      color: var(--accent);
      font-size: 22px;
      font-weight: 950;
      line-height: 1;
      letter-spacing: -0.03em;
    }
    .queue-progress-bar {
      height: 10px;
      border-radius: 999px;
      background: rgba(148,163,184,0.18);
      overflow: hidden;
      box-shadow: inset 0 1px 2px rgba(15,23,42,0.08);
    }
    .queue-progress-fill {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #0f6fff 0%, #33a1ff 100%);
      box-shadow: 0 6px 14px rgba(15,111,255,0.24);
    }
    .queue-item,
    .queue-empty {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 13px 14px;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(248,251,255,0.92);
    }
    .queue-main {
      min-width: 0;
    }
    .queue-main strong {
      display: block;
      margin-bottom: 4px;
    }
    .queue-copy,
    .muted-block {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      word-break: break-word;
    }
    .detail-panel {
      border-radius: 20px;
      border: 1px solid rgba(148,163,184,0.18);
      background: rgba(255,255,255,0.92);
      box-shadow: var(--shadow-soft);
    }
    .detail-summary {
      cursor: pointer;
      list-style: none;
      padding: 16px 18px;
      font-size: 15px;
      font-weight: 900;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.92));
    }
    .detail-summary::-webkit-details-marker {
      display: none;
    }
    .detail-body {
      padding: 0 18px 18px;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }
    .log-stack {
      display: grid;
      gap: 14px;
    }
    .log-card {
      border-radius: 18px;
      border: 1px solid rgba(148,163,184,0.16);
      background: rgba(248,251,255,0.96);
      overflow: hidden;
    }
    .log-card summary {
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 900;
      list-style: none;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.92));
    }
    .log-card summary::-webkit-details-marker {
      display: none;
    }
    .log-box {
      margin: 10px 0 0;
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
    }
    code {
      font-family: "Consolas", "SFMono-Regular", monospace;
      font-size: 12px;
    }
    @media (max-width: 1280px) {
      .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .sidebar {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .queue-summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .chart-grid {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 860px) {
      .page {
        padding: 16px 14px 28px;
      }
      .hero {
        grid-template-columns: 1fr;
        padding: 20px;
      }
      .hero h1 {
        font-size: 30px;
      }
      .hero-kpis,
      .summary-grid,
      .chart-grid,
      .confusion-summary,
      .content-row,
      .detail-grid,
      .insight-grid,
      .key-metric-grid,
      .queue-summary-grid,
      .sidebar,
      .action-grid {
        grid-template-columns: 1fr;
      }
      .insight-item {
        grid-template-columns: 1fr;
      }
      .filekey-class-header,
      .filekey-row {
        grid-template-columns: 1fr;
      }
      .filekey-class-meta {
        white-space: normal;
      }
      .filekey-badges {
        align-items: flex-start;
        flex-direction: row;
        flex-wrap: wrap;
      }
      .sidebar > .sidebar-panel:first-child {
        grid-row: auto;
      }
      .section-nav {
        top: 8px;
      }
      .data-table {
        min-width: 0;
      }
    }
  </style>
"""


def render_dashboard_live_fragments(overview: dict) -> dict[str, str]:
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
    logs = overview.get("logs") or {}
    continual_state = overview.get("continual_state") or {}
    artifacts = overview.get("artifacts") or {}
    skip_report = overview.get("skip_report") or {}
    cumulative_skip_report = overview.get("cumulative_skip_report") or {}
    insights = overview.get("insights") or {}
    labels = progress.get("labels") or metrics.get("labels") or []
    final_validation = progress.get("final_validation") or metrics.get("final_validation") or {}
    completed_jobs = launcher.get("completed_jobs") or []

    raw_total = int(((dataset.get("raw") or {}).get("total") or 0))
    prepared_total = sum(
        int(((dataset.get(key) or {}).get("total") or 0))
        for key in ("prepared_train", "prepared_val", "prepared_test")
    )
    updated_at = progress.get("updated_at") or pipeline.get("updated_at") or "-"
    latest = progress.get("latest") or {}
    latest_loss = _float_text(latest.get("val_loss"))
    dataset_rows = _build_dataset_rows(dataset)
    current_dataset_rows = _build_dataset_rows(current_dataset)
    history_bundle = _build_history_bundle(progress, metrics)
    history_rows = history_bundle["rows"]
    confusion = final_validation.get("confusion_matrix") or []
    support_map = _build_support_map(confusion if isinstance(confusion, list) else None)
    per_class_rows = _build_per_class_rows(final_validation, labels, support_map)
    confusion_matrix_html = _render_confusion_matrix(labels, confusion if isinstance(confusion, list) else None)
    recent_job_rows = _build_recent_job_rows(completed_jobs)
    issue_rows = _build_issue_rows(skip_report.get("issues") or [])
    train_distribution = progress.get("train_distribution") or metrics.get("train_distribution") or {}
    val_distribution = progress.get("val_distribution") or metrics.get("val_distribution") or {}
    epoch_labels = history_bundle["epoch_labels"]
    val_accuracy_values = history_bundle["val_accuracy_values"]
    val_f1_values = history_bundle["val_f1_values"]
    train_loss_values = history_bundle["train_loss_values"]
    val_loss_values = history_bundle["val_loss_values"]
    distribution_entries = _build_distribution_entries(dataset, labels)
    performance_chart_html = _render_line_chart(
        "성능 추이",
        "epoch별 검증 정확도와 macro F1을 바로 비교할 수 있습니다.",
        epoch_labels,
        [
            ("검증 정확도", "#2563eb", val_accuracy_values),
            ("검증 macro F1", "#059669", val_f1_values),
        ],
        fixed_min=0.0,
        fixed_max=1.0,
        decimals=3,
    )
    loss_chart_html = _render_line_chart(
        "손실 추이",
        "train / validation loss 변화를 함께 보면서 과적합 여부를 판단할 수 있습니다.",
        epoch_labels,
        [
            ("학습 손실", "#2563eb", train_loss_values),
            ("검증 손실", "#f97316", val_loss_values),
        ],
        decimals=4,
    )
    distribution_chart_html = _render_distribution_chart(
        "학습 데이터 분포",
        "전처리된 학습 세트 기준 클래스별 샘플 수입니다.",
        distribution_entries,
    )
    summary_cards = _build_summary_card_items(
        overview=overview,
        pipeline=pipeline,
        launcher=launcher,
        progress=progress,
        queue_progress=queue_progress,
        current_job_progress=current_job_progress,
        eta=eta,
        gpu=gpu,
        raw_total=raw_total,
        prepared_total=prepared_total,
    )

    return {
        "hero_kpis": _render_hero_kpis(
            updated_at=updated_at,
            progress=progress,
            skip_report=skip_report,
            continual_state=continual_state,
            element_id="hero-kpis",
        ),
        "hero_status": _render_hero_status_card(
            pipeline=pipeline,
            launcher=launcher,
            gpu=gpu,
            latest_loss=latest_loss,
            element_id="hero-status-card",
        ),
        "summary_cards": _render_summary_grid(summary_cards, element_id="summary-grid"),
        "system_panel": _render_system_panel(
            overview,
            overview.get("config_path") or "",
            artifacts,
            gpu,
            progress,
            queue_progress,
            element_id="system-panel",
        ),
        "main_live_sections": _render_main_live_sections(
            summary_cards=_render_summary_cards(summary_cards),
            insights=insights,
            performance_chart_html=performance_chart_html,
            loss_chart_html=loss_chart_html,
            distribution_chart_html=distribution_chart_html,
            progress=progress,
            latest=latest,
            raw_total=raw_total,
            prepared_total=prepared_total,
            train_distribution=train_distribution,
            val_distribution=val_distribution,
            skip_report=skip_report,
            cumulative_skip_report=cumulative_skip_report,
            dataset_rows=dataset_rows,
            current_dataset_rows=current_dataset_rows,
            history_rows=history_rows,
            per_class_rows=per_class_rows,
            confusion_matrix_html=confusion_matrix_html,
            recent_job_rows=recent_job_rows,
            issue_rows=issue_rows,
            element_id="main-live-sections",
        ),
        "queue_panel": _render_queue_panel(
            launcher,
            current_job_progress,
            queue_progress,
            element_id="queue-panel",
        ),
        "logs": _render_logs_stack(logs, element_id="logs-stack"),
    }


def render_dashboard_page(
    overview: dict,
    *,
    config_path: str,
    default_datasetkey: str = "",
    controls_enabled: bool = True,
    controls_notice: str | None = None,
    notice: str | None = None,
    notice_level: str = "info",
    refresh_seconds: int = 0,
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
    insights = overview.get("insights") or {}
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
    latest_loss = _float_text(latest.get("val_loss"))

    refresh_seconds = max(0, min(int(refresh_seconds), 300))
    meta_refresh = f"<meta http-equiv=\"refresh\" content=\"{refresh_seconds};url=/\" />" if refresh_seconds > 0 else ""

    banners = []
    if notice:
        notice_tone = _state_tone(notice_level)
        banners.append(_render_banner(notice, notice_tone, "작업 결과" if notice_tone != "danger" else "오류"))
    warning_banner = _render_warning_banner(diagnostics)
    if warning_banner:
        banners.append(warning_banner)
    banners_html = "".join(banners)

    summary_card_items = _build_summary_card_items(
        overview=overview,
        pipeline=pipeline,
        launcher=launcher,
        progress=progress,
        queue_progress=queue_progress,
        current_job_progress=current_job_progress,
        eta=eta,
        gpu=gpu,
        raw_total=raw_total,
        prepared_total=prepared_total,
    )
    summary_cards = _render_summary_cards(summary_card_items)
    hero_kpis = _render_hero_kpis(
        updated_at=updated_at,
        progress=progress,
        skip_report=skip_report,
        continual_state=continual_state,
        element_id="hero-kpis",
    )
    dataset_rows = _build_dataset_rows(dataset)
    current_dataset_rows = _build_dataset_rows(current_dataset)

    history_bundle = _build_history_bundle(progress, metrics)
    history_rows = history_bundle["rows"]

    confusion = final_validation.get("confusion_matrix") or []
    support_map = _build_support_map(confusion if isinstance(confusion, list) else None)
    per_class_rows = _build_per_class_rows(final_validation, labels, support_map)
    confusion_matrix_html = _render_confusion_matrix(labels, confusion if isinstance(confusion, list) else None)

    recent_job_rows = _build_recent_job_rows(completed_jobs)
    issue_rows = _build_issue_rows(skip_report.get("issues") or [])

    train_distribution = progress.get("train_distribution") or metrics.get("train_distribution") or {}
    val_distribution = progress.get("val_distribution") or metrics.get("val_distribution") or {}

    epoch_labels = history_bundle["epoch_labels"]
    val_accuracy_values = history_bundle["val_accuracy_values"]
    val_f1_values = history_bundle["val_f1_values"]
    train_loss_values = history_bundle["train_loss_values"]
    val_loss_values = history_bundle["val_loss_values"]

    distribution_entries = _build_distribution_entries(dataset, labels)

    performance_chart_html = _render_line_chart(
        "성능 추이",
        "epoch별 검증 정확도와 macro F1을 바로 비교할 수 있습니다.",
        epoch_labels,
        [
            ("검증 정확도", "#2563eb", val_accuracy_values),
            ("검증 macro F1", "#059669", val_f1_values),
        ],
        fixed_min=0.0,
        fixed_max=1.0,
        decimals=3,
    )
    loss_chart_html = _render_line_chart(
        "손실 추이",
        "train / validation loss 변화를 함께 보면서 과적합 여부를 판단할 수 있습니다.",
        epoch_labels,
        [
            ("학습 손실", "#2563eb", train_loss_values),
            ("검증 손실", "#f97316", val_loss_values),
        ],
        decimals=4,
    )
    distribution_chart_html = _render_distribution_chart(
        "학습 데이터 분포",
        "전처리된 학습 세트 기준 클래스별 샘플 수입니다.",
        distribution_entries,
    )
    main_live_sections_html = _render_main_live_sections(
        summary_cards=summary_cards,
        insights=insights,
        performance_chart_html=performance_chart_html,
        loss_chart_html=loss_chart_html,
        distribution_chart_html=distribution_chart_html,
        progress=progress,
        latest=latest,
        raw_total=raw_total,
        prepared_total=prepared_total,
        train_distribution=train_distribution,
        val_distribution=val_distribution,
        skip_report=skip_report,
        cumulative_skip_report=cumulative_skip_report,
        dataset_rows=dataset_rows,
        current_dataset_rows=current_dataset_rows,
        history_rows=history_rows,
        per_class_rows=per_class_rows,
        confusion_matrix_html=confusion_matrix_html,
        recent_job_rows=recent_job_rows,
        issue_rows=issue_rows,
        element_id="main-live-sections",
    )

    return f"""<!DOCTYPE html>
<html lang="ko" data-overview-revision="{_text(overview.get('overview_revision'), '')}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {meta_refresh}
  <title>detectWarning 학습 대시보드</title>
  {_styles()}
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
      <aside class="status-card" id="hero-status-card">
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

    <div id="action-notice-region"></div>
    {banners_html}

      {_render_section_nav()}

      <div class="layout">
        <aside class="sidebar">
          {_render_actions_panel(default_datasetkey, controls_enabled, controls_notice)}
          {_render_queue_panel(launcher, current_job_progress, queue_progress, element_id="queue-panel")}
          {_render_system_panel(overview, config_path, artifacts, gpu, progress, queue_progress, element_id="system-panel")}
        </aside>

        <main class="content">
          {main_live_sections_html}

          <section class="panel" id="logs">
            <div class="panel-head"><div><h2>로그</h2><p>필요한 로그만 열어 볼 수 있게 접이식으로 정리했습니다.</p></div></div>
            <div class="panel-body">
            <div class="log-stack" id="logs-stack">
              {_render_logs(logs)}
            </div>
          </div>
        </section>

        {_render_diagnostics_details(diagnostics)}
      </main>
    </div>
  </div>
  {_render_live_refresh_script()}
</body>
</html>
"""

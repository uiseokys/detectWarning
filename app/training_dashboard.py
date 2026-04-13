from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import uvicorn
from fastapi import FastAPI
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


def create_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    app = FastAPI(title="detectWarning Training Dashboard")

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
      --panel: rgba(255, 255, 255, 0.92);
      --panel-soft: rgba(248, 250, 252, 0.96);
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
      max-width: 1440px;
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
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .card,
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(14px);
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
    }
    .table th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .mono {
      font-family: "SF Mono", "JetBrains Mono", monospace;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }
    .empty {
      padding: 18px;
      border-radius: 18px;
      background: rgba(255,255,255,0.68);
      border: 1px dashed rgba(148,163,184,0.30);
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .main-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .wrap { padding: 18px; }
      .hero { flex-direction: column; align-items: stretch; }
      .grid, .two-col { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div>
        <h1>행동 학습 진행 대시보드</h1>
        <p>데이터 다운로드, pose 전처리, 학습 epoch 진행 상황과 현재 validation 성능을 한 화면에서 확인합니다.</p>
      </div>
      <div id="pipelineState" class="status-pill tone-neutral">상태 확인 중</div>
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
  </div>
  <script>
    function toneClass(state) {
      if (state === 'completed') return 'tone-good';
      if (state === 'running') return 'tone-accent';
      if (state === 'error') return 'tone-danger';
      if (state === 'download' || state === 'prepare' || state === 'train') return 'tone-warn';
      return 'tone-neutral';
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

    async function refresh() {
      const response = await fetch('/api/overview');
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      const pipeline = data.pipeline_status || {};
      const progress = data.training_progress || {};

      const stateEl = document.getElementById('pipelineState');
      stateEl.textContent = pipeline.state || 'unknown';
      stateEl.className = `status-pill ${toneClass(pipeline.state)}`;

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

      if (progress.latest) {
        document.getElementById('latestEpoch').textContent = `Epoch ${progress.latest.epoch}`;
        document.getElementById('latestMetrics').textContent =
          `train loss ${progress.latest.train_loss} / val acc ${progress.latest.val_accuracy} / val f1 ${progress.latest.val_macro_f1}`;
      } else {
        document.getElementById('latestEpoch').textContent = '-';
        document.getElementById('latestMetrics').textContent = '-';
      }

      document.getElementById('updatedAt').textContent = pipeline.updated_at || progress.updated_at || '-';
      document.getElementById('configPath').textContent = data.config_path || '-';

      renderChart(progress.history || []);
      renderDatasetTable(data.dataset || {});
    }

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
        return build_overview(paths, config_path)

    return app


def build_overview(paths: dict, config_path: Path) -> dict:
    return {
        "workspace_dir": str(paths["workspace_dir"]),
        "config_path": str(config_path),
        "pipeline_status": read_json(paths["pipeline_status"]),
        "training_progress": read_json(paths["training_progress"]),
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


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    app = create_app(config_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

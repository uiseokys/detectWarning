from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
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


def create_app(config_path: Path) -> FastAPI:
    config = load_config(config_path)
    paths = resolve_paths(config, config_path.parent)
    project_root = Path(__file__).resolve().parent.parent
    pipeline_script = Path(__file__).resolve().with_name("action_training_pipeline.py")
    launch_log_path = paths["workspace_dir"] / "launcher.log"
    runtime_config_dir = paths["workspace_dir"] / "runtime_configs"
    runtime_config_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="detectWarning Training Dashboard")

    launcher_state: dict[str, object] = {
        "process": None,
        "started_at": None,
        "runtime_config_path": None,
        "requested_filekeys": [],
        "last_state": "idle",
        "last_exit_code": None,
        "last_message": "아직 실행 기록이 없습니다.",
    }

    def current_timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def get_launcher_status() -> dict:
        process = launcher_state.get("process")
        if process is not None and isinstance(process, subprocess.Popen):
            exit_code = process.poll()
            if exit_code is None:
                launcher_state["last_state"] = "running"
                launcher_state["last_message"] = "다운로드 + 압축 해제 + 학습 파이프라인이 실행 중입니다."
            else:
                launcher_state["process"] = None
                launcher_state["last_exit_code"] = exit_code
                if exit_code == 0:
                    launcher_state["last_state"] = "completed"
                    launcher_state["last_message"] = "파이프라인이 정상 완료되었습니다."
                else:
                    launcher_state["last_state"] = "error"
                    launcher_state["last_message"] = f"파이프라인이 종료 코드 {exit_code}로 중단되었습니다."

        active_process = launcher_state.get("process")
        active_pid = active_process.pid if isinstance(active_process, subprocess.Popen) else None
        return {
            "state": launcher_state.get("last_state", "idle"),
            "message": launcher_state.get("last_message", ""),
            "pid": active_pid,
            "started_at": launcher_state.get("started_at"),
            "runtime_config_path": str(launcher_state["runtime_config_path"])
            if launcher_state.get("runtime_config_path")
            else None,
            "requested_filekeys": list(launcher_state.get("requested_filekeys", [])),
            "last_exit_code": launcher_state.get("last_exit_code"),
            "log_path": str(launch_log_path),
        }

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
      grid-template-columns: repeat(4, minmax(0, 1fr));
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
    @media (max-width: 1200px) {
      .control-panel,
      .main-grid {
        grid-template-columns: 1fr;
      }
      .grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
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
        <p>AIHub 분할 ZIP의 filekey를 직접 넣고, 다운로드부터 압축 해제, pose 전처리, 행동 학습까지 한 번에 실행할 수 있습니다.</p>
      </div>
      <div id="pipelineState" class="status-pill tone-neutral">상태 확인 중</div>
    </section>

    <section class="card control-panel">
      <div>
        <h2 class="control-title">AIHub filekey로 바로 학습 시작</h2>
        <div class="control-copy">
          여러 개의 분할 ZIP 파일 키를 쉼표 또는 줄바꿈으로 넣으면 됩니다.
          대시보드가 작업 폴더를 초기화한 뒤, 선택한 파일만 다운로드하고 자동으로 압축을 해제해서 학습을 시작합니다.
        </div>
        <div class="meta-row">
          <div class="meta-chip">datasetkey <span id="datasetKeyChip">-</span></div>
          <div class="meta-chip">workspace <span id="workspaceChip">-</span></div>
        </div>
        <label class="form-label" for="filekeysInput">분할 ZIP filekey 입력</label>
        <textarea id="filekeysInput" class="input-area" placeholder="예:&#10;123456&#10;123457&#10;123458"></textarea>
        <div class="control-actions">
          <button id="startButton" class="primary-button" type="button">다운로드 + 압축 해제 + 학습 시작</button>
          <div class="helper">실행 중에는 같은 대시보드에서 중복 시작이 막힙니다.</div>
        </div>
      </div>
      <div class="launch-box">
        <div class="launch-item">
          <div class="launch-label">런처 상태</div>
          <div class="launch-value" id="launcherState">대기 중</div>
        </div>
        <div class="launch-item">
          <div class="launch-label">최근 요청 filekey</div>
          <div class="launch-value" id="requestedFilekeys">-</div>
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

    function formatFilekeys(filekeys) {
      if (!filekeys || !filekeys.length) {
        return '-';
      }
      return filekeys.join(', ');
    }

    function setLaunchMessage(message, isError) {
      const box = document.getElementById('launchMessage');
      box.textContent = message || '-';
      box.style.color = isError ? '#dc2626' : '#64748b';
      box.style.borderColor = isError ? 'rgba(220,38,38,0.18)' : 'rgba(148,163,184,0.18)';
      box.style.background = isError ? 'rgba(220,38,38,0.06)' : 'rgba(255,255,255,0.84)';
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

    async function startTraining() {
      const input = document.getElementById('filekeysInput').value.trim();
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
          body: JSON.stringify({ filekeys: input }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.message || '학습 시작에 실패했습니다.');
        }
        setLaunchMessage(data.message || '학습을 시작했습니다.', false);
        await refresh();
      } catch (error) {
        setLaunchMessage(error.message || String(error), true);
      } finally {
        button.disabled = false;
        button.textContent = '다운로드 + 압축 해제 + 학습 시작';
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

      const stateEl = document.getElementById('pipelineState');
      stateEl.textContent = pipeline.state || 'unknown';
      stateEl.className = `status-pill ${toneClass(pipeline.state)}`;

      document.getElementById('datasetKeyChip').textContent = data.aihub?.datasetkey ?? '-';
      document.getElementById('workspaceChip').textContent = data.workspace_name || '-';
      document.getElementById('launcherState').textContent = launcher.state || 'idle';
      document.getElementById('requestedFilekeys').textContent = formatFilekeys(launcher.requested_filekeys || []);
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
    }

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

        launcher = get_launcher_status()
        if launcher["state"] == "running":
            raise HTTPException(status_code=409, detail="이미 학습 파이프라인이 실행 중입니다.")

        payload = await request.json()
        filekeys = parse_filekeys(payload.get("filekeys", ""))
        if not filekeys:
            raise HTTPException(status_code=400, detail="filekey를 하나 이상 입력해 주세요.")

        datasetkey = config.get("aihub_shell", {}).get("datasetkey")
        if datasetkey in (None, ""):
            raise HTTPException(status_code=400, detail="설정 파일에 aihub_shell.datasetkey 가 필요합니다.")

        reset_training_workspace(paths)
        runtime_config_dir.mkdir(parents=True, exist_ok=True)

        runtime_config = json.loads(json.dumps(config))
        runtime_config["dataset_source"] = "aihub_shell"
        runtime_shell = runtime_config.setdefault("aihub_shell", {})
        runtime_shell["filekey"] = filekeys if len(filekeys) > 1 else filekeys[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runtime_config_path = runtime_config_dir / f"launch_{timestamp}.json"
        with runtime_config_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_config, handle, ensure_ascii=False, indent=2)

        with launch_log_path.open("w", encoding="utf-8") as log_handle:
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

        launcher_state["process"] = process
        launcher_state["started_at"] = current_timestamp()
        launcher_state["runtime_config_path"] = runtime_config_path
        launcher_state["requested_filekeys"] = filekeys
        launcher_state["last_state"] = "running"
        launcher_state["last_exit_code"] = None
        launcher_state["last_message"] = "AIHub 다운로드와 학습 파이프라인을 시작했습니다."

        return {
            "ok": True,
            "message": f"filekey {', '.join(filekeys)} 기준으로 학습을 시작했습니다.",
            "launcher": get_launcher_status(),
        }

    return app


def build_overview(paths: dict, config_path: Path, *, config: dict, launcher_status: dict | None = None) -> dict:
    return {
        "workspace_dir": str(paths["workspace_dir"]),
        "workspace_name": paths["workspace_dir"].name,
        "config_path": str(config_path),
        "pipeline_status": read_json(paths["pipeline_status"]),
        "training_progress": read_json(paths["training_progress"]),
        "launcher": launcher_status or {},
        "aihub": {
            "datasetkey": config.get("aihub_shell", {}).get("datasetkey"),
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


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
      let data;
      try {
        const response = await fetch('/api/overview');
        if (!response.ok) {
          let detail = `대시보드 상태를 불러오지 못했습니다. (${response.status})`;
          try {
            const errorPayload = await response.json();
            detail = errorPayload?.detail || errorPayload?.message || detail;
          } catch (parseError) {
            // ignore response parse error
          }
          setLaunchMessage(detail, true);
          return;
        }
        data = await response.json();
      } catch (error) {
        setLaunchMessage(error?.message || '대시보드 상태 요청에 실패했습니다.', true);
        return;
      }
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
  
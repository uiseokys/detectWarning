from __future__ import annotations

import argparse
import base64
import io
import platform
import queue
import shutil
import socket
import subprocess
import threading
from dataclasses import dataclass
from time import monotonic, perf_counter
import wave

import cv2
import numpy as np
import uvicorn
from fastapi import Body
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Query, Request

from detector import FaceDetector, POSE_CONNECTIONS, PersonDetector
from person_classifier import PersonPresenceFilter
from risk_analyzer import RiskAnalyzer
from tracker import PersonTracker

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class ClientSession:
    tracker: PersonTracker
    person_filter: PersonPresenceFilter
    risk_analyzer: RiskAnalyzer
    last_seen: float
    latest_frame_jpeg: bytes | None = None
    latest_people: list[dict] | None = None
    latest_meta: dict | None = None


@dataclass
class AudioJob:
    client_id: str
    wav_bytes: bytes


@dataclass
class ServerSpeechResult:
    status: str
    transcript: str = ""
    audio_level: float = 0.0


class SystemMonitor:
    def __init__(self, args, audio_job_queue: queue.Queue) -> None:
        self.args = args
        self.audio_job_queue = audio_job_queue
        self._lock = threading.Lock()
        self._snapshot = self._collect_snapshot()
        if psutil is not None:
            psutil.cpu_percent(interval=None)

    def start(self) -> None:
        threading.Thread(target=self._run, name="system-monitor", daemon=True).start()

    def get_snapshot(self) -> dict:
        with self._lock:
            snapshot = dict(self._snapshot)
        snapshot["audio_queue_size"] = self.audio_job_queue.qsize()
        return snapshot

    def _run(self) -> None:
        while True:
            snapshot = self._collect_snapshot()
            with self._lock:
                self._snapshot = snapshot
            threading.Event().wait(1.0)

    def _collect_snapshot(self) -> dict:
        snapshot = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "yolo_device": self.args.yolo_device,
            "stt_device": self.args.stt_device,
            "stt_compute_type": self.args.stt_compute_type,
            "stt_beam_size": self.args.stt_beam_size,
            "stt_best_of": self.args.stt_best_of,
            "audio_queue_size": self.audio_job_queue.qsize(),
            "cpu_percent": None,
            "memory_percent": None,
            "memory_used_gb": None,
            "memory_total_gb": None,
            "gpu_name": "",
            "gpu_utilization_percent": None,
            "gpu_memory_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_watts": None,
            "gpu_status": "unavailable",
        }

        if psutil is not None:
            try:
                memory = psutil.virtual_memory()
                snapshot["cpu_percent"] = round(psutil.cpu_percent(interval=None), 1)
                snapshot["memory_percent"] = round(memory.percent, 1)
                snapshot["memory_used_gb"] = round(memory.used / (1024 ** 3), 1)
                snapshot["memory_total_gb"] = round(memory.total / (1024 ** 3), 1)
            except Exception:
                pass

        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            snapshot["gpu_status"] = "nvidia-smi not found"
            return snapshot

        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=True,
            )
            first_line = result.stdout.strip().splitlines()[0]
            parts = [part.strip() for part in first_line.split(",")]
            if len(parts) >= 7:
                snapshot["gpu_name"] = parts[0]
                snapshot["gpu_utilization_percent"] = _to_float(parts[1])
                snapshot["gpu_memory_percent"] = _to_float(parts[2])
                snapshot["gpu_memory_used_mb"] = _to_float(parts[3])
                snapshot["gpu_memory_total_mb"] = _to_float(parts[4])
                snapshot["gpu_temperature_c"] = _to_float(parts[5])
                snapshot["gpu_power_watts"] = _to_float(parts[6])
                snapshot["gpu_status"] = "ok"
        except Exception as exc:
            snapshot["gpu_status"] = f"error: {exc}"

        return snapshot


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


class ServerSpeechRecognizer:
    def __init__(
        self,
        model_size: str,
        compute_type: str,
        language: str,
        device: str,
        beam_size: int,
        best_of: int,
        no_speech_threshold: float,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper를 불러오지 못했습니다. `pip install -r requirements.txt`를 확인해 주세요."
            ) from exc

        self.language = normalize_language(language)
        self.device = device
        self.beam_size = beam_size
        self.best_of = best_of
        self.no_speech_threshold = no_speech_threshold
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )
        self._lock = threading.Lock()

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> tuple[str, float]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) == 0:
            return "", 0.0

        audio_level = float(np.sqrt(np.mean(np.square(audio))))
        with self._lock:
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                vad_filter=False,
                beam_size=self.beam_size,
                best_of=self.best_of,
                no_speech_threshold=self.no_speech_threshold,
                condition_on_previous_text=False,
                temperature=0.0,
            )
        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        return transcript, audio_level


def normalize_language(language: str) -> str:
    normalized = language.strip()
    if "-" in normalized:
        normalized = normalized.split("-", 1)[0]
    return normalized.lower() or "ko"


def draw_server_overlay(frame, client_id: str, faces, latency_ms: float, people_count: int) -> None:
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(
            frame,
            "Face",
            (x, max(y - 10, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )

    cv2.putText(
        frame,
        f"client: {client_id} | people: {people_count} | latency: {latency_ms:.0f}ms",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def draw_pose_overlay(frame, keypoints) -> None:
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
        start = keypoints[start_idx]
        end = keypoints[end_idx]
        if start.get("confidence", 0.0) < 0.35 or end.get("confidence", 0.0) < 0.35:
            continue
        cv2.line(
            frame,
            (int(start["x"]), int(start["y"])),
            (int(end["x"]), int(end["y"])),
            (60, 200, 255),
            2,
        )

    for point in keypoints:
        if point.get("confidence", 0.0) < 0.35:
            continue
        cv2.circle(frame, (int(point["x"]), int(point["y"])), 4, (0, 120, 255), -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원격 영상 추론 서버를 실행합니다.")
    parser.add_argument("--host", default="0.0.0.0", help="서버 바인드 주소")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    parser.add_argument(
        "--person-score-threshold",
        type=float,
        default=0.25,
        help="사람 감지 최소 신뢰도",
    )
    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=640,
        help="YOLO 입력 크기",
    )
    parser.add_argument(
        "--client-session-ttl",
        type=float,
        default=30.0,
        help="클라이언트 추적 상태를 유지할 최대 유휴 시간(초)",
    )
    parser.add_argument(
        "--show-windows",
        action="store_true",
        help="수신한 팀원 카메라 프레임을 데스크탑 OpenCV 창에 표시합니다.",
    )
    parser.add_argument(
        "--stt-model",
        default="medium",
        help="서버 STT용 Whisper 모델 크기",
    )
    parser.add_argument(
        "--stt-compute-type",
        default="int8",
        help="서버 STT용 Whisper 연산 타입",
    )
    parser.add_argument(
        "--stt-beam-size",
        type=int,
        default=3,
        help="서버 STT beam size. 클수록 보통 더 정확하지만 느려집니다.",
    )
    parser.add_argument(
        "--stt-best-of",
        type=int,
        default=3,
        help="서버 STT best_of. 클수록 보통 더 정확하지만 느려집니다.",
    )
    parser.add_argument(
        "--stt-no-speech-threshold",
        type=float,
        default=0.55,
        help="낮출수록 더 많은 오디오를 음성으로 간주합니다.",
    )
    parser.add_argument(
        "--stt-language",
        default="ko-KR",
        help="서버 STT 언어 코드. 예: ko-KR",
    )
    parser.add_argument(
        "--person-debug",
        action="store_true",
        help="사람 후보 상태와 제거 이유를 서버 오버레이/응답에 포함해 디버깅합니다.",
    )
    parser.add_argument(
        "--yolo-device",
        default="cuda:0",
        help="YOLO 추론 장치. 예: cuda:0, cpu",
    )
    parser.add_argument(
        "--stt-device",
        default="cuda",
        help="Whisper 추론 장치. 예: cuda, cpu",
    )
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="detectWarning Inference Server")
    person_detector = PersonDetector(
        score_threshold=args.person_score_threshold,
        resize_width=args.person_imgsz,
        device=args.yolo_device,
    )
    face_detector = FaceDetector()
    speech_recognizer = ServerSpeechRecognizer(
        model_size=args.stt_model,
        compute_type=args.stt_compute_type,
        language=args.stt_language,
        device=args.stt_device,
        beam_size=args.stt_beam_size,
        best_of=args.stt_best_of,
        no_speech_threshold=args.stt_no_speech_threshold,
    )
    sessions: dict[str, ClientSession] = {}
    session_lock = threading.Lock()
    audio_job_queue: queue.Queue[AudioJob] = queue.Queue(maxsize=32)
    system_monitor = SystemMonitor(args, audio_job_queue)
    system_monitor.start()

    def get_session(client_id: str) -> ClientSession:
        now = monotonic()
        with session_lock:
            expired = [
                key
                for key, session in sessions.items()
                if now - session.last_seen > args.client_session_ttl
            ]
            for key in expired:
                del sessions[key]

            session = sessions.get(client_id)
            if session is None:
                session = ClientSession(
                    tracker=PersonTracker(),
                    person_filter=PersonPresenceFilter(debug=args.person_debug),
                    risk_analyzer=RiskAnalyzer(),
                    last_seen=now,
                )
                sessions[client_id] = session
            else:
                session.last_seen = now
            return session

    def list_client_summaries() -> list[dict]:
        with session_lock:
            items = []
            for client_id, session in sessions.items():
                meta = session.latest_meta or {}
                items.append(
                    {
                        "client_id": client_id,
                        "last_seen_seconds": round(monotonic() - session.last_seen, 1),
                        "people_count": int(meta.get("people_count", 0)),
                        "face_count": int(meta.get("face_count", 0)),
                        "latency_ms": float(meta.get("latency_ms", 0.0)),
                        "speech_status": str(meta.get("speech_status", "idle")),
                        "transcript": str(meta.get("transcript", "")),
                        "risk_score": int(meta.get("risk_score", 0)),
                        "risk_level": str(meta.get("risk_level", "LOW")),
                        "risk_level_label": localize_risk_level(str(meta.get("risk_level", "LOW"))),
                        "risk_categories": list(meta.get("risk_categories", [])),
                        "has_frame": session.latest_frame_jpeg is not None,
                    }
                )
            return sorted(items, key=lambda item: item["client_id"])

    def render_dashboard() -> str:
        return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>detectWarning Dashboard</title>
  <style>
    :root {
      --bg-top: #f8fbff;
      --bg-bottom: #ecf2f8;
      --panel: rgba(255, 255, 255, 0.88);
      --panel-strong: rgba(255, 255, 255, 0.96);
      --panel-soft: rgba(248, 251, 255, 0.82);
      --ink: #0f172a;
      --muted: #64748b;
      --line: rgba(148, 163, 184, 0.22);
      --line-strong: rgba(148, 163, 184, 0.35);
      --accent: #2563eb;
      --accent-soft: rgba(37, 99, 235, 0.12);
      --success: #059669;
      --success-soft: rgba(5, 150, 105, 0.12);
      --warn: #d97706;
      --warn-soft: rgba(217, 119, 6, 0.12);
      --danger: #dc2626;
      --danger-soft: rgba(220, 38, 38, 0.12);
      --shadow-lg: 0 24px 60px rgba(15, 23, 42, 0.10);
      --shadow-md: 0 12px 28px rgba(15, 23, 42, 0.08);
      --shadow-sm: 0 8px 18px rgba(15, 23, 42, 0.06);
      --radius-xl: 28px;
      --radius-lg: 22px;
      --radius-md: 18px;
      --radius-sm: 14px;
    }
    * {
      box-sizing: border-box;
    }
    html, body {
      min-height: 100%;
    }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "SF Pro Display", "Pretendard", "SUIT", "Apple SD Gothic Neo", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 24%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
    }
    .wrap {
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 28px 36px;
    }
    .hero {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 24px;
    }
    .hero-copy {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.66);
      border: 1px solid rgba(37, 99, 235, 0.14);
      color: #1d4ed8;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      backdrop-filter: blur(12px);
    }
    .eyebrow::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(37, 99, 235, 0.10);
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(34px, 4vw, 48px);
      line-height: 1.02;
      letter-spacing: -0.04em;
      font-weight: 800;
    }
    .hero p {
      margin: 0;
      max-width: 760px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.65;
    }
    .hero-summary {
      min-width: 240px;
      padding: 16px 18px;
      border-radius: 20px;
      background: linear-gradient(145deg, rgba(255, 255, 255, 0.92), rgba(244, 248, 255, 0.82));
      border: 1px solid rgba(148, 163, 184, 0.22);
      box-shadow: var(--shadow-md);
      backdrop-filter: blur(14px);
    }
    .hero-summary-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .hero-summary-value {
      font-size: 18px;
      font-weight: 700;
      line-height: 1.4;
    }
    .system-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }
    .system-card,
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-sm);
      backdrop-filter: blur(16px);
    }
    .system-card {
      padding: 18px;
      min-height: 132px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .system-title {
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .system-value {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.6;
      white-space: pre-line;
      color: var(--ink);
    }
    .dashboard-grid {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 20px 22px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.52));
    }
    .panel-head-copy {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .panel-kicker {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .panel-title {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .panel-note {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .panel-note::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 0 5px rgba(5, 150, 105, 0.10);
    }
    .clients {
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 76vh;
      overflow: auto;
    }
    .clients::-webkit-scrollbar {
      width: 10px;
    }
    .clients::-webkit-scrollbar-thumb {
      background: rgba(148, 163, 184, 0.26);
      border-radius: 999px;
    }
    .client-card {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 249, 253, 0.88));
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    .client-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
      border-color: rgba(37, 99, 235, 0.24);
    }
    .client-card.active {
      border-color: rgba(37, 99, 235, 0.46);
      box-shadow:
        inset 0 0 0 1px rgba(37, 99, 235, 0.16),
        0 18px 34px rgba(37, 99, 235, 0.12);
      background: linear-gradient(180deg, rgba(255, 255, 255, 1), rgba(239, 246, 255, 0.96));
    }
    .client-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .client-name {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.45;
      word-break: break-all;
    }
    .client-badges {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 6px;
    }
    .mini-badge,
    .status-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      min-height: 32px;
      padding: 7px 12px;
      border-radius: 999px;
      border: 1px solid transparent;
      font-size: 12px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
    }
    .mini-badge::before,
    .status-pill::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
      opacity: 0.85;
    }
    .tone-neutral {
      color: #475569;
      background: rgba(148, 163, 184, 0.10);
      border-color: rgba(148, 163, 184, 0.18);
    }
    .tone-good {
      color: var(--success);
      background: var(--success-soft);
      border-color: rgba(5, 150, 105, 0.18);
    }
    .tone-warn {
      color: var(--warn);
      background: var(--warn-soft);
      border-color: rgba(217, 119, 6, 0.18);
    }
    .tone-danger {
      color: var(--danger);
      background: var(--danger-soft);
      border-color: rgba(220, 38, 38, 0.18);
    }
    .tone-accent {
      color: var(--accent);
      background: var(--accent-soft);
      border-color: rgba(37, 99, 235, 0.18);
    }
    .client-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .client-stat {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(248, 250, 252, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.12);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .client-stat strong {
      display: block;
      color: var(--ink);
      font-size: 15px;
      margin-top: 2px;
    }
    .client-transcript,
    .client-categories {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.6;
    }
    .client-transcript {
      min-height: 38px;
      margin-bottom: 8px;
    }
    .client-empty {
      padding: 18px;
      border-radius: 16px;
      border: 1px dashed rgba(148, 163, 184, 0.28);
      color: var(--muted);
      text-align: center;
      font-size: 14px;
      background: rgba(255, 255, 255, 0.56);
    }
    .viewer {
      padding: 20px 22px 22px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .metric-card {
      min-height: 98px;
      padding: 14px 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.84));
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
      display: flex;
      flex-direction: column;
      gap: 8px;
      justify-content: space-between;
    }
    .metric-label {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .metric-value {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.3;
      color: var(--ink);
    }
    .metric-value.metric-compact {
      font-size: 17px;
      font-weight: 700;
    }
    .metric-value.status-pill {
      width: fit-content;
      max-width: 100%;
      font-size: 13px;
      font-weight: 700;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .detail-card {
      min-height: 106px;
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.14);
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    }
    .detail-label {
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .detail-value {
      color: var(--ink);
      font-size: 15px;
      font-weight: 700;
      line-height: 1.65;
      word-break: keep-all;
    }
    .screen-shell {
      padding: 16px;
      border-radius: 24px;
      background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 28%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(243, 247, 252, 0.82));
      border: 1px solid rgba(148, 163, 184, 0.16);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6), var(--shadow-sm);
    }
    .screen-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .screen-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .screen-subtitle {
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    .screen {
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 20px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.96));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), var(--shadow-md);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .screen::before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(transparent, rgba(15, 23, 42, 0.08)),
        radial-gradient(circle at top left, rgba(96, 165, 250, 0.14), transparent 24%);
      pointer-events: none;
    }
    .screen img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #111827;
      position: relative;
      z-index: 1;
    }
    .screen-empty {
      position: relative;
      z-index: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      text-align: center;
      padding: 24px;
      color: rgba(226, 232, 240, 0.90);
    }
    .screen-empty-icon {
      width: 72px;
      height: 72px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.12);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 28px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .screen-empty-title {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .screen-empty-copy {
      font-size: 14px;
      line-height: 1.7;
      color: rgba(226, 232, 240, 0.72);
      max-width: 420px;
    }
    @media (max-width: 1280px) {
      .metric-grid,
      .detail-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 980px) {
      .dashboard-grid {
        grid-template-columns: 1fr;
      }
      .system-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 720px) {
      .wrap {
        padding: 18px;
      }
      .hero {
        flex-direction: column;
        align-items: stretch;
      }
      .system-grid,
      .metric-grid,
      .detail-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <div class="hero-copy">
        <span class="eyebrow">AI Monitoring System</span>
        <h1>detectWarning 실시간 모니터링 대시보드</h1>
        <p>팀원 노트북에서 들어오는 영상과 음성을 데스크탑 서버가 분석하고, 현재 상태를 한눈에 볼 수 있도록 정리한 실시간 대시보드입니다.</p>
      </div>
      <div class="hero-summary">
        <div class="hero-summary-label">Presentation Ready</div>
        <div class="hero-summary-value">실시간 분석 화면, 위험도, 시스템 상태를 한 화면에서 확인</div>
      </div>
    </header>

    <section class="system-grid">
      <article class="system-card">
        <div class="system-title">Desktop</div>
        <div class="system-value" id="desktopHost">-</div>
      </article>
      <article class="system-card">
        <div class="system-title">GPU</div>
        <div class="system-value" id="desktopGpu">-</div>
      </article>
      <article class="system-card">
        <div class="system-title">CPU / RAM</div>
        <div class="system-value" id="desktopCpuRam">-</div>
      </article>
      <article class="system-card">
        <div class="system-title">Runtime</div>
        <div class="system-value" id="desktopRuntime">-</div>
      </article>
    </section>

    <section class="dashboard-grid">
      <aside class="panel">
        <div class="panel-head">
          <div class="panel-head-copy">
            <div class="panel-kicker">Clients</div>
            <h2 class="panel-title">연결된 클라이언트</h2>
          </div>
          <div class="panel-note">자동 갱신</div>
        </div>
        <div id="clients" class="clients"></div>
      </aside>

      <main class="panel">
        <div class="panel-head">
          <div class="panel-head-copy">
            <div class="panel-kicker">Live View</div>
            <h2 class="panel-title">실시간 분석 화면</h2>
          </div>
          <div class="panel-note">AI 분석 중</div>
        </div>
        <div class="viewer">
          <section class="metric-grid">
            <article class="metric-card">
              <div class="metric-label">클라이언트</div>
              <div class="metric-value metric-compact" id="clientName">-</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">사람 수</div>
              <div class="metric-value" id="peopleCount">0</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">얼굴 수</div>
              <div class="metric-value" id="faceCount">0</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">지연 시간</div>
              <div class="metric-value" id="latency">0ms</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">최근 수신</div>
              <div class="metric-value metric-compact" id="lastSeen">-</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">음성 인식 상태</div>
              <div class="metric-value status-pill tone-neutral" id="speechStatus">대기</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">오디오 레벨</div>
              <div class="metric-value metric-compact" id="audioLevel">0.000</div>
            </article>
            <article class="metric-card">
              <div class="metric-label">위험도</div>
              <div class="metric-value status-pill tone-neutral" id="riskLevel">0/100 | 낮음</div>
            </article>
          </section>

          <section class="detail-grid">
            <article class="detail-card">
              <div class="detail-label">인식 내용</div>
              <div class="detail-value" id="transcript">-</div>
            </article>
            <article class="detail-card">
              <div class="detail-label">위험 카테고리</div>
              <div class="detail-value" id="riskCategories">없음</div>
            </article>
            <article class="detail-card">
              <div class="detail-label">위험 신호</div>
              <div class="detail-value" id="riskReasons">없음</div>
            </article>
          </section>

          <section class="screen-shell">
            <div class="screen-head">
              <div class="screen-title">실시간 분석 프레임</div>
              <div class="screen-subtitle">Pose / Face / Risk Overlay</div>
            </div>
            <div class="screen" id="screen">
              <div class="screen-empty">
                <div class="screen-empty-icon">AI</div>
                <div class="screen-empty-title">분석 화면을 준비하는 중입니다</div>
                <div class="screen-empty-copy">클라이언트가 연결되면 여기에 실시간 영상과 분석 오버레이가 표시됩니다.</div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </section>
  </div>
  <script>
    let selectedClientId = null;

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function speechTone(status) {
      if (status === 'recognized') return 'tone-good';
      if (status === 'processing') return 'tone-warn';
      if (status === 'error') return 'tone-danger';
      return 'tone-neutral';
    }

    function riskTone(level) {
      if (level === 'HIGH') return 'tone-danger';
      if (level === 'MEDIUM') return 'tone-warn';
      if (level === 'ELEVATED') return 'tone-accent';
      return 'tone-neutral';
    }

    function screenPlaceholder(title, copy) {
      return `
        <div class="screen-empty">
          <div class="screen-empty-icon">AI</div>
          <div class="screen-empty-title">${escapeHtml(title)}</div>
          <div class="screen-empty-copy">${escapeHtml(copy)}</div>
        </div>
      `;
    }

    async function refreshSystem() {
      if (document.hidden) {
        return;
      }
      const response = await fetch('/api/system');
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      document.getElementById('desktopHost').textContent = `${data.hostname}\n${data.platform}`;
      const gpuLine = data.gpu_status === 'ok'
        ? `${data.gpu_name}\nGPU ${data.gpu_utilization_percent ?? 0}% | VRAM ${Math.round(data.gpu_memory_used_mb ?? 0)}/${Math.round(data.gpu_memory_total_mb ?? 0)} MB\n온도 ${data.gpu_temperature_c ?? '-'}C | 전력 ${data.gpu_power_watts ?? '-'}W`
        : `GPU 정보 없음\n${data.gpu_status}`;
      document.getElementById('desktopGpu').textContent = gpuLine;
      const cpuRamLine =
        `CPU ${data.cpu_percent ?? 0}%\nRAM ${data.memory_percent ?? 0}% (${data.memory_used_gb ?? 0}/${data.memory_total_gb ?? 0} GB)`;
      document.getElementById('desktopCpuRam').textContent = cpuRamLine;
      const runtimeLine =
        `오디오 큐 ${data.audio_queue_size}\nYOLO ${data.yolo_device}\nSTT ${data.stt_device} / ${data.stt_compute_type}\nbeam ${data.stt_beam_size} | best_of ${data.stt_best_of}`;
      document.getElementById('desktopRuntime').textContent = runtimeLine;
    }

    async function refreshClients() {
      if (document.hidden) {
        return;
      }
      const response = await fetch('/api/clients');
      const clients = await response.json();
      const container = document.getElementById('clients');
      container.innerHTML = '';

      if (!clients.length) {
        container.innerHTML = '<div class="client-empty">아직 연결된 클라이언트가 없습니다.</div>';
        document.getElementById('screen').innerHTML = screenPlaceholder('클라이언트를 기다리는 중입니다', '맥북 업로더를 실행하면 이곳에 실시간 분석 화면이 표시됩니다.');
        updateMeta(null);
        return;
      }

      if (!selectedClientId || !clients.some(client => client.client_id === selectedClientId)) {
        selectedClientId = clients[0].client_id;
      }

      for (const client of clients) {
        const item = document.createElement('div');
        item.className = 'client-card' + (client.client_id === selectedClientId ? ' active' : '');
        item.innerHTML = `
          <div class="client-top">
            <div class="client-name">${escapeHtml(client.client_id)}</div>
            <div class="client-badges">
              <span class="mini-badge ${speechTone(client.speech_status)}">${escapeHtml(client.speech_status)}</span>
              <span class="mini-badge ${riskTone(client.risk_level)}">${escapeHtml(client.risk_level_label)}</span>
            </div>
          </div>
          <div class="client-stats">
            <div class="client-stat">사람<strong>${client.people_count}</strong></div>
            <div class="client-stat">얼굴<strong>${client.face_count}</strong></div>
            <div class="client-stat">최근 수신<strong>${client.last_seen_seconds}초 전</strong></div>
            <div class="client-stat">지연<strong>${client.latency_ms.toFixed(1)}ms</strong></div>
          </div>
          <div class="client-transcript">${escapeHtml(client.transcript ? client.transcript : '최근 인식된 음성이 없습니다.')}</div>
          <div class="client-categories">카테고리 ${escapeHtml(client.risk_categories && client.risk_categories.length ? client.risk_categories.join(', ') : '없음')}</div>
        `;
        item.onclick = () => {
          selectedClientId = client.client_id;
          refreshClients();
          refreshSelectedFrame();
        };
        container.appendChild(item);
      }

      refreshSelectedFrame();
    }

    async function refreshSelectedFrame() {
      if (document.hidden) {
        return;
      }
      if (!selectedClientId) {
        return;
      }
      const response = await fetch(`/api/client/${encodeURIComponent(selectedClientId)}`);
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      updateMeta(data);
      const screen = document.getElementById('screen');
      if (!data.frame_data_url) {
        screen.innerHTML = screenPlaceholder('프레임을 기다리는 중입니다', '선택한 클라이언트에서 아직 수신된 프레임이 없습니다.');
        return;
      }
      screen.innerHTML = '';
      const image = document.createElement('img');
      image.alt = '분석 화면';
      image.src = data.frame_data_url;
      screen.appendChild(image);
    }

    function updateMeta(data) {
      document.getElementById('clientName').textContent = data ? data.client_id : '-';
      document.getElementById('peopleCount').textContent = `${data ? data.people_count : 0}`;
      document.getElementById('faceCount').textContent = `${data ? data.face_count : 0}`;
      document.getElementById('latency').textContent = `${data ? data.latency_ms.toFixed(1) : 0}ms`;
      document.getElementById('lastSeen').textContent = data ? `${data.last_seen_seconds.toFixed(1)}초 전` : '-';

      const speechStatus = document.getElementById('speechStatus');
      speechStatus.textContent = data ? data.speech_status_label : '대기';
      speechStatus.className = `metric-value status-pill ${speechTone(data ? data.speech_status : 'idle')}`;

      document.getElementById('audioLevel').textContent = data ? data.audio_level.toFixed(3) : '0.000';

      const riskLevel = document.getElementById('riskLevel');
      riskLevel.textContent = `${data ? data.risk_score : 0}/100 | ${data ? data.risk_level_label : '낮음'}`;
      riskLevel.className = `metric-value status-pill ${riskTone(data ? data.risk_level : 'LOW')}`;

      document.getElementById('transcript').textContent = data && data.transcript ? data.transcript : '-';
      document.getElementById('riskCategories').textContent = data && data.risk_categories && data.risk_categories.length ? data.risk_categories.join(', ') : '없음';
      document.getElementById('riskReasons').textContent = data && data.risk_reasons && data.risk_reasons.length ? data.risk_reasons.join(', ') : '없음';
    }

    const params = new URLSearchParams(window.location.search);
    const requestedClientId = params.get('client_id');
    if (requestedClientId) {
      selectedClientId = requestedClientId;
    }

    refreshSystem();
    refreshClients();
    setInterval(refreshSystem, 1000);
    setInterval(refreshClients, 1000);
    setInterval(refreshSelectedFrame, 350);
  </script>
</body>
</html>"""

    @app.get("/health")
    def health() -> dict:
        snapshot = system_monitor.get_snapshot()
        snapshot.update({"status": "ok", "sessions": len(sessions)})
        return snapshot

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard()

    @app.get("/api/clients")
    def api_clients() -> list[dict]:
        return list_client_summaries()

    @app.get("/api/system")
    def api_system() -> dict:
        snapshot = system_monitor.get_snapshot()
        snapshot["sessions"] = len(sessions)
        return snapshot

    @app.get("/api/client/{client_id}")
    def api_client(client_id: str) -> dict:
        with session_lock:
            session = sessions.get(client_id)
            if session is None:
                raise HTTPException(status_code=404, detail="클라이언트를 찾을 수 없습니다.")
            frame_data_url = None
            if session.latest_frame_jpeg:
                encoded = base64.b64encode(session.latest_frame_jpeg).decode("ascii")
                frame_data_url = f"data:image/jpeg;base64,{encoded}"
            meta = session.latest_meta or {}
            return {
                "client_id": client_id,
                "people_count": int(meta.get("people_count", 0)),
                "face_count": int(meta.get("face_count", 0)),
                "latency_ms": float(meta.get("latency_ms", 0.0)),
                "speech_status": str(meta.get("speech_status", "idle")),
                "speech_status_label": localize_speech_status(str(meta.get("speech_status", "idle"))),
                "transcript": str(meta.get("transcript", "")),
                "audio_level": float(meta.get("audio_level", 0.0)),
                "risk_score": int(meta.get("risk_score", 0)),
                "risk_level": str(meta.get("risk_level", "LOW")),
                "risk_level_label": localize_risk_level(str(meta.get("risk_level", "LOW"))),
                "risk_categories": list(meta.get("risk_categories", [])),
                "risk_reasons": list(meta.get("risk_reasons", [])),
                "last_seen_seconds": monotonic() - session.last_seen,
                "frame_data_url": frame_data_url,
            }

    @app.post("/analyze/audio")
    def analyze_audio(
        audio_bytes: bytes = Body(..., media_type="audio/wav"),
        client_id: str = Query(..., min_length=3, description="팀원별 추적 상태 식별자"),
    ) -> dict:
        started_at = perf_counter()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="빈 오디오 요청입니다.")

        session = get_session(client_id)
        latency_ms = (perf_counter() - started_at) * 1000.0
        if session.latest_meta is None:
            session.latest_meta = {}
        session.latest_meta.update({"speech_status": "processing"})
        try:
            audio_job_queue.put_nowait(AudioJob(client_id=client_id, wav_bytes=audio_bytes))
            queued = True
        except queue.Full:
            queued = False
            session.latest_meta.update(
                {
                    "speech_status": "error",
                    "speech_error": "오디오 처리 대기열이 가득 찼습니다.",
                }
            )
        return {
            "client_id": client_id,
            "speech_status": "queued" if queued else "error",
            "transcript": "",
            "audio_level": 0.0,
            "latency_ms": round(latency_ms, 1),
            "queued": queued,
        }

    @app.post("/analyze/frame")
    def analyze_frame(
        image_bytes: bytes = Body(..., media_type="image/jpeg"),
        client_id: str = Query(..., min_length=3, description="팀원별 추적 상태 식별자"),
    ) -> dict:
        started_at = perf_counter()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="빈 이미지 요청입니다.")

        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="JPEG 이미지를 디코딩하지 못했습니다.")

        session = get_session(client_id)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        people = person_detector.detect(frame)
        tracked_people = session.tracker.update(people)
        faces = face_detector.detect(frame)
        evaluated_people = session.person_filter.evaluate(
            tracked_people=tracked_people,
            faces=faces,
            gray_frame=gray_frame,
            frame_shape=frame.shape,
        )
        confirmed_people = [
            person
            for person in evaluated_people
            if person.get("person_state") in {"full_body_person", "upper_body_person"}
        ]
        latency_ms = (perf_counter() - started_at) * 1000.0
        annotated = frame.copy()
        session.person_filter.draw_debug_overlay(annotated, evaluated_people, draw_pose_overlay)
        draw_server_overlay(annotated, client_id, faces, latency_ms, len(confirmed_people))
        success, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            session.latest_frame_jpeg = encoded.tobytes()
        session.latest_people = confirmed_people
        if session.latest_meta is None:
            session.latest_meta = {}
        session.latest_meta.update(
            {
                "people_count": len(confirmed_people),
                "uncertain_count": sum(
                    1 for person in evaluated_people if person.get("person_state") == "uncertain"
                ),
                "candidate_count": len(evaluated_people),
                "face_count": len(faces),
                "latency_ms": round(latency_ms, 1),
            }
        )
        if args.show_windows:
            cv2.imshow(f"detectWarning server - {client_id}", annotated)
            cv2.waitKey(1)
        return {
            "client_id": client_id,
            "tracked_people": evaluated_people,
            "faces": [list(map(int, face)) for face in faces],
            "latency_ms": round(latency_ms, 1),
        }

    def audio_worker() -> None:
        while True:
            job = audio_job_queue.get()
            session = get_session(job.client_id)
            if session.latest_meta is None:
                session.latest_meta = {}
            try:
                transcript, audio_level = speech_recognizer.transcribe_wav_bytes(job.wav_bytes)
                speech_status = "recognized" if transcript else "listening"
                tracked_people = session.latest_people or []
                face_count = int((session.latest_meta or {}).get("face_count", 0))
                risk = session.risk_analyzer.update(
                    ServerSpeechResult(
                        status=speech_status,
                        transcript=transcript,
                        audio_level=audio_level,
                    ),
                    tracked_people=tracked_people,
                    face_count=face_count,
                )
                session.latest_meta.update(
                    {
                        "speech_status": speech_status,
                        "transcript": transcript,
                        "audio_level": round(audio_level, 4),
                        "risk_score": risk.score,
                        "risk_level": risk.level,
                        "risk_categories": list(risk.categories),
                        "risk_reasons": list(risk.reasons),
                        "risk_context_flags": list(risk.context_flags),
                    }
                )
                session.latest_meta.pop("speech_error", None)
            except Exception as exc:
                session.latest_meta.update(
                    {
                        "speech_status": "error",
                        "speech_error": str(exc),
                    }
                )
            finally:
                audio_job_queue.task_done()

    threading.Thread(target=audio_worker, name="audio-worker", daemon=True).start()

    return app


def localize_speech_status(status: str) -> str:
    labels = {
        "idle": "대기",
        "listening": "듣는 중",
        "recognized": "인식됨",
        "error": "오류",
    }
    return labels.get(status, status)


def localize_risk_level(level: str) -> str:
    labels = {
        "LOW": "낮음",
        "ELEVATED": "주의",
        "MEDIUM": "경계",
        "HIGH": "위험",
    }
    return labels.get(level, level)


def main() -> None:
    args = parse_args()
    print(f"[detectWarning] YOLO device: {args.yolo_device}")
    print(f"[detectWarning] Whisper device: {args.stt_device}")
    print(f"[detectWarning] Whisper compute type: {args.stt_compute_type}")
    print(f"[detectWarning] Whisper beam/best_of: {args.stt_beam_size}/{args.stt_best_of}")
    try:
        app = create_app(args)
    except Exception as exc:
        raise RuntimeError(
            "서버 시작 중 추론 장치 초기화에 실패했습니다.\n"
            f"{exc}\n\n"
            "해결 방법:\n"
            "1. Windows 데스크탑에서 CUDA 지원 PyTorch를 설치합니다.\n"
            "2. 또는 임시로 --yolo-device cpu --stt-device cpu 로 실행합니다."
        ) from exc
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

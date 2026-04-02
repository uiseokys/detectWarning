from __future__ import annotations

import argparse
import base64
import io
import queue
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
from risk_analyzer import RiskAnalyzer
from tracker import PersonTracker


@dataclass
class ClientSession:
    tracker: PersonTracker
    risk_analyzer: RiskAnalyzer
    last_seen: float
    latest_frame_jpeg: bytes | None = None
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


class ServerSpeechRecognizer:
    def __init__(self, model_size: str, compute_type: str, language: str, device: str) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper를 불러오지 못했습니다. `pip install -r requirements.txt`를 확인해 주세요."
            ) from exc

        self.language = normalize_language(language)
        self.device = device
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
                beam_size=1,
                best_of=1,
                no_speech_threshold=0.6,
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


def draw_server_overlay(frame, client_id: str, tracked_people, faces, latency_ms: float) -> None:
    for person in tracked_people:
        x, y, w, h = person["bbox"]
        person_id = person["id"]
        draw_pose_overlay(frame, person.get("keypoints", []))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 180, 99), 1)
        cv2.putText(
            frame,
            f"Person {person_id}",
            (x, max(y - 10, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 180, 99),
            2,
        )

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
        f"client: {client_id}",
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
    cv2.putText(
        frame,
        f"people: {len(tracked_people)} | faces: {len(faces)} | latency: {latency_ms:.1f}ms",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


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
        "--stt-language",
        default="ko-KR",
        help="서버 STT 언어 코드. 예: ko-KR",
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
    )
    sessions: dict[str, ClientSession] = {}
    session_lock = threading.Lock()
    audio_job_queue: queue.Queue[AudioJob] = queue.Queue(maxsize=32)

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
      --bg: #f2efe6;
      --panel: #fffaf2;
      --ink: #18222f;
      --accent: #cf5f39;
      --line: #d6c8b7;
      --muted: #6d737b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Pretendard", "Apple SD Gothic Neo", sans-serif;
      background: radial-gradient(circle at top, #fff7e8 0%, var(--bg) 60%, #e9e4d8 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin-bottom: 20px;
    }
    .hero h1 {
      margin: 0;
      font-size: 34px;
      line-height: 1.05;
    }
    .hero p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
    }
    .grid {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 20px;
    }
    .panel {
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 20px 60px rgba(100, 78, 44, 0.08);
      overflow: hidden;
    }
    .panel-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }
    .clients {
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 72vh;
      overflow: auto;
    }
    .client {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #fffdf8;
      cursor: pointer;
    }
    .client.active {
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
    }
    .client-title {
      font-weight: 700;
      margin-bottom: 8px;
    }
    .client-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .viewer {
      padding: 18px;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 14px;
    }
    .screen {
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, #ddd4c8, #f7f2eb);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .screen img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #1f2328;
    }
    .placeholder {
      color: var(--muted);
      font-size: 16px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>detectWarning<br/>원격 분석 대시보드</h1>
        <p>팀원 노트북 카메라 영상을 데스크탑에서 분석하고 브라우저로 확인합니다.</p>
      </div>
    </div>
    <div class="grid">
      <section class="panel">
        <div class="panel-head">연결된 클라이언트</div>
        <div id="clients" class="clients"></div>
      </section>
      <section class="panel">
        <div class="panel-head">실시간 분석 화면</div>
        <div class="viewer">
          <div class="stats">
            <div class="stat" id="clientName">클라이언트: -</div>
            <div class="stat" id="peopleCount">사람: 0</div>
            <div class="stat" id="faceCount">얼굴: 0</div>
            <div class="stat" id="latency">지연: 0ms</div>
            <div class="stat" id="lastSeen">최근 수신: -</div>
            <div class="stat" id="speechStatus">음성 인식: 대기</div>
            <div class="stat" id="audioLevel">오디오 레벨: 0.000</div>
          <div class="stat" id="riskLevel">위험도: 0/100 | 낮음</div>
          </div>
          <div class="stat" id="transcript" style="display:block; border-radius:16px; margin-bottom:16px;">인식 내용: -</div>
          <div class="stat" id="riskCategories" style="display:block; border-radius:16px; margin-bottom:16px;">위험 카테고리: 없음</div>
          <div class="stat" id="riskReasons" style="display:block; border-radius:16px; margin-bottom:16px;">위험 신호: 없음</div>
          <div class="screen" id="screen">
            <div class="placeholder">클라이언트를 선택하면 분석 화면이 표시됩니다.</div>
          </div>
        </div>
      </section>
    </div>
  </div>
  <script>
    let selectedClientId = null;

    async function refreshClients() {
      const response = await fetch('/api/clients');
      const clients = await response.json();
      const container = document.getElementById('clients');
      container.innerHTML = '';

      if (!clients.length) {
        container.innerHTML = '<div class="client-meta">아직 연결된 클라이언트가 없습니다.</div>';
        document.getElementById('screen').innerHTML = '<div class="placeholder">맥북 업로더를 실행하면 화면이 나타납니다.</div>';
        updateMeta(null);
        return;
      }

      if (!selectedClientId || !clients.some(client => client.client_id === selectedClientId)) {
        selectedClientId = clients[0].client_id;
      }

      for (const client of clients) {
        const item = document.createElement('div');
        item.className = 'client' + (client.client_id === selectedClientId ? ' active' : '');
        item.innerHTML = `
          <div class="client-title">${client.client_id}</div>
          <div class="client-meta">
            사람 ${client.people_count}명 | 얼굴 ${client.face_count}개<br/>
            최근 수신 ${client.last_seen_seconds}초 전 | 지연 ${client.latency_ms.toFixed(1)}ms<br/>
            STT ${client.speech_status} ${client.transcript ? '| ' + client.transcript : ''}<br/>
            위험도 ${client.risk_score}/100 | ${client.risk_level_label}<br/>
            카테고리 ${client.risk_categories && client.risk_categories.length ? client.risk_categories.join(', ') : '-'}
          </div>
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
        screen.innerHTML = '<div class="placeholder">아직 수신된 프레임이 없습니다.</div>';
        return;
      }
      screen.innerHTML = `<img alt="분석 화면" src="${data.frame_data_url}" />`;
    }

    function updateMeta(data) {
      document.getElementById('clientName').textContent = `클라이언트: ${data ? data.client_id : '-'}`;
      document.getElementById('peopleCount').textContent = `사람: ${data ? data.people_count : 0}`;
      document.getElementById('faceCount').textContent = `얼굴: ${data ? data.face_count : 0}`;
      document.getElementById('latency').textContent = `지연: ${data ? data.latency_ms.toFixed(1) : 0}ms`;
      document.getElementById('lastSeen').textContent = `최근 수신: ${data ? data.last_seen_seconds.toFixed(1) : '-'}초 전`;
      document.getElementById('speechStatus').textContent = `음성 인식: ${data ? data.speech_status_label : '대기'}`;
      document.getElementById('audioLevel').textContent = `오디오 레벨: ${data ? data.audio_level.toFixed(3) : '0.000'}`;
      document.getElementById('riskLevel').textContent = `위험도: ${data ? data.risk_score : 0}/100 | ${data ? data.risk_level_label : '낮음'}`;
      document.getElementById('transcript').textContent = `인식 내용: ${data && data.transcript ? data.transcript : '-'}`;
      document.getElementById('riskCategories').textContent = `위험 카테고리: ${data && data.risk_categories && data.risk_categories.length ? data.risk_categories.join(', ') : '없음'}`;
      document.getElementById('riskReasons').textContent = `위험 신호: ${data && data.risk_reasons && data.risk_reasons.length ? data.risk_reasons.join(', ') : '없음'}`;
    }

    const params = new URLSearchParams(window.location.search);
    const requestedClientId = params.get('client_id');
    if (requestedClientId) {
      selectedClientId = requestedClientId;
    }

    refreshClients();
    setInterval(refreshClients, 1000);
    setInterval(refreshSelectedFrame, 350);
  </script>
</body>
</html>"""

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "sessions": len(sessions),
            "audio_queue_size": audio_job_queue.qsize(),
            "yolo_device": args.yolo_device,
            "stt_device": args.stt_device,
            "stt_compute_type": args.stt_compute_type,
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard()

    @app.get("/api/clients")
    def api_clients() -> list[dict]:
        return list_client_summaries()

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
        people = person_detector.detect(frame)
        tracked_people = session.tracker.update(people)
        faces = face_detector.detect(frame)
        latency_ms = (perf_counter() - started_at) * 1000.0
        annotated = frame.copy()
        draw_server_overlay(annotated, client_id, tracked_people, faces, latency_ms)
        success, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            session.latest_frame_jpeg = encoded.tobytes()
        if session.latest_meta is None:
            session.latest_meta = {}
        session.latest_meta.update(
            {
                "people_count": len(tracked_people),
                "face_count": len(faces),
                "latency_ms": round(latency_ms, 1),
            }
        )
        if args.show_windows:
            cv2.imshow(f"detectWarning server - {client_id}", annotated)
            cv2.waitKey(1)
        return {
            "client_id": client_id,
            "tracked_people": tracked_people,
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
                risk = session.risk_analyzer.update(
                    ServerSpeechResult(
                        status=speech_status,
                        transcript=transcript,
                        audio_level=audio_level,
                    ),
                    tracked_people=[],
                    face_count=0,
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
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import base64
import threading
from dataclasses import dataclass
from time import monotonic, perf_counter

import cv2
import numpy as np
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Query, Request

from detector import FaceDetector, PersonDetector
from tracker import PersonTracker


@dataclass
class ClientSession:
    tracker: PersonTracker
    last_seen: float
    latest_frame_jpeg: bytes | None = None
    latest_meta: dict | None = None


def draw_server_overlay(frame, client_id: str, tracked_people, faces, latency_ms: float) -> None:
    for person in tracked_people:
        x, y, w, h = person["bbox"]
        person_id = person["id"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 180, 99), 2)
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
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="detectWarning Inference Server")
    person_detector = PersonDetector(
        score_threshold=args.person_score_threshold,
        resize_width=args.person_imgsz,
    )
    face_detector = FaceDetector()
    sessions: dict[str, ClientSession] = {}
    session_lock = threading.Lock()

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
                session = ClientSession(tracker=PersonTracker(), last_seen=now)
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
          </div>
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
            최근 수신 ${client.last_seen_seconds}초 전 | 지연 ${client.latency_ms.toFixed(1)}ms
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
    }

    refreshClients();
    setInterval(refreshClients, 1000);
    setInterval(refreshSelectedFrame, 350);
  </script>
</body>
</html>"""

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "sessions": len(sessions)}

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return render_dashboard()

    @app.get("/api/clients")
    async def api_clients() -> list[dict]:
        return list_client_summaries()

    @app.get("/api/client/{client_id}")
    async def api_client(client_id: str) -> dict:
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
                "last_seen_seconds": monotonic() - session.last_seen,
                "frame_data_url": frame_data_url,
            }

    @app.post("/analyze/frame")
    async def analyze_frame(
        request: Request,
        client_id: str = Query(..., min_length=3, description="팀원별 추적 상태 식별자"),
    ) -> dict:
        started_at = perf_counter()
        image_bytes = await request.body()
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
        session.latest_meta = {
            "people_count": len(tracked_people),
            "face_count": len(faces),
            "latency_ms": round(latency_ms, 1),
        }
        if args.show_windows:
            cv2.imshow(f"detectWarning server - {client_id}", annotated)
            cv2.waitKey(1)
        return {
            "client_id": client_id,
            "tracked_people": tracked_people,
            "faces": [list(map(int, face)) for face in faces],
            "latency_ms": round(latency_ms, 1),
        }

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

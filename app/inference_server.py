from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from time import monotonic, perf_counter

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request

from detector import FaceDetector, PersonDetector
from tracker import PersonTracker


@dataclass
class ClientSession:
    tracker: PersonTracker
    last_seen: float


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

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "sessions": len(sessions)}

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
        if args.show_windows:
            annotated = frame.copy()
            draw_server_overlay(annotated, client_id, tracked_people, faces, latency_ms)
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

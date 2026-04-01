from __future__ import annotations

import argparse
import socket
import uuid
from time import perf_counter, sleep

import cv2
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="노트북 카메라 프레임을 원격 추론 서버로 전송합니다.")
    parser.add_argument("--source", default="0", help="웹캠 인덱스 또는 영상 파일 경로")
    parser.add_argument("--server-url", required=True, help="원격 추론 서버 주소. 예: http://100.x.x.x:8000")
    parser.add_argument("--client-id", default="", help="클라이언트 식별자. 비우면 자동 생성")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="전송용 JPEG 품질")
    parser.add_argument("--max-fps", type=float, default=6.0, help="최대 전송 FPS")
    parser.add_argument("--show-local-preview", action="store_true", help="노트북에서도 카메라 미리보기를 표시")
    parser.add_argument("--timeout-seconds", type=float, default=3.0, help="서버 요청 제한 시간")
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        index = int(source)
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(source)


def build_client_id(client_id: str) -> str:
    if client_id.strip():
        return client_id.strip()
    host = socket.gethostname().replace(" ", "-")
    return f"{host}-{uuid.uuid4().hex[:6]}"


def main() -> None:
    args = parse_args()
    capture = open_source(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"입력 소스를 열 수 없습니다: {args.source}")

    client_id = build_client_id(args.client_id)
    interval = 1.0 / max(args.max_fps, 0.1)
    session = requests.Session()
    last_sent_at = 0.0

    print(f"업로더 시작: client_id={client_id}")
    print(f"서버 주소: {args.server_url.rstrip('/')}")
    print("종료하려면 q 키를 누르세요.")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if args.show_local_preview:
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Uploader: {client_id}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("detectWarning uploader", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        now = perf_counter()
        if now - last_sent_at < interval:
            sleep(0.005)
            continue

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(max(min(args.jpeg_quality, 100), 40))],
        )
        if not success:
            continue

        try:
            response = session.post(
                f"{args.server_url.rstrip('/')}/analyze/frame",
                params={"client_id": client_id},
                data=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=args.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            print(
                f"\r전송 성공 | 사람 {len(payload.get('tracked_people', []))} | "
                f"얼굴 {len(payload.get('faces', []))} | "
                f"지연 {payload.get('latency_ms', 0)}ms",
                end="",
                flush=True,
            )
        except Exception as exc:
            print(f"\r전송 실패: {exc}", end="", flush=True)

        last_sent_at = now

    capture.release()
    cv2.destroyAllWindows()
    print("\n업로더를 종료했습니다.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import io
import socket
import threading
import uuid
import wave
from collections import deque
from time import perf_counter, sleep

import cv2
import numpy as np
import requests


def list_input_devices() -> list[tuple[int, str, int, float]]:
    try:
        import sounddevice as sounddevice
    except Exception:
        return []

    devices = []
    for index, device in enumerate(sounddevice.query_devices()):
        max_inputs = int(device.get("max_input_channels", 0))
        if max_inputs <= 0:
            continue
        devices.append(
            (
                index,
                str(device.get("name", "Unknown")),
                max_inputs,
                float(device.get("default_samplerate", 0.0)),
            )
        )
    return devices


def choose_audio_device_interactively() -> int:
    devices = list_input_devices()
    if not devices:
        raise RuntimeError(
            "사용 가능한 입력 오디오 장치를 찾지 못했습니다.\n"
            "마이크 권한과 장치 연결 상태를 확인해 주세요."
        )

    print("마이크를 선택하세요.")
    for index, name, channels, sample_rate in devices:
        print(f"{index}: {name} | 입력채널={channels} | 기본샘플레이트={sample_rate:.0f}")

    while True:
        selected = input("사용할 마이크 장치 번호를 입력하세요: ").strip()
        if not selected:
            print("장치 번호를 입력해 주세요.")
            continue
        if not selected.isdigit():
            print("숫자 장치 번호를 입력해 주세요.")
            continue
        selected_index = int(selected)
        if not any(device_index == selected_index for device_index, *_rest in devices):
            print("목록에 있는 장치 번호를 입력해 주세요.")
            continue
        print(f"선택된 마이크 장치: {selected_index}")
        return selected_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="노트북 카메라 프레임을 원격 추론 서버로 전송합니다.")
    parser.add_argument("--source", default="0", help="웹캠 인덱스 또는 영상 파일 경로")
    parser.add_argument("--server-url", required=True, help="원격 추론 서버 주소. 예: http://100.x.x.x:8000")
    parser.add_argument("--client-id", default="", help="클라이언트 식별자. 비우면 자동 생성")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="전송용 JPEG 품질")
    parser.add_argument("--max-fps", type=float, default=3.0, help="최대 전송 FPS")
    parser.add_argument("--show-local-preview", action="store_true", help="노트북에서도 카메라 미리보기를 표시")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="서버 요청 제한 시간")
    parser.add_argument("--stt", action="store_true", help="맥북 마이크 오디오를 서버로 보내 STT를 함께 수행합니다.")
    parser.add_argument("--stt-device", type=int, default=None, help="입력 오디오 장치 번호")
    parser.add_argument("--select-audio-device", action="store_true", help="실행 전에 마이크 장치를 직접 선택합니다.")
    parser.add_argument("--stt-phrase-seconds", type=float, default=1.8, help="한 번 인식할 오디오 길이")
    parser.add_argument("--stt-silence-seconds", type=float, default=0.35, help="이 시간 이상 조용하면 STT 전송")
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


def build_open_error(source: str) -> str:
    details = [f"입력 소스를 열 수 없습니다: {source}"]
    if source.isdigit():
        details.extend(
            [
                "",
                "macOS에서 이 터미널 또는 앱의 카메라 권한이 막혀 있을 수 있습니다.",
                "시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 현재 앱을 허용해 주세요.",
                "이전에 거부했다면 다음을 실행해 보세요: tccutil reset Camera",
                "그 뒤 터미널을 완전히 종료한 후 다시 실행해 주세요.",
            ]
        )
    return "\n".join(details)


def build_read_error(source: str) -> str:
    details = [f"카메라 첫 프레임을 읽지 못했습니다: {source}"]
    if source.isdigit():
        details.extend(
            [
                "",
                "가능한 원인:",
                "- macOS 카메라 권한이 허용되지 않음",
                "- 다른 앱이 카메라를 이미 사용 중임",
                "- 카메라 초기화가 늦어 첫 프레임을 받지 못함",
                "",
                "확인 방법:",
                "- FaceTime, Zoom, 브라우저 탭 등 카메라를 쓰는 앱 종료",
                "- 시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널 허용",
                "- 필요하면 `python3 app/camera_uploader.py --source 0 --server-url ... --show-local-preview`로 미리보기 확인",
            ]
        )
    return "\n".join(details)


def build_client_id(client_id: str) -> str:
    if client_id.strip():
        return client_id.strip()
    host = socket.gethostname().replace(" ", "-")
    return f"{host}-{uuid.uuid4().hex[:6]}"


class AudioStreamer:
    def __init__(
        self,
        session: requests.Session,
        server_url: str,
        client_id: str,
        timeout_seconds: float,
        input_device: int | None,
        phrase_seconds: float,
        silence_seconds: float,
        sample_rate: int = 16000,
    ) -> None:
        self.session = session
        self.server_url = server_url.rstrip("/")
        self.client_id = client_id
        self.timeout_seconds = timeout_seconds
        self.input_device = input_device
        self.phrase_seconds = phrase_seconds
        self.silence_seconds = silence_seconds
        self.sample_rate = sample_rate
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import sounddevice as sounddevice
        except Exception as exc:
            print(f"\n오디오 업로더를 시작하지 못했습니다: {exc}")
            return

        audio_queue = deque()
        speech_level = 0.008
        block_duration = 0.4
        max_phrase_bytes = int(self.sample_rate * self.phrase_seconds * 2)
        min_phrase_bytes = int(self.sample_rate * 0.55 * 2)
        last_voice_at = None
        speech_started_at = None
        buffer = bytearray()

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                print(f"\n오디오 상태 경고: {status}")
            chunk = bytes(indata)
            level = float(np.abs(np.frombuffer(chunk, dtype=np.int16)).mean() / 32768.0)
            audio_queue.append((chunk, level, perf_counter()))

        try:
            with sounddevice.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * block_duration),
                dtype="int16",
                channels=1,
                device=self.input_device,
                callback=callback,
            ):
                while not self._stop_event.is_set():
                    while audio_queue:
                        chunk, level, ts = audio_queue.popleft()
                        buffer.extend(chunk)
                        if level >= speech_level:
                            if speech_started_at is None:
                                speech_started_at = ts
                            last_voice_at = ts

                    now = perf_counter()
                    reached_max = len(buffer) >= max_phrase_bytes
                    speech_finished = (
                        speech_started_at is not None
                        and last_voice_at is not None
                        and len(buffer) >= min_phrase_bytes
                        and (now - last_voice_at) >= self.silence_seconds
                    )
                    if not reached_max and not speech_finished:
                        if len(buffer) >= max_phrase_bytes and speech_started_at is None:
                            buffer.clear()
                        sleep(0.05)
                        continue

                    wav_bytes = self._to_wav_bytes(bytes(buffer))
                    buffer.clear()
                    speech_started_at = None
                    last_voice_at = None
                    self._send_audio(wav_bytes)
        except Exception as exc:
            print(f"\n오디오 캡처 실패: {exc}")

    def _to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_bytes)
        return output.getvalue()

    def _send_audio(self, wav_bytes: bytes) -> None:
        try:
            response = self.session.post(
                f"{self.server_url}/analyze/audio",
                params={"client_id": self.client_id},
                data=wav_bytes,
                headers={"Content-Type": "audio/wav"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            print(
                "\n오디오 전송 실패: "
                f"{exc} | 서버 STT가 느리거나 현재 요청이 밀린 상태일 수 있습니다."
            )


def main() -> None:
    args = parse_args()
    if args.stt and (args.select_audio_device or args.stt_device is None):
        args.stt_device = choose_audio_device_interactively()

    capture = open_source(args.source)
    if not capture.isOpened():
        raise RuntimeError(build_open_error(args.source))

    client_id = build_client_id(args.client_id)
    interval = 1.0 / max(args.max_fps, 0.1)
    session = requests.Session()
    last_sent_at = 0.0
    audio_streamer = None

    print(f"업로더 시작: client_id={client_id}")
    print(f"서버 주소: {args.server_url.rstrip('/')}")
    print(f"웹 대시보드: {args.server_url.rstrip('/')}/?client_id={client_id}")
    print("종료하려면 q 키를 누르세요.")

    initial_ok = False
    for _ in range(20):
        ok, frame = capture.read()
        if ok and frame is not None:
            initial_ok = True
            break
        sleep(0.1)

    if not initial_ok:
        capture.release()
        cv2.destroyAllWindows()
        raise RuntimeError(build_read_error(args.source))

    if args.stt:
        audio_streamer = AudioStreamer(
            session=session,
            server_url=args.server_url,
            client_id=client_id,
            timeout_seconds=args.timeout_seconds,
            input_device=args.stt_device,
            phrase_seconds=args.stt_phrase_seconds,
            silence_seconds=args.stt_silence_seconds,
        )
        audio_streamer.start()

    while True:
        ok, frame = capture.read()
        if not ok:
            print("\n카메라 프레임을 더 이상 읽지 못했습니다. 업로더를 종료합니다.")
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
            print(
                "\r전송 실패: "
                f"{exc} | 서버 YOLO/STT 처리 시간이 타임아웃보다 길 수 있습니다.",
                end="",
                flush=True,
            )

        last_sent_at = now

    capture.release()
    if audio_streamer is not None:
        audio_streamer.stop()
    cv2.destroyAllWindows()
    print("\n업로더를 종료했습니다.")


if __name__ == "__main__":
    main()

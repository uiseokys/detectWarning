from __future__ import annotations

import argparse
import io
import platform
import queue
import socket
import subprocess
import threading
import uuid
import wave
from collections import deque
from dataclasses import dataclass
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


@dataclass
class OpenAttempt:
    capture: cv2.VideoCapture | None
    frame: object | None
    backend_name: str


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
    parser.add_argument("--source", default="0", help="웹캠 인덱스, 영상 파일 경로, 또는 iphone/continuity")
    parser.add_argument("--server-url", required=True, help="원격 추론 서버 주소. 예: http://100.x.x.x:8000")
    parser.add_argument("--client-id", default="", help="클라이언트 식별자. 비우면 자동 생성")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="전송용 JPEG 품질")
    parser.add_argument("--max-fps", type=float, default=5.0, help="최대 전송 FPS")
    parser.add_argument("--frame-width", type=int, default=960, help="전송 전 프레임 가로 크기. 0이면 원본 유지")
    parser.add_argument("--show-local-preview", action="store_true", help="노트북에서도 카메라 미리보기를 표시")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="서버 요청 제한 시간")
    parser.add_argument("--stt", action="store_true", help="맥북 마이크 오디오를 서버로 보내 STT를 함께 수행합니다.")
    parser.add_argument("--stt-device", type=int, default=None, help="입력 오디오 장치 번호")
    parser.add_argument("--select-audio-device", action="store_true", help="실행 전에 마이크 장치를 직접 선택합니다.")
    parser.add_argument("--stt-phrase-seconds", type=float, default=1.2, help="한 번 인식할 오디오 길이")
    parser.add_argument("--stt-silence-seconds", type=float, default=0.25, help="이 시간 이상 조용하면 STT 전송")
    parser.add_argument("--list-video-devices", action="store_true", help="사용 가능한 카메라 인덱스를 탐색하고 종료합니다.")
    parser.add_argument("--video-device-scan-limit", type=int, default=20, help="카메라 인덱스 탐색 범위. 기본 0~19")
    return parser.parse_args()


def _probe_camera(index: int, backend=None, backend_name: str = "default") -> OpenAttempt:
    if backend is None:
        capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        return OpenAttempt(capture=None, frame=None, backend_name=backend_name)

    frame = None
    for _ in range(25):
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
            break
        sleep(0.08)

    if frame is None:
        capture.release()
        return OpenAttempt(capture=None, frame=None, backend_name=backend_name)
    return OpenAttempt(capture=capture, frame=frame, backend_name=backend_name)


def list_macos_camera_names() -> list[str]:
    if platform.system().lower() != "darwin":
        return []
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=4.0,
            check=False,
        )
    except Exception:
        return []
    names = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line.endswith(":"):
            continue
        name = line[:-1].strip()
        if name and name not in {"Camera", "Cameras"} and name not in names:
            names.append(name)
    return names


def list_video_devices(max_index: int = 20) -> list[tuple[int, str]]:
    found = []
    for index in range(max_index):
        attempts = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            attempts.append((cv2.CAP_AVFOUNDATION, "AVFOUNDATION"))
        attempts.append((None, "DEFAULT"))
        for backend, backend_name in attempts:
            result = _probe_camera(index, backend=backend, backend_name=backend_name)
            if result.capture is not None:
                result.capture.release()
                found.append((index, backend_name))
                break
    return found


def open_iphone_source(max_index: int = 20) -> tuple[cv2.VideoCapture, object | None, str]:
    candidates = list(range(1, max(max_index, 2))) + [0]
    for index in candidates:
        attempts = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            attempts.append((cv2.CAP_AVFOUNDATION, "AVFOUNDATION"))
        attempts.append((None, "DEFAULT"))
        for backend, backend_name in attempts:
            result = _probe_camera(index, backend=backend, backend_name=backend_name)
            if result.capture is not None:
                return result.capture, result.frame, f"{backend_name}:iphone-auto:{index}"
    return cv2.VideoCapture(0), None, "NONE:iphone-auto"


def open_source(source: str, max_index: int = 20) -> tuple[cv2.VideoCapture, object | None, str]:
    normalized_source = source.strip().lower()
    if normalized_source in {"iphone", "ios", "continuity", "continuity-camera", "auto-iphone"}:
        return open_iphone_source(max_index=max_index)
    if source.isdigit():
        index = int(source)
        attempts = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            attempts.append((cv2.CAP_AVFOUNDATION, "AVFOUNDATION"))
        attempts.append((None, "DEFAULT"))

        for backend, backend_name in attempts:
            result = _probe_camera(index, backend=backend, backend_name=backend_name)
            if result.capture is not None:
                return result.capture, result.frame, result.backend_name
        return cv2.VideoCapture(index), None, "NONE"

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return capture, None, "FILE"
    ok, frame = capture.read()
    if ok and frame is not None:
        return capture, frame, "FILE"
    return capture, None, "FILE"


def build_open_error(source: str) -> str:
    details = [f"입력 소스를 열 수 없습니다: {source}"]
    if source.isdigit() or source.strip().lower() in {"iphone", "ios", "continuity", "continuity-camera", "auto-iphone"}:
        details.extend(
            [
                "",
                "macOS에서 이 터미널 또는 앱의 카메라 권한이 막혀 있을 수 있습니다.",
                "iPhone 카메라는 macOS 연속성 카메라로 먼저 잡혀야 합니다.",
                "FaceTime/QuickTime에서 iPhone Camera가 보이는지 먼저 확인해 주세요.",
                "시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 현재 앱을 허용해 주세요.",
                "이전에 거부했다면 다음을 실행해 보세요: tccutil reset Camera",
                "그 뒤 터미널을 완전히 종료한 후 다시 실행해 주세요.",
            ]
        )
    return "\n".join(details)


def build_read_error(source: str) -> str:
    details = [f"카메라 첫 프레임을 읽지 못했습니다: {source}"]
    if source.isdigit() or source.strip().lower() in {"iphone", "ios", "continuity", "continuity-camera", "auto-iphone"}:
        details.extend(
            [
                "",
                "가능한 원인:",
                "- macOS 카메라 권한이 허용되지 않음",
                "- iPhone이 연속성 카메라로 macOS에 노출되지 않음",
                "- 다른 앱이 카메라를 이미 사용 중임",
                "- 카메라 초기화가 늦어 첫 프레임을 받지 못함",
                "",
                "확인 방법:",
                "- FaceTime, Zoom, 브라우저 탭 등 카메라를 쓰는 앱 종료",
                "- FaceTime 또는 QuickTime에서 iPhone Camera가 선택 가능한지 확인",
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


def resize_frame_for_upload(frame, target_width: int):
    if target_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / float(width)
    target_height = max(int(height * scale), 1)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


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
        self._send_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        self._sender_thread = None

    def start(self) -> None:
        self._sender_thread = threading.Thread(target=self._send_worker, daemon=True)
        self._sender_thread.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._send_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import sounddevice as sounddevice
        except Exception as exc:
            print(f"\n오디오 업로더를 시작하지 못했습니다: {exc}")
            return

        audio_queue = deque()
        speech_level = 0.006
        min_peak_level = 0.035
        block_duration = 0.2
        max_phrase_bytes = int(self.sample_rate * self.phrase_seconds * 2)
        min_phrase_bytes = int(self.sample_rate * min(max(self.phrase_seconds * 0.25, 0.45), 0.8) * 2)
        last_voice_at = None
        speech_started_at = None
        buffer = bytearray()
        active_levels = deque(maxlen=10)

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
                            active_levels.append(level)
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

                    if not self._should_send_audio(buffer, active_levels, speech_level, min_peak_level):
                        buffer.clear()
                        active_levels.clear()
                        speech_started_at = None
                        last_voice_at = None
                        continue

                    wav_bytes = self._to_wav_bytes(bytes(buffer))
                    buffer.clear()
                    active_levels.clear()
                    speech_started_at = None
                    last_voice_at = None
                    self._queue_latest_audio(wav_bytes)
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

    def _queue_latest_audio(self, wav_bytes: bytes) -> None:
        while True:
            try:
                self._send_queue.get_nowait()
                self._send_queue.task_done()
            except queue.Empty:
                break
        try:
            self._send_queue.put_nowait(wav_bytes)
        except queue.Full:
            pass

    def _send_worker(self) -> None:
        while not self._stop_event.is_set():
            wav_bytes = self._send_queue.get()
            try:
                if wav_bytes is None:
                    return
                self._send_audio(wav_bytes)
            finally:
                self._send_queue.task_done()

    def _should_send_audio(
        self,
        buffer: bytearray,
        active_levels: deque,
        speech_level: float,
        min_peak_level: float,
    ) -> bool:
        if len(buffer) < int(self.sample_rate * 0.35 * 2):
            return False
        if not active_levels:
            return False
        peak_level = max(active_levels)
        mean_level = sum(active_levels) / max(len(active_levels), 1)
        return peak_level >= min_peak_level or (mean_level >= speech_level * 1.8 and len(active_levels) >= 2)


def main() -> None:
    args = parse_args()
    if args.list_video_devices:
        camera_names = list_macos_camera_names()
        if camera_names:
            print("macOS camera names:")
            for name in camera_names:
                print(f"- {name}")
            print()
        devices = list_video_devices(max_index=args.video_device_scan_limit)
        if not devices:
            print("프레임을 읽을 수 있는 카메라를 찾지 못했습니다.")
        else:
            print("사용 가능한 카메라:")
            for index, backend_name in devices:
                print(f"{index}: {backend_name}")
            print("\niPhone/Continuity Camera가 보이면 `--source iphone` 또는 해당 인덱스를 사용하세요.")
        return

    if args.stt and (args.select_audio_device or args.stt_device is None):
        args.stt_device = choose_audio_device_interactively()

    capture, initial_frame, backend_name = open_source(args.source, max_index=args.video_device_scan_limit)
    if not capture.isOpened():
        raise RuntimeError(build_open_error(args.source))

    client_id = build_client_id(args.client_id)
    interval = 1.0 / max(args.max_fps, 0.1)
    dynamic_interval = interval
    dynamic_frame_width = int(args.frame_width)
    session = requests.Session()
    last_sent_at = 0.0
    audio_streamer = None

    print(f"업로더 시작: client_id={client_id}")
    print(f"서버 주소: {args.server_url.rstrip('/')}")
    print(f"웹 대시보드: {args.server_url.rstrip('/')}/?client_id={client_id}")
    print(f"카메라 백엔드: {backend_name}")
    print("종료하려면 q 키를 누르세요.")

    frame = initial_frame
    if frame is None:
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

    try:
        while True:
            if frame is None:
                ok, frame = capture.read()
            else:
                ok = True
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
            if now - last_sent_at < dynamic_interval:
                sleep(0.005)
                frame = None
                continue

            upload_frame = resize_frame_for_upload(frame, dynamic_frame_width)
            success, encoded = cv2.imencode(
                ".jpg",
                upload_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(max(min(args.jpeg_quality, 100), 40))],
            )
            if not success:
                frame = None
                continue

            try:
                response = session.post(
                    f"{args.server_url.rstrip('/')}/analyze/frame",
                    params={"client_id": client_id, "lite": "1"},
                    data=encoded.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                    timeout=args.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                upload_hint = payload.get("upload_hint") if isinstance(payload, dict) else {}
                if isinstance(upload_hint, dict):
                    hinted_fps = float(upload_hint.get("max_fps") or args.max_fps)
                    hinted_width = int(upload_hint.get("frame_width") or args.frame_width)
                    hinted_fps = max(0.5, min(12.0, hinted_fps))
                    dynamic_interval = 1.0 / max(hinted_fps, 0.1)
                    if args.frame_width > 0:
                        dynamic_frame_width = max(480, min(int(args.frame_width), hinted_width))
                print(
                    f"\rupload ok | people {payload.get('people_count', len(payload.get('tracked_people', [])))} | "
                    f"latency {payload.get('latency_ms', 0)}ms | "
                    f"fps {1.0 / max(dynamic_interval, 1e-6):.1f} | width {dynamic_frame_width}",
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
            frame = None
    finally:
        capture.release()
        if audio_streamer is not None:
            audio_streamer.stop()
        session.close()
        cv2.destroyAllWindows()
        print("\n업로더를 종료했습니다.")


if __name__ == "__main__":
    main()

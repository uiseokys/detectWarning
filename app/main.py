import argparse
import sys
from time import perf_counter
from time import monotonic
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

from audio_detector import SpeechResult, SpeechToTextListener, list_input_devices
from detector import FaceDetector, PersonDetector
from event_logger import WarningEventLogger
from remote_inference import RemoteInferenceClient
from risk_analyzer import RiskAnalyzer
from tracker import PersonTracker

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


FONT_CANDIDATES = [
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]
FONT_CACHE = {}


@dataclass
class ServerStatus:
    mode: str = "로컬 추론"
    latency_ms: float = 0.0
    error: str | None = None


def load_overlay_font(size: int):
    if ImageFont is None:
        return None
    if size in FONT_CACHE:
        return FONT_CACHE[size]

    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                FONT_CACHE[size] = ImageFont.truetype(str(path), size=size)
                return FONT_CACHE[size]
            except Exception:
                continue
    FONT_CACHE[size] = None
    return None


def draw_unicode_text(frame, text: str, position: tuple[int, int], color, size: int):
    font = load_overlay_font(size)
    if font is None or Image is None or ImageDraw is None:
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="웹캠 또는 영상 파일에서 사람을 감지합니다.")
    parser.add_argument(
        "--source",
        default="0",
        help="0 같은 웹캠 인덱스 또는 영상 파일 경로.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.05,
        help="감지 창 스케일 비율.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="각 후보 사각형이 가져야 하는 최소 이웃 수.",
    )
    parser.add_argument(
        "--person-score-threshold",
        type=float,
        default=0.25,
        help="사람 감지 최소 신뢰도. 낮출수록 민감해집니다.",
    )
    parser.add_argument(
        "--person-nms-threshold",
        type=float,
        default=0.45,
        help="겹치는 사람 박스를 얼마나 강하게 합칠지 설정합니다.",
    )
    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=640,
        help="YOLO 사람 감지 입력 크기. 클수록 보통 더 정확합니다.",
    )
    parser.add_argument(
        "--person-detect-interval",
        type=int,
        default=2,
        help="사람 감지를 몇 프레임마다 수행할지 설정합니다. 클수록 더 빠릅니다.",
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="마이크 음성 인식(STT)을 켭니다.",
    )
    parser.add_argument(
        "--stt-language",
        default="ko-KR",
        help="음성 인식 언어 코드. 예: ko-KR, en-US",
    )
    parser.add_argument(
        "--stt-phrase-seconds",
        type=float,
        default=1.8,
        help="한 번 인식할 때 모을 오디오 길이(초).",
    )
    parser.add_argument(
        "--stt-model",
        default="base",
        help="Whisper 모델 크기. 예: tiny, base, small, medium, large-v3",
    )
    parser.add_argument(
        "--stt-compute-type",
        default="int8",
        help="Whisper 연산 타입. 예: int8, int16, float16",
    )
    parser.add_argument(
        "--stt-beam-size",
        type=int,
        default=1,
        help="값이 클수록 정확도는 오를 수 있지만 지연도 늘어납니다.",
    )
    parser.add_argument(
        "--stt-best-of",
        type=int,
        default=1,
        help="더 나은 인식을 위한 샘플 후보 수.",
    )
    parser.add_argument(
        "--stt-no-speech-threshold",
        type=float,
        default=0.6,
        help="낮출수록 더 많은 오디오를 음성으로 간주합니다.",
    )
    parser.add_argument(
        "--stt-device",
        type=int,
        default=None,
        help="입력 오디오 장치 인덱스. --list-audio-devices로 확인할 수 있습니다.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="사용 가능한 입력 오디오 장치를 출력하고 종료합니다.",
    )
    parser.add_argument(
        "--warning-log-path",
        default="logs/warnings.jsonl",
        help="위험 이벤트 로그를 저장할 JSONL 파일 경로.",
    )
    parser.add_argument(
        "--warning-log-min-score",
        type=int,
        default=60,
        help="이 점수 이상일 때만 위험 이벤트 로그를 저장합니다.",
    )
    parser.add_argument(
        "--server-url",
        default="",
        help="비어 있지 않으면 사람/얼굴 영상 추론을 원격 서버로 보냅니다. 예: http://100.x.x.x:8000",
    )
    parser.add_argument(
        "--server-client-id",
        default="",
        help="원격 추론 서버에서 팀원별 추적 상태를 구분할 ID입니다. 비우면 자동 생성합니다.",
    )
    parser.add_argument(
        "--server-timeout-seconds",
        type=float,
        default=3.0,
        help="원격 추론 서버 요청 제한 시간(초).",
    )
    parser.add_argument(
        "--server-jpeg-quality",
        type=int,
        default=80,
        help="원격 전송용 JPEG 품질. 낮을수록 빠르지만 화질이 떨어집니다.",
    )
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
                "macOS에서 이 터미널 또는 앱의 카메라 권한이 막혀 있는 것 같습니다.",
                "시스템 설정 > 개인정보 보호 및 보안 > 카메라에서 현재 앱을 허용해 주세요.",
                "이전에 거부했다면 다음을 실행해 보세요: tccutil reset Camera",
                "그 뒤 터미널 또는 앱을 완전히 종료한 후 다시 실행해 주세요.",
            ]
        )
    else:
        details.extend(["", "영상 파일 경로가 존재하고 읽을 수 있는지 확인해 주세요."])
    return "\n".join(details)


def draw_people(frame, tracked_people):
    for person in tracked_people:
        person_id = person["id"]
        x, y, w, h = person["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 180, 99), 2)
        draw_unicode_text(frame, f"사람 {person_id}", (x, max(y - 28, 20)), (40, 180, 99), 24)


def draw_faces(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        draw_unicode_text(frame, "얼굴", (x, max(y - 26, 20)), (255, 200, 0), 22)


def localize_status(status: str) -> str:
    labels = {
        "idle": "대기",
        "starting": "시작 중",
        "loading": "불러오는 중",
        "listening": "듣는 중",
        "processing": "분석 중",
        "recognized": "인식됨",
        "warning": "경고",
        "error": "오류",
        "unavailable": "사용 불가",
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


def draw_speech(frame, result):
    color = (0, 220, 255)
    if result.status in {"error", "unavailable"}:
        color = (0, 90, 255)
    elif result.status == "recognized":
        color = (80, 220, 120)
    elif result.status == "processing":
        color = (255, 220, 80)

    transcript = result.transcript.strip() or "-"
    status_text = f"음성 인식: {localize_status(result.status)} | 레벨: {result.audio_level:.3f}"
    draw_unicode_text(frame, status_text, (20, 42), color, 22)

    message = transcript
    if result.error:
        message = result.error

    max_length = 60
    if len(message) > max_length:
        message = message[: max_length - 3] + "..."

    speech_text = f"인식 내용: {message}"
    draw_unicode_text(frame, speech_text, (20, 72), color, 24)


def draw_risk(frame, assessment):
    color = (90, 200, 90)
    if assessment.level == "ELEVATED":
        color = (0, 215, 255)
    elif assessment.level == "MEDIUM":
        color = (0, 140, 255)
    elif assessment.level == "HIGH":
        color = (0, 70, 255)

    draw_unicode_text(
        frame,
        f"위험도: {assessment.score}/100 | {localize_risk_level(assessment.level)}",
        (20, 108),
        color,
        24,
    )

    reason_parts = []
    if assessment.matched_keywords:
        reason_parts.append("키워드=" + ",".join(assessment.matched_keywords))
    if assessment.reasons:
        reason_parts.append("신호=" + ", ".join(assessment.reasons))
    message = " | ".join(reason_parts) if reason_parts else "신호=없음"
    if len(message) > 85:
        message = message[:82] + "..."

    draw_unicode_text(frame, message, (20, 142), color, 20)


def draw_fps(frame, fps: float):
    draw_unicode_text(frame, f"FPS: {fps:.1f}", (20, 174), (200, 200, 200), 20)


def draw_server_status(frame, status: ServerStatus):
    color = (120, 210, 120)
    if status.error:
        color = (0, 90, 255)
    text = status.mode
    if status.latency_ms > 0:
        text += f" | 서버 지연: {status.latency_ms:.0f}ms"
    if status.error:
        text += " | 연결 문제"
    draw_unicode_text(frame, text, (20, 198), color, 20)


def draw_counts(frame, people_count: int, face_count: int):
    draw_unicode_text(
        frame,
        f"사람: {people_count} | 얼굴: {face_count}",
        (20, 10),
        (0, 255, 255),
        24,
    )


def box_contains_face(person_box, faces) -> bool:
    px, py, pw, ph = person_box
    for fx, fy, fw, fh in faces:
        face_cx = fx + fw / 2
        face_cy = fy + fh / 2
        if px <= face_cx <= px + pw and py <= face_cy <= py + ph:
            return True
    return False


def filter_people(tracked_people, faces):
    filtered = []
    for person in tracked_people:
        stationary_frames = person.get("stationary_frames", 0)
        has_face = box_contains_face(person["bbox"], faces)
        if stationary_frames >= 15 and not has_face:
            continue
        filtered.append(person)
    return filtered


def main() -> None:
    args = parse_args()
    if args.list_audio_devices:
        devices = list_input_devices()
        if not devices:
            print("사용 가능한 입력 오디오 장치를 찾지 못했습니다.")
        else:
            for index, name, channels, sample_rate in devices:
                print(f"{index}: {name} | 입력채널={channels} | 기본샘플레이트={sample_rate:.0f}")
        sys.exit(0)

    remote_client = None
    if args.server_url.strip():
        remote_client = RemoteInferenceClient(
            server_url=args.server_url.strip(),
            client_id=args.server_client_id.strip() or None,
            timeout_seconds=args.server_timeout_seconds,
            jpeg_quality=args.server_jpeg_quality,
        )
        person_detector = None
        face_detector = None
        tracker = None
    else:
        person_detector = PersonDetector(
            scale=args.scale,
            min_neighbors=args.min_neighbors,
            score_threshold=args.person_score_threshold,
            nms_threshold=args.person_nms_threshold,
            resize_width=args.person_imgsz,
        )
        face_detector = FaceDetector()
        tracker = PersonTracker()

    risk_analyzer = RiskAnalyzer()
    event_logger = WarningEventLogger(
        log_path=Path(args.warning_log_path),
        min_score=args.warning_log_min_score,
    )
    speech_listener = None
    if args.stt:
        speech_listener = SpeechToTextListener(
            language=args.stt_language,
            phrase_time_limit=args.stt_phrase_seconds,
            model_size=args.stt_model,
            compute_type=args.stt_compute_type,
            input_device=args.stt_device,
            beam_size=args.stt_beam_size,
            best_of=args.stt_best_of,
            no_speech_threshold=args.stt_no_speech_threshold,
        )
        speech_listener.start()
    capture = open_source(args.source)

    if not capture.isOpened():
        if speech_listener is not None:
            speech_listener.stop()
        raise RuntimeError(build_open_error(args.source))

    window_name = "detectWarning - 사람 및 얼굴 감지"
    frame_index = 0
    tracked_people = []
    faces = []
    last_frame_time = perf_counter()
    smoothed_fps = 0.0
    server_status = ServerStatus(mode="원격 추론" if remote_client else "로컬 추론")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        now = perf_counter()
        instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
        last_frame_time = now
        if smoothed_fps == 0.0:
            smoothed_fps = instant_fps
        else:
            smoothed_fps = smoothed_fps * 0.9 + instant_fps * 0.1

        if remote_client is not None:
            remote_result = remote_client.analyze_frame(frame)
            if remote_result.error:
                server_status.error = remote_result.error
                server_status.latency_ms = remote_result.latency_ms
            else:
                tracked_people = remote_result.tracked_people
                faces = remote_result.faces
                server_status.error = None
                server_status.latency_ms = remote_result.latency_ms
        else:
            if frame_index % max(args.person_detect_interval, 1) == 0:
                people = person_detector.detect(frame)
                tracked_people = tracker.update(people)
            faces = face_detector.detect(frame)
            server_status.latency_ms = 0.0
            server_status.error = None

        draw_faces(frame, faces)
        visible_people = filter_people(tracked_people, faces)
        draw_people(frame, visible_people)

        draw_counts(frame, len(visible_people), len(faces))
        draw_fps(frame, smoothed_fps)
        draw_server_status(frame, server_status)
        if speech_listener is not None:
            speech_result = speech_listener.get_result()
            draw_speech(frame, speech_result)
        else:
            speech_result = None

        if speech_result is None:
            speech_result = SpeechResult(status="idle")

        risk_assessment = risk_analyzer.update(
            speech_result,
            tracked_people=visible_people,
            face_count=len(faces),
        )
        draw_risk(frame, risk_assessment)
        event_logger.maybe_log(
            now_monotonic=monotonic(),
            source=str(args.source),
            assessment=risk_assessment,
            speech_result=speech_result,
            people_count=len(visible_people),
            face_count=len(faces),
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    capture.release()
    if speech_listener is not None:
        speech_listener.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

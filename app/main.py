import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from audio_detector import SpeechResult, SpeechToTextListener, list_input_devices
from detector import FaceDetector, PersonDetector
from risk_analyzer import RiskAnalyzer
from tracker import PersonTracker

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf"),
]


def load_overlay_font(size: int):
    if ImageFont is None:
        return None

    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
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
        default=960,
        help="YOLO 사람 감지 입력 크기. 클수록 보통 더 정확합니다.",
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
    for person_id, (x, y, w, h) in tracked_people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 180, 99), 2)
        draw_unicode_text(frame, f"사람 {person_id}", (x, max(y - 24, 20)), (40, 180, 99), 20)


def draw_faces(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        draw_unicode_text(frame, "얼굴", (x, max(y - 22, 20)), (255, 200, 0), 18)


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
    cv2.putText(
        frame,
        status_text,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    message = transcript
    if result.error:
        message = result.error

    max_length = 60
    if len(message) > max_length:
        message = message[: max_length - 3] + "..."

    speech_text = f"인식 내용: {message}"
    draw_unicode_text(frame, speech_text, (20, 78), color, 22)


def draw_risk(frame, assessment):
    color = (90, 200, 90)
    if assessment.level == "ELEVATED":
        color = (0, 215, 255)
    elif assessment.level == "MEDIUM":
        color = (0, 140, 255)
    elif assessment.level == "HIGH":
        color = (0, 70, 255)

    cv2.putText(
        frame,
        f"위험도: {assessment.score}/100 | {localize_risk_level(assessment.level)}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    reason_parts = []
    if assessment.matched_keywords:
        reason_parts.append("키워드=" + ",".join(assessment.matched_keywords))
    if assessment.reasons:
        reason_parts.append("신호=" + ", ".join(assessment.reasons))
    message = " | ".join(reason_parts) if reason_parts else "신호=없음"
    if len(message) > 85:
        message = message[:82] + "..."

    draw_unicode_text(frame, message, (20, 138), color, 18)


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

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        people = person_detector.detect(frame)
        tracked_people = tracker.update(people)
        faces = face_detector.detect(frame)
        draw_people(frame, tracked_people)
        draw_faces(frame, faces)

        cv2.putText(
            frame,
            f"사람: {len(tracked_people)} | 얼굴: {len(faces)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        if speech_listener is not None:
            speech_result = speech_listener.get_result()
            draw_speech(frame, speech_result)
        else:
            speech_result = None

        if speech_result is None:
            speech_result = SpeechResult(status="idle")

        risk_assessment = risk_analyzer.update(
            speech_result,
            people_count=len(tracked_people),
            face_count=len(faces),
        )
        draw_risk(frame, risk_assessment)

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

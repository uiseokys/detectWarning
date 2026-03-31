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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people from webcam or a video file.")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index like 0, or a video file path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.05,
        help="Detection window scale factor.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="How many neighbors each candidate rectangle should have.",
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="Enable microphone speech-to-text.",
    )
    parser.add_argument(
        "--stt-language",
        default="ko-KR",
        help="Speech recognition language code, for example ko-KR or en-US.",
    )
    parser.add_argument(
        "--stt-phrase-seconds",
        type=float,
        default=1.8,
        help="How many seconds of audio to capture before each recognition request.",
    )
    parser.add_argument(
        "--stt-model",
        default="base",
        help="Whisper model size, for example tiny, base, small, medium, or large-v3.",
    )
    parser.add_argument(
        "--stt-compute-type",
        default="int8",
        help="Whisper compute type, for example int8, int16, or float16.",
    )
    parser.add_argument(
        "--stt-beam-size",
        type=int,
        default=1,
        help="Higher values can improve accuracy, but add latency.",
    )
    parser.add_argument(
        "--stt-best-of",
        type=int,
        default=1,
        help="Sampling candidates for better recognition quality.",
    )
    parser.add_argument(
        "--stt-no-speech-threshold",
        type=float,
        default=0.6,
        help="Lower values treat more audio as speech.",
    )
    parser.add_argument(
        "--stt-device",
        type=int,
        default=None,
        help="Input audio device index. Use --list-audio-devices to find it.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Print available input audio devices and exit.",
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
    details = [f"Could not open source: {source}"]
    if source.isdigit():
        details.extend(
            [
                "",
                "macOS camera access looks blocked for this terminal/app.",
                "Check System Settings > Privacy & Security > Camera and allow the app you are using.",
                "If it was previously denied, run: tccutil reset Camera",
                "Then close the terminal/app completely and try again.",
            ]
        )
    else:
        details.extend(["", "Check that the video file path exists and is readable."])
    return "\n".join(details)


def draw_people(frame, tracked_people):
    for person_id, (x, y, w, h) in tracked_people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 180, 99), 2)
        cv2.putText(
            frame,
            f"Person {person_id}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 180, 99),
            2,
        )


def draw_faces(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(
            frame,
            "Face",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )


def draw_speech(frame, result):
    color = (0, 220, 255)
    if result.status in {"error", "unavailable"}:
        color = (0, 90, 255)
    elif result.status == "recognized":
        color = (80, 220, 120)
    elif result.status == "processing":
        color = (255, 220, 80)

    transcript = result.transcript.strip() or "-"
    status_text = f"STT: {result.status} | Level: {result.audio_level:.3f}"
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

    speech_text = f"Speech: {message}"
    font = load_overlay_font(22)
    if font is None or Image is None or ImageDraw is None:
        cv2.putText(
            frame,
            speech_text,
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image)
    draw.text((20, 78), speech_text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


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
        f"Risk: {assessment.score}/100 | {assessment.level}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    reason_parts = []
    if assessment.matched_keywords:
        reason_parts.append("keywords=" + ",".join(assessment.matched_keywords))
    if assessment.reasons:
        reason_parts.append("signals=" + ", ".join(assessment.reasons))
    message = " | ".join(reason_parts) if reason_parts else "signals=none"
    if len(message) > 85:
        message = message[:82] + "..."

    cv2.putText(
        frame,
        message,
        (20, 152),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def main() -> None:
    args = parse_args()
    if args.list_audio_devices:
        devices = list_input_devices()
        if not devices:
            print("No input audio devices found.")
        else:
            for index, name, channels, sample_rate in devices:
                print(f"{index}: {name} | input_channels={channels} | default_sr={sample_rate:.0f}")
        sys.exit(0)

    person_detector = PersonDetector(scale=args.scale, min_neighbors=args.min_neighbors)
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

    window_name = "detectWarning - person and face detection"

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
            f"People: {len(tracked_people)} | Faces: {len(faces)}",
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

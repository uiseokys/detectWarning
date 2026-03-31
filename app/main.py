import argparse

import cv2

from audio_detector import SpeechToTextListener
from detector import FaceDetector, PersonDetector
from tracker import PersonTracker


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
        default=3.0,
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

    transcript = result.transcript.strip() or "-"
    status_text = f"STT: {result.status}"
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

    cv2.putText(
        frame,
        f"Speech: {message}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main() -> None:
    args = parse_args()
    person_detector = PersonDetector(scale=args.scale, min_neighbors=args.min_neighbors)
    face_detector = FaceDetector()
    tracker = PersonTracker()
    speech_listener = None
    if args.stt:
        speech_listener = SpeechToTextListener(
            language=args.stt_language,
            phrase_time_limit=args.stt_phrase_seconds,
            model_size=args.stt_model,
            compute_type=args.stt_compute_type,
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
            draw_speech(frame, speech_listener.get_result())

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

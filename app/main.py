import argparse

import cv2

from detector import FaceDetector, PersonDetector


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


def draw_people(frame, people):
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 180, 99), 2)
        cv2.putText(
            frame,
            "Person",
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


def main() -> None:
    args = parse_args()
    person_detector = PersonDetector(scale=args.scale, min_neighbors=args.min_neighbors)
    face_detector = FaceDetector()
    capture = open_source(args.source)

    if not capture.isOpened():
        raise RuntimeError(build_open_error(args.source))

    window_name = "detectWarning - person and face detection"

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        people = person_detector.detect(frame)
        faces = face_detector.detect(frame)
        draw_people(frame, people)
        draw_faces(frame, faces)

        cv2.putText(
            frame,
            f"People: {len(people)} | Faces: {len(faces)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

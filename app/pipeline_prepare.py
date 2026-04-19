from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detector import FaceDetector, PersonDetector
from person_classifier import PersonPresenceFilter
from tracker import PersonTracker


CONFIRMED_PERSON_STATES = {"full_body_person", "upper_body_person"}


def extract_pose_sequence(
    *,
    video_path: Path,
    person_detector: PersonDetector,
    face_detector: FaceDetector,
    sequence_length: int,
    max_frames_to_scan: int,
    detector_batch_size: int,
) -> dict:
    payload = load_video_sequence_payload(
        video_path=video_path,
        sequence_length=sequence_length,
        max_frames_to_scan=max_frames_to_scan,
    )
    return extract_pose_sequence_from_payload(
        payload=payload,
        person_detector=person_detector,
        face_detector=face_detector,
        detector_batch_size=detector_batch_size,
    )


def load_video_sequence_payload(
    *,
    video_path: Path,
    sequence_length: int,
    max_frames_to_scan: int,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상 파일을 열지 못했습니다: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_indices = build_frame_indices(total_frames, sequence_length, max_frames_to_scan)

    sampled_frames = read_sampled_frames(capture, frame_indices)
    valid_frames: list[np.ndarray] = []
    valid_time_indices: list[int] = []
    for time_index, frame in enumerate(sampled_frames):
        if frame is None:
            continue
        valid_frames.append(frame)
        valid_time_indices.append(time_index)
    capture.release()

    return {
        "video_path": str(video_path),
        "sequence_length": int(sequence_length),
        "frames": valid_frames,
        "time_indices": valid_time_indices,
    }


def extract_pose_sequence_from_payload(
    *,
    payload: dict,
    person_detector: PersonDetector,
    face_detector: FaceDetector,
    detector_batch_size: int,
) -> dict:
    sequence_length = int(payload.get("sequence_length", 0) or 0)
    sampled_frames = payload.get("frames") or []
    sampled_time_indices = payload.get("time_indices") or []
    tracker = PersonTracker()
    presence_filter = PersonPresenceFilter(debug=False)
    track_frames: dict[int, dict[int, dict]] = defaultdict(dict)

    if not sampled_frames:
        return {
            "pose": np.zeros((sequence_length, 17, 3), dtype=np.float32),
            "mask": np.zeros((sequence_length,), dtype=np.float32),
            "valid_frames": 0,
            "confirmed_frames": 0,
            "chosen_track_id": -1,
        }

    batch_size = max(int(detector_batch_size), 1)
    for batch_start in range(0, len(sampled_frames), batch_size):
        frame_batch = sampled_frames[batch_start:batch_start + batch_size]
        time_batch = sampled_time_indices[batch_start:batch_start + batch_size]
        gray_batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frame_batch]
        detections_batch = person_detector.detect_batch(frame_batch)

        for time_index, frame, gray, detections in zip(time_batch, frame_batch, gray_batch, detections_batch):
            tracked = tracker.update(detections)
            faces = face_detector.detect_gray(gray)
            evaluated = presence_filter.evaluate(tracked, faces, gray, frame.shape)
            for candidate in evaluated:
                if candidate.get("person_state") == "rejected":
                    continue
                candidate_id = int(candidate["id"])
                previous = track_frames[candidate_id].get(time_index)
                if previous is None or candidate.get("person_score", 0) >= previous.get("person_score", 0):
                    track_frames[candidate_id][time_index] = candidate

    if not track_frames:
        return {
            "pose": np.zeros((sequence_length, 17, 3), dtype=np.float32),
            "mask": np.zeros((sequence_length,), dtype=np.float32),
            "valid_frames": 0,
            "confirmed_frames": 0,
            "chosen_track_id": -1,
        }

    chosen_track_id = choose_best_track(track_frames)
    chosen_frames = track_frames[chosen_track_id]
    pose = np.zeros((sequence_length, 17, 3), dtype=np.float32)
    mask = np.zeros((sequence_length,), dtype=np.float32)
    valid_frames = 0
    confirmed_frames = 0

    for time_index in range(sequence_length):
        candidate = chosen_frames.get(time_index)
        if candidate is None:
            continue
        pose[time_index] = normalize_pose(candidate.get("keypoints", []), candidate["bbox"])
        mask[time_index] = 1.0
        valid_frames += 1
        if candidate.get("person_state") in CONFIRMED_PERSON_STATES:
            confirmed_frames += 1

    return {
        "pose": pose,
        "mask": mask,
        "valid_frames": valid_frames,
        "confirmed_frames": confirmed_frames,
        "chosen_track_id": chosen_track_id,
    }


def build_frame_indices(total_frames: int, sequence_length: int, max_frames_to_scan: int) -> list[int]:
    if total_frames > 0:
        effective_total = min(total_frames, max_frames_to_scan)
        if effective_total <= sequence_length:
            return list(range(effective_total))
        return np.linspace(0, effective_total - 1, num=sequence_length, dtype=int).tolist()
    return list(range(sequence_length))


def read_sampled_frames(capture: cv2.VideoCapture, frame_indices: list[int]):
    if not frame_indices:
        return []

    sampled = [None] * len(frame_indices)
    current_frame_index = 0
    last_frame = None

    for output_index, target_index in enumerate(frame_indices):
        target_index = max(int(target_index), 0)

        if last_frame is not None and current_frame_index - 1 == target_index:
            sampled[output_index] = last_frame.copy()
            continue

        if target_index < current_frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            current_frame_index = target_index
            last_frame = None

        while current_frame_index <= target_index:
            ok, frame = capture.read()
            if not ok:
                last_frame = None
                break
            last_frame = frame
            current_frame_index += 1

        sampled[output_index] = None if last_frame is None else last_frame.copy()

    return sampled


def choose_best_track(track_frames: dict[int, dict[int, dict]]) -> int:
    best_track_id = -1
    best_score = None
    for track_id, frames in track_frames.items():
        if not frames:
            continue
        confirmed_count = sum(
            1 for frame in frames.values() if frame.get("person_state") in CONFIRMED_PERSON_STATES
        )
        avg_score = sum(float(frame.get("person_score", 0.0)) for frame in frames.values()) / max(len(frames), 1)
        score = confirmed_count * 100.0 + len(frames) * 10.0 + avg_score
        if best_score is None or score > best_score:
            best_score = score
            best_track_id = track_id
    if best_track_id < 0:
        return next(iter(track_frames))
    return best_track_id


def normalize_pose(keypoints: list[dict], bbox) -> np.ndarray:
    x, y, w, h = bbox
    normalized = np.zeros((17, 3), dtype=np.float32)
    for index in range(min(len(keypoints), 17)):
        point = keypoints[index]
        conf = float(point.get("confidence", 0.0))
        if conf <= 0.0:
            continue
        normalized[index, 0] = float((float(point.get("x", 0.0)) - x) / max(w, 1))
        normalized[index, 1] = float((float(point.get("y", 0.0)) - y) / max(h, 1))
        normalized[index, 2] = conf
    return normalized

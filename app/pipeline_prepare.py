from __future__ import annotations

from collections import Counter, defaultdict
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
    allow_rejected_pose_fallback: bool = True,
    fallback_min_keypoints: int = 3,
    fallback_min_detection_confidence: float = 0.15,
    fallback_min_person_score: int = 20,
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
        allow_rejected_pose_fallback=allow_rejected_pose_fallback,
        fallback_min_keypoints=fallback_min_keypoints,
        fallback_min_detection_confidence=fallback_min_detection_confidence,
        fallback_min_person_score=fallback_min_person_score,
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
    allow_rejected_pose_fallback: bool = True,
    fallback_min_keypoints: int = 3,
    fallback_min_detection_confidence: float = 0.15,
    fallback_min_person_score: int = 20,
) -> dict:
    sequence_length = int(payload.get("sequence_length", 0) or 0)
    sampled_frames = payload.get("frames") or []
    sampled_time_indices = payload.get("time_indices") or []
    tracker = PersonTracker()
    presence_filter = PersonPresenceFilter(debug=False)
    track_frames: dict[int, dict[int, dict]] = defaultdict(dict)
    frame_stats = {
        "sampled_frames": len(sampled_frames),
        "loaded_frames": len(sampled_frames),
        "detection_frames": 0,
        "candidate_frames": 0,
        "accepted_frames": 0,
        "fallback_frames": 0,
        "rejected_candidates": 0,
        "rejection_reasons": Counter(),
    }

    if not sampled_frames:
        return build_empty_pose_sequence(
            sequence_length,
            skip_reason="no_decodable_frames",
            frame_stats=frame_stats,
        )

    def should_keep_candidate(candidate: dict) -> tuple[bool, str | None]:
        if candidate.get("person_state") != "rejected":
            return True, None
        frame_stats["rejected_candidates"] += 1
        for reason in candidate.get("debug_reasons") or ["rejected"]:
            frame_stats["rejection_reasons"][str(reason)] += 1
        if not allow_rejected_pose_fallback:
            return False, None
        valid_keypoints = int(candidate.get("valid_keypoint_count", 0) or 0)
        det_conf = float(candidate.get("det_conf", 0.0) or 0.0)
        person_score = int(candidate.get("person_score", 0) or 0)
        has_bbox = is_valid_bbox(candidate.get("bbox"))
        has_pose = valid_keypoints >= max(int(fallback_min_keypoints), 0)
        strong_enough_detection = det_conf >= float(fallback_min_detection_confidence)
        strong_enough_score = person_score >= int(fallback_min_person_score)
        if has_bbox and has_pose and (strong_enough_detection or strong_enough_score):
            return True, "rejected_pose_fallback"
        return False, None

    def choose_candidate(existing: dict | None, candidate: dict) -> dict:
        if existing is None:
            return candidate
        existing_score = candidate_selection_score(existing)
        candidate_score = candidate_selection_score(candidate)
        return candidate if candidate_score >= existing_score else existing

    batch_size = max(int(detector_batch_size), 1)
    for batch_start in range(0, len(sampled_frames), batch_size):
        frame_batch = sampled_frames[batch_start:batch_start + batch_size]
        time_batch = sampled_time_indices[batch_start:batch_start + batch_size]
        gray_batch = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frame_batch]
        detections_batch = person_detector.detect_batch(frame_batch)

        for time_index, frame, gray, detections in zip(
            time_batch,
            frame_batch,
            gray_batch,
            detections_batch,
            strict=False,
        ):
            if detections:
                frame_stats["detection_frames"] += 1
            tracked = tracker.update(detections)
            if tracked:
                frame_stats["candidate_frames"] += 1
            faces = face_detector.detect_gray(gray)
            evaluated = presence_filter.evaluate(tracked, faces, gray, frame.shape)
            for candidate in evaluated:
                keep_candidate, recovery_action = should_keep_candidate(candidate)
                if not keep_candidate:
                    continue
                if recovery_action:
                    candidate = {**candidate, "recovery_action": recovery_action}
                    frame_stats["fallback_frames"] += 1
                else:
                    frame_stats["accepted_frames"] += 1
                candidate_id = int(candidate["id"])
                previous = track_frames[candidate_id].get(time_index)
                track_frames[candidate_id][time_index] = choose_candidate(previous, candidate)

    if not track_frames:
        skip_reason = "person_not_detected"
        if frame_stats["detection_frames"] > 0:
            skip_reason = "pose_candidates_rejected"
        return build_empty_pose_sequence(
            sequence_length,
            skip_reason=skip_reason,
            frame_stats=frame_stats,
        )

    chosen_track_id = choose_best_track(track_frames)
    chosen_frames = track_frames[chosen_track_id]
    pose = np.zeros((sequence_length, 17, 3), dtype=np.float32)
    mask = np.zeros((sequence_length,), dtype=np.float32)
    valid_frames = 0
    confirmed_frames = 0
    fallback_frames = 0
    total_valid_keypoints = 0
    pose_confidences: list[float] = []
    recovery_actions: list[str] = []

    for time_index in range(sequence_length):
        candidate = chosen_frames.get(time_index)
        if candidate is None:
            continue
        bbox = candidate.get("bbox")
        if not is_valid_bbox(bbox):
            continue
        keypoints = candidate.get("keypoints", [])
        pose[time_index] = normalize_pose(keypoints, bbox)
        mask[time_index] = 1.0
        valid_frames += 1
        valid_keypoints = int(candidate.get("valid_keypoint_count", 0) or count_positive_keypoints(keypoints))
        total_valid_keypoints += valid_keypoints
        pose_confidences.append(float(candidate.get("pose_mean_conf", 0.0) or 0.0))
        recovery_action = candidate.get("recovery_action")
        if recovery_action:
            fallback_frames += 1
            recovery_actions.append(str(recovery_action))
        if candidate.get("person_state") in CONFIRMED_PERSON_STATES:
            confirmed_frames += 1

    return {
        "pose": pose,
        "mask": mask,
        "valid_frames": valid_frames,
        "confirmed_frames": confirmed_frames,
        "fallback_frames": fallback_frames,
        "total_valid_keypoints": total_valid_keypoints,
        "avg_pose_confidence": (
            float(sum(pose_confidences) / len(pose_confidences))
            if pose_confidences
            else 0.0
        ),
        "chosen_track_id": chosen_track_id,
        "skip_reason": None if valid_frames > 0 else "pose_missing",
        "recovery_actions": sorted(set(recovery_actions)),
        "frame_stats": normalize_frame_stats(frame_stats),
    }


def build_empty_pose_sequence(sequence_length: int, *, skip_reason: str, frame_stats: dict) -> dict:
    return {
        "pose": np.zeros((sequence_length, 17, 3), dtype=np.float32),
        "mask": np.zeros((sequence_length,), dtype=np.float32),
        "valid_frames": 0,
        "confirmed_frames": 0,
        "fallback_frames": 0,
        "total_valid_keypoints": 0,
        "avg_pose_confidence": 0.0,
        "chosen_track_id": -1,
        "skip_reason": skip_reason,
        "recovery_actions": [],
        "frame_stats": normalize_frame_stats(frame_stats),
    }


def normalize_frame_stats(frame_stats: dict) -> dict:
    normalized = dict(frame_stats)
    rejection_reasons = normalized.get("rejection_reasons") or {}
    normalized["rejection_reasons"] = dict(rejection_reasons)
    return normalized


def is_valid_bbox(bbox) -> bool:
    if bbox is None:
        return False
    try:
        _x, _y, w, h = bbox
    except (TypeError, ValueError):
        return False
    return float(w) > 0 and float(h) > 0


def count_positive_keypoints(keypoints: list[dict]) -> int:
    return sum(1 for point in keypoints or [] if float(point.get("confidence", 0.0) or 0.0) > 0.0)


def candidate_selection_score(candidate: dict) -> float:
    recovery_penalty = -5.0 if candidate.get("recovery_action") else 0.0
    return (
        float(candidate.get("person_score", 0.0) or 0.0)
        + (float(candidate.get("pose_mean_conf", 0.0) or 0.0) * 20.0)
        + (float(candidate.get("valid_keypoint_count", 0) or 0) * 2.0)
        + recovery_penalty
    )


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
        avg_score = sum(
            float(frame.get("person_score", 0.0)) for frame in frames.values()
        ) / max(len(frames), 1)
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

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


# 1차 사람 검출 confidence 최소 기준.
PERSON_DET_CONF_THRES = 0.35
# pose 평균 confidence가 이 값보다 낮으면 skeleton 품질이 불안정하다고 본다.
POSE_MEAN_CONF_THRES = 0.40
# full/half-body로 보기 위한 최소 유효 keypoint 수.
MIN_VALID_KPTS_FULL = 7
# upper-body-only로 보기 위한 최소 유효 keypoint 수.
MIN_VALID_KPTS_UPPER = 4
# full-body 판정 시 필요한 핵심 body part 수(shoulder/hip/lower 조합).
MIN_CORE_KPTS_FULL = 3
# upper-body 판정 시 필요한 핵심 body part 수(head/shoulder 조합).
MIN_CORE_KPTS_UPPER = 2
# 너무 작은 bbox는 사람 판정에서 제외한다.
MIN_BBOX_AREA = 48 * 48
# 사람 bbox 종횡비 최소/최대 허용 범위.
MIN_ASPECT_RATIO = 0.18
MAX_ASPECT_RATIO = 1.20
# full/upper_body 확정 전 필요한 최소 연속 검출 프레임 수.
MIN_TRACK_CONFIRM_FRAMES = 3
# tracker state를 얼마나 오래 유지할지.
MAX_TRACK_MISSING_FRAMES = 10
# static false positive를 판단할 때 볼 시간 창.
STATIC_FP_TIME_WINDOW = 18
# ROI frame-diff 평균이 이 값보다 낮으면 거의 움직임이 없는 것으로 본다.
STATIC_MOTION_THRES = 1.6
# 최종 상태 구분용 점수 기준.
FULL_BODY_SCORE_THRES = 72
UPPER_BODY_SCORE_THRES = 58
UNCERTAIN_SCORE_THRES = 38
# 프레임 가장자리 crop 판정을 위한 margin 비율.
FRAME_EDGE_MARGIN_RATIO = 0.03
# keypoint가 유효하다고 보는 최소 confidence.
KPT_VISIBLE_CONF_THRES = 0.35


HEAD_KPTS = (0, 1, 2, 3, 4)
SHOULDER_KPTS = (5, 6)
ELBOW_KPTS = (7, 8)
WRIST_KPTS = (9, 10)
HIP_KPTS = (11, 12)
KNEE_KPTS = (13, 14)
ANKLE_KPTS = (15, 16)
LOWER_BODY_KPTS = HIP_KPTS + KNEE_KPTS + ANKLE_KPTS
TORSO_KPTS = SHOULDER_KPTS + HIP_KPTS

STATE_COLORS = {
    "full_body_person": (40, 180, 99),
    "upper_body_person": (80, 210, 220),
    "uncertain": (0, 215, 255),
    "rejected": (60, 80, 180),
}


@dataclass
class TrackEvidence:
    consecutive_frames: int = 0
    total_frames: int = 0
    missing_frames: int = 0
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_motion: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_pose_conf: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_det_conf: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_states: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_geometry_scores: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_bbox_centers: deque = field(default_factory=lambda: deque(maxlen=20))


class PersonPresenceFilter:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self._track_memory: dict[int, TrackEvidence] = {}
        self._previous_gray: np.ndarray | None = None

    def evaluate(
        self,
        tracked_people: list[dict],
        faces,
        gray_frame: np.ndarray,
        frame_shape,
    ) -> list[dict]:
        current_ids = {person["id"] for person in tracked_people}
        self._age_missing_tracks(current_ids)

        evaluations = []
        for person in tracked_people:
            evaluation = self._evaluate_candidate(person, faces, gray_frame, frame_shape)
            evaluations.append(evaluation)

        self._previous_gray = gray_frame.copy()
        return evaluations

    def draw_debug_overlay(self, frame, evaluations: list[dict], draw_pose_fn=None) -> None:
        for candidate in evaluations:
            state = candidate["person_state"]
            if not self.debug and state == "rejected":
                continue

            color = STATE_COLORS.get(state, (0, 255, 255))
            x, y, w, h = candidate["bbox"]
            if draw_pose_fn is not None and candidate.get("keypoints"):
                draw_pose_fn(frame, candidate["keypoints"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2 if state != "rejected" else 1)

            candidate_id = candidate.get("id", "-")
            label = f"#{candidate_id} {state} {candidate['person_score']}"
            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            if self.debug:
                debug_reason = ",".join(candidate.get("debug_reasons", [])[:2]) or "-"
                cv2.putText(
                    frame,
                    debug_reason,
                    (x, min(y + h + 18, frame.shape[0] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )

    def _evaluate_candidate(self, person: dict, faces, gray_frame: np.ndarray, frame_shape) -> dict:
        bbox = tuple(int(value) for value in person["bbox"])
        keypoints = list(person.get("keypoints", []))
        det_conf = float(person.get("det_conf", 0.0))
        pose_mean_conf = float(person.get("pose_mean_conf", 0.0))
        track_id = int(person["id"])
        has_face = self._box_contains_face(bbox, faces)
        motion_roi = self._compute_motion_magnitude(gray_frame, bbox)

        pose_quality = self.evaluate_pose_quality(keypoints, pose_mean_conf)
        geometry = self.evaluate_geometry(
            bbox=bbox,
            keypoints=keypoints,
            frame_shape=frame_shape,
            visibility_hint=pose_quality["visibility_hint"],
        )
        visibility_level = self.classify_visibility_level(
            bbox=bbox,
            pose_quality=pose_quality,
            geometry=geometry,
            has_face=has_face,
        )
        evidence = self._track_memory.setdefault(track_id, TrackEvidence())
        temporal = self.evaluate_temporal_consistency(
            evidence=evidence,
            bbox=bbox,
            det_conf=det_conf,
            pose_mean_conf=pose_quality["pose_mean_conf"],
            geometry_score=geometry["score"],
            motion_roi=motion_roi,
            candidate_motion=float(person.get("movement", 0.0)),
        )
        static_fp = self.evaluate_static_fp_risk(
            evidence=evidence,
            det_conf=det_conf,
            pose_quality=pose_quality,
            geometry=geometry,
            motion_roi=motion_roi,
            has_face=has_face,
            stationary_frames=int(person.get("stationary_frames", 0)),
            visibility_level=visibility_level,
        )
        person_score = self.compute_person_score(
            det_conf=det_conf,
            pose_quality=pose_quality,
            geometry=geometry,
            temporal=temporal,
            static_fp=static_fp,
            visibility_level=visibility_level,
            has_face=has_face,
        )
        person_state, decision_reasons = self.finalize_person_state(
            person_score=person_score,
            det_conf=det_conf,
            pose_quality=pose_quality,
            geometry=geometry,
            temporal=temporal,
            static_fp=static_fp,
            visibility_level=visibility_level,
            has_face=has_face,
        )

        evidence.consecutive_frames += 1
        evidence.total_frames += 1
        evidence.missing_frames = 0
        evidence.recent_scores.append(person_score)
        evidence.recent_motion.append(motion_roi)
        evidence.recent_pose_conf.append(pose_quality["pose_mean_conf"])
        evidence.recent_det_conf.append(det_conf)
        evidence.recent_states.append(person_state)
        evidence.recent_geometry_scores.append(geometry["score"])

        return {
            **person,
            "bbox": bbox,
            "keypoints": keypoints,
            "has_face": has_face,
            "det_conf": round(det_conf, 4),
            "pose_mean_conf": round(pose_quality["pose_mean_conf"], 4),
            "valid_keypoint_count": pose_quality["valid_count"],
            "core_keypoint_count": pose_quality["core_count"],
            "motion_roi": round(motion_roi, 3),
            "visibility_level": visibility_level,
            "person_score": int(round(person_score)),
            "person_state": person_state,
            "debug_reasons": decision_reasons,
            "geometry_score": round(geometry["score"], 3),
            "confirm_frames": evidence.consecutive_frames,
        }

    def classify_visibility_level(
        self,
        *,
        bbox,
        pose_quality: dict,
        geometry: dict,
        has_face: bool,
    ) -> str:
        del bbox
        if pose_quality["full_candidate"] and geometry["full_body_ok"]:
            return "full_body_person"
        if pose_quality["upper_candidate"] and geometry["upper_body_ok"]:
            return "upper_body_person"
        if has_face or pose_quality["head_count"] >= 1 or pose_quality["shoulder_count"] >= 1:
            return "uncertain"
        return "rejected"

    def evaluate_pose_quality(self, keypoints: list[dict], pose_mean_conf: float) -> dict:
        visible_indices = [
            index
            for index, point in enumerate(keypoints)
            if float(point.get("confidence", 0.0)) >= KPT_VISIBLE_CONF_THRES
        ]
        confidences = [
            float(keypoints[index].get("confidence", 0.0))
            for index in visible_indices
        ]
        mean_conf = sum(confidences) / len(confidences) if confidences else pose_mean_conf

        head_count = self._count_visible(keypoints, HEAD_KPTS)
        shoulder_count = self._count_visible(keypoints, SHOULDER_KPTS)
        hip_count = self._count_visible(keypoints, HIP_KPTS)
        lower_count = self._count_visible(keypoints, LOWER_BODY_KPTS)
        torso_count = self._count_visible(keypoints, TORSO_KPTS)

        full_core_count = shoulder_count + hip_count + min(lower_count, 2)
        upper_core_count = min(head_count, 2) + shoulder_count

        full_candidate = (
            len(visible_indices) >= MIN_VALID_KPTS_FULL
            and full_core_count >= MIN_CORE_KPTS_FULL
            and shoulder_count >= 1
            and (hip_count >= 1 or lower_count >= 2 or torso_count >= 3)
        )
        upper_candidate = (
            len(visible_indices) >= MIN_VALID_KPTS_UPPER
            and upper_core_count >= MIN_CORE_KPTS_UPPER
            and shoulder_count >= 1
            and (head_count >= 1 or shoulder_count == 2)
        )

        return {
            "pose_mean_conf": mean_conf,
            "valid_count": len(visible_indices),
            "visible_indices": visible_indices,
            "head_count": head_count,
            "shoulder_count": shoulder_count,
            "hip_count": hip_count,
            "lower_count": lower_count,
            "torso_count": torso_count,
            "core_count": full_core_count if full_candidate else upper_core_count,
            "full_candidate": full_candidate,
            "upper_candidate": upper_candidate,
            "visibility_hint": "full" if full_candidate else "upper" if upper_candidate else "borderline",
        }

    def evaluate_geometry(self, *, bbox, keypoints: list[dict], frame_shape, visibility_hint: str) -> dict:
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        area = w * h
        aspect_ratio = w / max(h, 1)
        score = 0.0
        penalties = 0.0
        reasons: list[str] = []

        border_margin_x = frame_w * FRAME_EDGE_MARGIN_RATIO
        border_margin_y = frame_h * FRAME_EDGE_MARGIN_RATIO
        border_crop = (
            x <= border_margin_x
            or y <= border_margin_y
            or x + w >= frame_w - border_margin_x
            or y + h >= frame_h - border_margin_y
        )

        if area < MIN_BBOX_AREA:
            penalties += 24
            reasons.append("low_bbox_area")
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            penalties += 16
            reasons.append("bad_aspect_ratio")

        head_points = self._visible_points(keypoints, HEAD_KPTS)
        shoulder_points = self._visible_points(keypoints, SHOULDER_KPTS)
        hip_points = self._visible_points(keypoints, HIP_KPTS)
        all_points = self._visible_points(keypoints, tuple(range(len(keypoints))))

        full_body_ok = False
        upper_body_ok = False

        if len(shoulder_points) == 2:
            shoulder_span = abs(shoulder_points[0]["x"] - shoulder_points[1]["x"])
            shoulder_ratio = shoulder_span / max(w, 1)
            if 0.16 <= shoulder_ratio <= 0.95:
                score += 10
            else:
                penalties += 10
                reasons.append("bad_shoulder_span")

        if head_points and shoulder_points:
            head_y = min(point["y"] for point in head_points)
            shoulder_y = sum(point["y"] for point in shoulder_points) / len(shoulder_points)
            if head_y < shoulder_y:
                score += 8
            else:
                penalties += 8
                reasons.append("bad_head_torso")

        if shoulder_points and hip_points:
            shoulder_y = sum(point["y"] for point in shoulder_points) / len(shoulder_points)
            hip_y = sum(point["y"] for point in hip_points) / len(hip_points)
            torso_height = hip_y - shoulder_y
            if 0.15 * h <= torso_height <= 0.75 * h:
                score += 12
                full_body_ok = True
            else:
                penalties += 10
                reasons.append("bad_torso_ratio")

        if all_points:
            xs = [point["x"] for point in all_points]
            ys = [point["y"] for point in all_points]
            spread_x = (max(xs) - min(xs)) / max(w, 1)
            spread_y = (max(ys) - min(ys)) / max(h, 1)
            if spread_x < 0.10 or spread_y < 0.12:
                penalties += 10
                reasons.append("collapsed_skeleton")
            else:
                score += 6

        if visibility_hint == "upper":
            if head_points and shoulder_points:
                upper_body_ok = True
                score += 8
            if not hip_points and not border_crop:
                score += 5
                reasons.append("occlusion_aware_upper")
        elif visibility_hint == "full":
            upper_body_ok = True
            if full_body_ok:
                score += 10
        else:
            if head_points or shoulder_points:
                upper_body_ok = True

        return {
            "score": score - penalties,
            "raw_score": score,
            "penalties": penalties,
            "reasons": reasons,
            "border_crop": border_crop,
            "full_body_ok": full_body_ok,
            "upper_body_ok": upper_body_ok,
        }

    def evaluate_temporal_consistency(
        self,
        *,
        evidence: TrackEvidence,
        bbox,
        det_conf: float,
        pose_mean_conf: float,
        geometry_score: float,
        motion_roi: float,
        candidate_motion: float,
    ) -> dict:
        center = self._bbox_center(bbox)
        evidence.recent_bbox_centers.append(center)

        score = 0.0
        reasons: list[str] = []
        if evidence.consecutive_frames + 1 >= MIN_TRACK_CONFIRM_FRAMES:
            score += min(18, (evidence.consecutive_frames + 1) * 4)
            reasons.append(f"temporal_confirm:{evidence.consecutive_frames + 1}")
        else:
            score -= 8
            reasons.append("unstable_track")

        if candidate_motion > 110:
            score -= 8
            reasons.append("bbox_jump")

        if evidence.recent_scores:
            avg_recent_score = sum(evidence.recent_scores) / len(evidence.recent_scores)
            if avg_recent_score >= 55:
                score += 6
        if evidence.recent_pose_conf:
            avg_pose = sum(evidence.recent_pose_conf) / len(evidence.recent_pose_conf)
            if avg_pose >= POSE_MEAN_CONF_THRES:
                score += 4
        if motion_roi >= STATIC_MOTION_THRES:
            score += 3
        if det_conf >= PERSON_DET_CONF_THRES + 0.20:
            score += 4
        if geometry_score > 0:
            score += 4

        return {"score": score, "reasons": reasons}

    def evaluate_static_fp_risk(
        self,
        *,
        evidence: TrackEvidence,
        det_conf: float,
        pose_quality: dict,
        geometry: dict,
        motion_roi: float,
        has_face: bool,
        stationary_frames: int,
        visibility_level: str,
    ) -> dict:
        penalty = 0.0
        reasons: list[str] = []

        avg_motion = (
            sum(evidence.recent_motion) / len(evidence.recent_motion)
            if evidence.recent_motion
            else motion_roi
        )
        long_static = max(stationary_frames, evidence.consecutive_frames) >= STATIC_FP_TIME_WINDOW
        low_motion = avg_motion < STATIC_MOTION_THRES and motion_roi < STATIC_MOTION_THRES
        weak_pose = (
            pose_quality["valid_count"] < MIN_VALID_KPTS_UPPER + 1
            or pose_quality["pose_mean_conf"] < POSE_MEAN_CONF_THRES
        )
        weak_geometry = geometry["score"] < 0
        low_det = det_conf < PERSON_DET_CONF_THRES + 0.08

        if long_static and low_motion and weak_pose and weak_geometry and low_det and not has_face:
            penalty += 28
            reasons.append("static_fp_risk")
        elif long_static and low_motion and weak_geometry and not has_face:
            penalty += 18
            reasons.append("static_like_pattern")

        if visibility_level == "upper_body_person" and has_face:
            penalty *= 0.3
        if visibility_level == "full_body_person" and pose_quality["valid_count"] >= MIN_VALID_KPTS_FULL:
            penalty *= 0.2

        return {"penalty": penalty, "reasons": reasons}

    def compute_person_score(
        self,
        *,
        det_conf: float,
        pose_quality: dict,
        geometry: dict,
        temporal: dict,
        static_fp: dict,
        visibility_level: str,
        has_face: bool,
    ) -> float:
        score = 0.0
        score += self._scaled_score(det_conf, PERSON_DET_CONF_THRES, 1.0, 24)
        score += self._scaled_score(pose_quality["pose_mean_conf"], POSE_MEAN_CONF_THRES, 1.0, 18)

        if visibility_level == "full_body_person":
            score += min(16, pose_quality["valid_count"] * 1.8)
        elif visibility_level == "upper_body_person":
            score += min(14, pose_quality["valid_count"] * 2.0)
        else:
            score += min(10, pose_quality["valid_count"] * 1.5)

        score += max(geometry["score"], -20) + temporal["score"]
        if has_face:
            score += 8
        if visibility_level == "upper_body_person" and has_face:
            score += 6
        if visibility_level == "full_body_person":
            score += 10

        score -= static_fp["penalty"]
        return max(min(score, 100), 0)

    def finalize_person_state(
        self,
        *,
        person_score: float,
        det_conf: float,
        pose_quality: dict,
        geometry: dict,
        temporal: dict,
        static_fp: dict,
        visibility_level: str,
        has_face: bool,
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []

        if det_conf < PERSON_DET_CONF_THRES * 0.75:
            reasons.append("low_det_conf")
        if pose_quality["valid_count"] < MIN_VALID_KPTS_UPPER:
            reasons.append("too_few_valid_kpts")
        if pose_quality["core_count"] < MIN_CORE_KPTS_UPPER:
            reasons.append("too_few_core_kpts")
        if geometry["score"] < -8:
            reasons.append("bad_geometry")
        if "unstable_track" in temporal["reasons"]:
            reasons.append("unstable_track")
        reasons.extend(static_fp["reasons"])

        if (
            person_score < UNCERTAIN_SCORE_THRES
            or (not has_face and pose_quality["valid_count"] < MIN_VALID_KPTS_UPPER)
        ):
            return "rejected", list(dict.fromkeys(reasons))[:3]

        if (
            visibility_level == "full_body_person"
            and person_score >= FULL_BODY_SCORE_THRES
            and "unstable_track" not in reasons
        ):
            return "full_body_person", ["confirmed_full_body", *reasons][:3]

        if (
            visibility_level == "upper_body_person"
            and person_score >= UPPER_BODY_SCORE_THRES
            and (has_face or pose_quality["shoulder_count"] >= 2 or "unstable_track" not in reasons)
        ):
            return "upper_body_person", ["confirmed_upper_body", *reasons][:3]

        return "uncertain", ["pending_temporal_check", *reasons][:3]

    def _age_missing_tracks(self, current_ids: set[int]) -> None:
        for track_id in list(self._track_memory):
            if track_id in current_ids:
                continue
            memory = self._track_memory[track_id]
            memory.missing_frames += 1
            memory.consecutive_frames = 0
            if memory.missing_frames > MAX_TRACK_MISSING_FRAMES:
                del self._track_memory[track_id]

    def _compute_motion_magnitude(self, gray_frame: np.ndarray, bbox) -> float:
        if self._previous_gray is None or self._previous_gray.shape != gray_frame.shape:
            return 0.0
        x, y, w, h = bbox
        x0 = max(int(x), 0)
        y0 = max(int(y), 0)
        x1 = min(int(x + w), gray_frame.shape[1])
        y1 = min(int(y + h), gray_frame.shape[0])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        diff = cv2.absdiff(self._previous_gray[y0:y1, x0:x1], gray_frame[y0:y1, x0:x1])
        return float(diff.mean()) if diff.size else 0.0

    @staticmethod
    def _bbox_center(bbox) -> tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _count_visible(keypoints: list[dict], indices: tuple[int, ...]) -> int:
        return sum(
            1
            for index in indices
            if index < len(keypoints) and float(keypoints[index].get("confidence", 0.0)) >= KPT_VISIBLE_CONF_THRES
        )

    @staticmethod
    def _visible_points(keypoints: list[dict], indices: tuple[int, ...]) -> list[dict]:
        return [
            keypoints[index]
            for index in indices
            if index < len(keypoints) and float(keypoints[index].get("confidence", 0.0)) >= KPT_VISIBLE_CONF_THRES
        ]

    @staticmethod
    def _scaled_score(value: float, lower: float, upper: float, cap: float) -> float:
        if value <= lower:
            return 0.0
        if value >= upper:
            return cap
        return ((value - lower) / max(upper - lower, 1e-6)) * cap

    @staticmethod
    def _box_contains_face(person_box, faces) -> bool:
        px, py, pw, ph = person_box
        for fx, fy, fw, fh in faces:
            face_cx = fx + fw / 2
            face_cy = fy + fh / 2
            if px <= face_cx <= px + pw and py <= face_cy <= py + ph:
                return True
        return False

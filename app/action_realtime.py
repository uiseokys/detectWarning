from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np

from extract_rgb_video_features import RgbFeatureExtractor
from hybrid_pose_ensemble import build_feature_matrix
from pipeline_prepare import normalize_pose
from reporting import read_json


@dataclass
class RealtimeActionResult:
    available: bool
    status: str
    label: str = "unknown"
    confidence: float = 0.0
    abnormal_score: float = 0.0
    threshold: float = 0.5
    model: str = ""
    reason: str = ""
    probabilities: dict[str, float] | None = None


class RealtimeActionRecognizer:
    def __init__(
        self,
        *,
        artifacts_dir: str | Path,
        enabled: bool = True,
        device: str = "cuda",
        rgb_model: str = "i3d_r50",
        clip_seconds: float = 4.0,
        min_interval_seconds: float = 1.5,
        sequence_length: int = 32,
        rgb_frames: int = 16,
        image_size: int = 112,
        normal_threshold: float = 0.78,
        min_action_confidence: float = 0.45,
        collapse_static_motion_threshold: float = 4.0,
        collapse_static_abnormal_threshold: float = 0.90,
    ) -> None:
        self.enabled = bool(enabled)
        self.artifacts_dir = Path(artifacts_dir)
        self.clip_seconds = max(float(clip_seconds), 1.0)
        self.min_interval_seconds = max(float(min_interval_seconds), 0.25)
        self.sequence_length = max(int(sequence_length), 4)
        self.normal_threshold = min(max(float(normal_threshold), 0.0), 0.99)
        self.min_action_confidence = min(max(float(min_action_confidence), 0.0), 0.99)
        self.collapse_static_motion_threshold = max(float(collapse_static_motion_threshold), 0.0)
        self.collapse_static_abnormal_threshold = min(max(float(collapse_static_abnormal_threshold), 0.0), 0.99)
        self.last_inference_at = 0.0
        self.frame_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=max(int(self.clip_seconds * 30), rgb_frames, 8))
        self.pose_buffer: deque[tuple[float, list[dict]]] = deque(maxlen=max(self.sequence_length * 2, 8))
        self._last_candidate_label = ""
        self._candidate_streak = 0
        self.latest = RealtimeActionResult(available=False, status="warming_up", reason="not_enough_frames")
        self.detection = self._load_feature_model("detection")
        self.classification = self._load_feature_model("classification")
        self.detection_threshold = self._load_detection_threshold()
        self.extractor = None
        if self.enabled and (self.detection or self.classification):
            try:
                self.extractor = RgbFeatureExtractor(
                    model_name=rgb_model,
                    device=device,
                    frames=rgb_frames,
                    image_size=image_size,
                    allow_fallback=True,
                )
            except Exception as exc:
                self.latest = RealtimeActionResult(
                    available=False,
                    status="unavailable",
                    reason=f"rgb extractor load failed: {exc}",
                )
                self.enabled = False

    def _load_feature_model(self, task_name: str) -> dict | None:
        model_path = self.artifacts_dir / "specialized_tasks" / task_name / "best_feature_model.pkl"
        if not model_path.exists():
            return None
        try:
            with model_path.open("rb") as handle:
                payload = pickle.load(handle)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _load_detection_threshold(self) -> float:
        metrics_path = self.artifacts_dir / "specialized_tasks" / "detection" / "metrics.json"
        payload = read_json(metrics_path) or {}
        best = payload.get("best_result") if isinstance(payload.get("best_result"), dict) else {}
        threshold = best.get("threshold")
        if threshold is None:
            validation = best.get("validation") if isinstance(best.get("validation"), dict) else {}
            detection = validation.get("detection") if isinstance(validation.get("detection"), dict) else {}
            threshold = detection.get("threshold")
        try:
            return float(threshold)
        except (TypeError, ValueError):
            return 0.5

    def update(self, frame_bgr: np.ndarray, tracked_people: list[dict]) -> RealtimeActionResult:
        now = monotonic()
        if not self.enabled:
            return self.latest
        self.frame_buffer.append((now, frame_bgr.copy()))
        self.pose_buffer.append((now, list(tracked_people or [])))
        self._trim_buffers(now)
        if now - self.last_inference_at < self.min_interval_seconds:
            return self.latest
        if len(self.frame_buffer) < 4:
            self.latest = RealtimeActionResult(available=False, status="warming_up", reason="not_enough_frames")
            return self.latest
        self.last_inference_at = now
        try:
            self.latest = self._predict()
        except Exception as exc:
            self.latest = RealtimeActionResult(available=False, status="error", reason=str(exc))
        return self.latest

    def _trim_buffers(self, now: float) -> None:
        while self.frame_buffer and now - self.frame_buffer[0][0] > self.clip_seconds:
            self.frame_buffer.popleft()
        while self.pose_buffer and now - self.pose_buffer[0][0] > self.clip_seconds:
            self.pose_buffer.popleft()

    def _predict(self) -> RealtimeActionResult:
        if self.extractor is None:
            return RealtimeActionResult(available=False, status="unavailable", reason="rgb extractor unavailable")
        frames = [frame for _ts, frame in self.frame_buffer]
        rgb_feature = self.extractor.extract_frames(frames)
        pose_payload = self._build_pose_payload()
        feature_vector = self._build_feature_vector(rgb_feature, pose_payload)

        detection_payload = self._predict_with_model(self.detection, feature_vector)
        abnormal_score = 0.0
        if detection_payload:
            detection_labels = detection_payload["labels"]
            abnormal_index = detection_labels.index("abnormal") if "abnormal" in detection_labels else min(len(detection_labels) - 1, 1)
            abnormal_score = float(np.clip(detection_payload["probabilities"][abnormal_index], 0.0, 1.0))

        classification_payload = self._predict_with_model(self.classification, feature_vector)
        effective_threshold = max(self.detection_threshold, self.normal_threshold)
        label = "normal"
        confidence = 1.0 - abnormal_score
        probabilities: dict[str, float] = {"normal": max(0.0, min(1.0, 1.0 - abnormal_score))}
        if abnormal_score >= effective_threshold and classification_payload:
            labels = classification_payload["labels"]
            probs = classification_payload["probabilities"]
            best_index = int(np.argmax(probs))
            candidate_label = str(labels[best_index])
            candidate_confidence = float(np.clip(probs[best_index], 0.0, 1.0))
            scaled_confidence = float(np.clip(abnormal_score * candidate_confidence, 0.0, 1.0))
            action_probabilities = {
                str(label_name): float(abnormal_score * probs[index])
                for index, label_name in enumerate(labels)
            }
            probabilities.update(action_probabilities)
            if self._accept_action_label(
                candidate_label,
                abnormal_score=abnormal_score,
                confidence=candidate_confidence,
                pose_payload=pose_payload,
            ):
                label = candidate_label
                confidence = scaled_confidence
        elif classification_payload:
            self._last_candidate_label = ""
            self._candidate_streak = 0
            labels = classification_payload["labels"]
            probs = classification_payload["probabilities"]
            probabilities.update({str(label_name): float(abnormal_score * probs[index]) for index, label_name in enumerate(labels)})
        else:
            self._last_candidate_label = ""
            self._candidate_streak = 0

        return RealtimeActionResult(
            available=True,
            status="ok",
            label=label,
            confidence=confidence,
            abnormal_score=abnormal_score,
            threshold=effective_threshold,
            model=str((self.classification or {}).get("model_name") or ""),
            probabilities=probabilities,
        )

    def _accept_action_label(
        self,
        label: str,
        *,
        abnormal_score: float,
        confidence: float,
        pose_payload: dict,
    ) -> bool:
        if label == self._last_candidate_label:
            self._candidate_streak += 1
        else:
            self._last_candidate_label = label
            self._candidate_streak = 1
        if confidence < self.min_action_confidence:
            return False
        if label == "collapse":
            avg_movement = float(pose_payload.get("avg_movement") or 0.0)
            if (
                avg_movement < self.collapse_static_motion_threshold
                and abnormal_score < self.collapse_static_abnormal_threshold
            ):
                return False
            if self._candidate_streak < 2 and abnormal_score < 0.94:
                return False
        elif label == "loitering":
            if self._candidate_streak < 2 and abnormal_score < 0.92:
                return False
        elif label == "violence":
            if self._candidate_streak < 2 and abnormal_score < 0.88 and confidence < 0.68:
                return False
        return True

    def _build_pose_payload(self) -> dict:
        pose = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
        mask = np.zeros((self.sequence_length,), dtype=np.float32)
        if not self.pose_buffer:
            return {
                "pose": pose,
                "mask": mask,
                "valid_frames": 0,
                "confirmed_frames": 0,
                "fallback_frames": 0,
                "total_valid_keypoints": 0,
                "avg_pose_confidence": 0.0,
                "avg_movement": 0.0,
            }
        items = list(self.pose_buffer)
        indices = np.linspace(0, len(items) - 1, num=self.sequence_length, dtype=int).tolist()
        valid_frames = 0
        total_keypoints = 0
        confidences: list[float] = []
        movements: list[float] = []
        for output_index, source_index in enumerate(indices):
            people = items[source_index][1]
            person = self._select_person(people)
            if person is None:
                continue
            bbox = person.get("bbox")
            keypoints = person.get("keypoints") or []
            pose[output_index] = normalize_pose(keypoints, bbox)
            mask[output_index] = 1.0
            valid_frames += 1
            total_keypoints += int(person.get("valid_keypoint_count", 0) or self._count_keypoints(keypoints))
            confidences.append(float(person.get("pose_mean_conf", 0.0) or 0.0))
            movements.append(float(person.get("movement", 0.0) or 0.0))
        return {
            "pose": pose,
            "mask": mask,
            "valid_frames": valid_frames,
            "confirmed_frames": valid_frames,
            "fallback_frames": 0,
            "total_valid_keypoints": total_keypoints,
            "avg_pose_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
            "avg_movement": float(sum(movements) / len(movements)) if movements else 0.0,
        }

    @staticmethod
    def _select_person(people: list[dict]) -> dict | None:
        if not people:
            return None
        return max(
            people,
            key=lambda person: (
                float(person.get("person_score", 0.0) or 0.0),
                float(person.get("pose_mean_conf", 0.0) or 0.0),
                float(person.get("movement", 0.0) or 0.0),
            ),
        )

    @staticmethod
    def _count_keypoints(keypoints: list[dict]) -> int:
        return sum(1 for point in keypoints if float(point.get("confidence", 0.0) or 0.0) > 0.0)

    def _build_feature_vector(self, rgb_feature: np.ndarray, pose_payload: dict) -> np.ndarray:
        row = {
            "label_idx": 0,
            "target_label": "normal",
            "pose_array": pose_payload["pose"],
            "pose_mask": pose_payload["mask"],
            "rgb_feature": np.asarray(rgb_feature, dtype=np.float32),
            "valid_frames": int(pose_payload.get("valid_frames") or 0),
            "confirmed_frames": int(pose_payload.get("confirmed_frames") or 0),
            "fallback_frames": int(pose_payload.get("fallback_frames") or 0),
            "total_valid_keypoints": int(pose_payload.get("total_valid_keypoints") or 0),
            "avg_pose_confidence": float(pose_payload.get("avg_pose_confidence") or 0.0),
            "clip_start_seconds": 0.0,
            "clip_end_seconds": self.clip_seconds,
            "sample_weight": 1.0,
        }
        matrix, _labels = build_feature_matrix([row])
        return matrix[0]

    def _predict_with_model(self, payload: dict | None, feature_vector: np.ndarray) -> dict | None:
        if not payload:
            return None
        model = payload.get("model")
        labels = [str(label) for label in payload.get("labels") or []]
        if model is None or not labels:
            return None
        expected_dim = self._expected_feature_dim(model)
        features = self._pad_or_trim(feature_vector, expected_dim).reshape(1, -1)
        probabilities = self._safe_probabilities(np.asarray(model.predict_proba(features), dtype=np.float64)[0])
        return {"labels": labels, "probabilities": probabilities}

    @staticmethod
    def _expected_feature_dim(model) -> int:
        value = getattr(model, "n_features_in_", None)
        if value is None and hasattr(model, "__getitem__"):
            try:
                value = getattr(model[-1], "n_features_in_", None)
            except Exception:
                value = None
        return int(value or 0)

    @staticmethod
    def _pad_or_trim(values: np.ndarray, expected_dim: int) -> np.ndarray:
        values = np.nan_to_num(np.asarray(values, dtype=np.float32).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        if expected_dim <= 0 or values.size == expected_dim:
            return values
        output = np.zeros(expected_dim, dtype=np.float32)
        output[: min(values.size, expected_dim)] = values[: min(values.size, expected_dim)]
        return output

    @staticmethod
    def _safe_probabilities(values: np.ndarray) -> np.ndarray:
        probabilities = np.nan_to_num(np.asarray(values, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        total = float(probabilities.sum())
        if total <= 0.0:
            if probabilities.size == 0:
                return probabilities
            return np.full_like(probabilities, 1.0 / probabilities.size, dtype=np.float64)
        return probabilities / total

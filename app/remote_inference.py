from __future__ import annotations

import socket
import uuid
from dataclasses import dataclass
from time import perf_counter

import cv2
import requests


@dataclass
class RemoteInferenceResult:
    tracked_people: list[dict]
    faces: list[tuple[int, int, int, int]]
    latency_ms: float
    error: str | None = None


class RemoteInferenceClient:
    def __init__(
        self,
        server_url: str,
        client_id: str | None = None,
        timeout_seconds: float = 3.0,
        jpeg_quality: int = 80,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.client_id = client_id or self._build_client_id()
        self.timeout_seconds = timeout_seconds
        self.jpeg_quality = int(max(min(jpeg_quality, 100), 40))
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def analyze_frame(self, frame) -> RemoteInferenceResult:
        started_at = perf_counter()
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            return RemoteInferenceResult(
                tracked_people=[],
                faces=[],
                latency_ms=0.0,
                error="프레임 JPEG 인코딩에 실패했습니다.",
            )

        try:
            response = self.session.post(
                f"{self.server_url}/analyze/frame",
                params={"client_id": self.client_id},
                data=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return RemoteInferenceResult(
                tracked_people=[],
                faces=[],
                latency_ms=(perf_counter() - started_at) * 1000.0,
                error=f"원격 추론 서버 요청 실패: {exc}",
            )

        tracked_people = []
        for person in payload.get("tracked_people", []):
            bbox = tuple(int(value) for value in person.get("bbox", [0, 0, 0, 0]))
            keypoints = []
            for point in person.get("keypoints", []):
                keypoints.append(
                    {
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                        "confidence": float(point.get("confidence", 0.0)),
                    }
                )
            parsed_person = dict(person)
            parsed_person.update(
                {
                    "id": int(person.get("id", 0)),
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "stationary_frames": int(person.get("stationary_frames", 0)),
                    "movement": float(person.get("movement", 0.0)),
                    "det_conf": float(person.get("det_conf", 0.0)),
                    "pose_mean_conf": float(person.get("pose_mean_conf", 0.0)),
                    "valid_keypoint_count": int(person.get("valid_keypoint_count", 0)),
                    "core_keypoint_count": int(person.get("core_keypoint_count", 0)),
                    "motion_roi": float(person.get("motion_roi", 0.0)),
                    "visibility_level": str(person.get("visibility_level", "")),
                    "person_score": int(person.get("person_score", 0)),
                    "person_state": str(person.get("person_state", "rejected")),
                    "geometry_score": float(person.get("geometry_score", 0.0)),
                    "confirm_frames": int(person.get("confirm_frames", 0)),
                    "has_face": bool(person.get("has_face", False)),
                    "debug_reasons": list(person.get("debug_reasons", [])),
                }
            )
            tracked_people.append(parsed_person)

        faces = [
            tuple(int(value) for value in face_box)
            for face_box in payload.get("faces", [])
        ]
        return RemoteInferenceResult(
            tracked_people=tracked_people,
            faces=faces,
            latency_ms=float(payload.get("latency_ms", (perf_counter() - started_at) * 1000.0)),
            error=payload.get("error"),
        )

    @staticmethod
    def _build_client_id() -> str:
        host = socket.gethostname().replace(" ", "-")
        return f"{host}-{uuid.uuid4().hex[:8]}"

import os
from pathlib import Path

import cv2


POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class PersonDetector:
    def __init__(
        self,
        scale: float = 1.03,
        min_neighbors: int = 5,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        resize_width: int = 640,
        device: str = "cuda:0",
    ) -> None:
        self.scale = scale
        self.min_neighbors = min_neighbors
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.resize_width = resize_width
        self.device = device
        self._validate_device(device)
        config_dir = Path(__file__).resolve().parent.parent / ".ultralytics"
        config_dir.mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Could not import Ultralytics YOLO. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.model = YOLO("yolo11n-pose.pt")

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.score_threshold,
            imgsz=self.resize_width,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        keypoints = results[0].keypoints
        if boxes is None or boxes.xyxy is None:
            return []

        people = []
        keypoint_xy = []
        keypoint_conf = []
        if keypoints is not None and keypoints.xy is not None:
            keypoint_xy = keypoints.xy.cpu().tolist()
        if keypoints is not None and keypoints.conf is not None:
            keypoint_conf = keypoints.conf.cpu().tolist()

        for index, xyxy in enumerate(boxes.xyxy.cpu().tolist()):
            x1, y1, x2, y2 = [int(value) for value in xyxy]
            x = max(x1, 0)
            y = max(y1, 0)
            w = max(x2 - x1, 0)
            h = max(y2 - y1, 0)
            if w == 0 or h == 0:
                continue
            person_keypoints = []
            xy_points = keypoint_xy[index] if index < len(keypoint_xy) else []
            conf_points = keypoint_conf[index] if index < len(keypoint_conf) else []
            for point_index, xy in enumerate(xy_points):
                px, py = xy
                confidence = conf_points[point_index] if point_index < len(conf_points) else 0.0
                person_keypoints.append(
                    {
                        "x": float(px),
                        "y": float(py),
                        "confidence": float(confidence),
                    }
                )
            people.append(
                {
                    "bbox": (x, y, w, h),
                    "keypoints": person_keypoints,
                }
            )
        return people

    @staticmethod
    def _validate_device(device: str) -> None:
        normalized = str(device).strip().lower()
        if not normalized.startswith("cuda"):
            return
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(
                "GPU 장치를 요청했지만 PyTorch를 불러오지 못했습니다."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "YOLO에 GPU 장치를 요청했지만 현재 PyTorch가 CUDA를 사용할 수 없습니다.\n"
                f"- 요청한 장치: {device}\n"
                f"- torch.cuda.is_available(): {torch.cuda.is_available()}\n"
                f"- torch.cuda.device_count(): {torch.cuda.device_count()}\n"
                "현재 Windows 파이썬 환경에 CUDA 지원 PyTorch가 설치되지 않았을 가능성이 큽니다."
            )

        if ":" in normalized:
            try:
                device_index = int(normalized.split(":", 1)[1])
            except ValueError:
                return
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"요청한 CUDA 장치 {device} 를 찾지 못했습니다. "
                    f"현재 사용 가능한 GPU 개수: {torch.cuda.device_count()}"
                )


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        if self.cascade.empty():
            raise RuntimeError(f"Could not load face cascade: {cascade_path}")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
        )
        return list(faces)

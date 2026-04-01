import os
from pathlib import Path

import cv2


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

        self.model = YOLO("yolo26n.pt")

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
        if boxes is None or boxes.xyxy is None:
            return []

        people = []
        for xyxy in boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = [int(value) for value in xyxy]
            x = max(x1, 0)
            y = max(y1, 0)
            w = max(x2 - x1, 0)
            h = max(y2 - y1, 0)
            if w == 0 or h == 0:
                continue
            people.append((x, y, w, h))
        return people


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

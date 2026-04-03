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

        self.detector_model = YOLO("yolo11n.pt")
        self.pose_model = YOLO("yolo11n-pose.pt")

    def detect(self, frame):
        candidates = self.detect_person_boxes(frame)
        return self.estimate_pose_in_boxes(frame, candidates)

    def detect_person_boxes(self, frame):
        results = self.detector_model.predict(
            source=frame,
            classes=[0],
            conf=self.score_threshold,
            iou=self.nms_threshold,
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
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        frame_h, frame_w = frame.shape[:2]
        for index, xyxy in enumerate(boxes.xyxy.cpu().tolist()):
            x1, y1, x2, y2 = [int(value) for value in xyxy]
            x = max(x1, 0)
            y = max(y1, 0)
            w = min(max(x2 - x1, 0), frame_w - x)
            h = min(max(y2 - y1, 0), frame_h - y)
            if w == 0 or h == 0:
                continue
            people.append(
                {
                    "bbox": (x, y, w, h),
                    "det_conf": float(confidences[index]) if index < len(confidences) else 0.0,
                }
            )
        return people

    def estimate_pose_in_boxes(self, frame, candidates):
        if not candidates:
            return []

        poses = []
        for candidate in candidates:
            x, y, w, h = candidate["bbox"]
            crop_box = self._expand_crop_box(frame.shape, x, y, w, h)
            cx, cy, cw, ch = crop_box
            if cw <= 0 or ch <= 0:
                poses.append({**candidate, "keypoints": [], "pose_mean_conf": 0.0})
                continue

            crop = frame[cy:cy + ch, cx:cx + cw]
            if crop.size == 0:
                poses.append({**candidate, "keypoints": [], "pose_mean_conf": 0.0})
                continue

            pose_imgsz = max(256, min(self.resize_width, max(crop.shape[:2])))
            pose_results = self.pose_model.predict(
                source=crop,
                classes=[0],
                conf=max(self.score_threshold * 0.5, 0.15),
                imgsz=pose_imgsz,
                device=self.device,
                verbose=False,
            )
            keypoints, pose_mean_conf = self._extract_pose_from_crop(pose_results, crop_box)
            poses.append(
                {
                    **candidate,
                    "keypoints": keypoints,
                    "pose_mean_conf": pose_mean_conf,
                    "crop_bbox": crop_box,
                }
            )
        return poses

    @staticmethod
    def _expand_crop_box(frame_shape, x: int, y: int, w: int, h: int):
        frame_h, frame_w = frame_shape[:2]
        pad = int(max(w, h) * 0.08)
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, frame_w)
        y1 = min(y + h + pad, frame_h)
        return (x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))

    @staticmethod
    def _extract_pose_from_crop(results, crop_box):
        if not results:
            return [], 0.0

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if (
            boxes is None
            or boxes.xyxy is None
            or keypoints is None
            or keypoints.xy is None
        ):
            return [], 0.0

        keypoint_xy = keypoints.xy.cpu().tolist()
        keypoint_conf = keypoints.conf.cpu().tolist() if keypoints.conf is not None else []
        box_conf = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        best_index = PersonDetector._select_best_pose_index(result, crop_box)
        if best_index is None:
            return [], 0.0

        xy_points = keypoint_xy[best_index] if best_index < len(keypoint_xy) else []
        conf_points = keypoint_conf[best_index] if best_index < len(keypoint_conf) else []
        crop_x, crop_y, _crop_w, _crop_h = crop_box
        person_keypoints = []
        visible_confidences = []
        for point_index, xy in enumerate(xy_points):
            px, py = xy
            confidence = conf_points[point_index] if point_index < len(conf_points) else 0.0
            confidence = float(confidence)
            person_keypoints.append(
                {
                    "x": float(px + crop_x),
                    "y": float(py + crop_y),
                    "confidence": confidence,
                }
            )
            if confidence > 0.0:
                visible_confidences.append(confidence)

        pose_mean_conf = (
            sum(visible_confidences) / len(visible_confidences)
            if visible_confidences
            else (float(box_conf[best_index]) if best_index < len(box_conf) else 0.0)
        )
        return person_keypoints, float(pose_mean_conf)

    @staticmethod
    def _select_best_pose_index(result, crop_box):
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return None

        crop_x, crop_y, crop_w, crop_h = crop_box
        crop_cx = crop_x + crop_w / 2.0
        crop_cy = crop_y + crop_h / 2.0

        best_index = None
        best_score = None
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        for index, xyxy in enumerate(boxes.xyxy.cpu().tolist()):
            x1, y1, x2, y2 = xyxy
            center_x = crop_x + (x1 + x2) / 2.0
            center_y = crop_y + (y1 + y2) / 2.0
            center_distance = abs(center_x - crop_cx) + abs(center_y - crop_cy)
            conf = float(confidences[index]) if index < len(confidences) else 0.0
            score = conf * 1000.0 - center_distance
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        return best_index

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

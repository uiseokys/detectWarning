from math import hypot


class PersonTracker:
    def __init__(
        self,
        max_distance: float = 90.0,
        max_missing: int = 12,
        stationary_distance: float = 12.0,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.stationary_distance = stationary_distance
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        if not detections:
            self._age_tracks()
            self._drop_missing_tracks()
            return []

        centroids = [self._get_centroid(self._get_bbox(detection)) for detection in detections]
        updated_ids = set()

        for detection, centroid in zip(detections, centroids):
            bbox = self._get_bbox(detection)
            keypoints = self._get_keypoints(detection)
            track_id = self._find_best_match(centroid, updated_ids)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                stationary_frames = 0
                movement = 0.0
            else:
                previous_centroid = self.tracks[track_id]["centroid"]
                movement = hypot(
                    centroid[0] - previous_centroid[0],
                    centroid[1] - previous_centroid[1],
                )
                stationary_frames = (
                    self.tracks[track_id].get("stationary_frames", 0) + 1
                    if movement < self.stationary_distance
                    else 0
                )

            self.tracks[track_id] = {
                "bbox": bbox,
                "keypoints": keypoints,
                "centroid": centroid,
                "missing": 0,
                "stationary_frames": stationary_frames,
                "movement": movement,
            }
            updated_ids.add(track_id)

        for track_id in list(self.tracks):
            if track_id not in updated_ids:
                self.tracks[track_id]["missing"] += 1

        self._drop_missing_tracks()

        results = []
        for track_id in sorted(updated_ids):
            track = self.tracks[track_id]
            results.append(
                {
                    "id": track_id,
                    "bbox": track["bbox"],
                    "keypoints": track.get("keypoints", []),
                    "stationary_frames": track.get("stationary_frames", 0),
                    "movement": track.get("movement", 0.0),
                }
            )
        return results

    def _find_best_match(self, centroid, updated_ids):
        best_track_id = None
        best_distance = self.max_distance

        for track_id, track in self.tracks.items():
            if track_id in updated_ids or track["missing"] > self.max_missing:
                continue

            distance = hypot(
                centroid[0] - track["centroid"][0],
                centroid[1] - track["centroid"][1],
            )
            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        return best_track_id

    def _age_tracks(self):
        for track_id in list(self.tracks):
            self.tracks[track_id]["missing"] += 1

    def _drop_missing_tracks(self):
        for track_id in list(self.tracks):
            if self.tracks[track_id]["missing"] > self.max_missing:
                del self.tracks[track_id]

    @staticmethod
    def _get_bbox(detection):
        if isinstance(detection, dict):
            return detection.get("bbox", (0, 0, 0, 0))
        return detection

    @staticmethod
    def _get_keypoints(detection):
        if isinstance(detection, dict):
            return detection.get("keypoints", [])
        return []

    @staticmethod
    def _get_centroid(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

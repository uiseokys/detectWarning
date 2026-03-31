from math import hypot


class PersonTracker:
    def __init__(self, max_distance: float = 90.0, max_missing: int = 12) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        if not detections:
            self._age_tracks()
            self._drop_missing_tracks()
            return []

        centroids = [self._get_centroid(box) for box in detections]
        updated_ids = set()

        for detection, centroid in zip(detections, centroids):
            track_id = self._find_best_match(centroid, updated_ids)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1

            self.tracks[track_id] = {
                "bbox": detection,
                "centroid": centroid,
                "missing": 0,
            }
            updated_ids.add(track_id)

        for track_id in list(self.tracks):
            if track_id not in updated_ids:
                self.tracks[track_id]["missing"] += 1

        self._drop_missing_tracks()

        results = []
        for track_id in sorted(updated_ids):
            results.append((track_id, self.tracks[track_id]["bbox"]))
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
    def _get_centroid(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

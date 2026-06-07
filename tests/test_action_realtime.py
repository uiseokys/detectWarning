from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from action_realtime import RealtimeActionRecognizer


class RealtimeActionRecognizerTests(unittest.TestCase):
    def make_recognizer(self) -> RealtimeActionRecognizer:
        return RealtimeActionRecognizer(
            artifacts_dir=Path(__file__).resolve().parents[1] / "training_data" / "missing-artifacts",
            enabled=False,
        )

    def test_loitering_rejects_static_waiting_person(self) -> None:
        recognizer = self.make_recognizer()
        payload = {
            "valid_frames": 24,
            "avg_movement": 1.2,
            "center_displacement": 8.0,
            "max_stationary_frames": 30,
        }

        for _ in range(5):
            accepted = recognizer._accept_action_label(
                "loitering",
                abnormal_score=0.99,
                confidence=0.95,
                pose_payload=payload,
            )

        self.assertFalse(accepted)

    def test_loitering_requires_repeated_motion_evidence(self) -> None:
        recognizer = self.make_recognizer()
        payload = {
            "valid_frames": 24,
            "avg_movement": 8.0,
            "center_displacement": 68.0,
            "max_stationary_frames": 2,
        }

        first = recognizer._accept_action_label(
            "loitering",
            abnormal_score=0.93,
            confidence=0.80,
            pose_payload=payload,
        )
        second = recognizer._accept_action_label(
            "loitering",
            abnormal_score=0.93,
            confidence=0.80,
            pose_payload=payload,
        )
        third = recognizer._accept_action_label(
            "loitering",
            abnormal_score=0.93,
            confidence=0.80,
            pose_payload=payload,
        )

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertTrue(third)

    def test_violence_rejects_single_borderline_spike(self) -> None:
        recognizer = self.make_recognizer()

        accepted = recognizer._accept_action_label(
            "violence",
            abnormal_score=0.89,
            confidence=0.70,
            pose_payload={},
        )

        self.assertFalse(accepted)

    def test_violence_accepts_repeated_supported_signal(self) -> None:
        recognizer = self.make_recognizer()

        first = recognizer._accept_action_label(
            "violence",
            abnormal_score=0.93,
            confidence=0.76,
            pose_payload={},
        )
        second = recognizer._accept_action_label(
            "violence",
            abnormal_score=0.93,
            confidence=0.76,
            pose_payload={},
        )

        self.assertFalse(first)
        self.assertTrue(second)

    def test_violence_rejects_low_abnormal_even_with_confidence(self) -> None:
        recognizer = self.make_recognizer()

        first = recognizer._accept_action_label(
            "violence",
            abnormal_score=0.84,
            confidence=0.90,
            pose_payload={},
        )
        second = recognizer._accept_action_label(
            "violence",
            abnormal_score=0.84,
            confidence=0.90,
            pose_payload={},
        )

        self.assertFalse(first)
        self.assertFalse(second)

    def test_violence_accepts_one_shot_only_when_very_strong(self) -> None:
        recognizer = self.make_recognizer()

        accepted = recognizer._accept_action_label(
            "violence",
            abnormal_score=0.98,
            confidence=0.92,
            pose_payload={},
        )

        self.assertTrue(accepted)

    def test_collapse_adaptive_threshold_rejects_weak_pose_quality(self) -> None:
        recognizer = self.make_recognizer()

        accepted = recognizer._accept_action_label(
            "collapse",
            abnormal_score=0.96,
            confidence=0.50,
            pose_payload={"valid_frames": 2, "avg_pose_confidence": 0.20, "avg_movement": 12.0},
        )

        self.assertFalse(accepted)

    def test_classification_only_mode_can_emit_abnormal_label_without_binary_detection(self) -> None:
        recognizer = self.make_recognizer()
        recognizer.classification = {"model_name": "fake"}
        recognizer.detection = None
        recognizer.extractor = type("DummyExtractor", (), {"extract_frames": lambda self, _frames: [0.0]})()
        recognizer.frame_buffer.append((0.0, None))
        recognizer.normal_threshold = 0.86
        recognizer._build_pose_payload = lambda: {}
        recognizer._build_feature_vector = lambda _rgb, _pose: [0.0]
        recognizer._accept_action_label = lambda label, **_kwargs: label == "violence"
        recognizer._predict_with_model = lambda payload, _feature: {
            "labels": ["violence", "collapse", "loitering"],
            "probabilities": [0.91, 0.05, 0.04],
        } if payload else None

        result = recognizer._predict()

        self.assertEqual(result.label, "violence")
        self.assertGreater(result.abnormal_score, 0.9)

    def test_fight_bilstm_does_not_raise_low_abnormal_score_by_itself(self) -> None:
        recognizer = self.make_recognizer()
        recognizer.classification = {"model_name": "fake"}
        recognizer.detection = {"model_name": "fake-detection"}
        recognizer.extractor = type("DummyExtractor", (), {"extract_frames": lambda self, _frames: [0.0]})()
        recognizer.frame_buffer.append((0.0, None))
        recognizer.normal_threshold = 0.78
        recognizer.detection_threshold = 0.50
        recognizer.fight_bilstm_min_probability = 0.90
        recognizer.fight_bilstm_min_abnormal_support = 0.70
        recognizer.fight_bilstm = type("DummyFightBilstm", (), {"available": True})()
        recognizer._build_pose_payload = lambda: {}
        recognizer._build_feature_vector = lambda _rgb, _pose: [0.0]
        recognizer._predict_fight_bilstm = lambda _frames: self.fail("fight BiLSTM should not run on weak base signal")
        recognizer._predict_with_model = lambda payload, _feature: (
            {"labels": ["normal", "abnormal"], "probabilities": [0.62, 0.38]}
            if payload is recognizer.detection
            else {"labels": ["violence", "collapse", "loitering"], "probabilities": [0.55, 0.25, 0.20]}
        )

        result = recognizer._predict()

        self.assertEqual(result.label, "normal")
        self.assertAlmostEqual(result.abnormal_score, 0.38)
        self.assertNotIn("violence_fight_bilstm", result.probabilities)

    def test_fight_bilstm_can_confirm_violence_when_base_signal_is_supported(self) -> None:
        recognizer = self.make_recognizer()
        recognizer.classification = {"model_name": "fake"}
        recognizer.detection = {"model_name": "fake-detection"}
        recognizer.extractor = type("DummyExtractor", (), {"extract_frames": lambda self, _frames: [0.0]})()
        recognizer.frame_buffer.append((0.0, None))
        recognizer.normal_threshold = 0.86
        recognizer.detection_threshold = 0.50
        recognizer.fight_bilstm_min_probability = 0.90
        recognizer.fight_bilstm_min_abnormal_support = 0.70
        recognizer.fight_bilstm = type("DummyFightBilstm", (), {"available": True})()
        recognizer._build_pose_payload = lambda: {}
        recognizer._build_feature_vector = lambda _rgb, _pose: [0.0]
        recognizer._predict_fight_bilstm = lambda _frames: 0.90
        recognizer._predict_with_model = lambda payload, _feature: (
            {"labels": ["normal", "abnormal"], "probabilities": [0.26, 0.74]}
            if payload is recognizer.detection
            else {"labels": ["violence", "collapse", "loitering"], "probabilities": [0.66, 0.20, 0.14]}
        )
        recognizer._accept_action_label("violence", abnormal_score=0.74, confidence=0.90, pose_payload={}, fight_supported=True)

        result = recognizer._predict()

        self.assertEqual(result.label, "violence")
        self.assertAlmostEqual(result.abnormal_score, 0.74)
        self.assertGreaterEqual(result.probabilities["violence_fight_bilstm"], 0.90)

    def test_fall_bilstm_does_not_confirm_collapse_without_base_class_support(self) -> None:
        recognizer = self.make_recognizer()
        recognizer.classification = {"model_name": "fake"}
        recognizer.detection = {"model_name": "fake-detection"}
        recognizer.extractor = type("DummyExtractor", (), {"extract_frames": lambda self, _frames: [0.0]})()
        recognizer.frame_buffer.append((0.0, None))
        recognizer.normal_threshold = 0.78
        recognizer.detection_threshold = 0.50
        recognizer.fall_bilstm_min_probability = 0.84
        recognizer.fall_bilstm_min_abnormal_support = 0.62
        recognizer.fall_bilstm_min_class_probability = 0.35
        recognizer.fall_bilstm = type("DummyFallBilstm", (), {"available": True})()
        recognizer._build_pose_payload = lambda: {"valid_frames": 20, "avg_movement": 12.0, "avg_pose_confidence": 0.6}
        recognizer._build_feature_vector = lambda _rgb, _pose: [0.0]
        recognizer._predict_fall_bilstm = lambda _frames: 0.96
        recognizer._predict_with_model = lambda payload, _feature: (
            {"labels": ["normal", "abnormal"], "probabilities": [0.30, 0.70]}
            if payload is recognizer.detection
            else {"labels": ["violence", "collapse", "loitering"], "probabilities": [0.70, 0.20, 0.10]}
        )

        result = recognizer._predict()

        self.assertNotEqual(result.label, "collapse")
        self.assertIn("collapse_fall_bilstm", result.probabilities)

    def test_fall_bilstm_can_confirm_collapse_when_base_signal_is_supported(self) -> None:
        recognizer = self.make_recognizer()
        recognizer.classification = {"model_name": "fake"}
        recognizer.detection = {"model_name": "fake-detection"}
        recognizer.extractor = type("DummyExtractor", (), {"extract_frames": lambda self, _frames: [0.0]})()
        recognizer.frame_buffer.append((0.0, None))
        recognizer.normal_threshold = 0.78
        recognizer.detection_threshold = 0.50
        recognizer.fall_bilstm_min_probability = 0.84
        recognizer.fall_bilstm_min_abnormal_support = 0.62
        recognizer.fall_bilstm_min_class_probability = 0.35
        recognizer.fall_bilstm = type("DummyFallBilstm", (), {"available": True})()
        recognizer._build_pose_payload = lambda: {"valid_frames": 20, "avg_movement": 12.0, "avg_pose_confidence": 0.6}
        recognizer._build_feature_vector = lambda _rgb, _pose: [0.0]
        recognizer._predict_fall_bilstm = lambda _frames: 0.95
        recognizer._predict_with_model = lambda payload, _feature: (
            {"labels": ["normal", "abnormal"], "probabilities": [0.30, 0.70]}
            if payload is recognizer.detection
            else {"labels": ["violence", "collapse", "loitering"], "probabilities": [0.25, 0.42, 0.33]}
        )

        result = recognizer._predict()

        self.assertEqual(result.label, "collapse")
        self.assertGreater(result.abnormal_score, 0.85)
        self.assertGreaterEqual(result.probabilities["collapse_fall_bilstm"], 0.95)


if __name__ == "__main__":
    unittest.main()

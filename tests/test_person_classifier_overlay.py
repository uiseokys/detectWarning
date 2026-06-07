from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from person_classifier import person_overlay_color, person_overlay_label


class PersonOverlayTests(unittest.TestCase):
    def test_confirmed_person_label_is_plain_person_without_track_id(self) -> None:
        label = person_overlay_label(
            {"id": 3, "person_state": "full_body_person", "person_score": 82.4},
            fallback_index=0,
        )

        self.assertEqual(label, "사람")
        self.assertNotIn("#", label)

    def test_upper_body_person_label_is_not_distinct(self) -> None:
        label = person_overlay_label(
            {"id": 2, "person_state": "upper_body_person", "person_score": 71},
            fallback_index=0,
        )

        self.assertEqual(label, "사람")

    def test_missing_track_id_still_uses_plain_person_label(self) -> None:
        label = person_overlay_label(
            {"person_state": "uncertain", "person_score": 41},
            fallback_index=4,
        )

        self.assertEqual(label, "사람")

    def test_track_color_is_stable_per_person(self) -> None:
        first = person_overlay_color({"id": 1}, "full_body_person")
        repeated = person_overlay_color({"id": 1}, "upper_body_person")
        second = person_overlay_color({"id": 2}, "full_body_person")

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()

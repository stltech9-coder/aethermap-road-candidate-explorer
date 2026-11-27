"""Checks that tile overlap does not create duplicate roof proposals."""

import unittest

import building_model


def roof(x1, y1, x2, y2, confidence=0.9, clipped=False):
    ring = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
    return {"box": (x1, y1, x2 + 1, y2 + 1), "pixel_ring": ring,
            "area": (x2 - x1) * (y2 - y1), "confidence": confidence,
            "clipped": clipped}


class BuildingDeduplication(unittest.TestCase):
    def test_same_roof_from_two_tiles_keeps_complete_outline(self):
        complete = roof(100, 100, 200, 200, confidence=0.8)
        partial = roof(120, 100, 200, 200, confidence=0.99, clipped=True)
        self.assertEqual(building_model.deduplicate_buildings([partial, complete]), [complete])

    def test_neighboring_roofs_are_kept(self):
        first = roof(100, 100, 200, 200)
        next_roof = roof(201, 100, 301, 200)
        self.assertEqual(len(building_model.deduplicate_buildings([first, next_roof])), 2)

    def test_overlapping_boxes_with_distinct_shapes_are_kept(self):
        first = roof(100, 100, 200, 200)
        second = roof(180, 180, 280, 280)
        self.assertEqual(len(building_model.deduplicate_buildings([first, second])), 2)


if __name__ == "__main__":
    unittest.main()

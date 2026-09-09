import random
import unittest

from src.safe_crop import annotation_boxes, sample_window


class SafeCropTests(unittest.TestCase):
    def test_multiple_annotations_and_locations(self):
        record = {"labelingInfo": [
            {"box": {"location": [
                {"x": 1, "y": 2, "width": 3, "height": 4},
                {"x": 10, "y": 20, "width": 30, "height": 40}]}},
            {"polygon": {"location": [
                {"x1": 50, "y1": 50, "x2": 80, "y2": 50,
                 "x3": 60, "y3": 90}]}}
        ]}
        self.assertEqual(annotation_boxes(record),
                         [[1, 2, 4, 6], [10, 20, 40, 60], [50, 50, 80, 90]])

    def test_all_lesions_preserved_across_seeds(self):
        boxes = [(20.5, 30.2, 130.7, 170.9), (500, 400, 640, 590)]
        windows = set()
        for seed in range(500):
            window = sample_window(1000, 800, boxes, rng=random.Random(seed))
            windows.add(window)
            x, y, r, b = window
            self.assertTrue(0 <= x < r <= 1000 and 0 <= y < b <= 800)
            for x1, y1, x2, y2 in boxes:
                self.assertTrue(x <= x1 < x2 <= r and y <= y1 < y2 <= b)
        self.assertGreater(len(windows), 10)

    def test_edges_and_missing_annotations_fall_back(self):
        for boxes in [[], [(0, 0, 1920, 1080)],
                      [(0, 0, 10, 10), (1910, 1070, 1920, 1080)]]:
            self.assertEqual(sample_window(1920, 1080, boxes), (0, 0, 1920, 1080))

    def test_bad_coordinates_fail(self):
        for box in [(0, 0, float('nan'), 4), (-1, 0, 3, 4), (5, 5, 2, 2)]:
            with self.assertRaises(ValueError):
                sample_window(100, 100, [box])

    def test_reproducible(self):
        args = (1000, 800, [(400, 300, 500, 400)])
        self.assertEqual(sample_window(*args, rng=random.Random(42)),
                         sample_window(*args, rng=random.Random(42)))

    def test_polygon_cannot_silently_drop_invalid_points(self):
        for location in [[], [{'x1': 1, 'y1': 2, 'x2': 2, 'y2': 3,
                              'x3': 3, 'y3': 4, 'x4': 'bad', 'y4': 99}],
                         [{'x1': 1, 'y1': 2, 'x3': 3, 'y3': 4, 'x4': 9, 'y4': 99}]]:
            with self.assertRaises(ValueError):
                annotation_boxes({'labelingInfo': [{'polygon': {'location': location}}]})


if __name__ == '__main__':
    unittest.main()

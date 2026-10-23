import unittest

import cv2
import numpy as np

from ballontranslator.modules.textdetector.detector_ctbd import get_inpaint_bboxes


def _mask_from_regions(shape, regions):
    mask = np.zeros(shape[:2], np.uint8)
    for x1, y1, x2, y2 in regions:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


class CTBDInpaintRegionTest(unittest.TestCase):
    def setUp(self) -> None:
        # Two vertical "columns" whose outer strokes sit exactly on the box
        # edges, as when the model's box hugs a vertical CJK block.
        self.image = np.full((200, 200, 3), 255, np.uint8)
        self.box = [50, 40, 150, 160]
        self.left_stroke = (slice(40, 160), slice(50, 60))
        self.right_stroke = (slice(40, 160), slice(140, 150))
        for rows, cols in (self.left_stroke, self.right_stroke):
            self.image[rows, cols] = 0
        # A bubble outline crossing the crop must stay out of the mask.
        cv2.circle(self.image, (100, 100), 90, (0, 0, 0), 3)

    def test_strokes_on_the_box_edge_are_masked(self) -> None:
        mask = _mask_from_regions(self.image.shape, get_inpaint_bboxes(self.box, self.image))

        for stroke in (self.left_stroke, self.right_stroke):
            self.assertTrue(np.all(mask[stroke] == 255))

    def test_outline_crossing_the_crop_is_not_masked(self) -> None:
        mask = _mask_from_regions(self.image.shape, get_inpaint_bboxes(self.box, self.image))

        self.assertEqual(mask[33, 100], 0)
        self.assertEqual(mask[167, 100], 0)


if __name__ == '__main__':
    unittest.main()

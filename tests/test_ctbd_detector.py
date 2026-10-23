import unittest

import cv2
import numpy as np

from ballontranslator.modules.textdetector.detector_ctbd import (
    LOGGER,
    ImageSlicer,
    get_inpaint_bboxes,
)


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


class CTBDSliceMergeTest(unittest.TestCase):
    """Tall pages are detected in overlapping slices and then merged."""

    # One seam: the lower slice starts at 2880, the upper one ends at 3600.
    BANDS = [(2880, 3600)]
    EDGE_TOL = 11.0

    def setUp(self) -> None:
        self.slicer = ImageSlicer(LOGGER)

    def merge(self, boxes):
        return self.slicer.merge_overlapping_boxes(
            np.array(boxes), 6800, self.BANDS, self.EDGE_TOL
        ).tolist()

    def test_stacked_bubbles_away_from_a_seam_stay_separate(self) -> None:
        boxes = [[177, 117, 423, 585], [180, 966, 520, 1540]]

        self.assertEqual(len(self.merge(boxes)), 2)

    def test_bubbles_inside_the_overlap_but_not_cut_stay_separate(self) -> None:
        # Neither box reaches the edge of the slice it was detected in.
        boxes = [[194, 2663, 525, 3214], [177, 3518, 420, 3967]]

        self.assertEqual(len(self.merge(boxes)), 2)

    def test_a_box_cut_by_the_seam_is_rejoined(self) -> None:
        upper_piece = [180, 2950, 520, 3600]  # cut by the upper slice's bottom
        lower_piece = [180, 3560, 520, 4100]

        self.assertEqual(self.merge([upper_piece, lower_piece]), [[180, 2950, 520, 4100]])


if __name__ == '__main__':
    unittest.main()

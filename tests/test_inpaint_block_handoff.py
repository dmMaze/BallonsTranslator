import unittest

import cv2
import numpy as np

from ballontranslator.modules.inpaint.base import filter_mask_by_bboxes
from ballontranslator.modules.inpaint.inpaint_default import OpenCVInpainter
from ballontranslator.utils.textblock import TextBlock


class InpaintBlockHandoffTests(unittest.TestCase):
    """Guard what detectors may hand to the block-wise inpaint path."""

    def setUp(self) -> None:
        self.inpainter = OpenCVInpainter()
        self.inpainter.inpaint_by_block = True

    def test_block_without_lines_keeps_its_rect(self):
        # filter_mask_by_bboxes documents a block carrying only xyxy.
        mask = np.full((40, 50), 255, dtype=np.uint8)
        kept = filter_mask_by_bboxes(mask, [TextBlock(xyxy=[10, 10, 30, 20])])
        self.assertGreater(int((kept > 0).sum()), 0)
        self.assertEqual(int(kept[15, 20]), 255)
        self.assertEqual(int(kept[0, 0]), 0)

    def test_degenerate_blocks_are_skipped(self):
        # enlarge_window() reports an empty rect for these, which would reach
        # OpenCV as an empty crop.
        img = np.full((120, 160, 3), 240, dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=np.uint8)
        for xyxy in ([5, 5, 5, 5], [80, 80, 40, 40], [150, 110, 200, 160]):
            with self.subTest(xyxy=xyxy):
                out = self.inpainter.inpaint(img.copy(), mask.copy(), [TextBlock(xyxy)])
                self.assertEqual(out.shape, img.shape)

    def test_block_without_mask_is_still_inpainted(self):
        img = np.full((120, 200, 3), 235, dtype=np.uint8)
        cv2.rectangle(img, (70, 50), (140, 80), (0, 0, 0), -1)
        blk = TextBlock(
            [60, 40, 150, 90],
            lines=[[[60, 40], [150, 40], [150, 90], [60, 90]]],
        )
        out = self.inpainter.inpaint(
            img.copy(), np.zeros(img.shape[:2], dtype=np.uint8), [blk]
        )
        self.assertGreater(int((out != img).any(axis=2).sum()), 0)


if __name__ == '__main__':
    unittest.main()

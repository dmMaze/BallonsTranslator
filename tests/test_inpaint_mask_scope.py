"""The skip-simple-cases shortcut must never paint outside the repair mask."""
import unittest
from unittest import mock

import numpy as np

from ballontranslator.modules.inpaint.base import InpainterBase
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock import TextBlock


def _balloon_and_non_text(img, mask):
    balloon = np.full(img.shape[:2], 255, np.uint8)
    non_text = balloon.copy()
    non_text[mask > 127] = 0
    return balloon, non_text


class DummyInpainter(InpainterBase):
    inpaint_by_block = True
    _load_model_keys = None

    def __init__(self):
        self.name = 'dummy'
        self.params = None
        self.calls = []

    def _inpaint(self, img, mask, textblock_list=None, **kwargs):
        self.calls.append(np.array(mask, copy=True))
        result = np.copy(img)
        result[mask > 127] = 255
        return result


class FlatFillMaskScopeTest(unittest.TestCase):
    def setUp(self):
        self.check_patch = mock.patch.object(pcfg.module, 'check_need_inpaint', True)
        self.filter_patch = mock.patch.object(pcfg.module, 'filter_mask_by_bboxes', False)
        self.check_patch.start()
        self.filter_patch.start()

    def tearDown(self):
        self.filter_patch.stop()
        self.check_patch.stop()

    def test_full_image_shortcut_only_fills_mask_pixels(self):
        inpainter = DummyInpainter()
        inpainter.inpaint_by_block = False
        img = np.full((12, 12, 3), 120, np.uint8)
        mask = np.zeros((12, 12), np.uint8)
        mask[4:7, 4:7] = 255
        img[4:7, 4:7] = 10
        with mock.patch(
            'ballontranslator.modules.inpaint.base.extract_ballon_mask',
            side_effect=_balloon_and_non_text,
        ):
            result = inpainter.inpaint(img, mask, None, check_need_inpaint=True)

        self.assertEqual(inpainter.calls, [])
        expected = img.copy()
        expected[mask > 127] = 120
        np.testing.assert_array_equal(result, expected)

    def test_block_shortcut_only_fills_mask_pixels(self):
        inpainter = DummyInpainter()
        img = np.full((20, 20, 3), 120, np.uint8)
        mask = np.zeros((20, 20), np.uint8)
        mask[8:12, 8:12] = 255
        img[8:12, 8:12] = 10
        blocks = [TextBlock(xyxy=[6, 6, 14, 14])]
        with mock.patch(
            'ballontranslator.modules.inpaint.base.extract_ballon_mask',
            side_effect=_balloon_and_non_text,
        ):
            result = inpainter.inpaint(img, mask.copy(), blocks)

        self.assertEqual(inpainter.calls, [])
        expected = img.copy()
        expected[mask > 127] = 120
        np.testing.assert_array_equal(result, expected)

    def test_nonuniform_background_uses_the_model_path(self):
        inpainter = DummyInpainter()
        img = np.zeros((16, 16, 3), np.uint8)
        img[:] = np.arange(16, dtype=np.uint8)[None, :, None] * 15
        mask = np.zeros((16, 16), np.uint8)
        mask[6:10, 6:10] = 255
        blocks = [TextBlock(xyxy=[5, 5, 11, 11])]
        with mock.patch(
            'ballontranslator.modules.inpaint.base.extract_ballon_mask',
            side_effect=_balloon_and_non_text,
        ):
            inpainter.inpaint(img, mask.copy(), blocks)

        self.assertEqual(len(inpainter.calls), 1)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest import mock

import numpy as np

from ballontranslator.modules.textdetector.base import TEXTDETECTORS
from ballontranslator.modules.textdetector.detector_ysg import (
    YSGYoloCTDDetector,
    _build_box_mask,
    _combine_refined_mask,
)
from ballontranslator.utils.textblock import TextBlock


def _block(xyxy):
    x1, y1, x2, y2 = xyxy
    blk = TextBlock(lines=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
    blk.adjust_bbox()
    return blk


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return FakeTensor(self.values[index])

    def __int__(self):
        return int(self.values)


class FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor([0] * len(xyxy))


class FakePrediction:
    names = {0: 'balloon'}
    boxes = None
    obb = None


class FakeYoloModel:
    def __init__(self, xyxy):
        self.prediction = FakePrediction()
        self.prediction.boxes = FakeBoxes(xyxy)

    def predict(self, **kwargs):
        return [self.prediction]


class FakeCtdModel:
    def __init__(self, fine_mask, ctd_blocks):
        self.fine_mask = fine_mask
        self.ctd_blocks = ctd_blocks

    def __call__(self, img):
        return None, self.fine_mask, self.ctd_blocks


class TestCombineRefinedMask(unittest.TestCase):
    def test_fine_mask_is_clipped_to_yolo_boxes(self):
        box_mask = np.zeros((60, 60), np.uint8)
        box_mask[10:31, 10:41] = 255
        fine_mask = np.zeros_like(box_mask)
        fine_mask[20:23, 20:23] = 255
        # CTD also fires on a stylized sound effect outside the body-text box.
        fine_mask[45:55, 45:55] = 255
        blk = _block((10, 10, 40, 30))
        ctd_blk = TextBlock(xyxy=[12, 12, 38, 28])

        combined = _combine_refined_mask(box_mask, fine_mask, [blk], [ctd_blk])

        self.assertTrue(np.all(combined[20:23, 20:23] == 255))
        self.assertFalse(np.any(combined[45:55, 45:55]))
        self.assertFalse(np.any(combined[10:13, 10:13]))

    def test_line_without_ctd_detection_keeps_coarse_box(self):
        box_mask = np.zeros((40, 40), np.uint8)
        box_mask[5:25, 5:25] = 255
        fine_mask = np.zeros_like(box_mask)
        blk = _block((5, 5, 24, 24))

        combined = _combine_refined_mask(box_mask, fine_mask, [blk], [])

        self.assertTrue(np.array_equal(combined, box_mask))

    def test_line_with_ctd_block_but_empty_refinement_keeps_coarse_box(self):
        box_mask = np.zeros((40, 40), np.uint8)
        box_mask[5:25, 5:25] = 255
        fine_mask = np.zeros_like(box_mask)
        blk = _block((5, 5, 24, 24))
        ctd_blk = TextBlock(xyxy=[5, 5, 24, 24])

        combined = _combine_refined_mask(box_mask, fine_mask, [blk], [ctd_blk])

        self.assertTrue(np.array_equal(combined, box_mask))

    def test_ctd_refined_line_does_not_keep_coarse_box(self):
        box_mask = np.zeros((40, 40), np.uint8)
        box_mask[5:25, 5:25] = 255
        fine_mask = np.zeros_like(box_mask)
        fine_mask[10:12, 10:12] = 255
        blk = _block((5, 5, 24, 24))
        ctd_blk = TextBlock(xyxy=[5, 5, 24, 24])

        combined = _combine_refined_mask(box_mask, fine_mask, [blk], [ctd_blk])

        self.assertTrue(np.array_equal(combined, fine_mask))

    def test_build_box_mask_uses_all_block_polygons(self):
        mask = _build_box_mask(
            [_block((0, 0, 9, 9)), _block((20, 20, 29, 29))], 40, 40)

        self.assertTrue(np.any(mask[0:10, 0:10]))
        self.assertTrue(np.any(mask[20:30, 20:30]))
        self.assertFalse(np.any(mask[10:20, 10:20]))


class TestHybridDetector(unittest.TestCase):
    def setUp(self):
        self.detector = YSGYoloCTDDetector()
        self.detector.set_param_value('merge text lines', False)
        self.detector.set_param_value('mask dilate size', 0)

    def test_hybrid_detect_keeps_blocks_and_refines_mask(self):
        img = np.zeros((60, 60, 3), np.uint8)
        fine_mask = np.zeros((60, 60), np.uint8)
        fine_mask[20:23, 20:23] = 255
        fine_mask[45:55, 45:55] = 255
        self.detector.model = FakeYoloModel(np.array([[10, 10, 40, 30]], np.float32))
        self.detector.ctd_model = FakeCtdModel(
            fine_mask, [TextBlock(xyxy=[12, 12, 38, 28])])

        mask, blk_list = self.detector._detect(img, None)

        self.assertEqual(len(blk_list), 1)
        self.assertTrue(np.all(mask[20:23, 20:23] == 255))
        self.assertFalse(np.any(mask[45:55, 45:55]))

    def test_ctd_failure_returns_yolo_mask(self):
        img = np.zeros((60, 60, 3), np.uint8)
        self.detector.model = FakeYoloModel(np.array([[10, 10, 40, 30]], np.float32))

        class BrokenCtdModel:
            def __call__(self, img):
                raise RuntimeError('boom')

        self.detector.ctd_model = BrokenCtdModel()
        with mock.patch.object(self.detector.logger, 'warning') as warning:
            mask, blk_list = self.detector._detect(img, None)
        warning.assert_called_once()

        self.assertEqual(len(blk_list), 1)
        self.assertTrue(np.any(mask[10:31, 10:41]))

    def test_ctd_params_reset_cached_model(self):
        self.detector.ctd_model = object()
        self.detector.updateParam('ctd detect size', 1024)
        self.assertIsNone(self.detector.ctd_model)

    def test_hybrid_detector_is_registered(self):
        self.assertIn('ysg_ctd', TEXTDETECTORS.module_dict)
        self.assertIn('ctd detect size', YSGYoloCTDDetector.params)
        self.assertEqual(len(YSGYoloCTDDetector.download_file_list), 4)


if __name__ == '__main__':
    unittest.main()

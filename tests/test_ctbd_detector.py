import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ballontranslator.modules.textdetector.detector_ctbd import (
    LOGGER,
    ImageSlicer,
    RTDetrV2TextDetector,
    get_inpaint_bboxes,
)


def setUpModule() -> None:
    # Module constructors update their class parameters; do not leak test options.
    global detector_params
    detector_params = patch.object(
        RTDetrV2TextDetector, 'params', deepcopy(RTDetrV2TextDetector.params)
    )
    detector_params.start()


def tearDownModule() -> None:
    detector_params.stop()


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
        self.outline = np.zeros(self.image.shape[:2], np.uint8)
        cv2.circle(self.outline, (100, 100), 90, 255, 3)
        self.image[self.outline > 0] = 0

    def test_strokes_on_the_box_edge_are_masked(self) -> None:
        mask = _mask_from_regions(self.image.shape, get_inpaint_bboxes(self.box, self.image))

        for stroke in (self.left_stroke, self.right_stroke):
            self.assertTrue(np.all(mask[stroke] == 255))

    def test_outline_crossing_the_crop_is_not_masked(self) -> None:
        mask = _mask_from_regions(self.image.shape, get_inpaint_bboxes(self.box, self.image))

        self.assertFalse(np.any(mask[self.outline > 0]))

    def test_enclosed_outline_survives_mask_building_and_inpainting(self) -> None:
        from ballontranslator.modules.inpaint.inpaint_default import OpenCVInpainter
        from ballontranslator.utils.config import pcfg

        image = np.full_like(self.image, 255)
        image[50:150, 65:75] = 0
        image[50:150, 125:135] = 0
        outline = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(outline, (44, 34), (156, 166), 255, 2)
        image[outline > 0] = 0
        for inverted in (False, True):
            source = 255 - image if inverted else image
            for method in ('none', 'rectangle', 'hull'):
                with self.subTest(inverted=inverted, method=method):
                    detector = RTDetrV2TextDetector(device='cpu', mask_unification_method=method)
                    blocks, mask = detector._create_text_blocks_and_mask(
                        source, np.array([self.box]), np.array([[43, 33, 157, 167]])
                    )
                    self.assertFalse(np.any(mask[outline > 0]))
                    self.assertTrue(np.all(mask[50:150, 65:75] == 255))
                    self.assertTrue(np.all(mask[50:150, 125:135] == 255))
                    if method == 'none' and not inverted:
                        self.assertEqual(mask[100, 100], 0)
                    with patch.object(pcfg.module, 'filter_mask_by_bboxes', False), patch.object(
                        pcfg.module, 'check_need_inpaint', False
                    ):
                        result = OpenCVInpainter().inpaint(source, mask, blocks)
                    np.testing.assert_array_equal(result[outline > 0], source[outline > 0])

    def test_padding_does_not_add_unrelated_components(self) -> None:
        self.image[80:120, 43:47] = 0
        mask = _mask_from_regions(self.image.shape, get_inpaint_bboxes(self.box, self.image))
        self.assertFalse(np.any(mask[80:120, 43:47]))
        self.assertTrue(np.all(mask[self.left_stroke] == 255))

    def test_single_glyph_filling_the_box_is_kept(self) -> None:
        image = np.full_like(self.image, 255)
        image[40:160, 50:150] = 0
        mask = _mask_from_regions(image.shape, get_inpaint_bboxes(self.box, image))
        self.assertTrue(np.all(mask[40:160, 50:150] == 255))

    def test_glyph_clipped_on_all_four_sides_is_kept(self) -> None:
        image = np.full_like(self.image, 255)
        cv2.line(image, (47, 37), (152, 162), (0, 0, 0), 4)
        cv2.line(image, (152, 37), (47, 162), (0, 0, 0), 4)
        glyph = np.all(image == 0, axis=2)
        for inverted in (False, True):
            with self.subTest(inverted=inverted):
                source = 255 - image if inverted else image
                mask = _mask_from_regions(source.shape, get_inpaint_bboxes(self.box, source))
                self.assertTrue(np.all(mask[glyph] == 255))

    def test_clipped_solid_glyph_is_not_mistaken_for_an_enclosing_border(self) -> None:
        image = np.full_like(self.image, 255)
        image[32:168, 42:158] = 0
        mask = _mask_from_regions(image.shape, get_inpaint_bboxes(self.box, image))
        self.assertTrue(np.all(mask[32:168, 42:158] == 255))

    def test_glyph_extending_past_one_box_edge_is_kept(self) -> None:
        image = np.full_like(self.image, 255)
        image[60:140, 47:57] = 0
        mask = _mask_from_regions(image.shape, get_inpaint_bboxes(self.box, image))
        self.assertTrue(np.all(mask[60:140, 47:57] == 255))

    def test_existing_vertical_margin_is_preserved(self) -> None:
        image = np.full_like(self.image, 255)
        image[33:38, 80:90] = 0
        image[162:167, 80:90] = 0
        mask = _mask_from_regions(image.shape, get_inpaint_bboxes(self.box, image))
        self.assertTrue(np.all(mask[33:38, 80:90] == 255))
        self.assertTrue(np.all(mask[162:167, 80:90] == 255))

    def test_empty_or_outside_crops_are_ignored(self) -> None:
        for box in (None, [], [1, 2, 3], [300, 40, 400, 160], [50, 300, 150, 400]):
            with self.subTest(box=box):
                self.assertEqual(get_inpaint_bboxes(box, self.image), [])


class CTBDSliceMergeTest(unittest.TestCase):
    """Tall pages are detected in overlapping slices and then merged."""

    def process(self, slice_boxes: list, height: int = 6800, **params: float) -> list:
        slicer = ImageSlicer(LOGGER, **params)
        predictions = [np.array(boxes, dtype=np.int64) for boxes in slice_boxes]
        snapshots = [boxes.copy() for boxes in predictions]
        detect = Mock(side_effect=[(boxes, boxes) for boxes in predictions])
        bubbles, text = slicer.process_slices_for_detection(
            np.zeros((height, 1200, 3), np.uint8), detect
        )
        self.assertEqual(detect.call_count, len(slice_boxes))
        np.testing.assert_array_equal(bubbles, text)
        for boxes, snapshot in zip(predictions, snapshots):
            np.testing.assert_array_equal(boxes, snapshot)
        return text.tolist()

    def test_stacked_bubbles_away_from_a_seam_stay_separate(self) -> None:
        boxes = [[177, 117, 423, 585], [180, 966, 520, 1540]]

        self.assertEqual(self.process([boxes, []]), boxes)

    def test_bubbles_inside_the_overlap_but_not_cut_stay_separate(self) -> None:
        # Neither box reaches the edge of the slice it was detected in.
        boxes = [[194, 2663, 525, 3214], [177, 3518, 420, 3967]]
        self.assertEqual(self.process([[boxes[0]], [[177, 638, 420, 1087]]]), boxes)

    def test_a_box_cut_by_the_seam_is_rejoined(self) -> None:
        upper_piece = [180, 2950, 520, 3600]  # cut by the upper slice's bottom
        lower_piece = [180, 680, 520, 1220]

        self.assertEqual(self.process([[upper_piece], [lower_piece]]), [[180, 2950, 520, 4100]])

    def test_near_edge_fragments_rejoin_without_slice_overlap(self) -> None:
        result = self.process(
            [[[180, 3200, 520, 3595]], [[180, 5, 520, 400]]],
            slice_overlap_ratio=0.0,
        )
        self.assertEqual(result, [[180, 3200, 520, 4000]])

    def test_same_slice_bubbles_near_another_slice_start_stay_separate(self) -> None:
        boxes = [[180, 2450, 520, 2883], [180, 2888, 520, 3340]]
        self.assertEqual(self.process([boxes, []]), boxes)

    def test_duplicate_winner_keeps_its_own_slice(self) -> None:
        for upper, lower in (
            ([100, 3000, 300, 3550], [100, 3005, 300, 3600]),  # containment
            ([100, 2750, 300, 3450], [100, 2900, 310, 3600]),  # IoU duplicate
        ):
            with self.subTest(upper=upper):
                lower_local = [lower[0], lower[1] - 2880, lower[2], lower[3] - 2880]
                neighbor = [100, 3610, 300, 4100]
                result = self.process([[upper], [lower_local, [100, 730, 300, 1220]]])
                self.assertEqual(result, [lower, neighbor])

    def test_box_rejoins_across_three_slices_without_absorbing_a_neighbor(self) -> None:
        result = self.process(
            [[[180, 1600, 520, 3600]], [[180, 0, 520, 3600]],
             [[180, 0, 520, 2040], [180, 2060, 520, 4040]]],
            height=10500,
        )
        self.assertEqual(result, [[180, 1600, 520, 7800], [180, 7820, 520, 9800]])

    def test_empty_slice_preserves_later_slice_indices(self) -> None:
        result = self.process(
            [[], [[180, 2500, 520, 3600]], [[180, 680, 520, 1800]]], height=10500
        )
        self.assertEqual(result, [[180, 5380, 520, 7560]])

    def test_final_slice_uses_its_actual_height_for_edge_tolerance(self) -> None:
        result = self.process([[[180, 2500, 520, 3400]], [[180, 12, 520, 920]]])
        self.assertEqual(result, [[180, 2500, 520, 3800]])

    def test_source_order_can_differ_from_predicted_top_edge_order(self) -> None:
        result = self.process([[[100, 2950, 300, 3600]], [[150, 0, 350, 520]]])
        self.assertEqual(result, [[100, 2880, 350, 3600]])

    def test_empty_and_single_box_results(self) -> None:
        self.assertEqual(self.process([[], []]), [])
        self.assertEqual(self.process([[], [[10, 20, 40, 60]]]), [[10, 2900, 40, 2940]])

    def test_unsliced_images_keep_detection_results(self) -> None:
        boxes = [[10, 20, 40, 60], [10, 65, 40, 100]]
        self.assertEqual(self.process([boxes], height=1200), boxes)

    def test_detector_routes_zero_overlap_boxes_through_postprocessing(self) -> None:
        detector = RTDetrV2TextDetector(device='cpu', slice_overlap_ratio=0.0)
        detector._initialize_slicer()
        detector.model = Mock()
        detector.model.run.side_effect = [
            (np.array([[1]]), np.array([[[180, 3200, 520, 3595]]]), np.array([[1.0]])),
            (np.array([[1]]), np.array([[[180, 5, 520, 400]]]), np.array([[1.0]])),
        ]
        mask, blocks = detector.detect(np.full((6800, 1200, 3), 255, np.uint8))

        self.assertEqual([list(block.xyxy) for block in blocks], [[180, 3200, 520, 4000]])
        self.assertFalse(np.any(mask))
        self.assertEqual(detector.model.run.call_count, 2)


if __name__ == '__main__':
    unittest.main()

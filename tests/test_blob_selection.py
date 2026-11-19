import unittest

import numpy as np

from ballontranslator.utils.blob_selection import (
    RULE_ANY,
    RULE_MAJORITY,
    BlobSelection,
    binarize_crop,
)


def two_letter_crop():
    """Two 4x6 dark letters on a white balloon, 6 columns apart."""
    crop = np.full((10, 20, 3), 255, dtype=np.uint8)
    crop[2:8, 2:6] = 0
    crop[2:8, 12:16] = 0
    return crop


def mask_for(shape, *slices):
    mask = np.zeros(shape, dtype=np.uint8)
    for rows, cols in slices:
        mask[rows, cols] = 255
    return mask


class BlobSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        crop = two_letter_crop()
        # The detector only caught the left letter.
        detected = mask_for(crop.shape[:2], (slice(2, 8), slice(2, 6)))
        self.sel = BlobSelection(crop, detected)
        self.left = self.sel.label_at(3, 4)
        self.right = self.sel.label_at(13, 4)

    def test_detector_choice_seeds_the_selection(self) -> None:
        self.assertEqual(self.sel.selected, {self.left})
        np.testing.assert_array_equal(
            self.sel.to_mask() > 0,
            mask_for((10, 20), (slice(2, 8), slice(2, 6))) > 0,
        )

    def test_available_mask_shows_only_unselected_blobs(self) -> None:
        available = self.sel.available_mask() > 0
        self.assertTrue(available[4, 13])
        self.assertFalse(available[4, 3])

    def test_click_adds_and_removes_one_blob(self) -> None:
        self.assertTrue(self.sel.click(13, 4, add=True))
        self.assertEqual(self.sel.selected, {self.left, self.right})
        self.assertTrue(self.sel.click(3, 4, add=False))
        self.assertEqual(self.sel.selected, {self.right})

    def test_click_on_background_or_outside_changes_nothing(self) -> None:
        self.assertFalse(self.sel.click(9, 0, add=True))
        self.assertFalse(self.sel.click(-5, 50, add=True))
        self.assertEqual(self.sel.selected, {self.left})

    def test_paint_touches_each_blob_once_per_gesture(self) -> None:
        self.sel.begin_gesture()
        self.assertTrue(self.sel.paint(13, 4, add=True))
        self.sel.click(13, 4, add=False)
        # Sweeping back over the same blob in the same gesture is ignored.
        self.assertFalse(self.sel.paint(14, 5, add=True))
        self.sel.begin_gesture()
        self.assertTrue(self.sel.paint(14, 5, add=True))

    def test_box_rules(self) -> None:
        # Covers 2 of the right letter's 4 columns: exactly half, so the
        # majority rule leaves it while the any rule takes it.
        self.assertEqual(
            self.sel.blobs_in_box(10, 0, 13, 9, RULE_MAJORITY), set()
        )
        self.assertEqual(
            self.sel.blobs_in_box(10, 0, 13, 9, RULE_ANY), {self.right}
        )
        self.assertEqual(
            self.sel.blobs_in_box(10, 0, 14, 9, RULE_MAJORITY), {self.right}
        )

    def test_box_accepts_corners_in_any_order_and_clamps(self) -> None:
        caught = self.sel.box(30, 30, -4, -4, add=False, rule=RULE_ANY)
        self.assertEqual(caught, 2)
        self.assertEqual(self.sel.selected, set())


class BackgroundFillGuardTest(unittest.TestCase):
    """The rect tool's flat fill paints the balloon, not the edited mask."""

    def setUp(self) -> None:
        crop = two_letter_crop()
        detected = mask_for(crop.shape[:2], (slice(2, 8), slice(2, 6)))
        self.sel = BlobSelection(crop, detected)
        # Balloon interior covering only the left letter.
        self.balloon = mask_for((10, 20), (slice(0, 10), slice(0, 10)))

    def test_unedited_selection_keeps_the_fill(self) -> None:
        self.assertFalse(self.sel.removed_any)
        self.assertFalse(self.sel.added_outside(self.balloon))

    def test_removing_a_detected_blob_is_reported(self) -> None:
        self.sel.click(3, 4, add=False)
        self.assertTrue(self.sel.removed_any)

    def test_adding_a_blob_outside_the_balloon_is_reported(self) -> None:
        self.sel.click(13, 4, add=True)
        self.assertFalse(self.sel.removed_any)
        self.assertTrue(self.sel.added_outside(self.balloon))

    def test_adding_a_blob_inside_the_balloon_keeps_the_fill(self) -> None:
        whole_crop = mask_for((10, 20), (slice(0, 10), slice(0, 20)))
        self.sel.click(13, 4, add=True)
        self.assertFalse(self.sel.added_outside(whole_crop))


class BinarizeCropTest(unittest.TestCase):
    def test_detector_mask_picks_the_text_polarity(self) -> None:
        # Light text on dark art: plain Otsu would make the art white.
        crop = np.full((10, 20, 3), 20, dtype=np.uint8)
        crop[2:8, 2:6] = 240
        detected = mask_for((10, 20), (slice(2, 8), slice(2, 6)))
        binary = binarize_crop(crop, detected)
        self.assertEqual(binary[4, 3], 255)
        self.assertEqual(binary[0, 0], 0)

    def test_rgba_crop_is_accepted(self) -> None:
        crop = np.dstack([
            two_letter_crop(), np.full((10, 20), 255, dtype=np.uint8)
        ])
        detected = mask_for((10, 20), (slice(2, 8), slice(2, 6)))
        self.assertEqual(binarize_crop(crop, detected).shape, (10, 20))


if __name__ == '__main__':
    unittest.main()

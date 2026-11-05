import doctest
import importlib.util
import itertools
import unittest
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

# Load the numerical module without importing the optional CTD inference backend.
TEXTMASK_SPEC = importlib.util.spec_from_file_location(
    "_ctd_textmask_for_tests",
    Path(__file__).resolve().parents[1] / "ballontranslator/modules/textdetector/ctd/textmask.py",
)
textmask = importlib.util.module_from_spec(TEXTMASK_SPEC)
TEXTMASK_SPEC.loader.exec_module(textmask)

REFINEMASK_ANNOTATION = textmask.REFINEMASK_ANNOTATION
REFINEMASK_INPAINT = textmask.REFINEMASK_INPAINT
merge_mask_list = textmask.merge_mask_list


def _reference_merge_mask_list(mask_list, pred_mask: np.ndarray, pred_thresh: int = 30, refine_mode: int = REFINEMASK_INPAINT) -> np.ndarray:
    """The original per-component loop, kept as the oracle for merge_mask_list."""
    mask_list.sort(key=lambda x: x[1])
    if pred_thresh > 0:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
        pred_mask = cv2.erode(pred_mask, element, iterations=1)
        _, pred_mask = cv2.threshold(pred_mask, 60, 255, cv2.THRESH_BINARY)

    def merge_if_improves(mask_merged: np.ndarray, labels: np.ndarray, label_index: int, stat: np.ndarray) -> None:
        x, y, w, h, area = stat
        x1, y1, x2, y2 = x, y, x + w, y + h
        label_local = labels[y1:y2, x1:x2]
        tmp_merged = np.zeros_like(label_local, np.uint8)
        tmp_merged[np.where(label_local == label_index)] = 255
        tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
        xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
        xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
        if xor_merged < xor_origin:
            mask_merged[y1:y2, x1:x2] = tmp_merged

    mask_merged = np.zeros_like(pred_mask)
    for candidate_mask, _ in mask_list:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8, cv2.CV_16U)
        for label_index in range(1, num_labels):
            if stats[label_index][2] * stats[label_index][3] >= 3:
                merge_if_improves(mask_merged, labels, label_index, stats[label_index])

    if refine_mode == REFINEMASK_INPAINT:
        mask_merged = cv2.dilate(mask_merged, np.ones((5, 5), np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(255 - mask_merged, 8, cv2.CV_16U)
    sorted_area = np.sort(stats[:, -1])
    area_thresh = sorted_area[-2] if len(sorted_area) > 1 else sorted_area[-1]
    for label_index in range(num_labels):
        if stats[label_index][4] < area_thresh:
            merge_if_improves(mask_merged, labels, label_index, stats[label_index])
    return mask_merged


def _random_mask(rng: np.random.Generator, shape, gray: bool = False) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    h, w = shape
    for _ in range(int(rng.integers(0, 40))):
        center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            axes = (int(rng.integers(1, 20)), int(rng.integers(1, 20)))
            cv2.ellipse(mask, center, axes, int(rng.integers(0, 180)), 0, 360, 255, -1)
        else:
            end = (center[0] + int(rng.integers(-50, 50)), center[1] + int(rng.integers(-50, 50)))
            cv2.line(mask, center, end, 255, int(rng.integers(1, 5)))
    # Speckle produces many tiny components, including ones below the w * h < 3 cutoff.
    mask[rng.random(shape) < 0.02] = 255
    return cv2.GaussianBlur(mask, (7, 7), 0) if gray else mask


class MergeMaskListEquivalenceTest(unittest.TestCase):
    def test_doctest(self) -> None:
        self.assertEqual(doctest.testmod(textmask).failed, 0)

    def assert_same_as_reference(self, mask_list, pred_mask: np.ndarray, pred_thresh: int, refine_mode: int) -> None:
        expected = _reference_merge_mask_list(deepcopy(mask_list), pred_mask.copy(), pred_thresh, refine_mode)
        actual = merge_mask_list(deepcopy(mask_list), pred_mask.copy(), pred_thresh=pred_thresh, refine_mode=refine_mode)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual, expected)

    def test_all_binary_2x3_masks(self) -> None:
        # Every (candidate, prediction) pair of binary 2x3 masks, both prediction paths and refine modes.
        masks = [np.array(bits, np.uint8).reshape(2, 3) * 255 for bits in itertools.product((0, 1), repeat=6)]
        for candidate, pred in itertools.product(masks, masks):
            for pred_thresh, refine_mode in itertools.product((0, 30), (REFINEMASK_INPAINT, REFINEMASK_ANNOTATION)):
                self.assert_same_as_reference([[candidate, 0]], pred, pred_thresh, refine_mode)

    def test_random_multi_candidate_masks(self) -> None:
        rng = np.random.default_rng(1370)
        for _ in range(150):
            shape = (int(rng.integers(8, 200)), int(rng.integers(8, 200)))
            pred_mask = _random_mask(rng, shape, gray=bool(rng.random() < 0.6))
            mask_list = [[_random_mask(rng, shape), int(rng.integers(0, 10**6))] for _ in range(int(rng.integers(0, 6)))]
            for pred_thresh, refine_mode in itertools.product((0, 30), (REFINEMASK_INPAINT, REFINEMASK_ANNOTATION)):
                self.assert_same_as_reference(mask_list, pred_mask, pred_thresh, refine_mode)

    def test_grayscale_ties_are_rejected(self) -> None:
        candidate = np.full((1, 4), 255, np.uint8)
        for values, accepted in (([127, 128, 127, 128], False), ([127, 128, 128, 128], True)):
            pred = np.array([values], np.uint8)
            actual = merge_mask_list([[candidate, 0]], pred, pred_thresh=0, refine_mode=REFINEMASK_ANNOTATION)
            np.testing.assert_array_equal(actual, candidate if accepted else np.zeros_like(candidate))

    def test_grayscale_decisions_with_few_and_many_components(self) -> None:
        for count in (1, 32, 33, 900):
            with self.subTest(components=count):
                candidate = np.zeros((3, count * 5), np.uint8)
                candidate[1].reshape(count, 5)[:, :4] = 255
                pred = np.zeros_like(candidate)
                pred[1].reshape(count, 5)[:, :4] = [127, 128, 127, 128]
                pred[1].reshape(count, 5)[1::2, 2] = 128
                expected = np.zeros_like(candidate)
                expected[1].reshape(count, 5)[1::2, :4] = 255
                actual = merge_mask_list(
                    [[candidate, 1], [candidate, 0]], pred,
                    pred_thresh=0, refine_mode=REFINEMASK_ANNOTATION,
                )
                np.testing.assert_array_equal(actual, expected)

    def test_overlapping_candidates_keep_sorted_pass_order(self) -> None:
        first = np.array([[0, 255, 255, 255, 0]], np.uint8)
        second = np.array([[0, 0, 255, 255, 255]], np.uint8)
        pred = np.array([[0, 0, 255, 255, 0]], np.uint8)
        actual = merge_mask_list([[second, 1], [first, 0]], pred, pred_thresh=0, refine_mode=REFINEMASK_ANNOTATION)
        np.testing.assert_array_equal(actual, first)

    def test_many_disjoint_components(self) -> None:
        candidate = np.zeros((180, 180), np.uint8)
        candidate.reshape(30, 6, 30, 6)[:, 1:4, :, 1:4] = 255
        pred = cv2.dilate(candidate, np.ones((3, 3), np.uint8))
        for pred_thresh, refine_mode in itertools.product((0, 30), (REFINEMASK_INPAINT, REFINEMASK_ANNOTATION)):
            self.assert_same_as_reference([[candidate, 0], [candidate, 1]], pred, pred_thresh, refine_mode)


if __name__ == '__main__':
    unittest.main()

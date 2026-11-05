"""Editable blob selection for the rect tool's preview mask.

Deliberately standalone: this module imports only numpy and cv2. It knows
nothing about Qt, the canvas, the project, or the rest of BallonsTranslator,
so it can be unit tested and exercised from a plain script.

The idea: after the rect tool extracts a preview mask, we binarize the same
crop, split it into connected components ("blobs"), and record which blobs
the detector chose. Editing the selection is then a set operation on blob
ids rather than any further image processing.

Coordinates are always in CROP space -- pixels relative to the top-left of
the rectangle the user drew, not the page. The caller is responsible for
translating.
"""

from typing import Iterable, Optional, Set, Tuple

import cv2
import numpy as np

# Box rules: which blobs a marquee catches.
RULE_ANY = 'any'            # a single pixel inside the box is enough
RULE_MAJORITY = 'majority'  # more than half the blob's pixels must be inside


def binarize_crop(crop_rgb: np.ndarray, reference_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Otsu binarization of a crop, with text coming out white.

    The extractors in textblock_mask don't return their internal
    binarization, so we recompute one here rather than modifying them.

    Polarity is ambiguous on its own (dark text on light balloons, light
    text on dark art), so when the detector's mask is available we use it
    to decide: whichever polarity overlaps the detected text more is the
    one where text ended up white.
    """
    if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
        crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    if reference_mask is not None and reference_mask.any():
        normal = np.count_nonzero(cv2.bitwise_and(thresh, reference_mask))
        flipped = np.count_nonzero(cv2.bitwise_and(255 - thresh, reference_mask))
        if flipped > normal:
            thresh = 255 - thresh

    return thresh


class BlobSelection:
    """A label map plus the set of blob ids currently selected.

    Two dark letters on a white balloon, where the detector only caught
    the left one:

    >>> crop = np.full((10, 20, 3), 255, dtype=np.uint8)
    >>> crop[2:8, 2:6] = 0
    >>> crop[2:8, 12:16] = 0
    >>> detected = np.zeros((10, 20), dtype=np.uint8)
    >>> detected[2:8, 2:6] = 255
    >>> sel = BlobSelection(crop, detected)
    >>> len(sel.selected)
    1
    >>> sel.begin_gesture()
    >>> sel.click(13, 4, add=True)
    True
    >>> int(np.count_nonzero(sel.to_mask()))
    48
    """

    def __init__(
        self,
        crop_rgb: np.ndarray,
        detector_mask: np.ndarray,
        coverage: float = 0.5,
    ):
        """Build the label map and seed the selection from the detector.

        Args:
            crop_rgb: the crop inside the user's rectangle, RGB or RGBA.
            detector_mask: the mask canny_flood / connected_canny_flood
                returned for that crop, undilated. Used both to pick the
                binarization polarity and to seed the initial selection.
            coverage: a blob counts as chosen when the detector's mask
                covers more than this fraction of it.
        """
        self.candidates = binarize_crop(crop_rgb, detector_mask)

        n_labels, labels, stats, _cent = cv2.connectedComponentsWithStats(
            self.candidates, connectivity=4, ltype=cv2.CV_32S
        )
        self.labels = labels
        self.n_labels = n_labels
        self.areas = stats[:, cv2.CC_STAT_AREA]

        covered = np.bincount(labels[detector_mask > 0].ravel(), minlength=n_labels)
        selected = set()
        for lab in range(1, n_labels):
            if self.areas[lab] > 0 and covered[lab] / self.areas[lab] > coverage:
                selected.add(lab)

        self.selected: Set[int] = selected
        self.initial: Set[int] = set(selected)
        self._painted: Set[int] = set()

    # ---------------------------------------------------------------- info

    @property
    def shape(self) -> Tuple[int, int]:
        return self.labels.shape

    @property
    def removed_any(self) -> bool:
        """True once the user has deselected a blob the detector chose.

        The flat-fill path in DrawingPanel.inpaintRect paints the whole
        balloon rather than the mask, so it cannot honour a deselection.
        Additions are checked separately by :meth:`added_outside`.

        >>> crop = np.full((10, 20, 3), 255, dtype=np.uint8)
        >>> crop[2:8, 2:6] = 0
        >>> detected = np.zeros((10, 20), dtype=np.uint8)
        >>> detected[2:8, 2:6] = 255
        >>> sel = BlobSelection(crop, detected)
        >>> sel.removed_any
        False
        >>> sel.click(3, 4, add=False)
        True
        >>> sel.removed_any
        True
        """
        return bool(self.initial - self.selected)

    def added_outside(self, region: np.ndarray) -> bool:
        """True when a blob the user added reaches outside ``region``.

        ``region`` is a crop-sized mask, such as the balloon interior the
        flat-fill path paints. Blobs the detector chose are not counted, so
        an unedited selection never reports True.

        >>> crop = np.full((10, 20, 3), 255, dtype=np.uint8)
        >>> crop[2:8, 2:6] = 0
        >>> crop[2:8, 12:16] = 0
        >>> detected = np.zeros((10, 20), dtype=np.uint8)
        >>> detected[2:8, 2:6] = 255
        >>> balloon = np.zeros((10, 20), dtype=np.uint8)
        >>> balloon[:, :10] = 255
        >>> sel = BlobSelection(crop, detected)
        >>> sel.added_outside(balloon)
        False
        >>> sel.click(13, 4, add=True)
        True
        >>> sel.added_outside(balloon)
        True
        """
        added = self.selected - self.initial
        if not added:
            return False
        outside = np.isin(self.labels, list(added)) & (region == 0)
        return bool(outside.any())

    def label_at(self, x: int, y: int) -> int:
        """Blob id under a point, or 0 for background / out of bounds."""
        h, w = self.labels.shape
        if 0 <= x < w and 0 <= y < h:
            return int(self.labels[y, x])
        return 0

    # ------------------------------------------------------------ gestures

    def begin_gesture(self) -> None:
        """Start a new gesture, so painting may touch every blob again."""
        self._painted = set()

    # ------------------------------------------------------------- editing

    def click(self, x: int, y: int, add: bool) -> bool:
        """Add or remove the single blob under a point.

        Returns True when the selection actually changed.
        """
        lab = self.label_at(x, y)
        if lab == 0:
            return False
        return self._apply({lab}, add)

    def paint(self, x: int, y: int, add: bool) -> bool:
        """Free-hand painting: like click, but ignores blobs already
        touched during this gesture so a sweep doesn't thrash."""
        lab = self.label_at(x, y)
        if lab == 0 or lab in self._painted:
            return False
        self._painted.add(lab)
        return self._apply({lab}, add)

    def box(self, x0: int, y0: int, x1: int, y1: int, add: bool,
            rule: str = RULE_MAJORITY) -> int:
        """Marquee selection. Coordinates are inclusive corners, any order.

        Returns how many blobs the box caught.
        """
        hit = self.blobs_in_box(x0, y0, x1, y1, rule)
        if hit:
            self._apply(hit, add)
        return len(hit)

    def blobs_in_box(self, x0: int, y0: int, x1: int, y1: int,
                     rule: str = RULE_MAJORITY) -> Set[int]:
        """Which blob ids a box would catch, without applying anything."""
        h, w = self.labels.shape
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            return set()

        sub = self.labels[y0:y1 + 1, x0:x1 + 1]
        ids, counts = np.unique(sub, return_counts=True)

        found = set()
        for lab, count in zip(ids, counts):
            lab = int(lab)
            if lab == 0:
                continue
            if rule == RULE_ANY:
                found.add(lab)
            else:
                total = self.areas[lab]
                if total > 0 and count / total > 0.5:
                    found.add(lab)
        return found

    def _apply(self, blobs: Iterable[int], add: bool) -> bool:
        before = len(self.selected)
        if add:
            self.selected |= set(blobs)
        else:
            self.selected -= set(blobs)
        return len(self.selected) != before

    # -------------------------------------------------------------- output

    def to_mask(self) -> np.ndarray:
        """The current selection as a uint8 mask, 0 or 255.

        This is what replaces the detector's output as the undilated
        source the preview and the inpainter are derived from.
        """
        if not self.selected:
            return np.zeros(self.labels.shape, dtype=np.uint8)
        return np.isin(self.labels, list(self.selected)).astype(np.uint8) * 255

    def available_mask(self) -> np.ndarray:
        """Blobs that exist but are NOT selected, as a uint8 mask.

        Drawn in a second colour to show what is clickable. Pixels tinted
        by neither colour never survived binarization and are unreachable.
        """
        selected = self.to_mask()
        return cv2.bitwise_and(self.candidates, cv2.bitwise_not(selected))

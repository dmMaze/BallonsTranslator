"""Optional CTD detection-box geometry; source-line geometry stays unchanged."""

import re
from logging import Logger
from typing import List, Tuple

import numpy as np

from ballontranslator.utils.imgproc_utils import rotate_polygons, xywh2xyxypoly
from ballontranslator.utils.textblock import TextBlock


def parse_ctd_padding(value: object) -> int:
    """Validate live CTD padding without loading a model.

    >>> parse_ctd_padding("4")
    4
    >>> parse_ctd_padding(0)
    0
    """
    if isinstance(value, bool) or not (
        isinstance(value, int)
        or isinstance(value, str) and re.fullmatch(r'[0-9]{1,2}', value.strip())
    ):
        raise ValueError('Detect box padding (px) must be a whole number from 0 to 64; 0 disables padding.')
    padding = int(value)
    if not 0 <= padding <= 64:
        raise ValueError('Detect box padding (px) must be a whole number from 0 to 64; 0 disables padding.')
    return padding


def _intersections(box: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Return positive-area AABB intersections; shared edges are permitted."""
    return np.all(np.minimum(box[2:], neighbors[:, 2:]) >
                  np.maximum(box[:2], neighbors[:, :2]), axis=1)

def _pad_rotated_block(block: TextBlock, padding: int, width: int, height: int,
                       neighbors: np.ndarray, neighbor_ids: np.ndarray, label: str, logger: Logger) -> None:
    """Expand local geometry without moving its source-line rotation center.

    Native sync truncates corners. Outward, symmetric integer bounds keep
    OCR coverage and TextBlock.center(), which source min_rect() depends on.
    """
    center = block.center()
    rect = np.asarray(block.bounding_rect(), dtype=float)
    if not np.isfinite(rect).all() or rect[2] <= 0 or rect[3] <= 0:
        logger.warning(f'{label}: invalid rotated rectangle; retained original geometry.')
        return
    local_min, local_max = rect[:2], rect[:2] + rect[2:]
    # Native bounding_rect uses truncated source vertices. Include their
    # floating extents too, so reduction at a page edge cannot cut lettering.
    if block.lines:
        local_lines = rotate_polygons(center, block.lines_array().reshape(-1, 8),
                                      block.angle, to_int=False).reshape(-1, 2)
        local_min = np.minimum(local_min, local_lines.min(axis=0))
        local_max = np.maximum(local_max, local_lines.max(axis=0))
    half_size = np.maximum(center - local_min, local_max - center)
    # Symmetric enclosure compensates native rounding without changing the
    # source rotation center. At boundaries reduce only the extra margin.
    reason = 'page edge'
    for margin in range(padding, 0, -1):
        half = half_size + margin
        padded_rect = [*(center - half), *(2 * half)]
        polygon = rotate_polygons(center, xywh2xyxypoly(np.array([padded_rect])),
                                  -block.angle, to_int=False).reshape(-1, 2)
        lower = np.floor(polygon.min(axis=0)).astype(int)
        upper = np.ceil(polygon.max(axis=0)).astype(int)
        # Integer CTD xyxy gives an integer doubled center, including .5
        # centers. Balance outward rounding rather than shifting that center.
        doubled_center = np.rint(center * 2).astype(int)
        lower = np.minimum(lower, doubled_center - upper)
        upper = doubled_center - lower
        if (lower < 0).any() or (upper > [width, height]).any():
            continue
        collisions = _intersections(np.r_[lower, upper], neighbors)
        if collisions.any():
            reason = f'original neighbor blocks {neighbor_ids[collisions].tolist()}'
            continue
        block._bounding_rect = [float(value) for value in padded_rect]
        block.sync_xyxy_from_bounding_rect()
        block.xyxy = [*lower.tolist(), *upper.tolist()]
        if margin < padding:
            logger.warning(
                f'{label}: {reason} reduced rotated box padding from {padding} to {margin}px; '
                'source position and lettering were retained.'
            )
        return
    logger.warning(
        f'{label}: rotated box has no room for padding ({reason}); '
        'retained original geometry instead of moving or shrinking lettering.'
    )


def pad_ctd_boxes(blocks: List[TextBlock], padding: int,
                  image_shape: Tuple[int, int], logger: Logger) -> None:
    """Expand new horizontal boxes against immutable original neighbor bounds.

    >>> import logging
    >>> block = TextBlock(xyxy=[20, 20, 60, 50])
    >>> pad_ctd_boxes([block], 4, (80, 100), logging.getLogger('ctd.padding'))
    >>> block.xyxy
    [16, 16, 64, 54]
    >>> block.bounding_rect()
    [16, 16, 48, 38]
    """
    if padding == 0:
        return
    height, width = image_shape
    # Immutable per-run ownership bounds prevent later candidates depending
    # on input order or treating another block's empty margin as lettering.
    originals = np.array([block.xyxy for block in blocks], dtype=float).reshape(-1, 4)
    valid = np.isfinite(originals).all(axis=1) & (originals[:, 2:] > originals[:, :2]).all(axis=1)
    for index, block in enumerate(blocks):
        label = f'ctd block #{index} xyxy={originals[index].tolist()}'
        if block.src_is_vertical:
            logger.warning(f'{label}: vertical source classification retained; '
                                'check source orientation manually if horizontal padding was expected.')
            continue
        x1, y1, x2, y2 = block.xyxy
        # Leave malformed/degenerate detections alone instead of inventing
        # a usable region. Real CTD returns integer original-image boxes.
        if not valid[index]:
            logger.warning(f'{label}: invalid source box retained; inspect detected geometry manually.')
            continue
        neighbor_ids = np.flatnonzero(valid & (np.arange(len(blocks)) != index))
        neighbors = originals[neighbor_ids]
        overlaps = _intersections(originals[index], neighbors)
        if overlaps.any():
            logger.warning(f'{label}: pre-existing overlap with original neighbor blocks '
                                f'{neighbor_ids[overlaps].tolist()}; padding skipped, source boxes retained.')
            continue
        if block.angle != 0:
            _pad_rotated_block(block, padding, width, height, neighbors, neighbor_ids, label, logger)
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            logger.warning(f'{label}: source box extends outside page; padding skipped '
                                'instead of moving or shrinking lettering.')
            continue
        reason = 'page edge'
        for margin in range(padding, 0, -1):
            left = max(0, min(width, int(x1) - margin))
            top = max(0, min(height, int(y1) - margin))
            right = max(0, min(width, int(x2) + margin))
            bottom = max(0, min(height, int(y2) + margin))
            if right <= left or bottom <= top:
                continue
            collisions = _intersections(np.array([left, top, right, bottom]), neighbors)
            if collisions.any():
                reason = f'original neighbor blocks {neighbor_ids[collisions].tolist()}'
                continue
            block._bounding_rect = [left, top, right - left, bottom - top]
            block.sync_xyxy_from_bounding_rect()
            if margin < padding:
                logger.warning(f'{label}: {reason} reduced box padding from {padding} to {margin}px; '
                                    'source boxes retained.')
            elif [left, top, right, bottom] != [x1-margin, y1-margin, x2+margin, y2+margin]:
                logger.warning(f'{label}: page edge clipped extra padding; source box retained.')
            break
        else:
            logger.warning(f'{label}: no room for padding ({reason}); source box retained.')
        # min_rect() is derived from source lines, not a cached edit box.
        # Preserve it, the line polygons, font estimate and upstream mask.

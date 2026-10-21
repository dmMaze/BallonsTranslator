"""Fit a vertical block's font size to its box instead of only shrinking it.

The pipeline's automatic layout can only shrink a block (its ratio is clamped to
1.0), so a short translation in a tall bubble stays small. This module measures
the settled layout and grows the font until the content fills the box, bounded
by ``max_ratio`` and verified by re-laying out.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_MAX_RATIO = 1.6
DEFAULT_TOLERANCE = 0.02
BISECTION_STEPS = 6

# Growing stops once the content already uses this much of the column
# height: vertical cells clamp to that height, so 'full' cannot be told
# apart from 'overflowing' by measuring the content alone.
_FULL_FRACTION = 0.9


def _content_height(layout) -> float:
    """Return the settled content height from the layout's own cell offsets.

    ``shrink_height`` saturates at the column height once the content overflows,
    so the cell offsets are what tells a full box from an overflowing one.
    """
    blocks = getattr(layout, 'y_offset_lst', None) or []
    top = None
    bottom = None
    for offsets in blocks:
        for pair in offsets:
            line_top, line_bottom = float(pair[0]), float(pair[1])
            top = line_top if top is None else min(top, line_top)
            bottom = line_bottom if bottom is None else max(bottom, line_bottom)
    if top is None or bottom is None or bottom <= top:
        return 0.0
    return bottom - top


def _settled_fits(item, tolerance: float) -> bool:
    """Return True when the settled layout stays inside the text box."""
    layout = item.layout
    available = getattr(layout, 'available_height', 0.0)
    content = _content_height(layout)
    if available <= 0 or content <= 0:
        return False
    if content > available * (1.0 + tolerance):
        return False
    # Columns flow leftwards; a negative left edge means one column too many.
    return getattr(layout, 'layout_left', 0.0) >= -tolerance


def content_fits(item, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Public check used by callers that only need the verdict.

    >>> callable(content_fits)
    True
    """
    return _settled_fits(item, tolerance)


def fit_vertical_font_size(
    item,
    max_ratio: float = DEFAULT_MAX_RATIO,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Grow a vertical block's font size until it fills its box.

    Returns the applied ratio (1.0 when nothing changed). The block is left
    exactly as it was unless a larger size verifiably still fits, so a block
    that already overflows is never touched.

    >>> callable(fit_vertical_font_size)
    True
    """
    block = getattr(item, 'blk', None)
    if block is None or not getattr(block, 'vertical', False):
        return 1.0
    base_size = float(item.fontformat.font_size)
    if base_size <= 0:
        return 1.0
    if not _settled_fits(item, tolerance):
        # Already overflowing: shrinking is the existing layout's job.
        return 1.0
    available = getattr(item.layout, 'available_height', 0.0)
    if available > 0 and _content_height(item.layout) >= available * _FULL_FRACTION:
        return 1.0

    def apply(ratio: float) -> None:
        size = base_size * ratio
        # setFontSize only restyles the document, so the block's own format has
        # to carry the value as well (that is what the panel and saves read).
        item.fontformat.font_size = size
        item.setFontSize(size)

    best = 1.0
    low, high = 1.0, max(max_ratio, 1.0)
    for _ in range(BISECTION_STEPS):
        if high - low <= 0.01:
            break
        mid = (low + high) / 2.0
        apply(mid)
        if _settled_fits(item, tolerance):
            best = mid
            low = mid
        else:
            high = mid
    apply(best)
    return best

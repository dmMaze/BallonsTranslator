"""Raster profiles and measured fitting for the existing Qt text layouts."""

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from typing import Optional, TYPE_CHECKING
import unicodedata

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, QTextBoundaryFinder
from qtpy.QtGui import QImage, QPainter, QTransform

from ballontranslator.utils.bubble_shape import BubbleShape, bubble_path
from ballontranslator.utils.logger import logger as LOGGER
from .rendering.glyph import glyph_geometry
from .rendering.indexing import _grapheme_ranges, _utf16_slice

if TYPE_CHECKING:
    from .item import TextBlkItem
    from .layout import SceneTextLayout


class BubbleProfile:
    """One bounded raster per path, retained only by the live document layout.

    Resizes and font searches reuse normalized scanline intervals. Taking the
    intersection across an ink band excludes tails, holes and narrow necks.

    >>> callable(BubbleProfile.band)
    True
    """

    def __init__(self, shape: BubbleShape) -> None:
        self.shape = shape
        path = bubble_path(shape.nodes)
        # The raster is normalized and never exceeds 512 squared pixels,
        # independent of page resolution. Leave one sample of safety at edges.
        side = 512
        image = QImage(side, side, QImage.Format.Format_Grayscale8)
        image.fill(0)
        painter = QPainter(image)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.white)
        painter.scale(side, side)
        painter.drawPath(path)
        painter.end()
        bits = image.constBits()
        if hasattr(bits, 'setsize'):
            bits.setsize(image.sizeInBytes())
        mask = np.frombuffer(bits, np.uint8).reshape(side, image.bytesPerLine())[:, :side] != 0
        self.intervals = []
        for raster in (mask, mask.T):
            # Use the widest continuous span, even when a tail or partition
            # puts the frame center outside the usable part of the bubble.
            counts = np.count_nonzero(raster, axis=1)
            left = raster.argmax(axis=1)
            right = side - raster[:, ::-1].argmax(axis=1)
            split = np.flatnonzero((counts > 0) & (right - left > counts))
            left += 1
            right -= 1
            left[counts == 0], right[counts == 0] = side, 0
            if len(split):
                # Rectangles/ellipses need no run extraction. Only concave
                # scanlines can contain multiple candidate spans.
                rows, edges = np.nonzero(np.diff(raster[split], axis=1, prepend=False, append=False))
                rows, starts, ends = rows[::2], edges[::2], edges[1::2]
                widths = ends - starts
                widest = np.zeros(len(split), np.intp)
                np.maximum.at(widest, rows, widths)
                selected = widths == widest[rows]
                left[split[rows[selected]]] = starts[selected] + 1
                right[split[rows[selected]]] = ends[selected] - 1
            self.intervals.append((left / side, right / side))

    def band(self, start: float, end: float, vertical: bool = False) -> tuple[float, float]:
        if start < 0 or end > 1 or start >= end:
            return .5, .5
        left, right = self.intervals[int(vertical)]
        first = max(0, math.floor(start * len(left)))
        last = min(len(left), max(first + 1, math.ceil(end * len(left))))
        return float(left[first:last].max()), float(right[first:last].min())


@dataclass(frozen=True)
class FitResult:
    font_size: float
    overflow: bool
    below_readability_floor: bool


@dataclass(frozen=True)
class FlowUnit:
    end: int
    advance: float
    trailing: float
    breakable: bool
    mandatory: bool
    penalty: float


def _flow_break_penalty(before: str, after: str) -> float:
    """Prefer sentence pauses and keep short English function words attached.

    >>> _flow_break_penalty('Stop! ', 'Now') < _flow_break_penalty('the ', 'book')
    True
    """
    before, after = before.rstrip(), after.lstrip()
    if before.endswith(('.', '!', '?', '…', '‼', '⁇', '⁈', '⁉', '。', '！', '？')):
        return 0.
    if before.endswith((',', ';', ':', '—', '–', '，', '、')):
        return 20.
    if re.match(r'[A-Za-z]*', after)[0].lower() in {
        'and', 'but', 'or', 'so', 'because', 'although', 'while', 'then',
    }:
        return 40.
    # These words have at most four letters; one extra rejects longer tokens
    # without backtracking through a long mixed-script prefix.
    if re.search(r'[A-Za-z]*$', before[-5:])[0].lower() in {
        'a', 'an', 'the', 'to', 'of', 'for', 'in', 'on', 'at', 'with', 'from',
    }:
        return 300.
    return 100.


def measure_flow_units(layout: 'SceneTextLayout', vertical: bool) -> list[FlowUnit]:
    """Reuse Qt's first layout pass; never reshape text in a second model."""
    # shortcut: Ruby and bidi keep Qt wrapping; balance them when their shaped
    # base/reading units and visual-order spans can be measured without loss.
    if any(layout._ruby_metrics):
        return []
    units = []
    block = layout.document().firstBlock()
    while block.isValid():
        text = block.text()
        if any(unicodedata.bidirectional(char) in ('R', 'AL', 'AN') for char in text):
            return []
        finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Line, text)
        legal = set()
        while (end := finder.toNextBoundary()) >= 0:
            legal.add(end)
        ranges = _grapheme_ranges(text)
        range_index = 0
        text_layout = block.layout()
        first_unit = len(units)
        for number in range(text_layout.lineCount()):
            line = text_layout.lineAt(number)
            end = line.textStart() + line.textLength()
            if line.textLength() == 0:
                units.append(FlowUnit(block.position() + end, 0., 0., True, True, 0.))
                continue
            if vertical:
                offsets = layout.line_spaces_lst[block.blockNumber()][number][2]
                cfmt = layout.get_char_fontfmt(block.blockNumber(), line.textStart())
                record = layout._line_record(block, number)
                trailing = (max(0., cfmt.tbr.height() * (cfmt.letter_spacing - 1.))
                            if cfmt is not None and 'text_combine_height' not in record else 0.)
                pieces = [(line.textStart(), end, max(0., offsets[-1] - offsets[0]), trailing)]
            else:
                pieces = []
                while range_index < len(ranges) and ranges[range_index][0] < end:
                    start, stop = ranges[range_index]
                    range_index += 1
                    advance = abs(layout._cursor_x(line, stop) - layout._cursor_x(line, start))
                    pieces.append((start, stop, advance, 0.))
            for start, stop, advance, trailing in pieces:
                content = _utf16_slice(text, start, stop - start)
                if not vertical and content.isspace():
                    if content == ' ':
                        cfmt = layout.get_char_fontfmt(block.blockNumber(), start)
                        if cfmt is not None:
                            # Qt clips hanging word separators at soft wraps;
                            # balancing may put that separator inside another line.
                            advance = max(0., float(cfmt.font_metrics.horizontalAdvance(' ')))
                    trailing = advance
                final = stop >= block.length() - 1 or '\u2028' in content
                breakable = stop in legal or final
                # English opportunities follow spaces; score the surrounding phrase,
                # not the final whitespace grapheme of the measured unit.
                penalty = (_flow_break_penalty(_utf16_slice(text, 0, stop),
                                               _utf16_slice(text, stop, block.length() - 1 - stop))
                           if breakable else 0.)
                unit = FlowUnit(block.position() + stop, advance, trailing,
                                breakable, final, penalty)
                if units and not units[-1].breakable and not units[-1].mandatory:
                    previous = units.pop()
                    unit = FlowUnit(unit.end, previous.advance + unit.advance, unit.trailing,
                                    unit.breakable, unit.mandatory, unit.penalty)
                units.append(unit)
                if len(units) > 1024:
                    return []
        if len(units) == first_unit:
            units.append(FlowUnit(block.position(), 0., 0., True, True, 0.))
        block = block.next()
    return units


def balanced_flow_breaks(units: list[FlowUnit], widths: list[float]) -> Optional[list[int]]:
    """Choose legal breaks that fill every planned line, preserving paragraphs.

    >>> units = [FlowUnit(i + 1, 10., 0., True, i == 5, 0.) for i in range(6)]
    >>> balanced_flow_breaks(units, [40., 40.])
    [3, 6]
    """
    size, count = len(units), len(widths)
    # Bound work for unusually long editing documents; Qt remains the fallback.
    if not count <= size <= 1024 or min(widths) < 1.:
        return None
    # Even removing every trailing gap cannot make this plan large enough.
    if sum(unit.advance - unit.trailing for unit in units) > sum(widths) + .1 * count:
        return None
    costs = np.full(size + 1, np.inf)
    costs[0] = 0.
    previous = np.full((count, size + 1), -1, np.int32)
    for row, width in enumerate(widths):
        following = np.full(size + 1, np.inf)
        last = size - (count - row - 1)
        for start in np.flatnonzero(np.isfinite(costs)):
            advance = 0.
            for end in range(start + 1, last + 1):
                unit = units[end - 1]
                advance += unit.advance
                visible = advance - unit.trailing
                if visible > width + .1:
                    break
                if unit.breakable:
                    cost = costs[start] + ((width - visible) / width) ** 2 * 1000. + unit.penalty
                    if cost < following[end]:
                        following[end] = cost
                        previous[row, end] = start
                if unit.mandatory:
                    break
        costs = following
    if not np.isfinite(costs[size]):
        return None
    ends = []
    end = size
    for row in range(count - 1, -1, -1):
        ends.append(units[end - 1].end)
        end = int(previous[row, end])
    return ends[::-1]


def _ink_fits(item: 'TextBlkItem', rect: QRectF) -> bool:
    layout = item.layout
    if layout.flow_overflow or layout.shrink_width > rect.width() + .1:
        return False
    if not item.fontformat.vertical and layout.shrink_height > rect.height():
        return False
    padding = layout.effectPadding()
    # Font metrics and raw glyph paths can disagree by a fractional pixel at
    # the baseline; this must not reject every size in an otherwise roomy box.
    frame = QRectF(padding, padding, rect.width(), rect.height()).adjusted(-.5, -.5, .5, .5)
    block = item.document().firstBlock()
    while block.isValid():
        text_layout = block.layout()
        finder = None
        if not item.fontformat.vertical and layout.flow_profile is None and text_layout.lineCount() > 1:
            finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Line, block.text())
        for index in range(text_layout.lineCount()):
            if item.fontformat.vertical:
                placement = layout.vertical_line_placement(block, index)
                if placement is None:
                    continue
                line, offset, orientation = placement
            else:
                line, offset, orientation = text_layout.lineAt(index), QPointF(), QTransform()
                if finder is not None:
                    # Qt's editable fallback can split a long word anywhere. That
                    # must not let automatic sizing claim a larger font fits.
                    finder.setPosition(line.textStart() + line.textLength())
                    if not finder.isAtBoundary():
                        return False
            ink = glyph_geometry(line, line.textStart(), line.textLength(), offset, orientation, 0.).bounds
            if ink.isEmpty():
                continue
            if not frame.contains(ink):
                return False
            if layout.flow_profile is not None:
                local = ink.translated(-padding, -padding)
                if item.fontformat.vertical:
                    low, high = layout.flow_profile.band(local.left() / rect.width(), local.right() / rect.width(), vertical=True)
                    first, last, extent = local.top(), local.bottom(), rect.height()
                else:
                    low, high = layout.flow_profile.band(local.top() / rect.height(), local.bottom() / rect.height())
                    first, last, extent = local.left(), local.right(), rect.width()
                if first < low * extent - .1 or last > high * extent + .1:
                    return False
        block = block.next()
    annotations = layout.annotation_ink_bounds()
    return annotations.isEmpty() or frame.contains(annotations)


def fit_text_in_shape(item: 'TextBlkItem', *, auto_size: bool, use_bubble_frame: bool = False,
                      bounding_rect: Optional[QRectF] = None) -> FitResult:
    """Fit or lay out text in its flow area without rewriting text.

    Automatic sizing probes detached Qt layouts. Coarse descending probes avoid
    assuming monotonic fits in tapered bubbles, then refine the final interval.

    >>> FitResult(8, False, True).below_readability_floor
    True
    """
    from .item import TextBlkItem

    block = item.blk
    rect = QRectF(*block.bubble_shape.frame) if use_bubble_frame and block.bubble_layout else (
        QRectF(bounding_rect) if bounding_rect is not None else item.absBoundingRect(qrect=True)
    )
    original_size = item.layout.max_font_size(to_px=True)
    if auto_size:
        block = deepcopy(block)
        block.rich_text = item.toHtml()
        block.translation = item.toPlainText()
        block._bounding_rect = list(rect.getRect())
        scratch = TextBlkItem(block, item.idx)
        scratch.repainting = True
        scratch.repaint_on_changed = False
        current_size = max(scratch.layout.max_font_size(to_px=True), .5)

        def measure(size: float) -> bool:
            nonlocal current_size
            if abs(size - current_size) > .001:
                scratch.setRelFontSize(size / current_size)
                current_size = max(scratch.layout.max_font_size(to_px=True), .5)
            if scratch.absBoundingRect(qrect=True) != rect:
                scratch.setRect(rect, repaint=False)
            return _ink_fits(scratch, rect)

        try:
            maximum = max(.5, rect.height() if block.vertical else rect.width())
            minimum = min(9., maximum)
            chosen = minimum
            previous = maximum
            # Geometric probes also cover useful sizes on high-resolution pages.
            for size in np.geomspace(maximum, minimum, 1 if maximum == minimum else 9):
                if measure(float(size)):
                    chosen = float(size)
                    low, high = chosen, previous
                    for _ in range(10):
                        if high - low < .1:
                            break
                        mid = (low + high) / 2
                        if measure(mid):
                            low = mid
                        else:
                            high = mid
                    chosen = low
                    break
                previous = float(size)
            measure(chosen)
            chosen = current_size
        finally:
            scratch.geometry_controller.release_render_resources()
            scratch.deleteLater()

    item.layout.set_flow_shape(block.bubble_shape if block.bubble_layout else None)
    was_repainting = item.repaint_on_changed
    item.repaint_on_changed = False
    try:
        if auto_size and original_size > 0:
            item.setRelFontSize(chosen / original_size)
        item.setRect(rect, repaint=False)
    finally:
        item.repaint_on_changed = was_repainting
    item.repaint_background()
    actual = item.layout.max_font_size(to_px=True)
    item.fontformat.font_size = item.minFontSize()
    # Qt rounds rich-text sizes independently; diagnose the applied layout.
    result = FitResult(actual, not _ink_fits(item, rect), actual < 9 - .1)
    if result.overflow:
        LOGGER.warning('Text block %s does not fit its flow area at %.1f px.', item.idx, actual)
    elif result.below_readability_floor:
        LOGGER.warning('Koharu text block %s fitted below 9 px (%.1f px).', item.idx, actual)
    return result

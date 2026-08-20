"""CSS-like emphasis-mark metrics and painting for the existing layouts."""

from __future__ import annotations

from typing import Iterable, Iterator, NamedTuple, Optional

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QFont,
    QFontMetricsF,
    QPainter,
    QPen,
    QTextBlock,
    QTextCharFormat,
    QTextLine,
    QTransform,
)

from ..annotations import (
    EMPHASIS_GLYPHS,
    TEXT_COMBINE_ALL,
    emphasis_values,
    text_combine_upright_values,
)
from .glyph import (
    GLYPH_STROKE_FORMAT_PROPERTY,
    PaintSpan,
    glyph_geometry,
    logical_span_rect,
    resolve_paint_spans,
)
from .indexing import _grapheme_ranges, _utf16_slice
from .native_document import (
    NativeTextDocument,
    draw_native_text_document,
    native_text_document,
)

EMPHASIS_FONT_SCALE = 0.5
EMPHASIS_GAP_SCALE = 0.08


class EmphasisMark(NamedTuple):
    """One layout-owned placement of a cached native mark document."""

    source: NativeTextDocument
    offset: QPointF

    @property
    def ink_bounds(self) -> QRectF:
        return self.source.ink_bounds.translated(self.offset)


def _mark_font(char_format: QTextCharFormat) -> QFont:
    font = QFont(char_format.font())
    point_size = font.pointSizeF()
    if point_size > 0:
        font.setPointSizeF(point_size * EMPHASIS_FONT_SCALE)
    elif font.pixelSize() > 0:
        font.setPixelSize(max(1, round(font.pixelSize() * EMPHASIS_FONT_SCALE)))
    font.setUnderline(False)
    font.setOverline(False)
    font.setStrikeOut(False)
    tag_type = getattr(QFont, 'Tag', None)
    if (
        tag_type is not None
        and hasattr(tag_type, 'fromString')
        and hasattr(font, 'setFeature')
    ):
        font.setFeature(tag_type.fromString('ruby'), 1)
    return font


def _mark_char_format(char_format: QTextCharFormat) -> QTextCharFormat:
    """Keep glyph paint inputs while dropping document semantics."""
    result = QTextCharFormat()
    result.setFont(_mark_font(char_format))
    foreground = char_format.foreground()
    if foreground.style() != Qt.BrushStyle.NoBrush:
        result.setForeground(foreground)
    outline = QPen(char_format.textOutline())
    if outline.style() != Qt.PenStyle.NoPen:
        if outline.widthF() > 0.0:
            outline.setWidthF(outline.widthF() * EMPHASIS_FONT_SCALE)
        result.setTextOutline(outline)
    return result


def _mark_document(
    style: str,
    char_format: QTextCharFormat,
) -> NativeTextDocument:
    mark_format = _mark_char_format(char_format)
    return native_text_document(EMPHASIS_GLYPHS[style], mark_format)


def _mark_extent(
    style: str,
    char_format: QTextCharFormat,
    *,
    vertical: bool,
) -> float:
    bounds = _mark_document(style, char_format).glyph_bounds
    ink_extent = bounds.width() if vertical else bounds.height()
    gap = QFontMetricsF(char_format.font()).height() * EMPHASIS_GAP_SCALE
    # Stroke rendering temporarily injects an outline into a cloned document.
    # Effect padding owns that extra ink; counting it here would reflow the
    # clone away from the live fill geometry.
    return ink_extent + gap


def emphasis_margins(
    block: QTextBlock,
    line: QTextLine,
    *,
    vertical: bool,
) -> tuple[float, float]:
    """Return line-relative margins required by its emphasis marks.

    The two values mean over/under horizontally and right/left vertically.
    Like CSS ruby spacing, these are leading around the base text rather than
    a change to the base glyph's own metrics.
    """
    first = second = 0.0
    for span in resolve_paint_spans(block, line, tuple(block.layout().formats())):
        style, position = emphasis_values(span.char_format)
        if style == 'none':
            continue
        extent = _mark_extent(
            style,
            span.char_format,
            vertical=vertical,
        )
        side = position.split()
        first_side = side[1] == 'right' if vertical else side[0] == 'over'
        if first_side:
            first = max(first, extent)
        else:
            second = max(second, extent)
    return first, second


def _effect_spans(
    block: QTextBlock,
    line: QTextLine,
    context: QAbstractTextDocumentLayout.PaintContext,
) -> Iterable[PaintSpan]:
    additional_formats = tuple(block.layout().formats())
    effect_selections = tuple(
        selection
        for selection in context.selections
        if bool(selection.format.property(GLYPH_STROKE_FORMAT_PROPERTY))
    )
    if not effect_selections or len(effect_selections) != len(context.selections):
        return resolve_paint_spans(block, line, additional_formats)
    spans = []
    line_start = line.textStart()
    line_end = line_start + line.textLength()
    for selection in effect_selections:
        selection_start = selection.cursor.selectionStart() - block.position()
        selection_end = selection.cursor.selectionEnd() - block.position()
        if selection_start >= line_end or selection_end <= line_start:
            continue
        for span in resolve_paint_spans(
            block, line, additional_formats, selection
        ):
            if selection_start <= span.start < selection_end:
                spans.append(span)
    return spans


def _iter_emphasis_marks(
    block: QTextBlock,
    line: QTextLine,
    *,
    vertical: bool,
    context: Optional[QAbstractTextDocumentLayout.PaintContext] = None,
    offset: QPointF = QPointF(),
    orientation: QTransform = QTransform(),
    side_offsets: tuple[float, float] = (0.0, 0.0),
) -> Iterator[EmphasisMark]:
    """Yield exact mark geometry for painting and effect-bound queries."""
    line_start = line.textStart()
    line_end = line_start + line.textLength()
    graphemes = tuple(
        (start, end)
        for start, end in _grapheme_ranges(block.text())
        if start < line_end and end > line_start
    )
    if not graphemes:
        return
    spans = (
        resolve_paint_spans(block, line, tuple(block.layout().formats()))
        if context is None
        else _effect_spans(block, line, context)
    )
    combined_unit = vertical and any(
        span.start <= line_start < span.start + span.length
        and text_combine_upright_values(span.char_format)[0]
        == TEXT_COMBINE_ALL
        for span in spans
    )
    if combined_unit:
        # A combined run occupies one vertical typographic unit.
        # If fragment styles differ, the first emphasized fragment owns its
        # single mark while every base glyph keeps its own normal formatting.
        graphemes = ((line_start, line_end),)
    for span in spans:
        style, position = emphasis_values(span.char_format)
        if style == 'none':
            continue
        source = _mark_document(style, span.char_format)
        span_end = span.start + span.length
        for start, end in graphemes:
            owns_mark = (
                span.start < line_end and span_end > line_start
                if combined_unit
                else span.start <= start < span_end
            )
            if not owns_mark:
                continue
            text = _utf16_slice(block.text(), start, end - start)
            if not text or text.isspace():
                continue
            cell = logical_span_rect(
                line,
                start,
                end - start,
                offset,
                orientation,
            )
            if combined_unit:
                run_bounds = glyph_geometry(
                    line,
                    line_start,
                    line.textLength(),
                    offset,
                    orientation,
                    0.0,
                ).bounds
                if not run_bounds.isEmpty():
                    cell = run_bounds
            if cell.isEmpty():
                continue
            mark_bounds = source.glyph_bounds
            gap = (
                QFontMetricsF(span.char_format.font()).height()
                * EMPHASIS_GAP_SCALE
            )
            horizontal_side, vertical_side = position.split()
            if vertical:
                side_offset = (
                    side_offsets[0]
                    if vertical_side == 'right'
                    else side_offsets[1]
                )
                x = (
                    cell.right() + side_offset + gap + mark_bounds.width() / 2
                    if vertical_side == 'right'
                    else cell.left() - side_offset - gap - mark_bounds.width() / 2
                )
                center = QPointF(x, cell.center().y())
            else:
                side_offset = (
                    side_offsets[0]
                    if horizontal_side == 'over'
                    else side_offsets[1]
                )
                y = (
                    cell.top() - side_offset - gap - mark_bounds.height() / 2
                    if horizontal_side == 'over'
                    else cell.bottom() + side_offset + gap + mark_bounds.height() / 2
                )
                center = QPointF(cell.center().x(), y)
            yield EmphasisMark(
                source,
                center - mark_bounds.center(),
            )
            if combined_unit:
                return


def draw_emphasis_marks(
    painter: QPainter,
    block: QTextBlock,
    line: QTextLine,
    context: QAbstractTextDocumentLayout.PaintContext,
    *,
    vertical: bool,
    offset: QPointF = QPointF(),
    orientation: QTransform = QTransform(),
    side_offsets: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Paint one mark per emphasized typographic unit using fragment style."""
    for mark in _iter_emphasis_marks(
        block,
        line,
        vertical=vertical,
        context=context,
        offset=offset,
        orientation=orientation,
        side_offsets=side_offsets,
    ):
        draw_native_text_document(
            painter,
            mark.source,
            QTransform.fromTranslate(mark.offset.x(), mark.offset.y()),
        )


def emphasis_ink_bounds(
    block: QTextBlock,
    line: QTextLine,
    *,
    vertical: bool,
    offset: QPointF = QPointF(),
    orientation: QTransform = QTransform(),
    side_offsets: tuple[float, float] = (0.0, 0.0),
) -> QRectF:
    """Return source-space mark ink for effect padding calculations."""
    bounds = QRectF()
    for mark in _iter_emphasis_marks(
        block,
        line,
        vertical=vertical,
        offset=offset,
        orientation=orientation,
        side_offsets=side_offsets,
    ):
        mark_bounds = mark.ink_bounds
        bounds = (
            QRectF(mark_bounds)
            if bounds.isNull()
            else bounds.united(mark_bounds)
        )
    return bounds

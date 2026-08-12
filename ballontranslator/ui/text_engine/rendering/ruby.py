"""Shared Ruby measurement and glyph geometry for both text layouts."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QTextBlock,
    QTextCharFormat,
    QTextLayout,
    QTextLine,
    QTransform,
)

from ..layout import get_punc_rect
from ..annotations import (
    AnnotationProperty,
    RubyContainerRange,
    RubyUnitRange,
    letter_spacing_value,
    ruby_containers_in_block,
)
from .glyph import (
    FallbackGlyph,
    GlyphGeometry,
    _composed_transform,
    draw_glyph_geometry,
    glyph_geometry,
)
from .indexing import (
    _grapheme_ranges,
    _utf16_char_at,
    _utf16_length,
    _utf16_slice,
)


RUBY_FONT_SCALE = 0.5
RUBY_GAP_SCALE = 0.06
RUBY_LAYOUT_SPACING_PROPERTY = int(AnnotationProperty.RUBY_POSITION) + 20


@dataclass(frozen=True)
class RubyUnitMetrics:
    """One unit's base and annotation inline measurements.

    >>> RubyUnitMetrics(None, None, 10.0, 14.0).extra
    4.0
    """

    container: RubyContainerRange
    unit: RubyUnitRange
    base_advance: float
    annotation_advance: float

    @property
    def extent(self) -> float:
        return max(self.base_advance, self.annotation_advance)

    @property
    def extra(self) -> float:
        return max(0.0, self.annotation_advance - self.base_advance)


@dataclass(frozen=True)
class RubyFormatIndex:
    """Block-local fragment lookup shared by Ruby measurement and paint."""

    block_format: QTextCharFormat
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    formats: tuple[QTextCharFormat, ...]
    layout_formats: tuple[QTextLayout.FormatRange, ...]

    @classmethod
    def from_block(cls, block: QTextBlock) -> RubyFormatIndex:
        starts = []
        ends = []
        formats = []
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.length() > 0:
                start = fragment.position() - block.position()
                starts.append(start)
                ends.append(start + fragment.length())
                formats.append(fragment.charFormat())
            iterator += 1
        layout_formats = tuple(
            format_range
            for format_range in block.layout().formats()
            if not bool(format_range.format.property(
                RUBY_LAYOUT_SPACING_PROPERTY
            ))
        )
        return cls(
            QTextCharFormat(block.charFormat()),
            tuple(starts),
            tuple(ends),
            tuple(formats),
            layout_formats,
        )

    def format_at(self, position: int) -> QTextCharFormat:
        index = bisect_right(self.starts, position) - 1
        result = QTextCharFormat(self.block_format)
        if index >= 0 and position < self.ends[index]:
            result = QTextCharFormat(self.formats[index])
        for format_range in self.layout_formats:
            start = int(format_range.start)
            if start <= position < start + int(format_range.length):
                result.merge(format_range.format)
        return result

    def ranges(self, start: int, length: int) -> list[QTextLayout.FormatRange]:
        end = start + length
        ranges = []
        index = bisect_right(self.ends, start)
        while index < len(self.starts) and self.starts[index] < end:
            overlap_start = max(start, self.starts[index])
            overlap_end = min(end, self.ends[index])
            format_range = QTextLayout.FormatRange()
            format_range.start = overlap_start - start
            format_range.length = overlap_end - overlap_start
            format_range.format = self.formats[index]
            ranges.append(format_range)
            index += 1
        return ranges


@dataclass(frozen=True)
class RubyBlockMetrics:
    """Ordered block metrics with logarithmic unit and fragment lookup."""

    units: tuple[RubyUnitMetrics, ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    format_index: Optional[RubyFormatIndex]

    @classmethod
    def empty(cls) -> RubyBlockMetrics:
        return cls((), (), (), None)

    @classmethod
    def create(
        cls,
        units: Iterable[RubyUnitMetrics],
        format_index: RubyFormatIndex,
    ) -> RubyBlockMetrics:
        normalized = tuple(units)
        return cls(
            normalized,
            tuple(metric.unit.start for metric in normalized),
            tuple(metric.unit.end for metric in normalized),
            format_index,
        )

    def __bool__(self) -> bool:
        return bool(self.units)

    def __iter__(self):
        return iter(self.units)

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> RubyUnitMetrics:
        return self.units[index]

    def overlapping(self, start: int, end: int) -> tuple[RubyUnitMetrics, ...]:
        index = bisect_right(self.ends, start)
        result = []
        while index < len(self.units) and self.starts[index] < end:
            result.append(self.units[index])
            index += 1
        return tuple(result)

    def contained(self, start: int, end: int) -> tuple[RubyUnitMetrics, ...]:
        index = bisect_right(self.starts, start - 1)
        result = []
        while index < len(self.units) and self.ends[index] <= end:
            result.append(self.units[index])
            index += 1
        return tuple(result)

    def containing(self, position: int) -> Optional[RubyUnitMetrics]:
        index = bisect_right(self.starts, position) - 1
        if index >= 0 and position <= self.ends[index]:
            return self.units[index]
        return None

    def split_by(self, position: int) -> Optional[RubyUnitMetrics]:
        index = bisect_right(self.starts, position) - 1
        if (
            index >= 0
            and self.starts[index] < position < self.ends[index]
        ):
            return self.units[index]
        return None


@dataclass(frozen=True)
class RubyPlacement:
    unit: RubyUnitRange
    cell: QRectF
    geometries: tuple[GlyphGeometry, ...]
    char_format: QTextCharFormat

    @property
    def ink_bounds(self) -> QRectF:
        bounds = QRectF()
        for geometry in self.geometries:
            if geometry.bounds.isEmpty():
                continue
            bounds = (
                QRectF(geometry.bounds)
                if bounds.isEmpty()
                else bounds.united(geometry.bounds)
            )
        return bounds


def ruby_font(char_format: QTextCharFormat) -> QFont:
    """Derive the fixed first-version Ruby font from its base fragment."""
    font = QFont(char_format.font())
    if font.pointSizeF() > 0:
        font.setPointSizeF(max(1.0, font.pointSizeF() * RUBY_FONT_SCALE))
    elif font.pixelSize() > 0:
        font.setPixelSize(max(1, round(font.pixelSize() * RUBY_FONT_SCALE)))
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100.0)
    return font


def _format_at(
    block: QTextBlock,
    position: int,
    format_index: Optional[RubyFormatIndex] = None,
) -> QTextCharFormat:
    if format_index is not None:
        return format_index.format_at(position)
    absolute = block.position() + position
    result = QTextCharFormat(block.charFormat())
    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        if (
            fragment.isValid()
            and fragment.position() <= absolute
            and absolute < fragment.position() + fragment.length()
        ):
            result = QTextCharFormat(fragment.charFormat())
            break
        iterator += 1
    for format_range in block.layout().formats():
        start = int(format_range.start)
        if start <= position < start + int(format_range.length):
            if not bool(format_range.format.property(RUBY_LAYOUT_SPACING_PROPERTY)):
                result.merge(format_range.format)
    return result


def ruby_char_format(
    block: QTextBlock,
    position: int,
    context: Optional[QAbstractTextDocumentLayout.PaintContext] = None,
    length: int = 1,
    format_index: Optional[RubyFormatIndex] = None,
) -> QTextCharFormat:
    result = _format_at(block, position, format_index)
    if context is not None:
        absolute = block.position() + position
        for selection in context.selections:
            if (
                selection.cursor.selectionStart() < absolute + length
                and selection.cursor.selectionEnd() > absolute
            ):
                result.merge(selection.format)
    result.setFont(ruby_font(result))
    result.setProperty(AnnotationProperty.LETTER_SPACING, 1.0)
    return result


def _measure_layout(text: str, formats: Iterable[QTextLayout.FormatRange]) -> float:
    if not text:
        return 0.0
    formats = list(formats)
    default_font = formats[0].format.font() if formats else QFont()
    layout = QTextLayout(text, default_font)
    layout.setFormats(formats)
    layout.beginLayout()
    line = layout.createLine()
    line.setLineWidth(1_000_000.0)
    layout.endLayout()
    return max(0.0, float(line.naturalTextWidth()))


def _measure_span(
    text: str,
    formats: Iterable[QTextLayout.FormatRange],
    start: int,
    end: int,
) -> float:
    formats = list(formats)
    default_font = formats[0].format.font() if formats else QFont()
    layout = QTextLayout(text, default_font)
    layout.setFormats(formats)
    layout.beginLayout()
    line = layout.createLine()
    line.setLineWidth(1_000_000.0)
    layout.endLayout()

    def cursor_x(position: int) -> float:
        value = line.cursorToX(position)
        if isinstance(value, (tuple, list)):
            value = value[0]
        return float(value)

    return max(0.0, cursor_x(end) - cursor_x(start))


def _base_formats(
    format_index: RubyFormatIndex,
    start: int,
    length: int,
) -> list[QTextLayout.FormatRange]:
    return format_index.ranges(start, length)


def _horizontal_base_probe(
    block_text: str,
    format_index: RubyFormatIndex,
    start: int,
    length: int,
) -> tuple[str, list[QTextLayout.FormatRange]]:
    end = start + length
    probe_end = end
    if end < _utf16_length(block_text):
        probe_end += _utf16_length(_utf16_char_at(block_text, end))
    return (
        _utf16_slice(block_text, start, probe_end - start),
        _base_formats(format_index, start, probe_end - start),
    )


def horizontal_ruby_metrics(
    block: QTextBlock,
) -> RubyBlockMetrics:
    metrics = []
    block_text = block.text()
    containers = ruby_containers_in_block(block)
    if not containers:
        return RubyBlockMetrics.empty()
    format_index = RubyFormatIndex.from_block(block)
    for container in containers:
        for unit in container.units:
            local_start = unit.start - block.position()
            probe_text, probe_formats = _horizontal_base_probe(
                block_text, format_index, local_start, unit.length
            )
            base_advance = _measure_span(
                probe_text, probe_formats, 0, unit.length
            )
            ruby_format = ruby_char_format(
                block, local_start, format_index=format_index
            )
            annotation_advance = QFontMetricsF(
                ruby_format.font()
            ).horizontalAdvance(unit.text)
            metrics.append(RubyUnitMetrics(
                container, unit, base_advance, annotation_advance
            ))
    return RubyBlockMetrics.create(metrics, format_index)


def vertical_ruby_metrics(
    block: QTextBlock,
    needs_rotation: Optional[Callable[[str], bool]] = None,
    letter_spacing_fallback: float = 1.0,
) -> RubyBlockMetrics:
    """Measure upright base and annotation runs along a vertical column."""
    metrics = []
    block_text = block.text()
    containers = ruby_containers_in_block(block)
    if not containers:
        return RubyBlockMetrics.empty()
    format_index = RubyFormatIndex.from_block(block)
    for container in containers:
        for unit in container.units:
            local_start = unit.start - block.position()
            base_text = _utf16_slice(block_text, local_start, unit.length)
            base_advance = 0.0
            for start, end in _grapheme_ranges(base_text):
                grapheme = _utf16_slice(base_text, start, end - start)
                char_format = _format_at(
                    block, local_start + start, format_index
                )
                font = char_format.font()
                reference_height = get_punc_rect(
                    '木',
                    font.family(),
                    font.pointSizeF(),
                    font.weight(),
                    font.italic(),
                )[0].height()
                natural_advance = (
                    _measure_layout(
                        grapheme,
                        _base_formats(
                            format_index, local_start + start, end - start
                        ),
                    )
                    if needs_rotation is not None
                    and needs_rotation(grapheme[0])
                    else reference_height
                )
                spacing = letter_spacing_value(
                    char_format, letter_spacing_fallback
                )
                base_advance += max(
                    0.0,
                    natural_advance + reference_height * (spacing - 1.0),
                )
            ruby_format = ruby_char_format(
                block, local_start, format_index=format_index
            )
            annotation_advance = (
                len(_grapheme_ranges(unit.text))
                * QFontMetricsF(ruby_format.font()).height()
            )
            metrics.append(RubyUnitMetrics(
                container, unit, base_advance, annotation_advance
            ))
    return RubyBlockMetrics.create(metrics, format_index)


def _remove_ruby_spacing_formats(block: QTextBlock) -> tuple[list, bool]:
    previous = list(block.layout().formats())
    formats = [
        format_range
        for format_range in previous
        if not bool(format_range.format.property(RUBY_LAYOUT_SPACING_PROPERTY))
    ]
    removed = len(formats) != len(previous)
    if removed:
        block.layout().setFormats(formats)
    return formats, removed


def prepare_horizontal_ruby_layout(
    block: QTextBlock,
) -> RubyBlockMetrics:
    """Reserve long-Ruby inline extent through one transient trailing range."""
    formats, _removed = _remove_ruby_spacing_formats(block)
    metrics = horizontal_ruby_metrics(block)
    added = False
    for metric in metrics:
        if metric.extra <= 1e-6:
            continue
        local_start = metric.unit.start - block.position()
        base_text = _utf16_slice(block.text(), local_start, metric.unit.length)
        graphemes = _grapheme_ranges(base_text)
        if not graphemes:
            continue
        last_start, last_end = graphemes[-1]
        format_index = metrics.format_index
        if format_index is None:
            continue
        char_format = _format_at(
            block, local_start + last_start, format_index
        )
        font = QFont(char_format.font())
        if font.letterSpacingType() == QFont.SpacingType.AbsoluteSpacing:
            existing_spacing = font.letterSpacing()
        else:
            glyph = _utf16_slice(base_text, last_start, last_end - last_start)
            natural = max(0.0, QFontMetricsF(font).horizontalAdvance(glyph))
            existing_spacing = (
                natural * (font.letterSpacing() - 100.0) / 100.0
            )
        block_text = block.text()
        probe_text, probe_formats = _horizontal_base_probe(
            block_text, format_index, local_start, metric.unit.length
        )

        def advance_with_spacing(spacing: float) -> float:
            probe_format = QTextCharFormat(char_format)
            probe_format.setFontLetterSpacingType(
                QFont.SpacingType.AbsoluteSpacing
            )
            probe_format.setFontLetterSpacing(spacing)
            probe_range = QTextLayout.FormatRange()
            probe_range.start = last_start
            probe_range.length = last_end - last_start
            probe_range.format = probe_format
            return _measure_span(
                probe_text,
                (*probe_formats, probe_range),
                0,
                metric.unit.length,
            )

        adjusted_advance = advance_with_spacing(existing_spacing)
        spacing_response = (
            advance_with_spacing(existing_spacing + 1.0) - adjusted_advance
        )
        required_extra = max(0.0, metric.extent - adjusted_advance)
        char_format.setFontLetterSpacingType(QFont.SpacingType.AbsoluteSpacing)
        char_format.setFontLetterSpacing(
            existing_spacing
            + required_extra / max(1e-6, spacing_response)
        )
        char_format.setProperty(RUBY_LAYOUT_SPACING_PROPERTY, True)
        format_range = QTextLayout.FormatRange()
        format_range.start = local_start + last_start
        format_range.length = last_end - last_start
        format_range.format = char_format
        formats.append(format_range)
        added = True
    if added:
        block.layout().setFormats(formats)
    return metrics


def clear_horizontal_ruby_layout(block: QTextBlock) -> None:
    _remove_ruby_spacing_formats(block)


def protect_horizontal_ruby_wrap(
    block: QTextBlock,
    line: QTextLine,
    metrics: RubyBlockMetrics,
) -> None:
    """Move an incomplete unit to the next line, or keep it whole if empty."""
    line_start = line.textStart()
    line_end = line_start + line.textLength()
    metric = metrics.split_by(block.position() + line_end)
    if metric is not None:
        unit_start = metric.unit.start - block.position()
        unit_end = metric.unit.end - block.position()
        target_end = unit_start if unit_start > line_start else unit_end
        line.setNumColumns(max(1, target_end - line_start))


def ruby_side_margins(
    block: QTextBlock,
    line: QTextLine,
    metrics: RubyBlockMetrics,
    *,
    vertical: bool,
) -> tuple[float, float]:
    """Return over/under or right/left block-axis margins for one line."""
    first = second = 0.0
    line_start = block.position() + line.textStart()
    line_end = line_start + line.textLength()
    for metric in metrics.overlapping(line_start, line_end):
        unit = metric.unit
        char_format = ruby_char_format(
            block,
            unit.start - block.position(),
            format_index=metrics.format_index,
        )
        font_metrics = QFontMetricsF(char_format.font())
        gap = QFontMetricsF(_format_at(
            block, unit.start - block.position(), metrics.format_index
        ).font()).height() * RUBY_GAP_SCALE
        extent = (
            font_metrics.maxWidth() if vertical else font_metrics.height()
        ) + gap
        if metric.container.position == 'over':
            first = max(first, extent)
        else:
            second = max(second, extent)
    return first, second


def _temporary_line(text: str, font: QFont) -> tuple[QTextLayout, QTextLine]:
    layout = QTextLayout(text, font)
    # QTextLine keeps the implicit-shared QTextLayout data alive.
    layout.beginLayout()
    line = layout.createLine()
    line.setLineWidth(1_000_000.0)
    layout.endLayout()
    return layout, line


def _translated_geometry(geometry: GlyphGeometry, offset: QPointF) -> GlyphGeometry:
    paths = []
    for source in geometry.paths:
        path = QPainterPath(source)
        path.translate(offset)
        paths.append(path)
    fallbacks = []
    translation = QTransform.fromTranslate(offset.x(), offset.y())
    for fallback in geometry.fallbacks:
        fallbacks.append(FallbackGlyph(
            fallback.run,
            _composed_transform(fallback.transform, translation),
            fallback.bounds.translated(offset),
            fallback.raw_bounds,
            fallback.native_color,
        ))
    return GlyphGeometry(
        tuple(paths),
        tuple(fallbacks),
        geometry.bounds.translated(offset),
    )


def _horizontal_geometry(
    text: str,
    char_format: QTextCharFormat,
    cell: QRectF,
    position: str,
    glyph_slant_angle: float,
) -> tuple[GlyphGeometry, ...]:
    _layout, line = _temporary_line(text, char_format.font())
    geometry = glyph_geometry(
        line, 0, _utf16_length(text), QPointF(), QTransform(), glyph_slant_angle
    )
    if geometry.bounds.isEmpty():
        return ()
    base_height = QFontMetricsF(char_format.font()).height() / RUBY_FONT_SCALE
    gap = base_height * RUBY_GAP_SCALE
    target_x = cell.center().x()
    target_y = (
        cell.top() - gap - geometry.bounds.height() / 2
        if position == 'over'
        else cell.bottom() + gap + geometry.bounds.height() / 2
    )
    return (_translated_geometry(
        geometry, QPointF(target_x, target_y) - geometry.bounds.center()
    ),)


def _vertical_geometries(
    text: str,
    char_format: QTextCharFormat,
    cell: QRectF,
    position: str,
    glyph_slant_angle: float,
) -> tuple[GlyphGeometry, ...]:
    source = []
    total_height = 0.0
    for start, end in _grapheme_ranges(text):
        grapheme = _utf16_slice(text, start, end - start)
        _layout, line = _temporary_line(grapheme, char_format.font())
        geometry = glyph_geometry(
            line, 0, end - start, QPointF(), QTransform(), glyph_slant_angle
        )
        height = max(
            geometry.bounds.height(), QFontMetricsF(char_format.font()).height()
        )
        source.append((geometry, height))
        total_height += height
    if not source:
        return ()
    base_height = QFontMetricsF(char_format.font()).height() / RUBY_FONT_SCALE
    gap = base_height * RUBY_GAP_SCALE
    x = (
        cell.right() + gap
        if position == 'over'
        else cell.left() - gap
    )
    y = cell.center().y() - total_height / 2
    result = []
    for geometry, height in source:
        center_x = (
            x + geometry.bounds.width() / 2
            if position == 'over'
            else x - geometry.bounds.width() / 2
        )
        center = QPointF(center_x, y + height / 2)
        result.append(_translated_geometry(
            geometry, center - geometry.bounds.center()
        ))
        y += height
    return tuple(result)


def ruby_placement(
    block: QTextBlock,
    container: RubyContainerRange,
    unit: RubyUnitRange,
    cell: QRectF,
    *,
    vertical: bool,
    context: Optional[QAbstractTextDocumentLayout.PaintContext] = None,
    glyph_slant_angle: float = 0.0,
    format_index: Optional[RubyFormatIndex] = None,
) -> RubyPlacement:
    char_format = ruby_char_format(
        block,
        unit.start - block.position(),
        context,
        unit.length,
        format_index,
    )
    geometries = (
        _vertical_geometries(
            unit.text, char_format, cell, container.position, glyph_slant_angle
        )
        if vertical
        else _horizontal_geometry(
            unit.text, char_format, cell, container.position, glyph_slant_angle
        )
    )
    return RubyPlacement(unit, QRectF(cell), geometries, char_format)


def draw_ruby_placement(painter: QPainter, placement: RubyPlacement) -> None:
    for geometry in placement.geometries:
        draw_glyph_geometry(painter, geometry, placement.char_format)

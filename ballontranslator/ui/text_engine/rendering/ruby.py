"""Shared Ruby measurement and glyph geometry for both text layouts."""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QBrush,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
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
    glyph_geometry,
    glyph_slant_transform,
)
from .indexing import (
    _grapheme_ranges,
    _utf16_char_at,
    _utf16_length,
    _utf16_slice,
)
from .native_document import (
    NativeTextDocument,
    draw_native_text_document,
    native_text_document,
)


RUBY_FONT_SCALE = 0.5
RUBY_GAP_SCALE = 0.06
RUBY_LAYOUT_SPACING_PROPERTY = int(AnnotationProperty.RUBY_POSITION) + 20


@dataclass(frozen=True)
class RubyUnitMetrics:
    """One unit's base and annotation inline measurements.

    >>> RubyUnitMetrics(None, None, 10.0, 14.0, 5.0).extra
    4.0
    """

    container: RubyContainerRange
    unit: RubyUnitRange
    base_advance: float
    annotation_advance: float
    annotation_cross_extent: float
    base_opportunity_ends: tuple[int, ...] = ()
    base_gap: float = 0.0
    annotation_center_offset: float = 0.0

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
    base_gap_prefix: tuple[float, ...]
    format_index: Optional[RubyFormatIndex]

    @classmethod
    def empty(cls) -> RubyBlockMetrics:
        return cls((), (), (), (0.0,), None)

    @classmethod
    def create(
        cls,
        units: Iterable[RubyUnitMetrics],
        format_index: RubyFormatIndex,
    ) -> RubyBlockMetrics:
        normalized = tuple(units)
        base_gap_prefix = [0.0]
        for metric in normalized:
            base_gap_prefix.append(base_gap_prefix[-1] + metric.base_gap)
        return cls(
            normalized,
            tuple(metric.unit.start for metric in normalized),
            tuple(metric.unit.end for metric in normalized),
            tuple(base_gap_prefix),
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

    def base_gap_before(self, position: int) -> float:
        """Return external edge spacing for units ending by ``position``."""
        return self.base_gap_prefix[bisect_right(self.ends, position)]


@dataclass(frozen=True)
class RubyPaintRun:
    source: NativeTextDocument
    transform: QTransform
    geometry: GlyphGeometry


@dataclass(frozen=True)
class RubyPlacement:
    unit: RubyUnitRange
    cell: QRectF
    paint_runs: tuple[RubyPaintRun, ...]
    char_format: QTextCharFormat

    @property
    def geometries(self) -> tuple[GlyphGeometry, ...]:
        return tuple(run.geometry for run in self.paint_runs)

    @property
    def ink_bounds(self) -> QRectF:
        bounds = QRectF()
        for run in self.paint_runs:
            geometry = run.geometry
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


def _ruby_document_format(
    char_format: QTextCharFormat,
) -> QTextCharFormat:
    """Keep Ruby glyph paint while dropping base-document semantics."""
    font = QFont(char_format.font())
    font.setUnderline(False)
    font.setOverline(False)
    font.setStrikeOut(False)
    result = QTextCharFormat()
    result.setFont(font)
    foreground = char_format.foreground()
    if foreground.style() != Qt.BrushStyle.NoBrush:
        result.setForeground(QBrush(foreground))
    outline = QPen(char_format.textOutline())
    if outline.style() != Qt.PenStyle.NoPen:
        if outline.widthF() > 0.0:
            outline.setWidthF(outline.widthF() * RUBY_FONT_SCALE)
        result.setTextOutline(outline)
    return result


def _space_around_spacing(
    text: str,
    extra: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], float]:
    """Return graphemes, eligible boundaries, and one full opportunity.

    The two half-size edge spaces together consume one full opportunity.

    >>> _space_around_spacing('漢字A', 20.0)[1:]
    ((1, 2), 6.666666666666667)
    >>> _space_around_spacing('ㄅㄆ', 20.0)[1:]
    ((), 20.0)
    """
    graphemes = tuple(_grapheme_ranges(text))

    def distributable(start: int, end: int) -> bool:
        grapheme = _utf16_slice(text, start, end - start)
        name = unicodedata.name(grapheme[0], '')
        return any(
            script in name
            for script in (
                'CJK',
                'IDEOGRAPHIC',
                'HIRAGANA',
                'KATAKANA',
                'HANGUL',
                'LATIN',
                'ROMAN NUMERAL',
            )
        )

    opportunities = tuple(
        end
        for (start, end), (next_start, next_end) in zip(
            graphemes, graphemes[1:]
        )
        if distributable(start, end)
        and distributable(next_start, next_end)
    )
    gap = max(0.0, extra) / (len(opportunities) + 1)
    return graphemes, opportunities, gap


def _space_around_positions(
    text: str,
    font: QFont,
    extent: float,
    *,
    vertical: bool,
) -> tuple[tuple[str, float], ...]:
    """Return space-around centers along one Ruby inline axis.

    A horizontal run keeps native shaping when it does not need expansion.

    >>> len(_space_around_positions('AB', QFont(), 100, vertical=False))
    2
    >>> len(_space_around_positions('哈佛', QFont(), 100, vertical=False))
    2
    >>> len(_space_around_positions('ＡＢ', QFont(), 100, vertical=False))
    2
    """
    ranges, opportunities, _gap = _space_around_spacing(text, 0.0)
    graphemes = tuple(
        _utf16_slice(text, start, end - start) for start, end in ranges
    )
    if not graphemes:
        return ()
    metrics = QFontMetricsF(font)
    if vertical:
        runs = tuple(zip(graphemes, ranges))
        advances = tuple(metrics.height() for _grapheme in graphemes)
    else:
        natural_advance = metrics.horizontalAdvance(text)
        if extent <= natural_advance + 1e-6:
            return ((text, extent / 2),)
        runs = []
        run_start = 0
        for opportunity_end in opportunities:
            runs.append((
                _utf16_slice(text, run_start, opportunity_end - run_start),
                (run_start, opportunity_end),
            ))
            run_start = opportunity_end
        text_end = ranges[-1][1]
        runs.append((
            _utf16_slice(text, run_start, text_end - run_start),
            (run_start, text_end),
        ))
        runs = tuple(runs)
        advances = tuple(
            metrics.horizontalAdvance(run) for run, _range in runs
        )
        if extent <= sum(advances) + 1e-6:
            return ((text, extent / 2),)
    free = max(0.0, extent - sum(advances))
    gap = free / (len(opportunities) + 1)
    cursor = gap / 2
    positions = []
    opportunity_ends = frozenset(opportunities)
    for (run, _range), advance in zip(runs, advances):
        positions.append((run, cursor + advance / 2))
        cursor += advance
        if _range[1] in opportunity_ends:
            cursor += gap
    return tuple(positions)


def _annotation_cross_extent(
    text: str,
    char_format: QTextCharFormat,
) -> float:
    """Measure the widest actual annotation glyph, not a font maximum."""
    width = 0.0
    for start, end in _grapheme_ranges(text):
        grapheme = _utf16_slice(text, start, end - start)
        _layout, line = _temporary_line(grapheme, char_format.font())
        geometry = glyph_geometry(
            line,
            0,
            _utf16_length(grapheme),
            QPointF(),
            QTransform(),
            0.0,
        )
        width = max(width, geometry.bounds.width())
    return width


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
            base_text = _utf16_slice(
                block_text, local_start, unit.length
            )
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
            _ranges, opportunities, gap = _space_around_spacing(
                base_text,
                max(0.0, annotation_advance - base_advance),
            )
            metrics.append(RubyUnitMetrics(
                container,
                unit,
                base_advance,
                annotation_advance,
                0.0,
                opportunities,
                gap,
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
            trailing_spacing = 0.0
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
                advance = max(
                    0.0,
                    natural_advance + reference_height * (spacing - 1.0),
                )
                base_advance += advance
                # Tracking trails the glyph frame. Half of the final trailing
                # advance separates the unit-cell center from its glyph center.
                trailing_spacing = advance - natural_advance
            ruby_format = ruby_char_format(
                block, local_start, format_index=format_index
            )
            annotation_advance = (
                len(_grapheme_ranges(unit.text))
                * QFontMetricsF(ruby_format.font()).height()
            )
            _ranges, opportunities, gap = _space_around_spacing(
                base_text,
                max(0.0, annotation_advance - base_advance),
            )
            metrics.append(RubyUnitMetrics(
                container,
                unit,
                base_advance,
                annotation_advance,
                _annotation_cross_extent(unit.text, ruby_format),
                opportunities,
                gap,
                -trailing_spacing / 2,
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
    """Reserve and distribute a shorter base with transient native spacing."""
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
        format_index = metrics.format_index
        if format_index is None:
            continue
        spacing_ends = frozenset(metric.base_opportunity_ends)
        if not spacing_ends:
            continue
        spacing_targets = []
        for start, end in graphemes:
            if end not in spacing_ends:
                continue
            char_format = _format_at(
                block, local_start + start, format_index
            )
            font = QFont(char_format.font())
            if font.letterSpacingType() == QFont.SpacingType.AbsoluteSpacing:
                existing_spacing = font.letterSpacing()
            else:
                glyph = _utf16_slice(base_text, start, end - start)
                natural = max(
                    0.0, QFontMetricsF(font).horizontalAdvance(glyph)
                )
                existing_spacing = (
                    natural * (font.letterSpacing() - 100.0) / 100.0
                )
            spacing_targets.append((
                start, end, char_format, existing_spacing
            ))
        block_text = block.text()
        probe_text, probe_formats = _horizontal_base_probe(
            block_text, format_index, local_start, metric.unit.length
        )

        def spacing_ranges(
            added_spacing: float,
            offset: int,
        ) -> list[QTextLayout.FormatRange]:
            ranges = []
            for start, end, source_format, existing in spacing_targets:
                char_format = QTextCharFormat(source_format)
                char_format.setFontLetterSpacingType(
                    QFont.SpacingType.AbsoluteSpacing
                )
                char_format.setFontLetterSpacing(existing + added_spacing)
                char_format.setProperty(RUBY_LAYOUT_SPACING_PROPERTY, True)
                format_range = QTextLayout.FormatRange()
                format_range.start = offset + start
                format_range.length = end - start
                format_range.format = char_format
                ranges.append(format_range)
            return ranges

        def advance_with_spacing(added_spacing: float) -> float:
            return _measure_span(
                probe_text,
                (*probe_formats, *spacing_ranges(added_spacing, 0)),
                0,
                metric.unit.length,
            )

        adjusted_advance = advance_with_spacing(0.0)
        spacing_response = (
            advance_with_spacing(1.0) - adjusted_advance
        )
        target_advance = (
            metric.base_advance
            + metric.base_gap * len(metric.base_opportunity_ends)
        )
        required_extra = max(0.0, target_advance - adjusted_advance)
        formats.extend(spacing_ranges(
            required_extra / max(1e-6, spacing_response),
            local_start,
        ))
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
            metric.annotation_cross_extent
            if vertical else font_metrics.height()
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


def _paint_run(
    text: str,
    char_format: QTextCharFormat,
    baseline: float,
    geometry: GlyphGeometry,
    translation: QPointF,
    glyph_slant_angle: float,
) -> RubyPaintRun:
    source = native_text_document(
        text, _ruby_document_format(char_format)
    )
    transform = _composed_transform(
        glyph_slant_transform(glyph_slant_angle, baseline),
        QTransform.fromTranslate(translation.x(), translation.y()),
    )
    return RubyPaintRun(
        source,
        transform,
        _translated_geometry(geometry, translation),
    )


def _horizontal_runs(
    text: str,
    char_format: QTextCharFormat,
    cell: QRectF,
    position: str,
    glyph_slant_angle: float,
) -> tuple[RubyPaintRun, ...]:
    base_height = QFontMetricsF(char_format.font()).height() / RUBY_FONT_SCALE
    gap = base_height * RUBY_GAP_SCALE
    target_y = (
        cell.top() - gap
        if position == 'over'
        else cell.bottom() + gap
    )
    result = []
    for cluster, inline_center in _space_around_positions(
        text, char_format.font(), cell.width(), vertical=False
    ):
        _layout, line = _temporary_line(cluster, char_format.font())
        geometry = glyph_geometry(
            line,
            0,
            _utf16_length(cluster),
            QPointF(),
            QTransform(),
            glyph_slant_angle,
        )
        if geometry.bounds.isEmpty():
            continue
        center_y = (
            target_y - geometry.bounds.height() / 2
            if position == 'over'
            else target_y + geometry.bounds.height() / 2
        )
        target = QPointF(cell.left() + inline_center, center_y)
        result.append(_paint_run(
            cluster,
            char_format,
            line.y() + line.ascent(),
            geometry,
            target - geometry.bounds.center(),
            glyph_slant_angle,
        ))
    return tuple(result)


def _vertical_runs(
    text: str,
    char_format: QTextCharFormat,
    cell: QRectF,
    position: str,
    glyph_slant_angle: float,
) -> tuple[RubyPaintRun, ...]:
    source = []
    for grapheme, inline_center in _space_around_positions(
        text, char_format.font(), cell.height(), vertical=True
    ):
        _layout, line = _temporary_line(grapheme, char_format.font())
        geometry = glyph_geometry(
            line,
            0,
            _utf16_length(grapheme),
            QPointF(),
            QTransform(),
            glyph_slant_angle,
        )
        source.append((
            grapheme,
            line.y() + line.ascent(),
            geometry,
            inline_center,
        ))
    if not source:
        return ()
    base_height = QFontMetricsF(char_format.font()).height() / RUBY_FONT_SCALE
    gap = base_height * RUBY_GAP_SCALE
    cross_extent = max(
        geometry.bounds.width()
        for _grapheme, _baseline, geometry, _inline_center in source
    )
    center_x = (
        cell.right() + gap + cross_extent / 2
        if position == 'over'
        else cell.left() - gap - cross_extent / 2
    )
    result = []
    for grapheme, baseline, geometry, inline_center in source:
        center = QPointF(center_x, cell.top() + inline_center)
        result.append(_paint_run(
            grapheme,
            char_format,
            baseline,
            geometry,
            center - geometry.bounds.center(),
            glyph_slant_angle,
        ))
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
    inline_offset: float = 0.0,
) -> RubyPlacement:
    char_format = ruby_char_format(
        block,
        unit.start - block.position(),
        context,
        unit.length,
        format_index,
    )
    annotation_cell = QRectF(cell)
    annotation_cell.translate(
        0.0 if vertical else inline_offset,
        inline_offset if vertical else 0.0,
    )
    paint_runs = (
        _vertical_runs(
            unit.text,
            char_format,
            annotation_cell,
            container.position,
            glyph_slant_angle,
        )
        if vertical
        else _horizontal_runs(
            unit.text,
            char_format,
            annotation_cell,
            container.position,
            glyph_slant_angle,
        )
    )
    return RubyPlacement(unit, QRectF(cell), paint_runs, char_format)


def draw_ruby_placement(painter: QPainter, placement: RubyPlacement) -> None:
    for run in placement.paint_runs:
        draw_native_text_document(painter, run.source, run.transform)

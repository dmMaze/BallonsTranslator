"""Source-space geometry for horizontal runs inside vertical text."""

from __future__ import annotations

import unicodedata
from typing import List, Tuple, Union

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QGlyphRun, QTextLayout, QTextLine, QTransform

from .glyph import glyph_geometry
from .indexing import _grapheme_count


class TateChuYokoRun(QTextLine):
    """Expose a run-local Qt line in the source block's UTF-16 coordinates.

    Only indexed glyph/cursor calls cross this boundary; Qt still owns shaping
    and spatial metrics. Keeping the layout here also owns the line's lifetime.

    >>> issubclass(TateChuYokoRun, QTextLine)
    True
    """

    def __init__(self, layout: QTextLayout, block_start: int) -> None:
        super().__init__(layout.lineAt(0))
        self.layout = layout
        self.block_start = block_start
        # This run's source line stays fixed until relayout replaces the run.
        # Placement, sizing, and interaction share its already measured ink.
        self.ink_bounds = _source_ink_bounds(layout.lineAt(0))

    def textStart(self) -> int:
        return self.block_start + super().textStart()

    def glyphRuns(self, from_: int = -1, length: int = -1) -> List[QGlyphRun]:
        if from_ < 0:
            return super().glyphRuns(from_, length)
        start = max(self.textStart(), from_)
        end = self.textStart() + self.textLength()
        if length >= 0:
            end = min(end, from_ + length)
        if end <= start:
            return []
        return super().glyphRuns(start - self.block_start, end - start)

    def cursorToX(
        self, cursorPos: int, edge: QTextLine.Edge = QTextLine.Leading,
    ) -> Union[float, Tuple[float, int]]:
        result = super().cursorToX(cursorPos - self.block_start, edge)
        if isinstance(result, tuple):
            x, position = result
            return x, position + self.block_start
        return result

    def xToCursor(
        self, x: float,
        edge: QTextLine.CursorPosition = QTextLine.CursorBetweenCharacters,
    ) -> int:
        return self.block_start + super().xToCursor(x, edge)


def normalize_tate_chu_yoko_text(text: str) -> str:
    """Reverse only explicit full-width forms in a multi-character TCY run.

    These one-to-one BMP mappings preserve the document's UTF-16 positions.

    >>> normalize_tate_chu_yoko_text('！!１２Ａ')
    '!!12A'
    >>> normalize_tate_chu_yoko_text('！')
    '！'
    >>> normalize_tate_chu_yoko_text('①㍿漢字')
    '①㍿漢字'
    """
    if _grapheme_count(text) < 2:
        return text
    result = []
    for char in text:
        decomposition = unicodedata.decomposition(char)
        result.append(
            chr(int(decomposition.split()[1], 16))
            if decomposition.startswith('<wide> ') else char
        )
    return ''.join(result)


def _source_ink_bounds(line: QTextLine) -> QRectF:
    if isinstance(line, TateChuYokoRun):
        return QRectF(line.ink_bounds)
    geometry = glyph_geometry(
        line,
        line.textStart(),
        line.textLength(),
        QPointF(),
        QTransform(),
        0.0,
    )
    bounds = geometry.bounds
    return QRectF(line.naturalTextRect() if bounds.isEmpty() else bounds)


def _source_natural_bounds(line: QTextLine, ink: QRectF) -> QRectF:
    logical = line.naturalTextRect()
    left = min(ink.left(), logical.left())
    right = max(ink.right(), logical.right())
    return QRectF(left, ink.top(), right - left, ink.height())


def tate_chu_yoko_natural_bounds(line: QTextLine) -> QRectF:
    """Include Qt's horizontal advance used by carets and decorations."""
    return _source_natural_bounds(line, _source_ink_bounds(line))


def _transform_from_ink(
    line: QTextLine,
    cell: QRectF,
    ink: QRectF,
) -> QTransform:
    natural = _source_natural_bounds(line, ink)
    if natural.isEmpty() or ink.isEmpty() or cell.isEmpty():
        return QTransform()
    scale_x = min(1.0, cell.width() / natural.width())
    source_center = ink.center()
    target_center = cell.center()
    return QTransform(
        scale_x,
        0.0,
        0.0,
        1.0,
        target_center.x() - source_center.x() * scale_x,
        target_center.y() - source_center.y(),
    )


def tate_chu_yoko_transform(
    line: QTextLine,
    cell: QRectF,
) -> QTransform:
    """Fit and center a horizontal run in its one-em vertical cell.

    Width-specific glyph variants are selected during shaping when Qt exposes
    them. This transform supplies the W3C geometric fallback when the resulting
    horizontal advance still exceeds the cell.

    >>> callable(tate_chu_yoko_transform)
    True
    """
    return _transform_from_ink(
        line,
        cell,
        _source_ink_bounds(line),
    )


def tate_chu_yoko_ink_bounds(
    line: QTextLine,
    cell: QRectF,
) -> QRectF:
    """Return the fitted ink used for visible-geometry checks."""
    source = _source_ink_bounds(line)
    return _transform_from_ink(line, cell, source).mapRect(source)

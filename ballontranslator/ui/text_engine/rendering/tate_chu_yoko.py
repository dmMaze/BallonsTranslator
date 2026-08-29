"""Source-space geometry for horizontal runs inside vertical text."""

from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QTextLine, QTransform

from .glyph import glyph_geometry


def _source_ink_bounds(line: QTextLine) -> QRectF:
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

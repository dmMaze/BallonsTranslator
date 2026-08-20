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


def tate_chu_yoko_natural_bounds(line: QTextLine) -> QRectF:
    """Include Qt's horizontal advance used by carets and decorations."""
    ink = _source_ink_bounds(line)
    logical = line.naturalTextRect()
    left = min(ink.left(), logical.left())
    right = max(ink.right(), logical.right())
    return QRectF(left, ink.top(), right - left, ink.height())


def tate_chu_yoko_transform(
    line: QTextLine,
    cell: QRectF,
) -> QTransform:
    """Center an unscaled horizontal run in its reserved vertical cell.

    The cell may grow wider than one em so the run follows Photoshop-like
    horizontal flow rather than CSS's one-em compression.

    >>> callable(tate_chu_yoko_transform)
    True
    """
    source = tate_chu_yoko_natural_bounds(line)
    if source.isEmpty() or cell.isEmpty():
        return QTransform()
    source_center = source.center()
    target_center = cell.center()
    return QTransform(
        1.0,
        0.0,
        0.0,
        1.0,
        target_center.x() - source_center.x(),
        target_center.y() - source_center.y(),
    )


def tate_chu_yoko_ink_bounds(
    line: QTextLine,
    cell: QRectF,
) -> QRectF:
    """Return the translated natural ink used for visible-geometry checks."""
    source = _source_ink_bounds(line)
    return tate_chu_yoko_transform(line, cell).mapRect(source)

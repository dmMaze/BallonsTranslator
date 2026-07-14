"""Pure helpers for the item-local post-layout text transform."""

import math

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPolygonF, QTransform

from ballontranslator.utils.fontformat import normalize_text_transform


def text_transform_point(
    point: QPointF,
    pivot: QPointF,
    horizontal_scale: float,
    vertical_scale: float,
    slant_angle: float,
) -> QPointF:
    """Map one point with the canonical scale-then-shear formula.

    ``slant_angle`` follows typographic convention: a positive value leans the
    top of horizontal text to the right, hence ``k = -tan(angle)`` in Qt's
    downward-positive coordinate system.

    >>> mapped = text_transform_point(QPointF(2, 3), QPointF(1, 1), 2, 3, 0)
    >>> (mapped.x(), mapped.y())
    (3.0, 7.0)
    """
    horizontal_scale, vertical_scale, slant_angle = normalize_text_transform(
        horizontal_scale, vertical_scale, slant_angle
    )
    k = -math.tan(math.radians(slant_angle))
    dx = point.x() - pivot.x()
    dy = point.y() - pivot.y()
    return QPointF(
        pivot.x() + horizontal_scale * dx + k * vertical_scale * dy,
        pivot.y() + vertical_scale * dy,
    )


def text_transform_matrix(
    horizontal_scale: float,
    vertical_scale: float,
    slant_angle: float,
    pivot: QPointF,
) -> QTransform:
    """Build the sole item-local affine matrix used for visual text geometry.

    >>> matrix = text_transform_matrix(2, 3, 0, QPointF(1, 1))
    >>> mapped = matrix.map(QPointF(2, 3))
    >>> (mapped.x(), mapped.y())
    (3.0, 7.0)
    """
    horizontal_scale, vertical_scale, slant_angle = normalize_text_transform(
        horizontal_scale, vertical_scale, slant_angle
    )
    k = -math.tan(math.radians(slant_angle))
    px, py = pivot.x(), pivot.y()
    return QTransform(
        horizontal_scale,
        0.0,
        k * vertical_scale,
        vertical_scale,
        px - horizontal_scale * px - k * vertical_scale * py,
        py - vertical_scale * py,
    )


def rect_polygon(rect: QRectF) -> QPolygonF:
    """Return a rectangle's four distinct corners in clockwise order."""
    return QPolygonF(
        [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
    )


def mapped_rect_polygon(rect: QRectF, transform: QTransform) -> QPolygonF:
    """Map all corners, preserving shear instead of collapsing to ``mapRect``."""
    return transform.map(rect_polygon(rect))

"""Pure helpers for the item-local post-layout *box* transform.

Glyph-local slant is deliberately rendered from shaped glyph runs and never
enters the matrix in this module.
"""

import math
from numbers import Real
from typing import Optional

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPolygonF, QTransform

from ballontranslator.utils.fontformat import TextTransform, normalize_text_transform


def _text_transform_coefficients(
    horizontal_scale: float,
    vertical_scale: float,
    slant_angle: float,
):
    """Return canonical scale and the Box Slant shear coefficient."""
    transform = normalize_text_transform(
        horizontal_scale, vertical_scale, slant_angle, 0.0
    )
    return (
        transform.horizontal_scale,
        transform.vertical_scale,
        -math.tan(math.radians(transform.slant_angle)),
    )


def text_transform_matrix(
    horizontal_scale: float,
    vertical_scale: float,
    slant_angle: float,
    pivot: QPointF,
) -> QTransform:
    """Build the Box-only affine matrix used for visual item geometry.

    >>> matrix = text_transform_matrix(2, 3, 0, QPointF(1, 1))
    >>> mapped = matrix.map(QPointF(2, 3))
    >>> (mapped.x(), mapped.y())
    (3.0, 7.0)
    >>> extreme = text_transform_matrix(4, 4, 85, QPointF())
    >>> all(math.isfinite(value) for value in (
    ...     extreme.m11(), extreme.m12(), extreme.m21(), extreme.m22()
    ... ))
    True
    """
    horizontal_scale, vertical_scale, shear = _text_transform_coefficients(
        horizontal_scale, vertical_scale, slant_angle
    )
    px, py = pivot.x(), pivot.y()
    return QTransform(
        horizontal_scale,
        0.0,
        shear * vertical_scale,
        vertical_scale,
        px - horizontal_scale * px - shear * vertical_scale * py,
        py - vertical_scale * py,
    )


def _rotation_about_pivot_matrix(angle: float, pivot: QPointF) -> QTransform:
    """Return Qt's clockwise-positive rotation about ``pivot``."""
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    px, py = pivot.x(), pivot.y()
    return QTransform(
        cosine,
        sine,
        -sine,
        cosine,
        px - cosine * px + sine * py,
        py - sine * px - cosine * py,
    )


def _transform_is_finite(transform: QTransform) -> bool:
    """Return whether every coefficient of ``transform`` is finite."""
    return all(
        math.isfinite(value)
        for value in (
            transform.m11(),
            transform.m12(),
            transform.m13(),
            transform.m21(),
            transform.m22(),
            transform.m23(),
            transform.m31(),
            transform.m32(),
            transform.m33(),
        )
    )


def compensated_text_transform_matrix(
    horizontal_scale: float,
    vertical_scale: float,
    slant_angle: float,
    box_pivot: QPointF,
    rotation_angle: float,
    rotation_pivot: Optional[QPointF] = None,
) -> QTransform:
    """Build the base transform that makes Box transform precede rotation.

    A :class:`QGraphicsItem` applies its built-in rotation before its base
    ``transform()`` when mapping a point.  For canonical Box transform ``S``
    and built-in rotation ``R``, install ``C = R^-1 * S * R`` as the base
    transform.  Qt then composes ``R * C == S * R``, which maps points as
    ``R(S(point))``.  ``box_pivot`` and ``rotation_pivot`` may differ.

    >>> base = compensated_text_transform_matrix(
    ...     2, 1, 0, QPointF(5, 7), 90, QPointF(5, 7)
    ... )
    >>> # Built-in rotation maps (6, 7) to (5, 8) before ``base`` is applied.
    >>> mapped = base.map(QPointF(5, 8))
    >>> (round(mapped.x(), 12), round(mapped.y(), 12))
    (5.0, 9.0)
    """
    if isinstance(rotation_angle, bool) or not isinstance(rotation_angle, Real):
        raise ValueError("rotation angle must be a finite number")
    rotation_angle = float(rotation_angle)
    if not math.isfinite(rotation_angle):
        raise ValueError("rotation angle must be a finite number")

    rotation_pivot = box_pivot if rotation_pivot is None else rotation_pivot
    for name, pivot in (
        ("box pivot", box_pivot),
        ("rotation pivot", rotation_pivot),
    ):
        if not math.isfinite(pivot.x()) or not math.isfinite(pivot.y()):
            raise ValueError(f"{name} coordinates must be finite numbers")

    box_transform = text_transform_matrix(
        horizontal_scale,
        vertical_scale,
        slant_angle,
        box_pivot,
    )
    if not _transform_is_finite(box_transform):
        raise ValueError("Box transform coefficients must be finite numbers")

    # Preserve exact canonical matrices for the common neutral paths.  Besides
    # avoiding trigonometric residue, this keeps identity/cache checks exact.
    same_pivot = (
        box_pivot.x() == rotation_pivot.x()
        and box_pivot.y() == rotation_pivot.y()
    )
    isotropic_box = (
        box_transform.m11() == box_transform.m22()
        and box_transform.m12() == 0.0
        and box_transform.m21() == 0.0
    )
    if (
        box_transform.isIdentity()
        or math.fmod(rotation_angle, 360.0) == 0.0
        or (same_pivot and isotropic_box)
    ):
        return box_transform

    rotation = _rotation_about_pivot_matrix(rotation_angle, rotation_pivot)
    if not _transform_is_finite(rotation):
        raise ValueError("rotation transform must be finite and invertible")
    inverse_rotation, rotation_is_invertible = rotation.inverted()
    if not rotation_is_invertible or not _transform_is_finite(inverse_rotation):
        raise ValueError("rotation transform must be finite and invertible")

    compensated = inverse_rotation * box_transform * rotation
    if not _transform_is_finite(compensated):
        raise ValueError("compensated Box transform must be finite and invertible")
    _, compensated_is_invertible = compensated.inverted()
    if not compensated_is_invertible:
        raise ValueError("compensated Box transform must be finite and invertible")
    return compensated


def rect_polygon(rect: QRectF) -> QPolygonF:
    """Return a rectangle's four distinct corners in clockwise order."""
    return QPolygonF(
        [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
    )


class TextTransformStrategy:
    """Rendering/geometry boundary implemented by each transform variant."""

    transform_type = 'base'

    def compensated_matrix(
        self,
        transform: TextTransform,
        box_pivot: QPointF,
        rotation_angle: float,
        rotation_pivot: QPointF,
    ) -> QTransform:
        raise NotImplementedError

    def visual_polygon(self, item, logical_rect: QRectF) -> QPolygonF:
        return QPolygonF(
            [item.mapToScene(point) for point in rect_polygon(logical_rect)]
        )

    def visual_is_neutral(self, item) -> bool:
        return item.transform().isIdentity()

    def apply_layout(
        self,
        item,
        transform: TextTransform,
        persistent_cache: bool = True,
    ) -> bool:
        """Apply variant-specific layout paint state."""
        return False

    def deactivate_layout(
        self,
        item,
        transform: TextTransform,
        persistent_cache: bool = True,
    ) -> bool:
        """Remove layout state owned by this strategy before a type switch."""
        return False

    def initialize_layout(
        self,
        item,
        transform: TextTransform,
        persistent_cache: bool = True,
    ) -> bool:
        """Install variant state into a newly attached text layout."""
        return False

    def requires_no_cache(self, transform: TextTransform) -> bool:
        return False

    def requires_custom_resize(self, transform: TextTransform) -> bool:
        return False


class NoTextTransformStrategy(TextTransformStrategy):
    """Identity geometry and layout for the explicit no-effect variant."""

    transform_type = 'none'

    def compensated_matrix(
        self,
        transform: TextTransform,
        box_pivot: QPointF,
        rotation_angle: float,
        rotation_pivot: QPointF,
    ) -> QTransform:
        return QTransform()


class SlantTextTransformStrategy(TextTransformStrategy):
    """Current affine box and glyph-slant implementation."""

    transform_type = 'slant'

    def __init__(self, layout_renderer_factory) -> None:
        self.layout_renderer_factory = layout_renderer_factory

    def compensated_matrix(
        self,
        transform: TextTransform,
        box_pivot: QPointF,
        rotation_angle: float,
        rotation_pivot: QPointF,
    ) -> QTransform:
        return compensated_text_transform_matrix(
            transform.horizontal_scale,
            transform.vertical_scale,
            transform.slant_angle,
            box_pivot,
            rotation_angle,
            rotation_pivot,
        )

    def visual_is_neutral(self, item) -> bool:
        return (
            super().visual_is_neutral(item)
            and not item.geometry_controller.has_layout_distortion()
        )

    def apply_layout(
        self,
        item,
        transform: TextTransform,
        persistent_cache: bool = True,
    ) -> bool:
        if item.layout is None:
            return False
        if transform.glyph_slant_angle == 0.0:
            return item.geometry_controller.detach_layout_renderer()
        renderer = item.geometry_controller.attach_layout_renderer(
            self.transform_type,
            self.layout_renderer_factory,
        )
        return renderer.apply(transform, persistent_cache)

    def deactivate_layout(
        self,
        item,
        transform: TextTransform,
        persistent_cache: bool = True,
    ) -> bool:
        if item.layout is None:
            return False
        return item.geometry_controller.detach_layout_renderer()

    def initialize_layout(
        self,
        item,
        transform: TextTransform,
        persistent_cache: bool = True,
    ) -> bool:
        if item.layout is None:
            return False
        if transform.glyph_slant_angle == 0.0:
            return item.geometry_controller.detach_layout_renderer()
        renderer = item.geometry_controller.attach_layout_renderer(
            self.transform_type,
            self.layout_renderer_factory,
        )
        return renderer.apply(transform, persistent_cache)

    def requires_no_cache(self, transform: TextTransform) -> bool:
        return (
            transform.horizontal_scale != 1.0
            or transform.vertical_scale != 1.0
            or transform.slant_angle != 0.0
        )

    def requires_custom_resize(self, transform: TextTransform) -> bool:
        return self.requires_no_cache(transform)

"""Pure helpers for the item-local post-layout *box* transform.

Glyph-local slant is deliberately rendered from shaped glyph runs and never
enters the matrix in this module.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Sequence

import numpy as np
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPainterPath, QPolygonF, QTransform

from ballontranslator.utils.fontformat import (
    CurvatureTextTransform,
    GridTextTransform,
    PerspectiveTextTransform,
    SlantTextTransform,
    TextTransformStack,
    normalize_text_transform,
)
from .text_effects.curvature import CurvatureMapper
from .text_effects.grid import GridMapper


def _text_transform_coefficients(
    horizontal_scale: float,
    vertical_scale: float,
    slant_angle: float,
):
    """Return canonical scale and the Box Slant shear coefficient."""
    transform = normalize_text_transform(
        horizontal_scale, vertical_scale, slant_angle
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


def compensated_box_transform_matrix(
    box_transform: QTransform,
    box_pivot: QPointF,
    rotation_angle: float,
    rotation_pivot: Optional[QPointF] = None,
) -> QTransform:
    """Build a Qt base transform that makes a box transform precede rotation.

    A :class:`QGraphicsItem` applies its built-in rotation before its base
    ``transform()`` when mapping a point.  For canonical Box transform ``S``
    and built-in rotation ``R``, install ``C = R^-1 * S * R`` as the base
    transform.  Qt then composes ``R * C == S * R``, which maps points as
    ``R(S(point))``.  ``box_pivot`` and ``rotation_pivot`` may differ.

    >>> box = text_transform_matrix(2, 1, 0, QPointF(5, 7))
    >>> base = compensated_box_transform_matrix(
    ...     box, QPointF(5, 7), 90, QPointF(5, 7)
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


def perspective_transform_matrix(
    transform: PerspectiveTextTransform,
    rect: QRectF,
) -> QTransform:
    """Return a centered, finite projective transform for ``rect``.

    Direction is clockwise in screen coordinates. Its L1-normalized vector
    keeps the homogeneous denominator at or above ``1 - strength``.

    >>> matrix = perspective_transform_matrix(
    ...     PerspectiveTextTransform(0.5, 0.0), QRectF(0, 0, 100, 50)
    ... )
    >>> mapped = matrix.map(QPointF(50, 25))
    >>> (mapped.x(), mapped.y())
    (50.0, 25.0)
    """
    transform = transform.normalized()
    if transform.strength == 0.0:
        return QTransform()
    if rect.width() <= 0.0 or rect.height() <= 0.0:
        raise ValueError('perspective rectangle must have positive dimensions')

    radians = math.radians(transform.direction)
    direction_x = math.cos(radians)
    direction_y = math.sin(radians)
    direction_length = abs(direction_x) + abs(direction_y)
    direction_x /= direction_length
    direction_y /= direction_length

    center = rect.center()
    half_width = rect.width() / 2.0
    half_height = rect.height() / 2.0
    projective_x = transform.strength * direction_x / half_width
    projective_y = transform.strength * direction_y / half_height
    denominator_offset = (
        1.0
        - projective_x * center.x()
        - projective_y * center.y()
    )

    matrix = QTransform(
        1.0 + center.x() * projective_x,
        center.y() * projective_x,
        projective_x,
        center.x() * projective_y,
        1.0 + center.y() * projective_y,
        projective_y,
        center.x() * denominator_offset - center.x(),
        center.y() * denominator_offset - center.y(),
        denominator_offset,
    )
    if not _transform_is_finite(matrix):
        raise ValueError('perspective transform must be finite and invertible')
    _, invertible = matrix.inverted()
    if not invertible:
        raise ValueError('perspective transform must be finite and invertible')
    return matrix


def rect_polygon(rect: QRectF) -> QPolygonF:
    """Return a rectangle's four distinct corners in clockwise order."""
    return QPolygonF(
        [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
    )


@dataclass(frozen=True)
class TransformStageContext:
    """Geometry presented to one registered transform stage.

    >>> context = TransformStageContext(
    ...     QRectF(0, 0, 10, 5), QRectF(-1, -1, 12, 7), False
    ... )
    >>> context.source_bounds.width()
    12.0
    """

    logical_bounds: QRectF
    source_bounds: QRectF
    vertical: bool


@dataclass(frozen=True)
class CompiledTransformStage:
    """Runtime mapper and input geometry for one persisted stack entry."""

    stack_index: int
    transform: object
    context: TransformStageContext
    mapper: object = None


def slant_transform_stage(
    transform: SlantTextTransform,
    context: TransformStageContext,
) -> QTransform:
    """Build one affine Slant stack stage."""
    return text_transform_matrix(
        transform.horizontal_scale,
        transform.vertical_scale,
        transform.slant_angle,
        context.logical_bounds.center(),
    )


def perspective_transform_stage(
    transform: PerspectiveTextTransform,
    context: TransformStageContext,
) -> QTransform:
    """Build one native projective stack stage over its current surface."""
    return perspective_transform_matrix(transform, context.source_bounds)


def curvature_transform_stage(
    transform: CurvatureTextTransform,
    context: TransformStageContext,
):
    """Build one nonlinear stage without coupling the compiler to its type."""
    return CurvatureMapper(
        context.logical_bounds,
        context.source_bounds,
        context.vertical,
        transform.curvature,
    )


def grid_transform_stage(
    transform: GridTextTransform,
    context: TransformStageContext,
):
    """Build one normalized free-form deformation stage."""
    return GridMapper(
        context.logical_bounds,
        context.source_bounds,
        transform,
    )


class MatrixTransformMapper:
    """Expose a finite ``QTransform`` through the composite mapper contract."""

    def __init__(self, matrix: QTransform) -> None:
        self.matrix = QTransform(matrix)
        self.inverse, invertible = self.matrix.inverted()
        if not invertible:
            raise ValueError('transform matrix must be finite and invertible')

    @property
    def geometry_key(self):
        matrix = self.matrix
        return (
            type(self),
            matrix.m11(), matrix.m12(), matrix.m13(),
            matrix.m21(), matrix.m22(), matrix.m23(),
            matrix.m31(), matrix.m32(), matrix.m33(),
        )

    def forward_point(self, source: QPointF) -> QPointF:
        return self.matrix.map(QPointF(source))

    def forward_arrays(self, source_x, source_y):
        source_x = np.asarray(source_x, dtype=np.float64)
        source_y = np.asarray(source_y, dtype=np.float64)
        matrix = self.matrix
        denominator = (
            matrix.m13() * source_x
            + matrix.m23() * source_y
            + matrix.m33()
        )
        return (
            (
                matrix.m11() * source_x
                + matrix.m21() * source_y
                + matrix.m31()
            ) / denominator,
            (
                matrix.m12() * source_x
                + matrix.m22() * source_y
                + matrix.m32()
            ) / denominator,
        )

    def inverse_point(
        self,
        visual: QPointF,
        previous_source: QPointF = None,
        *,
        extrapolate: bool = False,
    ) -> QPointF:
        return self.inverse.map(QPointF(visual))

    def inverse_arrays(self, visual_x, visual_y, *, return_valid=False):
        matrix = self.inverse
        denominator = (
            matrix.m13() * visual_x
            + matrix.m23() * visual_y
            + matrix.m33()
        )
        valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
        safe_denominator = np.where(valid, denominator, 1.0)
        source_x = (
            matrix.m11() * visual_x
            + matrix.m21() * visual_y
            + matrix.m31()
        ) / safe_denominator
        source_y = (
            matrix.m12() * visual_x
            + matrix.m22() * visual_y
            + matrix.m32()
        ) / safe_denominator
        valid &= np.isfinite(source_x) & np.isfinite(source_y)
        if return_valid:
            return source_x, source_y, valid
        return source_x, source_y

    def visual_bounds(self, source_rect: QRectF) -> QRectF:
        return self.matrix.map(rect_polygon(source_rect)).boundingRect()


def _point_segment_distance(point: QPointF, start: QPointF, end: QPointF) -> float:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(point.x() - start.x(), point.y() - start.y())
    ratio = (
        (point.x() - start.x()) * dx
        + (point.y() - start.y()) * dy
    ) / length_squared
    ratio = min(max(ratio, 0.0), 1.0)
    closest = QPointF(start.x() + ratio * dx, start.y() + ratio * dy)
    return math.hypot(point.x() - closest.x(), point.y() - closest.y())


class CompositeTextTransformMapper:
    """Compose matrix and nonlinear stages behind one mapping boundary.

    Painting inverse-samples this mapper once, even when it contains several
    nonlinear stages.

    >>> mapper = CompositeTextTransformMapper(
    ...     (MatrixTransformMapper(QTransform().scale(2, 1)),),
    ...     QRectF(0, 0, 10, 5),
    ...     QRectF(0, 0, 10, 5),
    ...     False,
    ... )
    >>> mapped = mapper.forward_point(QPointF(3, 2))
    >>> (mapped.x(), mapped.y())
    (6.0, 2.0)
    """

    OUTLINE_TOLERANCE = 0.25
    OUTLINE_MAX_DEPTH = 9

    def __init__(
        self,
        stages: Sequence,
        logical_rect: QRectF,
        source_rect: QRectF,
        vertical: bool,
    ) -> None:
        self.stages = tuple(stages)
        self.logical_rect = QRectF(logical_rect)
        self.source_rect = QRectF(source_rect)
        self.vertical = bool(vertical)
        self._rect_path_cache = {}
        self._visual_bounds_cache = {}

    @property
    def geometry_key(self):
        return (
            type(self),
            tuple(stage.geometry_key for stage in self.stages),
            self.vertical,
            self.logical_rect.x(),
            self.logical_rect.y(),
            self.logical_rect.width(),
            self.logical_rect.height(),
            self.source_rect.x(),
            self.source_rect.y(),
            self.source_rect.width(),
            self.source_rect.height(),
        )

    def forward_point(self, source: QPointF) -> QPointF:
        point = QPointF(source)
        for stage in self.stages:
            point = stage.forward_point(point)
        return point

    def forward_arrays(self, source_x, source_y):
        visual_x, visual_y = source_x, source_y
        for stage in self.stages:
            visual_x, visual_y = stage.forward_arrays(
                visual_x, visual_y
            )
        return visual_x, visual_y

    def _previous_stage_sources(self, previous_source: QPointF):
        if previous_source is None:
            return None
        points = []
        point = QPointF(previous_source)
        for stage in self.stages:
            points.append(QPointF(point))
            point = stage.forward_point(point)
        return points

    def inverse_point(
        self,
        visual: QPointF,
        previous_source: QPointF = None,
        *,
        extrapolate: bool = False,
    ) -> QPointF:
        point = QPointF(visual)
        previous = self._previous_stage_sources(previous_source)
        for index in range(len(self.stages) - 1, -1, -1):
            point = self.stages[index].inverse_point(
                point,
                None if previous is None else previous[index],
                extrapolate=extrapolate,
            )
        return point

    def inverse_arrays(self, visual_x, visual_y, *, return_valid=False):
        source_x, source_y = visual_x, visual_y
        valid = np.ones_like(visual_x, dtype=bool)
        for stage in reversed(self.stages):
            source_x, source_y, stage_valid = stage.inverse_arrays(
                source_x,
                source_y,
                return_valid=True,
            )
            valid &= stage_valid
        if return_valid:
            return source_x, source_y, valid
        return source_x, source_y

    def _append_mapped_edge(
        self,
        points: list[QPointF],
        source_start: QPointF,
        source_end: QPointF,
        mapped_start: QPointF,
        mapped_end: QPointF,
        depth: int,
    ) -> None:
        source_quarter = source_start * 0.75 + source_end * 0.25
        source_mid = (source_start + source_end) / 2.0
        source_three_quarter = source_start * 0.25 + source_end * 0.75
        mapped_quarter = self.forward_point(source_quarter)
        mapped_mid = self.forward_point(source_mid)
        mapped_three_quarter = self.forward_point(source_three_quarter)
        if (
            depth >= self.OUTLINE_MAX_DEPTH
            or max(
                _point_segment_distance(
                    point, mapped_start, mapped_end
                )
                for point in (
                    mapped_quarter,
                    mapped_mid,
                    mapped_three_quarter,
                )
            ) <= self.OUTLINE_TOLERANCE
        ):
            points.append(mapped_end)
            return
        self._append_mapped_edge(
            points,
            source_start,
            source_mid,
            mapped_start,
            mapped_mid,
            depth + 1,
        )
        self._append_mapped_edge(
            points,
            source_mid,
            source_end,
            mapped_mid,
            mapped_end,
            depth + 1,
        )

    def map_rect_path(self, rect: QRectF) -> QPainterPath:
        rect = QRectF(rect)
        cacheable = (
            rect == self.logical_rect
            or rect == self.source_rect
        )
        cache_key = (
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
        )
        if cacheable:
            cached = self._rect_path_cache.get(cache_key)
            if cached is not None:
                return QPainterPath(cached)

        if len(self.stages) == 1 and hasattr(
            self.stages[0], 'map_rect_path'
        ):
            path = self.stages[0].map_rect_path(rect)
            if cacheable:
                self._rect_path_cache[cache_key] = QPainterPath(path)
            return path

        corners = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft(),
        ]
        points = [self.forward_point(corners[0])]
        for index, source_start in enumerate(corners):
            source_end = corners[(index + 1) % len(corners)]
            self._append_mapped_edge(
                points,
                source_start,
                source_end,
                self.forward_point(source_start),
                self.forward_point(source_end),
                0,
            )
        path = QPainterPath()
        if points:
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        if cacheable:
            self._rect_path_cache[cache_key] = QPainterPath(path)
        return path

    def visual_bounds(self, source_rect: QRectF = None) -> QRectF:
        rect = self.source_rect if source_rect is None else source_rect
        cache_key = (
            rect.x(), rect.y(), rect.width(), rect.height()
        )
        cached = self._visual_bounds_cache.get(cache_key)
        if cached is not None:
            return QRectF(cached)
        bounds = QRectF(rect)
        for stage in self.stages:
            bounds = stage.visual_bounds(bounds)
        self._visual_bounds_cache[cache_key] = QRectF(bounds)
        return bounds

    def local_tangent(self, source: QPointF) -> QPointF:
        flow = QPointF(0.0, 1.0) if self.vertical else QPointF(1.0, 0.0)
        start = self.forward_point(source)
        tangent = self.forward_point(QPointF(source) + flow) - start
        length = math.hypot(tangent.x(), tangent.y())
        return tangent / length if length else flow


@dataclass(frozen=True)
class CompiledTextTransform:
    """One native matrix or one final surface mapper for an ordered stack."""

    stack: TextTransformStack
    native_matrix: QTransform
    surface_mapper: Optional[CompositeTextTransformMapper] = None
    stages: tuple[CompiledTransformStage, ...] = ()

    @property
    def geometry_key(self):
        if self.surface_mapper is not None:
            return ('surface', self.surface_mapper.geometry_key)
        matrix = self.native_matrix
        return (
            'matrix',
            matrix.m11(), matrix.m12(), matrix.m13(),
            matrix.m21(), matrix.m22(), matrix.m23(),
            matrix.m31(), matrix.m32(), matrix.m33(),
        )

    @property
    def is_identity(self) -> bool:
        return self.surface_mapper is None and self.native_matrix.isIdentity()

    @property
    def has_projective_mapping(self) -> bool:
        return (
            self.surface_mapper is None
            and not self.native_matrix.isAffine()
        )

    @property
    def needs_local_handle_frames(self) -> bool:
        return self.surface_mapper is not None or self.has_projective_mapping

    @property
    def requires_no_cache(self) -> bool:
        return not self.is_identity

    @property
    def requires_custom_resize(self) -> bool:
        return not self.is_identity

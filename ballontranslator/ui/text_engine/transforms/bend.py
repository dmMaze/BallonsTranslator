"""Analytic circular text-surface mapping."""

from __future__ import annotations

import math
from typing import Iterator, Optional, Tuple, Union

import numpy as np
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPainterPath


MAX_BEND_SWEEP = math.radians(350.0)
BEND_RADIAL_GUARD_RATIO = 0.02
BEND_OUTLINE_TOLERANCE = 0.25
BEND_OUTLINE_MAX_SEGMENTS = 512


def _rect_edge_samples(rect: QRectF, segments: int) -> Iterator[QPointF]:
    """Yield one clockwise sampled rectangle boundary without duplicates."""
    left, right = rect.left(), rect.right()
    top, bottom = rect.top(), rect.bottom()
    for index in range(segments):
        ratio = index / segments
        yield QPointF(left + rect.width() * ratio, top)
    for index in range(segments):
        ratio = index / segments
        yield QPointF(right, top + rect.height() * ratio)
    for index in range(segments):
        ratio = index / segments
        yield QPointF(right - rect.width() * ratio, bottom)
    for index in range(segments):
        ratio = index / segments
        yield QPointF(left, bottom - rect.height() * ratio)


class BendMapper:
    """Map a source rectangle to an invertible circular strip.

    The same analytic mapping is used by geometry, interaction, and raster
    sampling. The padded source extent may reduce the maximum logical sweep so
    effects never overlap across the near-closed seam.

    >>> mapper = BendMapper(
    ...     QRectF(0, 0, 100, 20), QRectF(0, 0, 100, 20), False, 0.5
    ... )
    >>> point = QPointF(25, 8)
    >>> restored = mapper.inverse_point(mapper.forward_point(point))
    >>> (round(restored.x(), 6), round(restored.y(), 6))
    (25.0, 8.0)
    """

    def __init__(
        self,
        logical_rect: QRectF,
        source_rect: QRectF,
        vertical: bool,
        bend: float,
    ) -> None:
        self.logical_rect = QRectF(logical_rect)
        self.source_rect = QRectF(source_rect)
        self.vertical = bool(vertical)
        self.bend = float(bend)
        self.direction = 1.0 if bend >= 0.0 else -1.0
        self.center = self.logical_rect.center()
        self.translation = QPointF()
        self.cross_scale = 1.0
        self.radius = math.inf
        self.sweep = 0.0
        self.source_angle_limit = 0.0

        if bend == 0.0:
            return
        flow_length = (
            self.logical_rect.height()
            if self.vertical
            else self.logical_rect.width()
        )
        source_flow_length = (
            self.source_rect.height()
            if self.vertical
            else self.source_rect.width()
        )
        if flow_length <= 0.0 or source_flow_length <= 0.0:
            raise ValueError('bend rectangles must have positive dimensions')

        requested_sweep = abs(bend) * MAX_BEND_SWEEP
        padded_sweep_limit = (
            MAX_BEND_SWEEP * flow_length / source_flow_length
        )
        self.sweep = min(requested_sweep, padded_sweep_limit)
        self.radius = flow_length / self.sweep
        self.source_angle_limit = (
            self.sweep * source_flow_length / flow_length / 2.0
        )

        cross_min, cross_max = self._cross_range(self.source_rect)
        radial_extent = max(
            self.direction * cross_min,
            self.direction * cross_max,
            0.0,
        )
        if radial_extent > 0.0:
            available_radius = self.radius * (
                1.0 - BEND_RADIAL_GUARD_RATIO
            )
            self.cross_scale = min(1.0, available_radius / radial_extent)

        raw_logical_bounds = self._raw_mapped_path(
            self.logical_rect
        ).boundingRect()
        self.translation = (
            self.logical_rect.center() - raw_logical_bounds.center()
        )

    @property
    def is_identity(self) -> bool:
        return self.bend == 0.0

    @property
    def geometry_key(self) -> tuple:
        return (
            type(self),
            self.bend,
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

    def _flow_cross(self, point: QPointF) -> Tuple[float, float]:
        if self.vertical:
            return point.y() - self.center.y(), point.x() - self.center.x()
        return point.x() - self.center.x(), point.y() - self.center.y()

    def _from_flow_cross(self, flow: float, cross: float) -> QPointF:
        if self.vertical:
            return QPointF(self.center.x() + cross, self.center.y() + flow)
        return QPointF(self.center.x() + flow, self.center.y() + cross)

    def _cross_range(self, rect: QRectF) -> Tuple[float, float]:
        if self.vertical:
            return rect.left() - self.center.x(), rect.right() - self.center.x()
        return rect.top() - self.center.y(), rect.bottom() - self.center.y()

    def _raw_forward(self, source: QPointF) -> QPointF:
        flow, cross = self._flow_cross(source)
        flow_length = (
            self.logical_rect.height()
            if self.vertical
            else self.logical_rect.width()
        )
        angle = self.sweep * flow / flow_length
        radial = self.radius - self.direction * self.cross_scale * cross
        mapped_flow = radial * math.sin(angle)
        mapped_cross = self.direction * (
            self.radius - radial * math.cos(angle)
        )
        return self._from_flow_cross(mapped_flow, mapped_cross)

    def forward_point(self, source: QPointF) -> QPointF:
        source = QPointF(source)
        if self.is_identity:
            return source
        return self._raw_forward(source) + self.translation

    def forward_arrays(
        self, source_x: np.ndarray, source_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        source_x = np.asarray(source_x, dtype=np.float64)
        source_y = np.asarray(source_y, dtype=np.float64)
        if self.is_identity:
            return source_x, source_y
        if self.vertical:
            flow = source_y - self.center.y()
            cross = source_x - self.center.x()
            flow_length = self.logical_rect.height()
        else:
            flow = source_x - self.center.x()
            cross = source_y - self.center.y()
            flow_length = self.logical_rect.width()
        angle = self.sweep * flow / flow_length
        radial = self.radius - self.direction * self.cross_scale * cross
        mapped_flow = radial * np.sin(angle)
        mapped_cross = self.direction * (
            self.radius - radial * np.cos(angle)
        )
        if self.vertical:
            visual_x = self.center.x() + mapped_cross
            visual_y = self.center.y() + mapped_flow
        else:
            visual_x = self.center.x() + mapped_flow
            visual_y = self.center.y() + mapped_cross
        return (
            visual_x + self.translation.x(),
            visual_y + self.translation.y(),
        )

    def inverse_point(
        self,
        visual: QPointF,
        previous_source: Optional[QPointF] = None,
        *,
        extrapolate: bool = False,
    ) -> QPointF:
        visual = QPointF(visual)
        if self.is_identity:
            return visual
        raw = visual - self.translation
        mapped_flow, mapped_cross = self._flow_cross(raw)
        circle_cross = mapped_cross - self.direction * self.radius
        radial = math.hypot(mapped_flow, circle_cross)
        angle = math.atan2(
            mapped_flow,
            -self.direction * circle_cross,
        )
        flow_length = (
            self.logical_rect.height()
            if self.vertical
            else self.logical_rect.width()
        )
        if extrapolate and previous_source is not None:
            previous_flow, _ = self._flow_cross(previous_source)
            previous_angle = self.sweep * previous_flow / flow_length
            angle += round(
                (previous_angle - angle) / (2.0 * math.pi)
            ) * (2.0 * math.pi)
        elif (
            angle < -self.source_angle_limit
            or angle > self.source_angle_limit
        ):
            if previous_source is not None:
                previous_flow, _ = self._flow_cross(previous_source)
                angle = math.copysign(
                    self.source_angle_limit,
                    previous_flow if previous_flow != 0.0 else angle,
                )
            else:
                angle = min(
                    max(angle, -self.source_angle_limit),
                    self.source_angle_limit,
                )
        flow = flow_length * angle / self.sweep
        cross = (
            self.direction * (self.radius - radial) / self.cross_scale
        )
        return self._from_flow_cross(flow, cross)

    def inverse_arrays(
        self,
        visual_x: np.ndarray,
        visual_y: np.ndarray,
        *,
        return_valid: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Vectorized inverse used by the bounded raster sampler."""
        if self.is_identity:
            if return_valid:
                return visual_x, visual_y, np.ones_like(
                    visual_x, dtype=bool
                )
            return visual_x, visual_y
        raw_x = visual_x - self.translation.x()
        raw_y = visual_y - self.translation.y()
        if self.vertical:
            mapped_flow = raw_y - self.center.y()
            mapped_cross = raw_x - self.center.x()
        else:
            mapped_flow = raw_x - self.center.x()
            mapped_cross = raw_y - self.center.y()
        circle_cross = mapped_cross - self.direction * self.radius
        radial = np.hypot(mapped_flow, circle_cross)
        angle = np.arctan2(
            mapped_flow,
            -self.direction * circle_cross,
        )
        valid = (
            (angle >= -self.source_angle_limit)
            & (angle <= self.source_angle_limit)
        )
        angle = np.clip(
            angle, -self.source_angle_limit, self.source_angle_limit
        )
        flow_length = (
            self.logical_rect.height()
            if self.vertical
            else self.logical_rect.width()
        )
        flow = flow_length * angle / self.sweep
        cross = (
            self.direction * (self.radius - radial) / self.cross_scale
        )
        if self.vertical:
            source_x = self.center.x() + cross
            source_y = self.center.y() + flow
        else:
            source_x = self.center.x() + flow
            source_y = self.center.y() + cross
        if return_valid:
            return source_x, source_y, valid
        return source_x, source_y

    def _segment_count(self) -> int:
        full_angle = self.source_angle_limit * 2.0
        cross_min, cross_max = self._cross_range(self.source_rect)
        maximum_radius = max(
            self.radius - self.direction * self.cross_scale * cross_min,
            self.radius - self.direction * self.cross_scale * cross_max,
        )
        if maximum_radius <= BEND_OUTLINE_TOLERANCE:
            maximum_segment_angle = math.pi
        else:
            maximum_segment_angle = 2.0 * math.acos(
                max(
                    -1.0,
                    1.0 - BEND_OUTLINE_TOLERANCE / maximum_radius,
                )
            )
            if maximum_segment_angle == 0.0:
                maximum_segment_angle = math.sqrt(
                    8.0
                    * BEND_OUTLINE_TOLERANCE
                    / maximum_radius
                )
        return max(
            8,
            min(
                BEND_OUTLINE_MAX_SEGMENTS,
                math.ceil(full_angle / maximum_segment_angle),
            ),
        )

    def _raw_mapped_path(self, rect: QRectF) -> QPainterPath:
        points = [
            self._raw_forward(point)
            for point in _rect_edge_samples(rect, self._segment_count())
        ]
        path = QPainterPath()
        if points:
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        return path

    def map_rect_path(self, rect: QRectF) -> QPainterPath:
        if self.is_identity:
            path = QPainterPath()
            path.addRect(rect)
            return path
        path = self._raw_mapped_path(rect)
        path.translate(self.translation)
        return path

    def visual_bounds(
        self, source_rect: Optional[QRectF] = None
    ) -> QRectF:
        rect = self.source_rect if source_rect is None else source_rect
        return self.map_rect_path(rect).boundingRect()

    def local_tangent(self, source: QPointF) -> QPointF:
        """Return the mapped unit-flow tangent at ``source``."""
        if self.is_identity:
            return QPointF(0.0, 1.0) if self.vertical else QPointF(1.0, 0.0)
        flow, cross = self._flow_cross(source)
        flow_length = (
            self.logical_rect.height()
            if self.vertical
            else self.logical_rect.width()
        )
        angle = self.sweep * flow / flow_length
        radial = self.radius - self.direction * self.cross_scale * cross
        derivative = radial * self.sweep / flow_length
        mapped_flow = derivative * math.cos(angle)
        mapped_cross = self.direction * derivative * math.sin(angle)
        tangent = (
            QPointF(mapped_cross, mapped_flow)
            if self.vertical
            else QPointF(mapped_flow, mapped_cross)
        )
        length = math.hypot(tangent.x(), tangent.y())
        return tangent / length if length else QPointF(1.0, 0.0)

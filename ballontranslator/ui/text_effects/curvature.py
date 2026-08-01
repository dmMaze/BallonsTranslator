"""Analytic circular text-surface mapping."""

import math

import cv2
import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QPainter, QPainterPath, QPixmap
from qtpy.QtWidgets import QStyleOptionGraphicsItem

from ..misc import ndarray2pixmap, pixmap2ndarray
from .raster import EffectRasterAllocationError, plan_effect_raster


MAX_CURVATURE_SWEEP = math.radians(350.0)
CURVATURE_RADIAL_GUARD_RATIO = 0.02
CURVATURE_OUTLINE_TOLERANCE = 0.25
CURVATURE_OUTLINE_MAX_SEGMENTS = 512


def _rect_edge_samples(rect: QRectF, segments: int):
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


class CurvatureMapper:
    """Map a source rectangle to an invertible circular strip.

    The same analytic mapping is used by geometry, interaction, and raster
    sampling. The padded source extent may reduce the maximum logical sweep so
    effects never overlap across the near-closed seam.

    >>> mapper = CurvatureMapper(
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
        curvature: float,
    ) -> None:
        self.logical_rect = QRectF(logical_rect)
        self.source_rect = QRectF(source_rect)
        self.vertical = bool(vertical)
        self.curvature = float(curvature)
        self.direction = 1.0 if curvature >= 0.0 else -1.0
        self.center = self.logical_rect.center()
        self.translation = QPointF()
        self.cross_scale = 1.0
        self.radius = math.inf
        self.sweep = 0.0
        self.source_angle_limit = 0.0

        if curvature == 0.0:
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
            raise ValueError('curvature rectangles must have positive dimensions')

        requested_sweep = abs(curvature) * MAX_CURVATURE_SWEEP
        padded_sweep_limit = (
            MAX_CURVATURE_SWEEP * flow_length / source_flow_length
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
                1.0 - CURVATURE_RADIAL_GUARD_RATIO
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
        return self.curvature == 0.0

    @property
    def geometry_key(self):
        return (
            type(self),
            self.curvature,
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

    def _flow_cross(self, point: QPointF):
        if self.vertical:
            return point.y() - self.center.y(), point.x() - self.center.x()
        return point.x() - self.center.x(), point.y() - self.center.y()

    def _from_flow_cross(self, flow: float, cross: float) -> QPointF:
        if self.vertical:
            return QPointF(self.center.x() + cross, self.center.y() + flow)
        return QPointF(self.center.x() + flow, self.center.y() + cross)

    def _cross_range(self, rect: QRectF):
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

    def forward_arrays(self, source_x, source_y):
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
        previous_source: QPointF = None,
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

    def inverse_arrays(self, visual_x, visual_y, *, return_valid=False):
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
        if maximum_radius <= CURVATURE_OUTLINE_TOLERANCE:
            maximum_segment_angle = math.pi
        else:
            maximum_segment_angle = 2.0 * math.acos(
                max(
                    -1.0,
                    1.0 - CURVATURE_OUTLINE_TOLERANCE / maximum_radius,
                )
            )
            if maximum_segment_angle == 0.0:
                maximum_segment_angle = math.sqrt(
                    8.0
                    * CURVATURE_OUTLINE_TOLERANCE
                    / maximum_radius
                )
        return max(
            8,
            min(
                CURVATURE_OUTLINE_MAX_SEGMENTS,
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

    def visual_bounds(self, source_rect: QRectF = None) -> QRectF:
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


class NonlinearTextSurfaceRenderer:
    """Warp a complete source text composite through a visual mapper.

    Float remap coordinates are built in row bands and cached by geometry.
    Rendered pixels have a separate content key so text, selection, and IME
    changes can reuse the inverse map. Both caches remain item-local.

    >>> renderer = NonlinearTextSurfaceRenderer()
    >>> renderer.cached_pixmap is None
    True
    """

    REMAP_ROW_BAND = 64

    def __init__(self) -> None:
        self.cached_pixmap = None
        self.cached_key = None
        self.cached_remap = None
        self.cached_remap_key = None

    def invalidate_surface(self) -> None:
        self.cached_pixmap = None
        self.cached_key = None

    def release(self) -> None:
        self.invalidate_surface()
        self.cached_remap = None
        self.cached_remap_key = None

    @staticmethod
    def _device_scale(painter: QPainter) -> float:
        transform = painter.deviceTransform()
        x_scale = math.hypot(transform.m11(), transform.m12())
        y_scale = math.hypot(transform.m21(), transform.m22())
        return max(1.0, x_scale, y_scale)

    @staticmethod
    def _capture_source(
        source_rect: QRectF,
        scale: float,
        option,
        paint_source,
    ) -> QPixmap:
        width = max(1, math.ceil(source_rect.width() * scale))
        height = max(1, math.ceil(source_rect.height() * scale))
        pixmap = QPixmap(width, height)
        if pixmap.isNull():
            raise EffectRasterAllocationError(
                f'failed to allocate nonlinear text source {width}x{height}'
            )
        pixmap.fill(Qt.GlobalColor.transparent)
        source_option = QStyleOptionGraphicsItem(option)
        source_option.exposedRect = QRectF(source_rect)
        source_painter = QPainter(pixmap)
        try:
            source_painter.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.TextAntialiasing
                | QPainter.RenderHint.SmoothPixmapTransform
            )
            source_painter.scale(scale, scale)
            source_painter.translate(-source_rect.topLeft())
            paint_source(source_painter, source_option, None)
        finally:
            source_painter.end()
        return pixmap

    def _warp(
        self,
        source_pixmap: QPixmap,
        source_rect: QRectF,
        destination_rect: QRectF,
        mapper,
        scale: float,
    ) -> QPixmap:
        source = pixmap2ndarray(source_pixmap, keep_alpha=True)
        # Interpolate premultiplied color so transparent glyph edges do not
        # borrow RGB from fully transparent pixels.
        alpha = source[:, :, 3].astype(np.uint16)
        for channel_index in range(3):
            channel = source[:, :, channel_index].astype(np.uint16)
            channel *= alpha
            channel += 127
            channel //= 255
            source[:, :, channel_index] = channel.astype(np.uint8)
        del alpha
        output_width = max(1, math.ceil(destination_rect.width() * scale))
        output_height = max(1, math.ceil(destination_rect.height() * scale))
        remap_key = (
            mapper.geometry_key,
            source_rect.x(),
            source_rect.y(),
            source_rect.width(),
            source_rect.height(),
            destination_rect.x(),
            destination_rect.y(),
            destination_rect.width(),
            destination_rect.height(),
            scale,
            output_width,
            output_height,
        )
        remap = (
            self.cached_remap
            if self.cached_remap_key == remap_key
            else None
        )
        if remap is None:
            map_x = np.empty(
                (output_height, output_width), dtype=np.float32
            )
            map_y = np.empty(
                (output_height, output_width), dtype=np.float32
            )
            x = (
                destination_rect.left()
                + (np.arange(output_width, dtype=np.float32) + 0.5) / scale
            )
            for top in range(0, output_height, self.REMAP_ROW_BAND):
                bottom = min(output_height, top + self.REMAP_ROW_BAND)
                y = (
                    destination_rect.top()
                    + (
                        np.arange(top, bottom, dtype=np.float32) + 0.5
                    ) / scale
                )
                visual_x, visual_y = np.meshgrid(x, y)
                source_x, source_y, valid = mapper.inverse_arrays(
                    visual_x, visual_y, return_valid=True
                )
                band_x = (
                    (source_x - source_rect.left()) * scale - 0.5
                ).astype(np.float32, copy=False)
                band_y = (
                    (source_y - source_rect.top()) * scale - 0.5
                ).astype(np.float32, copy=False)
                band_x[~valid] = -1.0
                band_y[~valid] = -1.0
                map_x[top:bottom] = band_x
                map_y[top:bottom] = band_y
            remap = (map_x, map_y)
            self.cached_remap = remap
            self.cached_remap_key = remap_key
        output = cv2.remap(
            source,
            remap[0],
            remap[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        output_alpha = output[:, :, 3]
        nonzero = output_alpha > 0
        if np.any(nonzero):
            alpha_values = output_alpha[nonzero].astype(np.float32)
            for channel_index in range(3):
                channel = output[:, :, channel_index]
                values = channel[nonzero].astype(np.float32)
                values *= 255.0
                values /= alpha_values
                channel[nonzero] = np.clip(
                    np.rint(values), 0, 255
                ).astype(np.uint8)
        pixmap = ndarray2pixmap(output)
        if pixmap.isNull():
            raise EffectRasterAllocationError(
                f'failed to allocate nonlinear text output '
                f'{output_width}x{output_height}'
            )
        return pixmap

    def paint(
        self,
        painter: QPainter,
        option,
        mapper,
        source_rect: QRectF,
        cache_key,
        cache_allowed: bool,
        paint_source,
        maximum_scale: float = None,
    ) -> bool:
        destination_rect = mapper.visual_bounds(source_rect)
        requested_scale = self._device_scale(painter)
        if maximum_scale is not None:
            requested_scale = min(requested_scale, float(maximum_scale))
        source_plan = plan_effect_raster(
            source_rect.width(), source_rect.height(), requested_scale
        )
        destination_plan = plan_effect_raster(
            destination_rect.width(),
            destination_rect.height(),
            requested_scale,
        )
        if source_plan.mode != 'full' or destination_plan.mode != 'full':
            raise EffectRasterAllocationError(
                'nonlinear text surface exceeds bounded full-raster policy'
            )
        render_scale = min(source_plan.tier, destination_plan.tier)
        key = cache_key + (render_scale,)
        pixmap = (
            self.cached_pixmap
            if cache_allowed and self.cached_key == key
            else None
        )
        cache_hit = pixmap is not None
        if pixmap is None:
            source = self._capture_source(
                source_rect, render_scale, option, paint_source
            )
            pixmap = self._warp(
                source,
                source_rect,
                destination_rect,
                mapper,
                render_scale,
            )
            if cache_allowed:
                self.cached_pixmap = pixmap
                self.cached_key = key
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(
                destination_rect,
                pixmap,
                QRectF(pixmap.rect()),
            )
        finally:
            painter.restore()
        return cache_hit

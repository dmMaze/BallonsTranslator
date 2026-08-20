"""Rendering for complete text surfaces under nonlinear mappings."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import cv2
import numpy as np
from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QPainter, QPixmap
from qtpy.QtWidgets import QStyleOptionGraphicsItem, QWidget

from ...misc import ndarray2pixmap, pixmap2ndarray
from .raster import (
    EffectRasterAllocationError,
    plan_effect_raster,
    quality_raster_request,
)


PaintSource = Callable[
    [QPainter, QStyleOptionGraphicsItem, Optional[QWidget]], None
]


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
        option: QStyleOptionGraphicsItem,
        paint_source: PaintSource,
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
        mapper: Any,
        scale: float,
        interpolation: int,
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
            interpolation=interpolation,
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
        option: QStyleOptionGraphicsItem,
        mapper: Any,
        source_rect: QRectF,
        cache_key: tuple,
        cache_allowed: bool,
        paint_source: PaintSource,
        maximum_scale: Optional[float] = None,
        high_quality: bool = True,
    ) -> bool:
        destination_rect = mapper.visual_bounds(source_rect)
        requested_scale = self._device_scale(painter)
        if maximum_scale is not None:
            requested_scale = min(requested_scale, float(maximum_scale))
        raster_request = requested_scale
        if high_quality:
            raster_request = quality_raster_request(requested_scale)
        source_plan = plan_effect_raster(
            source_rect.width(), source_rect.height(), raster_request
        )
        destination_plan = plan_effect_raster(
            destination_rect.width(),
            destination_rect.height(),
            raster_request,
        )
        if source_plan.mode != 'full' or destination_plan.mode != 'full':
            raise EffectRasterAllocationError(
                'nonlinear text surface exceeds bounded full-raster policy'
            )
        render_scale = min(source_plan.tier, destination_plan.tier)
        interpolation = (
            cv2.INTER_CUBIC if high_quality else cv2.INTER_LINEAR
        )
        key = cache_key + (render_scale, interpolation)
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
                interpolation,
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

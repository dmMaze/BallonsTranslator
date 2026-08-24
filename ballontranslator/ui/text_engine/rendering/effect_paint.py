"""Rasterize immutable text-effect paints with shared visual semantics."""

import math

import numpy as np

from qtpy.QtCore import QRectF
from qtpy.QtGui import QImage, QPainter, QPalette

from ballontranslator.utils.text_effects import (
    EffectPaint,
    LinearGradientPaint,
    SolidPaint,
)

from ...misc import ndarray2pixmap


def colorize_effect_paint_rgba(
    paint: EffectPaint,
    rgba: np.ndarray,
    surface_rect: QRectF,
    logical_rect: QRectF,
    render_scale: float,
) -> np.ndarray:
    """Colorize an existing straight-RGBA coverage surface in place.

    Stop RGB and opacity interpolate independently. Chunking bounds temporary
    storage and lets the effect renderer reuse its positioned Stroke surface.

    >>> pixels = np.full((1, 2, 4), 255, dtype=np.uint8)
    >>> colorize_effect_paint_rgba(
    ...     SolidPaint((1, 2, 3)), pixels, QRectF(), QRectF(), 1.0
    ... )[0, 0].tolist()
    [1, 2, 3, 255]
    """
    if (
        not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.ndim != 3
        or rgba.shape[2] != 4
    ):
        raise TypeError('effect paint target must be a uint8 RGBA array')
    if not math.isfinite(render_scale) or render_scale <= 0.0:
        raise ValueError('effect paint raster scale must be positive')
    if isinstance(paint, SolidPaint):
        rgba[..., :3] = paint.color
        return rgba
    if not isinstance(paint, LinearGradientPaint):
        raise TypeError('effect paint raster requires EffectPaint')

    height, width = rgba.shape[:2]
    center = logical_rect.center()
    length = max(logical_rect.width(), logical_rect.height()) * paint.scale
    if length <= 0.0:
        rgba[..., :3] = paint.stops[0].color
        alpha = int(round(paint.stops[0].opacity * 255))
        product = rgba[..., 3].astype(np.uint16)
        product *= alpha
        product += 127
        product //= 255
        rgba[..., 3] = product.astype(np.uint8)
        return rgba

    radians = math.radians(paint.angle)
    direction_x = math.cos(radians)
    direction_y = math.sin(radians)
    x = (
        surface_rect.left()
        + (np.arange(width, dtype=np.float32) + 0.5) / render_scale
        - center.x()
    )
    positions = np.asarray(
        [stop.position for stop in paint.stops], dtype=np.float32
    )
    colors = np.asarray(
        [stop.color for stop in paint.stops], dtype=np.float32
    )
    opacities = np.asarray(
        [stop.opacity * 255.0 for stop in paint.stops], dtype=np.float32
    )
    for row_start in range(0, height, 256):
        row_end = min(height, row_start + 256)
        y = (
            surface_rect.top()
            + (np.arange(row_start, row_end, dtype=np.float32) + 0.5)
            / render_scale
            - center.y()
        )
        parameter = (
            0.5
            + (y[:, None] * direction_y + x[None, :] * direction_x)
            / length
        )
        np.clip(parameter, 0.0, 1.0, out=parameter)
        right = np.ones(parameter.shape, dtype=np.uint8)
        # Advancing on equality preserves equal-position hard transitions.
        for index in range(1, len(paint.stops) - 1):
            right[parameter >= positions[index]] = index + 1
        target = rgba[row_start:row_end]
        paint_alpha = np.empty(parameter.shape, dtype=np.uint8)
        for right_index in range(1, len(paint.stops)):
            selected = right == right_index
            if not np.any(selected):
                continue
            left_index = right_index - 1
            span = positions[right_index] - positions[left_index]
            if span <= 0.0:
                ratio = (
                    parameter[selected] >= positions[right_index]
                ).astype(np.float32)
            else:
                ratio = (
                    parameter[selected] - positions[left_index]
                ) / span
                np.clip(ratio, 0.0, 1.0, out=ratio)
            for channel in range(3):
                values = colors[left_index, channel] + (
                    colors[right_index, channel]
                    - colors[left_index, channel]
                ) * ratio
                target[..., channel][selected] = np.rint(values).astype(
                    np.uint8
                )
            alpha = opacities[left_index] + (
                opacities[right_index] - opacities[left_index]
            ) * ratio
            paint_alpha[selected] = np.rint(alpha).astype(np.uint8)
        product = target[..., 3].astype(np.uint16)
        product *= paint_alpha.astype(np.uint16)
        product += 127
        product //= 255
        target[..., 3] = product.astype(np.uint8)
    return rgba


def rasterize_effect_paint(
    paint: EffectPaint,
    surface_rect: QRectF,
    logical_rect: QRectF,
    render_scale: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Return deterministic straight RGBA in absolute item coordinates.

    Evaluating a Qt gradient independently in overlapping tile painters can
    differ by a few channel values. This bounded raster helper makes the same
    logical pixel receive the same paint in full and tiled surfaces.

    >>> rasterize_effect_paint(
    ...     SolidPaint((1, 2, 3)), QRectF(), QRectF(), 1.0, 2, 1
    ... )[0, 0].tolist()
    [1, 2, 3, 255]
    """
    if width < 0 or height < 0:
        raise ValueError('effect paint raster dimensions must be nonnegative')
    if not math.isfinite(render_scale) or render_scale <= 0.0:
        raise ValueError('effect paint raster scale must be positive')
    result = np.empty((height, width, 4), dtype=np.uint8)
    result[..., 3] = 255
    return colorize_effect_paint_rgba(
        paint, result, surface_rect, logical_rect, render_scale
    )


def effect_paint_preview_image(
    paint: EffectPaint,
    logical_rect: QRectF,
    render_scale: float,
) -> QImage:
    """Return a small deterministic preview image for an effect paint.

    >>> callable(effect_paint_preview_image)
    True
    """
    width = max(1, int(round(logical_rect.width() * render_scale)))
    height = max(1, int(round(logical_rect.height() * render_scale)))
    rgba = rasterize_effect_paint(
        paint, logical_rect, logical_rect, render_scale, width, height
    )
    # Copy detaches QImage from the temporary NumPy buffer while retaining
    # straight RGBA values that QPixmap would premultiply and round.
    return ndarray2pixmap(rgba, return_qimg=True).copy()


def paint_effect_paint_preview(
    painter: QPainter,
    rect: QRectF,
    paint: EffectPaint,
    palette: QPalette,
    render_scale: float,
) -> None:
    """Paint a palette checkerboard and deterministic alpha-aware strip.

    >>> callable(paint_effect_paint_preview)
    True
    """
    painter.save()
    painter.setClipRect(rect)
    checker_size = 5.0
    colors = (palette.base().color(), palette.mid().color())
    rows = int(math.ceil(rect.height() / checker_size))
    columns = int(math.ceil(rect.width() / checker_size))
    for row in range(rows):
        for column in range(columns):
            painter.fillRect(
                QRectF(
                    rect.left() + column * checker_size,
                    rect.top() + row * checker_size,
                    checker_size,
                    checker_size,
                ),
                colors[(row + column) % 2],
            )
    image = effect_paint_preview_image(paint, rect, render_scale)
    painter.drawImage(rect, image, QRectF(image.rect()))
    painter.restore()

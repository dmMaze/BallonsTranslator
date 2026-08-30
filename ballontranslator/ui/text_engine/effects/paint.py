"""Rasterize immutable text-effect paints with shared visual semantics."""

import math
import threading

import cv2
import numpy as np

from qtpy.QtCore import QRectF
from qtpy.QtGui import QImage, QPainter, QPalette

from ballontranslator.utils.text_effects import (
    GeneratedEffectPaint,
    LinearGradientPaint,
    SolidPaint,
    TexturePaint,
)
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.rgba import (
    premultiply_rgba_in_place,
    unpremultiply_rgba_in_place,
)

from ...misc import ndarray2pixmap


_numba_colorize_linear_gradient_rgba = None


def start_effect_paint_numba_warmup() -> threading.Thread:
    """Load or compile gradient kernels outside the Qt event thread.

    >>> callable(start_effect_paint_numba_warmup)
    True
    """
    def warmup() -> None:
        global _numba_colorize_linear_gradient_rgba
        try:
            from .paint_numba import (
                colorize_linear_gradient_rgba,
                warm_effect_paint_numba_cache,
            )
            warm_effect_paint_numba_cache()
            _numba_colorize_linear_gradient_rgba = (
                colorize_linear_gradient_rgba
            )
            LOGGER.info('Text effect gradient acceleration is ready.')
        except Exception as error:
            LOGGER.warning(
                f'Text effect gradient acceleration is unavailable: {error}'
            )

    thread = threading.Thread(
        target=warmup,
        name='EffectPaintNumbaWarmup',
        daemon=True,
    )
    thread.start()
    return thread


def _compiled_colorize_linear_gradient_rgba(*args, **kwargs) -> bool:
    """Use the warmed backend without depending on startup timing."""
    if _numba_colorize_linear_gradient_rgba is None:
        return False
    return _numba_colorize_linear_gradient_rgba(*args, **kwargs)


def colorize_effect_paint_rgba(
    paint: GeneratedEffectPaint,
    rgba: np.ndarray,
    surface_rect: QRectF,
    logical_rect: QRectF,
    render_scale: float,
) -> np.ndarray:
    """Colorize an existing straight-RGBA surface in place.

    Stop RGB and opacity interpolate independently. Chunking bounds temporary
    storage and lets the effect renderer reuse its captured effect surface.

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
        raise TypeError('generated effect paint requires Solid or Gradient')

    height, width = rgba.shape[:2]
    center = logical_rect.center()
    length = max(logical_rect.width(), logical_rect.height()) * paint.scale
    if length <= 0.0:
        alpha = int(round(paint.stops[0].opacity * 255))
        rgba[..., :3] = paint.stops[0].color
        product = rgba[..., 3].astype(np.uint16)
        product *= alpha
        product += 127
        product //= 255
        rgba[..., 3] = product.astype(np.uint8)
        return rgba

    radians = math.radians(paint.angle)
    direction_x = math.cos(radians)
    direction_y = math.sin(radians)
    positions = np.asarray(
        [stop.position for stop in paint.stops], dtype=np.float32
    )
    colors = np.asarray(
        [stop.color for stop in paint.stops], dtype=np.float32
    )
    opacities = np.asarray(
        [stop.opacity * 255.0 for stop in paint.stops], dtype=np.float32
    )
    if _compiled_colorize_linear_gradient_rgba(
        rgba,
        surface_rect.left(),
        surface_rect.top(),
        center.x(),
        center.y(),
        render_scale,
        direction_x,
        direction_y,
        length,
        positions,
        colors,
        opacities,
    ):
        return rgba
    x = (
        surface_rect.left()
        + (np.arange(width, dtype=np.float32) + 0.5) / render_scale
        - center.x()
    )
    two_stop = len(paint.stops) == 2
    opaque_two_stop = two_stop and all(
        stop.opacity == 1.0 for stop in paint.stops
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
        target = rgba[row_start:row_end]
        # Avoid per-stop masks and a full RGB intermediate while retaining
        # the generic path's float32 interpolation and integer rounding.
        if two_stop:
            span = positions[1] - positions[0]
            if span <= 0.0:
                ratio = (parameter >= positions[1]).astype(np.float32)
            else:
                ratio = (parameter - positions[0]) / span
                np.clip(ratio, 0.0, 1.0, out=ratio)

            for channel in range(3):
                values = colors[0, channel] + (
                    colors[1, channel] - colors[0, channel]
                ) * ratio
                target[..., channel] = np.rint(values).astype(np.uint8)

            if opaque_two_stop:
                continue
            paint_alpha = np.rint(
                opacities[0]
                + (opacities[1] - opacities[0]) * ratio
            ).astype(np.uint8)
            product = target[..., 3].astype(np.uint16)
            product *= paint_alpha.astype(np.uint16)
            product += 127
            product //= 255
            target[..., 3] = product.astype(np.uint8)
            continue

        right = np.ones(parameter.shape, dtype=np.uint8)
        # Advancing on equality preserves equal-position hard transitions.
        for index in range(1, len(paint.stops) - 1):
            right[parameter >= positions[index]] = index + 1
        paint_rgb = target[..., :3]
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
                paint_values = np.rint(values).astype(np.uint8)
                paint_rgb[..., channel][selected] = paint_values
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


def colorize_texture_paint_rgba(
    paint: TexturePaint,
    rgba: np.ndarray,
    texture_rgba: np.ndarray,
    surface_rect: QRectF,
    logical_rect: QRectF,
    render_scale: float,
    *,
    texture_premultiplied: bool = False,
) -> np.ndarray:
    """Map required RGBA8 texture pixels in absolute logical coordinates.

    >>> callable(colorize_texture_paint_rgba)
    True
    """
    if not isinstance(paint, TexturePaint):
        raise TypeError('texture colorization requires TexturePaint')
    if (
        not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.ndim != 3
        or rgba.shape[2] != 4
    ):
        raise TypeError('texture paint target must be a uint8 RGBA array')
    if not math.isfinite(render_scale) or render_scale <= 0.0:
        raise ValueError('texture paint raster scale must be positive')
    if (
        not isinstance(texture_rgba, np.ndarray)
        or texture_rgba.dtype != np.uint8
        or texture_rgba.ndim != 3
        or texture_rgba.shape[2] != 4
        or not texture_rgba.shape[0]
        or not texture_rgba.shape[1]
    ):
        raise ValueError('texture paint requires a uint8 RGBA raster')
    height, width = rgba.shape[:2]
    texture_height, texture_width = texture_rgba.shape[:2]
    logical_width = logical_rect.width()
    logical_height = logical_rect.height()
    if logical_width <= 0.0 or logical_height <= 0.0:
        rgba[..., :3] = texture_rgba[0, 0, :3]
        rgba[..., 3] = 0
        return rgba
    has_transparency = texture_premultiplied or not np.all(
        texture_rgba[..., 3] == 255
    )
    texture_source = texture_rgba
    if has_transparency and not texture_premultiplied:
        texture_source = np.array(texture_rgba, copy=True, order='C')
        premultiply_rgba_in_place(texture_source)

    x = (
        surface_rect.left()
        + (np.arange(width, dtype=np.float32) + 0.5) / render_scale
    )
    if paint.mapping == 'fill':
        map_x = (
            (x - logical_rect.left()) * texture_width / logical_width - 0.5
        )
        mapped_width = logical_width
        mapped_height = logical_height
        mapped_left = logical_rect.left()
        mapped_top = logical_rect.top()
        border_mode = cv2.BORDER_REPLICATE
    elif paint.mapping in {'fit', 'crop'}:
        factor = (
            min(
                logical_width / texture_width,
                logical_height / texture_height,
            )
            if paint.mapping == 'fit'
            else max(
                logical_width / texture_width,
                logical_height / texture_height,
            )
        )
        mapped_width = texture_width * factor
        mapped_height = texture_height * factor
        mapped_left = logical_rect.center().x() - mapped_width / 2.0
        mapped_top = logical_rect.center().y() - mapped_height / 2.0
        map_x = (x - mapped_left) * texture_width / mapped_width - 0.5
        border_mode = cv2.BORDER_CONSTANT
    else:
        mapped_width = texture_width * paint.scale
        mapped_height = texture_height * paint.scale
        mapped_left = logical_rect.left()
        mapped_top = logical_rect.top()
        map_x = (
            (x - mapped_left) * texture_width / mapped_width - 0.5
        )
        border_mode = cv2.BORDER_WRAP

    for row_start in range(0, height, 256):
        row_end = min(height, row_start + 256)
        y = (
            surface_rect.top()
            + (np.arange(row_start, row_end, dtype=np.float32) + 0.5)
            / render_scale
        )
        map_y = (y - mapped_top) * texture_height / mapped_height - 0.5
        map_x_grid = np.broadcast_to(
            map_x[None, :], (row_end - row_start, width)
        )
        map_y_grid = np.broadcast_to(
            map_y[:, None], (row_end - row_start, width)
        )
        sampled = cv2.remap(
            texture_source,
            map_x_grid,
            map_y_grid,
            cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=(0, 0, 0, 0),
        )
        if paint.mapping == 'fit':
            inside = (
                (x[None, :] >= mapped_left)
                & (x[None, :] < mapped_left + mapped_width)
                & (y[:, None] >= mapped_top)
                & (y[:, None] < mapped_top + mapped_height)
            )
            sampled[~inside] = 0
        if has_transparency:
            unpremultiply_rgba_in_place(sampled)
        target = rgba[row_start:row_end]
        target[..., :3] = sampled[..., :3]
        product = target[..., 3].astype(np.uint16)
        product *= sampled[..., 3].astype(np.uint16)
        product += 127
        product //= 255
        target[..., 3] = product.astype(np.uint8)
    return rgba


def rasterize_effect_paint(
    paint: GeneratedEffectPaint,
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
        paint,
        result,
        surface_rect,
        logical_rect,
        render_scale,
    )


def effect_paint_preview_image(
    paint: GeneratedEffectPaint,
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
    paint: GeneratedEffectPaint,
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

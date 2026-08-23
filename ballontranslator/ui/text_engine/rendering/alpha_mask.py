"""Bounded item-local rasterization for TextBlock alpha-mask histories."""

import math

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from ballontranslator.utils.text_alpha_mask import TextAlphaMask
from .raster import (
    EFFECT_CACHE_MAX_BYTES,
    EFFECT_CACHE_MAX_DIMENSION,
    EFFECT_CACHE_MAX_PIXELS,
    RASTER_BOUNDARY_FAILURES,
    EffectRasterAllocationError,
)


def render_text_alpha_mask(
    mask: TextAlphaMask,
    surface_rect: QRectF,
    logical_origin: QPointF,
    render_scale: float,
) -> QImage:
    """Rasterize ordered hard round strokes into one bounded alpha image.

    The transform is anchored in item coordinates, making full and tiled
    surfaces sample the same mask geometry.

    >>> from ballontranslator.utils.text_alpha_mask import AlphaBrushStroke
    >>> mask = TextAlphaMask(strokes=(AlphaBrushStroke('erase', 4, ((0, 0),)),))
    >>> render_text_alpha_mask(mask, QRectF(-2, -2, 4, 4), QPointF(), 1).isNull()
    False
    """
    if not isinstance(mask, TextAlphaMask) or mask.is_neutral():
        raise ValueError('render_text_alpha_mask requires an active typed mask')
    if not math.isfinite(render_scale) or render_scale <= 0.0:
        raise ValueError('text alpha mask render scale must be finite and positive')

    pixel_width = max(1, math.ceil(surface_rect.width() * render_scale))
    pixel_height = max(1, math.ceil(surface_rect.height() * render_scale))
    if (
        pixel_width > EFFECT_CACHE_MAX_DIMENSION
        or pixel_height > EFFECT_CACHE_MAX_DIMENSION
        or pixel_width * pixel_height > EFFECT_CACHE_MAX_PIXELS
        or pixel_width * pixel_height > EFFECT_CACHE_MAX_BYTES
    ):
        raise EffectRasterAllocationError(
            f'text alpha mask surface {pixel_width}x{pixel_height} exceeds policy'
        )

    try:
        image = QImage(
            pixel_width,
            pixel_height,
            QImage.Format.Format_Alpha8,
        )
        if image.isNull():
            raise EffectRasterAllocationError(
                f'unable to allocate text alpha mask surface '
                f'{pixel_width}x{pixel_height}'
            )
        image.fill(255)
        image.setDevicePixelRatio(render_scale)
        painter = QPainter(image)
        if not painter.isActive():
            raise EffectRasterAllocationError(
                'unable to begin text alpha mask painter'
            )
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # DevicePixelRatio owns scaling; translate only in logical units
            # so full and tile surfaces share the same item-space sample grid.
            painter.translate(-surface_rect.topLeft())
            for stroke in mask.strokes:
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_Clear
                    if stroke.mode == 'erase'
                    else QPainter.CompositionMode.CompositionMode_SourceOver
                )
                pen = QPen(
                    QColor(255, 255, 255, 255),
                    stroke.diameter,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                painter.setPen(pen)
                points = [
                    QPointF(
                        logical_origin.x() + point[0],
                        logical_origin.y() + point[1],
                    )
                    for point in stroke.points
                ]
                if len(points) == 1:
                    painter.drawPoint(points[0])
                else:
                    path = QPainterPath(points[0])
                    for point in points[1:]:
                        path.lineTo(point)
                    painter.drawPath(path)
        finally:
            painter.end()
    except RASTER_BOUNDARY_FAILURES as error:
        if isinstance(error, EffectRasterAllocationError):
            raise
        raise EffectRasterAllocationError(
            'unable to rasterize text alpha mask'
        ) from error
    return image

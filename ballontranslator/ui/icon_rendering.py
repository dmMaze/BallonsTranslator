import math
from functools import lru_cache

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPixmap
from qtpy.QtSvg import QSvgRenderer


@lru_cache(maxsize=256)
def render_svg_pixmap(
    path: str,
    width: int,
    height: int,
    device_pixel_ratio: float,
    inset: int = 0,
    background_rgba=(0, 0, 0, 0),
    background_radius: int = 0,
) -> QPixmap:
    """Render and cache a device-pixel-aware SVG pixmap.

    >>> render_svg_pixmap('', 20, 20, 1.0).isNull()
    True
    """
    if not path or width <= 0 or height <= 0:
        return QPixmap()
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return QPixmap()

    dpr = max(1.0, float(device_pixel_ratio or 1.0))
    physical_width = max(1, math.ceil(width * dpr))
    physical_height = max(1, math.ceil(height * dpr))
    pixmap = QPixmap(physical_width, physical_height)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    if background_rgba[3] > 0:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*background_rgba))
        painter.drawRoundedRect(
            QRectF(0, 0, width, height),
            background_radius,
            background_radius,
        )
    renderer.render(
        painter,
        QRectF(inset, inset, width - inset * 2, height - inset * 2),
    )
    painter.end()
    return pixmap

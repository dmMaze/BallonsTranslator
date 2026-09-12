import math
import os.path as osp
from functools import cached_property, lru_cache

from qtpy.QtCore import QLineF, QPointF, Qt
from qtpy.QtGui import QBrush, QColor, QCursor, QPainter, QPainterPath, QPen, QPixmap

from ballontranslator.utils import shared


def _icon_path(filename: str) -> str:
    return osp.join(shared.ICON_DIR, filename)


def _wand_star_path(cx: float, cy: float, outer: float, inner: float) -> QPainterPath:
    """Four-pointed star used by the wand cursor.

    Example:
        >>> _wand_star_path(0, 0, 2, 1).elementCount() > 4
        True
    """
    path = QPainterPath()
    for i in range(8):
        radius = outer if i % 2 == 0 else inner
        angle = math.radians(-90 + i * 45)
        point = QPointF(cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        if i == 0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    return path


def _paint_magic_wand(painter: QPainter, color: QColor, width: float) -> None:
    painter.setPen(QPen(
        color,
        width,
        Qt.PenStyle.SolidLine,
        Qt.PenCapStyle.RoundCap,
        Qt.PenJoinStyle.RoundJoin,
    ))
    painter.setBrush(QBrush(color))
    painter.drawLine(QLineF(9, 9, 24, 26))
    painter.drawLine(QLineF(21.5, 24.5, 26.5, 21))
    painter.drawPath(_wand_star_path(7, 7, 5.5, 2.2))
    painter.drawPath(_wand_star_path(20, 5.5, 2.2, 0.8))
    painter.drawPath(_wand_star_path(26, 12, 1.8, 0.7))


@lru_cache(maxsize=1)
def magic_wand_cursor() -> QCursor:
    """Bitmap cursor with the hotspot on the wand star."""
    size = 32
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    _paint_magic_wand(painter, QColor(Qt.GlobalColor.white), 4.0)
    _paint_magic_wand(painter, QColor(Qt.GlobalColor.black), 2.0)
    painter.end()
    return QCursor(pixmap, 7, 7)


def scene_angle_to_cursor_index(angle: float) -> int:
    """Map an ``atan2`` scene angle to the cursor lists' handle order.

    Scene angles start at the right edge, while the cursor lists start at the
    top-left handle and advance clockwise.

    >>> [scene_angle_to_cursor_index(angle) for angle in (-135, -90, -45, 0)]
    [0, 1, 2, 3]
    """
    return int((angle + 135.0 + 22.5) % 360 / 45)


def resize_handle_scene_angle(
    horizontal_axis: QPointF,
    handle_index: int,
) -> float:
    """Return the semantic resize direction for an eight-handle box.

    The top edge supplies accumulated rotation.  Aspect ratio, non-uniform
    scale, and projective slant must not change a handle's semantic role.

    >>> round(resize_handle_scene_angle(QPointF(1000, 0), 0))
    -135
    """
    rotation = math.degrees(
        math.atan2(horizontal_axis.y(), horizontal_axis.x())
    )
    return rotation + 45.0 * handle_index - 135.0


class RotateCursorList:
    @cached_property
    def Cursor0(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor0.png')))

    @cached_property
    def Cursor1(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor1.png')))

    @cached_property
    def Cursor2(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor2.png')))

    @cached_property
    def Cursor3(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor3.png')))

    @cached_property
    def Cursor4(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor4.png')))

    @cached_property
    def Cursor5(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor5.png')))

    @cached_property
    def Cursor6(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor6.png')))

    @cached_property
    def Cursor7(self):
        return QCursor(QPixmap(_icon_path('rotate_cursor7.png')))

    def __getitem__(self, idx):
        return self.__getattribute__('Cursor' + str(idx))
        
resizeCursorList = [
    Qt.CursorShape.SizeFDiagCursor, 
    Qt.CursorShape.SizeVerCursor, 
    Qt.CursorShape.SizeBDiagCursor, 
    Qt.CursorShape.SizeHorCursor
]
rotateCursorList = RotateCursorList()

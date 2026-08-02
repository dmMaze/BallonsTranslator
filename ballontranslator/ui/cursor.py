import math
import os.path as osp
from functools import cached_property

from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QCursor, QPixmap

from ballontranslator.utils import shared


def _icon_path(filename: str) -> str:
    return osp.join(shared.ICON_DIR, filename)


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

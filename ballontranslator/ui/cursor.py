from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtGui import QCursor
from functools import cached_property
import os.path as osp

from ballontranslator.utils import shared


def _icon_path(filename: str) -> str:
    return osp.join(shared.ICON_DIR, filename)


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

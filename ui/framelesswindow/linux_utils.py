# coding: utf-8

from utils import shared

# coding: utf-8
from qtpy.QtCore import QObject, QEvent
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget


class LinuxMoveResize:
    """ Tool class for moving and resizing window """

    @classmethod
    def startSystemMove(cls, window, globalPos):
        """ move window """
        window.windowHandle().startSystemMove()

    @classmethod
    def starSystemResize(cls, window, globalPos, edges):
        """ resize window

        Parameters
        ----------
        window: QWidget
            window

        globalPos: QPoint
            the global point of mouse release event

        edges: `Qt.Edges`
            window edges
        """
        window.windowHandle().startSystemResize(edges)

    @classmethod
    def toggleMaxState(cls, window):
        if shared.HEADLESS:
            return
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()


def getSystemAccentColor():
    """ get the accent color of system

    Returns
    -------
    color: QColor
        accent color
    """
    return QColor()


class LinuxScreenCaptureFilter(QObject):
    """ Filter for screen capture """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setScreenCaptureEnabled(False)

    def eventFilter(self, watched, event):
        if watched == self.parent():
            if event.type() == QEvent.Type.WinIdChange:
                self.setScreenCaptureEnabled(self.isScreenCaptureEnabled)

        return super().eventFilter(watched, event)

    def setScreenCaptureEnabled(self, enabled: bool):
        """ Set screen capture enabled """
        self.isScreenCaptureEnabled = enabled

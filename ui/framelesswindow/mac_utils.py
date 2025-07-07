# coding:utf-8
from ctypes import c_void_p

import Cocoa
import objc
from qtpy.QtCore import qVersion, QEvent, QObject
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget
from Quartz.CoreGraphics import (CGEventCreateMouseEvent,
                                 kCGEventLeftMouseDown, kCGMouseButtonLeft)

QT_VERSION = tuple(int(v) for v in qVersion().split('.'))


class MacMoveResize:
    """ Tool class for moving and resizing Mac OS window """

    @staticmethod
    def startSystemMove(window: QWidget, globalPos):
        """ resize window

        Parameters
        ----------
        window: QWidget
            window

        globalPos: QPoint
            the global point of mouse release event
        """
        if QT_VERSION >= (5, 15, 0):
            window.windowHandle().startSystemMove()
            return

        nsWindow = getNSWindow(window.winId())

        # send click event
        cgEvent = CGEventCreateMouseEvent(
            None, kCGEventLeftMouseDown, (globalPos.x(), globalPos.y()), kCGMouseButtonLeft)
        clickEvent = Cocoa.NSEvent.eventWithCGEvent_(cgEvent)

        if clickEvent:
            nsWindow.performWindowDragWithEvent_(clickEvent)

        # CFRelease(cgEvent)

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
        pass

    @classmethod
    def toggleMaxState(cls, window):
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()


def getNSWindow(winId):
    """ convert window handle to NSWindow

    Parameters
    ----------
    winId: int or `sip.voidptr`
        window handle
    """
    view = objc.objc_object(c_void_p=c_void_p(int(winId)))
    return view.window()


def getSystemAccentColor():
    """ get the accent color of system

    Returns
    -------
    color: QColor
        accent color
    """
    color = Cocoa.NSColor.controlAccentColor()
    color = color.colorUsingColorSpace_(Cocoa.NSColorSpace.sRGBColorSpace())
    r = int(color.redComponent() * 255)
    g = int(color.greenComponent() * 255)
    b = int(color.blueComponent() * 255)
    return QColor(r, g, b)


class MacScreenCaptureFilter(QObject):
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

        nsWindow = getNSWindow(self.parent().winId())
        if nsWindow:
            NSWindowSharingNone = 0
            NSWindowSharingReadOnly = 1
            nsWindow.setSharingType_(NSWindowSharingReadOnly if enabled else NSWindowSharingNone)
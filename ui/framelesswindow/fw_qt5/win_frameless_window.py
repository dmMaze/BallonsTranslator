# coding:utf-8
import sys
from ctypes import cast
from ctypes.wintypes import LPRECT, MSG
from platform import platform

import win32api
import win32con
import win32gui
from qtpy.QtCore import Qt, QSize, QRect
from qtpy.QtGui import QCloseEvent, QCursor
from qtpy.QtWidgets import QApplication, QWidget, QMainWindow

# from ..titlebar import TitleBar
from .. import win32_utils as win_utils
from ..win32_utils import Taskbar, isSystemBorderAccentEnabled, getSystemAccentColor
from ..win_c_structures import LPNCCALCSIZE_PARAMS
from ..win_window_effect import WindowsWindowEffect


class WindowsFramelessWindow(QWidget):
    """  Frameless window for Windows system """

    BORDER_WIDTH = 5

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.windowEffect = WindowsWindowEffect(self)
        # self.titleBar = TitleBar(self)
        self._isSystemButtonVisible = False
        self._isResizeEnabled = True

        self.updateFrameless()

        # solve issue #5
        self.windowHandle().screenChanged.connect(self.__onScreenChanged)

        # self.resize(500, 500)
        # self.titleBar.raise_()

    def updateFrameless(self):
        """ update frameless window """
        stayOnTop = Qt.WindowStaysOnTopHint if self.windowFlags() & Qt.WindowStaysOnTopHint else 0

        if not win_utils.isWin7():
            self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        elif self.parent():
            self.setWindowFlags(self.parent().windowFlags() | Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint | stayOnTop)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint | stayOnTop)

        # add DWM shadow and window animation
        self.windowEffect.addWindowAnimation(self.winId())
        if not isinstance(self, AcrylicWindow):
            self.windowEffect.addShadowEffect(self.winId())

    # def setTitleBar(self, titleBar):
    #     """ set custom title bar

    #     Parameters
    #     ----------
    #     titleBar: TitleBar
    #         title bar
    #     """
    #     self.titleBar.deleteLater()
    #     self.titleBar.hide()
    #     self.titleBar = titleBar
    #     self.titleBar.setParent(self)
    #     self.titleBar.raise_()

    def setResizeEnabled(self, isEnabled: bool):
        """ set whether resizing is enabled """
        self._isResizeEnabled = isEnabled

    def setStayOnTop(self, isTop: bool):
        """ set the stay on top status """
        if isTop:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)

        self.updateFrameless()
        self.show()

    def toggleStayOnTop(self):
        """ toggle the stay on top status """
        if self.windowFlags() & Qt.WindowStaysOnTopHint:
            self.setStayOnTop(False)
        else:
            self.setStayOnTop(True)

    # def resizeEvent(self, e):
    #     super().resizeEvent(e)
    #     self.titleBar.resize(self.width(), self.titleBar.height())

    def isSystemButtonVisible(self):
        """ Returns whether the system title bar button is visible """
        return self._isSystemButtonVisible

    def setSystemTitleBarButtonVisible(self, isVisible):
        """ set the visibility of system title bar button, only works for macOS """
        pass

    def systemTitleBarRect(self, size: QSize) -> QRect:
        """ Returns the system title bar rect, only works for macOS

        Parameters
        ----------
        size: QSize
            original system title bar rect
        """
        return QRect(0, 0, size.width(), size.height())

    def nativeEvent(self, eventType, message):
        """ Handle the Windows message """
        msg = MSG.from_address(message.__int__())
        if not msg.hWnd:
            return super().nativeEvent(eventType, message)

        if msg.message == win32con.WM_NCHITTEST and self._isResizeEnabled:
            xPos, yPos = win32gui.ScreenToClient(msg.hWnd, win32api.GetCursorPos())
            clientRect = win32gui.GetClientRect(msg.hWnd)

            w = clientRect[2] - clientRect[0]
            h = clientRect[3] - clientRect[1]

            # fixes issue https://github.com/zhiyiYo/PyQt-Frameless-Window/issues/98
            bw = 0 if win_utils.isMaximized(msg.hWnd) or win_utils.isFullScreen(msg.hWnd) else self.BORDER_WIDTH
            lx = xPos < bw  # left
            rx = xPos > w - bw  # right
            ty = yPos < bw  # top
            by = yPos > h - bw  # bottom
            if lx and ty:
                return True, win32con.HTTOPLEFT
            elif rx and by:
                return True, win32con.HTBOTTOMRIGHT
            elif rx and ty:
                return True, win32con.HTTOPRIGHT
            elif lx and by:
                return True, win32con.HTBOTTOMLEFT
            elif ty:
                return True, win32con.HTTOP
            elif by:
                return True, win32con.HTBOTTOM
            elif lx:
                return True, win32con.HTLEFT
            elif rx:
                return True, win32con.HTRIGHT
        elif msg.message == win32con.WM_NCCALCSIZE:
            if msg.wParam:
                rect = cast(msg.lParam, LPNCCALCSIZE_PARAMS).contents.rgrc[0]
            else:
                rect = cast(msg.lParam, LPRECT).contents

            isMax = win_utils.isMaximized(msg.hWnd)
            isFull = win_utils.isFullScreen(msg.hWnd)

            # adjust the size of client rect
            if isMax and not isFull:
                ty = win_utils.getResizeBorderThickness(msg.hWnd, False)
                rect.top += ty
                rect.bottom -= ty

                tx = win_utils.getResizeBorderThickness(msg.hWnd, True)
                rect.left += tx
                rect.right -= tx

            # handle the situation that an auto-hide taskbar is enabled
            if (isMax or isFull) and Taskbar.isAutoHide():
                position = Taskbar.getPosition(msg.hWnd)
                if position == Taskbar.LEFT:
                    rect.top += Taskbar.AUTO_HIDE_THICKNESS
                elif position == Taskbar.BOTTOM:
                    rect.bottom -= Taskbar.AUTO_HIDE_THICKNESS
                elif position == Taskbar.LEFT:
                    rect.left += Taskbar.AUTO_HIDE_THICKNESS
                elif position == Taskbar.RIGHT:
                    rect.right -= Taskbar.AUTO_HIDE_THICKNESS

            result = 0 if not msg.wParam else win32con.WVR_REDRAW
            return True, result
        elif msg.message == win32con.WM_SETFOCUS and isSystemBorderAccentEnabled():
            self.windowEffect.setBorderAccentColor(self.winId(), getSystemAccentColor())
            return True, 0
        elif msg.message == win32con.WM_KILLFOCUS:
            self.windowEffect.removeBorderAccentColor(self.winId())
            return True, 0

        return super().nativeEvent(eventType, message)

    def __onScreenChanged(self):
        hWnd = int(self.windowHandle().winId())
        win32gui.SetWindowPos(hWnd, None, 0, 0, 0, 0, win32con.SWP_NOMOVE |
                              win32con.SWP_NOSIZE | win32con.SWP_FRAMECHANGED)


class AcrylicWindow(WindowsFramelessWindow):
    """ A frameless window with acrylic effect """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.__closedByKey = False
        self.setStyleSheet("AcrylicWindow{background:transparent}")

    def updateFrameless(self):
        super().updateFrameless()
        self.windowEffect.enableBlurBehindWindow(self.winId())

        stayOnTop = Qt.WindowStaysOnTopHint if self.windowFlags() & Qt.WindowStaysOnTopHint else 0

        if win_utils.isWin7() and self.parent():
            self.setWindowFlags(self.parent().windowFlags() | Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint | stayOnTop)
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint | stayOnTop)

        self.windowEffect.addWindowAnimation(self.winId())

        if win_utils.isWin7():
            self.windowEffect.addShadowEffect(self.winId())
            self.windowEffect.setAeroEffect(self.winId())
        else:
            self.windowEffect.setAcrylicEffect(self.winId())
            if win_utils.isGreaterEqualWin11():
                self.windowEffect.addShadowEffect(self.winId())

    def nativeEvent(self, eventType, message):
        """ Handle the Windows message """
        msg = MSG.from_address(message.__int__())

        # handle Alt+F4
        if msg.message == win32con.WM_SYSKEYDOWN:
            if msg.wParam == win32con.VK_F4:
                self.__closedByKey = True
                QApplication.sendEvent(self, QCloseEvent())
                return False, 0

        return super().nativeEvent(eventType, message)

    def closeEvent(self, e):
        if not self.__closedByKey or QApplication.quitOnLastWindowClosed():
            self.__closedByKey = False
            return super().closeEvent(e)

        # system tray icon
        self.__closedByKey = False
        self.hide()
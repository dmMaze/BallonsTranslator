# coding:utf-8
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QWidget

# from ..titlebar import TitleBar
from ..linux_utils import LinuxMoveResize
from ..linux_window_effect import LinuxWindowEffect


class LinuxFramelessWindow(QWidget):
    """ Frameless window for Linux system """

    BORDER_WIDTH = 5

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.windowEffect = LinuxWindowEffect(self)
        # self.titleBar = TitleBar(self)
        self._isSystemButtonVisible = False
        self._isResizeEnabled = True

        self.updateFrameless()
        QCoreApplication.instance().installEventFilter(self)

        # self.titleBar.raise_()
    #     self.resize(500, 500)

    # def resizeEvent(self, e):
    #     super().resizeEvent(e)
    #     self.titleBar.resize(self.width(), self.titleBar.height())

    def updateFrameless(self):
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)

    def setStayOnTop(self, isTop: bool):
        """ set the stay on top status """
        if isTop:
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)

        self.updateFrameless()
        self.show()

    def toggleStayOnTop(self):
        """ toggle the stay on top status """
        if self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint:
            self.setStayOnTop(False)
        else:
            self.setStayOnTop(True)

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

    def isSystemButtonVisible(self):
        """ Returns whether the system title bar button is visible """
        return self._isSystemButtonVisible

    def setSystemTitleBarButtonVisible(self, isVisible):
        """ set the visibility of system title bar button, only works for macOS """
        pass

    # def systemTitleBarRect(self, size: QSize) -> QRect:
    #     """ Returns the system title bar rect, only works for macOS

    #     Parameters
    #     ----------
    #     size: QSize
    #         original system title bar rect
    #     """
    #     return QRect(0, 0, size.width(), size.height())

    def eventFilter(self, obj, event):
        et = event.type()
        if et != QEvent.Type.MouseButtonPress and et != QEvent.Type.MouseMove or not self._isResizeEnabled:
            return False

        edges = Qt.Edge(0)
        pos = event.globalPosition().toPoint() - self.pos()
        if pos.x() < self.BORDER_WIDTH:
            edges |= Qt.Edge.LeftEdge
        if pos.x() >= self.width()-self.BORDER_WIDTH:
            edges |= Qt.Edge.RightEdge
        if pos.y() < self.BORDER_WIDTH:
            edges |= Qt.Edge.TopEdge
        if pos.y() >= self.height()-self.BORDER_WIDTH:
            edges |= Qt.Edge.BottomEdge

        # change cursor
        if et == QEvent.Type.MouseMove and self.windowState() == Qt.WindowState.WindowNoState:
            if edges in (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.Edge.RightEdge | Qt.Edge.BottomEdge):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif edges in (Qt.Edge.RightEdge | Qt.Edge.TopEdge, Qt.Edge.LeftEdge | Qt.Edge.BottomEdge):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif edges in (Qt.Edge.TopEdge, Qt.Edge.BottomEdge):
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        # elif obj in (self, self.titleBar) and et == QEvent.Type.MouseButtonPress and edges:
        #     LinuxMoveResize.starSystemResize(self, event.globalPosition(), edges)

        return super().eventFilter(obj, event)

# coding:utf-8
import Cocoa
import objc
from qtpy.QtCore import QEvent, Qt, QRect, QSize, QPoint
from qtpy.QtWidgets import QWidget, QMainWindow, QDialog

# from ..titlebar import TitleBar
from ..mac_utils import QT_VERSION
from ..mac_window_effect import MacWindowEffect


class MacFramelessWindowBase:
    """ Frameless window base class for mac """

    def __init__(self, *args, **kwargs):
        self._isSystemButtonVisible = False

    def _initFrameless(self):
        self.windowEffect = MacWindowEffect(self)
        # must enable acrylic effect before creating title bar
        # if isinstance(self, AcrylicWindow):
        #     self.windowEffect.setAcrylicEffect(self.winId())

        # self.titleBar = TitleBar(self)
        self._isResizeEnabled = True

        # remove content margin
        if QT_VERSION >= (6, 8, 0):
            self.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)
            # self.titleBar.setAttribute(Qt.WidgetAttribute.WA_LayoutOnEntireRect, True)

        self.updateFrameless()

        # self.resize(500, 500)
        # self.titleBar.raise_()

    def updateFrameless(self):
        view = objc.objc_object(c_void_p=self.winId().__int__())
        self.__nsWindow = view.window()

        # hide system title bar
        isButtonVisible = self.isSystemButtonVisible()
        self.hideSystemTitleBar()
        # self.setSystemTitleBarButtonVisible(isButtonVisible)

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

    #     if QT_VERSION >= (6, 8, 0):
    #         self.titleBar.setAttribute(Qt.WidgetAttribute.WA_LayoutOnEntireRect)

    def setResizeEnabled(self, isEnabled: bool):
        """ set whether resizing is enabled """
        self._isResizeEnabled = isEnabled

    # def resizeEvent(self, e):
    #     self.titleBar.resize(self.width(), self.titleBar.height())

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            # self.setSystemTitleBarButtonVisible(False)
            self.hideSystemTitleBar()
        elif event.type() == QEvent.Resize:
            # self._updateSystemButtonRect()
            self.hideSystemTitleBar()

    def hideSystemTitleBar(self):
        # extend view to title bar region
        self.__nsWindow.setStyleMask_(
            self.__nsWindow.styleMask() | Cocoa.NSFullSizeContentViewWindowMask)
        self.__nsWindow.setTitlebarAppearsTransparent_(True)

        # disable the moving behavior of system
        self.__nsWindow.setMovableByWindowBackground_(False)
        self.__nsWindow.setMovable_(False)

        # hide title bar buttons and title
        self.__nsWindow.setTitleVisibility_(Cocoa.NSWindowTitleHidden)
        self.setSystemTitleBarButtonVisible(False)

    def isSystemButtonVisible(self):
        return self._isSystemButtonVisible

    def setSystemTitleBarButtonVisible(self, isVisible):
        self._isSystemButtonVisible = isVisible
        self.__nsWindow.setShowsToolbarButton_(isVisible)

        isHidden = not isVisible
        # self.__nsWindow.standardWindowButton_(Cocoa.NSWindowCloseButton).setHidden_(isHidden)
        # self.__nsWindow.standardWindowButton_(Cocoa.NSWindowZoomButton).setHidden_(isHidden)
        # self.__nsWindow.standardWindowButton_(Cocoa.NSWindowMiniaturizeButton).setHidden_(isHidden)

        if isVisible:
            self._updateSystemButtonRect()

    def _updateSystemButtonRect(self):
        if not self.isSystemButtonVisible():
            return

        # get system title bar button
        leftButton = self.__nsWindow.standardWindowButton_(Cocoa.NSWindowCloseButton)
        midButton =  self.__nsWindow.standardWindowButton_(Cocoa.NSWindowMiniaturizeButton)
        rightButton = self.__nsWindow.standardWindowButton_(Cocoa.NSWindowZoomButton)

        # get system title bar
        titlebar = rightButton.superview()
        titlebarHeight = titlebar.frame().size.height

        spacing = midButton.frame().origin.x - leftButton.frame().origin.x
        width = midButton.frame().size.width
        height = midButton.frame().size.height

        if self.__nsWindow.contentView():
            viewSize = self.__nsWindow.contentView().frame().size
        else:
            viewSize = self.__nsWindow.frame().size

        center = self.systemTitleBarRect(QSize(int(viewSize.width), titlebarHeight)).center()

        # The origin of the NSWindow coordinate system is in the lower left corner, we do the necessary transformations
        center.setY(titlebarHeight - center.y())

        # adjust the position of minimize button
        centerOrigin = Cocoa.NSPoint(center.x() - width // 2, center.y() - height // 2)
        midButton.setFrameOrigin_(centerOrigin)

        # adjust the position of close button
        leftOrigin = Cocoa.NSPoint(centerOrigin.x - spacing, centerOrigin.y)
        leftButton.setFrameOrigin_(leftOrigin)

        # adjust the position of zoom button
        rightOrigin = Cocoa.NSPoint(centerOrigin.x + spacing, centerOrigin.y)
        rightButton.setFrameOrigin_(rightOrigin)

    def systemTitleBarRect(self, size: QSize) -> QRect:
        """ Returns the system title bar rect

        Parameters
        ----------
        size: QSize
            original system title bar rect
        """
        return QRect(0, 0, 75, size.height())


class MacFramelessWindow(QWidget, MacFramelessWindowBase):
    """ Frameless window for Linux system """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._isSystemButtonVisible = False
        self._initFrameless()

    # def resizeEvent(self, e):
    #     MacFramelessWindowBase.resizeEvent(self, e)

    def changeEvent(self, e):
        MacFramelessWindowBase.changeEvent(self, e)

    def paintEvent(self, e):
        QWidget.paintEvent(self, e)
        self.setSystemTitleBarButtonVisible(self.isSystemButtonVisible())
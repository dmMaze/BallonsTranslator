# coding:utf-8
import Cocoa
import objc
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QWidget

# from ..titlebar import TitleBar
from .window_effect import MacWindowEffect


class MacFramelessWindow(QWidget):
    """ Frameless window for Linux system """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.windowEffect = MacWindowEffect(self)
        # https://bugreports.qt.io/browse/QTBUG-133215
        self.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)

        # self.titleBar = TitleBar(self)

        view = objc.objc_object(c_void_p=self.winId().__int__())
        self.__nsWindow = view.window()

        # hide system title bar
        self.hideSystemTitleBar()

        # self.resize(500, 500)
        # self.titleBar.raise_()

    # def setTitleBar(self, titleBar):
    #     """ set custom title bar

    #     Parameters
    #     ----------
    #     titleBar: TitleBar
    #         title bar
    #     """
    #     self.titleBar.deleteLater()
    #     self.titleBar = titleBar
    #     self.titleBar.setParent(self)
    #     self.titleBar.raise_()

    # def resizeEvent(self, e):
    #     super().resizeEvent(e)
    #     self.titleBar.resize(self.width(), self.titleBar.height())

    def paintEvent(self, e):
        super().paintEvent(e)
        self.__nsWindow.setTitlebarAppearsTransparent_(True)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
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
        self.__nsWindow.setShowsToolbarButton_(False)
        self.__nsWindow.setTitleVisibility_(Cocoa.NSWindowTitleHidden)
        # self.__nsWindow.standardWindowButton_(Cocoa.NSWindowCloseButton).setHidden_(True)
        # self.__nsWindow.standardWindowButton_(Cocoa.NSWindowZoomButton).setHidden_(True)
        # self.__nsWindow.standardWindowButton_(Cocoa.NSWindowMiniaturizeButton).setHidden_(True)

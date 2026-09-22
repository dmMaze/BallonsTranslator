from qtpy.QtCore import QElapsedTimer, QTimer, Qt
from qtpy.QtGui import QHideEvent, QPainter, QPaintEvent, QShowEvent
from qtpy.QtWidgets import QToolButton, QPushButton, QSizePolicy, QWidget

from ..icon_rendering import render_svg_pixmap
from ..misc import themed_icon_path


class NoBorderPushBtn(QPushButton):
    pass


class RefreshButton(QToolButton):
    """Show the themed reload SVG and animate only while busy and visible.

    >>> issubclass(RefreshButton, QToolButton)
    True
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName('RefreshButton')
        self._busy = False
        self._angle = 0.0
        self._elapsed = QElapsedTimer()
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setInterval(30)
        self._rotation_timer.timeout.connect(self._advance_rotation)
        self.setAccessibleName(self.tr('Refresh'))

    def set_busy(self, busy: bool) -> None:
        if self._busy == busy:
            return
        self._busy = busy
        self.setEnabled(not busy)
        if busy and self.isVisible():
            self._elapsed.start()
            self._rotation_timer.start()
        else:
            self._rotation_timer.stop()
        self._angle = 0.0
        self.update()

    def _advance_rotation(self) -> None:
        self._angle = (self._elapsed.elapsed() % 900) * 360.0 / 900
        self.update()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._busy:
            self._elapsed.start()
            self._rotation_timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self._rotation_timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        icon = (
            'fontfmt_reload_activate.svg'
            if self.isDown() and self.isEnabled()
            else 'fontfmt_reload.svg'
        )
        pixmap = render_svg_pixmap(
            themed_icon_path(icon), 20, 20, self.devicePixelRatioF(),
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)
        if not self.isEnabled():
            painter.setOpacity(0.45)
        painter.drawPixmap(-10, -10, pixmap)
        painter.end()


class ExpandingToolButton(QToolButton):
    """A left-aligned tool button that fills the available horizontal space.

    >>> ExpandingToolButton.__name__
    'ExpandingToolButton'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

import sys

from qtpy.QtCore import QEvent, QObject, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QRegion
from qtpy.QtWidgets import QMenu, QProxyStyle, QStyle, QWidget

from .icon_rendering import render_svg_pixmap
from .misc import themed_icon_path
from ballontranslator.utils import shared


_MENU_CORNER_RADIUS = 8
_MENU_CHEVRON_SIZE = 12


class _MenuChevronStyle(QProxyStyle):
    """Replace native submenu arrows with the shared themed SVG chevron.

    >>> _MenuChevronStyle.__name__
    '_MenuChevronStyle'
    """

    def drawPrimitive(self, element, option, painter, widget=None):
        primitive = getattr(QStyle, 'PrimitiveElement', QStyle)
        if (
            element == primitive.PE_IndicatorArrowRight
            and isinstance(widget, QMenu)
        ):
            pixmap = render_svg_pixmap(
                themed_icon_path('chevron-right.svg'),
                _MENU_CHEVRON_SIZE,
                _MENU_CHEVRON_SIZE,
                widget.devicePixelRatioF(),
            )
            target = option.rect
            x = target.center().x() - _MENU_CHEVRON_SIZE // 2
            y = target.center().y() - _MENU_CHEVRON_SIZE // 2
            painter.drawPixmap(x, y, pixmap)
            return
        return super().drawPrimitive(element, option, painter, widget)


def _windows_menu_mask(rect):
    """Clip compositor edges and round only the artifact-prone corner.

    A binary mask around the whole menu makes every submenu corner jagged.

    >>> from qtpy.QtCore import QRect
    >>> _windows_menu_mask(QRect(0, 0, 100, 80)).isEmpty()
    False
    """
    width, height = rect.width(), rect.height()
    edge_clip = QRegion(
        0, 0, max(0, width - 1), max(0, height - 1),
    )
    corner_size = min(_MENU_CORNER_RADIUS + 2, width, height)
    if corner_size <= 0:
        return edge_clip

    corner = QRegion(
        width - corner_size,
        height - corner_size,
        corner_size,
        corner_size,
    )
    path = QPainterPath()
    path.addRoundedRect(
        QRectF(rect).adjusted(0, 0, -1, -1),
        _MENU_CORNER_RADIUS,
        _MENU_CORNER_RADIUS,
    )
    rounded_corner = QRegion(
        path.toFillPolygon().toPolygon()
    ).intersected(corner).intersected(edge_clip)
    return edge_clip.subtracted(corner).united(rounded_corner)


class _MenuBorderOverlay(QWidget):
    """Paint a smooth border without changing QMenu's widget hit region."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('MenuBorderOverlay')
        attr_enum = getattr(Qt, 'WidgetAttribute', Qt)
        self.setAttribute(attr_enum.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        render_hint = getattr(QPainter, 'RenderHint', QPainter)
        painter.setRenderHint(render_hint.Antialiasing, True)
        brush_style = getattr(Qt, 'BrushStyle', Qt)
        painter.setBrush(brush_style.NoBrush)
        painter.setPen(QPen(QColor(*shared.BORDER_COLOR), 1))
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = _MENU_CORNER_RADIUS - 1
        painter.drawRoundedRect(rect, radius, radius)


class MenuStyleFilter(QObject):
    """Apply shared popup geometry and text-based checked markers to menus."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._menu_chevron_style = _MenuChevronStyle()
        self._menu_chevron_style.setParent(self)

    def eventFilter(self, watched, event):
        if not isinstance(watched, QMenu):
            return False
        event_type = event.type()
        if event_type == QEvent.Type.Polish:
            attr_enum = getattr(Qt, 'WidgetAttribute', Qt)
            watched.setAttribute(attr_enum.WA_TranslucentBackground, True)
            watched.setStyle(self._menu_chevron_style)
            if sys.platform == 'darwin':
                # Use the stylesheet border instead of macOS's native popup
                # frame, which is black and leaves an outer gap around menus.
                window_type = getattr(Qt, 'WindowType', Qt)
                watched.setWindowFlag(window_type.NoDropShadowWindowHint, True)
            elif sys.platform == 'win32':
                window_type = getattr(Qt, 'WindowType', Qt)
                watched.setWindowFlag(window_type.FramelessWindowHint, True)
                watched.setWindowFlag(window_type.NoDropShadowWindowHint, True)
                watched.setProperty('windowsBorderOverlay', True)
                if not hasattr(watched, '_menu_border_overlay'):
                    watched._menu_border_overlay = _MenuBorderOverlay(watched)
        elif event_type in (QEvent.Type.Show, QEvent.Type.Resize):
            if sys.platform == 'win32':
                window = watched.windowHandle()
                if window is not None:
                    window.setMask(_windows_menu_mask(watched.rect()))
                overlay = getattr(watched, '_menu_border_overlay', None)
                if overlay is not None:
                    overlay.setGeometry(watched.rect().adjusted(0, 0, -1, -1))
                    overlay.show()
                    overlay.raise_()
                    overlay.update()
            elif sys.platform != 'darwin':
                path = QPainterPath()
                path.addRoundedRect(
                    QRectF(watched.rect()),
                    _MENU_CORNER_RADIUS,
                    _MENU_CORNER_RADIUS,
                )
                watched.setMask(QRegion(path.toFillPolygon().toPolygon()))
        if event_type == QEvent.Type.Show:
            self._sync_checked_action_text(watched)
        return super().eventFilter(watched, event)

    @staticmethod
    def _sync_checked_action_text(menu: QMenu):
        for action in menu.actions():
            if not action.isCheckable():
                continue
            base_text = action.property('_menuBaseText')
            if base_text is None:
                base_text = action.text()
                action.setProperty('_menuBaseText', base_text)
            action.setText(base_text + ('\t\u2713' if action.isChecked() else ''))

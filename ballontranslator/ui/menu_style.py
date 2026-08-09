import sys

from qtpy.QtCore import QEvent, QObject, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QRegion
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFontComboBox,
    QFrame,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from ballontranslator.utils import shared


_MENU_CORNER_RADIUS = 8


class DropDownStyleFilter(QObject):
    """Keep combo popup hover colors independent of platform styling.

    >>> DropDownStyleFilter.__name__
    'DropDownStyleFilter'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Qt owns the delegates as children, while this registry keeps their
        # Python wrappers alive so overridden paint methods remain callable.
        self._delegates = {}

    def eventFilter(self, watched, event):
        if isinstance(watched, QComboBox):
            if event.type() in (
                QEvent.Type.Show,
                QEvent.Type.MouseButtonPress,
                QEvent.Type.KeyPress,
            ):
                self._style_view(watched, watched.view())
            return super().eventFilter(watched, event)

        if not isinstance(watched, QWidget):
            return False
        if watched.objectName() != 'qt_scrollarea_viewport':
            return False
        if event.type() not in (
            QEvent.Type.MouseMove,
            QEvent.Type.HoverMove,
        ):
            return False
        view = watched.parentWidget()
        if not isinstance(view, QAbstractItemView):
            return False
        popup = view.parentWidget()
        if not isinstance(popup, QFrame):
            return False
        combo = popup.parentWidget()
        if not isinstance(combo, QComboBox):
            return False
        self._style_view(combo, view)
        return super().eventFilter(watched, event)

    def _style_view(self, combo: QComboBox, view: QAbstractItemView):
        # Never reach this from Polish or Paint: replacing a delegate while Qt
        # is styling or drawing the same view can re-enter platform style code.
        if not view.hasMouseTracking():
            view.setMouseTracking(True)
        # QFontComboBox's native delegate previews every item in its own font.
        if isinstance(combo, QFontComboBox):
            return
        key = id(combo)
        record = self._delegates.get(key)
        if record is None or record[0] is not combo or record[1] is not view:
            delegate = _DropDownItemDelegate(view)
            self._delegates[key] = (combo, view, delegate)
            if record is None or record[0] is not combo:
                # Follow the C++ widget lifetime; Python wrappers may disappear
                # earlier or be recreated by the binding.
                combo.setProperty('_dropDownStyleRegistryKey', key)
                combo.destroyed.connect(self._discard_combo_delegate)
        else:
            delegate = record[2]
        if view.itemDelegate() is not delegate:
            view.setItemDelegate(delegate)

    def _discard_combo_delegate(self, combo: QObject = None) -> None:
        if combo is None:
            return
        key = combo.property('_dropDownStyleRegistryKey')
        if key is not None:
            self._delegates.pop(int(key), None)


class _DropDownItemDelegate(QStyledItemDelegate):
    """Paint combo popup hover without relying on stylesheet precedence."""

    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        state_enum = getattr(QStyle, 'StateFlag', QStyle)
        highlighted = state_enum.State_MouseOver | state_enum.State_Selected
        is_highlighted = bool(option.state & highlighted)
        if is_highlighted:
            option.state &= ~highlighted
        style = QApplication.style() if option.widget is None else option.widget.style()
        control = getattr(QStyle, 'ControlElement', QStyle)
        style.drawControl(
            control.CE_ItemViewItem,
            option,
            painter,
            option.widget,
        )
        if is_highlighted:
            painter.fillRect(option.rect, QColor(30, 147, 229, 51))


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
        # Keep one live Python overlay wrapper per menu for paintEvent dispatch.
        self._menu_border_overlays = {}

    def _menu_border_overlay(self, menu: QMenu, create=False):
        key = id(menu)
        record = self._menu_border_overlays.get(key)
        if record is not None and record[0] is menu:
            return record[1]
        if not create:
            return None
        overlay = _MenuBorderOverlay(menu)
        self._menu_border_overlays[key] = (menu, overlay)
        # QMenu owns the QWidget; the registry only mirrors that lifetime.
        menu.setProperty('_menuStyleRegistryKey', key)
        menu.destroyed.connect(self._discard_menu_border_overlay)
        return overlay

    def _discard_menu_border_overlay(self, menu: QObject = None) -> None:
        if menu is None:
            return
        key = menu.property('_menuStyleRegistryKey')
        if key is not None:
            self._menu_border_overlays.pop(int(key), None)

    def eventFilter(self, watched, event):
        if not isinstance(watched, QMenu):
            return False
        event_type = event.type()
        if event_type == QEvent.Type.Polish:
            attr_enum = getattr(Qt, 'WidgetAttribute', Qt)
            watched.setAttribute(attr_enum.WA_TranslucentBackground, True)
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
                self._menu_border_overlay(watched, create=True)
        elif event_type in (QEvent.Type.Show, QEvent.Type.Resize):
            if sys.platform == 'win32':
                window = watched.windowHandle()
                if window is not None:
                    window.setMask(_windows_menu_mask(watched.rect()))
                overlay = self._menu_border_overlay(watched)
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


def install_app_style_filters(app):
    """Install one application-owned instance of each popup style filter.

    >>> hasattr(QObject(), '_menu_style_filter')
    False
    """
    filter_specs = (
        ('_menu_style_filter', MenuStyleFilter),
        ('_dropdown_style_filter', DropDownStyleFilter),
    )
    for attribute, filter_class in filter_specs:
        if getattr(app, attribute, None) is not None:
            continue
        event_filter = filter_class(app)
        setattr(app, attribute, event_filter)
        app.installEventFilter(event_filter)

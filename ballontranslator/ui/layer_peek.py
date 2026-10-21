"""Peek at the original page or the inpainted page without losing context.

The canvas stacks three things: the page image (``baseLayer`` paints the
original array), the inpainted layer, and the text items. Holding a key (or the
floating button) temporarily hides the layers above the one being peeked at,
then restores exactly the previous visibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Optional, Tuple

from qtpy.QtCore import QEvent, QObject, QPoint, Qt
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from qtpy.QtWidgets import QApplication, QToolButton, QWidget


class PeekMode(Enum):
    """Which layer stack the canvas should show."""

    CURRENT = 'current'
    ORIGINAL = 'original'
    INPAINTED = 'inpainted'


BUTTON_GLYPHS = {
    # Japanese for the untouched page, Chinese for the current one, and an
    # empty bubble for the page with its lettering removed.
    PeekMode.CURRENT: '\u539f',
    PeekMode.ORIGINAL: '\u5143',
    PeekMode.INPAINTED: '',
}

BUTTON_COLORS = {
    PeekMode.CURRENT: '#3b82f6',
    PeekMode.ORIGINAL: '#60a5fa',
    PeekMode.INPAINTED: '#1d4ed8',
}

_HANDWRITING_FAMILIES = (
    'KaiTi',
    'STKaiti',
    'Kaiti SC',
    'Yu Mincho',
    '\u6977\u4f53',
    '\u534e\u6587\u884c\u6977',
    'Segoe Print',
)


def _handwriting_font(size: int) -> QFont:
    """Prefer a brush-like face so the glyph reads as hand lettering."""
    from qtpy.QtGui import QFontDatabase

    families = set(QFontDatabase.families())
    font = QFont()
    for family in _HANDWRITING_FAMILIES:
        if family in families:
            font.setFamily(family)
            break
    font.setPixelSize(size)
    font.setBold(False)
    font.setItalic(True)
    return font


def _bubble_path(size: int):
    """A round speech bubble with the classic manga tail."""
    from qtpy.QtGui import QPainterPath

    inset = size * 0.06
    diameter = size - inset * 2
    path = QPainterPath()
    path.addEllipse(inset, inset, diameter, diameter)
    tail = QPainterPath()
    tail.moveTo(size * 0.28, size * 0.80)
    tail.lineTo(size * 0.10, size * 0.99)
    tail.lineTo(size * 0.46, size * 0.90)
    tail.closeSubpath()
    return path.united(tail)


def _draw_screentone(painter, size: int) -> None:
    """Light halftone dots, the way a manga background is shaded."""
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(150, 160, 175, 140))
    step = max(3.0, size / 13.0)
    radius = max(0.7, step * 0.22)
    y = step / 2
    row = 0
    while y < size:
        x = step / 2 + (step / 2 if row % 2 else 0.0)
        while x < size:
            painter.drawEllipse(QPointF(x, y), radius, radius)
            x += step
        y += step
        row += 1


def make_peek_icon(mode: PeekMode, size: int = 44) -> QIcon:
    """Draw one state as a screentoned speech bubble.

    >>> from qtpy.QtWidgets import QApplication
    >>> _ = QApplication.instance() or QApplication([])
    >>> isinstance(make_peek_icon(PeekMode.ORIGINAL), QIcon)
    True
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    bubble = _bubble_path(size)
    painter.save()
    painter.setClipPath(bubble)
    painter.fillRect(0, 0, size, size, QColor(252, 252, 250))
    _draw_screentone(painter, size)
    painter.restore()

    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(BUTTON_COLORS[mode]), max(1.6, size * 0.045)))
    painter.drawPath(bubble)

    glyph = BUTTON_GLYPHS[mode]
    if glyph:
        painter.setPen(QColor(30, 32, 36))
        painter.setFont(_handwriting_font(int(size * 0.46)))
        painter.drawText(
            QRectF(0, size * 0.06, size, size * 0.86),
            Qt.AlignmentFlag.AlignCenter,
            glyph,
        )
    painter.end()
    return QIcon(pixmap)


class PeekController(QObject):
    """Switch the canvas between the three states and restore exactly."""

    def __init__(
        self,
        canvas,
        text_items: callable,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.canvas = canvas
        self._text_items = text_items
        self.mode = PeekMode.CURRENT
        self._saved: List[Tuple[QObject, bool]] = []
        self._saved_pixmap = None

    def _items(self) -> Iterable:
        try:
            return list(self._text_items() or [])
        except RuntimeError:
            return []

    def _inpaint_layer(self):
        return getattr(self.canvas, 'inpaintLayer', None)

    def _original_array(self):
        project = getattr(self.canvas, 'imgtrans_proj', None)
        return getattr(project, 'img_array', None)

    def _show_original_page(self) -> None:
        """Paint the untouched page; the base normally shows the inpaint."""
        layer = self._inpaint_layer()
        array = self._original_array()
        if layer is None or array is None:
            return
        from ballontranslator.ui.misc import ndarray2pixmap

        self._saved_pixmap = layer.pixmap()
        layer.setPixmap(ndarray2pixmap(array))

    def _restore(self) -> None:
        if self._saved_pixmap is not None:
            layer = self._inpaint_layer()
            if layer is not None:
                try:
                    layer.setPixmap(self._saved_pixmap)
                except RuntimeError:
                    pass
            self._saved_pixmap = None
        for obj, visible in self._saved:
            try:
                obj.setVisible(visible)
            except RuntimeError:
                continue
        self._saved = []

    def _hide(self, obj) -> None:
        if obj is None:
            return
        try:
            self._saved.append((obj, obj.isVisible()))
            obj.setVisible(False)
        except RuntimeError:
            return

    def set_mode(self, mode: PeekMode) -> None:
        """Apply ``mode``; switching away from CURRENT restores first."""
        if mode is self.mode:
            return
        self._restore()
        if mode is PeekMode.ORIGINAL:
            self._show_original_page()
        if mode is not PeekMode.CURRENT:
            for item in self._items():
                self._hide(item)
        self.mode = mode

    def reset(self) -> None:
        """Force the canvas back to the current state."""
        self._restore()
        self.mode = PeekMode.CURRENT


class LayerPeekButton(QToolButton):
    """A draggable corner button: hold to peek, release to return.

    The left button shows the untouched page, the right button shows the page
    with its lettering removed; letting go always returns to the current view.
    """

    def __init__(self, controller: PeekController, parent: QWidget) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName('LayerPeekButton')
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setAutoRaise(False)
        self._drag_from: Optional[QPoint] = None
        self._dragged = False
        self._peeking = False
        self.user_moved = False
        self.setIconSize(self.size())
        self.setIcon(make_peek_icon(PeekMode.CURRENT, self._icon_size()))
        self.setFixedSize(self._button_size(), self._button_size())
        self._apply_style()
        self.refresh()

    @staticmethod
    def _button_size() -> int:
        return 52

    def _icon_size(self) -> int:
        return self._button_size() - 8

    def _apply_style(self) -> None:
        radius = self._button_size() // 2
        self.setStyleSheet(
            'QToolButton#LayerPeekButton { border: none;'
            ' background: transparent; padding: 0px;'
            ' border-radius: %dpx; }' % radius
        )

    def begin_peek(self, mode: PeekMode) -> None:
        """Start a hold; ignored while another hold is running."""
        if self._peeking:
            return
        self._peeking = True
        self.controller.set_mode(mode)
        self.refresh()

    def end_peek(self) -> None:
        if not self._peeking:
            return
        self._peeking = False
        self.controller.reset()
        self.refresh()

    def refresh(self) -> None:
        self.setIcon(make_peek_icon(
            self.controller.mode, self._icon_size()
        ))
        self.setToolTip({
            PeekMode.CURRENT: '当前图（按住左键看原图 / 右键看修复图）',
            PeekMode.ORIGINAL: '原图（松开返回）',
            PeekMode.INPAINTED: '修复图（松开返回）',
        }[self.controller.mode])

    # -- mouse ------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        button = event.button()
        if button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._drag_from = event.position().toPoint()
            self._dragged = False
            self.begin_peek(
                PeekMode.ORIGINAL
                if button == Qt.MouseButton.LeftButton
                else PeekMode.INPAINTED
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None:
            return super().mouseMoveEvent(event)
        delta = event.position().toPoint() - self._drag_from
        if not self._dragged and delta.manhattanLength() < 4:
            return
        self._dragged = True
        self.user_moved = True
        target = self.pos() + delta
        parent = self.parentWidget()
        if parent is not None:
            bounds = parent.rect().adjusted(
                0, 0, -self.width(), -self.height()
            )
            target.setX(max(bounds.left(), min(target.x(), bounds.right())))
            target.setY(max(bounds.top(), min(target.y(), bounds.bottom())))
        self.move(target)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
        self.end_peek()
        event.accept()

    def contextMenuEvent(self, event) -> None:
        # The right button is a peek gesture, not a menu gesture.
        event.accept()


def _parse_shortcut(text: str):
    """Turn ``'Ctrl+Tab'`` into a key code and its modifier mask.

    >>> _parse_shortcut('Ctrl+Tab')[0] == Qt.Key.Key_Tab
    True
    >>> _parse_shortcut('')[0] is None
    True
    """
    text = (text or '').strip()
    if not text:
        return None, Qt.KeyboardModifier.NoModifier
    sequence = QKeySequence(text)
    if sequence.isEmpty():
        return None, Qt.KeyboardModifier.NoModifier
    combined = sequence[0]
    return int(combined.key()), int(combined.keyboardModifiers().value)


def peek_key_map(original: str, inpainted: str) -> dict:
    """Map (key, modifiers) pairs to peek modes from the configured text.

    >>> modes = peek_key_map('Ctrl+Tab', 'Alt+Tab')
    >>> sorted(mode.value for mode in modes.values())
    ['inpainted', 'original']
    >>> peek_key_map('', '')
    {}
    """
    modes = {}
    for text, mode in (
        (original, PeekMode.ORIGINAL),
        (inpainted, PeekMode.INPAINTED),
    ):
        key, modifiers = _parse_shortcut(text)
        if key is None:
            continue
        modes[(key, int(modifiers))] = mode
    return modes


class PeekResizeWatcher(QObject):
    """Call back when the canvas view is resized, so the button follows."""

    def __init__(self, on_resize, parent=None) -> None:
        super().__init__(parent)
        self._on_resize = on_resize

    def eventFilter(self, obj, event) -> bool:
        if event.type() is QEvent.Type.Resize and self._on_resize is not None:
            self._on_resize()
        return False


class PeekKeyFilter(QObject):
    """Hold-to-peek while the canvas owns the focus.

    Tab is Qt's focus-navigation key, so it is only consumed when the canvas
    view has focus and no modal dialog is open; everywhere else it passes
    through untouched.
    """

    def __init__(self, controller: PeekController, view_getter, keys: dict,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._view_getter = view_getter
        self.keys = dict(keys)
        self._active = None

    def _canvas_focused(self) -> bool:
        """Only peek while the canvas owns the keyboard.

        Tab is Qt's focus-navigation key, so it has to pass through every
        other widget untouched. An empty focus counts as the canvas,
        because the view is the window's main widget.
        """
        window = QApplication.activeWindow()
        if window is None:
            return False
        view = self._view_getter()
        if view is None:
            return False
        focused = QApplication.focusWidget()
        if focused is None:
            return True
        if focused is view or focused is view.viewport():
            return True
        return view.isAncestorOf(focused)

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        if QApplication.activeModalWidget() is not None:
            return False
        if event.isAutoRepeat():
            return False
        combination = (int(event.key()), int(event.modifiers().value))
        mode = self.keys.get(combination)
        if mode is None:
            return False
        if not self._canvas_focused():
            return False
        if kind is QEvent.Type.KeyPress:
            self._active = mode
            self.controller.set_mode(mode)
            return True
        if self._active is None:
            return False
        self._active = None
        self.controller.set_mode(PeekMode.CURRENT)
        return True

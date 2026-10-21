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
    PeekMode.CURRENT: '当',
    PeekMode.ORIGINAL: '原',
    PeekMode.INPAINTED: '修',
}


def make_peek_icon(
    mode: PeekMode,
    size: int = 30,
    background: str = '#2b2b2b',
    foreground: str = '#f0f0f0',
) -> QIcon:
    """Draw one state's icon so no assets are needed.

    >>> from qtpy.QtWidgets import QApplication
    >>> _ = QApplication.instance() or QApplication([])
    >>> isinstance(make_peek_icon(PeekMode.ORIGINAL), QIcon)
    True
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(background), 2))
    painter.setBrush(QColor(background))
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 6, 6)
    painter.setPen(QPen(QColor(foreground), 1))
    font = QFont()
    font.setPixelSize(int(size * 0.52))
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(
        pixmap.rect(), Qt.AlignmentFlag.AlignCenter,
        BUTTON_GLYPHS[mode],
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

    def _items(self) -> Iterable:
        try:
            return list(self._text_items() or [])
        except RuntimeError:
            return []

    def _inpaint_layer(self):
        return getattr(self.canvas, 'inpaintLayer', None)

    def _restore(self) -> None:
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
            self._hide(self._inpaint_layer())
        if mode is not PeekMode.CURRENT:
            for item in self._items():
                self._hide(item)
        self.mode = mode

    def reset(self) -> None:
        """Force the canvas back to the current state."""
        self._restore()
        self.mode = PeekMode.CURRENT


class LayerPeekButton(QToolButton):
    """A draggable corner button showing what is being peeked at."""

    def __init__(self, controller: PeekController, parent: QWidget) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName('LayerPeekButton')
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setToolTip(
            '当前图 / 按住看原图（默认 Tab）/ 反引号看修复图'
        )
        self._drag_from: Optional[QPoint] = None
        self._dragged = False
        self.setIcon(make_peek_icon(PeekMode.CURRENT))
        self.setAutoRaise(True)
        self.clicked.connect(self._cycle)

    def _cycle(self) -> None:
        order = [PeekMode.CURRENT, PeekMode.ORIGINAL, PeekMode.INPAINTED]
        index = order.index(self.controller.mode)
        self.controller.set_mode(order[(index + 1) % len(order)])
        self.refresh()

    def refresh(self) -> None:
        self.setIcon(make_peek_icon(self.controller.mode))
        self.setToolTip({
            PeekMode.CURRENT: '当前图（点击切换 / 按住 Tab 看原图）',
            PeekMode.ORIGINAL: '原图（松开恢复）',
            PeekMode.INPAINTED: '修复图（松开恢复）',
        }[self.controller.mode])

    # -- dragging ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.position().toPoint()
            self._dragged = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None:
            return super().mouseMoveEvent(event)
        delta = event.position().toPoint() - self._drag_from
        if not self._dragged and delta.manhattanLength() < 4:
            return
        self._dragged = True
        target = self.pos() + delta
        parent = self.parentWidget()
        if parent is not None:
            bounds = parent.rect().adjusted(0, 0, -self.width(), -self.height())
            target.setX(max(bounds.left(), min(target.x(), bounds.right())))
            target.setY(max(bounds.top(), min(target.y(), bounds.bottom())))
        self.move(target)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
        super().mouseReleaseEvent(event)


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
        window = QApplication.activeWindow()
        if window is None:
            return False
        view = self._view_getter()
        if view is None:
            return False
        return view.hasFocus()

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

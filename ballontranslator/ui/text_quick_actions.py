"""Quick actions beside a selected text block: scale text and free wrapping.

Two small buttons float at the right edge of the active block. Dragging the
``T`` handle changes the font size (down grows, up shrinks, Shift applies to
every block on the page); the second button lets the text ignore its box so it
wraps naturally instead of being forced at the frame edge.
"""

from __future__ import annotations

from typing import Callable, List

from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QToolButton, QWidget

BUTTON_SIZE = 30
DRAG_SPAN = 240.0
MIN_FONT_SIZE = 4.0
MAX_FONT_SIZE = 400.0


def scaled_font_size(base: float, dy: float) -> float:
    """Return the font size for a vertical drag of ``dy`` pixels.

    Dragging down grows the text and dragging up shrinks it, with the same drag
    always giving the same result so a gesture can be repeated.

    >>> round(scaled_font_size(50.0, 0.0), 2)
    50.0
    >>> round(scaled_font_size(50.0, DRAG_SPAN), 2)
    100.0
    >>> round(scaled_font_size(50.0, -DRAG_SPAN / 2), 2)
    25.0
    >>> scaled_font_size(50.0, -DRAG_SPAN)
    4.0
    """
    size = base * (1.0 + dy / DRAG_SPAN)
    return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, size))


def restore_original_height(item) -> bool:
    """Shrink a frame back to the height it had before free wrapping.

    The layout grows the frame to hold unconstrained text; turning the switch
    off should put the frame back rather than leave it stretched. A block that
    was already free-wrapped when the page loaded has no remembered height, so
    it falls back to fitting its content.

    >>> callable(restore_original_height)
    True
    """
    height = getattr(item, 'free_wrap_original_height', None)
    rect = item.boundingRect()
    try:
        if height is not None and height < rect.height():
            item.set_size(
                rect.width(), float(height), set_layout_maxsize=True
            )
            item.layout.reLayoutEverything()
            return True
        if height is None:
            item.squeezeBoundingRect()
            item.layout.reLayoutEverything()
            return True
    except (RuntimeError, AttributeError):
        return False
    return False


class _ScaleHandle(QToolButton):
    """The ``T`` button: press, drag vertically, release."""

    def __init__(self, actions: 'TextQuickActions', parent: QWidget) -> None:
        super().__init__(parent)
        self.actions = actions
        self.setText('T')
        self.setToolTip('拖动：向下放大 / 向上缩小（按住 Shift：整页）')

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.actions.start_scale(event.globalPosition().y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self.actions.drag_scale(event.globalPosition().y())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self.actions.end_scale()
        event.accept()


class _WrapHandle(QToolButton):
    """The free-wrapping button: one click toggles the block's mode."""

    def __init__(self, actions: 'TextQuickActions', parent: QWidget) -> None:
        super().__init__(parent)
        self.actions = actions
        self.setText('\u21b5')
        self.setToolTip('点击：文字不受文字框限制（按住 Shift：整页）')


class TextQuickActions:
    """Own the two buttons and apply their gestures."""

    STYLE = (
        'QToolButton { border: 1px solid #8a8f99; border-radius: %dpx;'
        ' background: rgba(255, 255, 255, 225); font-size: 15px;'
        ' font-weight: 600; color: #22252b; }'
        'QToolButton:hover { background: rgba(240, 244, 252, 245); }'
        % (BUTTON_SIZE // 2)
    )

    def __init__(
        self,
        view: QWidget,
        page_items: Callable[[], List],
    ) -> None:
        self.view = view
        self.page_items = page_items
        self.target = None

        self.scale_button = _ScaleHandle(self, view)
        self.scale_button.setObjectName('TextScaleHandle')
        self.wrap_button = _WrapHandle(self, view)
        self.wrap_button.setObjectName('TextWrapHandle')
        for button in (self.scale_button, self.wrap_button):
            button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
            button.setCursor(Qt.CursorShape.ArrowCursor)
            button.setStyleSheet(self.STYLE)
            button.setVisible(False)

        self.wrap_button.clicked.connect(self._on_wrap_clicked)

        self._base_size = 0.0
        self._press_y = 0.0
        self._dragging = False
        self._all_blocks = False

    # -- geometry ---------------------------------------------------------
    def set_target(self, item) -> None:
        self.target = item
        self.refresh()

    def refresh(self) -> None:
        item = self.target
        buttons = (self.scale_button, self.wrap_button)
        if item is None:
            for button in buttons:
                button.setVisible(False)
            return
        try:
            if not item.isVisible():
                raise RuntimeError
            rect = item.mapToScene(item.boundingRect()).boundingRect()
        except RuntimeError:
            for button in buttons:
                button.setVisible(False)
            return
        anchor = self.view.mapFromScene(rect.topRight())
        middle = self.view.mapFromScene(rect.center())
        x = max(0, anchor.x() + 8)
        self.scale_button.move(x, max(0, middle.y() - BUTTON_SIZE - 4))
        self.wrap_button.move(x, max(0, middle.y() + 4))
        for button in buttons:
            button.setVisible(True)

    # -- helpers ----------------------------------------------------------
    def _items(self, all_blocks: bool) -> List:
        if all_blocks:
            try:
                return [item for item in self.page_items() if item is not None]
            except RuntimeError:
                return []
        return [self.target] if self.target is not None else []

    @staticmethod
    def _shift_down() -> bool:
        from qtpy.QtWidgets import QApplication

        return bool(
            QApplication.keyboardModifiers()
            & Qt.KeyboardModifier.ShiftModifier
        )

    def _capture_frames(self) -> None:
        """Remember each target's frame so scaling never resizes it."""
        self._frames = {}
        for item in self._items(self._all_blocks):
            try:
                rect = item.boundingRect()
            except RuntimeError:
                continue
            self._frames[id(item)] = (rect.width(), rect.height())

    def _restore_frames(self) -> None:
        """Put the captured frame size back after the layout reflowed."""
        for item in self._items(self._all_blocks):
            frame = getattr(self, '_frames', {}).get(id(item))
            if frame is None:
                continue
            try:
                rect = item.boundingRect()
                if (
                    abs(rect.width() - frame[0]) < 0.5
                    and abs(rect.height() - frame[1]) < 0.5
                ):
                    continue
                item.set_size(
                    frame[0], frame[1], set_layout_maxsize=True
                )
            except (RuntimeError, AttributeError):
                continue

    def _apply_size(self, size: float) -> None:
        """Apply a size in the model's pixels, like the toolbar control.

        ``FontFormat.font_size`` is stored in pixels while ``setFontSize``
        takes points, and the toolbar converts between them. Skipping the
        conversion made every drag jump far past the intended size, which
        in turn forced the frame to grow.
        """
        from ballontranslator.utils.fontformat import px2pt

        for item in self._items(self._all_blocks):
            try:
                item.fontformat.font_size = size
                item.setFontSize(px2pt(size))
            except RuntimeError:
                continue
        self._restore_frames()
        self.refresh()

    # -- scale gesture ----------------------------------------------------
    def start_scale(self, global_y: float) -> None:
        item = self.target
        if item is None:
            return
        self._all_blocks = self._shift_down()
        self._base_size = float(item.fontformat.font_size)
        self._press_y = float(global_y)
        self._dragging = True
        self._capture_frames()

    def drag_scale(self, global_y: float) -> None:
        if not self._dragging:
            return
        self._apply_size(
            scaled_font_size(self._base_size, float(global_y) - self._press_y)
        )

    def end_scale(self) -> None:
        self._dragging = False
        self._restore_frames()

    # -- wrap toggle ------------------------------------------------------
    def _on_wrap_clicked(self) -> None:
        items = self._items(self._shift_down())
        if not items:
            return
        enabled = not bool(
            getattr(items[0].fontformat, 'no_forced_wrap', False)
        )
        for item in items:
            try:
                if enabled:
                    # Remember the frame the user chose, so the switch is
                    # reversible instead of leaving a grown frame behind.
                    item.free_wrap_original_height = (
                        item.boundingRect().height()
                    )
                item.fontformat.no_forced_wrap = enabled
                item.layout.reLayoutEverything()
                if not enabled:
                    restore_original_height(item)
            except (RuntimeError, AttributeError):
                continue
        self.refresh()

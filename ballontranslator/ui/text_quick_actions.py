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


def drag_progress(dx: float, dy: float) -> float:
    """Combine a pointer delta into one "enlarge" amount.

    Moving right or up enlarges, moving left or down shrinks, so the same
    diagonal gesture always produces the same change.

    >>> drag_progress(0, 0)
    0
    >>> drag_progress(120, -120)
    240
    >>> drag_progress(-120, 120)
    -240
    """
    return dx - dy


def scaled_font_size(base: float, progress: float) -> float:
    """Return the font size for a drag of ``progress`` pixels.

    ``progress`` comes from :func:`drag_progress`, so dragging up or right
    enlarges. The same drag always gives the same result, which keeps a gesture
    repeatable.

    >>> round(scaled_font_size(50.0, 0.0), 2)
    50.0
    >>> round(scaled_font_size(50.0, DRAG_SPAN), 2)
    100.0
    >>> round(scaled_font_size(50.0, -DRAG_SPAN / 2), 2)
    25.0
    >>> scaled_font_size(50.0, -DRAG_SPAN)
    4.0
    """
    size = base * (1.0 + progress / DRAG_SPAN)
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
            self.actions.start_scale(
                event.globalPosition().y(), event.globalPosition().x()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self.actions.drag_scale(
            event.globalPosition().y(), event.globalPosition().x()
        )
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
        self._last_size = 0.0
        self._press_y = 0.0
        self._press_x = 0.0
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

    def _snapshot(self, item) -> dict:
        """Capture everything a scale step has to be able to restore."""
        from ballontranslator.utils.fontformat import px2pt

        rect = item.boundingRect()
        centre = item.mapToScene(rect.center())
        size = float(item.fontformat.font_size)
        return {
            'size': size,
            'pt': px2pt(size),
            'w': rect.width(),
            'h': rect.height(),
            'cx': centre.x(),
            'cy': centre.y(),
        }

    def _snapshot(self, item) -> dict:
        """Capture everything a scale step has to be able to restore."""
        from ballontranslator.utils.fontformat import px2pt

        rect = item.boundingRect()
        centre = item.mapToScene(rect.center())
        size = float(item.fontformat.font_size)
        return {
            'size': size,
            'pt': px2pt(size),
            'w': rect.width(),
            'h': rect.height(),
            'cx': centre.x(),
            'cy': centre.y(),
        }

    def _capture_frames(self) -> None:
        """Remember each target's frame and snapshot before the gesture."""
        self._frames = {}
        self._initial = {}
        self._initial_items = []
        for item in self._items(self._all_blocks):
            try:
                snapshot = self._snapshot(item)
            except RuntimeError:
                continue
            self._frames[id(item)] = (
                snapshot['w'], snapshot['h'],
                item.mapToScene(item.boundingRect().center()),
            )
            self._initial[id(item)] = snapshot
            self._initial_items.append(item)

    def _scale_frame(self, item, ratio: float) -> None:
        """Resize the frame by ``ratio`` and keep its centre where it was."""
        frame = getattr(self, '_frames', {}).get(id(item))
        if frame is None:
            return
        width, height, centre = frame
        try:
            item.set_size(
                width * ratio, height * ratio, set_layout_maxsize=True
            )
            current = item.mapToScene(item.boundingRect().center())
            item.setPos(item.pos() + (centre - current))
            item.layout.reLayoutEverything()
        except (RuntimeError, AttributeError):
            return

    def _apply_size(self, size: float) -> None:
        """Scale text and frame as one unit, like resizing a picture.

        ``FontFormat.font_size`` is stored in pixels while ``setFontSize``
        takes points, and the toolbar control converts between them.

        >>> callable(TextQuickActions._apply_size)
        True
        """
        from ballontranslator.utils.fontformat import px2pt

        self._last_size = size
        ratio = size / self._base_size if self._base_size else 1.0
        for item in self._items(self._all_blocks):
            try:
                item.fontformat.font_size = size
                item.setFontSize(px2pt(size))
            except RuntimeError:
                continue
            self._scale_frame(item, ratio)
        self.refresh()

    # -- scale gesture ----------------------------------------------------
    def start_scale(self, global_y: float, global_x: float = 0.0) -> None:
        item = self.target
        if item is None:
            return
        self._all_blocks = self._shift_down()
        self._base_size = float(item.fontformat.font_size)
        self._last_size = self._base_size
        self._press_y = float(global_y)
        self._press_x = float(global_x)
        self._dragging = True
        self._capture_frames()

    def drag_scale(self, global_y: float, global_x: float = 0.0) -> None:
        if not self._dragging:
            return
        progress = drag_progress(
            float(global_x) - self._press_x,
            float(global_y) - self._press_y,
        )
        self._apply_size(scaled_font_size(self._base_size, progress))

    def end_scale(self) -> None:
        self._dragging = False
        initial = getattr(self, '_initial', {})
        items = getattr(self, '_initial_items', [])
        if not items:
            return
        changed = any(
            abs(initial[id(item)]['size']
                - float(item.fontformat.font_size)) > 0.01
            for item in items if id(item) in initial
        )
        if not changed:
            return
        from ballontranslator.ui.text_engine.formatting.commands import (
            push_scale_snapshot,
        )

        canvas = self.view.scene() if hasattr(self.view, 'scene') else None
        before = [initial[id(item)] for item in items]
        after = [self._snapshot(item) for item in items]
        push_scale_snapshot(canvas, items, before, after)

    # -- wrap toggle ------------------------------------------------------
    def _on_wrap_clicked(self) -> None:
        items = self._items(self._shift_down())
        if not items:
            return
        enabled = not bool(
            getattr(items[0].fontformat, 'no_forced_wrap', False)
        )
        if enabled:
            for item in items:
                try:
                    # Remember the frame the user chose, so the switch is
                    # reversible instead of leaving a grown frame behind.
                    item.free_wrap_original_height = (
                        item.boundingRect().height()
                    )
                except RuntimeError:
                    continue

        from ballontranslator.ui.text_engine.formatting.commands import (
            push_no_forced_wrap,
        )

        # Pushing onto the text undo stack is what marks the project unsaved,
        # so the toggle survives a page switch or a restart.
        canvas = self.view.scene() if hasattr(self.view, 'scene') else None
        push_no_forced_wrap(canvas, items, enabled)

        if not enabled:
            for item in items:
                try:
                    restore_original_height(item)
                except RuntimeError:
                    continue
        self.refresh()

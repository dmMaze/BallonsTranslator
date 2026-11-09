"""Spacing scrubs preview detached text and commit one document edit per item."""

from copy import deepcopy
from math import isclose
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QRectF, QSignalBlocker, Qt
from qtpy.QtGui import QTextCharFormat, QTextCursor
from qtpy.QtWidgets import QGraphicsItem
try:
    from qtpy.QtGui import QUndoCommand
except ImportError:
    from qtpy.QtWidgets import QUndoCommand

from ballontranslator.utils.fontformat import LineSpacingType
from ... import shared_widget as SW
from ...custom_widget import SizeComboBox
from ..item import TextBlkItem
from .commands import restore_canvas_view_focus

if TYPE_CHECKING:
    from .panel import FontFormatPanel, FormattingDragLabel


class SpacingEditCommand(QUndoCommand):
    """Keep spacing defaults, insertion formats, bounds and document undo together.

    >>> callable(SpacingEditCommand.undo)
    True
    """

    def __init__(
        self, session: 'SpacingDragSession', items: list[TextBlkItem], value: float,
    ) -> None:
        super().__init__()
        self.session = session
        self.items = items
        self.value = value
        self.before = [self._snapshot(item) for item in items]
        self.after: list[tuple[QRectF, float, int | None, QTextCharFormat]] = []
        self.changed: list[bool] = []

    def _snapshot(self, item: TextBlkItem) -> tuple[QRectF, float, int | None, QTextCharFormat]:
        cursor = item.textCursor()
        return (
            QRectF(item.absBoundingRect(qrect=True)),
            getattr(item.fontformat, self.session.param_name),
            # A live QTextCursor moves when later edits delete its position.
            None if cursor.hasSelection() else cursor.position(),
            QTextCharFormat(cursor.charFormat()),
        )

    def _restore(
        self, states: list[tuple[QRectF, float, int | None, QTextCharFormat]], redo: bool,
    ) -> None:
        for item, changed, (rect, default, position, char_format) in zip(self.items, self.changed, states):
            was_repainting = item.repaint_on_changed
            item.repaint_on_changed = False
            try:
                setattr(item.fontformat, self.session.param_name, default)
                setattr(item.layout, self.session.param_name, default)
                if changed:
                    if redo:
                        item.redo()
                    else:
                        item.undo()
                elif position is not None:
                    # Insertion formatting has no QTextDocument undo command.
                    cursor = QTextCursor(item.document())
                    cursor.setPosition(position)
                    cursor.setCharFormat(char_format)
                    item.setTextCursor(cursor)
            finally:
                item.repaint_on_changed = was_repainting
            item.setRect(rect)
        self.session.refresh()

    def redo(self) -> None:
        if self.after:
            self._restore(self.after, True)
            return
        for item in self.items:
            steps = item.document().availableUndoSteps()
            was_replaying = item.in_redo_undo
            item.in_redo_undo = True
            try:
                self.session.apply(item, self.value)
            finally:
                item.in_redo_undo = was_replaying
                item.updateUndoSteps()
            self.changed.append(item.old_undo_steps != steps)
            self.after.append(self._snapshot(item))
        self.session.refresh()

    def undo(self) -> None:
        self._restore(self.before, False)


class SpacingDragSession(QObject):
    """Render scratch items without consuming the original documents' history.

    >>> callable(SpacingDragSession.cancel)
    True
    """

    def __init__(
        self, panel: 'FontFormatPanel', label: 'FormattingDragLabel',
        param_name: str, box: SizeComboBox,
    ) -> None:
        super().__init__(panel)
        self.panel = panel
        self.label = label
        self.param_name = param_name
        self.box = box
        self.items: list[TextBlkItem] = []
        self.previews: list[TextBlkItem] = []
        self.opacities: list[float] = []
        self.value = self.start_value = 0.0
        self.multiplier = 0.01
        self.active = False
        label.drag_started.connect(self.begin)
        label.size_ctrl_changed.connect(self.preview)
        label.btn_released.connect(self.commit)
        label.drag_canceled.connect(self.cancel)
        panel.destroyed.connect(self.cancel)

    def begin(self) -> None:
        self.panel.resolve_text_transform_edits_for_save()
        item = self.panel.textblk_item
        if item is not None:
            self.items = [item]
        elif SW.canvas is not None:
            self.items = SW.canvas.selected_text_items()
        source = item
        if source is None and self.items:
            source = SW.canvas.primary_selected_text_item(self.items)
        source_format = source.get_fontformat() if source is not None else self.panel.global_format
        self.value = self.start_value = getattr(source_format, self.param_name)
        self.multiplier = (
            0.1 if self.param_name == 'line_spacing'
            and source_format.line_spacing_type == LineSpacingType.Distance
            else 0.01
        )
        self.active = True

    def apply(self, item: TextBlkItem, value: float) -> None:
        if self.param_name == 'letter_spacing':
            item.setLetterSpacing(value)
        else:
            item.setLineSpacing(value)

    def _create_previews(self) -> None:
        for item in self.items:
            block = deepcopy(item.blk)
            block.rich_text = item.toHtml()
            block._bounding_rect = item.absBoundingRect()
            block.angle = item.rotation()
            preview = TextBlkItem(block, item.idx)
            self.previews.append(preview)
            self.opacities.append(item.opacity())
            preview.setTextInteractionFlags(item.textInteractionFlags())
            cursor = preview.textCursor()
            original = item.textCursor()
            cursor.setPosition(original.anchor())
            cursor.setPosition(original.position(), QTextCursor.MoveMode.KeepAnchor)
            if not original.hasSelection():
                cursor.setCharFormat(original.charFormat())
            preview.setTextCursor(cursor)
            preview.document().clearUndoRedoStacks()
            preview.setRect(item.absBoundingRect(qrect=True))
            preview.setFlags(QGraphicsItem.GraphicsItemFlag(0))
            preview.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            preview.setAcceptHoverEvents(False)
            preview.setParentItem(item.parentItem())
            if preview.scene() is None and item.scene() is not None:
                item.scene().addItem(preview)
            preview.setZValue(item.zValue())
            preview.stackBefore(item)
            # Native opacity hides the source without changing selection or
            # the persisted effect opacity; the scratch item owns all layout.
            item._set_effective_opacity(0)

    def preview(self, delta: int) -> None:
        if not self.active or not delta:
            return
        value = min(
            self.box.max_val,
            max(self.box.min_val, self.value + delta * self.multiplier),
        )
        if isclose(value, self.start_value, rel_tol=0, abs_tol=1e-9):
            value = self.start_value
        if value == self.value:
            return
        self.value = value
        self._display(value)
        if not self.previews:
            self._create_previews()
        for item, preview in zip(self.items, self.previews):
            preview.repaint_on_changed = False
            try:
                # Rebase only the scratch item; the real setter owns growth
                # and rotated alignment, including when a drag reverses.
                default = getattr(item.fontformat, self.param_name)
                setattr(preview.fontformat, self.param_name, default)
                setattr(preview.layout, self.param_name, default)
                preview.undo()
                rect = item.absBoundingRect(qrect=True)
                if preview.absBoundingRect(qrect=True) != rect:
                    preview.setRect(rect, repaint=False)
                self.apply(preview, value)
            finally:
                preview.repaint_on_changed = True
            preview.repaint_background()

    def _display(self, value: float) -> None:
        with QSignalBlocker(self.box):
            self.box.setValue(value)

    def commit(self) -> None:
        if not self.active:
            return
        items, value, changed = self.items, self.value, self.value != self.start_value
        self.cancel()
        if not changed:
            return
        if self.panel.global_mode():
            setattr(self.panel.global_format, self.param_name, value)
            self.panel.update_text_style_label()
        if items:
            SW.canvas.push_undo_command(SpacingEditCommand(self, items, value))
            restore_canvas_view_focus()
        else:
            self._display(value)

    def cancel(self) -> None:
        if not self.active:
            return
        self.active = False
        self.label.cancel_drag()
        for item, preview, opacity in zip(self.items, self.previews, self.opacities):
            item._set_effective_opacity(opacity)
            preview.geometry_controller.release_render_resources()
            if preview.scene() is not None:
                preview.scene().removeItem(preview)
            preview.setParentItem(None)
            preview.deleteLater()
        self.previews = []
        self.items = []
        self.opacities = []
        self._display(self.start_value)

    def refresh(self) -> None:
        item = self.panel.textblk_item
        if item is not None:
            self.panel.sync_inline_format(
                item.get_fontformat(), not item.isEditing() and item.isMultiFontSize(),
            )
        elif SW.canvas is not None:
            item = SW.canvas.primary_selected_text_item()
            source = item.get_fontformat() if item is not None else self.panel.global_format
            self._display(getattr(source, self.param_name))

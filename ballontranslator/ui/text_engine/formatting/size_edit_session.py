"""Font-size scrubbing: transient whole-item geometry or a selected-range draft."""

from dataclasses import replace
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QRectF, QSignalBlocker
from qtpy.QtGui import QFont, QTextCharFormat, QTextCursor
try:
    from qtpy.QtGui import QUndoCommand
except ImportError:
    from qtpy.QtWidgets import QUndoCommand

from ballontranslator.utils.fontformat import (
    LineSpacingType, ProjectiveTextTransform, TextTransformStack, px2pt,
)
from ... import shared_widget as SW
from ..item import TextBlkItem
from ..annotations import apply_line_spacing, line_spacing_values

if TYPE_CHECKING:
    from .panel import FontFormatPanel


class ResizeTextItemsCommand(QUndoCommand):
    """Scale rich-text sizes and logical boxes in one canvas history step.

    Keep the document's own edit block so earlier typing history survives.

    >>> callable(ResizeTextItemsCommand.undo)
    True
    """

    def __init__(
        self, items: list[TextBlkItem], ratio: float,
        session: 'FontSizeEditSession',
    ) -> None:
        super().__init__()
        self.items = items
        self.ratio = ratio
        self.session = session
        self.before = [QRectF(item.absBoundingRect(qrect=True)) for item in items]
        self.after: list[QRectF] = []
        self.changed: list[bool] = []
        self.old_sizes = [item.fontformat.font_size for item in items]
        self.new_sizes = [item.get_fontformat().font_size * ratio for item in items]
        self.old_spacing = [item.fontformat.line_spacing for item in items]
        self.old_fonts = [item.document().defaultFont() for item in items]
        self.new_fonts: list[QFont] = []
        self.empty_formats = {
            index: QTextCharFormat(item.textCursor().charFormat())
            for index, item in enumerate(items) if item.document().isEmpty()
        }

    def _set_defaults(self, index: int, redo: bool) -> None:
        item = self.items[index]
        item.fontformat.font_size = self.new_sizes[index] if redo else self.old_sizes[index]
        if item.fontformat.line_spacing_type == LineSpacingType.Distance:
            item.fontformat.line_spacing = self.old_spacing[index] * (self.ratio if redo else 1.0)
            item.layout.line_spacing = item.fontformat.line_spacing
        if index in self.empty_formats:
            # An empty cursor's insertion format is not document undo state.
            cursor = item.textCursor()
            char_format = QTextCharFormat(self.empty_formats[index])
            if redo:
                char_format.setFontPointSize(px2pt(self.new_sizes[index]))
            cursor.setCharFormat(char_format)
            item.setTextCursor(cursor)

    def redo(self) -> None:
        first = not self.after
        for index, item in enumerate(self.items):
            if first:
                rect = QRectF(self.before[index])
                center = rect.center()
                rect.setSize(rect.size() * self.ratio)
                rect.moveCenter(center)
                steps = item.document().availableUndoSteps()
                was_replaying = item.in_redo_undo
                was_repainting = item.repaint_on_changed
                item.in_redo_undo = True
                item.repaint_on_changed = False
                cursor = QTextCursor(item.document())
                cursor.beginEditBlock()
                try:
                    # Grow the box before the glyphs, avoiding auto-enlargement
                    # against the old width during the document edit.
                    item.setRect(rect, repaint=False, notify=False)
                    if item.document().isEmpty():
                        item.setFontSize(
                            px2pt(self.new_sizes[index]),
                            restore_cursor=True,
                        )
                    else:
                        item.setRelFontSize(self.ratio, restore_cursor=True)
                    block = item.document().firstBlock()
                    while block.isValid():
                        cursor.setPosition(block.position())
                        block_size = block.charFormat().fontPointSize()
                        if block_size > 0:
                            char_format = QTextCharFormat()
                            char_format.setFontPointSize(block_size * self.ratio)
                            cursor.mergeBlockCharFormat(char_format)
                        spacing, kind = line_spacing_values(
                            block.blockFormat(), self.old_spacing[index],
                            item.fontformat.line_spacing_type,
                        )
                        if kind == LineSpacingType.Distance:
                            apply_line_spacing(cursor, spacing * self.ratio, kind)
                        block = block.next()
                    font = QFont(self.old_fonts[index])
                    font.setPointSizeF(px2pt(self.new_sizes[index]))
                    item.document().setDefaultFont(font)
                    self._set_defaults(index, True)
                    cursor.endEditBlock()
                    cursor = None
                    item.setRect(rect, repaint=False)
                finally:
                    if cursor is not None:
                        cursor.endEditBlock()
                    item.in_redo_undo = was_replaying
                    item.repaint_on_changed = was_repainting
                    item.old_undo_steps = item.document().availableUndoSteps()
                self.changed.append(item.old_undo_steps != steps)
                self.after.append(QRectF(item.absBoundingRect(qrect=True)))
                self.new_fonts.append(item.document().defaultFont())
            else:
                self._set_defaults(index, True)
                if self.changed[index]:
                    item.redo()
                item.document().setDefaultFont(self.new_fonts[index])
                item.setRect(self.after[index], repaint=False)
            item.repaint_background()
        self.session.refresh()

    def undo(self) -> None:
        for index, item in enumerate(self.items):
            self._set_defaults(index, False)
            if self.changed[index]:
                item.undo()
            # QTextDocument does not include its default font in undo history.
            item.document().setDefaultFont(self.old_fonts[index])
            item.setRect(self.before[index], repaint=False)
            item.repaint_background()
        self.session.refresh()


class FontSizeEditSession(QObject):
    """Hold a stable drag target without publishing intermediate formatting.

    >>> callable(FontSizeEditSession.cancel)
    True
    """

    def __init__(self, host: 'FontFormatPanel') -> None:
        super().__init__(host)
        self.host = host
        self.box = host.fontsizebox
        self.items: list[TextBlkItem] = []
        self.before: list[TextTransformStack] = []
        self.selection: QTextCursor | None = None
        self.start_text: str | None = None
        self.start_size = 1.0
        self.ratio = 1.0
        self.value = 1.0
        self.delta = 0
        self.box.drag_label.drag_started.connect(self.begin)
        self.box.drag_label.size_ctrl_changed.connect(self.preview)
        self.box.drag_label.btn_released.connect(self.commit)
        self.box.drag_label.drag_canceled.connect(self.cancel)

    def begin(self) -> None:
        self.cancel()
        self.host.text_transform_session.resolve_for_save()
        self.host.text_effect_session.resolve_for_save()
        self.box.fcombobox.finish_edit()
        self.start_text = self.box.getFontSize()
        item = self.host.textblk_item
        self.items = [item] if item is not None else list(SW.canvas.selected_text_items())
        source_format = self.items[0].get_fontformat() if self.items else self.host.global_format
        self.start_size = source_format.font_size
        self.value = self.start_size
        self.before = [item.geometry_controller.canonical() for item in self.items]
        if item is not None and item.isEditing() and item.textCursor().hasSelection():
            self.selection = QTextCursor(item.textCursor())

    def preview(self, delta: int) -> None:
        if self.start_text is None or not delta:
            return
        self.delta += delta
        # Equal pointer distances give equal proportional changes at any size.
        exponent = min(20.0, max(-20.0, self.delta / 100.0))
        self.value = min(1000.0, max(1.0, self.start_size * 2 ** exponent))
        self.ratio = self.value / self.start_size
        self._display(self.value)
        if self.selection is not None:
            return
        ratio = self.ratio
        for item, state in zip(self.items, self.before):
            preview = replace(state, transforms=state.transforms + (
                ProjectiveTextTransform(horizontal_scale=ratio, vertical_scale=ratio),
            )) if ratio != 1 else state
            if item.set_text_transform(preview, preview=True):
                item.update()

    def _display(self, value: float) -> None:
        suffix = '+' if self.selection is None and self.start_text.endswith('+') else ''
        with QSignalBlocker(self.box.fcombobox):
            self.box.fcombobox.setCurrentText(f'{value:.1f}'.rstrip('0').rstrip('.') + suffix)

    def commit(self) -> None:
        if self.start_text is None:
            return
        value = round(self.value, 1)
        ratio = self.ratio
        items, selection = self.items, self.selection
        self.cancel()
        if ratio == 1:
            return
        if selection is not None:
            items[0].setTextCursor(selection)
            self.box.param_changed.emit('font_size', value)
        elif items:
            SW.canvas.push_undo_command(ResizeTextItemsCommand(items, ratio, self))
            return
        else:
            self.box.param_changed.emit('font_size', value)
        self.refresh()

    def cancel(self) -> None:
        if self.start_text is not None:
            for item in self.items:
                if item.clear_text_transform_preview():
                    item.update()
            self.box.fcombobox.set_committed_text(self.start_text)
        self.start_text = None
        self.items = []
        self.before = []
        self.selection = None
        self.delta = 0
        self.ratio = 1.0

    def refresh(self) -> None:
        item = self.host.textblk_item
        if item is not None:
            self.host.sync_inline_format(
                item.get_fontformat(), not item.isEditing() and item.isMultiFontSize()
            )
        else:
            items = SW.canvas.selected_text_items()
            if items:
                self.host.global_format.font_size = items[0].get_fontformat().font_size
            self.host.sync_inline_format(
                self.host.global_format,
                bool(items) and any(item.isMultiFontSize() for item in items),
            )

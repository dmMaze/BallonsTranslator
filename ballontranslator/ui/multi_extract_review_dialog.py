"""Review multi-pass extraction disagreements block by block.

The workflow never merges readings on its own: every block whose passes agree
on nothing keeps its payload, shows a red corner badge, and waits here. Picking
a reading applies it straight to the block's source text.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from qtpy.QtCore import QRectF, Qt, Signal
from qtpy.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.multi_extract import (
    diff_readings,
    has_conflict,
    iter_conflicts,
    readings,
    resolve,
)
from ballontranslator.utils.textblock import TextBlock

ALL_FILTER = 'all'
TEXT_FILTER = 'text'
WEAK_FILTER = 'weak'


def prefixed(text: str, limit: int = 28) -> str:
    """Single-line preview of a reading.

    >>> prefixed('そういえば')
    'そういえば'
    >>> prefixed('あ' * 40)
    'あああああああああああああああああああああああああああ…'
    """
    flat = ' '.join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + '…'


class _DiffTextLabel(QWidget):
    """A reading whose differing characters carry a red text outline.

    A ``QTextDocument`` is used instead of rich-text HTML because Qt's rich text
    has no CSS property for a text stroke. The document wraps to the label's
    width and paints itself with the theme's normal text colour.

    >>> _DiffTextLabel.DIFF_COLOR.red()
    214
    """

    DIFF_COLOR = QColor(214, 48, 49)
    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._segments: List[Tuple[str, bool]] = []
        self._height = 0
        self._document = QTextDocument(self)
        self._document.setDocumentMargin(0)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

    def set_segments(self, segments: Sequence[Tuple[str, bool]]) -> None:
        self._segments = list(segments)
        self._document.clear()
        self._document.setDefaultFont(self.font())
        normal = QTextCharFormat()
        normal.setForeground(self.palette().color(QPalette.ColorRole.Text))
        differing = QTextCharFormat(normal)
        differing.setTextOutline(QPen(self.DIFF_COLOR, 1.4))
        cursor = QTextCursor(self._document)
        for text, differs in self._segments:
            if text:
                cursor.insertText(text, differing if differs else normal)
        self._document.setTextWidth(max(1, self.width()))
        self._sync_height()

    def plain_text(self) -> str:
        return ''.join(text for text, _ in self._segments)

    def resizeEvent(self, event) -> None:
        self._document.setTextWidth(max(1, self.width()))
        self._sync_height()
        super().resizeEvent(event)

    def _sync_height(self) -> None:
        height = max(1, int(self._document.size().height()) + 2)
        if height != self._height:
            self._height = height
            self.setFixedHeight(height)
            self.updateGeometry()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            self._document.drawContents(
                painter, QRectF(0, 0, self.width(), self.height())
            )
        finally:
            painter.end()

    def mouseReleaseEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.pos())
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class MultiExtractReviewDialog(QDialog):
    """Pick a reading for each block the extraction passes disagreed on."""

    applied = Signal(int)

    def __init__(
        self,
        project,
        apply_callback: Optional[Callable[[str, TextBlock, int], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self._apply_callback = apply_callback
        self._page_cache = {}
        self._resolved = set()
        self._entries: List[Tuple[str, TextBlock]] = []
        self._radio_buttons: List[QRadioButton] = []
        self._reading_labels: List[_DiffTextLabel] = []
        self._button_group = QButtonGroup(self)

        self.setWindowTitle(self.tr('Review Multi-Pass Extraction'))
        self.resize(820, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top = QHBoxLayout()
        self.filter_combo = QComboBox(self)
        self.filter_combo.addItem(self.tr('All'), ALL_FILTER)
        self.filter_combo.addItem(self.tr('Text conflicts'), TEXT_FILTER)
        self.filter_combo.addItem(self.tr('Single-pass regions'), WEAK_FILTER)
        self.filter_combo.currentIndexChanged.connect(self._rebuild_list)
        self.summary_label = QLabel(self)
        top.addWidget(self.filter_combo)
        top.addWidget(self.summary_label)
        top.addStretch()
        layout.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(10)
        self.list_widget = QListWidget(self)
        self.list_widget.setFixedWidth(250)
        self.list_widget.setWordWrap(True)
        self.list_widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.list_widget.currentRowChanged.connect(self._show_current)
        body.addWidget(self.list_widget)

        detail = QVBoxLayout()
        detail.setSpacing(6)
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 150)
        self.preview_label.setStyleSheet('background: #202225; color: #bbb;')
        detail.addWidget(self.preview_label, 1)

        self.readings_label = QLabel(self.tr('Readings'), self)
        detail.addWidget(self.readings_label)
        self.readings_container = QWidget(self)
        self.readings_layout = QVBoxLayout(self.readings_container)
        self.readings_layout.setContentsMargins(0, 0, 0, 0)
        self.readings_layout.setSpacing(4)
        detail.addWidget(self.readings_container)
        self.hint_label = QLabel(self)
        self.hint_label.setWordWrap(True)
        detail.addWidget(self.hint_label)
        body.addLayout(detail, 1)
        layout.addLayout(body, 1)

        buttons = QHBoxLayout()
        self.status_label = QLabel(self)
        self.status_label.setStyleSheet('color: #d63031;')
        buttons.addWidget(self.status_label)
        buttons.addStretch()
        self.apply_button = QPushButton(self.tr('Apply to source text'), self)
        self.apply_button.clicked.connect(self.apply_current)
        self.close_button = QPushButton(self.tr('Close'), self)
        self.close_button.clicked.connect(self.accept)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self._rebuild_list()

    # ------------------------------------------------------------------ list

    def filter_mode(self) -> str:
        return self.filter_combo.currentData()

    def _matches(self, block: TextBlock) -> bool:
        if self.filter_mode() == WEAK_FILTER:
            return self.is_single_pass(block)
        if self.filter_mode() == TEXT_FILTER:
            return not self.is_single_pass(block) and self._text_differs(block)
        return True

    @staticmethod
    def is_single_pass(block: TextBlock) -> bool:
        payload = getattr(block, 'multi_extract', None) or {}
        return len(payload.get('support', [])) == 1

    @staticmethod
    def _text_differs(block: TextBlock) -> bool:
        texts = {' '.join(lines).strip() for _, lines in readings(block)}
        return len({text for text in texts if text}) > 1

    def _rebuild_list(self) -> None:
        self.list_widget.clear()
        self._entries = [
            (page, block)
            for page, block in iter_conflicts(self.project.pages)
            if self._matches(block)
        ]
        for page, block in self._entries:
            item = QListWidgetItem(self._item_label(page, block))
            self.list_widget.addItem(item)
        self._refresh_summary()
        if self._entries:
            self.list_widget.setCurrentRow(0)
        else:
            self._show_empty()

    def _item_label(self, page: str, block: TextBlock) -> str:
        page_blocks = self.project.pages.get(page, [])
        number = next(
            (index + 1 for index, candidate in enumerate(page_blocks)
             if candidate is block),
            0,
        )
        kind = self.tr('single pass') if self.is_single_pass(block) else self.tr('differs')
        preview = prefixed(block.get_text())
        return f'{page} #{number} · {kind}\n{preview}'

    def _refresh_summary(self) -> None:
        pending = sum(1 for _, block in iter_conflicts(self.project.pages))
        self.summary_label.setText(
            self.tr('{shown} shown · {pending} pending').format(
                shown=len(self._entries), pending=pending
            )
        )
        self.apply_button.setEnabled(bool(self._entries))

    # ---------------------------------------------------------------- detail

    def current_entry(self) -> Optional[Tuple[str, TextBlock]]:
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._entries):
            return None
        return self._entries[row]

    def _show_empty(self) -> None:
        self.preview_label.clear()
        self._clear_readings()
        self.hint_label.setText(self.tr('Nothing left to review.'))
        self.apply_button.setEnabled(False)

    def _show_current(self, _row: int) -> None:
        entry = self.current_entry()
        if entry is None:
            self._show_empty()
            return
        page, block = entry
        self.preview_label.setPixmap(self._crop_pixmap(page, block))
        self._rebuild_readings(block)
        self.status_label.clear()

    def _clear_readings(self) -> None:
        for button in self._radio_buttons:
            self._button_group.removeButton(button)
        self._radio_buttons = []
        self._reading_labels = []
        while self.readings_layout.count():
            item = self.readings_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _rebuild_readings(self, block: TextBlock) -> None:
        self._clear_readings()
        single = self.is_single_pass(block)
        options = readings(block)
        texts = [
            ' '.join(line for line in lines if line.strip())
            for _, lines in options
        ]
        diff_rows = diff_readings(texts)
        for index, (pass_index, _) in enumerate(options):
            row = QWidget(self.readings_container)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            label = self.tr('Pass {pass}').format(**{'pass': pass_index + 1})
            radio = QRadioButton(label, row)
            radio.setChecked(index == 0)
            radio.setEnabled(not single)
            self._button_group.addButton(radio, index)
            self._radio_buttons.append(radio)

            text_label = _DiffTextLabel(row)
            text_label.setProperty('reading_index', index)
            text_label.clicked.connect(self._on_reading_label_clicked)
            text_label.set_segments(diff_rows[index])
            self._reading_labels.append(text_label)

            row_layout.addWidget(radio, 0, Qt.AlignmentFlag.AlignTop)
            row_layout.addWidget(text_label, 1)
            self.readings_layout.addWidget(row)
        if single:
            self.hint_label.setText(
                self.tr(
                    'Only one pass detected this region. Nothing to choose; '
                    'the text is kept as detected.'
                )
            )
            self.apply_button.setEnabled(False)
        else:
            self.hint_label.setText(
                self.tr(
                    'Applying overwrites the source text of this block. '
                    'Red outlines mark the characters that differ between passes.'
                )
            )
            self.apply_button.setEnabled(True)

    def _on_reading_label_clicked(self) -> None:
        widget = self.sender()
        index = widget.property('reading_index') if widget is not None else None
        if isinstance(index, int) and 0 <= index < len(self._radio_buttons):
            self._radio_buttons[index].setChecked(True)

    def chosen_row(self) -> int:
        return self._button_group.checkedId()

    def _crop_pixmap(self, page: str, block: TextBlock) -> QPixmap:
        image = self._page_image(page)
        if image is None:
            return QPixmap()
        x, y, w, h = block.bounding_rect()
        pad = 8
        top = max(0, int(y) - pad)
        left = max(0, int(x) - pad)
        bottom = min(image.shape[0], int(y + h) + pad)
        right = min(image.shape[1], int(x + w) + pad)
        if bottom <= top or right <= left:
            return QPixmap()
        crop = np.ascontiguousarray(image[top:bottom, left:right])
        height, width = crop.shape[:2]
        qimage = QImage(
            crop.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).copy()
        return QPixmap.fromImage(qimage).scaled(
            320, 150,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _page_image(self, page: str) -> Optional[np.ndarray]:
        if page in self._page_cache:
            return self._page_cache[page]
        try:
            image = self.project.read_img(page)
        except Exception:
            image = None
        self._page_cache[page] = image
        return image

    # ---------------------------------------------------------------- action

    def apply_current(self) -> None:
        entry = self.current_entry()
        if entry is None:
            return
        page, block = entry
        if self.is_single_pass(block):
            return
        options = readings(block)
        row = self.chosen_row()
        if row < 0 or row >= len(options):
            return
        pass_index = options[row][0]
        if self._apply_callback is not None:
            self._apply_callback(page, block, pass_index)
        else:
            resolve(block, pass_index)
        if not has_conflict(block):
            self._resolved.add(id(block))
        self.applied.emit(len(self._resolved))
        self._advance_after_apply()

    def _advance_after_apply(self) -> None:
        row = self.list_widget.currentRow()
        remaining = [
            index
            for index, (_, block) in enumerate(self._entries)
            if has_conflict(block)
        ]
        if not remaining:
            self._resolved = set()
            self._rebuild_list()
            return
        next_row = next((index for index in remaining if index > row), remaining[0])
        self.list_widget.setCurrentRow(next_row)
        self._refresh_summary()

    def keyPressEvent(self, event) -> None:
        if event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ) and self.apply_button.isEnabled():
            self.apply_current()
            return
        if Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_3:
            index = event.key() - Qt.Key.Key_1
            if index < len(self._radio_buttons):
                self._radio_buttons[index].setChecked(True)
                return
        super().keyPressEvent(event)

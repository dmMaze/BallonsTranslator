import re
from typing import Callable, Dict, Iterable, Optional

from qtpy.QtCore import QEvent, QRect, QSize, Qt
from qtpy.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPalette,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextFormat,
)
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from qtpy.QtWidgets import QUndoCommand
except ImportError:
    from qtpy.QtGui import QUndoCommand

from ballontranslator.ui.custom_widget import ScrollBar
from ballontranslator.ui.framelesswindow import (
    DialogCloseButton,
    FramelessWindow,
    OutsideClickFramelessMixin,
)
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.shared import ON_MACOS, ON_WINDOWS


PAGE_HEADING_PATTERN = re.compile(r'^### ([^\r\n]+)\r?$', re.MULTILINE)
ORDERED_LIST_PATTERN = re.compile(r'^\s*\d+\.')


def page_summary_markdown(project: ProjImgTrans) -> str:
    r"""Render every project page and its editable summary in project order.

    >>> project = ProjImgTrans()
    >>> project.pages = {'001.png': []}
    >>> project._image_info = {'001.png': {}}
    >>> page_summary_markdown(project)
    '### 001.png\n\n'
    """
    sections = []
    for page_name in project.pages:
        record = project.get_llm_visual_summary(page_name)
        summary = str(record['text']) if record is not None else ''
        section = '### ' + page_name + '\n\n' + summary
        sections.append(section)
    return '\n\n\n'.join(sections)


def parse_page_summary_markdown(
    markdown: str,
    page_names: Iterable[str],
) -> Dict[str, str]:
    r"""Match exact Markdown page headings; omitted pages become empty.

    Unknown sections are ignored and a later duplicate replaces an earlier
    section, mirroring the project's page-heading import behavior.

    >>> parse_page_summary_markdown(
    ...     '### A.png\nold\n\n### A.png\nnew', ['A.png', 'B.png'])
    {'A.png': 'new', 'B.png': ''}
    """
    summaries = {str(page_name): '' for page_name in page_names}
    markdown = str(markdown).replace('\r\n', '\n')
    matches = list(PAGE_HEADING_PATTERN.finditer(markdown))
    for index, match in enumerate(matches):
        page_name = match.group(1).strip()
        if page_name not in summaries:
            continue
        body_start = match.end()
        has_next = index + 1 < len(matches)
        body_end = matches[index + 1].start() if has_next else len(markdown)
        body = markdown[body_start:body_end]
        if body.startswith('\n\n'):
            body = body[2:]
        elif body.startswith('\n'):
            body = body[1:]
        if has_next:
            for separator in ('\n\n\n', '\n\n', '\n'):
                if body.endswith(separator):
                    body = body[:-len(separator)]
                    break
        summaries[page_name] = body
    return summaries


SummaryRecord = Dict[str, object]
SummaryRecords = Dict[str, Optional[SummaryRecord]]


def _summary_records(
    project: ProjImgTrans,
    page_names: Iterable[str],
) -> SummaryRecords:
    return {
        page_name: project.get_llm_visual_summary(page_name)
        for page_name in page_names
        if page_name in project.pages
    }


def _summary_texts(records: SummaryRecords) -> Dict[str, str]:
    return {
        page_name: str(record['text']) if record is not None else ''
        for page_name, record in records.items()
    }


class PageSummaryEditCommand(QUndoCommand):
    """Apply one project-wide summary edit through the page-local undo stack.

    >>> issubclass(PageSummaryEditCommand, QUndoCommand)
    True
    """

    def __init__(
        self,
        project: ProjImgTrans,
        before: SummaryRecords,
        after: Dict[str, str],
        refresh: Callable[[], None],
        text: str,
    ) -> None:
        super().__init__()
        self.setText(text)
        self._project = project
        self._before = before
        self._after = after
        self._refresh = refresh

    def redo(self) -> None:
        for page_name, summary in self._after.items():
            if page_name in self._project.pages:
                self._project.set_llm_visual_summary_text(page_name, summary)
        self._refresh()

    def undo(self) -> None:
        for page_name, record in self._before.items():
            if page_name not in self._project.pages:
                continue
            if record is None:
                self._project.clear_llm_visual_summary(page_name)
            else:
                self._project.set_llm_visual_summary(page_name, record)
        self._refresh()


class _SummaryMarkdownHighlighter(QSyntaxHighlighter):
    def __init__(self, editor: QPlainTextEdit) -> None:
        super().__init__(editor.document())
        self._editor = editor
        self._heading_format = QTextCharFormat()
        self._list_format = QTextCharFormat()
        self.update_palette()

    def update_palette(self) -> None:
        palette = self._editor.palette()
        role = getattr(QPalette, 'ColorRole', QPalette)
        base = palette.color(role.Base)
        accent = palette.color(role.Highlight)
        marker = (
            accent.lighter(125)
            if base.lightness() < 128
            else accent.darker(115)
        )
        self._heading_format.setForeground(QColor('#d5686f'))
        self._heading_format.setFontWeight(QFont.Weight.DemiBold)
        self._list_format.setForeground(marker)
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        if PAGE_HEADING_PATTERN.fullmatch(text):
            self.setFormat(0, len(text), self._heading_format)
            return
        marker = ORDERED_LIST_PATTERN.match(text)
        if marker is not None:
            self.setFormat(
                marker.start(),
                marker.end() - marker.start(),
                self._list_format,
            )


class _LineNumberArea(QWidget):
    def __init__(self, editor: 'PageSummaryMarkdownEditor') -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event) -> None:
        self._editor.paint_line_number_area(event)


class PageSummaryMarkdownEditor(QPlainTextEdit):
    """Plain Markdown editor with a compact line-number gutter.

    >>> issubclass(PageSummaryMarkdownEditor, QPlainTextEdit)
    True
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName('BulkPageSummaryTextEdit')
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._line_number_area = _LineNumberArea(self)
        self._highlighter = _SummaryMarkdownHighlighter(self)
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_line_number_area_width()
        self._highlight_current_line()
        ScrollBar(Qt.Orientation.Vertical, self)

    def line_number_area_width(self) -> int:
        digits = len(str(max(1, self.blockCount())))
        return 10 + self.fontMetrics().horizontalAdvance('9') * digits

    def _update_line_number_area_width(self, _count: int = 0) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect: QRect, dy: int) -> None:
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(
                0,
                rect.y(),
                self._line_number_area.width(),
                rect.height(),
            )
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        contents = self.contentsRect()
        self._line_number_area.setGeometry(
            contents.left(),
            contents.top(),
            self.line_number_area_width(),
            contents.height(),
        )

    def _highlight_current_line(self) -> None:
        selection = QTextEdit.ExtraSelection()
        role = getattr(QPalette, 'ColorRole', QPalette)
        color = QColor(self.palette().color(role.Highlight))
        color.setAlpha(28)
        selection.format.setBackground(color)
        text_property = getattr(QTextFormat, 'Property', QTextFormat)
        selection.format.setProperty(
            text_property.FullWidthSelection,
            True,
        )
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        self.setExtraSelections([selection])
        self._line_number_area.update()

    def paint_line_number_area(self, event) -> None:
        painter = QPainter(self._line_number_area)
        palette = self.palette()
        role = getattr(QPalette, 'ColorRole', QPalette)
        gutter_color = palette.color(role.Base)
        painter.fillRect(event.rect(), gutter_color)

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(
            self.blockBoundingGeometry(block).translated(
                self.contentOffset()
            ).top()
        )
        bottom = top + round(self.blockBoundingRect(block).height())
        current_block = self.textCursor().blockNumber()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                if block_number == current_block:
                    active = QColor(palette.color(role.Highlight))
                    active.setAlpha(28)
                    painter.fillRect(
                        0,
                        top,
                        self._line_number_area.width(),
                        self.fontMetrics().height(),
                        active,
                    )
                    color = palette.color(role.Highlight)
                else:
                    color = palette.color(role.PlaceholderText)
                    color.setAlpha(150)
                painter.setPen(color)
                painter.drawText(
                    0,
                    top,
                    self._line_number_area.width() - 5,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    str(block_number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        event_type = event.type()
        if (
            event_type == QEvent.Type.FontChange
            and hasattr(self, '_line_number_area')
        ):
            self._update_line_number_area_width()
            self._line_number_area.update()
        if hasattr(self, '_highlighter') and event_type in {
            QEvent.Type.PaletteChange,
            QEvent.Type.StyleChange,
        }:
            self._highlighter.update_palette()
            self._highlight_current_line()


class BulkPageSummaryDialog(OutsideClickFramelessMixin, FramelessWindow):
    """Edit all page summaries as one transient Markdown document.

    >>> issubclass(BulkPageSummaryDialog, FramelessWindow)
    True
    """

    def __init__(
        self,
        project: ProjImgTrans,
        push_command: Callable[[QUndoCommand], None],
        refresh: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        window_type = getattr(Qt, 'WindowType', Qt)
        super().__init__(parent, window_type.Dialog)
        opaque_frame = ON_WINDOWS or ON_MACOS
        self.setObjectName('BulkPageSummaryDialog')
        self.setProperty('opaqueFrame', opaque_frame)
        self.setProperty('nativeFrame', ON_WINDOWS)
        self.setWindowTitle(self.tr('Page Summaries'))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumSize(620, 440)
        self.resize(760, 560)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        if not opaque_frame:
            self.setAttribute(widget_attribute.WA_TranslucentBackground)
        self.setAttribute(widget_attribute.WA_StyledBackground)
        self.setAttribute(widget_attribute.WA_DeleteOnClose)
        if ON_MACOS:
            self.windowEffect.removeShadowEffect(self.winId())

        self._project = project
        self._push_command = push_command
        self._refresh = refresh
        self._page_names = tuple(project.pages)
        self._opening_texts = _summary_texts(
            _summary_records(project, self._page_names)
        )
        self._committed = False

        root_layout = QVBoxLayout(self)
        margin = 0 if opaque_frame else 5
        root_layout.setContentsMargins(margin, margin, margin, margin)
        surface = QFrame(self)
        surface.setObjectName('BulkPageSummarySurface')
        root_layout.addWidget(surface)

        layout = QVBoxLayout(surface)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(14)

        self.title_bar = QWidget(surface)
        self.title_bar.setObjectName('BulkPageSummaryTitleBar')
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(self.tr('Page Summaries'), self.title_bar)
        title.setObjectName('BulkPageSummaryTitle')
        title_layout.addWidget(title)
        title_layout.addStretch()
        self.close_button = DialogCloseButton(self.title_bar)
        self.close_button.clicked.connect(self.close)
        title_layout.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        self.editor = PageSummaryMarkdownEditor(surface)
        self.editor.setPlainText(page_summary_markdown(project))
        self._baseline = parse_page_summary_markdown(
            self.editor.toPlainText(),
            self._page_names,
        )
        layout.addWidget(self.editor)

    def _dismiss_transient_window(self) -> None:
        self.close()

    def discard_and_close(self) -> None:
        self._committed = True
        self.close()

    def _commit(self) -> None:
        if self._committed:
            return
        self._committed = True
        parsed = parse_page_summary_markdown(
            self.editor.toPlainText(),
            self._page_names,
        )
        if parsed == self._baseline:
            return
        after = {
            page_name: (
                self._opening_texts[page_name]
                if summary == self._baseline[page_name]
                else summary
            )
            for page_name, summary in parsed.items()
            if page_name in self._project.pages
        }
        before = _summary_records(self._project, after)
        if not before or after == _summary_texts(before):
            return
        self._push_command(
            PageSummaryEditCommand(
                self._project,
                before,
                after,
                self._refresh,
                self.tr('Edit page summaries'),
            )
        )

    def closeEvent(self, event) -> None:
        self._commit()
        super().closeEvent(event)

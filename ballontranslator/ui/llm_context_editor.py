from typing import Optional

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils import shared
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class _ContextEditorCard(QFrame):
    def __init__(
        self,
        title: str,
        placeholder: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('LLMContextEditorCard')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName('LLMContextEditorTitle')
        title_font = self.title_label.font()
        if shared.ON_MACOS:
            title_font.setPointSize(13)
        else:
            title_font.setPointSizeF(10)
        self.title_label.setFont(title_font)
        self.detail_label = QLabel(self)
        self.detail_label.setObjectName('LLMContextEditorDetail')
        self.detail_label.setFont(title_font)

        header = QFrame(self)
        header.setObjectName('LLMContextEditorHeader')
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header.setFixedHeight(26)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 10, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.detail_label)

        self.editor = QPlainTextEdit(self)
        self.editor.setObjectName('LLMContextTextEdit')
        self.editor.setPlaceholderText(placeholder)
        self.editor.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
        )
        self.editor.setTabChangesFocus(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.editor, 1)


class LLMContextEditor(QWidget):
    """Edit the current page summary and project memory in one split pane.

    Writes update project-owned state only; the window decides when to save the
    project file.

    >>> LLMContextEditor.__name__
    'LLMContextEditor'
    """

    project_changed = Signal()

    def __init__(
        self,
        project: Optional[ProjImgTrans] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('LLMContextEditor')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(300)
        self._project = project
        self._page_key: Optional[str] = getattr(
            project,
            'current_img',
            None,
        )

        self.summary_card = _ContextEditorCard(
            self.tr('Page Summary'),
            self.tr('Write or revise context for the current page.'),
            self,
        )
        self.memory_card = _ContextEditorCard(
            self.tr('Compact Memory'),
            self.tr(
                'Review or revise the project memory applied to every page.'
            ),
            self,
        )
        self.summary_editor = self.summary_card.editor
        self.memory_editor = self.memory_card.editor

        self.editor_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.editor_splitter.setObjectName('LLMContextVerticalSplitter')
        self.editor_splitter.setChildrenCollapsible(False)
        self.editor_splitter.addWidget(self.summary_card)
        self.editor_splitter.addWidget(self.memory_card)
        self.editor_splitter.setStretchFactor(0, 1)
        self.editor_splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor_splitter)

        self.summary_editor.textChanged.connect(self._on_summary_changed)
        self.memory_editor.textChanged.connect(self._on_memory_changed)
        self.refresh()

    @property
    def page_key(self) -> Optional[str]:
        return self._page_key

    def set_project(self, project: Optional[ProjImgTrans]) -> None:
        self._project = project
        self._page_key = getattr(project, 'current_img', None)
        self.refresh()

    def set_page(self, page_key: Optional[str]) -> None:
        self._page_key = str(page_key) if page_key is not None else None
        self.refresh()

    @staticmethod
    def _replace_text(editor: QPlainTextEdit, text: str) -> None:
        if editor.toPlainText() == text:
            return
        blocker = QSignalBlocker(editor)
        editor.setPlainText(text)
        del blocker

    def refresh(self) -> None:
        project = self._project
        has_project = bool(
            project is not None and getattr(project, 'directory', None)
        )
        page_key = self._page_key
        has_page = bool(
            has_project
            and page_key is not None
            and page_key in project._image_info
        )

        summary_text = ''
        if has_page:
            summary_record = project.get_llm_visual_summary(page_key)
            if summary_record is not None:
                summary_text = str(summary_record['text'])
        self._replace_text(self.summary_editor, summary_text)
        self.summary_editor.setEnabled(has_page)
        self.summary_card.detail_label.setText(
            page_key if has_page else self.tr('No page selected')
        )

        memory_text = ''
        covered_count = 0
        if has_project:
            memory_record = project.get_llm_compact_memory()
            if memory_record is not None:
                memory_text = str(memory_record['text'])
                covered_count = len(memory_record['covered_pages'])
        self._replace_text(self.memory_editor, memory_text)
        self.memory_editor.setEnabled(has_project)
        if has_project:
            memory_detail = self.tr(
                'All pages · {count} covered'
            ).format(count=covered_count)
        else:
            memory_detail = self.tr('No project open')
        self.memory_card.detail_label.setText(memory_detail)

    def _on_summary_changed(self) -> None:
        project = self._project
        page_key = self._page_key
        if (
            project is None
            or page_key is None
            or page_key not in project._image_info
        ):
            return
        project.set_llm_visual_summary_text(
            page_key,
            self.summary_editor.toPlainText(),
        )
        self.project_changed.emit()

    def _on_memory_changed(self) -> None:
        project = self._project
        if project is None or not getattr(project, 'directory', None):
            return
        project.set_llm_compact_memory_text(
            self.memory_editor.toPlainText()
        )
        record = project.get_llm_compact_memory()
        covered_count = len(record['covered_pages']) if record else 0
        self.memory_card.detail_label.setText(
            self.tr('All pages · {count} covered').format(
                count=covered_count
            )
        )
        self.project_changed.emit()

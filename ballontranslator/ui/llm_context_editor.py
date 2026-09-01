from typing import Optional

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWIDGETSIZE_MAX,
    QWidget,
)

from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.ui.custom_widget import ScrollBar, ViewWidget


class _ContextView(ViewWidget):
    """Let a splitter reclaim the content area of a collapsed view."""

    def __init__(
        self,
        content_widget: QWidget,
        panel_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(content_widget, panel_name, parent)
        # Keep the card border visible around children that fill the view.
        self.layout().setContentsMargins(1, 1, 1, 1)
        self.detail_label = QLabel(self.title_label)
        self.detail_label.setObjectName('LLMContextEditorDetail')
        self.detail_label.setFont(self.title_label.textlabel.font())
        title_layout = self.title_label.layout()
        title_layout.insertWidget(
            title_layout.count() - 1,
            self.detail_label,
        )

    def set_expend_area(
        self,
        expend: Optional[bool] = None,
        set_config: bool = True,
    ) -> None:
        if expend is None:
            expend = self.title_label.expanded
        super().set_expend_area(expend, set_config)
        self.setMaximumHeight(
            QWIDGETSIZE_MAX if expend else self.sizeHint().height()
        )
        self.expend_changed.emit()


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
        self._page_key = project.current_img if project is not None else None

        self.summary_editor = self._create_editor(
            self.tr('Write or revise context for the current page.'),
        )
        self.memory_editor = self._create_editor(
            self.tr(
                'Review or revise the project memory applied to every page.'
            ),
        )

        self.summary_view = _ContextView(
            self.summary_editor,
            self.tr('Summary'),
            self,
        )
        self.summary_view.setObjectName('LLMContextView')
        self.memory_view = _ContextView(
            self.memory_editor,
            self.tr('Compact Memory'),
            self,
        )
        self.memory_view.setObjectName('LLMContextView')

        self.editor_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.editor_splitter.setObjectName('LLMContextVerticalSplitter')
        self.editor_splitter.setChildrenCollapsible(False)
        self.editor_splitter.addWidget(self.summary_view)
        self.editor_splitter.addWidget(self.memory_view)
        self.editor_splitter.setStretchFactor(0, 1)
        self.editor_splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.editor_splitter)

        self.summary_view.expend_changed.connect(
            self._sync_expanded_height
        )
        self.memory_view.expend_changed.connect(
            self._sync_expanded_height
        )
        self.summary_editor.textChanged.connect(self._on_summary_changed)
        self.memory_editor.textChanged.connect(self._on_memory_changed)
        self.refresh()

    def register_views(self) -> None:
        self.summary_view.register_view_widget(
            config_name='show_llm_page_summary',
            config_expand_name='expand_llm_page_summary',
            action_name=self.tr('Summary'),
        )
        self.memory_view.register_view_widget(
            config_name='show_llm_compact_memory',
            config_expand_name='expand_llm_compact_memory',
            action_name=self.tr('Compact Memory'),
        )

    def _create_editor(self, placeholder: str) -> QPlainTextEdit:
        editor = QPlainTextEdit(self)
        editor.setObjectName('LLMContextTextEdit')
        editor.setPlaceholderText(placeholder)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        editor.setTabChangesFocus(True)
        ScrollBar(Qt.Orientation.Vertical, editor)
        return editor

    def sync_view_visibility(self) -> None:
        """Hide the editor area when both View-menu entries are hidden."""
        has_visible_view = any(
            not view.isHidden()
            for view in (self.summary_view, self.memory_view)
        )
        self.setVisible(has_visible_view)
        self._sync_expanded_height()

    def _sync_expanded_height(self) -> None:
        has_expanded_view = any(
            not view.isHidden() and view.title_label.expanded
            for view in (self.summary_view, self.memory_view)
        )
        minimum_height = self.editor_splitter.minimumSizeHint().height()
        if has_expanded_view:
            self.setMaximumHeight(QWIDGETSIZE_MAX)
            self.setMinimumHeight(minimum_height)
        else:
            self.setMinimumHeight(minimum_height)
            self.setMaximumHeight(minimum_height)

    def set_project(self, project: Optional[ProjImgTrans]) -> None:
        self._project = project
        self._page_key = project.current_img if project is not None else None
        self.refresh()

    def set_page(self, page_key: Optional[str]) -> None:
        normalized_page_key = (
            str(page_key) if page_key is not None else None
        )
        if normalized_page_key == self._page_key:
            return
        self._page_key = normalized_page_key
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
        has_project = bool(project is not None and project.directory)
        page_key = self._page_key
        has_page = bool(
            has_project
            and page_key is not None
            and page_key in project.pages
        )

        summary_text = ''
        if has_page:
            summary_record = project.get_llm_visual_summary(page_key)
            if summary_record is not None:
                summary_text = str(summary_record['text'])
        self._replace_text(self.summary_editor, summary_text)
        self.summary_editor.setEnabled(has_page)
        self.summary_view.detail_label.setText(
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
        self.memory_view.detail_label.setText(memory_detail)

    def _on_summary_changed(self) -> None:
        project = self._project
        page_key = self._page_key
        if (
            project is None
            or page_key is None
            or page_key not in project.pages
        ):
            return
        project.set_llm_visual_summary_text(
            page_key,
            self.summary_editor.toPlainText(),
        )
        self.project_changed.emit()

    def _on_memory_changed(self) -> None:
        project = self._project
        if project is None or not project.directory:
            return
        project.set_llm_compact_memory_text(
            self.memory_editor.toPlainText()
        )
        record = project.get_llm_compact_memory()
        covered_count = len(record['covered_pages']) if record else 0
        self.memory_view.detail_label.setText(
            self.tr('All pages · {count} covered').format(
                count=covered_count
            )
        )
        self.project_changed.emit()

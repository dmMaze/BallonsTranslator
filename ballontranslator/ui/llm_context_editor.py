from typing import Optional

from qtpy.QtCore import QSignalBlocker, Qt, Signal
from qtpy.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QWIDGETSIZE_MAX,
    QWidget,
)

from ballontranslator.utils.config import pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.ui.custom_widget import ScrollBar, ViewWidget


class _ContextView(ViewWidget):
    """Let a splitter reclaim the content area of a collapsed view."""

    def __init__(
        self,
        content_widget: QWidget,
        panel_name: str,
        parent: Optional[QWidget] = None,
        show_detail: bool = False,
    ) -> None:
        super().__init__(content_widget, panel_name, parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.detail_label: Optional[QLabel] = None
        if show_detail:
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
        if expend:
            self.setMinimumHeight(0)
            self.setMaximumHeight(QWIDGETSIZE_MAX)
        else:
            self.setFixedHeight(self.title_label.height())
        self.expend_changed.emit()


class LLMContextEditor(QSplitter):
    """Place page context and its two editors in one split pane.

    Writes update project-owned state only; the window decides when to save the
    project file.

    >>> LLMContextEditor.__name__
    'LLMContextEditor'
    """

    project_changed = Signal()

    def __init__(
        self,
        page_widget: QWidget,
        project: Optional[ProjImgTrans] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(Qt.Orientation.Vertical, parent)
        self.setObjectName('LLMContextPanelSplitter')
        self.setChildrenCollapsible(False)
        self.setHandleWidth(1)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumWidth(300)
        self._page_widget = page_widget
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
            self.tr('Page Summary'),
            self,
        )
        self.summary_view.setObjectName('LLMContextView')
        self.memory_view = _ContextView(
            self.memory_editor,
            self.tr('Memory'),
            self,
            show_detail=True,
        )
        self.memory_view.setObjectName('LLMContextView')
        self.memory_detail_label = self.memory_view.detail_label
        assert self.memory_detail_label is not None

        page_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.addWidget(page_widget)
        self.addWidget(self.summary_view)
        self.addWidget(self.memory_view)
        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)
        self.setStretchFactor(2, 0)

        self.summary_view.expend_changed.connect(
            self._redistribute_view_space
        )
        self.memory_view.expend_changed.connect(
            self._redistribute_view_space
        )
        self.summary_editor.textChanged.connect(self._on_summary_changed)
        self.memory_editor.textChanged.connect(self._on_memory_changed)
        self.refresh_context()

    def register_views(self) -> None:
        self.summary_view.register_view_widget(
            config_name='show_llm_page_summary',
            config_expand_name='expand_llm_page_summary',
            action_name=self.tr('Page Summary'),
        )
        self.memory_view.register_view_widget(
            config_name='show_llm_compact_memory',
            config_expand_name='expand_llm_compact_memory',
            action_name=self.tr('Memory'),
        )

    def _create_editor(self, placeholder: str) -> QPlainTextEdit:
        editor = QPlainTextEdit(self)
        editor.setObjectName('LLMContextTextEdit')
        editor.setPlaceholderText(placeholder)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        editor.setTabChangesFocus(True)
        editor.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        ScrollBar(Qt.Orientation.Vertical, editor)
        return editor

    def set_context_visible(self, visible: bool) -> None:
        self.summary_view.setVisible(
            visible and pcfg.show_llm_page_summary
        )
        self.memory_view.setVisible(
            visible and pcfg.show_llm_compact_memory
        )

    def _redistribute_view_space(self) -> None:
        """Give folded editor space to the page list and reclaim it on reopen."""
        view = self.sender()
        if not isinstance(view, _ContextView):
            return
        sizes = self.sizes()
        view_index = self.indexOf(view)
        if view_index < 1 or len(sizes) != self.count():
            return

        current_height = sizes[view_index]
        target_height = (
            view.sizeHint().height()
            if view.title_label.expanded
            else view.title_label.height()
        )
        delta = target_height - current_height
        if delta > 0:
            page_minimum = self._page_widget.minimumSizeHint().height()
            delta = min(
                delta,
                max(0, sizes[0] - page_minimum),
            )
        sizes[0] -= delta
        sizes[view_index] += delta
        self.setSizes(sizes)

    def set_project(self, project: Optional[ProjImgTrans]) -> None:
        self._project = project
        self._page_key = project.current_img if project is not None else None
        self.refresh_context()

    def set_page(self, page_key: Optional[str]) -> None:
        normalized_page_key = (
            str(page_key) if page_key is not None else None
        )
        if normalized_page_key == self._page_key:
            return
        self._page_key = normalized_page_key
        self.refresh_context()

    @staticmethod
    def _replace_text(editor: QPlainTextEdit, text: str) -> None:
        if editor.toPlainText() == text:
            return
        blocker = QSignalBlocker(editor)
        editor.setPlainText(text)
        del blocker

    def refresh_context(self) -> None:
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
                '{count} covered'
            ).format(count=covered_count)
        else:
            memory_detail = self.tr('No project open')
        self.memory_detail_label.setText(memory_detail)

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
        self.memory_detail_label.setText(
            self.tr('{count} covered').format(
                count=covered_count
            )
        )
        self.project_changed.emit()

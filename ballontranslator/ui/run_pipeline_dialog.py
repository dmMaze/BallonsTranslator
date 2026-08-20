import os

from qtpy.QtCore import (
    QEvent,
    QObject,
    QSize,
    QSignalBlocker,
    Qt,
    Signal,
)
from qtpy.QtGui import QIcon, QMouseEvent
from qtpy.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icon_rendering import render_svg_pixmap
from .misc import themed_icon_path
from .framelesswindow import DialogCloseButton, FramelessMoveResize
from .llm_modality import (
    LLM_MODALITY_IMAGE,
    LLM_MODALITY_IMAGE_COLOR,
    LLM_MODALITY_TEXT,
    LLM_MODALITY_TEXT_COLOR,
    LLM_MODALITY_VISION,
    LLM_MODALITY_VISION_COLOR,
    modality_badge_qcolor,
)
from .page_range_progress import PageRangeProgressWidget, PageRangeSpinBox
from .custom_widget import ExpandingToolButton
from .custom_widget.combobox import BottomBorderComboBox
from ballontranslator.utils.config import (
    LLMGlossaryMode,
    LLMTranslateContext,
    OCRTextPostprocess,
    pcfg,
    save_config,
    TranslateContext,
)
from ballontranslator.utils.llm_profiles import LLM_TRANSLATOR_KEY
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.modules import (
    GET_VALID_INPAINTERS,
    GET_VALID_OCR,
    GET_VALID_TEXTDETECTORS,
    GET_VALID_TRANSLATORS,
)


RUN_PIPELINE_DIALOG_WIDTH = 510
RUN_PIPELINE_SETTING_CONTROL_WIDTH = 100
RUN_PIPELINE_GLOSSARY_DISPLAY_WIDTH = 100


class PipelineModuleButton(QAbstractButton):
    """Locally toggle a pipeline module with profile-style modality visuals.

    >>> PipelineModuleButton.__name__
    'PipelineModuleButton'
    """

    _MODALITY_VISUALS = {
        LLM_MODALITY_TEXT: ('text.svg', 'text_disabled.svg', LLM_MODALITY_TEXT_COLOR),
        LLM_MODALITY_VISION: ('eye.svg', 'eye_disabled.svg', LLM_MODALITY_VISION_COLOR),
        LLM_MODALITY_IMAGE: ('image.svg', 'image_disabled.svg', LLM_MODALITY_IMAGE_COLOR),
    }

    def __init__(
        self,
        text: str,
        modality: str,
        parent: QWidget = None,
        active_icon_name: str = '',
        inactive_icon_name: str = '',
    ):
        super().__init__(parent)
        self.modality = modality
        default_active_icon, default_inactive_icon, _ = self._MODALITY_VISUALS[modality]
        self.active_icon_name = active_icon_name or default_active_icon
        self.inactive_icon_name = inactive_icon_name or default_inactive_icon
        self.setObjectName('RunPipelineModuleButton')
        self.setCheckable(True)
        self.setChecked(True)
        cursor_shape = getattr(Qt, 'CursorShape', Qt)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        self.setCursor(cursor_shape.PointingHandCursor)
        self.setAccessibleName(text)
        self.setToolTip(text)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(0)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName('RunPipelineModuleIcon')
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setAttribute(widget_attribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.icon_label)

        self.toggled.connect(self._refresh_visuals)
        self._refresh_visuals(self.isChecked())

    def _refresh_visuals(self, active: bool):
        _, _, modality_color = self._MODALITY_VISUALS[self.modality]
        icon_path = themed_icon_path(
            self.active_icon_name if active else self.inactive_icon_name
        )
        badge_color = modality_badge_qcolor(modality_color)
        background = badge_color.getRgb() if active else (0, 0, 0, 0)
        self.icon_label.setPixmap(
            render_svg_pixmap(
                icon_path,
                self.icon_label.width(),
                self.icon_label.height(),
                self.icon_label.devicePixelRatioF(),
                inset=2,
                background_rgba=background,
                background_radius=6,
            )
        )

    def paintEvent(self, event):
        option = QStyleOptionButton()
        option.initFrom(self)
        state_flag = getattr(QStyle, 'StateFlag', QStyle)
        if self.isDown():
            option.state |= state_flag.State_Sunken
        if self.isChecked():
            option.state |= state_flag.State_On
        control_element = getattr(QStyle, 'ControlElement', QStyle)
        painter = QStylePainter(self)
        painter.drawControl(control_element.CE_PushButton, option)

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            self._refresh_visuals(self.isChecked())
        return super().changeEvent(event)


class PipelineModuleActivator(QWidget):
    """Toggleable pipeline stage with its own always-visible module selector.

    >>> PipelineModuleActivator.__name__
    'PipelineModuleActivator'
    """

    module_selected = Signal(str, str)
    config_requested = Signal(str, str)

    def __init__(
        self,
        module_type: str,
        module_name: str,
        options,
        text: str,
        modality: str,
        parent: QWidget = None,
        active_icon_name: str = '',
        inactive_icon_name: str = '',
    ) -> None:
        super().__init__(parent)
        self.module_type = module_type
        self._hovered = False
        self.setObjectName('RunPipelineModuleActivator')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.button = PipelineModuleButton(
            text,
            modality,
            self,
            active_icon_name=active_icon_name,
            inactive_icon_name=inactive_icon_name,
        )
        layout.addWidget(self.button)

        self.config_button = QToolButton(self)
        self.config_button.setObjectName('RunPipelineModuleConfigButton')
        self.config_button.setIcon(
            QIcon(themed_icon_path('leftbar_config_activate.svg'))
        )
        self.config_button.setToolTip(self.tr('Config'))
        self.config_button.setAccessibleName(self.tr('Config'))
        self.config_button.clicked.connect(self._request_config)
        self.deactivate_button = QToolButton(self)
        self.deactivate_button.setObjectName('RunPipelineModuleDeactivateButton')
        self.deactivate_button.setIcon(
            QIcon(themed_icon_path('titlebar_close.svg'))
        )
        self.deactivate_button.setIconSize(QSize(12, 12))
        self.deactivate_button.setToolTip(self.tr('Deactivate module'))
        self.deactivate_button.setAccessibleName(self.tr('Deactivate module'))
        self.deactivate_button.clicked.connect(self._deactivate)
        self.selector = BottomBorderComboBox(self)
        self.selector.setObjectName('RunPipelineModuleSelector')
        self.selector.setFixedWidth(136)
        self.selector.addItems(options)
        self.selector.setCurrentText(module_name)
        self.selector.setToolTip(module_name)
        self.selector.currentTextChanged.connect(self._on_module_selected)
        self.selector.installEventFilter(self)
        layout.addWidget(self.selector)
        layout.addStretch(1)
        layout.addWidget(self.config_button)
        layout.addWidget(self.deactivate_button)

        self.button.toggled.connect(self._refresh_active_state)
        self._refresh_active_state(self.button.isChecked())

    def enterEvent(self, event) -> None:
        self._hovered = True
        self._refresh_aux_buttons()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self._refresh_aux_buttons()
        super().leaveEvent(event)

    def _refresh_aux_buttons(self, _checked: bool = False) -> None:
        active = self.button.isChecked()
        visible = self._hovered and active
        self.config_button.setVisible(visible)
        self.deactivate_button.setVisible(visible)

    def _refresh_active_state(self, active: bool) -> None:
        self._refresh_aux_buttons()
        for widget in (self, self.selector):
            widget.setProperty('moduleActive', active)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is not self.selector:
            return super().eventFilter(watched, event)
        if (
            not self.button.isChecked()
            and isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.button.setChecked(True)
            return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            not self.button.isChecked()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.button.setChecked(True)
            event.accept()
            return
        super().mousePressEvent(event)

    def _deactivate(self, _checked: bool = False) -> None:
        self.button.setChecked(False)

    def _request_config(self, _checked: bool = False) -> None:
        self.config_requested.emit(
            self.module_type,
            self.selector.currentText(),
        )

    def _on_module_selected(self, module_name: str) -> None:
        self.selector.setToolTip(module_name)
        self.module_selected.emit(self.module_type, module_name)

    def setModule(self, module_name: str) -> None:
        if self.selector.currentText() == module_name:
            return
        blocker = QSignalBlocker(self.selector)
        self.selector.setCurrentText(module_name)
        del blocker
        self.selector.setToolTip(module_name)


class GlossaryPathEdit(QLineEdit):
    """Read-only glossary filename field with an embedded picker button.

    >>> GlossaryPathEdit.__name__
    'GlossaryPathEdit'
    """


    def __init__(self, parent: QWidget = None, button_size: int = 20):
        super().__init__(parent)
        self._button_size = button_size
        self.select_button = QPushButton(self)
        self.select_button.setParent(self)
        self.select_button.setObjectName('RunPipelineGlossaryFileButton')
        self.select_button.setFixedSize(button_size, button_size)
        self.select_button.setIcon(QIcon(themed_icon_path('files.svg')))
        self.select_button.setIconSize(QSize(16, 16))
        self.setTextMargins(0, 0, button_size + 2, 0)
        self._position_select_button()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_select_button()

    def _position_select_button(self):
        self.select_button.move(
            self.width() - self._button_size,
            max(0, (self.height() - self.select_button.height()) // 2),
        )


class RunPipelineDialog(QDialog):
    """Choose and configure the pipeline action to run.

    >>> len({RunPipelineDialog.RUN, RunPipelineDialog.CONTINUE, RunPipelineDialog.RENDER})
    3
    """

    RUN = 1
    CONTINUE = 2
    RENDER = 3
    translate_source_changed = Signal(str)
    translate_target_changed = Signal(str)
    module_selected = Signal(str, str)
    module_config_requested = Signal(str, str)
    RESIZE_BORDER_WIDTH = 5
    _module_settings_expanded = (False, False, False, False)
    _page_range = (1, None)

    def __init__(
        self,
        parent: QWidget = None,
        project: ProjImgTrans = None,
        translator_metadata: dict = None,
    ):
        super().__init__(parent)
        self.project = project
        self.translator_metadata = translator_metadata or {}
        self._app_event_filter_installed = False
        self._checkbox_settings: dict[QCheckBox, tuple[object, str]] = {}
        self.setObjectName('RunPipelineDialog')
        self.setWindowTitle(self.tr('Run'))
        self.setMinimumWidth(RUN_PIPELINE_DIALOG_WIDTH)
        self.setModal(True)
        self.setMouseTracking(True)

        window_type = getattr(Qt, 'WindowType', Qt)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        self.setWindowFlags(window_type.Dialog | window_type.FramelessWindowHint)
        self.setAttribute(widget_attribute.WA_TranslucentBackground)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        surface = QFrame(self)
        surface.setObjectName('RunPipelineSurface')
        surface.setMouseTracking(True)
        outer_layout.addWidget(surface)

        layout = QVBoxLayout(surface)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(14)

        self.title_bar = QWidget(surface)
        self.title_bar.setObjectName('RunPipelineTitleBar')
        self.title_bar.setMouseTracking(True)
        title_row = QHBoxLayout(self.title_bar)
        title_row.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(self.tr('Run'), self.title_bar)
        self.title_label.setObjectName('RunPipelineTitle')
        self.title_label.setMouseTracking(True)
        title_row.addWidget(self.title_label)
        title_row.addSpacing(12)
        self.workflow_selector = BottomBorderComboBox(surface)
        self.workflow_selector.setObjectName('RunPipelineWorkflowSelector')
        self.workflow_selector.addItems((self.tr('Pipeline'), self.tr('Rendering')))
        pipeline_mode = str(pcfg.run_pipeline_mode).lower()
        mode_indexes = {'pipeline': 0, 'rendering': 1}
        self.workflow_selector.setCurrentIndex(mode_indexes.get(pipeline_mode, 0))
        self.workflow_selector.setFixedWidth(126)
        title_row.addWidget(self.workflow_selector)
        title_row.addStretch()
        self.close_button = DialogCloseButton(surface)
        self.close_button.clicked.connect(self.reject)
        title_row.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        self.content_stack = QStackedWidget(surface)
        self.content_stack.setObjectName('RunPipelineContentStack')
        self.content_stack.addWidget(self._build_pipeline_page())
        self.content_stack.addWidget(self._build_rendering_page())

        self.content_dock = QDockWidget(surface)
        self.content_dock.setObjectName('RunPipelineContentDock')
        dock_features = getattr(QDockWidget, 'DockWidgetFeature', QDockWidget)
        self.content_dock.setFeatures(dock_features.NoDockWidgetFeatures)
        dock_title = QWidget(self.content_dock)
        dock_title.setFixedHeight(0)
        self.content_dock.setTitleBarWidget(dock_title)
        self.content_dock.setWidget(self.content_stack)
        layout.addWidget(self.content_dock)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 10, 0, 0)
        button_row.setSpacing(8)
        button_row.addStretch()

        self.run_button = QPushButton(self.tr('Run'), surface)
        self.run_button.setObjectName('RunPipelinePrimaryButton')
        self.run_button.clicked.connect(self._finish_run)

        self.continue_button = QPushButton(self.tr('Continue'), surface)
        self.continue_button.setObjectName('RunPipelineSecondaryButton')
        self.continue_button.setDefault(True)
        self.continue_button.clicked.connect(self._finish_continue)
        button_row.addWidget(self.continue_button)
        button_row.addWidget(self.run_button)

        self.render_button = QPushButton(self.tr('Render'), surface)
        self.render_button.setObjectName('RunPipelinePrimaryButton')
        self.render_button.clicked.connect(self._finish_render)
        self.render_button.hide()
        button_row.addWidget(self.render_button)
        layout.addLayout(button_row)

        self.workflow_selector.currentIndexChanged.connect(self._set_pipeline_page)
        self.finished.connect(self._save_config_on_finish)
        self._set_pipeline_page(self.workflow_selector.currentIndex(), persist=False)
        initial_height = self.sizeHint().height()
        self.setMinimumHeight(initial_height)
        self.resize(RUN_PIPELINE_DIALOG_WIDTH, initial_height)

    def _finish_run(self, _checked: bool = False) -> None:
        self.done(self.RUN)

    def _finish_continue(self, _checked: bool = False) -> None:
        self.done(self.CONTINUE)

    def _finish_render(self, _checked: bool = False) -> None:
        self.done(self.RENDER)

    def _save_config_on_finish(self, _result: int) -> None:
        save_config()

    def showEvent(self, event):
        super().showEvent(event)
        app = QApplication.instance()
        if app is not None and not self._app_event_filter_installed:
            app.installEventFilter(self)
            self._app_event_filter_installed = True

    def hideEvent(self, event):
        app = QApplication.instance()
        if app is not None and self._app_event_filter_installed:
            app.removeEventFilter(self)
            self._app_event_filter_installed = False
        super().hideEvent(event)

    def eventFilter(self, watched, event):
        if (
            watched is getattr(self, 'glossary_path_edit', None)
            and isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and not pcfg.module.llm_glossary_path.strip()
        ):
            self._select_glossary_file()
            return True
        if (
            not self.isVisible()
            or not isinstance(watched, QWidget)
            or watched.window() is not self
            or not isinstance(event, QMouseEvent)
        ):
            return super().eventFilter(watched, event)

        event_type = event.type()
        if event_type not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove):
            return super().eventFilter(watched, event)

        global_pos = self._global_mouse_pos(event)
        edges = self._resize_edges(global_pos)
        if event_type == QEvent.Type.MouseMove:
            self._update_resize_cursor(edges)
        elif event.button() == Qt.MouseButton.LeftButton:
            if edges:
                FramelessMoveResize.starSystemResize(self, global_pos, edges)
                return True
            if self._can_drag_title(watched, global_pos):
                FramelessMoveResize.startSystemMove(self, global_pos)
                return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _global_mouse_pos(event: QMouseEvent):
        if hasattr(event, 'globalPosition'):
            return event.globalPosition().toPoint()
        return event.globalPos()

    def _resize_edges(self, global_pos):
        pos = self.mapFromGlobal(global_pos)
        edges = Qt.Edge(0)
        if pos.x() < self.RESIZE_BORDER_WIDTH:
            edges |= Qt.Edge.LeftEdge
        if pos.x() >= self.width() - self.RESIZE_BORDER_WIDTH:
            edges |= Qt.Edge.RightEdge
        if pos.y() < self.RESIZE_BORDER_WIDTH:
            edges |= Qt.Edge.TopEdge
        if pos.y() >= self.height() - self.RESIZE_BORDER_WIDTH:
            edges |= Qt.Edge.BottomEdge
        return edges

    def _update_resize_cursor(self, edges):
        if edges in (
            Qt.Edge.LeftEdge | Qt.Edge.TopEdge,
            Qt.Edge.RightEdge | Qt.Edge.BottomEdge,
        ):
            cursor = Qt.CursorShape.SizeFDiagCursor
        elif edges in (
            Qt.Edge.RightEdge | Qt.Edge.TopEdge,
            Qt.Edge.LeftEdge | Qt.Edge.BottomEdge,
        ):
            cursor = Qt.CursorShape.SizeBDiagCursor
        elif edges in (Qt.Edge.TopEdge, Qt.Edge.BottomEdge):
            cursor = Qt.CursorShape.SizeVerCursor
        elif edges in (Qt.Edge.LeftEdge, Qt.Edge.RightEdge):
            cursor = Qt.CursorShape.SizeHorCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)

    def _can_drag_title(self, watched: QWidget, global_pos) -> bool:
        local_pos = self.mapFromGlobal(global_pos)
        title_bottom = self.title_bar.mapTo(
            self,
            self.title_bar.rect().bottomLeft(),
        ).y()
        if not (
            0 <= local_pos.x() < self.width()
            and 0 <= local_pos.y() <= title_bottom
        ):
            return False
        for control in (self.workflow_selector, self.close_button):
            if watched is control or control.isAncestorOf(watched):
                return False
        return True

    def _build_pipeline_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName('RunPipelinePipelinePage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)

        self._add_section_divider(layout, self.tr('Activate Modules'))

        stages = QFrame(page)
        stages.setObjectName('RunPipelineStages')
        stage_layout = QGridLayout(stages)
        stage_layout.setContentsMargins(2, 4, 2, 2)
        stage_layout.setHorizontalSpacing(24)
        stage_layout.setVerticalSpacing(8)
        stage_specs = (
            (
                0,
                'textdetector',
                pcfg.module.textdetector,
                GET_VALID_TEXTDETECTORS(),
                self.tr('Detection'),
                LLM_MODALITY_VISION,
                'textdetect_activate.svg',
                'textdetect.svg',
            ),
            (
                1,
                'ocr',
                pcfg.module.ocr,
                GET_VALID_OCR(),
                self.tr('OCR'),
                LLM_MODALITY_VISION,
                '',
                '',
            ),
            (
                3,
                'inpainter',
                pcfg.module.inpainter,
                GET_VALID_INPAINTERS(),
                self.tr('Inpainting'),
                LLM_MODALITY_IMAGE,
                '',
                '',
            ),
            (
                2,
                'translator',
                pcfg.module.translator,
                GET_VALID_TRANSLATORS(),
                self.tr('Translation'),
                LLM_MODALITY_TEXT,
                '',
                '',
            ),
        )
        self.module_buttons = []
        self.module_activators = []
        for display_index, (
            stage_index,
            module_type,
            module_name,
            module_options,
            name,
            modality,
            active_icon,
            inactive_icon,
        ) in enumerate(stage_specs):
            activator = PipelineModuleActivator(
                module_type,
                module_name,
                module_options,
                name,
                modality,
                stages,
                active_icon_name=active_icon,
                inactive_icon_name=inactive_icon,
            )
            button = activator.button
            button.setChecked(pcfg.module.stage_enabled(stage_index))
            button.setProperty('stageIndex', stage_index)
            button.setProperty('sectionIndex', display_index)
            button.toggled.connect(self._on_stage_button_toggled)
            activator.module_selected.connect(self.module_selected.emit)
            activator.config_requested.connect(self.module_config_requested.emit)
            stage_layout.addWidget(
                activator,
                display_index // 2,
                display_index % 2,
            )
            self.module_buttons.append(button)
            self.module_activators.append(activator)
        layout.addWidget(stages)

        self._add_section_divider(layout, self.tr('Settings'))

        self.settings_body = QWidget(page)
        self.settings_body.setObjectName('RunPipelineSettingsBody')
        settings_layout = QVBoxLayout(self.settings_body)
        settings_layout.setContentsMargins(16, 0, 16, 0)
        settings_layout.setSpacing(10)
        self.settings_sections = {}
        self.module_settings_headers = {}
        self.module_settings_bodies = {}
        general_section, general_layout = self._add_settings_section(
            settings_layout,
            'general',
            show_header=False,
        )
        self._build_general_settings(general_section, general_layout)
        layout.addWidget(self.settings_body)

        section_specs = (
            (self.tr('Text Detection'), self._build_detector_settings),
            (self.tr('OCR'), self._build_ocr_settings),
            (self.tr('Inpainting'), self._build_inpainting_settings),
            (self.tr('Translation'), self._build_translation_settings),
        )
        for index, (title, builder) in enumerate(section_specs):
            section, section_body, section_layout = self._add_settings_section(
                layout,
                index,
                title,
            )
            builder(section_body, section_layout)
            section.setVisible(
                self.module_buttons[index].isChecked()
                and self._settings_section_has_content(section_body)
            )
        return page

    def _build_general_settings(self, section: QWidget, layout: QVBoxLayout):
        start, end = type(self)._page_range
        self.page_range_progress = PageRangeProgressWidget(
            self._project_page_names(),
            start=start,
            end=end,
            parent=section,
        )
        self.range_start = self.page_range_progress.range_start
        self.range_end = self.page_range_progress.range_end
        self.progress_bar = self.page_range_progress.range_bar
        self.page_range_progress.range_changed.connect(
            self._on_page_range_changed
        )
        layout.addWidget(self.page_range_progress)
        self._refresh_progress()

    def _add_checkbox_setting(
        self,
        parent: QWidget,
        layout: QVBoxLayout,
        object_name: str,
        text: str,
        checked: bool,
        target: object,
        attribute: str,
    ) -> QCheckBox:
        checkbox = QCheckBox(text, parent)
        checkbox.setObjectName(object_name)
        checkbox.setChecked(checked)
        self._checkbox_settings[checkbox] = (target, attribute)
        checkbox.toggled.connect(self._on_checkbox_setting_toggled)
        layout.addWidget(checkbox)
        return checkbox

    def _on_checkbox_setting_toggled(self, checked: bool) -> None:
        checkbox = self.sender()
        if not isinstance(checkbox, QCheckBox):
            return
        target_and_attribute = self._checkbox_settings.get(checkbox)
        if target_and_attribute is None:
            return
        target, attribute = target_and_attribute
        setattr(target, attribute, checked)

    def _build_detector_settings(self, section: QWidget, layout: QVBoxLayout):
        self.keep_existing_lines = self._add_checkbox_setting(
            section,
            layout,
            'RunPipelineKeepExistingLines',
            self.tr('Keep Existing Lines'),
            pcfg.module.keep_exist_textlines,
            pcfg.module,
            'keep_exist_textlines',
        )

    def _build_ocr_settings(self, section: QWidget, layout: QVBoxLayout):
        self.remove_empty_textblocks = self._add_checkbox_setting(
            section,
            layout,
            'RunPipelineRemoveEmptyTextblocks',
            self.tr('Remove empty textblocks'),
            pcfg.restore_ocr_empty,
            pcfg,
            'restore_ocr_empty',
        )
        self.font_detection = self._add_checkbox_setting(
            section,
            layout,
            'RunPipelineFontDetection',
            self.tr('Font Detection'),
            pcfg.module.ocr_font_detect,
            pcfg.module,
            'ocr_font_detect',
        )

        postprocess_options_row = QWidget(section)
        postprocess_options_row.setObjectName('RunPipelineGeneralSettingRow')
        postprocess_options_row.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        postprocess_layout = QHBoxLayout(postprocess_options_row)
        postprocess_layout.setContentsMargins(0, 0, 0, 0)
        postprocess_layout.setSpacing(12)

        postprocess_label = QLabel(self.tr('Letter Case'), postprocess_options_row)
        postprocess_label.setObjectName('RunPipelineSettingLabel')
        postprocess_label.setToolTip(self.tr(
            'Choose how OCR text letter case is adjusted after keyword substitution.'
        ))
        postprocess_layout.addWidget(postprocess_label)

        self.ocr_text_postprocess_group = QButtonGroup(postprocess_options_row)
        self.ocr_text_postprocess_buttons = {}
        postprocess_options = (
            (
                self.tr('None'),
                OCRTextPostprocess.NONE,
                self.tr('Keep OCR text letter case unchanged.'),
            ),
            (
                self.tr('Capitalize'),
                OCRTextPostprocess.CAPITALIZE,
                self.tr(
                    'Lowercase OCR text, then capitalize the first letter of each sentence.'
                ),
            ),
            (
                self.tr('Uppercase'),
                OCRTextPostprocess.UPPERCASE,
                self.tr('Convert OCR text to uppercase.'),
            ),
        )
        for text, mode, tooltip in postprocess_options:
            button = QRadioButton(text, postprocess_options_row)
            button.setObjectName('RunPipelineOCRTextPostprocessOption')
            button.setChecked(pcfg.module.ocr_text_postprocess == mode)
            button.setProperty('textPostprocessMode', mode)
            button.setToolTip(tooltip)
            button.toggled.connect(self._on_ocr_text_postprocess_toggled)
            self.ocr_text_postprocess_group.addButton(button)
            self.ocr_text_postprocess_buttons[mode] = button
            postprocess_layout.addWidget(button)
        postprocess_layout.addStretch()
        layout.addWidget(postprocess_options_row)

    def _on_ocr_text_postprocess_toggled(self, checked: bool) -> None:
        button = self.sender()
        if checked and isinstance(button, QRadioButton):
            pcfg.module.ocr_text_postprocess = button.property(
                'textPostprocessMode'
            )

    def _build_inpainting_settings(self, section: QWidget, layout: QVBoxLayout):
        self.skip_simple_cases = self._add_checkbox_setting(
            section,
            layout,
            'RunPipelineSkipSimpleCases',
            self.tr('Skip simple cases'),
            pcfg.module.check_need_inpaint,
            pcfg.module,
            'check_need_inpaint',
        )
        self.filter_mask_by_text_boxes = self._add_checkbox_setting(
            section,
            layout,
            'RunPipelineFilterMaskByTextBoxes',
            self.tr('Filter mask by text boxes'),
            pcfg.module.filter_mask_by_bboxes,
            pcfg.module,
            'filter_mask_by_bboxes',
        )

    def _translation_options(self, key: str, current: str):
        options = list(self.translator_metadata.get(key, ()))
        if current and current not in options:
            options.append(current)
        return options

    def _build_translation_settings(self, section: QWidget, layout: QVBoxLayout):
        self._llm_settings_visible = (
            self.translator_metadata.get('name') == LLM_TRANSLATOR_KEY
        )
        translation_grid = QGridLayout()
        translation_grid.setContentsMargins(0, 0, 0, 0)
        translation_grid.setHorizontalSpacing(16)
        translation_grid.setVerticalSpacing(8)
        translation_grid.setColumnStretch(0, 1)
        translation_grid.setColumnStretch(1, 1)
        layout.addLayout(translation_grid)

        source_row = QWidget(section)
        source_row.setObjectName('RunPipelineGeneralSettingRow')
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(8)
        source_label = QLabel(self.tr('Source'), source_row)
        source_label.setObjectName('RunPipelineSettingLabel')
        source_layout.addWidget(source_label)
        source_layout.addStretch()
        self.source_combobox = BottomBorderComboBox(source_row)
        self.source_combobox.setObjectName('RunPipelineSourceComboBox')
        self.source_combobox.addItems(self._translation_options(
            'supported_src_list',
            pcfg.module.translate_source,
        ))
        self.source_combobox.setCurrentText(pcfg.module.translate_source)
        self.source_combobox.setFixedWidth(RUN_PIPELINE_SETTING_CONTROL_WIDTH)
        source_layout.addWidget(self.source_combobox)

        target_row = QWidget(section)
        target_row.setObjectName('RunPipelineGeneralSettingRow')
        target_layout = QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(8)
        target_label = QLabel(self.tr('Target'), target_row)
        target_label.setObjectName('RunPipelineSettingLabel')
        target_layout.addWidget(target_label)
        target_layout.addStretch()
        self.target_combobox = BottomBorderComboBox(target_row)
        self.target_combobox.setObjectName('RunPipelineTargetComboBox')
        self.target_combobox.addItems(self._translation_options(
            'supported_tgt_list',
            pcfg.module.translate_target,
        ))
        self.target_combobox.setCurrentText(pcfg.module.translate_target)
        self.target_combobox.setFixedWidth(RUN_PIPELINE_SETTING_CONTROL_WIDTH)
        target_layout.addWidget(self.target_combobox)
        translation_grid.addWidget(source_row, 0, 0)
        translation_grid.addWidget(target_row, 0, 1)

        context_row = QWidget(section)
        self.context_row = context_row
        context_row.setObjectName('RunPipelineGeneralSettingRow')
        context_layout = QHBoxLayout(context_row)
        context_layout.setContentsMargins(0, 0, 0, 0)
        context_layout.setSpacing(8)
        context_label = QLabel(self.tr('Context'), context_row)
        context_label.setObjectName('RunPipelineSettingLabel')
        context_layout.addWidget(context_label)
        context_layout.addStretch()
        self.context_combobox = BottomBorderComboBox(context_row)
        self.context_combobox.setObjectName('RunPipelineContextComboBox')
        self.context_combobox.addItem(self.tr('textblock'), TranslateContext.TextBlock)
        self.context_combobox.addItem(self.tr('page'), TranslateContext.Page)
        context_index = self.context_combobox.findData(pcfg.module.translate_context)
        self.context_combobox.setCurrentIndex(max(context_index, 0))
        self.context_combobox.setFixedWidth(RUN_PIPELINE_SETTING_CONTROL_WIDTH)
        context_layout.addWidget(self.context_combobox)

        llm_context_row = QWidget(section)
        self.llm_context_row = llm_context_row
        llm_context_row.setObjectName('RunPipelineGeneralSettingRow')
        llm_context_layout = QHBoxLayout(llm_context_row)
        llm_context_layout.setContentsMargins(0, 0, 0, 0)
        llm_context_layout.setSpacing(8)
        llm_context_label = QLabel(self.tr('LLM Context'), llm_context_row)
        llm_context_label.setObjectName('RunPipelineSettingLabel')
        llm_context_layout.addWidget(llm_context_label)
        llm_context_layout.addStretch()
        self.llm_context_combobox = BottomBorderComboBox(llm_context_row)
        self.llm_context_combobox.setObjectName('RunPipelineLLMContextComboBox')
        self.llm_context_combobox.addItem(
            self.tr('page'),
            LLMTranslateContext.PAGE,
        )
        self.llm_context_combobox.addItem(
            self.tr('+history'),
            LLMTranslateContext.HISTORY,
        )
        llm_context_index = self.llm_context_combobox.findData(
            pcfg.module.llm_translate_context
        )
        self.llm_context_combobox.setCurrentIndex(max(llm_context_index, 0))
        self.llm_context_combobox.setFixedWidth(
            RUN_PIPELINE_SETTING_CONTROL_WIDTH
        )
        llm_context_layout.addWidget(self.llm_context_combobox)
        llm_context_row.setVisible(self._llm_settings_visible)

        history_budget_row = QWidget(section)
        self.history_budget_row = history_budget_row
        history_budget_row.setObjectName('RunPipelineGeneralSettingRow')
        history_budget_layout = QHBoxLayout(history_budget_row)
        history_budget_layout.setContentsMargins(0, 0, 0, 0)
        history_budget_layout.setSpacing(8)
        budget_label = QLabel(
            self.tr('Token budget'),
            history_budget_row,
        )
        budget_label.setObjectName('RunPipelineSettingLabel')
        history_budget_layout.addWidget(budget_label)
        history_budget_layout.addStretch()
        self.prior_context_token_budget = PageRangeSpinBox(history_budget_row)
        self.prior_context_token_budget.setObjectName(
            'RunPipelinePriorContextTokenBudget'
        )
        self.prior_context_token_budget.setRange(128, 2_147_483_647)
        self.prior_context_token_budget.setSingleStep(128)
        self.prior_context_token_budget.setFixedWidth(
            RUN_PIPELINE_SETTING_CONTROL_WIDTH
        )
        self.prior_context_token_budget.setValue(
            max(128, pcfg.module.llm_prior_context_token_budget)
        )
        history_limit_help = self.tr(
            'Maximum translation history sent to the model. The current page, '
            'instructions, glossary, and generated reply are not included.'
        )
        budget_label.setToolTip(history_limit_help)
        self.prior_context_token_budget.setToolTip(history_limit_help)
        history_budget_layout.addWidget(self.prior_context_token_budget)
        history_budget_row.setVisible(
            self._llm_settings_visible
            and pcfg.module.llm_translate_context == LLMTranslateContext.HISTORY
        )
        translation_grid.addWidget(context_row, 1, 0)
        translation_grid.addWidget(llm_context_row, 1, 0)
        translation_grid.addWidget(history_budget_row, 1, 1)

        glossary_row = QWidget(section)
        self.glossary_row = glossary_row
        glossary_row.setObjectName('RunPipelineGeneralSettingRow')
        glossary_layout = QHBoxLayout(glossary_row)
        glossary_layout.setContentsMargins(0, 0, 0, 0)
        glossary_layout.setSpacing(8)
        glossary_label = QLabel(self.tr('Glossary'), glossary_row)
        glossary_label.setObjectName('RunPipelineSettingLabel')
        glossary_layout.addWidget(glossary_label)
        glossary_layout.addStretch()
        self.glossary_path_edit = GlossaryPathEdit(glossary_row)
        self.glossary_path_edit.setObjectName('RunPipelineGlossaryPath')
        self.glossary_path_edit.setReadOnly(True)
        self.glossary_path_edit.setFixedWidth(RUN_PIPELINE_GLOSSARY_DISPLAY_WIDTH)
        self._set_glossary_path_display(pcfg.module.llm_glossary_path)
        glossary_layout.addWidget(self.glossary_path_edit)
        self.glossary_file_button = self.glossary_path_edit.select_button
        select_glossary_text = self.tr('Select Glossary File')
        self.glossary_file_button.setToolTip(select_glossary_text)
        self.glossary_file_button.setAccessibleName(select_glossary_text)

        self.glossary_mode_combobox = BottomBorderComboBox(glossary_row)
        self.glossary_mode_combobox.setObjectName(
            'RunPipelineGlossaryModeComboBox'
        )
        self.glossary_mode_combobox.addItem(
            self.tr('Matching'),
            LLMGlossaryMode.Matching,
        )
        self.glossary_mode_combobox.addItem(
            self.tr('All'),
            LLMGlossaryMode.All,
        )
        glossary_mode_index = self.glossary_mode_combobox.findData(
            pcfg.module.llm_glossary_mode
        )
        self.glossary_mode_combobox.setCurrentIndex(max(glossary_mode_index, 0))
        self.glossary_mode_combobox.setFixedWidth(
            RUN_PIPELINE_SETTING_CONTROL_WIDTH
        )
        glossary_row.setVisible(self._llm_settings_visible)
        mode_row = QWidget(section)
        self.glossary_mode_row = mode_row
        mode_row.setObjectName('RunPipelineGeneralSettingRow')
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        mode_label = QLabel(self.tr('Mode'), mode_row)
        mode_label.setObjectName('RunPipelineSettingLabel')
        mode_layout.addWidget(mode_label)
        mode_layout.addStretch()
        mode_layout.addWidget(self.glossary_mode_combobox)
        mode_row.setVisible(self._llm_settings_visible)
        translation_grid.addWidget(glossary_row, 2, 0)
        translation_grid.addWidget(mode_row, 2, 1)

        self.context_row.setVisible(not self._llm_settings_visible)

        self.source_combobox.currentTextChanged.connect(
            self._on_translate_source_changed
        )
        self.target_combobox.currentTextChanged.connect(
            self._on_translate_target_changed
        )
        self.context_combobox.currentIndexChanged.connect(
            self._on_translate_context_changed
        )
        self.llm_context_combobox.currentIndexChanged.connect(
            self._on_llm_context_changed
        )
        self.prior_context_token_budget.valueChanged.connect(
            self._on_prior_context_token_budget_changed
        )
        self.glossary_path_edit.textChanged.connect(
            self._on_glossary_path_changed
        )
        self.glossary_file_button.clicked.connect(self._select_glossary_file)
        self.glossary_mode_combobox.currentIndexChanged.connect(
            self._on_glossary_mode_changed
        )

    def setTranslatorMetadata(self, metadata: dict) -> None:
        self.translator_metadata = metadata or {}
        source = self.translator_metadata.get(
            'lang_source',
            pcfg.module.translate_source,
        )
        target = self.translator_metadata.get(
            'lang_target',
            pcfg.module.translate_target,
        )
        for combobox, key, current in (
            (self.source_combobox, 'supported_src_list', source),
            (self.target_combobox, 'supported_tgt_list', target),
        ):
            blocker = QSignalBlocker(combobox)
            combobox.clear()
            combobox.addItems(self._translation_options(key, current))
            combobox.setCurrentText(current)
            del blocker

        self._llm_settings_visible = (
            self.translator_metadata.get('name') == LLM_TRANSLATOR_KEY
        )
        self.context_row.setVisible(not self._llm_settings_visible)
        self.llm_context_row.setVisible(self._llm_settings_visible)
        self.history_budget_row.setVisible(
            self._llm_settings_visible
            and pcfg.module.llm_translate_context == LLMTranslateContext.HISTORY
        )
        self.glossary_row.setVisible(self._llm_settings_visible)
        self.glossary_mode_row.setVisible(self._llm_settings_visible)
        self._fit_to_current_workflow()

    def _on_translate_source_changed(self, source: str):
        pcfg.module.translate_source = source
        self.translate_source_changed.emit(source)

    def _on_translate_target_changed(self, target: str):
        pcfg.module.translate_target = target
        self.translate_target_changed.emit(target)

    def _on_translate_context_changed(self):
        context = self.context_combobox.currentData()
        pcfg.module.translate_context = context

    def _on_llm_context_changed(self):
        context = self.llm_context_combobox.currentData()
        pcfg.module.llm_translate_context = context
        self.history_budget_row.setVisible(
            self._llm_settings_visible
            and context == LLMTranslateContext.HISTORY
        )

    def _on_prior_context_token_budget_changed(self, budget: int):
        pcfg.module.llm_prior_context_token_budget = budget

    def _on_glossary_path_changed(self, path: str):
        self._set_glossary_path_display(path)

    def _set_glossary_path_display(self, path: str):
        pcfg.module.llm_glossary_path = path
        blocker = QSignalBlocker(self.glossary_path_edit)
        self.glossary_path_edit.setText(os.path.basename(path) if path else '')
        del blocker

    def _on_glossary_mode_changed(self):
        pcfg.module.llm_glossary_mode = self.glossary_mode_combobox.currentData()

    def _select_glossary_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr('Select Glossary File'),
            self.glossary_path_edit.text(),
            self.tr('Glossary Files (*.json *.txt *.tsv)'),
        )
        if path:
            self._set_glossary_path_display(path)

    def _add_settings_section(
        self,
        layout: QVBoxLayout,
        key,
        title: str = '',
        show_header: bool = True,
    ):
        section = QWidget(self)
        section.setObjectName('RunPipelineSettingsSection')
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        section.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(6)

        if show_header:
            expanded = type(self)._module_settings_expanded[key]
            header = self._create_expanding_header(
                section,
                title,
                'RunPipelineModuleSettingsHeader',
                expanded,
            )
            section_layout.addWidget(header)
            body = QWidget(section)
            body.setObjectName('RunPipelineModuleSettingsBody')
            body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(18, 0, 0, 0)
            body_layout.setSpacing(8)
            section_layout.addWidget(body)
            body.setVisible(header.isChecked())
            self.module_settings_headers[key] = header
            self.module_settings_bodies[key] = body
            header.setProperty('settingsKey', key)
            header.toggled.connect(self._on_settings_header_toggled)
        else:
            body = section
            body_layout = section_layout
        self.settings_sections[key] = section
        layout.addWidget(section)
        if show_header:
            return section, body, body_layout
        return section, body_layout

    def _on_settings_header_toggled(self, expanded: bool) -> None:
        header = self.sender()
        if not isinstance(header, ExpandingToolButton):
            return
        key = header.property('settingsKey')
        body = self.module_settings_bodies.get(key)
        if body is not None:
            self._set_module_settings_expanded(key, header, body, expanded)

    def _project_page_names(self):
        pages = getattr(self.project, 'pages', None)
        return list(pages) if pages is not None else []

    def selected_pages(self):
        pages = self._project_page_names()
        if not pages:
            return []
        return pages[self.range_start.value() - 1 : self.range_end.value()]

    def _on_page_range_changed(self, start: int, end: int):
        type(self)._page_range = (start, end)

    @staticmethod
    def _settings_section_has_content(section: QWidget) -> bool:
        return section.layout().count() > 0

    def _on_stage_toggled(
        self,
        stage_index: int,
        section_index: int,
        checked: bool,
    ):
        pcfg.module.set_stage_enabled(stage_index, checked)
        section = self.settings_sections.get(section_index)
        section_body = self.module_settings_bodies.get(section_index)
        if section is not None:
            section.setVisible(
                checked
                and section_body is not None
                and self._settings_section_has_content(section_body)
            )
        self._refresh_progress()
        self._fit_to_current_workflow()

    def _on_stage_button_toggled(self, checked: bool) -> None:
        button = self.sender()
        if not isinstance(button, PipelineModuleButton):
            return
        self._on_stage_toggled(
            int(button.property('stageIndex')),
            int(button.property('sectionIndex')),
            checked,
        )

    def setModuleSelection(self, module_type: str, module_name: str) -> None:
        for activator in self.module_activators:
            if activator.module_type == module_type:
                activator.setModule(module_name)
                return

    def _set_module_settings_expanded(
        self,
        key: int,
        header: ExpandingToolButton,
        body: QWidget,
        expanded: bool,
    ):
        states = list(type(self)._module_settings_expanded)
        states[key] = expanded
        type(self)._module_settings_expanded = tuple(states)
        self._refresh_expanding_header_chevron(header, expanded)
        body.setVisible(expanded)
        section = body.parentWidget()
        section.layout().invalidate()
        section.updateGeometry()
        self._fit_to_current_workflow()

    def _fit_to_current_workflow(self):
        current_page = self.content_stack.currentWidget()
        current_page.layout().invalidate()
        current_page.layout().activate()
        current_page.updateGeometry()
        self.content_dock.setMinimumHeight(0)
        self.content_dock.setMaximumHeight(current_page.sizeHint().height())
        self.content_stack.updateGeometry()
        self.content_dock.updateGeometry()
        surface = self.content_dock.parentWidget()
        surface.layout().invalidate()
        surface.layout().activate()
        surface.updateGeometry()
        self.setMinimumHeight(0)
        self.layout().invalidate()
        self.layout().activate()
        target_height = self.sizeHint().height()
        self.resize(self.width(), target_height)
        self.setMinimumHeight(target_height)

    def _create_expanding_header(
        self,
        parent: QWidget,
        text: str,
        object_name: str,
        expanded: bool = False,
    ) -> ExpandingToolButton:
        header = ExpandingToolButton(parent)
        header.setObjectName(object_name)
        button_style = getattr(Qt, 'ToolButtonStyle', Qt)
        header.setToolButtonStyle(button_style.ToolButtonTextBesideIcon)
        header.setText('\u2009' + text)
        header.setCheckable(True)
        header.setChecked(expanded)
        header.setIconSize(QSize(12, 12))
        self._refresh_expanding_header_chevron(header, expanded)
        return header

    def _refresh_expanding_header_chevron(
        self,
        header: ExpandingToolButton,
        expanded: bool,
    ):
        icon_name = 'chevron-down.svg' if expanded else 'chevron-right.svg'
        pixmap = render_svg_pixmap(
            themed_icon_path(icon_name),
            12,
            12,
            self.devicePixelRatioF(),
        )
        header.setIcon(QIcon(pixmap))

    def changeEvent(self, event):
        if (
            event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange)
            and hasattr(self, 'module_settings_headers')
        ):
            for header in self.module_settings_headers.values():
                self._refresh_expanding_header_chevron(
                    header,
                    header.isChecked(),
                )
        return super().changeEvent(event)

    def _refresh_progress(self):
        if not hasattr(self, 'progress_bar') or self.project is None:
            return
        pages = self._project_page_names()
        finished_pages = [bool(self.project.get_page_progress(page)) for page in pages]
        self.page_range_progress.set_finished_pages(finished_pages)

    def _build_rendering_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName('RunPipelineRenderingPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)

        self._add_section_divider(layout, self.tr('Settings'), show_line=False)

        setting_row = QWidget(page)
        setting_row.setObjectName('RunPipelineRenderingSetting')
        setting_layout = QHBoxLayout(setting_row)
        setting_layout.setContentsMargins(2, 2, 2, 2)
        setting_layout.setSpacing(7)
        self.render_without_text_style_update = QCheckBox(setting_row)
        self.render_without_text_style_update.setObjectName(
            'RunPipelineRenderWithoutTextStyleUpdate'
        )
        self.render_without_text_style_update.setChecked(
            pcfg.render_without_text_style_update
        )
        self.render_without_text_style_update.toggled.connect(
            self._on_render_without_text_style_update_changed
        )
        setting_layout.addWidget(self.render_without_text_style_update)
        setting_label = QLabel(self.tr('Render without update text style'), setting_row)
        setting_label.setObjectName('RunPipelineRenderingSettingLabel')
        setting_layout.addWidget(setting_label)
        setting_layout.addStretch()
        layout.addWidget(setting_row)
        layout.addStretch()
        return page

    def _set_pipeline_page(self, index: int, persist: bool = True):
        rendering = index == 1
        if persist:
            pcfg.run_pipeline_mode = 'rendering' if rendering else 'pipeline'
        self.content_stack.setCurrentIndex(1 if rendering else 0)
        self.run_button.setVisible(not rendering)
        self.continue_button.setVisible(not rendering)
        self.render_button.setVisible(rendering)
        self.render_button.setDefault(rendering)
        self.continue_button.setDefault(not rendering)
        self._fit_to_current_workflow()

    def _on_render_without_text_style_update_changed(self, checked: bool):
        pcfg.render_without_text_style_update = checked

    def _add_section_divider(
        self,
        layout: QVBoxLayout,
        text: str,
        show_line: bool = True,
    ):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        title = QLabel(text, self)
        title.setObjectName('RunPipelineSectionTitle')
        row.addWidget(title)

        if not show_line:
            row.addStretch(1)
        else:
            line = QFrame(self)
            line.setObjectName('RunPipelineSectionLine')
            frame_shape = getattr(QFrame, 'Shape', QFrame)
            line.setFrameShape(frame_shape.HLine)
            row.addWidget(line, 1)
        layout.addLayout(row)
        return title

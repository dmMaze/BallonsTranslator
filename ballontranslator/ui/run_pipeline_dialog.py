from qtpy.QtCore import QCoreApplication, QEvent, QPointF, QSize, Qt
from qtpy.QtGui import QIcon, QMouseEvent, QPainter, QPalette, QPen
from qtpy.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .misc import themed_icon_path
from .framelesswindow import FramelessMoveResize
from .llm_modality import (
    LLM_MODALITY_IMAGE,
    LLM_MODALITY_IMAGE_COLOR,
    LLM_MODALITY_TEXT,
    LLM_MODALITY_TEXT_COLOR,
    LLM_MODALITY_VISION,
    LLM_MODALITY_VISION_COLOR,
    modality_badge_qcolor,
)
from ballontranslator.utils.config import pcfg


RUN_PIPELINE_DIALOG_WIDTH = 460


def _pipeline_text(source: str) -> str:
    # Keep using the existing translation catalog entries after moving the UI
    # out of MainWindow.
    return QCoreApplication.translate('MainWindow', source)


class DialogCloseButton(QAbstractButton):
    """Small title-bar button that paints its own close glyph.

    >>> DialogCloseButton.__name__
    'DialogCloseButton'
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName('RunPipelineCloseButton')
        self.setFixedSize(26, 26)
        self.setToolTip(self.tr('Close'))
        self.setAccessibleName(self.tr('Close'))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        render_hint = getattr(QPainter, 'RenderHint', QPainter).Antialiasing
        painter.setRenderHint(render_hint)

        if self.underMouse() or self.isDown():
            color_role = getattr(QPalette, 'ColorRole', QPalette)
            background = self.palette().color(color_role.Highlight)
            background.setAlpha(55 if self.isDown() else 35)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(self.rect(), 6, 6)

        color_role = getattr(QPalette, 'ColorRole', QPalette)
        color = self.palette().color(color_role.WindowText)
        color.setAlpha(210)
        pen = QPen(color)
        pen.setWidthF(1.6)
        pen.setCapStyle(getattr(getattr(Qt, 'PenCapStyle', Qt), 'RoundCap'))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inset = 8.5
        end = self.width() - inset
        painter.drawLine(QPointF(inset, inset), QPointF(end, end))
        painter.drawLine(QPointF(end, inset), QPointF(inset, end))


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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 6, 1)
        layout.setSpacing(4)

        self.icon_label = QLabel(self)
        self.icon_label.setObjectName('RunPipelineModuleIcon')
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setAttribute(widget_attribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.icon_label)

        self.text_label = QLabel(text, self)
        self.text_label.setObjectName('RunPipelineModuleLabel')
        self.text_label.setAttribute(widget_attribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.text_label)
        layout.addStretch()

        self.toggled.connect(self._refresh_visuals)
        self._refresh_visuals(self.isChecked())

    def _refresh_visuals(self, active: bool):
        _, _, modality_color = self._MODALITY_VISUALS[self.modality]
        icon_path = themed_icon_path(
            self.active_icon_name if active else self.inactive_icon_name
        )
        self.icon_label.setPixmap(QIcon(icon_path).pixmap(QSize(16, 16)))

        badge_color = modality_badge_qcolor(modality_color)
        if active:
            background = 'rgba({}, {}, {}, {})'.format(
                badge_color.red(),
                badge_color.green(),
                badge_color.blue(),
                badge_color.alpha(),
            )
        else:
            background = 'transparent'
        badge_style = (
            'QLabel#RunPipelineModuleIcon {{ '
            'background-color: {}; border-radius: 6px; '
            '}}'.format(background)
        )
        if self.icon_label.styleSheet() != badge_style:
            self.icon_label.setStyleSheet(badge_style)
        self.text_label.setProperty('moduleActive', active)
        self.text_label.style().unpolish(self.text_label)
        self.text_label.style().polish(self.text_label)

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


class RunPipelineDialog(QDialog):
    """Choose and configure the pipeline action to run.

    >>> len({RunPipelineDialog.RUN, RunPipelineDialog.CONTINUE, RunPipelineDialog.RENDER})
    3
    """

    RUN = 1
    CONTINUE = 2
    RENDER = 3
    RESIZE_BORDER_WIDTH = 5

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._app_event_filter_installed = False
        self.setObjectName('RunPipelineDialog')
        self.setWindowTitle(self.tr('Run Pipeline'))
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
        self.title_label = QLabel(self.tr('Run Pipeline'), self.title_bar)
        self.title_label.setObjectName('RunPipelineTitle')
        self.title_label.setMouseTracking(True)
        title_row.addWidget(self.title_label)
        title_row.addSpacing(12)
        self.workflow_selector = QComboBox(surface)
        self.workflow_selector.setObjectName('RunPipelineWorkflowSelector')
        self.workflow_selector.addItems((self.tr('Automation'), self.tr('Rendering')))
        pipeline_mode = str(pcfg.run_pipeline_mode).lower()
        self.workflow_selector.setCurrentIndex(1 if pipeline_mode == 'rendering' else 0)
        self.workflow_selector.setFixedWidth(126)
        title_row.addWidget(self.workflow_selector)
        title_row.addStretch()
        self.close_button = DialogCloseButton(surface)
        self.close_button.clicked.connect(self.reject)
        title_row.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        self.content_stack = QStackedWidget(surface)
        self.content_stack.setObjectName('RunPipelineContentStack')
        self.content_stack.addWidget(self._build_automation_page())
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
        button_row.setContentsMargins(0, 4, 0, 0)
        button_row.setSpacing(8)
        button_row.addStretch()

        self.run_button = QPushButton(_pipeline_text('Run'), surface)
        self.run_button.setObjectName('RunPipelineSecondaryButton')
        self.run_button.clicked.connect(lambda: self.done(self.RUN))
        button_row.addWidget(self.run_button)

        self.continue_button = QPushButton(_pipeline_text('Continue'), surface)
        self.continue_button.setObjectName('RunPipelinePrimaryButton')
        self.continue_button.setDefault(True)
        self.continue_button.clicked.connect(lambda: self.done(self.CONTINUE))
        button_row.addWidget(self.continue_button)

        self.render_button = QPushButton(self.tr('Render'), surface)
        self.render_button.setObjectName('RunPipelinePrimaryButton')
        self.render_button.clicked.connect(lambda: self.done(self.RENDER))
        self.render_button.hide()
        button_row.addWidget(self.render_button)
        layout.addLayout(button_row)

        self.workflow_selector.currentIndexChanged.connect(self._set_pipeline_page)
        self._set_pipeline_page(self.workflow_selector.currentIndex())
        initial_height = self.sizeHint().height()
        self.setMinimumHeight(initial_height)
        self.resize(RUN_PIPELINE_DIALOG_WIDTH, initial_height)

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

    def _build_automation_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName('RunPipelineAutomationPage')
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
                self.tr('Text Detection'),
                LLM_MODALITY_VISION,
                'textdetect_activate.svg',
                'textdetect.svg',
            ),
            (self.tr('OCR'), LLM_MODALITY_VISION, '', ''),
            (self.tr('Translation'), LLM_MODALITY_TEXT, '', ''),
            (self.tr('Inpainting'), LLM_MODALITY_IMAGE, '', ''),
        )
        self.module_buttons = []
        for index, (name, modality, active_icon, inactive_icon) in enumerate(stage_specs):
            button = PipelineModuleButton(
                name,
                modality,
                stages,
                active_icon_name=active_icon,
                inactive_icon_name=inactive_icon,
            )
            button.setChecked(pcfg.module.stage_enabled(index))
            button.toggled.connect(
                lambda checked, stage_index=index: pcfg.module.set_stage_enabled(
                    stage_index,
                    checked,
                )
            )
            stage_layout.addWidget(button, index // 2, index % 2)
            self.module_buttons.append(button)
        layout.addWidget(stages)

        self.settings_header = self._add_section_divider(
            layout,
            self.tr('Settings'),
            folded=True,
        )
        return page

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

    def _set_pipeline_page(self, index: int):
        rendering = index == 1
        pcfg.run_pipeline_mode = 'rendering' if rendering else 'automation'
        self.content_stack.setCurrentIndex(1 if rendering else 0)
        self.run_button.setVisible(not rendering)
        self.continue_button.setVisible(not rendering)
        self.render_button.setVisible(rendering)
        self.render_button.setDefault(rendering)
        self.continue_button.setDefault(not rendering)

    def _on_render_without_text_style_update_changed(self, checked: bool):
        pcfg.render_without_text_style_update = checked

    def _add_section_divider(
        self,
        layout: QVBoxLayout,
        text: str,
        folded: bool = False,
        show_line: bool = True,
    ):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        if folded:
            title = QToolButton(self)
            title.setObjectName('RunPipelineSettingsHeader')
            arrow_type = getattr(Qt, 'ArrowType', Qt)
            button_style = getattr(Qt, 'ToolButtonStyle', Qt)
            title.setArrowType(arrow_type.RightArrow)
            title.setToolButtonStyle(button_style.ToolButtonTextBesideIcon)
            title.setText(text)
        else:
            title = QLabel(text, self)
            title.setObjectName('RunPipelineSectionTitle')
        row.addWidget(title)

        if folded or not show_line:
            row.addStretch(1)
        else:
            line = QFrame(self)
            line.setObjectName('RunPipelineSectionLine')
            frame_shape = getattr(QFrame, 'Shape', QFrame)
            line.setFrameShape(frame_shape.HLine)
            row.addWidget(line, 1)
        layout.addLayout(row)
        return title

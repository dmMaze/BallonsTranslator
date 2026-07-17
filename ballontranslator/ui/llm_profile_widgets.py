import copy
import json
import uuid
from typing import get_type_hints

from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QPlainTextEdit,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QFrame,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QToolButton,
    QGroupBox,
    QScrollArea,
    QStyle,
    QStyleOptionGroupBox,
)
from qtpy.QtCore import QEvent, QRectF, QTimer, Qt, Signal
from qtpy.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction

from .custom_widget import ParamComboBox, NoBorderPushBtn, ScrollBar
from .icon_rendering import render_svg_pixmap
from .llm_modality import (
    LLM_MODALITY_IMAGE_COLOR,
    LLM_MODALITY_TEXT_COLOR,
    LLM_MODALITY_VISION_COLOR,
    modality_badge_qcolor,
)
from .misc import themed_icon_path
from .module_parse_widgets import ParamWidget, SecretParamWidget
from ballontranslator.utils.shared import size2width
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLM_INPAINT_KEY,
    LLM_OCR_KEY,
    LLM_TRANSLATOR_KEY,
    LLMProfile,
    copy_profile,
    profile_by_id,
    profile_to_export_dict,
    profiles_from_json,
    restore_builtin_profiles,
    resolve_api_key,
    store_api_key,
)


PROFILE_COMMON_PARAM_DEFS = [
    ('require_api_key', 'checkbox'),
    ('base_url', 'line_editor'),
    ('max_tokens', 'line_editor'),
    ('temperature', 'line_editor'),
    ('top_p', 'line_editor'),
    ('frequency_penalty', 'line_editor'),
    ('presence_penalty', 'line_editor'),
    ('low_vram_mode', 'checkbox'),
    ('json_schema_response_format', 'checkbox'),
]
PROFILE_MODALITY_PARAM_DEFS = {
    'text': [
        ('thinking_level', 'selector'),
        ('prompt', 'editor'),
    ],
    'vision': [
        ('vision_detail_level', 'selector'),
        ('vision_prompt', 'editor'),
    ],
    'image': [
        ('image_base_url', 'line_editor'),
        ('image_prompt', 'editor'),
    ],
}
PROFILE_EDITOR_WIDTH_SCALE = 1.15

PROFILE_FIELD_TYPES = get_type_hints(LLMProfile)


def _widen_profile_editor(editor: QWidget):
    editor.setFixedWidth(round(editor.width() * PROFILE_EDITOR_WIDTH_SCALE))


class ProfileNameEdit(QLineEdit):
    """Title-like profile name editor that only edits on demand.

    Example:
        >>> ProfileNameEdit.__name__
        'ProfileNameEdit'
    """

    edit_finished = Signal()
    edit_requested = Signal()

    def __init__(self, text: str = '', parent: QWidget = None):
        super().__init__(text, parent)
        self.setObjectName('LLMProfileNameEdit')
        self.editingFinished.connect(self.finishEdit)
        font = self.font()
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)
        self.setFixedHeight(18)
        self.resizeToContent()

    def focusOutEvent(self, event):
        self.finishEdit()
        return super().focusOutEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.finishEdit()
            event.accept()
            return
        return super().keyPressEvent(event)

    def resizeToContent(self):
        width = self.fontMetrics().boundingRect(self.text() or 'Profile').width() + 14
        self.setFixedWidth(max(90, min(width, 260)))

    def startEdit(self, select_all: bool = True):
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.setFocus()
        if select_all:
            self.selectAll()

    def finishEdit(self):
        if not self.isVisible():
            return
        self.clearFocus()
        self.resizeToContent()
        self.edit_finished.emit()


class CachedSvgStatusIcon(QLabel):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._icon_path = ''
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def setIconFile(self, filename: str):
        self.setIconPath(themed_icon_path(filename))

    def setIconPath(self, path: str):
        self._icon_path = path
        self.update()

    def paintEvent(self, event):
        pixmap = self._renderIconPixmap()
        if pixmap.isNull():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

    def _renderIconPixmap(self) -> QPixmap:
        return render_svg_pixmap(
            self._icon_path,
            self.width(),
            self.height(),
            self.devicePixelRatioF(),
            0,
            (0, 0, 0, 0),
            0,
        )


class CapabilityBadgeLabel(CachedSvgStatusIcon):
    """Clickable LLM capability icon shown beside its model label.

    Example:
        >>> CapabilityBadgeLabel.__name__
        'CapabilityBadgeLabel'
    """

    clicked = Signal()

    def __init__(self, active_color: QColor, parent: QWidget = None):
        super().__init__(parent)
        self._active_color = active_color
        self.setObjectName('LLMProfileCapabilityBadge')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(20, 20)

    def paintEvent(self, event):
        pixmap = self._renderIconPixmap()
        if pixmap.isNull():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

    def _renderIconPixmap(self) -> QPixmap:
        background = self._active_color.getRgb() if bool(self.property('capabilityActive')) else (0, 0, 0, 0)
        return render_svg_pixmap(
            self._icon_path,
            self.width(),
            self.height(),
            self.devicePixelRatioF(),
            2,
            background,
            6,
        )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        return super().mouseReleaseEvent(event)


class ClickableFieldLabel(QLabel):
    """Field label that can act as a compact toggle target.

    Example:
        >>> ClickableFieldLabel.__name__
        'ClickableFieldLabel'
    """

    clicked = Signal()

    def __init__(self, text: str = '', parent: QWidget = None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        return super().mouseReleaseEvent(event)


class ProfileDetailsWidget(QWidget):
    """Profile detail editor grouped by modality while keeping one lookup path.

    Example:
        >>> ProfileDetailsWidget.__name__
        'ProfileDetailsWidget'
    """

    paramwidget_edited = Signal(str, dict)

    def __init__(self, common_params: dict, sections: dict, scrollWidget: QWidget = None, parent=None):
        super().__init__(parent)
        self.setObjectName('LLMProfileDetails')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.param_widgets = {}
        self._param_owners = {}
        self._param_groups = []
        self.section_widgets = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._addParamWidget(layout, common_params, scrollWidget, self)
        for section_key, (title, params) in sections.items():
            section = QWidget(self)
            section.setObjectName('LLMProfileDetailSection')
            section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(6)
            title_label = QLabel(title, section)
            title_label.setObjectName('LLMProfileDetailSectionTitle')
            title_lines = [QFrame(section), QFrame(section)]
            for title_line in title_lines:
                title_line.setObjectName('LLMProfileDetailSectionLine')
                title_line.setFrameShape(QFrame.Shape.HLine)
                title_line.setFrameShadow(QFrame.Shadow.Plain)
                title_line.setFixedHeight(1)
                title_line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            header_widget = QWidget(section)
            header_widget.setFixedWidth(240)
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 8, 0, 8)
            header_layout.setSpacing(0)
            header_layout.addWidget(title_lines[0], 1, Qt.AlignmentFlag.AlignVCenter)
            header_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(title_lines[1], 1, Qt.AlignmentFlag.AlignVCenter)
            section_layout.addWidget(header_widget, 0, Qt.AlignmentFlag.AlignHCenter)
            self._addParamWidget(section_layout, params, scrollWidget, section)
            self.section_widgets[section_key] = section
            layout.addWidget(section, 0, Qt.AlignmentFlag.AlignLeft)
        self._alignParamColumns()

    def _addParamWidget(self, layout, params: dict, scrollWidget: QWidget, parent: QWidget):
        param_widget = ParamWidget(params, scrollWidget=scrollWidget, parent=parent)
        param_widget.layout().setContentsMargins(0, 0, 0, 0)
        param_widget.paramwidget_edited.connect(self.paramwidget_edited.emit)
        self._param_groups.append(param_widget)
        for key, editor in param_widget.param_widgets.items():
            self.param_widgets[key] = editor
            self._param_owners[key] = param_widget
            if isinstance(editor, (QLineEdit, ParamComboBox, QPlainTextEdit)):
                _widen_profile_editor(editor)
        layout.addWidget(param_widget, 0, Qt.AlignmentFlag.AlignLeft)

    def _alignParamColumns(self):
        prompt_width = max(
            (
                self.param_widgets[key].minimumWidth()
                for key in ('prompt', 'vision_prompt', 'image_prompt')
                if key in self.param_widgets
            ),
            default=0,
        )
        for group in self._param_groups:
            inline_rows = [row for row in group.param_rows.values() if len(row) > 1]
            inline_editors = [
                editor
                for editor in group.param_widgets.values()
                if isinstance(editor, (QLineEdit, ParamComboBox))
            ]
            if not inline_rows or not inline_editors:
                continue
            spacing = max(0, group.param_layout.horizontalSpacing())
            # Preserve editor widths and align their right edge with the full-width prompt.
            label_width = max(
                max(row[0].sizeHint().width() for row in inline_rows),
                prompt_width - max(editor.width() for editor in inline_editors) - spacing,
            )
            group.param_layout.setColumnMinimumWidth(0, label_width)

    def setParamVisible(self, param_key: str, visible: bool):
        owner = self._param_owners.get(param_key)
        if owner is not None:
            owner.setParamVisible(param_key, visible)

    def setSectionVisible(self, section_key: str, visible: bool):
        section = self.section_widgets.get(section_key)
        if section is not None:
            section.setVisible(visible)


class ProfileCardWidget(QGroupBox):
    """Compact card editor for a single LLM profile.

    Example:
        >>> ProfileCardWidget.__name__
        'ProfileCardWidget'
    """

    profile_changed = Signal()
    profile_selector_changed = Signal()
    profile_summary_changed = Signal()
    copy_json_requested = Signal(str)
    copy_requested = Signal(str)
    delete_requested = Signal(str)
    set_translator_requested = Signal(str)
    set_ocr_requested = Signal(str)
    set_inpainter_requested = Signal(str)

    def __init__(self, profile: LLMProfile, scrollWidget: QWidget = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('LLMProfileCard')
        self.profile = profile
        self.setTitle(profile.name)
        self.setToolTip(self.tr('Double click the name to edit. Right click for profile actions.'))
        self.scrollWidget = scrollWidget
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self._name_editing = False
        self._model_editing = False
        self._vision_model_editing = False
        self._image_model_editing = False
        self._selection_border_colors = []
        self._previous_model_text = ''
        self._previous_vision_model_text = ''
        self._previous_image_model_text = ''
        self._action_buttons_visible = False
        self.profile_param_display_names = {
            'base_url': self.tr('Base URL'),
            'image_base_url': self.tr('Image Base URL'),
            'require_api_key': self.tr('Require API Key'),
            'vision_model': self.tr('Vision Model'),
            'image_model': self.tr('Image Model'),
            'vision_detail_level': self.tr('Vision Detail Level'),
            'thinking_level': self.tr('Thinking Level'),
            'max_tokens': self.tr('Max Tokens'),
            'temperature': self.tr('Temperature'),
            'top_p': self.tr('Top P'),
            'frequency_penalty': self.tr('Frequency Penalty'),
            'presence_penalty': self.tr('Presence Penalty'),
            'json_schema_response_format': self.tr('JSON Schema Response'),
            'prompt': self.tr('Prompt'),
            'vision_prompt': self.tr('Prompt'),
            'image_prompt': self.tr('Prompt'),
            'low_vram_mode': self.tr('Low VRAM Mode'),
        }
        self.profile_param_descriptions = {
            'base_url': self.tr('OpenAI-compatible API base URL.'),
            'image_base_url': self.tr('OpenAI-compatible image API base URL used only by LLMInpaint.'),
            'require_api_key': self.tr('Require API key before running this LLM task.'),
            'vision_model': self.tr('Model used by LLMOCR for image OCR.'),
            'image_model': self.tr('Model used by LLMInpaint for image cleanup.'),
            'vision_detail_level': self.tr('Image detail level sent to vision-capable providers.'),
            'thinking_level': self.tr('Reasoning effort sent only when it is not None.'),
            'prompt': self.tr('Additional translation instructions for style and wording.'),
            'vision_prompt': self.tr('Instructions sent to the vision model for OCR.'),
            'image_prompt': self.tr('Instructions sent to the image model for cleanup.'),
            'max_tokens': self.tr('Maximum generated response tokens, not input/context tokens.'),
            'temperature': self.tr('Sampling temperature.'),
            'top_p': self.tr('Top-p sampling.'),
            'frequency_penalty': self.tr('Sent only when greater than 0. Some OpenAI-compatible providers may ignore or reject it.'),
            'presence_penalty': self.tr('Sent only when greater than 0. Some OpenAI-compatible providers may ignore or reject it.'),
            'json_schema_response_format': self.tr('Request responses with the translation JSON schema. Useful for LM Studio; disable it if a provider rejects json_schema response_format.'),
            'low_vram_mode': self.tr('Preserved compatibility flag for local profiles.'),
        }
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 14)
        layout.setSpacing(8)

        self.name_edit = ProfileNameEdit(profile.name, self)
        self.name_edit.edit_requested.connect(self.startNameEdit)
        self.name_edit.edit_finished.connect(self.on_name_edit_finished)
        self.name_edit.hide()

        self.text_badge = CapabilityBadgeLabel(modality_badge_qcolor(LLM_MODALITY_TEXT_COLOR), self)
        self.text_badge.clicked.connect(self.toggleTextSupport)
        self.vision_badge = CapabilityBadgeLabel(modality_badge_qcolor(LLM_MODALITY_VISION_COLOR), self)
        self.vision_badge.clicked.connect(self.toggleVisionSupport)
        self.image_badge = CapabilityBadgeLabel(modality_badge_qcolor(LLM_MODALITY_IMAGE_COLOR), self)
        self.image_badge.clicked.connect(self.toggleImageSupport)

        self.key_status_icon = CachedSvgStatusIcon(self)
        self.key_status_icon.setObjectName('LLMProfileKeyStatusIcon')
        self.key_status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.key_status_icon.setFixedSize(16, 16)

        self.edit_icon = QIcon(themed_icon_path('edit.svg'))
        self.edit_icon_active = QIcon(themed_icon_path('edit_activate.svg'))
        self.more_btn = QToolButton(self)
        self.more_btn.setObjectName('LLMProfileConfigButton')
        self.more_btn.setIcon(self.edit_icon)
        self.more_btn.setToolTip(self.tr('Edit'))
        self.more_btn.clicked.connect(self.toggleExpanded)
        self.more_btn.installEventFilter(self)
        self.more_btn.setFixedSize(22, 22)

        self.delete_btn = QToolButton(self)
        self.delete_btn.setObjectName('LLMProfileDeleteButton')
        self.delete_btn.setIcon(QIcon(themed_icon_path('titlebar_close.svg')))
        self.delete_btn.setToolTip(self.tr('Delete'))
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.profile.id))
        self.delete_btn.setFixedSize(18, 18)

        self.summary_widget = QWidget(self)
        self.summary_widget.setObjectName('LLMProfileSummaryGrid')
        self.summary_layout = QGridLayout(self.summary_widget)
        self.summary_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_layout.setHorizontalSpacing(12)
        self.summary_layout.setVerticalSpacing(8)
        self.summary_layout.setColumnStretch(1, 1)

        self.api_summary_widget = QWidget(self)
        self.api_summary_widget.setObjectName('LLMProfileSummaryColumn')
        self.api_summary_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        center_column = QVBoxLayout(self.api_summary_widget)
        center_column.setContentsMargins(0, 0, 0, 0)
        center_column.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        center_column.setSpacing(2)
        self.api_label = QLabel(self.tr('API Key'), self)
        self.api_label.setObjectName('LLMProfileFieldLabel')
        self.api_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.api_label.setFixedHeight(self.text_badge.height())
        api_label_row = QHBoxLayout()
        api_label_row.setContentsMargins(0, 0, 0, 0)
        api_label_row.setSpacing(6)
        api_label_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        api_label_row.addWidget(self.api_label, 0, Qt.AlignmentFlag.AlignLeft)
        api_label_row.addStretch(1)
        self.api_key_widget = SecretParamWidget('api_key', size='short')
        self.api_key_widget.editor.setObjectName('LLMProfileApiKeyEditor')
        self.api_key_widget.setToolTip(self.api_key_widget.editor.toolTip())
        self.api_label.setToolTip(self.api_key_widget.editor.toolTip())
        self.api_key_widget.setText(resolve_api_key(profile))
        self._api_key_editor_is_empty = self._apiKeyEditorIsEmpty()
        api_editor_row = QHBoxLayout()
        api_editor_row.setContentsMargins(0, 0, 0, 0)
        api_editor_row.setSpacing(4)
        api_editor_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        api_editor_row.addWidget(self.api_key_widget, 0, Qt.AlignmentFlag.AlignLeft)
        api_editor_row.addWidget(self.key_status_icon, 0, Qt.AlignmentFlag.AlignLeft)
        api_editor_row.addStretch(1)

        center_column.addLayout(api_label_row)
        center_column.addLayout(api_editor_row)

        self.model_summary_widget = QWidget(self)
        self.model_summary_widget.setObjectName('LLMProfileSummaryColumn')
        self.model_summary_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        right_column = QVBoxLayout(self.model_summary_widget)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        right_column.setSpacing(2)
        self.model_label = ClickableFieldLabel(self.tr('Text Model'), self)
        self.model_label.setObjectName('LLMProfileFieldLabel')
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.model_label.clicked.connect(self.toggleTextSupport)
        model_tooltip = self.tr('Text translation model used by LLMTranslator.')
        self.add_model_btn = QToolButton(self)
        self.add_model_btn.setObjectName('LLMProfileModelAddButton')
        self.add_model_btn.setIcon(QIcon(themed_icon_path('add.svg')))
        self.add_model_btn.setToolTip(self.tr('Add model'))
        self.add_model_btn.setFixedSize(16, 16)
        self.add_model_btn.clicked.connect(self.startModelEdit)
        self.remove_model_btn = QToolButton(self)
        self.remove_model_btn.setObjectName('LLMProfileModelRemoveButton')
        self.remove_model_btn.setIcon(QIcon(themed_icon_path('titlebar_min.svg')))
        self.remove_model_btn.setToolTip(self.tr('Delete current model'))
        self.remove_model_btn.setFixedSize(16, 16)
        self.remove_model_btn.clicked.connect(self.deleteCurrentModel)
        self.model_modality_row = QWidget(self.model_summary_widget)
        self.model_modality_row.setObjectName('LLMProfileModalityRow')
        self.model_modality_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        model_label_row = QHBoxLayout(self.model_modality_row)
        model_label_row.setContentsMargins(0, 0, 0, 0)
        model_label_row.setSpacing(4)
        model_label_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        model_label_row.addWidget(self.text_badge, 0, Qt.AlignmentFlag.AlignLeft)
        model_label_row.addWidget(self.model_label, 0, Qt.AlignmentFlag.AlignLeft)
        model_label_row.addWidget(self.add_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        model_label_row.addWidget(self.remove_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        model_label_row.addStretch(1)
        self.model_combo = ParamComboBox('model', profile.model_options, size=size2width('short'), scrollWidget=scrollWidget)
        self.model_combo.setObjectName('LLMProfileModelCombo')
        self.model_label.setToolTip(model_tooltip)
        self.model_combo.setToolTip(model_tooltip)
        self.model_combo.setEditable(False)
        self.model_combo.setCurrentText(profile.model)
        right_column.addWidget(self.model_modality_row)
        right_column.addWidget(self.model_combo, 0, Qt.AlignmentFlag.AlignLeft)

        self.vision_model_summary_widget = QWidget(self)
        self.vision_model_summary_widget.setObjectName('LLMProfileSummaryColumn')
        self.vision_model_summary_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        vision_column = QVBoxLayout(self.vision_model_summary_widget)
        vision_column.setContentsMargins(0, 0, 0, 0)
        vision_column.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        vision_column.setSpacing(2)
        self.vision_model_label = ClickableFieldLabel(self.tr('Vision Model'), self)
        self.vision_model_label.setObjectName('LLMProfileFieldLabel')
        self.vision_model_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.vision_model_label.clicked.connect(self.toggleVisionSupport)
        vision_model_tooltip = self.tr('Model used by LLMOCR for image OCR.')
        self.add_vision_model_btn = QToolButton(self)
        self.add_vision_model_btn.setObjectName('LLMProfileModelAddButton')
        self.add_vision_model_btn.setIcon(QIcon(themed_icon_path('add.svg')))
        self.add_vision_model_btn.setToolTip(self.tr('Add vision model'))
        self.add_vision_model_btn.setFixedSize(16, 16)
        self.add_vision_model_btn.clicked.connect(self.startVisionModelEdit)
        self.remove_vision_model_btn = QToolButton(self)
        self.remove_vision_model_btn.setObjectName('LLMProfileModelRemoveButton')
        self.remove_vision_model_btn.setIcon(QIcon(themed_icon_path('titlebar_min.svg')))
        self.remove_vision_model_btn.setToolTip(self.tr('Delete current vision model'))
        self.remove_vision_model_btn.setFixedSize(16, 16)
        self.remove_vision_model_btn.clicked.connect(self.deleteCurrentVisionModel)
        self.vision_model_modality_row = QWidget(self.vision_model_summary_widget)
        self.vision_model_modality_row.setObjectName('LLMProfileModalityRow')
        self.vision_model_modality_row.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        vision_model_label_row = QHBoxLayout(self.vision_model_modality_row)
        vision_model_label_row.setContentsMargins(0, 0, 0, 0)
        vision_model_label_row.setSpacing(4)
        vision_model_label_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        vision_model_label_row.addWidget(self.vision_badge, 0, Qt.AlignmentFlag.AlignLeft)
        vision_model_label_row.addWidget(self.vision_model_label, 0, Qt.AlignmentFlag.AlignLeft)
        vision_model_label_row.addWidget(self.add_vision_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        vision_model_label_row.addWidget(self.remove_vision_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        vision_model_label_row.addStretch(1)
        self.vision_model_combo = ParamComboBox(
            'vision_model',
            profile.vision_model_options,
            size=size2width('short'),
            scrollWidget=scrollWidget,
        )
        self.vision_model_combo.setObjectName('LLMProfileModelCombo')
        self.vision_model_label.setToolTip(vision_model_tooltip)
        self.vision_model_combo.setToolTip(vision_model_tooltip)
        self.vision_model_combo.setEditable(False)
        self.vision_model_combo.setCurrentText(profile.vision_model)
        vision_column.addWidget(self.vision_model_modality_row)
        vision_column.addWidget(self.vision_model_combo, 0, Qt.AlignmentFlag.AlignLeft)

        self.image_model_summary_widget = QWidget(self)
        self.image_model_summary_widget.setObjectName('LLMProfileSummaryColumn')
        self.image_model_summary_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        image_column = QVBoxLayout(self.image_model_summary_widget)
        image_column.setContentsMargins(0, 0, 0, 0)
        image_column.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        image_column.setSpacing(2)
        self.image_model_label = ClickableFieldLabel(self.tr('Image Model'), self)
        self.image_model_label.setObjectName('LLMProfileFieldLabel')
        self.image_model_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.image_model_label.clicked.connect(self.toggleImageSupport)
        image_model_tooltip = self.tr('Model used by LLMInpaint for image cleanup.')
        self.add_image_model_btn = QToolButton(self)
        self.add_image_model_btn.setObjectName('LLMProfileModelAddButton')
        self.add_image_model_btn.setIcon(QIcon(themed_icon_path('add.svg')))
        self.add_image_model_btn.setToolTip(self.tr('Add image model'))
        self.add_image_model_btn.setFixedSize(16, 16)
        self.add_image_model_btn.clicked.connect(self.startImageModelEdit)
        self.remove_image_model_btn = QToolButton(self)
        self.remove_image_model_btn.setObjectName('LLMProfileModelRemoveButton')
        self.remove_image_model_btn.setIcon(QIcon(themed_icon_path('titlebar_min.svg')))
        self.remove_image_model_btn.setToolTip(self.tr('Delete current image model'))
        self.remove_image_model_btn.setFixedSize(16, 16)
        self.remove_image_model_btn.clicked.connect(self.deleteCurrentImageModel)
        self.image_model_modality_row = QWidget(self.image_model_summary_widget)
        self.image_model_modality_row.setObjectName('LLMProfileModalityRow')
        self.image_model_modality_row.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        image_model_label_row = QHBoxLayout(self.image_model_modality_row)
        image_model_label_row.setContentsMargins(0, 0, 0, 0)
        image_model_label_row.setSpacing(4)
        image_model_label_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        image_model_label_row.addWidget(self.image_badge, 0, Qt.AlignmentFlag.AlignLeft)
        image_model_label_row.addWidget(self.image_model_label, 0, Qt.AlignmentFlag.AlignLeft)
        image_model_label_row.addWidget(self.add_image_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        image_model_label_row.addWidget(self.remove_image_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        image_model_label_row.addStretch(1)
        self.image_model_combo = ParamComboBox(
            'image_model',
            profile.image_model_options,
            size=size2width('short'),
            scrollWidget=scrollWidget,
        )
        self.image_model_combo.setObjectName('LLMProfileModelCombo')
        self.image_model_label.setToolTip(image_model_tooltip)
        self.image_model_combo.setToolTip(image_model_tooltip)
        self.image_model_combo.setEditable(False)
        self.image_model_combo.setCurrentText(profile.image_model)
        image_column.addWidget(self.image_model_modality_row)
        image_column.addWidget(self.image_model_combo, 0, Qt.AlignmentFlag.AlignLeft)

        _widen_profile_editor(self.api_key_widget)
        _widen_profile_editor(self.api_key_widget.editor)
        for editor in (self.model_combo, self.vision_model_combo, self.image_model_combo):
            _widen_profile_editor(editor)
            # The finish handlers own option insertion; otherwise Enter also lets
            # QComboBox insert the edit text before editingFinished is emitted.
            editor.setInsertPolicy(editor.InsertPolicy.NoInsert)
        self._setSummaryColumnWidth(self.model_summary_widget, model_label_row, self.model_combo)
        self._setSummaryColumnWidth(self.vision_model_summary_widget, vision_model_label_row, self.vision_model_combo)
        self._setSummaryColumnWidth(self.image_model_summary_widget, image_model_label_row, self.image_model_combo)
        self._sync_summary_grid()
        layout.addWidget(self.summary_widget)

        self.details = ProfileDetailsWidget(
            self._detail_params(PROFILE_COMMON_PARAM_DEFS),
            {
                'text': (self.tr('Text'), self._detail_params(PROFILE_MODALITY_PARAM_DEFS['text'])),
                'vision': (self.tr('Vision'), self._detail_params(PROFILE_MODALITY_PARAM_DEFS['vision'])),
                'image': (self.tr('Image'), self._detail_params(PROFILE_MODALITY_PARAM_DEFS['image'])),
            },
            scrollWidget=scrollWidget,
            parent=self,
        )
        self._install_detail_editor_scrollbars()
        layout.addWidget(self.details)
        self._sync_minimum_width_with_content()
        self.details.setVisible(False)
        self.setActionButtonsVisible(False)

        self.model_combo.paramwidget_edited.connect(self.on_model_edited)
        self.vision_model_combo.paramwidget_edited.connect(self.on_vision_model_edited)
        self.image_model_combo.paramwidget_edited.connect(self.on_image_model_edited)
        self.api_key_widget.editor.editingFinished.connect(self.on_api_key_finished)
        self.api_key_widget.editor.textChanged.connect(self.on_api_key_text_changed)
        self.details.paramwidget_edited.connect(self.on_detail_edited)
        self._position_header_controls()
        self.refreshTextBadge()
        self.refreshVisionBadge()
        self.refreshImageBadge()
        self.refreshConditionalVisibility()
        self.refreshSelectionBorder()

    def _install_detail_editor_scrollbars(self):
        for editor in self.details.findChildren(QPlainTextEdit):
            editor.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, editor, fadeout=False, hover_style=True)
            editor.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, editor, fadeout=False, hover_style=True)

    def syncFromProfile(self):
        self._syncComboBox(self.model_combo, self.profile.model_options, self.profile.model)
        self._syncComboBox(self.vision_model_combo, self.profile.vision_model_options, self.profile.vision_model)
        self._syncComboBox(self.image_model_combo, self.profile.image_model_options, self.profile.image_model)
        vision_detail_combo = self.details.param_widgets.get('vision_detail_level')
        if isinstance(vision_detail_combo, ParamComboBox):
            self._syncComboBox(
                vision_detail_combo,
                self.profile.vision_detail_level_options,
                self.profile.vision_detail_level,
            )
        thinking_combo = self.details.param_widgets.get('thinking_level')
        if isinstance(thinking_combo, ParamComboBox):
            self._syncComboBox(thinking_combo, self.profile.thinking_level_options, self.profile.thinking_level)
        for key in ('prompt', 'vision_prompt', 'image_prompt'):
            editor = self.details.param_widgets.get(key)
            if isinstance(editor, QPlainTextEdit):
                editor.blockSignals(True)
                editor.setPlainText(str(getattr(self.profile, key) or ''))
                editor.blockSignals(False)
        self.refreshTextBadge()
        self.refreshVisionBadge()
        self.refreshImageBadge()
        self.refreshConditionalVisibility()

    def _syncComboBox(self, combo: ParamComboBox, options, value: str):
        combo.blockSignals(True)
        option_texts = [str(option) for option in options if str(option)]
        current_options = [combo.itemText(i) for i in range(combo.count())]
        if current_options != option_texts:
            combo.clear()
            combo.addItems(option_texts)
        value = str(value or '')
        if value and combo.findText(value) < 0:
            combo.addItem(value)
        combo.setCurrentText(value)
        combo.blockSignals(False)

    def _detail_params(self, param_defs):
        params = {}
        for key, widget_type in param_defs:
            value = getattr(self.profile, key)
            display_name = self.profile_param_display_names.get(key, key)
            description = self.profile_param_descriptions.get(key, '')
            if key == 'thinking_level':
                options = self.profile.thinking_level_options
            elif key == 'vision_detail_level':
                options = self.profile.vision_detail_level_options
            else:
                options = None
            if widget_type == 'selector':
                params[key] = {
                    'type': 'selector',
                    'options': options,
                    'value': value,
                    'display_name': display_name,
                    'description': description,
                }
            elif widget_type == 'checkbox':
                params[key] = {
                    'type': 'checkbox',
                    'value': bool(value),
                    'display_name': display_name,
                    'description': description,
                }
            elif widget_type == 'editor':
                params[key] = {
                    'type': 'editor',
                    'value': str(value or ''),
                    'display_name': display_name,
                    'description': description,
                    'label_above': True,
                }
            else:
                params[key] = {
                    'type': 'line_editor',
                    'value': value,
                    'display_name': display_name,
                    'description': description,
                }
        return params

    def _sync_minimum_width_with_content(self):
        margins = self.layout().contentsMargins()
        details_width = self.details.sizeHint().width()
        summary_width = self.summary_widget.sizeHint().width()
        title_width = max(
            self.fontMetrics().boundingRect(self.title() or '').width() + 14,
            self.name_edit.sizeHint().width(),
        )
        title_width += self.more_btn.width() + self.delete_btn.width() + 42
        self.setMinimumWidth(max(details_width, summary_width, title_width) + margins.left() + margins.right())

    def _summary_grid_widgets(self):
        visible_widgets = [
            self.model_summary_widget,
            self.vision_model_summary_widget,
            self.image_model_summary_widget,
        ]
        if not self.api_summary_widget.isHidden():
            visible_widgets.insert(0, self.api_summary_widget)
        return visible_widgets

    def _setSummaryColumnWidth(self, column_widget: QWidget, label_row: QHBoxLayout, selector: QWidget):
        selector_width = max(selector.minimumWidth(), selector.width())
        column_widget.setMinimumWidth(max(
            label_row.sizeHint().width(),
            selector_width,
        ))

    def _sync_summary_grid(self):
        while self.summary_layout.count():
            self.summary_layout.takeAt(0)
        visible_widgets = self._summary_grid_widgets()
        self.summary_widget.ensurePolished()
        for widget in visible_widgets:
            widget.ensurePolished()
        for index, widget in enumerate(visible_widgets):
            row, column = divmod(index, 2)
            alignment = Qt.AlignmentFlag.AlignRight if column else Qt.AlignmentFlag.AlignLeft
            alignment |= Qt.AlignmentFlag.AlignTop
            self.summary_layout.addWidget(widget, row, column, alignment)
        row_widths = []
        for row_start in range(0, len(visible_widgets), 2):
            row_widgets = visible_widgets[row_start:row_start + 2]
            row_width = sum(widget.sizeHint().width() for widget in row_widgets)
            if len(row_widgets) > 1:
                row_width += self.summary_layout.horizontalSpacing() * (len(row_widgets) - 1)
            row_widths.append(row_width)
        self.summary_widget.setMinimumWidth(max(row_widths, default=0))
        self.summary_widget.setMinimumHeight(0)
        self.summary_widget.setMaximumHeight(16777215)
        self.summary_layout.invalidate()
        self.summary_widget.setMinimumHeight(self.summary_layout.sizeHint().height())
        self.summary_widget.updateGeometry()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._selection_border_colors:
            return

        radius = 6
        style_option = QStyleOptionGroupBox()
        self.initStyleOption(style_option)
        frame_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_GroupBox,
            style_option,
            QStyle.SubControl.SC_GroupBoxFrame,
            self,
        )
        rect = QRectF(frame_rect).adjusted(0.5, 0.5, -0.5, -0.5)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen()
        pen.setWidthF(1.0)
        if len(self._selection_border_colors) == 1:
            pen.setColor(QColor(self._selection_border_colors[0]))
        else:
            gradient = QLinearGradient(rect.left(), 0, rect.right(), 0)
            last_index = len(self._selection_border_colors) - 1
            for index, color in enumerate(self._selection_border_colors):
                gradient.setColorAt(index / last_index, QColor(color))
            pen.setBrush(gradient)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        path.moveTo(rect.left() + radius, rect.top())
        if self.title():
            title_rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox,
                style_option,
                QStyle.SubControl.SC_GroupBoxLabel,
                self,
            )
            title_left = max(rect.left() + radius, title_rect.left() - 2)
            title_right = min(rect.right() - radius, title_rect.right() + 2)
            path.lineTo(title_left, rect.top())
            path.moveTo(title_right, rect.top())
        path.lineTo(rect.right() - radius, rect.top())
        path.arcTo(rect.right() - 2 * radius, rect.top(), 2 * radius, 2 * radius, 90, -90)
        path.lineTo(rect.right(), rect.bottom() - radius)
        path.arcTo(rect.right() - 2 * radius, rect.bottom() - 2 * radius, 2 * radius, 2 * radius, 0, -90)
        path.lineTo(rect.left() + radius, rect.bottom())
        path.arcTo(rect.left(), rect.bottom() - 2 * radius, 2 * radius, 2 * radius, 270, -90)
        path.lineTo(rect.left(), rect.top() + radius)
        path.arcTo(rect.left(), rect.top(), 2 * radius, 2 * radius, 180, -90)
        painter.drawPath(path)
        painter.end()

    def refreshSelectionBorder(self):
        self._selection_border_colors = [
            color
            for selected_id, color in (
                (
                    pcfg.module.translator_llm_id
                    if pcfg.module.translator == LLM_TRANSLATOR_KEY
                    else '',
                    LLM_MODALITY_TEXT_COLOR,
                ),
                (
                    pcfg.module.ocr_llm_id
                    if pcfg.module.ocr == LLM_OCR_KEY
                    else '',
                    LLM_MODALITY_VISION_COLOR,
                ),
                (
                    pcfg.module.inpaint_llm_id
                    if pcfg.module.inpainter == LLM_INPAINT_KEY
                    else '',
                    LLM_MODALITY_IMAGE_COLOR,
                ),
            )
            if selected_id == self.profile.id
        ]
        self.update()

    def toggleExpanded(self):
        self.setExpanded(not self.details.isVisible())

    def setExpanded(self, expanded: bool):
        self.details.setVisible(expanded)
        self.more_btn.setToolTip(self.tr('Edit'))
        self.more_btn.setIcon(self.edit_icon_active if expanded else self.edit_icon)

    def expand(self):
        if not self.details.isVisible():
            self.setExpanded(True)

    def collapse(self):
        if self.details.isVisible():
            self.setExpanded(False)

    def focusApiKey(self):
        self.api_key_widget.setFocus()

    def focusModel(self, vision: bool = False):
        if vision:
            self.startVisionModelEdit()
        else:
            self.startModelEdit()

    def focusImageModel(self):
        self.startImageModelEdit()

    def focusDetailLineEditor(self, target: str, placeholder: str = ''):
        widget = self.details.param_widgets.get(target)
        if widget is None:
            return
        if placeholder and hasattr(widget, 'setPlaceholderText'):
            widget.setPlaceholderText(placeholder)
        widget.setFocus()
        if hasattr(widget, 'selectAll'):
            widget.selectAll()

    def startNameEdit(self):
        self._name_editing = True
        self.name_edit.setText(self.profile.name or self.tr('LLM Profile'))
        self.name_edit.resizeToContent()
        self._position_header_controls()
        self.setTitle('')
        self.name_edit.show()
        self.name_edit.raise_()
        self.name_edit.startEdit(select_all=True)

    def on_name_edit_finished(self):
        self.profile.name = self.name_edit.text().strip() or self.tr('LLM Profile')
        self.name_edit.setText(self.profile.name)
        self.setTitle(self.profile.name)
        self.name_edit.resizeToContent()
        self.name_edit.hide()
        self._sync_minimum_width_with_content()
        self._position_header_controls()
        QTimer.singleShot(0, self._finishNameEditCycle)
        self.profile_changed.emit()
        self.profile_selector_changed.emit()

    def _finishNameEditCycle(self):
        self._name_editing = False

    def setActionButtonsVisible(self, visible: bool):
        self._action_buttons_visible = visible
        self.more_btn.setVisible(visible)
        self.delete_btn.setVisible(visible)
        self.add_model_btn.setVisible(visible and bool(self.profile.support_text))
        self.remove_model_btn.setVisible(visible and bool(self.profile.support_text))
        self.add_vision_model_btn.setVisible(visible and bool(self.profile.support_vision))
        self.remove_vision_model_btn.setVisible(visible and bool(self.profile.support_vision))
        self.add_image_model_btn.setVisible(visible and bool(self.profile.support_image))
        self.remove_image_model_btn.setVisible(visible and bool(self.profile.support_image))

    def _position_header_controls(self):
        border_y = 9
        title_y = border_y - self.name_edit.height() // 2
        self.name_edit.move(18, title_y)
        button_y = 14
        spacing = 6
        delete_x = max(18, self.width() - 18 - self.delete_btn.width())
        more_x = max(18, delete_x - spacing - self.more_btn.width())
        self.more_btn.move(more_x, button_y)
        self.delete_btn.move(delete_x, button_y + 2)
        if self.name_edit.isVisible():
            self.name_edit.raise_()
        self.more_btn.raise_()
        self.delete_btn.raise_()

    def resizeEvent(self, event):
        result = super().resizeEvent(event)
        self._position_header_controls()
        return result

    def enterEvent(self, event):
        self.setActionButtonsVisible(True)
        return super().enterEvent(event)

    def leaveEvent(self, event):
        self.setActionButtonsVisible(False)
        if not self.details.isVisible():
            self.more_btn.setIcon(self.edit_icon)
        return super().leaveEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        edit_action = QAction(self.tr('Edit name'), menu)
        delete_action = QAction(self.tr('Delete'), menu)
        copy_action = QAction(self.tr('Copy'), menu)
        copy_json_action = QAction(self.tr('Copy Profile as JSON'), menu)
        more_action = QAction(self.tr('Edit'), menu)
        set_translator_action = QAction(self.tr('Set for Translator'), menu)
        set_ocr_action = QAction(self.tr('Set for OCR'), menu)
        set_inpainter_action = QAction(self.tr('Set for Inpainter'), menu)
        set_translator_action.setEnabled(bool(self.profile.support_text))
        set_ocr_action.setEnabled(bool(self.profile.support_vision))
        set_inpainter_action.setEnabled(bool(self.profile.support_image))
        menu.addAction(edit_action)
        menu.addAction(delete_action)
        menu.addAction(copy_action)
        menu.addAction(copy_json_action)
        menu.addSeparator()
        menu.addAction(set_translator_action)
        menu.addAction(set_ocr_action)
        menu.addAction(set_inpainter_action)
        menu.addSeparator()
        menu.addAction(more_action)
        action = menu.exec(event.globalPos()) if hasattr(menu, 'exec') else menu.exec_(event.globalPos())
        if action == edit_action:
            self.startNameEdit()
        elif action == delete_action:
            self.delete_requested.emit(self.profile.id)
        elif action == copy_action:
            self.copy_requested.emit(self.profile.id)
        elif action == copy_json_action:
            self.copy_json_requested.emit(self.profile.id)
        elif action == set_translator_action:
            self.set_translator_requested.emit(self.profile.id)
        elif action == set_ocr_action:
            self.set_ocr_requested.emit(self.profile.id)
        elif action == set_inpainter_action:
            self.set_inpainter_requested.emit(self.profile.id)
        elif action == more_action:
            self.toggleExpanded()

    def eventFilter(self, obj, event):
        if obj is self.more_btn:
            if event.type() == QEvent.Type.Enter:
                self.more_btn.setIcon(self.edit_icon_active)
            elif event.type() == QEvent.Type.Leave and not self.details.isVisible():
                self.more_btn.setIcon(self.edit_icon)
        return super().eventFilter(obj, event)

    def mouseDoubleClickEvent(self, event):
        pos_y = event.position().y() if hasattr(event, 'position') else event.y()
        if pos_y <= 24:
            self.startNameEdit()
            event.accept()
            return
        return super().mouseDoubleClickEvent(event)

    def on_model_edited(self, param_key, value):
        if self._model_editing:
            return
        self.profile.model = value
        options = self.profile.model_options
        if value and value not in options:
            options.append(value)
        self.profile_changed.emit()
        self.profile_summary_changed.emit()

    def on_vision_model_edited(self, param_key, value):
        if self._vision_model_editing:
            return
        self.profile.vision_model = value
        options = self.profile.vision_model_options
        if value and value not in options:
            options.append(value)
        self.profile_changed.emit()
        self.profile_summary_changed.emit()

    def on_image_model_edited(self, param_key, value):
        if self._image_model_editing:
            return
        self.profile.image_model = value
        options = self.profile.image_model_options
        if value and value not in options:
            options.append(value)
        self.profile_changed.emit()
        self.profile_summary_changed.emit()

    def startModelEdit(self):
        if self._model_editing:
            return
        self._model_editing = True
        self._previous_model_text = self.model_combo.currentText()
        self.model_combo.setEditable(True)
        editor = self.model_combo.lineEdit()
        if editor is None:
            self._model_editing = False
            return
        editor.setObjectName('LLMProfileModelEditor')
        editor.setPlaceholderText(self.tr('Model name'))
        try:
            editor.editingFinished.disconnect(self.finishModelEdit)
        except Exception:
            pass
        editor.editingFinished.connect(self.finishModelEdit)
        self.model_combo.setEditText('')
        editor.setFocus()
        editor.selectAll()

    def finishModelEdit(self):
        if not self._model_editing:
            return
        editor = self.model_combo.lineEdit()
        text = editor.text().strip() if editor is not None else ''
        self._model_editing = False
        self.model_combo.setEditable(False)
        if not text:
            self._setModelText(self._previous_model_text, emit_changed=False)
            return
        options = self.profile.model_options
        if text not in options:
            options.append(text)
            self.model_combo.blockSignals(True)
            self.model_combo.addItem(text)
            self.model_combo.blockSignals(False)
        self._setModelText(text, emit_changed=True)

    def deleteCurrentModel(self):
        if self._model_editing:
            self.finishModelEdit()
        current = self.model_combo.currentText()
        options = [str(option) for option in self.profile.model_options if str(option)]
        if current not in options:
            return
        removed_idx = options.index(current)
        options.pop(removed_idx)
        self.profile.model_options = options
        next_model = options[min(removed_idx, len(options) - 1)] if options else ''
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(options)
        self.model_combo.blockSignals(False)
        self._setModelText(next_model, emit_changed=True)

    def _setModelText(self, text: str, emit_changed: bool):
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText(text)
        self.model_combo.blockSignals(False)
        self.profile.model = text
        if emit_changed:
            self.profile_changed.emit()
            self.profile_summary_changed.emit()

    def startVisionModelEdit(self):
        if self._vision_model_editing:
            return
        self._vision_model_editing = True
        self._previous_vision_model_text = self.vision_model_combo.currentText()
        self.vision_model_combo.setEditable(True)
        editor = self.vision_model_combo.lineEdit()
        if editor is None:
            self._vision_model_editing = False
            return
        editor.setObjectName('LLMProfileModelEditor')
        editor.setPlaceholderText(self.tr('Vision model name'))
        try:
            editor.editingFinished.disconnect(self.finishVisionModelEdit)
        except Exception:
            pass
        editor.editingFinished.connect(self.finishVisionModelEdit)
        self.vision_model_combo.setEditText('')
        editor.setFocus()
        editor.selectAll()

    def finishVisionModelEdit(self):
        if not self._vision_model_editing:
            return
        editor = self.vision_model_combo.lineEdit()
        text = editor.text().strip() if editor is not None else ''
        self._vision_model_editing = False
        self.vision_model_combo.setEditable(False)
        if not text:
            self._setVisionModelText(self._previous_vision_model_text, emit_changed=False)
            return
        options = self.profile.vision_model_options
        if text not in options:
            options.append(text)
            self.vision_model_combo.blockSignals(True)
            self.vision_model_combo.addItem(text)
            self.vision_model_combo.blockSignals(False)
        self._setVisionModelText(text, emit_changed=True)

    def deleteCurrentVisionModel(self):
        if self._vision_model_editing:
            self.finishVisionModelEdit()
        current = self.vision_model_combo.currentText()
        options = [str(option) for option in self.profile.vision_model_options if str(option)]
        if current not in options:
            return
        removed_idx = options.index(current)
        options.pop(removed_idx)
        self.profile.vision_model_options = options
        next_model = options[min(removed_idx, len(options) - 1)] if options else ''
        self.vision_model_combo.blockSignals(True)
        self.vision_model_combo.clear()
        self.vision_model_combo.addItems(options)
        self.vision_model_combo.blockSignals(False)
        self._setVisionModelText(next_model, emit_changed=True)

    def _setVisionModelText(self, text: str, emit_changed: bool):
        self.vision_model_combo.blockSignals(True)
        self.vision_model_combo.setCurrentText(text)
        self.vision_model_combo.blockSignals(False)
        self.profile.vision_model = text
        if emit_changed:
            self.profile_changed.emit()
            self.profile_summary_changed.emit()

    def _syncVisionModelCombo(self):
        self._syncComboBox(self.vision_model_combo, self.profile.vision_model_options, self.profile.vision_model)

    def startImageModelEdit(self):
        if self._image_model_editing:
            return
        self._image_model_editing = True
        self._previous_image_model_text = self.image_model_combo.currentText()
        self.image_model_combo.setEditable(True)
        editor = self.image_model_combo.lineEdit()
        if editor is None:
            self._image_model_editing = False
            return
        editor.setObjectName('LLMProfileModelEditor')
        editor.setPlaceholderText(self.tr('Image model name'))
        try:
            editor.editingFinished.disconnect(self.finishImageModelEdit)
        except Exception:
            pass
        editor.editingFinished.connect(self.finishImageModelEdit)
        self.image_model_combo.setEditText('')
        editor.setFocus()
        editor.selectAll()

    def finishImageModelEdit(self):
        if not self._image_model_editing:
            return
        editor = self.image_model_combo.lineEdit()
        text = editor.text().strip() if editor is not None else ''
        self._image_model_editing = False
        self.image_model_combo.setEditable(False)
        if not text:
            self._setImageModelText(self._previous_image_model_text, emit_changed=False)
            return
        options = self.profile.image_model_options
        if text not in options:
            options.append(text)
            self.image_model_combo.blockSignals(True)
            self.image_model_combo.addItem(text)
            self.image_model_combo.blockSignals(False)
        self._setImageModelText(text, emit_changed=True)

    def deleteCurrentImageModel(self):
        if self._image_model_editing:
            self.finishImageModelEdit()
        current = self.image_model_combo.currentText()
        options = [str(option) for option in self.profile.image_model_options if str(option)]
        if current not in options:
            return
        removed_idx = options.index(current)
        options.pop(removed_idx)
        self.profile.image_model_options = options
        next_model = options[min(removed_idx, len(options) - 1)] if options else ''
        self.image_model_combo.blockSignals(True)
        self.image_model_combo.clear()
        self.image_model_combo.addItems(options)
        self.image_model_combo.blockSignals(False)
        self._setImageModelText(next_model, emit_changed=True)

    def _setImageModelText(self, text: str, emit_changed: bool):
        self.image_model_combo.blockSignals(True)
        self.image_model_combo.setCurrentText(text)
        self.image_model_combo.blockSignals(False)
        self.profile.image_model = text
        if emit_changed:
            self.profile_changed.emit()
            self.profile_summary_changed.emit()

    def on_api_key_finished(self):
        store_api_key(self.profile, self.api_key_widget.text())
        self.profile_changed.emit()

    def _apiKeyEditorIsEmpty(self) -> bool:
        return not self.api_key_widget.text().strip()

    def on_api_key_text_changed(self, *_):
        is_empty = self._apiKeyEditorIsEmpty()
        if is_empty == self._api_key_editor_is_empty:
            return
        self._api_key_editor_is_empty = is_empty
        self.refreshKeyStatus()

    def on_detail_edited(self, param_key, param_content):
        content = param_content.get('content')
        profile_type = PROFILE_FIELD_TYPES.get(param_key)
        if profile_type is int:
            try:
                content = int(str(content).strip())
            except (TypeError, ValueError):
                return
        elif profile_type is float:
            try:
                content = float(str(content).strip())
            except (TypeError, ValueError):
                return
        setattr(self.profile, param_key, content)
        if param_key == 'require_api_key':
            self.refreshConditionalVisibility()
            self.refreshKeyStatus()
        elif param_key == 'vision_detail_level':
            self.profile_summary_changed.emit()
        self.profile_changed.emit()
        if param_key == 'thinking_level':
            self.profile_summary_changed.emit()

    def toggleVisionSupport(self):
        self.profile.support_vision = not bool(self.profile.support_vision)
        if self.profile.support_vision and not self.profile.vision_model:
            self.profile.vision_model = self.profile.model
        if self.profile.support_vision and self.profile.vision_model:
            options = self.profile.vision_model_options
            if self.profile.vision_model not in options:
                options.insert(0, self.profile.vision_model)
        self._syncVisionModelCombo()
        self.refreshVisionBadge()
        self.refreshConditionalVisibility()
        self.profile_changed.emit()
        self.profile_selector_changed.emit()
        self.profile_summary_changed.emit()

    def toggleTextSupport(self):
        self.profile.support_text = not bool(self.profile.support_text)
        if self.profile.support_text and not self.profile.model:
            options = [str(option) for option in self.profile.model_options if str(option)]
            self.profile.model = options[0] if options else ''
        if self.profile.support_text and self.profile.model:
            options = self.profile.model_options
            if self.profile.model not in options:
                options.insert(0, self.profile.model)
        self._syncComboBox(self.model_combo, self.profile.model_options, self.profile.model)
        self.refreshTextBadge()
        self.refreshConditionalVisibility()
        self.profile_changed.emit()
        self.profile_selector_changed.emit()
        self.profile_summary_changed.emit()

    def toggleImageSupport(self):
        self.profile.support_image = not bool(self.profile.support_image)
        if self.profile.support_image and not self.profile.image_model:
            options = [str(option) for option in self.profile.image_model_options if str(option)]
            self.profile.image_model = options[0] if options else ''
        if self.profile.support_image and self.profile.image_model:
            options = self.profile.image_model_options
            if self.profile.image_model not in options:
                options.insert(0, self.profile.image_model)
        self._syncComboBox(self.image_model_combo, self.profile.image_model_options, self.profile.image_model)
        self.refreshImageBadge()
        self.refreshConditionalVisibility()
        self.profile_changed.emit()
        self.profile_selector_changed.emit()
        self.profile_summary_changed.emit()

    def _setModalityLabelState(self, label: QLabel, active: bool, tooltip: str):
        label.setProperty('modalityActive', active)
        label.setStyleSheet('')
        label.setToolTip(tooltip)
        label.setAccessibleDescription(tooltip)
        label.style().unpolish(label)
        label.style().polish(label)

    def refreshTextBadge(self):
        active = bool(self.profile.support_text)
        icon_path = themed_icon_path('text.svg' if active else 'text_disabled.svg')
        self.text_badge.setIconPath(icon_path)
        self.text_badge.setProperty('capabilityActive', active)
        tooltip = (
            self.tr('Text translation model used by LLMTranslator. Click to disable text translation for this profile.')
            if active
            else self.tr('Text translation model used by LLMTranslator. Click to enable text translation for this profile.')
        )
        self.text_badge.setToolTip(tooltip)
        self.text_badge.setAccessibleDescription(tooltip)
        self._setModalityLabelState(self.model_label, active, tooltip)

    def refreshVisionBadge(self):
        active = bool(self.profile.support_vision)
        icon_path = themed_icon_path('eye.svg' if active else 'eye_disabled.svg')
        self.vision_badge.setIconPath(icon_path)
        self.vision_badge.setProperty('capabilityActive', active)
        tooltip = (
            self.tr('Vision OCR model used by LLMOCR. Click to disable vision OCR for this profile.')
            if active
            else self.tr('Vision OCR model used by LLMOCR. Click to enable vision OCR for this profile.')
        )
        self.vision_badge.setToolTip(tooltip)
        self.vision_badge.setAccessibleDescription(tooltip)
        self._setModalityLabelState(self.vision_model_label, active, tooltip)

    def refreshImageBadge(self):
        active = bool(self.profile.support_image)
        icon_path = themed_icon_path('image.svg' if active else 'image_disabled.svg')
        self.image_badge.setIconPath(icon_path)
        self.image_badge.setProperty('capabilityActive', active)
        tooltip = (
            self.tr('Image cleanup model used by LLMInpaint. Click to disable image cleanup for this profile.')
            if active
            else self.tr('Image cleanup model used by LLMInpaint. Click to enable image cleanup for this profile.')
        )
        self.image_badge.setToolTip(tooltip)
        self.image_badge.setAccessibleDescription(tooltip)
        self._setModalityLabelState(self.image_model_label, active, tooltip)

    def refreshConditionalVisibility(self):
        require_key = bool(self.profile.require_api_key)
        support_text = bool(self.profile.support_text)
        support_vision = bool(self.profile.support_vision)
        support_image = bool(self.profile.support_image)
        self.api_summary_widget.setVisible(require_key)
        self.model_summary_widget.setVisible(True)
        self.vision_model_summary_widget.setVisible(True)
        self.image_model_summary_widget.setVisible(True)
        self.model_combo.setVisible(support_text)
        self.vision_model_combo.setVisible(support_vision)
        self.image_model_combo.setVisible(support_image)
        self.setActionButtonsVisible(self._action_buttons_visible)
        self.details.setParamVisible('low_vram_mode', not require_key)
        self.details.setSectionVisible('text', support_text)
        self.details.setSectionVisible('vision', support_vision)
        self.details.setSectionVisible('image', support_image)
        self.refreshKeyStatus()
        self._sync_summary_grid()
        self._sync_minimum_width_with_content()

    def refreshKeyStatus(self):
        require_key = bool(self.profile.require_api_key)
        self.key_status_icon.setVisible(require_key)
        if not require_key:
            return
        has_key = not self._apiKeyEditorIsEmpty()
        if has_key:
            self.key_status_icon.setIconFile('llm_key_ok.svg')
            self.key_status_icon.setProperty('status', 'ok')
            self.key_status_icon.setToolTip(self.tr('Required API key is configured.'))
        else:
            self.key_status_icon.setIconFile('llm_key_missing.svg')
            self.key_status_icon.setProperty('status', 'missing')
            self.key_status_icon.setToolTip(self.tr('Required API key is missing.'))
        self._position_header_controls()


class LLMProfilesWidget(QWidget):
    """Config-panel editor for all LLM profiles.

    Example:
        >>> LLMProfilesWidget.__name__
        'LLMProfilesWidget'
    """

    # Raised when the profile list or display names changed and selector UIs
    # need to rebuild their profile entries.
    profile_ui_updated = Signal()
    # Raised when the selected profile's bottom-bar summary can change, such
    # as model or thinking-level edits, without rebuilding selector entries.
    profile_summary_changed = Signal()
    set_translator_requested = Signal(str)
    set_ocr_requested = Signal(str)
    set_inpainter_requested = Signal(str)

    def __init__(self, scrollWidget: QWidget = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scrollWidget = scrollWidget
        self.rows = {}
        self._selected_profile_ids = self._currentSelectedProfileIds()
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(14)
        self.actions_layout = QHBoxLayout()
        self.actions_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.new_btn = QToolButton(self)
        self.new_btn.setObjectName('LLMProfileNewButton')
        self.new_btn.setText(self.tr('New'))
        self.new_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.new_btn.setFixedHeight(24)
        new_menu = QMenu(self.new_btn)
        new_empty_action = new_menu.addAction(self.tr('New Empty Profile'))
        import_action = new_menu.addAction(self.tr('Import from Clipboard'))
        new_empty_action.triggered.connect(self.newProfile)
        import_action.triggered.connect(self.importProfiles)
        self.new_btn.setMenu(new_menu)
        self.restore_btn = NoBorderPushBtn(self.tr('Restore Built-ins...'), self)
        self.restore_btn.setObjectName('LLMProfileRestoreButton')
        self.restore_btn.setFixedHeight(24)
        self.filter_edit = QLineEdit(self)
        self.filter_edit.setObjectName('LLMProfileFilterEdit')
        self.filter_edit.setFixedHeight(24)
        self.filter_edit.setFixedWidth(size2width('short'))
        self.filter_edit.setPlaceholderText(self.tr('Filter profiles'))
        self.filter_edit.setToolTip(self.tr('Filter displayed profiles by name, model, or base URL.'))
        self.actions_layout.addWidget(self.new_btn)
        self.actions_layout.addWidget(self.restore_btn)
        self.actions_layout.addStretch(-1)
        self.actions_layout.addWidget(self.filter_edit)
        self.layout.addLayout(self.actions_layout)
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.rows_layout.setSpacing(12)
        self.layout.addLayout(self.rows_layout)
        self.restore_btn.clicked.connect(self.restoreBuiltins)
        self.filter_edit.textChanged.connect(self.applyFilter)
        self.rebuild()

    def clearRows(self):
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.rows.clear()

    def addProfileRow(self, profile: LLMProfile):
        row = ProfileCardWidget(profile, scrollWidget=self.scrollWidget)
        row.profile_changed.connect(self.onProfileChanged)
        row.profile_selector_changed.connect(self.onProfileSelectorChanged)
        row.profile_summary_changed.connect(self.onProfileSummaryChanged)
        row.copy_requested.connect(self.copyProfile)
        row.copy_json_requested.connect(self.copyProfileAsJson)
        row.delete_requested.connect(self.deleteProfile)
        row.set_translator_requested.connect(self.set_translator_requested.emit)
        row.set_ocr_requested.connect(self.set_ocr_requested.emit)
        row.set_inpainter_requested.connect(self.set_inpainter_requested.emit)
        self.rows_layout.addWidget(row)
        self.rows[profile.id] = row
        return row

    def rebuild(self):
        self.clearRows()
        for profile in pcfg.module.llm_profiles:
            self.addProfileRow(profile)
        self.applyFilter()
        self._selected_profile_ids = self._currentSelectedProfileIds()

    def onProfileChanged(self):
        sender = self.sender()
        if isinstance(sender, ProfileCardWidget):
            self.applyFilterToRow(sender)
        else:
            self.applyFilter()

    def onProfileSelectorChanged(self):
        self.profile_ui_updated.emit()

    def onProfileSummaryChanged(self):
        self.profile_summary_changed.emit()

    def refreshSelectionBorders(self, *profile_ids: str):
        for profile_id in {profile_id for profile_id in profile_ids if profile_id}:
            row = self.rows.get(profile_id)
            if row is not None:
                row.refreshSelectionBorder()

    @staticmethod
    def _currentSelectedProfileIds():
        return {
            'translator': pcfg.module.translator_llm_id,
            'ocr': pcfg.module.ocr_llm_id,
            'inpainter': pcfg.module.inpaint_llm_id,
        }

    def setSelectedProfile(self, role: str, profile_id: str):
        previous_profile_id = self._selected_profile_ids.get(role, '')
        if profile_id == previous_profile_id:
            return
        self._selected_profile_ids[role] = profile_id
        self.refreshSelectionBorders(previous_profile_id, profile_id)

    def filterQuery(self):
        return self.filter_edit.text().strip().lower() if hasattr(self, 'filter_edit') else ''

    def applyFilterToRow(self, row: ProfileCardWidget, query: str = None):
        query = self.filterQuery() if query is None else query
        profile = row.profile
        haystack = ' '.join((
            profile.name,
            profile.model,
            profile.vision_model,
            profile.image_model,
            profile.base_url,
            profile.image_base_url,
            profile.id,
        )).lower()
        row.setVisible(not query or query in haystack)

    def applyFilter(self, *args):
        query = self.filter_edit.text().strip().lower() if hasattr(self, 'filter_edit') else ''
        for row in self.rows.values():
            self.applyFilterToRow(row, query)

    def syncProfile(self, profile_id: str):
        row = self.rows.get(profile_id)
        if row is None:
            return
        row.syncFromProfile()
        self.applyFilterToRow(row)

    def collapseProfiles(self):
        for row in self.rows.values():
            row.collapse()

    def _new_custom_profile_id(self):
        used_ids = {profile.id for profile in pcfg.module.llm_profiles}
        while True:
            profile_id = f"custom-{uuid.uuid4().hex[:10]}"
            if profile_id not in used_ids:
                return profile_id

    def newProfile(self):
        self.filter_edit.clear()
        profile = LLMProfile()
        profile.id = self._new_custom_profile_id()
        profile.name = self.tr('New Profile')
        profile.built_in = False
        pcfg.module.llm_profiles.append(profile)
        row = self.addProfileRow(profile)
        self.applyFilterToRow(row)
        self.focusProfileName(profile.id, deferred=True)
        self.profile_ui_updated.emit()

    def copyProfileAsJson(self, profile_id: str):
        profile = profile_by_id(pcfg.module.llm_profiles, profile_id)
        if profile is None:
            return
        payload = json.dumps(profile_to_export_dict(profile), ensure_ascii=False, indent=2)
        QApplication.clipboard().setText(payload)

    def importProfiles(self):
        imported = profiles_from_json(QApplication.clipboard().text())
        if not imported:
            QMessageBox.warning(
                self,
                self.tr('Import Profiles'),
                self.tr('The clipboard does not contain valid LLM profile JSON.'),
            )
            return
        self.filter_edit.clear()
        imported_rows = []
        for profile in imported:
            profile.id = self._new_custom_profile_id()
            profile.built_in = False
            pcfg.module.llm_profiles.append(profile)
            row = self.addProfileRow(profile)
            self.applyFilterToRow(row)
            imported_rows.append(row)
        self.profile_ui_updated.emit()
        QTimer.singleShot(0, lambda: self.ensureRowVisible(imported_rows[-1]))

    def copyProfile(self, profile_id: str):
        profile = profile_by_id(pcfg.module.llm_profiles, profile_id)
        if profile is None:
            return
        self.filter_edit.clear()
        copied = copy_profile(copy.deepcopy(profile))
        copied.id = self._new_custom_profile_id()
        pcfg.module.llm_profiles.append(copied)
        row = self.addProfileRow(copied)
        self.applyFilterToRow(row)
        self.focusProfileName(copied.id, deferred=True)
        self.profile_ui_updated.emit()

    def deleteProfile(self, profile_id: str):
        if len(pcfg.module.llm_profiles) <= 1:
            return
        pcfg.module.llm_profiles = [p for p in pcfg.module.llm_profiles if p.id != profile_id]
        if pcfg.module.translator_llm_id == profile_id:
            pcfg.module.translator_llm_id = pcfg.module.llm_profiles[0].id
        if pcfg.module.ocr_llm_id == profile_id:
            pcfg.module.ocr_llm_id = pcfg.module.llm_profiles[0].id
        if pcfg.module.inpaint_llm_id == profile_id:
            pcfg.module.inpaint_llm_id = pcfg.module.llm_profiles[0].id
        row = self.rows.pop(profile_id, None)
        if row is not None:
            row.collapse()
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.setSelectedProfile('translator', pcfg.module.translator_llm_id)
        self.setSelectedProfile('ocr', pcfg.module.ocr_llm_id)
        self.setSelectedProfile('inpainter', pcfg.module.inpaint_llm_id)
        self.profile_ui_updated.emit()

    def restoreBuiltins(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning if hasattr(QMessageBox, 'Icon') else QMessageBox.Warning)
        msg.setWindowTitle(self.tr('Restore Built-in Profiles'))
        msg.setText(self.tr('Restore built-in LLM profiles to their default values?'))
        msg.setInformativeText(self.tr(
            'This may overwrite current built-in profile settings such as base URL, model, and prompts. '
            'Filled API keys will be kept.'
        ))
        restore_btn = msg.addButton(self.tr('Restore'), QMessageBox.ButtonRole.AcceptRole if hasattr(QMessageBox, 'ButtonRole') else QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel if hasattr(QMessageBox, 'StandardButton') else QMessageBox.Cancel)
        msg.exec()
        if msg.clickedButton() != restore_btn:
            return
        pcfg.module.llm_profiles = restore_builtin_profiles(pcfg.module.llm_profiles)
        if not profile_by_id(pcfg.module.llm_profiles, pcfg.module.translator_llm_id):
            pcfg.module.translator_llm_id = pcfg.module.llm_profiles[0].id
        if not profile_by_id(pcfg.module.llm_profiles, pcfg.module.ocr_llm_id):
            pcfg.module.ocr_llm_id = pcfg.module.llm_profiles[0].id
        if not profile_by_id(pcfg.module.llm_profiles, pcfg.module.inpaint_llm_id):
            pcfg.module.inpaint_llm_id = pcfg.module.llm_profiles[0].id
        self.rebuild()
        self.profile_ui_updated.emit()

    def focusProfileControl(
        self,
        profile_id: str,
        target: str = 'api_key',
        deferred: bool = False,
        expand_details: bool = True,
    ):
        row = self.rows.get(profile_id)
        if row is None:
            return
        if deferred:
            # A zero-timeout timer runs on the next Qt event-loop turn, after
            # rebuild/show/layout work has settled enough for focus and scroll.
            QTimer.singleShot(
                0,
                lambda profile_id=profile_id, target=target, expand_details=expand_details: self.focusProfileControl(
                    profile_id,
                    target=target,
                    expand_details=expand_details,
                ),
            )
            return
        if not row.isVisible():
            self.filter_edit.clear()
        if expand_details:
            row.expand()
        if target == 'model':
            row.focusModel(vision=False)
            self.ensureWidgetVisible(row.model_combo)
        elif target == 'vision_model':
            row.focusModel(vision=True)
            self.ensureWidgetVisible(row.vision_model_combo)
        elif target == 'image_model':
            row.focusImageModel()
            self.ensureWidgetVisible(row.image_model_combo)
        elif target == 'image_base_url':
            row.focusDetailLineEditor(target, self.tr('Image Base URL'))
            widget = row.details.param_widgets.get(target)
            if widget is not None:
                self.ensureWidgetVisible(widget)
        else:
            row.focusApiKey()
            self.ensureWidgetVisible(row.api_key_widget.editor)

    def focusProfileApiKey(self, profile_id: str, deferred: bool = False, expand_details: bool = True):
        self.focusProfileControl(
            profile_id,
            target='api_key',
            deferred=deferred,
            expand_details=expand_details,
        )

    def focusProfileName(self, profile_id: str, deferred: bool = False):
        row = self.rows.get(profile_id)
        if row is None:
            return
        if deferred:
            QTimer.singleShot(0, lambda profile_id=profile_id: self.focusProfileName(profile_id))
            return
        row.startNameEdit()
        self.ensureRowVisible(row)

    def ensureRowVisible(self, row: QWidget):
        self.ensureWidgetVisible(row)

    def ensureWidgetVisible(self, widget: QWidget):
        scroll_area = self.parentWidget()
        while scroll_area is not None:
            if isinstance(scroll_area, QScrollArea):
                scroll_area.ensureWidgetVisible(widget, 0, 16)
                QTimer.singleShot(0, lambda scroll_area=scroll_area, widget=widget: scroll_area.ensureWidgetVisible(widget, 0, 16))
                return
            scroll_area = scroll_area.parentWidget()

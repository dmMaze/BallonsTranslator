import copy
import uuid

from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QPlainTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QGroupBox,
    QMenu,
    QScrollArea,
)
from qtpy.QtCore import QEvent, QRectF, QTimer, Qt, Signal
from qtpy.QtGui import QFont, QIcon, QPainter
from qtpy.QtSvg import QSvgRenderer

try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction

from .custom_widget import ParamComboBox, NoBorderPushBtn, ScrollBar
from .misc import themed_icon_path
from .module_parse_widgets import ParamWidget, SecretParamWidget
from ballontranslator.utils.shared import size2width
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    copy_profile,
    default_profile,
    dedupe_profiles,
    profile_by_id,
    restore_builtin_profiles,
    resolve_api_key,
    store_api_key,
)


PROFILE_PARAM_DEFS = [
    ('require api key', 'checkbox'),
    ('base url', 'line_editor'),
    ('thinking level', 'selector'),
    ('system prompt', 'editor'),
    ('chat sample', 'editor'),
    ('invalid repeat count', 'line_editor'),
    ('max tokens', 'line_editor'),
    ('temperature', 'line_editor'),
    ('top p', 'line_editor'),
    ('frequency penalty', 'line_editor'),
    ('presence penalty', 'line_editor'),
    ('low vram mode', 'checkbox'),
]


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


class SvgStatusIcon(QLabel):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._renderer = None
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def setIconFile(self, filename: str):
        if self._renderer is not None:
            self._renderer.deleteLater()
        self._renderer = QSvgRenderer(themed_icon_path(filename), self)
        self.update()

    def paintEvent(self, event):
        if self._renderer is None or not self._renderer.isValid():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._renderer.render(painter, QRectF(0, 0, self.width(), self.height()))
        painter.end()


class ProfileCardWidget(QGroupBox):
    """Compact card editor for a single LLM profile.

    Example:
        >>> ProfileCardWidget.__name__
        'ProfileCardWidget'
    """

    profile_changed = Signal()
    copy_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, profile: dict, scrollWidget: QWidget = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('LLMProfileCard')
        self.profile = profile
        self.setTitle(profile.get('name', ''))
        self.setToolTip(self.tr('Double click the name to edit. Right click for profile actions.'))
        self.scrollWidget = scrollWidget
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self._name_editing = False
        self._model_editing = False
        self._previous_model_text = ''
        self._app_filter_installed = False
        self.profile_param_display_names = {
            'base url': self.tr('Base URL'),
            'require api key': self.tr('Require API Key'),
            'thinking level': self.tr('Thinking Level'),
            'system prompt': self.tr('System Prompt'),
            'chat sample': self.tr('Chat Sample'),
            'invalid repeat count': self.tr('Invalid Repeat Count'),
            'max tokens': self.tr('Max Tokens'),
            'temperature': self.tr('Temperature'),
            'top p': self.tr('Top P'),
            'frequency penalty': self.tr('Frequency Penalty'),
            'presence penalty': self.tr('Presence Penalty'),
            'low vram mode': self.tr('Low VRAM Mode'),
        }
        self.profile_param_descriptions = {
            'base url': self.tr('OpenAI-compatible API base URL.'),
            'require api key': self.tr('Require API key before running translation.'),
            'thinking level': self.tr('Reasoning effort sent only when it is not None.'),
            'system prompt': self.tr('System prompt used for structured JSON translation.'),
            'chat sample': self.tr('Few-shot samples preserved from migrated profiles.'),
            'invalid repeat count': self.tr('Retries when response count does not match source count.'),
            'max tokens': self.tr('Maximum response tokens.'),
            'temperature': self.tr('Sampling temperature.'),
            'top p': self.tr('Top-p sampling.'),
            'frequency penalty': self.tr('OpenAI frequency penalty.'),
            'presence penalty': self.tr('OpenAI presence penalty.'),
            'low vram mode': self.tr('Preserved compatibility flag for local profiles.'),
        }
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 24, 16, 14)
        layout.setSpacing(8)

        self.name_edit = ProfileNameEdit(profile.get('name', ''), self)
        self.name_edit.edit_requested.connect(self.startNameEdit)
        self.name_edit.edit_finished.connect(self.on_name_edit_finished)
        self.name_edit.hide()

        self.key_status_icon = SvgStatusIcon(self)
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
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.profile['id']))
        self.delete_btn.setFixedSize(18, 18)

        summary = QHBoxLayout()
        summary.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        summary.setSpacing(12)

        self.api_summary_widget = QWidget(self)
        self.api_summary_widget.setObjectName('LLMProfileSummaryColumn')
        self.api_summary_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        center_column = QVBoxLayout(self.api_summary_widget)
        center_column.setContentsMargins(0, 0, 0, 0)
        center_column.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        center_column.setSpacing(6)
        self.api_label = QLabel(self.tr('API Key'), self)
        self.api_label.setObjectName('LLMProfileFieldLabel')
        self.api_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        api_label_row = QHBoxLayout()
        api_label_row.setContentsMargins(0, 0, 0, 0)
        api_label_row.setSpacing(6)
        api_label_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        api_label_row.addWidget(self.api_label, 0, Qt.AlignmentFlag.AlignLeft)
        api_label_row.addWidget(self.key_status_icon, 0, Qt.AlignmentFlag.AlignLeft)
        api_label_row.addStretch(1)
        self.api_key_widget = SecretParamWidget('api key', size='short')
        self.api_key_widget.editor.setObjectName('LLMProfileApiKeyEditor')
        self.api_key_widget.setText(resolve_api_key(profile))

        center_column.addLayout(api_label_row)
        center_column.addWidget(self.api_key_widget)

        self.model_summary_widget = QWidget(self)
        self.model_summary_widget.setObjectName('LLMProfileSummaryColumn')
        self.model_summary_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        right_column = QVBoxLayout(self.model_summary_widget)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        right_column.setSpacing(6)
        self.model_label = QLabel(self.tr('Model'), self)
        self.model_label.setObjectName('LLMProfileFieldLabel')
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
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
        model_label_row = QHBoxLayout()
        model_label_row.setContentsMargins(0, 0, 0, 0)
        model_label_row.setSpacing(4)
        model_label_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        model_label_row.addWidget(self.model_label, 0, Qt.AlignmentFlag.AlignLeft)
        model_label_row.addWidget(self.add_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        model_label_row.addWidget(self.remove_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        model_label_row.addStretch(1)
        self.model_combo = ParamComboBox('model', profile.get('model options', []), size=size2width('short'), scrollWidget=scrollWidget)
        self.model_combo.setObjectName('LLMProfileModelCombo')
        self.model_combo.setEditable(False)
        self.model_combo.setCurrentText(profile.get('model', ''))
        right_column.addLayout(model_label_row)
        right_column.addWidget(self.model_combo, 0, Qt.AlignmentFlag.AlignLeft)

        self.summary_spacer = QWidget(self)
        self.summary_spacer.setObjectName('LLMProfileSummarySpacer')
        self.summary_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        summary.addWidget(self.api_summary_widget, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        summary.addWidget(self.summary_spacer)
        summary.addWidget(self.model_summary_widget, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addLayout(summary)

        self.details = ParamWidget(self._detail_params(), scrollWidget=scrollWidget)
        self.details.setObjectName('LLMProfileDetails')
        self.details.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._install_detail_editor_scrollbars()
        layout.addWidget(self.details)
        self._sync_minimum_width_with_content()
        self.details.setVisible(False)
        self.setActionButtonsVisible(False)

        self.model_combo.paramwidget_edited.connect(self.on_model_edited)
        self.api_key_widget.editor.editingFinished.connect(self.on_api_key_finished)
        self.details.paramwidget_edited.connect(self.on_detail_edited)
        self._position_header_controls()
        self.refreshConditionalVisibility()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._app_filter_installed = True
        self.destroyed.connect(self._remove_app_event_filter)

    def _install_detail_editor_scrollbars(self):
        for editor in self.details.findChildren(QPlainTextEdit):
            editor.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, editor, fadeout=False, hover_style=True)
            editor.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, editor, fadeout=False, hover_style=True)

    def _detail_params(self):
        params = {}
        for key, widget_type in PROFILE_PARAM_DEFS:
            value = self.profile.get(key)
            display_name = self.profile_param_display_names.get(key, key)
            description = self.profile_param_descriptions.get(key, '')
            if key == 'thinking level':
                options = self.profile.get('thinking level options', [])
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
        summary_width = self.layout().itemAt(0).sizeHint().width()
        title_width = max(
            self.fontMetrics().boundingRect(self.title() or '').width() + 14,
            self.name_edit.sizeHint().width(),
        ) + self.more_btn.width() + self.delete_btn.width() + 64
        self.setMinimumWidth(max(details_width, summary_width, title_width) + margins.left() + margins.right())

    def toggleExpanded(self):
        self.setExpanded(not self.details.isVisible())

    def setExpanded(self, expanded: bool):
        self.details.setVisible(expanded)
        self.more_btn.setToolTip(self.tr('Edit'))
        self.more_btn.setIcon(self.edit_icon_active if expanded else self.edit_icon)
        if expanded:
            self.refreshConditionalVisibility()
        self._sync_minimum_width_with_content()

    def expand(self):
        if not self.details.isVisible():
            self.setExpanded(True)

    def collapse(self):
        if self.details.isVisible():
            self.setExpanded(False)

    def focusApiKey(self):
        self.api_key_widget.setFocus()

    def startNameEdit(self):
        self._name_editing = True
        self.name_edit.setText(self.profile.get('name', '') or self.tr('LLM Profile'))
        self.name_edit.resizeToContent()
        self._position_header_controls()
        self.setTitle('')
        self.name_edit.show()
        self.name_edit.raise_()
        self.name_edit.startEdit(select_all=True)

    def on_name_edit_finished(self):
        self.profile['name'] = self.name_edit.text().strip() or self.tr('LLM Profile')
        self.name_edit.setText(self.profile['name'])
        self.setTitle(self.profile['name'])
        self.name_edit.resizeToContent()
        self.name_edit.hide()
        self._sync_minimum_width_with_content()
        self._position_header_controls()
        QTimer.singleShot(0, self._finishNameEditCycle)
        self.profile_changed.emit()

    def _finishNameEditCycle(self):
        self._name_editing = False

    def setActionButtonsVisible(self, visible: bool):
        self.more_btn.setVisible(visible)
        self.delete_btn.setVisible(visible)
        self.add_model_btn.setVisible(visible)
        self.remove_model_btn.setVisible(visible)

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
        self._position_header_controls()
        return super().resizeEvent(event)

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
        more_action = QAction(self.tr('Edit'), menu)
        menu.addAction(edit_action)
        menu.addAction(delete_action)
        menu.addAction(copy_action)
        menu.addSeparator()
        menu.addAction(more_action)
        action = menu.exec(event.globalPos()) if hasattr(menu, 'exec') else menu.exec_(event.globalPos())
        if action == edit_action:
            self.startNameEdit()
        elif action == delete_action:
            self.delete_requested.emit(self.profile['id'])
        elif action == copy_action:
            self.copy_requested.emit(self.profile['id'])
        elif action == more_action:
            self.toggleExpanded()

    def eventFilter(self, obj, event):
        if obj is self.more_btn:
            if event.type() == QEvent.Type.Enter:
                self.more_btn.setIcon(self.edit_icon_active)
            elif event.type() == QEvent.Type.Leave and not self.details.isVisible():
                self.more_btn.setIcon(self.edit_icon)
        elif event.type() == QEvent.Type.MouseButtonPress and self.details.isVisible():
            if not self._name_editing and not self._globalMouseEventInsideCard(event):
                self.collapse()
        return super().eventFilter(obj, event)

    def _globalMouseEventInsideCard(self, event) -> bool:
        if hasattr(event, 'globalPosition'):
            global_pos = event.globalPosition().toPoint()
        elif hasattr(event, 'globalPos'):
            global_pos = event.globalPos()
        else:
            return True
        return self.rect().contains(self.mapFromGlobal(global_pos))

    def _remove_app_event_filter(self, *args):
        if not self._app_filter_installed:
            return
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._app_filter_installed = False

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
        self.profile['model'] = value
        options = self.profile.setdefault('model options', [])
        if value and value not in options:
            options.append(value)
        self.profile_changed.emit()

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
        options = self.profile.setdefault('model options', [])
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
        options = [str(option) for option in self.profile.get('model options', []) if str(option)]
        if current not in options:
            return
        removed_idx = options.index(current)
        options.pop(removed_idx)
        self.profile['model options'] = options
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
        self.profile['model'] = text
        if emit_changed:
            self.profile_changed.emit()

    def on_api_key_finished(self):
        store_api_key(self.profile, self.api_key_widget.text())
        self.refreshKeyStatus()
        self.profile_changed.emit()

    def on_detail_edited(self, param_key, param_content):
        content = param_content.get('content')
        self.profile[param_key] = content
        if param_key == 'require api key':
            self.refreshConditionalVisibility()
            self.refreshKeyStatus()
        self.profile_changed.emit()

    def refreshConditionalVisibility(self):
        require_key = bool(self.profile.get('require api key'))
        self.api_summary_widget.setVisible(require_key)
        self.summary_spacer.setVisible(require_key)
        if hasattr(self.details, 'setParamVisible'):
            self.details.setParamVisible('low vram mode', not require_key)
        self.refreshKeyStatus()

    def refreshKeyStatus(self):
        require_key = bool(self.profile.get('require api key'))
        self.key_status_icon.setVisible(require_key)
        if not require_key:
            return
        has_key = bool(resolve_api_key(self.profile).strip())
        if has_key:
            self.key_status_icon.setIconFile('llm_key_ok.svg')
            self.key_status_icon.setProperty('status', 'ok')
            self.key_status_icon.setToolTip(self.tr('Required API key is configured.'))
        else:
            self.key_status_icon.setIconFile('llm_key_missing.svg')
            self.key_status_icon.setProperty('status', 'missing')
            self.key_status_icon.setToolTip(self.tr('Required API key is missing.'))
        self.key_status_icon.style().unpolish(self.key_status_icon)
        self.key_status_icon.style().polish(self.key_status_icon)
        self._position_header_controls()


class LLMProfilesWidget(QWidget):
    """Config-panel editor for all LLM profiles.

    Example:
        >>> LLMProfilesWidget.__name__
        'LLMProfilesWidget'
    """

    profiles_changed = Signal()

    def __init__(self, scrollWidget: QWidget = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scrollWidget = scrollWidget
        self.rows = {}
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(14)
        self.actions_layout = QHBoxLayout()
        self.actions_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.new_btn = NoBorderPushBtn(self.tr('New'), self)
        self.new_btn.setObjectName('LLMProfileNewButton')
        self.new_btn.setFixedHeight(24)
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
        self.new_btn.clicked.connect(self.newProfile)
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

    def rebuild(self):
        pcfg.module.llm_profiles = dedupe_profiles(pcfg.module.llm_profiles, pcfg.module.llm_profile)
        if not profile_by_id(pcfg.module.llm_profiles, pcfg.module.llm_profile) and pcfg.module.llm_profiles:
            pcfg.module.llm_profile = pcfg.module.llm_profiles[0]['id']
        self.clearRows()
        for profile in pcfg.module.llm_profiles:
            row = ProfileCardWidget(profile, scrollWidget=self.scrollWidget)
            row.profile_changed.connect(self.onProfileChanged)
            row.copy_requested.connect(self.copyProfile)
            row.delete_requested.connect(self.deleteProfile)
            self.rows_layout.addWidget(row)
            self.rows[profile['id']] = row
        self.applyFilter()

    def onProfileChanged(self):
        self.applyFilter()
        self.profiles_changed.emit()

    def applyFilter(self):
        query = self.filter_edit.text().strip().lower() if hasattr(self, 'filter_edit') else ''
        for row in self.rows.values():
            profile = row.profile
            haystack = ' '.join(str(profile.get(key, '')) for key in ('name', 'model', 'base url', 'id')).lower()
            row.setVisible(not query or query in haystack)

    def newProfile(self):
        self.filter_edit.clear()
        profile = default_profile('OpenAI')
        profile['id'] = f"custom-{uuid.uuid4().hex[:10]}"
        profile['name'] = self.tr('New Profile')
        profile['built_in'] = False
        profile['api key'] = ''
        pcfg.module.llm_profiles.append(profile)
        pcfg.module.llm_profile = profile['id']
        self.rebuild()
        self.focusProfileName(profile['id'], deferred=True)
        self.profiles_changed.emit()

    def copyProfile(self, profile_id: str):
        profile = profile_by_id(pcfg.module.llm_profiles, profile_id)
        if profile is None:
            return
        self.filter_edit.clear()
        copied = copy_profile(copy.deepcopy(profile))
        copied['id'] = f"custom-{uuid.uuid4().hex[:10]}"
        pcfg.module.llm_profiles.append(copied)
        pcfg.module.llm_profile = copied['id']
        self.rebuild()
        self.focusProfileName(copied['id'], deferred=True)
        self.profiles_changed.emit()

    def deleteProfile(self, profile_id: str):
        if len(pcfg.module.llm_profiles) <= 1:
            return
        pcfg.module.llm_profiles = [p for p in pcfg.module.llm_profiles if p.get('id') != profile_id]
        if pcfg.module.llm_profile == profile_id:
            pcfg.module.llm_profile = pcfg.module.llm_profiles[0]['id']
        self.rebuild()
        self.profiles_changed.emit()

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
        if not profile_by_id(pcfg.module.llm_profiles, pcfg.module.llm_profile):
            pcfg.module.llm_profile = pcfg.module.llm_profiles[0]['id']
        self.rebuild()
        self.profiles_changed.emit()

    def focusProfileApiKey(self, profile_id: str, deferred: bool = False):
        row = self.rows.get(profile_id)
        if row is None:
            return
        if deferred:
            QTimer.singleShot(0, lambda profile_id=profile_id: self.focusProfileApiKey(profile_id))
            return
        if not row.isVisible():
            self.filter_edit.clear()
        row.expand()
        row.focusApiKey()
        self.ensureWidgetVisible(row.api_key_widget.editor)

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

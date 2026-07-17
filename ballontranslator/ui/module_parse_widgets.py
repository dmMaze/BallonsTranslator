from typing import Callable

from ballontranslator.modules import GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS, GET_VALID_TRANSLATORS, GET_VALID_OCR
from ballontranslator.utils.logger import logger as LOGGER
from .custom_widget import ConfigComboBox, ParamComboBox, NoBorderPushBtn, ParamNameLabel
from ballontranslator.utils.shared import (
    CONFIG_COMBOBOX_HEIGHT,
    CONFIG_COMBOBOX_LONG,
    CONFIG_CONTENT_ROW_SPACING,
    CONFIG_MODULE_PARAM_BODY_MIN_WIDTH,
    size2width,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import LLM_TRANSLATOR_KEY
from .module_param_i18n import (
    tr_module_description,
    tr_param_description,
    tr_param_display_name,
)

from qtpy.QtWidgets import (
    QApplication,
    QPlainTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QLineEdit,
    QGridLayout,
    QPushButton,
    QSizePolicy,
    QLayout,
    QMenu,
)
from qtpy.QtCore import QTimer, Qt, Signal
from qtpy.QtGui import QDoubleValidator, QKeySequence

try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction

LAYOUT_SET_MINIMUM_SIZE = getattr(getattr(QLayout, 'SizeConstraint', QLayout), 'SetMinimumSize')


class ParamCheckGroup(QWidget):

    paramwidget_edited = Signal(str, dict)

    def __init__(self, param_key, check_group: dict, parent=None) -> None:
        super().__init__(parent=parent)
        self.param_key = param_key
        layout = QHBoxLayout(self)
        self.label2widget = {}
        for k, v in check_group.items():
            checker = QCheckBox(text=k, parent=self)
            checker.setObjectName('ConfigCheckBox')
            checker.setChecked(v)
            layout.addWidget(checker)
            self.label2widget[k] = checker
            checker.clicked.connect(self.on_checker_clicked)

    def on_checker_clicked(self):
        new_state_dict = {}
        w = QCheckBox()
        for k, w in self.label2widget.items():
            new_state_dict[k] = w.isChecked()
        self.paramwidget_edited.emit(self.param_key, new_state_dict)


class ParamLineEditor(QLineEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, force_digital, size='short', *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size2width(size))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

        if force_digital:
            validator = QDoubleValidator()
            self.setValidator(validator)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())


class SecretLineEditor(QLineEdit):
    """Password editor with an explicit copy path for hidden text.

    Example:
        >>> SecretLineEditor.__name__
        'SecretLineEditor'
    """

    def copySecretText(self):
        text = self.selectedText() if self.hasSelectedText() else self.text()
        if not text:
            return
        QApplication.clipboard().setText(text)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copySecretText()
            event.accept()
            return
        return super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        copy_action = QAction(self.tr('Copy'), menu)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.setEnabled(bool(self.text()))
        paste_action = QAction(self.tr('Paste'), menu)
        paste_action.setEnabled(not self.isReadOnly() and bool(QApplication.clipboard().text()))
        select_all_action = QAction(self.tr('Select All'), menu)
        select_all_action.setEnabled(bool(self.text()))
        menu.addAction(copy_action)
        menu.addAction(paste_action)
        menu.addSeparator()
        menu.addAction(select_all_action)
        action = menu.exec(event.globalPos()) if hasattr(menu, 'exec') else menu.exec_(event.globalPos())
        if action == copy_action:
            self.copySecretText()
        elif action == paste_action:
            self.paste()
        elif action == select_all_action:
            self.selectAll()


class SecretParamWidget(QWidget):
    """Password-style parameter editor.

    Example:
        >>> SecretParamWidget.__name__
        'SecretParamWidget'
    """

    paramwidget_edited = Signal(str, str)

    def __init__(self, param_key: str, size='short', fixed_size: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setObjectName('SecretParamWidget')
        self.param_key = param_key
        self.editor = SecretLineEditor(self)
        if fixed_size:
            self.editor.setFixedWidth(size2width(size))
            self.setFixedWidth(size2width(size))
        else:
            self.editor.setMinimumWidth(size2width(size))
            self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.setMinimumWidth(size2width(size))
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.editor.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.editor.setEchoMode(QLineEdit.EchoMode.Password)
        self.editor.setToolTip(self.tr(
            'Stored in portable obfuscated form. This hides the key from plain-text scans, '
            'but it is not a secure password vault.'
        ))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor)
        self.editor.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.editor.text())

    def setText(self, text: str):
        self.editor.setText(text)

    def text(self):
        return self.editor.text()

    def setFocus(self, *args, **kwargs):
        return self.editor.setFocus(*args, **kwargs)

class ParamEditor(QPlainTextEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self._auto_max_height = 100

        self._showed = False

        self.setFixedWidth(int(CONFIG_COMBOBOX_LONG))
        self.textChanged.connect(self.on_text_changed)
        self.document().documentLayout().documentSizeChanged.connect(lambda *_: self.adjustSize())

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)

    def text(self):
        return self.toPlainText()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._showed:
            self._showed = True
            QTimer.singleShot(0, self.adjustSize)

    def adjustSize(self):

        # QPlainTextDocumentLayout.documentSize().height() reports a block
        # count here; block bounds provide the actual wrapped visual heights.
        document_layout = self.document().documentLayout()
        block = self.document().begin()
        content_height = 0.0
        while block.isValid():
            content_height += document_layout.blockBoundingRect(block).height()
            if content_height >= self._auto_max_height:
                break
            block = block.next()

        content_height += self.frameWidth() * 2
        height = min(round(content_height), self._auto_max_height)
        if self.height() != height:
            self.setFixedHeight(height)


class ParamCheckerBox(QWidget):
    checker_changed = Signal(bool)
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.checker = QCheckBox()
        self.checker.setObjectName('ConfigCheckBox')
        name_label = ParamNameLabel(param_key)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.checker)
        hlayout.addWidget(name_label)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.checker.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        is_checked = self.checker.isChecked()
        self.checker_changed.emit(is_checked)
        checked = 'true' if is_checked else 'false'
        self.paramwidget_edited.emit(self.param_key, checked)

    def isChecked(self):
        return self.checker.isChecked()


class ParamCheckBox(QCheckBox):
    paramwidget_edited = Signal(str, bool)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('ParamCheckBox')
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def ensure_current_device_option(param_dict: dict):
    not_supported = param_dict.get('__device_not_supported', [])
    current_value = str(param_dict.get('value', 'cpu'))
    options = [str(opt) for opt in param_dict.get('options', [])]
    if current_value not in options and all(device not in current_value for device in not_supported):
        options.append(current_value)
    param_dict['options'] = options
    param_dict['value'] = current_value if current_value in options else 'cpu'


def set_label_tooltip_from_widget(label: QWidget, widget: QWidget):
    """Mirror an editor tooltip onto its visible label.

    Example:
        >>> set_label_tooltip_from_widget.__name__
        'set_label_tooltip_from_widget'
    """
    tooltip = widget.toolTip()
    if tooltip:
        label.setToolTip(tooltip)


class ParamPushButton(QPushButton):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, display_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.setText(display_name)
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.paramwidget_edited.emit(self.param_key, '')


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(
        self,
        params,
        scrollWidget: QWidget = None,
        module_type: str = '',
        module_key: str = '',
        body_min_width: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.module_type = module_type
        self.module_key = module_key
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout = param_layout = QGridLayout()
        self.param_widgets = {}
        self.param_rows = {}
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setVerticalSpacing(CONFIG_CONTENT_ROW_SPACING)
        if body_min_width > 0:
            self.setMinimumWidth(body_min_width)
            param_layout.setColumnStretch(1, 1)
            layout.addLayout(param_layout, 1)
        else:
            param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            layout.addLayout(param_layout)
            layout.addStretch(-1)

        module_description = tr_module_description(params, module_type, module_key)
        if module_description:
            self.setToolTip(module_description)

        for ii, param_key in enumerate(params):
            if param_key == 'description' or param_key.startswith('__'):
                continue
            display_param_name = tr_param_display_name(
                params, param_key, module_type=module_type, module_key=module_key)
            param_description = tr_param_description(
                params, param_key, module_type=module_type, module_key=module_key)

            require_label = True
            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
            param_widget = None
            label_above = False

            if isinstance(params[param_key], bool):
                param_widget = ParamCheckBox(param_key)
                val = params[param_key]
                param_widget.setChecked(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif is_str or is_digital:
                param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                val = params[param_key]
                if is_digital:
                    val = str(val)
                param_widget.setText(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif isinstance(params[param_key], dict):
                param_dict = params[param_key]
                display_param_name = tr_param_display_name(
                    params, param_key, param_dict, module_type, module_key)
                param_description = tr_param_description(
                    params, param_key, param_dict, module_type, module_key)
                value = params[param_key]['value']
                param_widget = None  # Ensure initialization
                param_type = param_dict['type'] if 'type' in param_dict else 'line_editor'
                flush_btn = param_dict.get('flush_btn', False)
                path_selector = param_dict.get('path_selector', False)
                param_size = param_dict.get('size', 'short')
                label_above = param_dict.get('label_above', False)
                if param_key == 'device' and param_type == 'selector':
                    ensure_current_device_option(param_dict)
                    value = param_dict['value']
                if param_type == 'selector':
                    if 'url' in param_key:
                        size = size2width('median')
                    else:
                        size = size2width(param_size)

                    param_widget = ParamComboBox(
                        param_key, param_dict['options'], size=size, scrollWidget=scrollWidget, flush_btn=flush_btn, path_selector=path_selector)

                    param_widget.setEditable(param_dict.get('editable', False))
                    param_widget.setCurrentText(str(value))

                elif param_type == 'editor':
                    param_widget = ParamEditor(param_key)
                    param_widget.setText(value)

                elif param_type == 'checkbox':
                    param_widget = ParamCheckBox(param_key)
                    if isinstance(value, str):
                        value = value.lower().strip() == 'true'
                        params[param_key]['value'] = value
                    param_widget.setChecked(value)

                elif param_type == 'pushbtn':
                    param_widget = ParamPushButton(param_key, display_param_name)
                    require_label = False

                elif param_type == 'line_editor':
                    param_widget = ParamLineEditor(param_key, force_digital=isinstance(value, (float, int)))
                    param_widget.setText(str(value))

                elif param_type == 'secret':
                    param_widget = SecretParamWidget(param_key, size=param_size)
                    param_widget.setText(str(value))

                elif param_type == 'check_group':
                    param_widget = ParamCheckGroup(param_key, check_group=value)

                if param_widget is not None:
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                    if param_description:
                        param_widget.setToolTip(param_description)

            if param_widget is not None and require_label and label_above:
                self.param_widgets[param_key] = param_widget
                row_widget = QWidget(self)
                row_widget.setObjectName('ParamLabelAboveRow')
                row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                row_layout = QVBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                param_label = ParamNameLabel(display_param_name)
                param_label.setObjectName('ParamLabelAboveLabel')
                set_label_tooltip_from_widget(param_label, param_widget)
                row_layout.addWidget(param_label, 0, Qt.AlignmentFlag.AlignLeft)
                row_layout.addWidget(param_widget, 0, Qt.AlignmentFlag.AlignLeft)
                column_span = 3 if body_min_width > 0 else 2
                param_layout.addWidget(row_widget, ii, 0, 1, column_span)
                self.param_rows[param_key] = [row_widget]
                continue

            widget_idx = 2 if body_min_width > 0 else 0
            row_widgets = []
            if require_label:
                param_label = ParamNameLabel(display_param_name)
                param_label.setObjectName('ParamFieldLabel')
                if param_widget is not None:
                    set_label_tooltip_from_widget(param_label, param_widget)
                param_layout.addWidget(param_label, ii, 0)
                row_widgets.append(param_label)
                widget_idx = 2 if body_min_width > 0 else 1
            if param_widget is not None:
                self.param_widgets[param_key] = param_widget
                row_widgets.append(param_widget)
                pw_lo = None
                if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                    pw_lo = QHBoxLayout()
                    pw_lo.addWidget(param_widget)
                if hasattr(param_widget, 'flush_btn'):
                    pw_lo.addWidget(param_widget.flush_btn)
                    param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                if hasattr(param_widget, 'path_select_btn'):
                    pw_lo.addWidget(param_widget.path_select_btn)
                    param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                if pw_lo is None:
                    param_layout.addWidget(
                        param_widget,
                        ii,
                        widget_idx,
                        Qt.AlignmentFlag.AlignLeft,
                    )
                else:
                    param_layout.addLayout(
                        pw_lo,
                        ii,
                        widget_idx,
                        Qt.AlignmentFlag.AlignLeft,
                    )
                self.param_rows[param_key] = row_widgets
            else:
                v = params[param_key]
                raise ValueError(f"Failed to initialize widget for key-value pair: {param_key}-{v}")
            
    def on_flushbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'flush': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_pathbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'select_path': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_paramwidget_edited(self, param_key, param_content):
        content_dict = {'content': param_content}
        self.paramwidget_edited.emit(param_key, content_dict)

    def setParamVisible(self, param_key: str, visible: bool):
        for widget in self.param_rows.get(param_key, []):
            widget.setVisible(visible)

class ModuleParseWidgets(QWidget):
    def addModulesParamWidgets(self, ocr_instance):
        self.params = ocr_instance.get_params()
        self.on_module_changed()

    def on_module_changed(self):
        self.updateModuleParamWidget()

    def updateModuleParamWidget(self):
        widget = ParamWidget(self.params, scrollWidget=self)
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)

class ModuleConfigParseWidget(QWidget):
    module_changed = Signal(str)
    paramwidget_edited = Signal(str, dict)
    def __init__(self, module_name: str, get_valid_module_keys: Callable, scrollWidget: QWidget, add_from: int = 1, module_type: str = '', *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.module_type = module_type
        self.get_valid_module_keys = get_valid_module_keys
        self.module_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.params_layout = QHBoxLayout()
        self.params_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.params_layout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        self.header_widget = QWidget(self)
        self.header_widget.setObjectName('PipelineModuleHeader')
        self.header_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.header_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        p_layout = QHBoxLayout(self.header_widget)
        p_layout.setContentsMargins(0, 4, 6, 4)
        p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.module_label = ParamNameLabel(module_name)
        p_layout.addWidget(self.module_label)
        p_layout.addWidget(self.module_combobox)
        p_layout.addStretch(-1)
        self.p_layout = p_layout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        self.param_widget_map = {}
        layout.addWidget(self.header_widget)
        layout.addLayout(self.params_layout)
        layout.setSpacing(CONFIG_CONTENT_ROW_SPACING)
        self.vlayout = layout
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.visibleWidget: QWidget = None
        self.module_dict: dict = {}

    def setJumpHighlighted(self, highlighted: bool):
        if self.header_widget.property('jumpHighlighted') == highlighted:
            return
        self.header_widget.setProperty('jumpHighlighted', highlighted)
        style = self.header_widget.style()
        style.unpolish(self.header_widget)
        style.polish(self.header_widget)
        self.header_widget.update()

    def addModulesParamWidgets(self, module_dict: dict, selected_module: str = None):
        invalid_module_keys = []
        valid_modulekeys = self.get_valid_module_keys()

        num_widgets_before = len(self.param_widget_map)

        for module in module_dict:
            if module not in valid_modulekeys:
                invalid_module_keys.append(module)
                continue

            if module in self.param_widget_map:
                LOGGER.warning(f'duplicated module key: {module}')
                continue

            self.module_combobox.addItem(module)
            params = module_dict[module]
            if params is not None:
                self.param_widget_map[module] = None

        if len(invalid_module_keys) > 0:
            LOGGER.warning(F'Invalid module keys: {invalid_module_keys}')
            for ik in invalid_module_keys:
                module_dict.pop(ik)

        self.module_dict = module_dict

        num_widgets_after = len(self.param_widget_map)
        if num_widgets_before == 0 and num_widgets_after > 0:
            if selected_module in self.module_dict:
                self.module_combobox.setCurrentText(selected_module)
            self.updateModuleParamWidget()
            self.module_combobox.currentTextChanged.connect(self.on_module_changed)

    def setModule(self, module: str):
        if self.module_combobox.currentText() == module and self.visibleWidget is not None:
            return
        self.blockSignals(True)
        self.module_combobox.setCurrentText(module)
        self.updateModuleParamWidget()
        self.blockSignals(False)

    def updateModuleParamWidget(self):
        module = self.module_combobox.currentText()
        if self.visibleWidget is not None:
            self.visibleWidget.hide()
        if module in self.param_widget_map:
            widget: QWidget = self.param_widget_map[module]
            if widget is None:
                # Build parameter widgets only for the selected manifest entry.
                params = self.module_dict[module]
                widget = ParamWidget(
                    params,
                    scrollWidget=self,
                    module_type=self.module_type,
                    module_key=module,
                    body_min_width=CONFIG_MODULE_PARAM_BODY_MIN_WIDTH,
                )
                widget.paramwidget_edited.connect(self.paramwidget_edited)
                self.param_widget_map[module] = widget
                self.params_layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            else:
                widget.show()
            self.visibleWidget = widget

    def on_module_changed(self):
        self.updateModuleParamWidget()
        self.module_changed.emit(self.module_combobox.currentText())


class TranslatorConfigPanel(ModuleConfigParseWidget):

    show_pre_MT_keyword_window = Signal()
    show_MT_keyword_window = Signal()
    show_OCR_keyword_window = Signal()
    llm_profile_changed = Signal(str)
    llm_profile_config_clicked = Signal(str)

    def __init__(self, module_name, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TRANSLATORS, scrollWidget, *args, module_type='translator', **kwargs)
        self.translator_changed = self.module_changed
    
        self.llm_profile_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.llm_profile_config_btn = NoBorderPushBtn(self.tr('Config'), self)
        self.llm_profile_config_btn.clicked.connect(self.on_llm_profile_config_clicked)
        self.llm_profile_layout = QHBoxLayout()
        self.llm_profile_layout.setSpacing(15)
        self.llm_profile_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.llm_profile_label = ParamNameLabel(self.tr('LLM Profile'))
        self.llm_profile_layout.addWidget(self.llm_profile_label)
        self.llm_profile_layout.addWidget(self.llm_profile_combobox)
        self.llm_profile_layout.addWidget(self.llm_profile_config_btn)
        self.llm_profile_layout.addStretch(-1)
        self.llm_profile_combobox.currentIndexChanged.connect(self.on_llm_profile_changed)
        
        self.vlayout.insertLayout(1, self.llm_profile_layout)
        self.replaceOCRkeywordBtn = NoBorderPushBtn(
            self.tr('Keyword substitution for source text'),
            self,
        )
        self.replaceOCRkeywordBtn.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        self.replacePreMTkeywordBtn = NoBorderPushBtn(
            self.tr('Keyword substitution for machine translation source text'),
            self,
        )
        self.replacePreMTkeywordBtn.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.replacePreMTkeywordBtn.clicked.connect(
            self.show_pre_MT_keyword_window
        )
        self.replaceMTkeywordBtn = NoBorderPushBtn(
            self.tr('Keyword substitution for machine translation'),
            self,
        )
        self.replaceMTkeywordBtn.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        self.vlayout.addWidget(self.replaceOCRkeywordBtn)
        self.vlayout.addWidget(self.replacePreMTkeywordBtn)
        self.vlayout.addWidget(self.replaceMTkeywordBtn)
        self.refreshLLMProfiles()
        self.setLLMProfileControlsVisible(False)

    def setLLMProfileControlsVisible(self, visible: bool):
        for widget in [self.llm_profile_label, self.llm_profile_combobox, self.llm_profile_config_btn]:
            widget.setVisible(visible)

    def refreshLLMProfiles(self):
        self.llm_profile_combobox.blockSignals(True)
        self.llm_profile_combobox.clear()
        for profile in pcfg.module.llm_profiles:
            self.llm_profile_combobox.addItem(profile.name or profile.id, profile.id)
        idx = self.llm_profile_combobox.findData(pcfg.module.translator_llm_id)
        if idx >= 0:
            self.llm_profile_combobox.setCurrentIndex(idx)
        self.llm_profile_combobox.blockSignals(False)

    def on_llm_profile_changed(self):
        profile_id = self.llm_profile_combobox.currentData()
        if profile_id:
            pcfg.module.translator_llm_id = profile_id
            self.llm_profile_changed.emit(profile_id)

    def on_llm_profile_config_clicked(self):
        profile_id = self.llm_profile_combobox.currentData() or pcfg.module.translator_llm_id
        self.llm_profile_config_clicked.emit(profile_id)

    def setTranslatorMetadata(self, name: str, supported_src_list, supported_tgt_list, lang_source: str, lang_target: str):
        refresh_params = self.module_combobox.currentText() != name or self.visibleWidget is None
        self.module_combobox.blockSignals(True)
        self.module_combobox.setCurrentText(name)
        if refresh_params:
            self.updateModuleParamWidget()
        self.module_combobox.blockSignals(False)
        self.refreshLLMProfiles()
        self.setLLMProfileControlsVisible(name == LLM_TRANSLATOR_KEY)



class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_INPAINTERS, scrollWidget, *args, module_type='inpainter', **kwargs)
        self.inpainter_changed = self.module_changed
        self.setInpainter = self.setModule

    def showEvent(self, e) -> None:
        self.p_layout.insertWidget(1, self.module_combobox)
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        self.p_layout.removeWidget(self.module_combobox)
        return super().hideEvent(e)

class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TEXTDETECTORS, scrollWidget, *args, module_type='textdetector', **kwargs)
        self.detector_changed = self.module_changed
        self.setDetector = self.setModule
        

class OCRConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_OCR, scrollWidget, *args, module_type='ocr', **kwargs)
        self.ocr_changed = self.module_changed
        self.setOCR = self.setModule

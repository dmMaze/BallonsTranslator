from typing import Callable

from ballontranslator.modules import GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS, GET_VALID_TRANSLATORS, GET_VALID_OCR
from ballontranslator.utils.logger import logger as LOGGER
from .custom_widget import ConfigComboBox, ParamComboBox, NoBorderPushBtn, ParamNameLabel
from ballontranslator.utils.shared import CONFIG_COMBOBOX_LONG, size2width, CONFIG_COMBOBOX_HEIGHT
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import LLM_TRANSLATOR_KEY

from qtpy.QtWidgets import QPlainTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QLineEdit, QGridLayout, QPushButton, QSizePolicy, QLayout
from qtpy.QtCore import QTimer, Qt, Signal
from qtpy.QtGui import QDoubleValidator

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
        self.editor = QLineEdit(self)
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
        self._auto_height = param_key == 'prompt'
        self._auto_max_height = 100
        self._auto_min_height = self._auto_max_height

        self.setFixedWidth(int(CONFIG_COMBOBOX_LONG))
        if self._auto_height:
            self.setMinimumHeight(self._auto_min_height)
            self.setMaximumHeight(self._auto_max_height)
            self.setFixedHeight(self._auto_min_height)
        else:
            self.setFixedHeight(100)
        # self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)
        if self._auto_height:
            self.document().documentLayout().documentSizeChanged.connect(lambda *_: self.updateAutoHeight())

    def on_text_changed(self):
        self.updateAutoHeight()
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)
        self.updateAutoHeight()
        if self._auto_height:
            QTimer.singleShot(0, self.updateAutoHeight)

    def text(self):
        return self.toPlainText()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateAutoHeight()

    def updateAutoHeight(self):
        if not self._auto_height:
            return
        available_width = max(1, self.viewport().width() - 16)
        font_metrics = self.fontMetrics()
        visual_lines = 0
        for line in (self.toPlainText() or ' ').splitlines() or [' ']:
            text_width = max(1, font_metrics.horizontalAdvance(line))
            visual_lines += max(1, (text_width + available_width - 1) // available_width)
        content_height = visual_lines * font_metrics.lineSpacing() + 14
        height = max(self._auto_min_height, min(content_height, self._auto_max_height))
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


class ParamCheckBox(QCheckBox):
    paramwidget_edited = Signal(str, bool)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('ParamCheckBox')
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def get_param_display_name(param_key: str, param_dict: dict = None):
    if param_dict is not None and isinstance(param_dict, dict):
        if 'display_name' in param_dict:
            return param_dict['display_name']
    return param_key


def ensure_current_device_option(param_dict: dict):
    not_supported = param_dict.get('__device_not_supported', [])
    current_value = str(param_dict.get('value', 'cpu'))
    options = [str(opt) for opt in param_dict.get('options', [])]
    if current_value not in options and all(device not in current_value for device in not_supported):
        options.append(current_value)
    param_dict['options'] = options
    param_dict['value'] = current_value if current_value in options else 'cpu'


class ParamPushButton(QPushButton):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, param_dict: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.setText(get_param_display_name(param_key, param_dict))
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.paramwidget_edited.emit(self.param_key, '')


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(self, params, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        layout = QHBoxLayout(self)
        layout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        self.param_layout = param_layout = QGridLayout()
        self.param_widgets = {}
        self.param_rows = {}
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(param_layout)
        layout.addStretch(-1)

        if 'description' in params:
            self.setToolTip(params['description'])

        for ii, param_key in enumerate(params):
            if param_key == 'description' or param_key.startswith('__'):
                continue
            display_param_name = param_key

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
                display_param_name = get_param_display_name(param_key, param_dict)
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
                    param_widget = ParamPushButton(param_key, param_dict)
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
                    if 'description' in param_dict:
                        param_widget.setToolTip(param_dict['description'])

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
                row_layout.addWidget(param_label, 0, Qt.AlignmentFlag.AlignLeft)
                row_layout.addWidget(param_widget, 0, Qt.AlignmentFlag.AlignLeft)
                param_layout.addWidget(row_widget, ii, 0, 1, 2)
                self.param_rows[param_key] = [row_widget]
                continue

            widget_idx = 0
            row_widgets = []
            if require_label:
                param_label = ParamNameLabel(display_param_name)
                param_label.setObjectName('ParamFieldLabel')
                param_layout.addWidget(param_label, ii, 0)
                row_widgets.append(param_label)
                widget_idx = 1
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
                    param_layout.addWidget(param_widget, ii, widget_idx)
                else:
                    param_layout.addLayout(pw_lo, ii, widget_idx)
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
    def __init__(self, module_name: str, get_valid_module_keys: Callable, scrollWidget: QWidget, add_from: int = 1, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.get_valid_module_keys = get_valid_module_keys
        self.module_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.params_layout = QHBoxLayout()
        self.params_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.params_layout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        p_layout = QHBoxLayout()
        p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.module_label = ParamNameLabel(module_name)
        p_layout.addWidget(self.module_label)
        p_layout.addWidget(self.module_combobox)
        p_layout.addStretch(-1)
        self.p_layout = p_layout

        layout = QVBoxLayout(self)
        layout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        self.param_widget_map = {}
        layout.addLayout(p_layout) 
        layout.addLayout(self.params_layout)
        layout.setSpacing(14)
        self.vlayout = layout
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        self.visibleWidget: QWidget = None
        self.module_dict: dict = {}

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
                widget = ParamWidget(params, scrollWidget=self)
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
        super().__init__(module_name, GET_VALID_TRANSLATORS, scrollWidget=scrollWidget, *args, **kwargs)
        self.translator_changed = self.module_changed
    
        self.source_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.target_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.replacePreMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation source text"), self)
        self.replacePreMTkeywordBtn.clicked.connect(self.show_pre_MT_keyword_window)
        self.replacePreMTkeywordBtn.setFixedWidth(420)
        self.replaceMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        self.replaceMTkeywordBtn.setFixedWidth(420)
        self.replaceOCRkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for source text"), self)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        self.replaceOCRkeywordBtn.setFixedWidth(420)
        self.translateByTextblockBox = ParamCheckerBox(self.tr('Translate each text block individually'))

        st_layout = QHBoxLayout()
        st_layout.setSpacing(15)
        st_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        st_layout.addWidget(ParamNameLabel(self.tr('Source')))
        st_layout.addWidget(self.source_combobox)
        st_layout.addWidget(ParamNameLabel(self.tr('Target')))
        st_layout.addWidget(self.target_combobox)

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
        
        self.vlayout.insertLayout(1, st_layout) 
        self.vlayout.insertLayout(2, self.llm_profile_layout)
        self.vlayout.addWidget(self.translateByTextblockBox)
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
        idx = self.llm_profile_combobox.findData(pcfg.module.llm_profile)
        if idx >= 0:
            self.llm_profile_combobox.setCurrentIndex(idx)
        self.llm_profile_combobox.blockSignals(False)

    def on_llm_profile_changed(self):
        profile_id = self.llm_profile_combobox.currentData()
        if profile_id:
            pcfg.module.llm_profile = profile_id
            self.llm_profile_changed.emit(profile_id)

    def on_llm_profile_config_clicked(self):
        profile_id = self.llm_profile_combobox.currentData() or pcfg.module.llm_profile
        self.llm_profile_config_clicked.emit(profile_id)

    def setTranslatorMetadata(self, name: str, supported_src_list, supported_tgt_list, lang_source: str, lang_target: str):
        refresh_params = self.module_combobox.currentText() != name or self.visibleWidget is None
        self.source_combobox.blockSignals(True)
        self.target_combobox.blockSignals(True)
        self.module_combobox.blockSignals(True)

        self.source_combobox.clear()
        self.target_combobox.clear()

        self.source_combobox.addItems(supported_src_list)
        self.target_combobox.addItems(supported_tgt_list)
        self.module_combobox.setCurrentText(name)
        self.source_combobox.setCurrentText(lang_source)
        self.target_combobox.setCurrentText(lang_target)
        if refresh_params:
            self.updateModuleParamWidget()
        self.source_combobox.blockSignals(False)
        self.target_combobox.blockSignals(False)
        self.module_combobox.blockSignals(False)
        self.refreshLLMProfiles()
        self.setLLMProfileControlsVisible(name == LLM_TRANSLATOR_KEY)


class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_INPAINTERS, scrollWidget = scrollWidget, *args, **kwargs)
        self.inpainter_changed = self.module_changed
        self.setInpainter = self.setModule
        self.needInpaintChecker = ParamCheckerBox(self.tr('Let the program decide whether it is necessary to use the selected inpaint method.'))
        self.filter_mask_by_bboxes_checker = QCheckBox()
        self.filter_mask_by_bboxes_checker.setObjectName('ConfigCheckBox')
        filter_mask_label = ParamNameLabel(self.tr('Filter mask by text boxes'))
        filter_mask_row = QWidget(self)
        filter_mask_row.setObjectName('ConfigInlineRow')
        filter_mask_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        filter_mask_layout = QHBoxLayout(filter_mask_row)
        filter_mask_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        filter_mask_layout.addWidget(self.filter_mask_by_bboxes_checker)
        filter_mask_layout.addWidget(filter_mask_label)
        self.vlayout.addWidget(self.needInpaintChecker)
        self.vlayout.addWidget(filter_mask_row)

    def showEvent(self, e) -> None:
        self.p_layout.insertWidget(1, self.module_combobox)
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        self.p_layout.removeWidget(self.module_combobox)
        return super().hideEvent(e)

class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TEXTDETECTORS, scrollWidget = scrollWidget, *args, **kwargs)
        self.detector_changed = self.module_changed
        self.setDetector = self.setModule
        self.keep_existing_checker = QCheckBox(text=self.tr('Keep Existing Lines'))
        self.keep_existing_checker.setObjectName('ConfigCheckBox')
        self.vlayout.insertWidget(1, self.keep_existing_checker)
        

class OCRConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_OCR, scrollWidget = scrollWidget, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.setOCR = self.setModule
        self.restoreEmptyOCRChecker = QCheckBox(self.tr("Delete and restore region where OCR return empty string."), self)
        self.restoreEmptyOCRChecker.setObjectName('ConfigCheckBox')
        self.restoreEmptyOCRChecker.clicked.connect(self.on_restore_empty_ocr)
        self.vlayout.addWidget(self.restoreEmptyOCRChecker)
        # 字体检测选项
        self.fontDetectChecker = QCheckBox(self.tr("Font Detection"), self)
        self.fontDetectChecker.setObjectName('ConfigCheckBox')
        self.fontDetectChecker.setChecked(pcfg.module.ocr_font_detect)
        self.fontDetectChecker.clicked.connect(self.on_fontdetect_changed)
        self.vlayout.addWidget(self.fontDetectChecker)

    def on_restore_empty_ocr(self):
        pcfg.restore_ocr_empty = self.restoreEmptyOCRChecker.isChecked()

    def on_fontdetect_changed(self):
        pcfg.module.ocr_font_detect = self.fontDetectChecker.isChecked()

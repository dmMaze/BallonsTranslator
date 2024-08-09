from typing import List, Callable

from modules import GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS, GET_VALID_TRANSLATORS, GET_VALID_OCR, \
    BaseTranslator, DEFAULT_DEVICE, GPUINTENSIVE_SET
from utils.logger import logger as LOGGER
from .stylewidgets import ConfigComboBox, NoBorderPushBtn, CustomComboBox
from utils.shared import CONFIG_FONTSIZE_CONTENT, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from utils.config import pcfg
import logging

# Настройка логгера
logging.basicConfig(level=logging.DEBUG)

from qtpy.QtWidgets import QPlainTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QComboBox, QCheckBox, QLineEdit, QGridLayout, QPushButton
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFontMetricsF, QDoubleValidator


class ParamNameLabel(QLabel):
    def __init__(self, param_name: str, alignment = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if alignment is None:
            self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        else:
            self.setAlignment(alignment)

        font = self.font()
        font.setPointSizeF(CONFIG_FONTSIZE_CONTENT-2)
        self.setFont(font)
        self.setText(param_name)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

class ParamLineEditor(QLineEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, force_digital, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

        if force_digital:
            validator = QDoubleValidator()
            self.setValidator(validator)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

class ParamEditor(QPlainTextEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key

        if param_key == 'chat sample':
            self.setFixedWidth(int(CONFIG_COMBOBOX_LONG * 1.2))
            self.setFixedHeight(200)
        else:
            self.setFixedWidth(CONFIG_COMBOBOX_LONG)
            self.setFixedHeight(100)
        # self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)

    def text(self):
        return self.toPlainText()


class ParamComboBox(CustomComboBox):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, options: List[str], size=CONFIG_COMBOBOX_SHORT, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(scrollWidget=scrollWidget, *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size)
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        options = [str(opt) for opt in options]
        self.addItems(options)
        self.currentTextChanged.connect(self.on_select_changed)

    def on_select_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.currentText())


class ParamCheckerBox(QWidget):
    checker_changed = Signal(bool)
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.checker = QCheckBox()
        name_label = ParamNameLabel(param_key)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(name_label)
        hlayout.addWidget(self.checker)
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
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def get_param_display_name(param_key: str, param_dict: dict = None):
    if param_dict is not None and isinstance(param_dict, dict):
        if 'display_name' in param_dict:
            return param_dict['display_name']
    return param_key


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
        layout = QHBoxLayout(self)
        self.param_layout = param_layout = QGridLayout()
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(param_layout)
        layout.addStretch(-1)

        if 'description' in params:
            self.setToolTip(params['description'])

        for ii, param_key in enumerate(params):
            if param_key == 'description':
                continue
            display_param_name = param_key

            require_label = True
            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
            param_widget = None  # Инициализация переменной

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

                if param_dict['type'] == 'selector':
                    if 'url' in param_key:
                        size = CONFIG_COMBOBOX_MIDEAN
                    else:
                        size = CONFIG_COMBOBOX_SHORT

                    param_widget = ParamComboBox(param_key, param_dict['options'], size=size, scrollWidget=scrollWidget)

                    if param_key == 'device' and DEFAULT_DEVICE == 'cpu':
                        param_dict['value'] = 'cpu'
                        for ii, device in enumerate(param_dict['options']):
                            if device in GPUINTENSIVE_SET:
                                model = param_widget.model()
                                item = model.item(ii, 0)
                                item.setEnabled(False)
                    param_widget.setCurrentText(str(value))
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

                elif param_dict['type'] == 'editor':
                    param_widget = ParamEditor(param_key)
                    param_widget.setText(value)
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

                elif param_dict['type'] == 'checkbox':
                    param_widget = ParamCheckBox(param_key)
                    if isinstance(value, str):
                        value = value.lower().strip() == 'true'
                        params[param_key]['value'] = value
                    param_widget.setChecked(value)
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

                elif param_dict['type'] == 'pushbtn':
                    param_widget = ParamPushButton(param_key, param_dict)
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                    require_label = False

                elif param_dict['type'] == 'float':  # Добавлен новый тип 'float'
                    param_widget = ParamLineEditor(param_key, force_digital=True)
                    param_widget.setText(str(value))
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

                if 'description' in param_dict and param_widget is not None:
                    param_widget.setToolTip(param_dict['description'])

            widget_idx = 0
            if require_label:
                param_label = ParamNameLabel(display_param_name)
                param_layout.addWidget(param_label, ii, 0)
                widget_idx = 1
            if param_widget:
                param_layout.addWidget(param_widget, ii, widget_idx)
            else:
                raise ValueError(f"Failed to initialize widget for key: {param_key}")

    def on_paramwidget_edited(self, param_key, param_content):
        content_dict = {'content': param_content}
        self.paramwidget_edited.emit(param_key, content_dict)

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
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        p_layout = QHBoxLayout()
        p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.module_label = ParamNameLabel(module_name)
        p_layout.addWidget(self.module_label)
        p_layout.addWidget(self.module_combobox)
        p_layout.addStretch(-1)
        self.p_layout = p_layout

        layout = QVBoxLayout(self)
        self.param_widget_map = {}
        layout.addLayout(p_layout) 
        layout.addLayout(self.params_layout)
        layout.setSpacing(30)
        self.vlayout = layout

        self.visibleWidget: QWidget = None
        self.module_dict: dict = {}

    def addModulesParamWidgets(self, module_dict: dict):
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
            self.on_module_changed()
            self.module_combobox.currentTextChanged.connect(self.on_module_changed)

    def setModule(self, module: str):
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
                # lazy load widgets
                params = self.module_dict[module]
                widget = ParamWidget(params, scrollWidget=self)
                widget.paramwidget_edited.connect(self.paramwidget_edited)
                self.param_widget_map[module] = widget
                self.params_layout.addWidget(widget)
            else:
                widget.show()
            self.visibleWidget = widget

    def on_module_changed(self):
        self.updateModuleParamWidget()
        self.module_changed.emit(self.module_combobox.currentText())


class TranslatorConfigPanel(ModuleConfigParseWidget):

    show_MT_keyword_window = Signal()

    def __init__(self, module_name, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TRANSLATORS, scrollWidget=scrollWidget, *args, **kwargs)
        self.translator_changed = self.module_changed
    
        self.source_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.target_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.replaceMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        self.replaceMTkeywordBtn.setFixedWidth(500)

        st_layout = QHBoxLayout()
        st_layout.setSpacing(15)
        st_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        st_layout.addWidget(ParamNameLabel(self.tr('Source')))
        st_layout.addWidget(self.source_combobox)
        st_layout.addWidget(ParamNameLabel(self.tr('Target')))
        st_layout.addWidget(self.target_combobox)
        
        self.vlayout.insertLayout(1, st_layout) 
        self.vlayout.addWidget(self.replaceMTkeywordBtn)

    def finishSetTranslator(self, translator: BaseTranslator):
        self.source_combobox.blockSignals(True)
        self.target_combobox.blockSignals(True)
        self.module_combobox.blockSignals(True)

        self.source_combobox.clear()
        self.target_combobox.clear()

        self.source_combobox.addItems(translator.supported_src_list)
        self.target_combobox.addItems(translator.supported_tgt_list)
        self.module_combobox.setCurrentText(translator.name)
        self.source_combobox.setCurrentText(translator.lang_source)
        self.target_combobox.setCurrentText(translator.lang_target)
        self.updateModuleParamWidget()
        self.source_combobox.blockSignals(False)
        self.target_combobox.blockSignals(False)
        self.module_combobox.blockSignals(False)


class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_INPAINTERS, scrollWidget = scrollWidget, *args, **kwargs)
        self.inpainter_changed = self.module_changed
        self.setInpainter = self.setModule
        self.needInpaintChecker = ParamCheckerBox(self.tr('Let the program decide whether it is necessary to use the selected inpaint method.'))
        self.vlayout.addWidget(self.needInpaintChecker)

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


class OCRConfigPanel(ModuleConfigParseWidget):
    
    show_OCR_keyword_window = Signal()

    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_OCR, scrollWidget = scrollWidget, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.setOCR = self.setModule

        self.replaceOCRkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for OCR results"), self)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        self.replaceOCRkeywordBtn.setFixedWidth(500)

        self.restoreEmptyOCRChecker = QCheckBox(self.tr("Delete and restore region where OCR return empty string."), self)
        self.restoreEmptyOCRChecker.clicked.connect(self.on_restore_empty_ocr)

        self.vlayout.addWidget(self.replaceOCRkeywordBtn)
        self.vlayout.addWidget(self.restoreEmptyOCRChecker)

    def on_restore_empty_ocr(self):
        pcfg.restore_ocr_empty = self.restoreEmptyOCRChecker.isChecked()
from typing import List

from modules import GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS, GET_VALID_TRANSLATORS, GET_VALID_OCR, \
    BaseTranslator, DEFAULT_DEVICE
from utils.logger import logger as LOGGER
from .stylewidgets import ConfigComboBox, NoBorderPushBtn
from .constants import CONFIG_FONTSIZE_CONTENT, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT

from qtpy.QtWidgets import QPlainTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QComboBox, QCheckBox, QLineEdit
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFontMetricsF, QDoubleValidator


class ParamNameLabel(QLabel):
    def __init__(self, param_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        font = self.font()
        font.setPointSizeF(CONFIG_FONTSIZE_CONTENT-2)
        self.setFont(font)
        labelwidth = 120
        fm = QFontMetricsF(font)
        fmw = fm.boundingRect(param_name).width()
        labelwidth = int(max(fmw, labelwidth))
        self.setFixedWidth(labelwidth)
        self.setText(param_name)

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


class ParamComboBox(QComboBox):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, options: List[str], size=CONFIG_COMBOBOX_SHORT, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
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


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        param_layout = QVBoxLayout(self)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(14)
        for param_key in params:
            param_label = ParamNameLabel(param_key)

            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)

            if param_key == 'description':
                continue
            
            elif isinstance(params[param_key], bool):
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
                if param_dict['type'] == 'selector':
                    if 'url' in param_key:
                        size = CONFIG_COMBOBOX_MIDEAN
                    else:
                        size = CONFIG_COMBOBOX_SHORT
                    param_widget = ParamComboBox(param_key, param_dict['options'], size=size)

                    # if cuda is not available, disable combobox 'cuda' item
                    # https://stackoverflow.com/questions/38915001/disable-specific-items-in-qcombobox
                    if param_key == 'device' and DEFAULT_DEVICE == 'cpu':
                        param_dict['select'] = 'cpu'
                        for ii, device in enumerate(param_dict['options']):
                            if device == 'cuda':
                                model = param_widget.model()
                                item = model.item(ii, 0)
                                item.setEnabled(False)
                    if 'select' in param_dict:
                        param_widget.setCurrentText(str(param_dict['select']))
                    else:
                        param_widget.setCurrentIndex(0)
                        param_dict['select'] = param_widget.currentText()
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                elif param_dict['type'] == 'editor':
                    param_widget = ParamEditor(param_key)
                    val = params[param_key]['content']
                    param_widget.setText(val)
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)
            layout.addWidget(param_label)
            layout.addWidget(param_widget)
            layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            param_layout.addLayout(layout)

    def on_paramwidget_edited(self, param_key, param_content):
            content_dict = {'content': param_content}
            self.paramwidget_edited.emit(param_key, content_dict)


class ModuleConfigParseWidget(QWidget):
    module_changed = Signal(str)
    paramwidget_edited = Signal(str, dict)
    def __init__(self, module_name: str, get_valid_module_keys, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.get_valid_module_keys = get_valid_module_keys
        self.module_combobox = ConfigComboBox()
        self.params_layout = QHBoxLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        p_layout = QHBoxLayout()
        p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        p_layout.addWidget(ParamNameLabel(module_name))
        p_layout.addWidget(self.module_combobox)
        p_layout.setSpacing(15)

        layout = QVBoxLayout(self)
        self.param_widget_map = {}
        layout.addLayout(p_layout) 
        layout.addLayout(self.params_layout)
        layout.setSpacing(30)
        self.vlayout = layout

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
                param_widget = ParamWidget(params)
                param_widget.paramwidget_edited.connect(self.paramwidget_edited)
                self.param_widget_map[module] = param_widget
                self.params_layout.addWidget(param_widget)
                param_widget.hide()

        if len(invalid_module_keys) > 0:
            LOGGER.warning(F'Invalid module keys: {invalid_module_keys}')
            for ik in invalid_module_keys:
                module_dict.pop(ik)

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
        for key in self.param_widget_map:
            if key == module:
                self.param_widget_map[key].show()
            else:
                self.param_widget_map[key].hide()

    def on_module_changed(self):
        self.updateModuleParamWidget()
        self.module_changed.emit(self.module_combobox.currentText())


class TranslatorConfigPanel(ModuleConfigParseWidget):

    show_MT_keyword_window = Signal()

    def __init__(self, module_name, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TRANSLATORS, *args, **kwargs)
        self.translator_combobox = self.module_combobox
        self.translator_changed = self.module_changed
    
        self.source_combobox = ConfigComboBox()
        self.target_combobox = ConfigComboBox()
        self.replaceMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        self.replaceMTkeywordBtn.setFixedWidth(500)

        st_layout = QHBoxLayout()
        st_layout.setSpacing(15)
        st_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        st_layout.addWidget(ParamNameLabel(self.tr('Source ')))
        st_layout.addWidget(self.source_combobox)
        st_layout.addWidget(ParamNameLabel(self.tr('Target ')))
        st_layout.addWidget(self.target_combobox)
        
        self.vlayout.insertLayout(1, st_layout) 
        self.vlayout.addWidget(self.replaceMTkeywordBtn)

    def finishSetTranslator(self, translator: BaseTranslator):
        self.source_combobox.blockSignals(True)
        self.target_combobox.blockSignals(True)
        self.translator_combobox.blockSignals(True)

        self.source_combobox.clear()
        self.target_combobox.clear()

        self.source_combobox.addItems(translator.supported_src_list)
        self.target_combobox.addItems(translator.supported_tgt_list)
        self.translator_combobox.setCurrentText(translator.name)
        self.source_combobox.setCurrentText(translator.lang_source)
        self.target_combobox.setCurrentText(translator.lang_target)
        self.updateModuleParamWidget()
        self.source_combobox.blockSignals(False)
        self.target_combobox.blockSignals(False)
        self.translator_combobox.blockSignals(False)


class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_INPAINTERS, *args, **kwargs)
        self.inpainter_changed = self.module_changed
        self.inpainter_combobox = self.module_combobox
        self.setInpainter = self.setModule
        self.needInpaintChecker = ParamCheckerBox(self.tr('Let the program decide whether it is necessary to use the selected inpaint method.'))
        self.vlayout.addWidget(self.needInpaintChecker)

class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TEXTDETECTORS, *args, **kwargs)
        self.detector_changed = self.module_changed
        self.detector_combobox = self.module_combobox
        self.setDetector = self.setModule


class OCRConfigPanel(ModuleConfigParseWidget):
    
    show_OCR_keyword_window = Signal()

    def __init__(self, module_name: str, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_OCR, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.ocr_combobox = self.module_combobox
        self.setOCR = self.setModule

        self.replaceOCRkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for OCR results"), self)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        self.replaceOCRkeywordBtn.setFixedWidth(500)

        self.vlayout.addWidget(self.replaceOCRkeywordBtn)

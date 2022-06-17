from typing import List

from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QLabel, QComboBox, QListView, QToolBar, QMenu, QSpacerItem, QPushButton, QAction, QCheckBox, QToolButton, QSplitter, QStylePainter, QStyleOption, QStyle, QScrollArea, QLineEdit, QGroupBox, QGraphicsSimpleTextItem
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFontMetricsF

from dl import VALID_INPAINTERS, VALID_TEXTDETECTORS, VALID_TRANSLATORS, VALID_OCR, \
    TranslatorBase, DEFAULT_DEVICE
from utils.logger import logger as LOGGER

from .stylewidgets import ConfigComboBox
from .constants import CONFIG_FONTSIZE_CONTENT, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_SHORT


class ParamNameLabel(QLabel):
    def __init__(self, param_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        font = self.font()
        font.setPointSizeF(CONFIG_FONTSIZE_CONTENT-2)
        self.setFont(font)
        labelwidth = 120
        fm = QFontMetricsF(font)
        fmw = fm.width(param_name)
        labelwidth = max(fmw, labelwidth)
        self.setFixedWidth(labelwidth)
        self.setText(param_name)

class ParamEditor(QLineEdit):
    
    paramwidget_edited = pyqtSignal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        self.setFixedHeight(45)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())


class ParamComboBox(QComboBox):
    paramwidget_edited = pyqtSignal(str, str)
    def __init__(self, param_key: str, options: List[str], size=CONFIG_COMBOBOX_SHORT, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size)
        self.setFixedHeight(45)
        options = [str(opt) for opt in options]
        self.addItems(options)
        self.currentTextChanged.connect(self.on_select_changed)

    def on_select_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.currentText())


class ParamCheckerBox(QWidget):
    checker_changed = pyqtSignal(bool)
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
        self.checker_changed.emit(self.checker.isChecked())


class ParamWidget(QWidget):

    paramwidget_edited = pyqtSignal(str, str)
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        param_layout = QVBoxLayout(self)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(20)
        for param_key in params:
            param_label = ParamNameLabel(param_key)
            if param_key == 'description':
                continue
            
            elif isinstance(params[param_key], str):
                param_widget = ParamEditor(param_key)
                param_widget.setText(params[param_key])
                param_widget.paramwidget_edited.connect(self.paramwidget_edited)
            elif isinstance(params[param_key], dict):
                param_dict = params[param_key]
                if param_dict['type'] == 'selector':
                    param_widget = ParamComboBox(param_key, param_dict['options'])

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
                    param_widget.paramwidget_edited.connect(self.paramwidget_edited)
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(15)
            layout.addWidget(param_label)
            layout.addWidget(param_widget)
            layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            param_layout.addLayout(layout)


class ModuleConfigParseWidget(QWidget):
    module_changed = pyqtSignal(str)
    paramwidget_edited = pyqtSignal(str, str)
    def __init__(self, module_name: str, valid_param_keys, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.valid_param_keys = valid_param_keys
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

    def setupModulesParamWidgets(self, module_dict: dict):
        invalid_module_keys = []
        for module in module_dict:
            if module not in self.valid_param_keys:
                invalid_module_keys.append(module)
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

    def __init__(self, module_name, *args, **kwargs) -> None:
        super().__init__(module_name, VALID_TRANSLATORS, *args, **kwargs)
        self.translator_combobox = self.module_combobox
        self.translator_changed = self.module_changed
    
        self.source_combobox = ConfigComboBox()
        self.target_combobox = ConfigComboBox()
        st_layout = QHBoxLayout()
        st_layout.setSpacing(15)
        st_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        st_layout.addWidget(ParamNameLabel(self.tr('Source ')))
        st_layout.addWidget(self.source_combobox)
        st_layout.addWidget(ParamNameLabel(self.tr('Target ')))
        st_layout.addWidget(self.target_combobox)
        self.vlayout.insertLayout(1, st_layout) 

    def finishSetTranslator(self, translator: TranslatorBase):
        self.source_combobox.blockSignals(True)
        self.target_combobox.blockSignals(True)
        self.translator_combobox.blockSignals(True)

        self.source_combobox.clear()
        self.target_combobox.clear()

        for lang in translator.lang_map:
            if translator.lang_map[lang] != '':
                self.source_combobox.addItem(lang)
                self.target_combobox.addItem(lang)
        self.translator_combobox.setCurrentText(translator.name)
        self.source_combobox.setCurrentText(translator.lang_source)
        self.target_combobox.setCurrentText(translator.lang_target)
        self.updateModuleParamWidget()
        self.source_combobox.blockSignals(False)
        self.target_combobox.blockSignals(False)
        self.translator_combobox.blockSignals(False)


class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, *args, **kwargs) -> None:
        super().__init__(module_name, VALID_INPAINTERS, *args, **kwargs)
        self.inpainter_changed = self.module_changed
        self.inpainter_combobox = self.module_combobox
        self.setInpainter = self.setModule
        self.needInpaintChecker = ParamCheckerBox(self.tr('Let the program decide whether it is necessary to use the selected inpaint method.'))
        self.vlayout.addWidget(self.needInpaintChecker)

class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, *args, **kwargs) -> None:
        super().__init__(module_name, VALID_TEXTDETECTORS, *args, **kwargs)
        self.detector_changed = self.module_changed
        self.detector_combobox = self.module_combobox
        self.setDetector = self.setModule


class OCRConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, *args, **kwargs) -> None:
        super().__init__(module_name, VALID_OCR, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.ocr_combobox = self.module_combobox
        self.setOCR = self.setModule

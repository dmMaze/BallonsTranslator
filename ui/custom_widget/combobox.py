from typing import List

from qtpy.QtWidgets import QComboBox, QWidget
from qtpy.QtCore import Signal, Qt

from utils.shared import CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT



class ComboBox(QComboBox):

    index_changed = Signal()

    # https://stackoverflow.com/questions/3241830/qt-how-to-disable-mouse-scrolling-of-qcombobox
    def __init__(self, parent: QWidget = None, scrollWidget: QWidget = None, options: List[str] = None) -> None:
        super().__init__(parent)
        self.scrollWidget = scrollWidget
        if options is not None:
            self.addItems(options)
        self.currentIndexChanged.connect(self.on_current_index_changed)
        self._program_set_flag = False

    def setScrollWidget(self, scrollWidget: QWidget):
        self.scrollWidget = scrollWidget

    def on_current_index_changed(self):
        if self._program_set_flag:
            return
        self.index_changed.emit()

    def set_current_index(self, index: int, user_input: bool = True):
        '''
        emit index_changed only if user_input is True
        '''
        
        self._program_set_flag = not user_input
        self.setCurrentIndex(index)
        self._program_set_flag = False

    def wheelEvent(self, *args, **kwargs):
        if self.scrollWidget is None or self.hasFocus():
            return super().wheelEvent(*args, **kwargs)
        else:
            return self.scrollWidget.wheelEvent(*args, **kwargs)


class ConfigComboBox(ComboBox):

    def __init__(self, fix_size=True, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(scrollWidget, *args, **kwargs)
        self.fix_size = fix_size
        self.adjustSize()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def addItems(self, texts: List[str]) -> None:
        super().addItems(texts)
        self.adjustSize()

    def adjustSize(self) -> None:
        super().adjustSize()
        width = self.minimumSizeHint().width()
        if width < CONFIG_COMBOBOX_SHORT:
            width = CONFIG_COMBOBOX_SHORT
        elif width < CONFIG_COMBOBOX_MIDEAN:
            width = CONFIG_COMBOBOX_MIDEAN
        else:
            width = CONFIG_COMBOBOX_LONG
        if self.fix_size:
            self.setFixedWidth(width)
        else:
            self.setMaximumWidth(width)


class ParamComboBox(ComboBox):
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


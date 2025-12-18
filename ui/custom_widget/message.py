from typing import Callable, List, Dict
import time
import datetime

from qtpy.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QMessageBox, QSizePolicy, QProgressBar, QPushButton
from qtpy.QtGui import  QCloseEvent, QShowEvent
from qtpy.QtCore import Qt, Signal

from utils.shared import remove_from_runtime_widget_set, add_to_runtime_widget_set
from .widget import Widget


class MessageBox(QMessageBox):

    def __init__(self, info_msg: str = None, btn_type = QMessageBox.StandardButton.Ok, frame_less: bool = False, modal: bool = False, signal_slot_map_list: List[Dict] = None, *args, **kwargs):
        super().__init__(text=info_msg, *args, **kwargs)
        self.register_signal_slot_map = []
        add_to_runtime_widget_set(self)

        if frame_less:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        if modal:
            self.setModal(modal)
        if btn_type is not None:
            self.setStandardButtons(btn_type)

        if signal_slot_map_list is not None:
            self.connect_signals(signal_slot_map_list)

    def connect_signals(self, signal_slot_map_list):
        if signal_slot_map_list is None:
            return
        if isinstance(signal_slot_map_list, dict):
            signal_slot_map_list = [signal_slot_map_list]
        for signal_slot_map in signal_slot_map_list:
            slot = signal_slot_map['slot']
            if isinstance(slot, Callable):
                slot_func = slot
            else:
                assert isinstance(slot, str)
                slot_func = getattr(self, slot)
            signal_slot_map['signal'].connect(slot_func)
            signal_slot_map['slot_func'] = slot_func
            self.register_signal_slot_map.append(signal_slot_map)

    def disconnect_all(self):
        # https://stackoverflow.com/a/48501804/17671327
        for signal_slot_map in self.register_signal_slot_map:
            signal_slot_map['signal'].disconnect(signal_slot_map['slot_func'])
        self.register_signal_slot_map.clear()

    def clear_before_close(self):
        remove_from_runtime_widget_set(self)
        self.disconnect_all()

    def done(self, v: int = 0):
        self.clear_before_close()
        super().done(v)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.clear_before_close()
        return super().closeEvent(event)
    

class TaskProgressBar(Widget):
    def __init__(self, description: str = '', verbose=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.progressbar = QProgressBar(self)
        self.progressbar.setTextVisible(False)
        self.textlabel = QLabel(self)
        self.description = description
        self.text_len = 89
        layout = QVBoxLayout(self)

        self.verbose = verbose
        # if not verbose:
        
        if verbose:
            self.start_time = 0
            self.verbose_label = QLabel(self)
            hl = QHBoxLayout()
            hl.addWidget(self.textlabel)
            hl.addStretch(1)
            hl.addWidget(self.verbose_label)
            layout.addLayout(hl)
        else:
            layout.addWidget(self.textlabel)
            
        layout.addWidget(self.progressbar)
        self.updateProgress(0)

    def updateProgress(self, progress: int, msg: str = ''):
        self.progressbar.setValue(progress)
        if self.description:
            msg = self.description + msg
        if len(msg) > self.text_len - 3:
            msg = msg[:self.text_len - 3] + '...'
        elif len(msg) < self.text_len:
            pads = self.text_len - len(msg)
            msg = msg + ' ' * pads
        self.textlabel.setText(msg)
        self.progressbar.setValue(progress)

        if self.verbose:
            if progress == 0:
                self.verbose_label.setText('')
                self.start_time = time.time()
            elif progress == 100:
                self.verbose_label.setText('')
            else:
                cur_time = time.time()
                left_progress = 100 - progress
                eta = left_progress / progress * (cur_time - self.start_time + 1e-6)
                eta = datetime.timedelta(seconds=int(round(eta)))
                added_str = f'{progress}% ETA {eta}'
                self.verbose_label.setText(added_str)


class FrameLessMessageBox(QMessageBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        

class ProgressMessageBox(QDialog):
    showed = Signal()
    stop_clicked = Signal()
    
    def __init__(self, task_name: str = None, show_stop_btn: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(20, 10, 20, 30)

        self.task_progress_bar: TaskProgressBar = None
        if task_name is not None:
            self.task_progress_bar = TaskProgressBar(task_name, True)
            layout.addWidget(self.task_progress_bar)

    def on_stop_clicked(self):
        self.stop_clicked.emit()
        self.hide()

    def updateTaskProgress(self, value: int, msg: str = ''):
        if self.task_progress_bar is not None:
            self.task_progress_bar.updateProgress(value, msg)

    def setTaskName(self, task_name: str):
        if self.task_progress_bar is not None:
            self.task_progress_bar.description = task_name
    
    def zero_progress(self):
        if self.task_progress_bar is not None:
            self.task_progress_bar.updateProgress(0)

    def showEvent(self, e: QShowEvent) -> None:
        self.showed.emit()
        return super().showEvent(e)


class ImgtransProgressMessageBox(ProgressMessageBox):
    stop_clicked = Signal()
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(None, *args, **kwargs)
        
        self.detect_bar = TaskProgressBar(self.tr('Detecting: '), True, self)
        self.ocr_bar = TaskProgressBar(self.tr('OCR: '), True, self)
        self.inpaint_bar = TaskProgressBar(self.tr('Inpainting: '), True, self)
        self.translate_bar = TaskProgressBar(self.tr('Translating: '), True, self)

        layout = self.layout()
        layout.addWidget(self.detect_bar)
        layout.addWidget(self.ocr_bar)
        layout.addWidget(self.inpaint_bar)
        layout.addWidget(self.translate_bar)
        
        # 添加停止按钮
        self.stop_button = QPushButton(self.tr('Stop'), self)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setFixedWidth(self.sizeHint().width())
    
    def on_stop_clicked(self):
        self.stop_clicked.emit()
        # 重置按钮状态（为下次使用准备）
        self.stop_button.setEnabled(False)
        self.stop_button.setText(self.tr('trying to stop...'))


    def updateDetectProgress(self, value: int, msg: str = ''):
        self.detect_bar.updateProgress(value, msg)

    def updateOCRProgress(self, value: int, msg: str = ''):
        self.ocr_bar.updateProgress(value, msg)

    def updateInpaintProgress(self, value: int, msg: str = ''):
        self.inpaint_bar.updateProgress(value, msg)

    def updateTranslateProgress(self, value: int, msg: str = ''):
        self.translate_bar.updateProgress(value, msg)
    
    def zero_progress(self):
        self.updateDetectProgress(0)
        self.updateOCRProgress(0)
        self.updateInpaintProgress(0)
        self.updateTranslateProgress(0)
        # 重置停止按钮状态
        self.stop_button.setEnabled(True)
        self.stop_button.setText(self.tr('Stop'))

    def show_all_bars(self):
        self.detect_bar.show()
        self.ocr_bar.show()
        self.translate_bar.show()
        self.inpaint_bar.show()

    def hide_all_bars(self):
        self.detect_bar.hide()
        self.ocr_bar.hide()
        self.translate_bar.hide()
        self.inpaint_bar.hide()

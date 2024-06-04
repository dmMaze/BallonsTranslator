from typing import Callable, List, Dict
from functools import partial

from qtpy.QtWidgets import QDialog, QLabel, QHBoxLayout, QVBoxLayout, QMessageBox
from qtpy.QtGui import  QCloseEvent
from qtpy.QtCore import Qt

from utils.shared import remove_from_runtime_widget_set, add_to_runtime_widget_set


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
import os
import threading
import time
import unittest
from typing import Callable
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtTest import QSignalSpy, QTest
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.custom_widget import ImgtransProgressMessageBox
from ballontranslator.ui import module_manager as M
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class ControlledInpainter:
    name = 'lama_large_512px'

    def __init__(self, **params) -> None:
        self.loaded = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def all_model_loaded(self) -> bool:
        return self.loaded

    def load_model(self) -> None:
        self.loaded = True

    def unload_model(self, **kwargs) -> None:
        self.loaded = False

    def inpaint(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError('Test did not release inference.')
        if self.fail:
            self.fail = False
            raise ValueError('inference failed')
        return img + 1


class CanvasInpaintLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_config = pcfg.module.copy()
        pcfg.module.inpainter = ControlledInpainter.name
        self.project = ProjImgTrans()
        self.project.inpainted_array = np.full((8, 8, 3), 10, np.uint8)
        self.window = QWidget()
        self.config = ConfigPanel(self.window)
        self.progress = ImgtransProgressMessageBox(self.window)
        self.manager = M.ModuleManager(self.project)
        self.manager.setupThread(self.config, self.progress, self.window)
        self.model = ControlledInpainter()
        self.model.load_model()
        self.manager.inpaint_thread.module = self.model
        self.completed = QSignalSpy(self.manager.canvas_inpaint_finished)
        self.failed = QSignalSpy(self.manager.canvas_inpaint_failed)

    def tearDown(self) -> None:
        self.model.release.set()
        self.manager.inpaint_thread.wait(3000)
        self.app.processEvents()
        self.manager.deleteLater()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        pcfg.module.merge(self.old_config)

    def request(self) -> dict:
        return {
            'img': self.project.inpainted_array,
            'mask': np.full((8, 8), 255, np.uint8),
            'inpaint_rect': [0, 0, 8, 8],
        }

    def wait_until(self, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 3
        while not predicate():
            self.assertLess(time.monotonic(), deadline, 'Worker did not finish.')
            QTest.qWait(5)

    def test_preparation_signal_before_thread_exit_does_not_lose_click(self) -> None:
        thread = self.manager.inpaint_thread
        thread.module = None
        prepared = threading.Event()
        release = threading.Event()
        set_module = thread._set_module

        def prepare_then_hold(name: str) -> None:
            set_module(name)
            prepared.set()
            release.wait(3)

        with (
            patch.object(thread, '_prepare_module_class', return_value=ControlledInpainter),
            patch.object(thread, '_set_module', prepare_then_hold),
            patch.object(self.manager, '_missing_module_requirements_for_modules', return_value=[]),
        ):
            try:
                self.manager.canvas_inpaint(self.request())
                self.wait_until(prepared.is_set)
                QTest.qWait(30)
                self.assertFalse(self.completed)
            finally:
                release.set()
            self.wait_until(lambda: len(self.completed) == 1)
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], 11)

    def test_page_change_discards_old_result_and_runs_new_page(self) -> None:
        self.model.release.clear()
        self.manager.canvas_inpaint(self.request())
        self.wait_until(self.model.started.is_set)
        self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
        self.manager.handle_page_changed()
        self.manager.canvas_inpaint(self.request())
        self.model.release.set()
        self.wait_until(lambda: len(self.completed) == 1)
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], 21)
        np.testing.assert_array_equal(self.project.inpainted_array, 20)
        self.assertFalse(self.failed)

    def test_failure_on_old_page_does_not_cancel_new_request(self) -> None:
        self.model.release.clear()
        self.model.fail = True
        with patch.object(M, 'create_error_dialog'):
            self.manager.canvas_inpaint(self.request())
            self.wait_until(self.model.started.is_set)
            self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
            self.manager.handle_page_changed()
            self.manager.canvas_inpaint(self.request())
            self.model.release.set()
            self.wait_until(lambda: len(self.completed) == 1)
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], 21)
        self.assertFalse(self.failed)

    def test_preparation_failure_notifies_canvas_and_allows_retry(self) -> None:
        thread = self.manager.inpaint_thread
        thread.module = None
        with (
            patch.object(thread, '_prepare_module_class', side_effect=ValueError('prepare failed')),
            patch.object(self.manager, '_missing_module_requirements_for_modules', return_value=[]),
            patch.object(M, 'create_error_dialog'),
        ):
            self.manager.canvas_inpaint(self.request())
            self.wait_until(lambda: len(self.failed) == 1)
        self.wait_until(lambda: not thread.isRunning())
        thread.module = self.model
        self.manager.canvas_inpaint(self.request())
        self.wait_until(lambda: len(self.completed) == 1)
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], 11)


if __name__ == '__main__':
    unittest.main()

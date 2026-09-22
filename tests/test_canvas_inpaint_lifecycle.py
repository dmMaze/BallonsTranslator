import os
import io
import copy
import threading
import time
import unittest
from typing import Callable
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtTest import QSignalSpy, QTest
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.custom_widget import ImgtransProgressMessageBox
from ballontranslator.ui import module_manager as M
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_profile, sync_codex_profile
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

    def test_codex_canvas_recovers_after_pipeline_stop_and_cancels_old_page(self) -> None:
        from ballontranslator.modules import codex
        from ballontranslator.modules.exceptions import LLMRequestStopped
        from ballontranslator.modules.inpaint.inpaint_llm import LLMInpaint

        account = codex.CodexAccount()
        account._loaded = True
        account._credentials = {
            'access_token': 'test-access', 'refresh_token': 'test-refresh',
            'account_id': 'test-account', 'expires_at': time.time() + 3600,
        }
        profile = default_profile('Codex')
        sync_codex_profile(profile, {'text-model': {'modalities': ['text'], 'efforts': []}})
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = profile.id
        pcfg.module.inpainter = 'LLMInpaint'
        inpainter = LLMInpaint()
        inpainter.params = copy.deepcopy(inpainter.params)
        inpainter.set_param_value('delay', 0)
        inpainter.set_param_value('max requests per minute', 0)
        previous_stop = threading.Event()
        previous_stop.set()
        inpainter.set_stop_event(previous_stop)
        self.manager.inpaint_thread.module = inpainter
        self.manager.inpaint_thread.pipeline_stop_event = previous_stop
        entered = threading.Event()
        cancelled = threading.Event()
        png = io.BytesIO()
        Image.new('RGB', (8, 8), (40, 50, 60)).save(png, format='PNG')

        def request(model, prompt, image, mask, stop_event, **kwargs) -> bytes:
            with Image.open(io.BytesIO(image)) as sent:
                first_page = sent.getpixel((0, 0)) == (10, 10, 10)
            if first_page:
                entered.set()
                if not stop_event.wait(3):
                    raise TimeoutError('Page change did not cancel the image request.')
                cancelled.set()
                raise LLMRequestStopped()
            self.assertFalse(stop_event.is_set())
            return png.getvalue()

        with patch.object(codex, 'account', account), \
                patch.object(codex, 'request_image', side_effect=request):
            self.manager.canvas_inpaint(self.request())
            self.wait_until(entered.is_set)
            self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
            self.manager.handle_page_changed()
            self.manager.canvas_inpaint(self.request())
            self.wait_until(lambda: len(self.completed) == 1)

        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.failed)
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], np.full((8, 8, 3), (40, 50, 60), np.uint8))
        np.testing.assert_array_equal(self.project.inpainted_array, 20)


if __name__ == '__main__':
    unittest.main()

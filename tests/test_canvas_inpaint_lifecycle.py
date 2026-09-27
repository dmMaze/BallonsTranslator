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
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawing_commands import InpaintUndoCommand
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.custom_widget import ImgtransProgressMessageBox
from ballontranslator.ui import module_manager as M
from ballontranslator.modules.inpaint.inpaint_llm import LLMInpaint
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, default_profile, sync_codex_profile
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
        self.old_draw_config = pcfg.drawpanel.copy()
        pcfg.module.inpainter = ControlledInpainter.name
        pcfg.drawpanel.inpainter = ControlledInpainter.name
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
        self.manager.imgtrans_thread.wait(3000)
        self.app.processEvents()
        self.manager.deleteLater()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        pcfg.module.merge(self.old_config)
        pcfg.drawpanel.merge(self.old_draw_config)

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
        with self.assertLogs(M.LOGGER, level='INFO') as logs:
            self.manager.canvas_inpaint(self.request())
            self.wait_until(self.model.started.is_set)
            self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
            self.manager.handle_page_changed()
            self.manager.canvas_inpaint(self.request())
            self.model.release.set()
            self.wait_until(lambda: len(self.completed) == 1)
            self.wait_until(lambda: not self.manager.inpaint_thread.isRunning())
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], 21)
        np.testing.assert_array_equal(self.project.inpainted_array, 20)
        self.assertFalse(self.failed)
        self.assertTrue(any('cancellation requested: page changed' in line for line in logs.output))
        self.assertTrue(any('result discarded: page changed' in line for line in logs.output))

    def test_failure_on_old_page_does_not_cancel_new_request(self) -> None:
        self.model.release.clear()
        self.model.fail = True
        with patch.object(M, 'create_error_dialog') as dialog, self.assertLogs(M.LOGGER, level='DEBUG') as logs:
            self.manager.canvas_inpaint(self.request())
            self.wait_until(self.model.started.is_set)
            self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
            self.manager.handle_page_changed()
            self.manager.canvas_inpaint(self.request())
            self.model.release.set()
            self.wait_until(lambda: len(self.completed) == 1)
            self.wait_until(lambda: not self.manager.inpaint_thread.isRunning())
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], 21)
        self.assertFalse(self.failed)
        dialog.assert_called_once()
        failures = [record for record in logs.records if 'Draw inpaint failed:' in record.getMessage()]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].levelname, 'DEBUG')
        self.assertIn('elapsed=', failures[0].getMessage())

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

    def test_queued_draw_snapshot_survives_selection_and_profile_edits(self) -> None:
        profile = default_profile('OpenRouter')
        profile.image_model = 'profile-model'
        profile.image_prompt = 'Profile prompt.'
        profile.api_key = 'test-secret-never-log'
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = 'run-profile'
        self.model.release.clear()
        self.manager.canvas_inpaint(self.request())
        self.wait_until(self.model.started.is_set)
        pcfg.drawpanel.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpaint_llm_id = profile.id
        pcfg.drawpanel.inpaint_llm_model = 'gpt-draw → gpt-image-2'
        pcfg.drawpanel.inpaint_prompt_override = '  Draw prompt.\n'
        with (
            patch.object(self.manager.inpaint_thread, '_prepare_module_class',
                         side_effect=lambda name: LLMInpaint if name == 'LLMInpaint' else ControlledInpainter),
            patch.object(self.manager, '_missing_module_requirements_for_modules', return_value=[]),
            patch.object(LLMInpaint, '_request_inpaint', side_effect=lambda profile, img, **kwargs: img + 5) as request,
            self.assertLogs(M.LOGGER, level='INFO') as logs,
        ):
            queued_request = self.request()
            queued_request['use_mask'] = False
            queued_request['mask'][:4] = 0
            self.manager.canvas_inpaint(queued_request)
            queued_request['use_mask'] = True
            profile.image_prompt = 'Changed profile prompt.'
            profile.image_model = 'gpt-changed → gpt-image-other'
            pcfg.drawpanel.inpainter = ControlledInpainter.name
            pcfg.drawpanel.inpaint_llm_model = 'changed-model'
            pcfg.drawpanel.inpaint_prompt_override = 'Changed override.'
            self.model.release.set()
            self.wait_until(lambda: len(self.completed) == 2)
            snapshot = request.call_args.args[0]
            self.assertIsNot(snapshot, profile)
            self.assertEqual(snapshot.image_model, 'gpt-draw → gpt-image-2')
            self.assertEqual(snapshot.image_prompt, 'Draw prompt.')
            self.assertIsNone(request.call_args.kwargs['mask'])
            self.assertEqual(pcfg.module.inpainter, ControlledInpainter.name)
            self.assertEqual(pcfg.module.inpaint_llm_id, 'run-profile')
            np.testing.assert_array_equal(self.completed[1][0]['inpainted'], 15)
            np.testing.assert_array_equal(self.completed[1][0]['mask'], 255)
            np.testing.assert_array_equal(queued_request['mask'][:4], 0)
            self.manager.canvas_inpaint(self.request())
            self.wait_until(lambda: len(self.completed) == 3)
            self.wait_until(lambda: not self.manager.inpaint_thread.isRunning())
            np.testing.assert_array_equal(self.completed[2][0]['inpainted'], 11)
        self.assertEqual(profile.image_model, 'gpt-changed → gpt-image-other')
        log_text = '\n'.join(logs.output)
        for event in ('submitted', 'started', 'completed'):
            self.assertIn('Draw inpaint ' + event, log_text)
        for value in ("backend='openai'", "model='gpt-draw → gpt-image-2'", 'rectangle=[0, 0, 8, 8]', 'use_mask=False', 'elapsed='):
            self.assertIn(value, log_text)
        for secret in ('test-secret-never-log', 'Profile prompt.', 'Draw prompt.', 'Changed override.'):
            self.assertNotIn(secret, log_text)

    def test_whitespace_draw_prompt_uses_profile_prompt(self) -> None:
        profile = default_profile('OpenRouter')
        profile.image_model = 'profile-model'
        profile.image_prompt = 'Profile prompt.'
        pcfg.module.llm_profiles = [profile]
        pcfg.drawpanel.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpaint_llm_id = profile.id
        pcfg.drawpanel.inpaint_llm_model = ''
        pcfg.drawpanel.inpaint_prompt_override = ' \t\n'
        self.manager.inpaint_thread.module = LLMInpaint()
        with patch.object(LLMInpaint, '_request_inpaint', side_effect=lambda profile, img, **kwargs: img) as request:
            self.manager.canvas_inpaint(self.request())
            self.wait_until(lambda: len(self.completed) == 1)
        snapshot = request.call_args.args[0]
        self.assertEqual(snapshot.image_prompt, 'Profile prompt.')
        self.assertEqual(snapshot.image_model, 'profile-model')

    def test_full_rectangle_mask_reaches_native_module_without_llm_kwargs(self) -> None:
        request = self.request()
        request['mask'][:4] = 0
        request['use_mask'] = False
        with patch.object(self.model, 'inpaint', wraps=self.model.inpaint) as inpaint:
            self.manager.canvas_inpaint(request)
            self.wait_until(lambda: len(self.completed) == 1)
        self.assertEqual(inpaint.call_args.kwargs, {})
        np.testing.assert_array_equal(inpaint.call_args.args[1], 255)
        np.testing.assert_array_equal(self.completed[0][0]['mask'], 255)
        np.testing.assert_array_equal(request['mask'][:4], 0)

    def test_unmasked_llm_rectangle_preserves_alpha_before_base_compositing(self) -> None:
        profile = default_profile('OpenRouter')
        profile.image_model = 'draw-model'
        pcfg.module.llm_profiles = [profile]
        pcfg.drawpanel.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpaint_llm_id = profile.id
        self.manager.inpaint_thread.module = LLMInpaint()
        image = np.zeros((8, 8, 4), dtype=np.uint8)
        image[:, :, 3] = 20
        image[3:5, 3:5, 3] = 230
        self.project.inpainted_array = image
        request = self.request()
        request['mask'][:] = 0
        request['mask'][3:5, 3:5] = 255
        request['use_mask'] = False
        with patch.object(LLMInpaint, '_request_inpaint',
                          side_effect=lambda profile, img, **kwargs: np.full_like(img, 33)) as provider:
            self.manager.canvas_inpaint(request)
            self.wait_until(lambda: len(self.completed) == 1)
        self.assertIsNone(provider.call_args.kwargs['mask'])
        result = self.completed[0][0]
        np.testing.assert_array_equal(result['inpainted'][:, :, :3], 33)
        np.testing.assert_array_equal(result['inpainted'][:, :, 3], image[:, :, 3])
        np.testing.assert_array_equal(result['mask'], 255)

    def test_deleted_draw_profile_fails_without_using_another_provider(self) -> None:
        pcfg.module.llm_profiles = [default_profile('OpenRouter')]
        pcfg.drawpanel.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpaint_llm_id = 'deleted-profile'
        with patch.object(M, '_show_llm_user_action_required_dialog') as dialog:
            self.manager.canvas_inpaint(self.request())
        self.assertEqual(len(self.failed), 1)
        self.assertFalse(self.completed)
        self.assertFalse(self.manager.inpaint_thread.isRunning())
        self.assertIn('drawing LLM profile is unavailable', str(dialog.call_args.args[0]))

    def test_unsigned_draw_codex_uses_existing_signin_dialog(self) -> None:
        from ballontranslator.modules import codex
        from ballontranslator.ui import codex_account

        account = codex.CodexAccount()
        account._loaded = True
        profile = default_codex_profile()
        pcfg.module.llm_profiles = [profile]
        pcfg.drawpanel.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpaint_llm_id = profile.id
        self.manager.inpaint_thread.module = LLMInpaint()
        with patch.object(codex, 'account', account), \
                patch.object(codex, 'request_image') as request, \
                patch.object(codex_account, 'show_codex_sign_in_required') as dialog:
            self.manager.canvas_inpaint(self.request())
            self.wait_until(lambda: len(self.failed) == 1)
            request.assert_not_called()
            dialog.assert_called_once()
        self.assertEqual(pcfg.module.inpainter, ControlledInpainter.name)

    def test_run_waits_for_canvas_before_preparing_its_own_inpainter(self) -> None:
        self.project.pages = {'page.png': []}
        pcfg.module.enable_detect = pcfg.module.enable_ocr = pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = True
        pcfg.module.inpainter = 'LLMInpaint'
        self.model.release.clear()
        self.manager.canvas_inpaint(self.request())
        self.wait_until(self.model.started.is_set)
        with (
            patch.object(self.manager.inpaint_thread, '_prepare_module_class', return_value=LLMInpaint) as prepare,
            patch.object(self.manager, '_missing_module_requirements_for_modules', return_value=[]),
            patch.object(self.manager, '_startImgtransPipeline') as start_run,
        ):
            self.manager.runImgtransPipeline()
            QTest.qWait(20)
            prepare.assert_not_called()
            start_run.assert_not_called()
            self.model.release.set()
            self.wait_until(lambda: start_run.call_count == 1)
            self.assertFalse(self.manager.inpaint_thread.isRunning())
            self.assertEqual(self.manager.inpainter.name, 'LLMInpaint')
        self.assertEqual(pcfg.drawpanel.inpainter, ControlledInpainter.name)
        self.assertEqual(pcfg.module.inpainter, 'LLMInpaint')

    def test_draw_waits_for_run_before_swapping_the_shared_inpainter(self) -> None:
        profile = default_profile('OpenRouter')
        profile.image_model = 'draw-model'
        pcfg.module.llm_profiles = [profile]
        pcfg.drawpanel.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpaint_llm_id = profile.id
        self.model.release.clear()
        run_results = []
        self.manager.imgtrans_thread.job = lambda: run_results.append(
            self.manager.imgtrans_thread.inpainter.inpaint(self.project.inpainted_array, self.request()['mask'])
        )
        self.manager.imgtrans_thread.start()
        self.wait_until(self.model.started.is_set)
        with (
            patch.object(self.manager.inpaint_thread, '_prepare_module_class', return_value=LLMInpaint) as prepare,
            patch.object(self.manager, '_missing_module_requirements_for_modules', return_value=[]),
            patch.object(LLMInpaint, '_request_inpaint', side_effect=lambda profile, img, **kwargs: img + 5),
        ):
            self.manager.canvas_inpaint(self.request())
            QTest.qWait(20)
            prepare.assert_not_called()
            self.assertFalse(self.completed)
            self.model.release.set()
            self.wait_until(lambda: len(self.completed) == 1)
        np.testing.assert_array_equal(run_results[0], 11)
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], 15)
        self.assertEqual(pcfg.module.inpainter, ControlledInpainter.name)

    def test_run_waits_even_when_canvas_module_is_already_loaded(self) -> None:
        self.project.pages = {'page.png': []}
        pcfg.module.enable_detect = pcfg.module.enable_ocr = pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = True
        self.model.release.clear()
        self.manager.canvas_inpaint(self.request())
        self.wait_until(self.model.started.is_set)
        with patch.object(self.manager, '_missing_module_requirements_for_modules', return_value=[]), \
                patch.object(self.manager, '_startImgtransPipeline') as start_run:
            self.manager.runImgtransPipeline()
            QTest.qWait(20)
            start_run.assert_not_called()
            self.model.release.set()
            self.wait_until(lambda: start_run.call_count == 1)
        self.assertFalse(self.manager.inpaint_thread.isRunning())
        self.assertIs(self.manager.inpainter, self.model)

    def test_page_change_discards_draw_queued_behind_run(self) -> None:
        self.model.release.clear()
        self.manager.imgtrans_thread.job = lambda: self.model.inpaint(
            self.project.inpainted_array, self.request()['mask']
        )
        self.manager.imgtrans_thread.start()
        self.wait_until(self.model.started.is_set)
        self.manager.canvas_inpaint(self.request())
        self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
        self.manager.handle_page_changed()
        self.assertFalse(self.manager.imgtrans_thread.isStopRequested())
        self.model.release.set()
        self.wait_until(lambda: not self.manager.imgtrans_thread.isRunning())
        self.app.processEvents()
        self.assertFalse(self.completed)
        self.assertIsNone(self.manager._pending_canvas_inpaint)

    def test_same_page_reload_discards_queued_draw_before_module_preparation(self) -> None:
        self.model.release.clear()
        self.manager.imgtrans_thread.job = lambda: self.model.inpaint(
            self.project.inpainted_array, self.request()['mask']
        )
        self.manager.imgtrans_thread.start()
        self.wait_until(self.model.started.is_set)
        self.manager.canvas_inpaint(self.request())
        # The current-page RUN completion reloads arrays without a page-change signal.
        self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
        with patch.object(self.manager, '_prepare_modules_then', side_effect=AssertionError('Stale request prepared')):
            self.model.release.set()
            self.wait_until(lambda: len(self.failed) == 1)
        self.assertFalse(self.completed)
        self.assertIsNone(self.manager._pending_canvas_inpaint)
        np.testing.assert_array_equal(self.project.inpainted_array, 20)

    def test_source_edit_during_module_preparation_discards_queued_draw(self) -> None:
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
                self.project.inpainted_array[2:4, 2:4] = 20
            finally:
                release.set()
            self.wait_until(lambda: len(self.failed) == 1)
        self.assertFalse(self.completed)
        self.assertFalse(thread.module.started.is_set())

    def test_late_result_cannot_overwrite_real_canvas_undo(self) -> None:
        canvas = Canvas()
        canvas.imgtrans_proj = self.project
        self.project.mask_array = np.zeros((8, 8), np.uint8)
        panel = DrawingPanel(canvas, self.window)
        panel.initDLModule(self.manager)
        try:
            with patch.object(canvas, 'updateLayers'):
                canvas.push_undo_command(InpaintUndoCommand(
                    canvas, np.full((8, 8, 3), 20, np.uint8),
                    np.full((8, 8), 255, np.uint8), [0, 0, 8, 8],
                ))
                self.model.release.clear()
                self.manager.canvas_inpaint(self.request())
                self.wait_until(self.model.started.is_set)
                canvas.undo()
                self.model.release.set()
                self.wait_until(lambda: len(self.failed) == 1)
                self.assertFalse(self.completed)
                np.testing.assert_array_equal(self.project.inpainted_array, 10)
                np.testing.assert_array_equal(self.project.mask_array, 0)
                self.assertEqual(canvas.draw_undo_stack.index(), 0)
                canvas.redo()
                np.testing.assert_array_equal(self.project.inpainted_array, 20)
        finally:
            self.model.release.set()
            canvas.deleteLater()

    def test_outside_crop_edit_and_mutating_worker_keep_rgba_result_valid(self) -> None:
        self.project.inpainted_array = np.full((8, 8, 4), 10, np.uint8)
        self.project.mask_array = np.zeros((8, 8), np.uint8)
        canvas = Canvas()
        canvas.imgtrans_proj = self.project
        panel = DrawingPanel(canvas, self.window)
        panel.initDLModule(self.manager)
        request = {
            'img': self.project.inpainted_array[2:6, 2:6],
            'mask': np.full((4, 4), 255, np.uint8),
            'inpaint_rect': [2, 2, 6, 6],
        }
        self.model.release.clear()

        def mutate_input(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            self.model.started.set()
            self.model.release.wait(3)
            image[..., :3] += 1
            return image

        try:
            with patch.object(self.model, 'inpaint', side_effect=mutate_input), patch.object(canvas, 'updateLayers'):
                self.manager.canvas_inpaint(request)
                self.wait_until(self.model.started.is_set)
                self.project.inpainted_array[0, 0] = 99
                self.model.release.set()
                self.wait_until(lambda: len(self.completed) == 1)
                self.assertFalse(self.failed)
                np.testing.assert_array_equal(self.project.inpainted_array[2:6, 2:6, :3], 11)
                np.testing.assert_array_equal(self.project.inpainted_array[2:6, 2:6, 3], 10)
                np.testing.assert_array_equal(self.project.inpainted_array[0, 0], 99)
                canvas.undo()
                np.testing.assert_array_equal(self.project.inpainted_array[2:6, 2:6], 10)
                np.testing.assert_array_equal(self.project.inpainted_array[0, 0], 99)
        finally:
            self.model.release.set()
            canvas.deleteLater()

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
        profile = default_codex_profile()
        sync_codex_profile(profile, {'text-model': {'modalities': ['text'], 'efforts': []}})
        pcfg.module.llm_profiles = [profile]
        pcfg.module.inpaint_llm_id = profile.id
        pcfg.module.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpainter = 'LLMInpaint'
        pcfg.drawpanel.inpaint_llm_id = profile.id
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
                patch.object(codex, 'request_image', side_effect=request), \
                self.assertLogs(M.LOGGER, level='INFO') as logs:
            self.manager.canvas_inpaint(self.request())
            self.wait_until(entered.is_set)
            self.project.inpainted_array = np.full((8, 8, 3), 20, np.uint8)
            self.manager.handle_page_changed()
            self.manager.canvas_inpaint(self.request())
            self.wait_until(lambda: len(self.completed) == 1)
            self.wait_until(lambda: not self.manager.inpaint_thread.isRunning())

        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.failed)
        self.assertTrue(any('Draw inpaint cancelled:' in line and "backend='codex'" in line
                            and 'elapsed=' in line for line in logs.output))
        np.testing.assert_array_equal(self.completed[0][0]['inpainted'], np.full((8, 8, 3), (40, 50, 60), np.uint8))
        np.testing.assert_array_equal(self.project.inpainted_array, 20)


if __name__ == '__main__':
    unittest.main()

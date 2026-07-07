import threading
import unittest
from unittest import mock

from ballontranslator.modules.translators.exceptions import LLMApiKeyRequiredError
from ballontranslator.ui import module_manager
from ballontranslator.utils import shared


class FakeModuleSignal:
    def connect(self, _slot):
        pass


class FakeTranslateThread:
    module_thread_stopped = FakeModuleSignal()
    finished = FakeModuleSignal()

    def __init__(self, translator):
        self.module = translator
        self.translator = translator

    def isRunning(self):
        return False


class MissingKeyTranslator:
    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def translate_textblk_lst(self, _blk_list):
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class LLMKeyDialogDedupTest(unittest.TestCase):
    def setUp(self):
        self._old_headless = shared.HEADLESS
        self._old_emit = shared.show_llm_key_dialog_in_mainthread
        self.calls = []
        shared.HEADLESS = False
        shared.show_llm_key_dialog_in_mainthread = lambda profile_id, profile_name: self.calls.append(
            (profile_id, profile_name)
        )
        module_manager._reset_llm_key_required_dialogs()

    def tearDown(self):
        shared.HEADLESS = self._old_headless
        shared.show_llm_key_dialog_in_mainthread = self._old_emit
        module_manager._reset_llm_key_required_dialogs()

    def test_missing_llm_key_dialog_emits_once_until_reset(self):
        error = LLMApiKeyRequiredError('profile-1', 'Profile 1')

        module_manager._show_llm_key_required_dialog(error)
        module_manager._show_llm_key_required_dialog(error)

        self.assertEqual(self.calls, [('profile-1', 'Profile 1')])

        module_manager._reset_llm_key_required_dialogs()
        module_manager._show_llm_key_required_dialog(error)

        self.assertEqual(
            self.calls,
            [('profile-1', 'Profile 1'), ('profile-1', 'Profile 1')],
        )

    def test_missing_llm_key_stops_imgtrans_thread(self):
        translator = MissingKeyTranslator()
        thread = module_manager.ImgtransThread(
            None,
            None,
            FakeTranslateThread(translator),
            None,
        )

        self.assertFalse(thread.isStopRequested())

        result = thread._translate_textblocks([])

        self.assertFalse(result)
        self.assertTrue(thread.isStopRequested())
        self.assertIs(translator.stop_event, thread.stop_event)

    def test_module_thread_fatal_llm_key_error_sets_pipeline_stop_event(self):
        stop_event = threading.Event()
        thread = module_manager.ModuleThread('translator', None)
        thread.pipeline_stop_event = stop_event
        thread.job = lambda: (_ for _ in ()).throw(LLMApiKeyRequiredError('profile-1', 'Profile 1'))

        thread.run()

        self.assertTrue(stop_event.is_set())

    def test_standalone_translate_page_clears_stale_pipeline_stop_event(self):
        thread = module_manager.TranslateThread()
        thread.pipeline_stop_event = threading.Event()

        with mock.patch.object(module_manager.TranslateThread, 'start', lambda self: None):
            thread.translatePage({'page-1': []}, 'page-1')

        self.assertIsNone(thread.pipeline_stop_event)


if __name__ == '__main__':
    unittest.main()

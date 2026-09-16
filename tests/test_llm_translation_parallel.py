import os
import threading
import unittest
from copy import deepcopy
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from _llm_translation_test_support import LLMTranslationTestMixin
from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.llm_chat import LLMChatResult
from ballontranslator.modules.llm_codex import CodexBusyError, CodexRequestError
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.ui import module_manager
from ballontranslator.ui.module_parse_widgets import ParamWidget
from ballontranslator.utils.config import LLMTranslateContext, RunStatus, pcfg


class LLMParallelTranslationTest(LLMTranslationTestMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        super().setUp()
        self.profile.transport = 'Codex App Server'
        self.param_backup = {
            key: self.translator.get_param_value(key)
            for key in ('codex parallel requests', 'delay', 'max requests per minute')
        }
        self.translator.set_param_value('codex parallel requests', '2')
        self.translator.set_param_value('delay', 0)
        self.translator.set_param_value('max requests per minute', 0)
        self.translator.set_param_value('retry attempts', 1)
        self.addCleanup(self._restore_params)
        profile_patch = mock.patch.object(
            type(self.translator), 'profile',
            new_callable=mock.PropertyMock, return_value=self.profile,
        )
        profile_patch.start()
        self.addCleanup(profile_patch.stop)

    def _restore_params(self) -> None:
        for key, value in self.param_backup.items():
            self.translator.set_param_value(key, value)

    def _worker(self, count: int = 3) -> module_manager.TranslateThread:
        worker = module_manager.TranslateThread()
        worker.translator = self.translator
        worker.num_process_pages = count
        worker.initImgtransPipeline(self._project(count), threading.Event())
        for key in worker.imgtrans_proj.pages:
            worker.push_pagekey_queue(key)
        return worker

    def test_parallel_requests_overlap_but_commit_pages_and_memory_in_order(self) -> None:
        worker = self._worker()
        second_finished = threading.Event()
        committed = []
        pcfg.module.llm_translate_summary_memory = True
        original_finish = self.translator.on_page_translation_finished

        def finish(project, page_key):
            committed.append(page_key)
            original_finish(project, page_key)

        def request(_profile, args, _stop):
            if 'response_format' not in args:
                self.assertEqual(committed, list(worker.imgtrans_proj.pages))
                return LLMChatResult('All three page summaries.', {'total_tokens': 1})
            prompt = str(args['messages'][-1]['content'])
            if 'source-1' in prompt:
                self.assertTrue(second_finished.wait(3), 'requests were serialized')
                number = 1
            elif 'source-2' in prompt:
                number = 2
                second_finished.set()
            else:
                number = 3
            return LLMChatResult(
                '{"translations":[{"id":1,"translation":"translated-%d"}],'
                '"page_summary":"Summary %d"}' % (number, number),
                {'total_tokens': 1},
            )

        with mock.patch.object(self.translator, 'on_page_translation_finished', side_effect=finish), \
                mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=request):
            worker._run_translate_pipeline()

        self.assertEqual(worker.finished_counter, 3)
        self.assertEqual(committed, list(worker.imgtrans_proj.pages))
        for index, (key, blocks) in enumerate(worker.imgtrans_proj.pages.items(), 1):
            self.assertEqual(blocks[0].translation, f'translated-{index}')
            self.assertEqual(worker.imgtrans_proj.get_llm_visual_summary(key)['text'], f'Summary {index}')
        self.assertEqual(worker.imgtrans_proj.get_llm_compact_memory()['covered_pages'], committed)
        self.assertEqual(self.translator.usage_totals.requests, 4)
        self.assertEqual(self.translator.usage_totals.total_tokens, 4)

    def test_fatal_later_request_cancels_earlier_request_and_queued_pages(self) -> None:
        worker = self._worker()
        first_started = threading.Event()
        first_closed = threading.Event()
        calls = []

        def request(_profile, args, stop):
            calls.append(args)
            if 'source-1' in str(args['messages']):
                first_started.set()
                try:
                    self.assertTrue(stop.wait(3), 'later failure did not stop active request')
                    raise LLMRequestStopped()
                finally:
                    first_closed.set()
            self.assertTrue(first_started.wait(3))
            raise CodexRequestError('needs attention')

        with mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=request), \
                mock.patch.object(module_manager, '_show_llm_user_action_required_dialog') as dialog:
            worker._run_translate_pipeline()

        self.assertTrue(worker.pipeline_stop_event.is_set())
        self.assertTrue(first_closed.is_set())
        self.assertEqual(len(calls), 2)
        self.assertEqual(worker.finished_counter, 0)
        dialog.assert_called_once()
        for key in ('001.png', '002.png'):
            self.assertFalse(worker.imgtrans_proj._image_info[key]['finish_code'] & RunStatus.FIN_TRANSLATE)

    def test_capacity_cooldown_preserves_active_work_and_delays_new_pages(self) -> None:
        worker = self._worker()
        self.translator.set_param_value('retry attempts', 2)
        first_started = threading.Event()
        cooldown_started = threading.Event()
        both_waiting = threading.Event()
        clock = [1000.0]
        lock = threading.Lock()
        calls = {1: 0, 2: 0, 3: 0}
        waits = []

        def wait(seconds: float) -> None:
            with lock:
                waits.append(seconds)
                cooldown_started.set()
                if len(waits) == 2:
                    both_waiting.set()
            self.assertTrue(both_waiting.wait(3), 'new page bypassed the shared cooldown')
            clock[0] = 1060.0

        def request(_profile, args: dict, _stop) -> LLMChatResult:
            number = next(i for i in calls if f'source-{i}' in str(args['messages']))
            calls[number] += 1
            if number == 1:
                first_started.set()
                self.assertTrue(cooldown_started.wait(3))
            elif number == 2 and calls[number] == 1:
                self.assertTrue(first_started.wait(3))
                raise CodexBusyError('Selected model is at capacity', {'total_tokens': 2})
            else:
                self.assertGreaterEqual(clock[0], 1060.0)
            return LLMChatResult('{"1":"translated-%d"}' % number, {'total_tokens': 3})

        with mock.patch('ballontranslator.modules.llm_chat.time.monotonic', side_effect=lambda: clock[0]), \
                mock.patch('ballontranslator.modules.llm_chat.random.uniform', return_value=0), \
                mock.patch.object(self.translator, '_wait', side_effect=wait), \
                mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=request):
            worker._run_translate_pipeline()

        self.assertEqual(waits, [60.0, 60.0])
        self.assertEqual(calls, {1: 1, 2: 2, 3: 1})
        self.assertEqual(worker.finished_counter, 3)
        self.assertFalse(worker.pipeline_stop_event.is_set())
        self.assertEqual(self.translator.usage_totals.requests, 4)
        self.assertEqual(self.translator.usage_totals.total_tokens, 11)
        for index, blocks in enumerate(worker.imgtrans_proj.pages.values(), 1):
            self.assertEqual(blocks[0].translation, f'translated-{index}')

    def test_history_and_other_transports_keep_sequential_execution(self) -> None:
        self.assertEqual(self.translator.parallel_request_workers(), 2)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        self.assertEqual(self.translator.parallel_request_workers(), 1)
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        self.profile.transport = 'OpenAI Compatible'
        self.assertEqual(self.translator.parallel_request_workers(), 1)

    def test_stopping_during_shared_rpm_wait_joins_all_workers(self) -> None:
        worker = self._worker()
        self.translator.set_param_value('max requests per minute', 1)
        throttled = threading.Event()

        def wait(_seconds):
            throttled.set()
            self.assertTrue(worker.pipeline_stop_event.wait(3))
            raise LLMRequestStopped()

        def request(_profile, _args, stop):
            self.assertTrue(throttled.wait(3), 'RPM was not shared across workers')
            stop.set()
            raise LLMRequestStopped()

        with mock.patch.object(self.translator, '_wait', side_effect=wait), \
                mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=request) as provider:
            worker._run_translate_pipeline()

        self.assertEqual(provider.call_count, 1)
        self.assertEqual(self.translator.usage_totals.requests, 1)
        self.assertEqual(worker.finished_counter, 0)
        self.assertFalse(any(t.name.startswith('codex-translate') for t in threading.enumerate()))

    def test_request_starts_share_delay_while_provider_calls_overlap(self) -> None:
        worker = self._worker(2)
        self.translator.set_param_value('delay', 0.3)
        clock = [10.0]
        starts = []
        overlap = threading.Barrier(2)

        def wait(seconds):
            clock[0] += seconds

        def request(_profile, _args, _stop):
            starts.append(clock[0])
            overlap.wait(3)
            return LLMChatResult('{"1":"translated"}')

        with mock.patch('ballontranslator.modules.llm_chat.time.time', side_effect=lambda: clock[0]), \
                mock.patch.object(self.translator, '_wait', side_effect=wait), \
                mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=request):
            worker._run_translate_pipeline()

        self.assertEqual(starts, [10.0, 10.3])
        self.assertEqual(worker.finished_counter, 2)

    def test_invalid_saved_worker_count_falls_back_without_losing_other_settings(self) -> None:
        for value in ('0', '-1', '1.5', '1e2', '', 'broken', None, True, [], {}, 1.5, '9' * 5000):
            self.translator.set_param_value('codex parallel requests', value, convert_dtype=False)
            with self.assertLogs(self.translator.logger, level='WARNING'):
                self.assertEqual(self.translator.parallel_request_workers(), 1)
            self.assertEqual(self.translator.get_param_value('codex parallel requests'), '1')
            self.assertEqual(self.translator.get_param_value('retry attempts'), 1)

    def test_saved_positive_counts_and_old_selector_settings_still_load(self) -> None:
        for value, expected in (
            ('1', 1), ('4', 4), ('6', 6), ('8', 8), ('16', 16), (8, 8), (' 6 ', 6),
            ({'type': 'selector', 'options': ['1', '2', '3', '4'], 'value': '3'}, 3),
        ):
            with self.subTest(value=value):
                translator = LLMTranslator('日本語', '简体中文', **{'codex parallel requests': value})
                self.assertEqual(translator.parallel_request_workers(), expected)

    def test_manual_entry_starts_eight_concurrent_requests(self) -> None:
        panel = ParamWidget(deepcopy(self.translator.params))
        self.addCleanup(panel.deleteLater)
        edits = []
        panel.paramwidget_edited.connect(lambda key, content: edits.append((key, content)))
        panel.show()
        self.app.processEvents()
        editor = panel.param_widgets['codex parallel requests']
        editor.setFocus()
        editor.selectAll()
        QTest.keyClicks(editor, '8')
        QTest.keyClick(editor, Qt.Key.Key_Return)
        panel.hide()
        self.assertEqual(edits, [('codex parallel requests', {'content': '8'})])
        self.translator.updateParam(edits[0][0], edits[0][1]['content'])
        worker = self._worker(8)
        overlap = threading.Barrier(8)

        def request(_profile, _args, _stop) -> LLMChatResult:
            overlap.wait(5)
            return LLMChatResult('{"1":"translated"}')

        with mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=request) as requests:
            worker._run_translate_pipeline()
        self.assertEqual(requests.call_count, 8)
        self.assertEqual(worker.finished_counter, 8)
        self.assertFalse(worker.pipeline_stop_event.is_set())

    def test_failed_page_stays_incomplete_while_following_pages_continue(self) -> None:
        worker = self._worker()

        def request(_profile, args, _stop):
            if 'source-2' in str(args['messages']):
                return LLMChatResult('invalid response')
            return LLMChatResult('{"1":"translated"}')

        with mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=request), \
                mock.patch.object(module_manager, '_create_page_error_dialog') as dialog:
            worker._run_translate_pipeline()

        self.assertEqual(worker.finished_counter, 3)
        self.assertFalse(worker.pipeline_stop_event.is_set())
        self.assertEqual([
            bool(info['finish_code'] & RunStatus.FIN_TRANSLATE)
            for info in worker.imgtrans_proj._image_info.values()
        ], [True, False, True])
        dialog.assert_called_once()

    def test_translation_only_pipeline_waits_for_translation_completion(self) -> None:
        worker = self._worker()
        pipeline = module_manager.ImgtransThread(None, None, worker, None)
        pipeline.imgtrans_proj = worker.imgtrans_proj
        pipeline.parallel_trans = True
        pipeline.num_pages = 3
        with mock.patch.object(pcfg.module, 'enable_ocr', False), \
                mock.patch.object(pcfg.module, 'enable_translate', True):
            self.assertFalse(pipeline.translate_finished())
            worker.finished_counter = 3
            self.assertTrue(pipeline.translate_finished())


if __name__ == '__main__':
    unittest.main()

import threading
import unittest
from unittest import mock

from _llm_translation_test_support import LLMTranslationTestMixin
from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.llm_chat import LLMChatResult
from ballontranslator.modules.llm_codex import CodexRequestError
from ballontranslator.ui import module_manager
from ballontranslator.utils.config import LLMTranslateContext, RunStatus, pcfg


class LLMParallelTranslationTest(LLMTranslationTestMixin, unittest.TestCase):
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
        for value in ('0', '5', 'broken', None, True, []):
            self.translator.set_param_value('codex parallel requests', value, convert_dtype=False)
            with self.assertLogs(self.translator.logger, level='WARNING'):
                self.assertEqual(self.translator.parallel_request_workers(), 1)
            self.assertEqual(self.translator.get_param_value('codex parallel requests'), '1')
            self.assertEqual(self.translator.get_param_value('retry attempts'), 1)

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

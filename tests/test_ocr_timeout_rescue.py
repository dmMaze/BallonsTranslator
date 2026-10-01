import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from _llm_translation_test_support import LLMTranslationTestMixin
from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.llm_chat import LLMChatResult
from ballontranslator.modules.llm_codex import CodexRequestError, CodexTimeoutError
from ballontranslator.utils.textblock import TextBlock


class TimeoutRescueTest(LLMTranslationTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        for key in ('delay', 'max requests per minute', 'timeout rescue'):
            self.addCleanup(self.translator.set_param_value, key, self.translator.get_param_value(key))
        self.profile.transport = 'Codex App Server'
        self.profile.model = 'gpt-6-luna'
        self.profile.model_options = ['gpt-6-luna']
        self.profile.thinking_level = 'none'
        self.translator.set_param_value('delay', 0)
        self.translator.set_param_value('max requests per minute', 0)
        self.translator.set_param_value('retry attempts', 3)
        self.translator.set_param_value('timeout rescue', True)
        self.enterContext(mock.patch.object(type(self.translator), 'profile', new_callable=mock.PropertyMock, return_value=self.profile))
        self.enterContext(mock.patch.object(self.translator, 'all_model_loaded', return_value=True))
        self.project = self._project(1)
        self.blocks = [
            TextBlock(text=['IC' * 40], translation='old-1'),
            TextBlock(text=['Goodbye'], translation='old-2'),
            TextBlock(text=[''], translation=''),
        ]
        self.project.pages['001.png'] = self.blocks
        self.read_image = self.enterContext(mock.patch.object(self.project, 'read_img', side_effect=AssertionError('No OCR rescan')))
        self.filter = self.enterContext(mock.patch(
            'ballontranslator.modules.ocr.jev_filter.filter_ocr_texts', return_value=['', 'Goodbye'],
        ))

    def run_page(self, responses) -> mock.MagicMock:
        with mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=responses) as request:
            self.translator.translate_textblk_lst(self.blocks, project=self.project, page_key='001.png', full_page=True)
        return request

    def test_first_timeout_cleans_once_reuses_model_and_preserves_saved_ocr(self) -> None:
        original = [b.get_text() for b in self.blocks]
        events = []

        def clean(texts, stop, source, totals):
            events.append('jev')
            self.assertEqual(list(texts), original[:2])
            self.assertTrue(source.endswith('001.png'))
            totals.update(requests=3, input_tokens=100, reported_input_requests=3,
                          calls=1, cleared_blocks=1, removed_characters=80)
            return ['', 'Goodbye']

        def response(profile, args, stop):
            events.append('codex')
            if len(events) == 1:
                raise CodexTimeoutError('first')
            return LLMChatResult('{"1":"invented","2":"再見"}', {'input_tokens': 20})

        self.filter.side_effect = clean
        request = self.run_page(response)
        self.assertEqual(events, ['codex', 'jev', 'codex'])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.translator.usage_totals.requests, 2)
        self.filter.assert_called_once()
        self.assertEqual([b.get_text() for b in self.blocks], original)
        self.assertEqual([b.translation for b in self.blocks], ['', '再見', ''])
        first, second = request.call_args_list
        self.assertIs(first.args[0], second.args[0])
        self.assertEqual(second.args[1]['model'], 'gpt-6-luna')
        self.assertNotIn('ICIC', str(second.args[1]))
        self.assertEqual(self.translator.jev_cleanup_totals['requests'], 3)
        self.assertEqual(self.translator.jev_cleanup_totals['input_tokens'], 100)
        self.read_image.assert_not_called()

    def test_successful_repetition_does_not_trigger_jev(self) -> None:
        self.run_page([LLMChatResult(json.dumps({'1': 'IC' * 40, '2': '再見'}))])
        self.filter.assert_not_called()

    def test_unchanged_cleanup_still_retries_only_once(self) -> None:
        self.filter.return_value = [b.get_text() for b in self.blocks[:2]]
        request = self.run_page([CodexTimeoutError('first'), LLMChatResult('{"1":"聲音","2":"再見"}')])
        self.assertEqual(request.call_count, 2)
        self.assertIn('ICIC', str(request.call_args_list[1].args[1]))

    def test_second_failure_has_no_provider_or_parse_retry_and_preserves_page(self) -> None:
        for failure in [
            CodexTimeoutError('again'), CodexRequestError('Selected model is at capacity'),
            CodexRequestError('out of credits'), LLMChatResult('invalid JSON'),
            LLMChatResult('{"1":"missing ID"}'),
        ]:
            with self.subTest(failure=failure):
                self.filter.reset_mock()
                original = [(b.get_text(), b.translation) for b in self.blocks]
                with mock.patch('ballontranslator.modules.llm_codex.request_codex_completion',
                                side_effect=[CodexTimeoutError('first'), failure]) as request:
                    with self.assertRaises(Exception):
                        self.translator.translate_textblk_lst(self.blocks, project=self.project, page_key='001.png', full_page=True)
                self.assertEqual(request.call_count, 2)
                self.filter.assert_called_once()
                self.assertEqual([(b.get_text(), b.translation) for b in self.blocks], original)

    def test_cancellation_during_cleanup_keeps_usage_and_prevents_retry(self) -> None:
        def cancelled(texts, stop, source, totals):
            totals.update(requests=1, input_tokens=12)
            raise LLMRequestStopped()
        self.filter.side_effect = cancelled
        with self.assertRaises(LLMRequestStopped):
            self.run_page([CodexTimeoutError('first')])
        self.assertEqual(self.translator.jev_cleanup_totals, dict(requests=1, input_tokens=12))
        self.assertEqual([b.translation for b in self.blocks], ['old-1', 'old-2', ''])

    def test_quota_cancellation_and_memory_timeout_do_not_trigger_jev(self) -> None:
        for error in [CodexRequestError('out of credits'), LLMRequestStopped()]:
            with self.subTest(error=type(error)), self.assertRaises(type(error)):
                self.run_page([error])
        with mock.patch.object(self.translator, '_snapshot_request_context', side_effect=CodexTimeoutError('memory')), self.assertRaises(CodexTimeoutError):
            self.run_page([])
        self.filter.assert_not_called()

    def test_partial_selection_and_disabled_rescue_keep_normal_timeout_retries(self) -> None:
        for partial in (False, True):
            with self.subTest(partial=partial):
                self.translator.set_param_value('timeout rescue', partial)
                result = '{"1":"聲音"}' if partial else '{"1":"聲音","2":"再見"}'
                with mock.patch.object(self.translator, '_wait'), mock.patch(
                    'ballontranslator.modules.llm_chat.time.monotonic', side_effect=range(1000, 3000, 100),
                ), mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=[
                    CodexTimeoutError('first'), LLMChatResult(result),
                ]) as request:
                    self.translator.translate_textblk_lst(
                        self.blocks[:1] if partial else self.blocks, project=self.project,
                        page_key='001.png', full_page=not partial,
                    )
                self.assertEqual(request.call_count, 2)
        self.filter.assert_not_called()

    def test_parallel_rescues_keep_page_sources_and_usage_separate(self) -> None:
        barrier = threading.Barrier(2)
        calls = threading.local()
        other = [TextBlock(text=['NOISE'], translation='old')]
        self.project.pages['002.png'] = other

        def clean(texts, stop, source, totals):
            barrier.wait(timeout=5)
            totals.update(requests=2, input_tokens=20, calls=1)
            return ['', 'Goodbye'] if source.endswith('001.png') else ['Hello']

        def response(profile, args, stop):
            calls.count = getattr(calls, 'count', 0) + 1
            if calls.count == 1:
                raise CodexTimeoutError('first')
            count = 2 if 'Goodbye' in str(args) else 1
            return LLMChatResult(json.dumps({str(i): '譯文' for i in range(1, count + 1)}))

        self.filter.side_effect = clean
        with mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=response), ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.translator.translate_textblk_lst, blocks,
                                   project=self.project, page_key=key, full_page=True)
                       for key, blocks in [('001.png', self.blocks), ('002.png', other)]]
            for future in futures:
                future.result(timeout=10)
        self.assertEqual(self.translator.jev_cleanup_totals, dict(requests=4, input_tokens=40, calls=2))
        self.assertEqual([b.translation for b in self.blocks], ['', '譯文', ''])
        self.assertEqual(other[0].translation, '譯文')
        self.assertEqual(other[0].get_text(), 'NOISE')

    def test_second_timeout_is_page_error_without_cancelling_queue(self) -> None:
        from qtpy.QtWidgets import QApplication
        from ballontranslator.ui import module_manager

        app = QApplication.instance() or QApplication([])
        worker = module_manager.TranslateThread()
        worker.translator = self.translator
        worker.pipeline_stop_event = threading.Event()
        with mock.patch('ballontranslator.modules.llm_codex.request_codex_completion', side_effect=[
            CodexTimeoutError('first'), CodexTimeoutError('again'),
        ]), mock.patch.object(module_manager, '_create_page_error_dialog'), mock.patch.object(
            module_manager, '_show_llm_user_action_required_dialog',
        ) as fatal:
            self.assertFalse(worker._translate_page(self.project, '001.png'))
        self.assertFalse(worker.pipeline_stop_event.is_set())
        fatal.assert_not_called()

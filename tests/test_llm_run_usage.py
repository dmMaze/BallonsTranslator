import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication
from ballontranslator.modules.context.token_usage import (
    LLMUsageTotals, estimated_token_cost, format_run_token_usage,
)
from ballontranslator.modules.llm_chat import LLMChatRequester
from ballontranslator.modules.ocr.base import OCRBase
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.ui.module_manager import ModuleManager
from ballontranslator.utils.config import pcfg


class LLMRunUsageTest(unittest.TestCase):
    def test_jev_input_requests_and_cost_are_separate_and_reset_for_each_run(self) -> None:
        app = QApplication.instance() or QApplication([])
        manager = ModuleManager(None)
        ocr, translator = OCRBase(), LLMTranslator('日本語', '繁體中文')
        manager.ocr_thread = SimpleNamespace(ocr=ocr)
        manager.translate_thread = SimpleNamespace(translator=translator)
        manager.imgtrans_thread = SimpleNamespace(isStopRequested=lambda: False)
        try:
            with mock.patch('ballontranslator.ui.module_manager.LOGGER.info') as log:
                translator.jev_cleanup_totals.update(requests=99, input_tokens=999, cleared_blocks=99)
                manager._begin_llm_usage_run(True, True)
                self.assertEqual(translator.jev_cleanup_totals, {})
                translator.jev_cleanup_totals.update(
                    requests=3, input_tokens=100, reported_input_requests=1,
                    cost_nano_usd=6200, estimated_cost_requests=1, reported_cost_requests=1,
                    calls=3, changed_calls=2, cleared_blocks=1, trimmed_blocks=2,
                    kept_blocks=10, removed_characters=50,
                )
                translator.usage_totals.requests = 1
                translator.usage_totals.add('gpt-6-luna', {'input_tokens': 1000, 'output_tokens': 100})
                manager._finish_llm_usage_run()
                manager._finish_llm_usage_run()
                messages = [call.args[0] for call in log.call_args_list]
                self.assertEqual(len(messages), 3)
                cleanup = next(s for s in messages if s.startswith('Jev OCR cleanup run:'))
                self.assertIn('requests=3, input_tokens=100, missing_input_usage_requests=2,', cleanup)
                self.assertIn('cost_usd=unavailable, cost_source=mixed, priced_subtotal_usd=0.000006200,', cleanup)
                self.assertIn('unpriced_requests=1,', cleanup)
                self.assertIn('changed_blocks=3 (cleared=1, trimmed=2)', cleanup)
                self.assertIn('removed_characters=50, kept_blocks=10, changed_calls=2/3', cleanup)
                for excluded in ('output_tokens', 'total_tokens', 'completion', 'estimated_cost'):
                    self.assertNotIn(excluded, cleanup)
                # Do not mix input-only Jev usage into full-token translation totals.
                total = messages[-1]
                self.assertIn('requests=1, total_tokens=1100, missing_usage_requests=0,', total)
                self.assertIn('estimated_cost_usd=0.000150,', total)

                log.reset_mock()
                manager._begin_llm_usage_run(True, False)
                manager._finish_llm_usage_run()
                log.assert_not_called()
                manager._begin_llm_usage_run(False, True)
                self.assertEqual(translator.jev_cleanup_totals, {})
                self.assertEqual(list(manager._llm_usage_totals), ['translation'])
                # Skipped filtering still reports zero API requests and removals.
                translator.jev_cleanup_totals.update(calls=1, changed_calls=0, cleared_blocks=0,
                                              trimmed_blocks=0, kept_blocks=3, removed_characters=0)
                manager.imgtrans_thread.isStopRequested = lambda: True
                manager._finish_llm_usage_run()
                manager._finish_llm_usage_run()
                log.assert_called_once()
                self.assertIn('status=stopped, requests=0, input_tokens=0,', log.call_args.args[0])
                self.assertIn('cost_usd=0.000000000, cost_source=none,', log.call_args.args[0])
                self.assertIn('unpriced_requests=0,', log.call_args.args[0])
                self.assertIn('changed_blocks=0', log.call_args.args[0])
                self.assertIn('changed_calls=0/1', log.call_args.args[0])
            with mock.patch('ballontranslator.ui.module_manager.LOGGER.info') as log:
                manager._begin_llm_usage_run(True, False)
                manager._finish_llm_usage_run()
                log.assert_not_called()
        finally:
            manager.deleteLater()
            app.processEvents()

    def test_prices_cache_reasoning_long_context_and_unknown_usage(self) -> None:
        counts = {'prompt': 1000, 'completion': 100}
        for model, cost in (('gpt-6-astra', '0.015'), ('gpt-6-sol', '0.003'),
                            ('gpt-6-luna', '0.00015'), ('gpt-5.6-sol', '0.006'),
                            ('gpt-5.6-terra', '0.0032'), ('gpt-5.6-luna', '0.00032'),
                            ('gpt-5.5', '0.008')):
            with self.subTest(model=model):
                self.assertEqual(estimated_token_cost(model, counts), Decimal(cost))
        cached = dict(counts, cache_hit=500, cache_write=100, reasoning=80)
        self.assertEqual(estimated_token_cost('gpt-5.6-sol', cached), Decimal('0.0043'))
        self.assertEqual(estimated_token_cost('gpt-5.6-sol', dict(counts, prompt=272000)), Decimal('1.09'))
        self.assertEqual(estimated_token_cost('gpt-5.6-sol', dict(counts, prompt=272001)), Decimal('2.179008'))
        self.assertIsNone(estimated_token_cost('unknown', counts))
        self.assertIsNone(estimated_token_cost('gpt-5.6-sol', {'total': 10}))
        self.assertIsNone(estimated_token_cost('gpt-5.6-sol', dict(counts, cache_hit=1001)))
        self.assertIsNone(estimated_token_cost('gpt-5.5', cached))

        totals = LLMUsageTotals(requests=4)
        totals.add('gpt-5.6-sol', {'input_tokens': 1000, 'output_tokens': 100})
        totals.add('unknown', {'total_tokens': 12})
        totals.add('gpt-5.6-sol', None)  # Fourth request raised without a report.
        report = format_run_token_usage([totals])
        self.assertIn('total_tokens=1112,', report)
        self.assertIn('missing_usage_requests=2,', report)
        self.assertIn('estimated_cost_usd=unavailable,', report)
        self.assertIn('priced_subtotal_usd=0.006000,', report)
        self.assertIn('unpriced_requests=3,', report)

    def test_run_resets_counts_waits_for_workers_and_logs_only_once(self) -> None:
        app = QApplication.instance() or QApplication([])
        manager = ModuleManager(None)
        translator, ocr = LLMChatRequester(), LLMChatRequester()
        manager.translate_thread = SimpleNamespace(translator=translator, isRunning=mock.Mock(return_value=True))
        manager.ocr_thread = SimpleNamespace(ocr=ocr)
        manager.imgtrans_thread = SimpleNamespace(isRunning=mock.Mock(return_value=False),
                                                  isStopRequested=mock.Mock(return_value=False))
        manager.progress_msgbox = SimpleNamespace(hide=mock.Mock())
        try:
            translator.usage_totals.total_tokens = 99999  # Previous run.
            manager._begin_llm_usage_run(True, True)
            for requester, requests in ((translator, 2), (ocr, 1)):
                # Translation includes a retry/summary; OCR has one request.
                requester.usage_totals.requests = requests
                for _ in range(requests):
                    requester.usage_totals.add('gpt-5.6-sol', {'input_tokens': 1000, 'output_tokens': 100})
            with mock.patch('ballontranslator.ui.module_manager.LOGGER.info') as log:
                manager.on_finish_blktrans(1, [])  # Intermediate OCR-stage signal.
                manager._finish_llm_usage_when_idle()
                log.assert_not_called()
                manager.translate_thread.isRunning.return_value = False
                manager._finish_llm_usage_when_idle()
                manager._finish_llm_usage_when_idle()
                self.assertEqual(log.call_count, 3)
                for call, prefix, tokens, cost in zip(log.call_args_list, (
                    'LLM OCR run usage:', 'LLM translation run usage:', 'LLM run usage:',
                ), (1100, 2200, 3300), ('0.006000', '0.012000', '0.018000')):
                    message = call.args[0]
                    self.assertTrue(message.startswith(prefix))
                    self.assertIn(f'total_tokens={tokens},', message)
                    self.assertIn(f'estimated_cost_usd={cost},', message)

                # A translation-only next run excludes old OCR costs, even when stopped.
                log.reset_mock()
                manager._begin_llm_usage_run(False, True)
                translator.usage_totals.requests = 1
                manager.imgtrans_thread.isStopRequested.return_value = True
                manager.on_imgtrans_thread_stopped()
                manager._finish_llm_usage_when_idle()
                self.assertEqual(log.call_count, 2)
                self.assertTrue(log.call_args_list[0].args[0].startswith('LLM translation run usage:'))
                for call in log.call_args_list:
                    self.assertNotIn('OCR', call.args[0])
                    self.assertIn('status=stopped', call.args[0])
                    self.assertIn('total_tokens=0, missing_usage_requests=1,', call.args[0])

                log.reset_mock()
                manager._begin_llm_usage_run(True, False)
                ocr.usage_totals.requests = 1
                ocr.usage_totals.add('unknown', {'total_tokens': 12})
                manager._finish_llm_usage_when_idle()
                manager._finish_llm_usage_when_idle()
                self.assertEqual(log.call_count, 2)
                self.assertTrue(log.call_args_list[0].args[0].startswith('LLM OCR run usage:'))
                for call in log.call_args_list:
                    self.assertNotIn('translation', call.args[0])
                    self.assertIn('total_tokens=12,', call.args[0])
                    self.assertIn('estimated_cost_usd=unavailable,', call.args[0])
        finally:
            manager.deleteLater()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()

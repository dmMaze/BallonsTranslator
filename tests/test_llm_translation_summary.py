import json
import unittest
from unittest import mock

import numpy as np

from _llm_translation_test_support import (
    LLMTranslationTestMixin,
    _block,
)
from ballontranslator.modules.context.translation_context import (
    PageSummary,
)
from ballontranslator.utils.config import LLMTranslateContext, pcfg


class LLMTranslationSummaryTest(
    LLMTranslationTestMixin,
    unittest.TestCase,
):
    def setUp(self) -> None:
        super().setUp()
        last_page_compaction = mock.patch.object(
            self.translator,
            '_compact_last_page_memory',
        )
        last_page_compaction.start()
        self.addCleanup(last_page_compaction.stop)

    def test_summary_persists_only_after_page_finalization(self):
        project = self._project(1)
        project.pages['001.png'][0].translation = ''
        pcfg.module.llm_translate_summary_memory = True
        response = json.dumps({
            'translations': {'1': 'translated'},
            'page_summary': 'A short-haired girl waits at a station.',
        })

        with mock.patch.object(
            type(self.translator),
            'profile',
            new_callable=mock.PropertyMock,
            return_value=self.profile,
        ), mock.patch.object(
            self.translator,
            'all_model_loaded',
            return_value=True,
        ), mock.patch.object(
            self.translator,
            '_request_translation',
            return_value=response,
        ) as request:
            self.translator.translate_textblk_lst(
                project.pages['001.png'],
                project=project,
                page_key='001.png',
                full_page=True,
            )

        self.assertEqual(request.call_count, 1)
        self.assertIsNone(project.get_llm_visual_summary('001.png'))
        project.mark_translation_finished('001.png', '简体中文')
        self.translator.on_page_translation_finished(project, '001.png')
        record = project.get_llm_visual_summary('001.png')
        self.assertEqual(
            record['text'],
            'A short-haired girl waits at a station.',
        )

    def test_saved_summaries_are_independent_from_exact_history_completion(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(3)
        self._complete(project, '001.png')
        project.set_llm_visual_summary_text(
            '001.png',
            'Completed page summary.',
        )
        project.set_llm_visual_summary_text(
            '002.png',
            'User summary from an incomplete page.',
        )
        project.set_llm_visual_summary_text(
            '003.png',
            'User summary for the current page.',
        )

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ), mock.patch(
            'ballontranslator.modules.context.translation_context.messages_token_count',
            return_value=1,
        ):
            context = self._snapshot_request_context(
                project,
                '003.png',
                self.profile,
                summary_enabled=True,
            )

        self.assertEqual(
            [page.page_key for page in context.history],
            ['001.png'],
        )
        self.assertEqual(
            context.history[0].snapshot.summary,
            'Completed page summary.',
        )
        self.assertIsNone(self.translator._snapshot_history_page(
            project,
            '002.png',
            '简体中文',
            summary_enabled=True,
        ))
        self.assertEqual(
            context.page_summaries,
            (
                PageSummary(
                    '002.png',
                    'User summary from an incomplete page.',
                ),
                PageSummary(
                    '003.png',
                    'User summary for the current page.',
                ),
            ),
        )

        messages, _ = self._assemble_request(
            ['current'],
            self.profile,
            request_context=context,
            summary_enabled=True,
        )
        history_response = messages[2]['content']
        current_prompt = messages[-1]['content']
        self.assertIn('Completed page summary.', history_response)
        self.assertNotIn('Completed page summary.', current_prompt)
        self.assertIn(
            'User summary from an incomplete page.',
            current_prompt,
        )
        self.assertIn('User summary for the current page.', current_prompt)

    def test_saved_summaries_apply_without_history(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(2)
        project.set_llm_visual_summary_text(
            '001.png',
            'The masked speaker promised to return the key.',
        )
        project.set_llm_visual_summary_text(
            '002.png',
            'The user identifies the masked speaker as Kuro.',
        )

        context = self._snapshot_request_context(
            project,
            '002.png',
            self.profile,
            summary_enabled=True,
        )
        messages, _ = self._assemble_request(
            ['current'],
            self.profile,
            request_context=context,
            summary_enabled=True,
        )

        self.assertEqual(context.history, ())
        self.assertEqual(
            context.page_summaries,
            (
                PageSummary(
                    '001.png',
                    'The masked speaker promised to return the key.',
                ),
                PageSummary(
                    '002.png',
                    'The user identifies the masked speaker as Kuro.',
                ),
            ),
        )
        self.assertIn(
            'The masked speaker promised to return the key.',
            messages[-1]['content'],
        )
        self.assertIn(
            'The user identifies the masked speaker as Kuro.',
            messages[-1]['content'],
        )

    def test_existing_summary_skips_summary_request_and_is_preserved(self):
        project = self._project(1)
        old_record = {
            'version': 1,
            'text': 'Old summary.',
        }
        project.set_llm_visual_summary('001.png', old_record)
        pcfg.module.llm_translate_summary_memory = True

        with mock.patch.object(
            type(self.translator),
            'profile',
            new_callable=mock.PropertyMock,
            return_value=self.profile,
        ), mock.patch.object(
            self.translator,
            'all_model_loaded',
            return_value=True,
        ), mock.patch.object(
            self.translator,
            '_request_translation',
            return_value='{"1":"translated"}',
        ) as request:
            self.translator.translate(
                ['source-1'],
                project=project,
                page_key='001.png',
                commit_history_window=True,
            )

        self.assertNotIn('summary_enabled', request.call_args.kwargs)
        messages = request.call_args.args[1]
        self.assertNotIn('page_summary', messages[0]['content'])
        self.assertIn('Old summary.', messages[-1]['content'])
        self.assertIsNotNone(project.get_llm_visual_summary('001.png'))
        self.translator.on_page_translation_finished(project, '001.png')

        self.assertEqual(
            project.get_llm_visual_summary('001.png')['text'],
            'Old summary.',
        )

    def test_overwrite_summary_ignores_current_context_and_replaces_it(self):
        project = self._project(2)
        project.set_llm_visual_summary_text('001.png', 'Prior summary.')
        project.set_llm_visual_summary_text('002.png', 'Old current summary.')
        pcfg.module.llm_translate_summary_memory = True
        pcfg.module.llm_translate_overwrite_summary = True
        response = json.dumps({
            'translations': {'1': 'translated'},
            'page_summary': 'Replacement summary.',
        })

        with mock.patch.object(
            type(self.translator),
            'profile',
            new_callable=mock.PropertyMock,
            return_value=self.profile,
        ), mock.patch.object(
            self.translator,
            'all_model_loaded',
            return_value=True,
        ), mock.patch.object(
            self.translator,
            '_request_translation',
            return_value=response,
        ) as request:
            self.translator.translate(
                ['source-2'],
                project=project,
                page_key='002.png',
                commit_history_window=True,
            )

        self.assertTrue(request.call_args.kwargs['summary_enabled'])
        messages = request.call_args.args[1]
        self.assertIn('page_summary', messages[0]['content'])
        self.assertIn('Prior summary.', messages[-1]['content'])
        self.assertNotIn('Old current summary.', messages[-1]['content'])
        self.assertEqual(
            project.get_llm_visual_summary('002.png')['text'],
            'Old current summary.',
        )

        self.translator.on_page_translation_finished(project, '002.png')

        self.assertEqual(
            project.get_llm_visual_summary('002.png')['text'],
            'Replacement summary.',
        )

    def test_user_clear_during_overwrite_is_not_replaced_by_summary(self):
        pcfg.module.llm_translate_summary_memory = True
        pcfg.module.llm_translate_overwrite_summary = True
        project = self._project(1)
        project.set_llm_visual_summary_text('001.png', 'User summary.')
        response = json.dumps({
            'translations': {'1': 'translated'},
            'page_summary': 'Generated summary.',
        })

        def complete_request(
            *_args: object,
            **_kwargs: object,
        ) -> str:
            project.clear_llm_visual_summary('001.png')
            return response

        with mock.patch.object(
            type(self.translator),
            'profile',
            new_callable=mock.PropertyMock,
            return_value=self.profile,
        ), mock.patch.object(
            self.translator,
            'all_model_loaded',
            return_value=True,
        ), mock.patch.object(
            self.translator,
            '_request_translation',
            side_effect=complete_request,
        ):
            self.translator.translate(
                ['source-1'],
                project=project,
                page_key='001.png',
                commit_history_window=True,
            )

        self.translator.on_page_translation_finished(project, '001.png')

        self.assertIsNone(project.get_llm_visual_summary('001.png'))

    def test_partial_selection_does_not_persist_generated_summary(self):
        project = self._project(1)
        project.pages['001.png'] = [
            _block('selected'),
            _block('unselected', 'existing translation'),
        ]
        project.read_img = mock.Mock(
            return_value=np.zeros((32, 24, 3), dtype=np.uint8)
        )
        pcfg.module.llm_translate_vision = True
        pcfg.module.llm_translate_summary_memory = True
        response = json.dumps({
            'translations': {'1': 'translated'},
            'page_summary': 'Must not be committed.',
        })

        with mock.patch.object(
            type(self.translator),
            'profile',
            new_callable=mock.PropertyMock,
            return_value=self.profile,
        ), mock.patch.object(
            self.translator,
            'all_model_loaded',
            return_value=True,
        ), mock.patch.object(
            self.translator,
            '_request_translation',
            return_value=response,
        ):
            self.translator.translate_textblk_lst(
                project.pages['001.png'][:1],
                project=project,
                page_key='001.png',
            )

        project.mark_translation_finished('001.png', '简体中文')
        self.translator.on_page_translation_finished(project, '001.png')
        self.assertIsNone(project.get_llm_visual_summary('001.png'))


if __name__ == '__main__':
    unittest.main()

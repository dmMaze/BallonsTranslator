import json
import unittest
from types import SimpleNamespace
from unittest import mock

from _llm_translation_test_support import LLMTranslationTestMixin
from ballontranslator.modules.context.history import (
    ContextAction,
    HistoryPage,
    HistoryWindow,
    HistoryWindowKey,
    RenderedHistoryPage,
)
from ballontranslator.modules.context.translation_context import (
    MemoryCheckpoint,
    PageSummary,
    memory_window_signature,
    plan_page_summary_context,
)
from ballontranslator.modules.exceptions import (
    LLMMemoryCompactionError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.utils.config import LLMTranslateContext, pcfg


class LLMTranslationMemoryTest(
    LLMTranslationTestMixin,
    unittest.TestCase,
):
    def test_summary_overflow_plans_oldest_low_water_batch(self):
        summaries = tuple(
            PageSummary(f'00{index}.png', f'summary-{index}')
            for index in range(1, 4)
        )

        with mock.patch(
            'ballontranslator.modules.context.translation_context.'
            'page_summary_context_token_count',
            side_effect=lambda selected, _model: len(selected) * 10,
        ):
            selected, compact = plan_page_summary_context(
                summaries,
                self.profile.model,
                25,
                required_page_key='003.png',
            )

        self.assertEqual(selected, summaries[1:])
        self.assertEqual(compact, summaries[:2])

    def test_covered_overflow_waits_for_new_summary_pressure(self):
        summaries = tuple(
            PageSummary(f'00{index}.png', f'summary-{index}')
            for index in range(1, 6)
        )

        with mock.patch(
            'ballontranslator.modules.context.translation_context.'
            'page_summary_context_token_count',
            side_effect=lambda selected, _model: len(selected) * 10,
        ):
            selected, compact = plan_page_summary_context(
                summaries[:4],
                self.profile.model,
                25,
                required_page_key='004.png',
                covered_page_keys=('001.png', '002.png'),
            )
            next_selected, next_compact = plan_page_summary_context(
                summaries,
                self.profile.model,
                25,
                required_page_key='005.png',
                covered_page_keys=('001.png', '002.png'),
            )

        self.assertEqual(selected, summaries[2:4])
        self.assertEqual(compact, ())
        self.assertEqual(next_selected, summaries[3:])
        self.assertEqual(next_compact, summaries[:4])

    def test_memory_compacts_evicted_summary_before_recent_history(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_prior_context_token_budget = 10
        pcfg.module.llm_translate_vision = True
        pcfg.module.llm_translate_summary_memory = True
        pages = {
            key: HistoryPage(key, (key,), (f't-{key}',), f'summary-{key}')
            for key in ('001.png', '002.png', '003.png')
        }
        summaries = {
            key: PageSummary(key, f'summary-{key}')
            for key in pages
        }
        project = self._project(4)
        for summary in summaries.values():
            project.set_llm_visual_summary_text(
                summary.page_key,
                summary.text,
            )
        model = self.translator._vision_model(self.profile)
        key = HistoryWindowKey(
            load_identity=project.load_identity,
            settings=(
                ('source_language', str(self.translator.lang_source)),
                ('model', model),
                (
                    'system_prompt',
                    self._prompt_spec(
                        self.profile,
                        summary_enabled=True,
                    ).system_prompt,
                ),
                ('token_budget', 10),
                ('memory_enabled', True),
                ('memory_signature', ''),
            ),
        )

        def rendered(page, *_args, **_kwargs):
            return RenderedHistoryPage(
                snapshot=page,
                messages=(('user', page.page_key), ('assistant', page.summary)),
                token_count=4,
            )

        self.translator._history_window = HistoryWindow(
            key=key,
            request_page_key='003.png',
            history=(rendered(pages['001.png']), rendered(pages['002.png'])),
            token_count=8,
        )
        completion = SimpleNamespace(
            content='{"memory":"The station meeting remains unresolved."}',
            usage=None,
        )

        with mock.patch.object(
            self.translator,
            '_snapshot_history_page',
            side_effect=lambda _project, page_key, *_args, **_kwargs: pages.get(
                page_key
            ),
        ), mock.patch(
            'ballontranslator.modules.translators.trans_llm.render_history_page',
            side_effect=rendered,
        ), mock.patch(
            'ballontranslator.modules.translators.trans_llm.messages_token_count',
            side_effect=lambda messages, _model: (
                6
                if str(messages[0].get('content', '')).startswith(
                    'Compacted translation memory'
                )
                else 10
            ),
        ), mock.patch.object(
            self.translator,
            'request_chat_completion',
            return_value=completion,
        ) as compact_request:
            context = self._snapshot_request_context(
                project,
                '004.png',
                self.profile,
                model=model,
                summary_enabled=True,
            )

        self.assertEqual(compact_request.call_count, 1)
        self.assertEqual(context.memory.covered_page_keys, ('001.png',))
        self.assertEqual(
            [page.page_key for page in context.history],
            ['003.png'],
        )
        self.assertEqual(context.diagnostic.token_count, 10)
        messages, _ = self._assemble_request(
            ['current'],
            self.profile,
            request_context=context,
            summary_enabled=True,
        )
        self.assertEqual(
            [message['role'] for message in messages],
            ['system', 'system', 'user', 'assistant', 'user'],
        )
        self.assertTrue(
            messages[1]['content'].startswith('Compacted translation memory')
        )

    def test_memory_waits_while_saved_summaries_fit_context_budget(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(2)
        project.set_llm_visual_summary_text(
            '001.png',
            'User-owned summary without completed translation.',
        )

        with mock.patch.object(
            self.translator,
            '_compact_summary_batch',
            return_value=None,
        ) as compact:
            context = self._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                summary_enabled=True,
            )

        compact.assert_not_called()
        self.assertEqual(context.history, ())
        self.assertEqual(
            context.page_summaries,
            (PageSummary(
                '001.png',
                'User-owned summary without completed translation.',
            ),),
        )

    def test_memory_compacts_summary_overflow_without_history(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_prior_context_token_budget = 128
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(2)
        summary_text = ('An unresolved identity clue. ' * 200).strip()
        project.set_llm_visual_summary_text('001.png', summary_text)
        checkpoint = MemoryCheckpoint(
            'Memory.',
            ('001.png',),
            32,
        )

        with mock.patch.object(
            self.translator,
            '_compact_summary_batch',
            return_value=checkpoint,
        ) as compact:
            context = self._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                summary_enabled=True,
            )

        self.assertEqual(context.history, ())
        self.assertEqual(
            compact.call_args.kwargs['summaries'],
            (PageSummary('001.png', summary_text),),
        )
        self.assertEqual(
            compact.call_args.kwargs['target_language'],
            'Simplified Chinese',
        )
        self.assertIs(context.memory, checkpoint)
        self.assertEqual(
            project.get_llm_compact_memory(),
            {
                'version': 1,
                'text': 'Memory.',
                'covered_pages': ['001.png'],
            },
        )

    def test_memory_discovers_late_summary_during_adjacent_growth(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_translate_summary_memory = True
        pcfg.module.llm_prior_context_token_budget = 10
        project = self._project(3)

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ), mock.patch.object(
            self.translator,
            '_request_translation',
            return_value='{"1":"target-2"}',
        ):
            self._translate(
                ['source-2'],
                profile=self.profile,
                request_context=self._snapshot_request_context(
                    project,
                    '002.png',
                    self.profile,
                    summary_enabled=True,
                ),
            )
            self._complete(project, '002.png')
            project.set_llm_visual_summary_text(
                '001.png',
                'Late summary from an incomplete page.',
            )
            project.set_llm_visual_summary_text(
                '002.png',
                'Summary from the exact page at risk of eviction.',
            )
            with mock.patch.object(
                self.translator,
                '_compact_summary_batch',
                return_value=MemoryCheckpoint(
                    'Memory.',
                    ('001.png',),
                    2,
                ),
            ) as compact, mock.patch(
                'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
                return_value=8,
            ):
                context = self._snapshot_request_context(
                    project,
                    '003.png',
                    self.profile,
                    summary_enabled=True,
                )

        self.assertEqual(context.diagnostic.action, ContextAction.GROW)
        self.assertEqual(
            tuple(page.page_key for page in context.history),
            ('002.png',),
        )
        self.assertEqual(context.diagnostic.evicted, 0)
        self.assertEqual(context.diagnostic.token_count, 10)
        self.assertLessEqual(
            context.diagnostic.token_count,
            context.history_budget,
        )
        self.assertEqual(
            compact.call_args.kwargs['summaries'],
            (PageSummary(
                '001.png',
                'Late summary from an incomplete page.',
            ),),
        )
        self.assertEqual(context.memory.token_count, 2)

    def test_memory_compaction_failure_propagates_after_retries(self):
        previous = MemoryCheckpoint('old memory', ('001.png',), 1)
        retired = (
            PageSummary('002.png', 'new summary'),
        )
        self.translator.set_param_value('retry attempts', 1)
        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            side_effect=RuntimeError('provider unavailable'),
        ):
            with self.assertRaisesRegex(
                LLMMemoryCompactionError,
                'after 1 attempt: provider unavailable',
            ):
                self.translator._compact_summary_batch(
                    previous=previous,
                    summaries=retired,
                    profile=self.profile,
                    model=self.profile.vision_model,
                    target_language='Simplified Chinese',
                )

    def test_memory_compaction_user_action_error_bypasses_retries(self):
        self.translator.set_param_value('retry attempts', 5)
        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            side_effect=LLMUserActionRequiredError('update profile'),
        ) as request:
            with self.assertRaisesRegex(
                LLMUserActionRequiredError,
                'update profile',
            ):
                self.translator._compact_summary_batch(
                    previous=None,
                    summaries=(PageSummary('001.png', 'summary'),),
                    profile=self.profile,
                    model=self.profile.vision_model,
                    target_language='Simplified Chinese',
                )

        self.assertEqual(request.call_count, 1)

    def test_memory_compaction_honors_cancellation(self):
        self.translator.stop_event = SimpleNamespace(is_set=lambda: True)
        summary = PageSummary('001.png', 'summary')

        with self.assertRaises(LLMRequestStopped):
            self.translator._compact_summary_batch(
                previous=None,
                summaries=(summary,),
                profile=self.profile,
                model=self.profile.vision_model,
                target_language='Simplified Chinese',
            )

    def test_memory_compaction_uses_its_own_response_format_name(self):
        completion = SimpleNamespace(
            content='{"memory":"merged memory"}',
            usage=None,
        )
        for strict in (False, True):
            with self.subTest(strict=strict):
                self.profile.json_schema_response_format = strict
                with mock.patch.object(
                    self.translator,
                    'request_chat_completion',
                    return_value=completion,
                ) as request, mock.patch(
                    'ballontranslator.modules.translators.trans_llm.messages_token_count',
                    return_value=1,
                ):
                    self.translator._compact_memory(
                        previous=None,
                        summaries=(PageSummary('001.png', 'summary'),),
                        profile=self.profile,
                        model=self.profile.model,
                        target_language='Simplified Chinese',
                    )

                request_args = request.call_args.args[1]
                self.assertIn(
                    'Keep the memory concise.',
                    request_args['messages'][0]['content'],
                )
                self.assertIn(
                    'complete memory body in Simplified Chinese',
                    request_args['messages'][0]['content'],
                )
                self.assertEqual(
                    request_args['max_completion_tokens'],
                    self.profile.max_tokens,
                )
                response_format = request_args['response_format']
                if strict:
                    self.assertEqual(response_format['type'], 'json_schema')
                    self.assertEqual(
                        response_format['json_schema']['name'],
                        'translation_memory',
                    )
                    self.assertEqual(
                        response_format['json_schema']['schema']['required'],
                        ['memory'],
                    )
                else:
                    self.assertEqual(
                        response_format,
                        {'type': 'json_object'},
                    )

    def test_memory_compaction_sends_only_not_yet_covered_summaries(self):
        previous = MemoryCheckpoint('old memory', ('001.png',), 1)
        summaries = (
            PageSummary('001.png', 'covered summary'),
            PageSummary('002.png', 'new summary'),
        )
        completion = SimpleNamespace(
            content='{"memory":"merged memory"}',
            usage=None,
        )

        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            return_value=completion,
        ) as request, mock.patch(
            'ballontranslator.modules.translators.trans_llm.messages_token_count',
            return_value=1,
        ):
            checkpoint = self.translator._compact_summary_batch(
                previous=previous,
                summaries=summaries,
                profile=self.profile,
                model=self.profile.vision_model,
                target_language='Simplified Chinese',
            )

        user_payload = json.loads(
            request.call_args.args[1]['messages'][1]['content']
        )
        self.assertEqual(
            user_payload['page_summaries'],
            [{'page': '002.png', 'summary': 'new summary'}],
        )
        self.assertEqual(
            checkpoint.covered_page_keys,
            ('001.png', '002.png'),
        )
        self.assertEqual(checkpoint.text, 'merged memory')
        self.assertEqual(
            request.call_args.args[1]['model'],
            self.profile.model,
        )

    def test_memory_compaction_sends_all_selected_summaries(self):
        summaries = tuple(
            PageSummary(f'00{index}.png', f'summary-{index}')
            for index in range(1, 4)
        )
        completion = SimpleNamespace(
            content='{"memory":"merged memory"}',
            usage=None,
        )

        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            return_value=completion,
        ) as request, mock.patch(
            'ballontranslator.modules.translators.trans_llm.'
            'messages_token_count',
            return_value=1,
        ):
            self.translator._compact_summary_batch(
                previous=None,
                summaries=summaries,
                profile=self.profile,
                model=self.profile.model,
                target_language='Simplified Chinese',
            )

        payload = json.loads(request.call_args.args[1]['messages'][1]['content'])
        self.assertEqual(
            [summary['page'] for summary in payload['page_summaries']],
            ['001.png', '002.png', '003.png'],
        )

    def test_memory_compaction_keeps_its_actual_translation_context_size(self):
        completion = SimpleNamespace(
            content='{"memory":"merged memory"}',
            usage=None,
        )

        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            return_value=completion,
        ) as request, mock.patch(
            'ballontranslator.modules.translators.trans_llm.messages_token_count',
            return_value=300,
        ):
            checkpoint = self.translator._compact_summary_batch(
                previous=None,
                summaries=(PageSummary('001.png', 'summary'),),
                profile=self.profile,
                model='vision-model',
                target_language='Simplified Chinese',
            )

        self.assertEqual(checkpoint.token_count, 300)
        request.assert_called_once()

    def test_persisted_memory_applies_without_vision_summary_or_history(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(2)
        project.set_llm_compact_memory({
            'version': 1,
            'text': 'The masked hero is named Kuro.',
            'covered_pages': ['002.png'],
        })

        contexts = tuple(
            self._snapshot_request_context(
                project,
                page_key,
                self.profile,
                model=self.profile.model,
            )
            for page_key in ('001.png', '002.png')
        )

        self.assertEqual(contexts[0].memory.text, contexts[1].memory.text)
        messages, _ = self._assemble_request(
            ['current'],
            self.profile,
            request_context=contexts[0],
        )
        self.assertEqual(
            [message['role'] for message in messages],
            ['system', 'system', 'user'],
        )
        self.assertIn('The masked hero is named Kuro.', messages[1]['content'])
        self.assertNotIn('002.png', messages[1]['content'])

    def test_coverage_metadata_does_not_change_prompt_signature(self):
        first = MemoryCheckpoint('Shared fact.', ('001.png',), 4)
        second = MemoryCheckpoint(
            'Shared fact.',
            ('001.png', '002.png'),
            4,
        )

        self.assertEqual(
            memory_window_signature(first),
            memory_window_signature(second),
        )

    def test_compacted_memory_persists_before_translation_request(self):
        pcfg.module.llm_translate_summary_memory = True
        pcfg.module.llm_prior_context_token_budget = 0
        project = self._project(2)
        project.set_llm_visual_summary_text('001.png', 'An earlier clue.')
        checkpoint = MemoryCheckpoint(
            'Shared fact.',
            ('001.png',),
            4,
        )

        def fail_translation(*_args: object, **_kwargs: object) -> list[str]:
            self.assertEqual(
                project.get_llm_compact_memory()['text'],
                'Shared fact.',
            )
            raise RuntimeError('translation failed')

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
            '_compact_summary_batch',
            return_value=checkpoint,
        ) as compact, mock.patch.object(
            self.translator,
            '_translate',
            side_effect=fail_translation,
        ):
            with self.assertRaisesRegex(RuntimeError, 'translation failed'):
                self.translator.translate(
                    ['source-2'],
                    project=project,
                    page_key='002.png',
                    commit_history_window=True,
                )
            retry_context = self._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                summary_enabled=True,
            )

        compact.assert_called_once()
        self.assertEqual(retry_context.memory.text, 'Shared fact.')
        self.assertEqual(
            project.get_llm_compact_memory(),
            {
                'version': 1,
                'text': checkpoint.text,
                'covered_pages': ['001.png'],
            },
        )

    def test_user_edit_wins_over_in_flight_compaction(self):
        pcfg.module.llm_translate_summary_memory = True
        pcfg.module.llm_prior_context_token_budget = 0
        project = self._project(2)
        project.set_llm_compact_memory({
            'version': 1,
            'text': 'Original.',
            'covered_pages': [],
        })
        project.set_llm_visual_summary_text('001.png', 'An earlier clue.')
        generated = MemoryCheckpoint(
            'Generated.',
            ('001.png',),
            4,
        )

        def edit_during_compaction(**_kwargs: object) -> MemoryCheckpoint:
            project.set_llm_compact_memory_text('User edit.')
            return generated

        with mock.patch.object(
            self.translator,
            '_compact_summary_batch',
            side_effect=edit_during_compaction,
        ) as compact:
            context = self._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                summary_enabled=True,
            )

        compact.assert_called_once()
        self.assertEqual(context.memory.text, 'User edit.')
        self.assertEqual(
            project.get_llm_compact_memory()['text'],
            'User edit.',
        )

    def test_summary_edit_during_compaction_rebuilds_latest_context(self):
        pcfg.module.llm_translate_summary_memory = True
        pcfg.module.llm_prior_context_token_budget = 128
        project = self._project(2)
        project.set_llm_visual_summary_text(
            '001.png',
            ('Original summary. ' * 500).strip(),
        )
        generated = MemoryCheckpoint('Generated.', ('001.png',), 4)

        def edit_during_compaction(**_kwargs: object) -> MemoryCheckpoint:
            project.set_llm_visual_summary_text('001.png', 'User edit.')
            return generated

        with mock.patch.object(
            self.translator,
            '_compact_summary_batch',
            side_effect=edit_during_compaction,
        ) as compact:
            context = self._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                summary_enabled=True,
            )

        compact.assert_called_once()
        self.assertIsNone(project.get_llm_compact_memory())
        self.assertIsNone(context.memory)
        self.assertEqual(
            context.page_summaries,
            (PageSummary('001.png', 'User edit.'),),
        )


if __name__ == '__main__':
    unittest.main()

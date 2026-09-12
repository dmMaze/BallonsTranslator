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
)
from ballontranslator.modules.exceptions import (
    LLMMemoryCompactionError,
)
from ballontranslator.utils.config import LLMTranslateContext, pcfg


class LLMTranslationMemoryTest(
    LLMTranslationTestMixin,
    unittest.TestCase,
):
    def test_summary_overflow_compacts_when_history_rebuilds(self) -> None:
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_prior_context_token_budget = 1000
        pcfg.module.llm_translate_summary_memory = True
        for completed_page in (False, True):
            with self.subTest(completed_page=completed_page):
                project = self._project(3)
                old_summary = 'An earlier established fact. ' * 1000
                project.set_llm_visual_summary_text('001.png', old_summary)
                project.set_llm_visual_summary_text('003.png', 'Current clue.')
                if completed_page:
                    self._complete(project, '002.png')
                    project.set_llm_visual_summary_text('002.png', 'A recent event.')
                with mock.patch.object(
                    self.translator, 'request_chat_completion',
                    return_value=SimpleNamespace(
                        content='Earlier fact retained.', usage=None, finish_reason='stop',
                    ),
                ):
                    context = self._snapshot_request_context(
                        project, '003.png', self.profile, summary_enabled=True,
                    )

                self.assertIsNotNone(context.memory)
                self.assertEqual(context.memory.covered_page_keys, ('001.png',))
                self.assertEqual(len(context.history), int(completed_page))
                self.assertEqual(context.diagnostic.action, ContextAction.EVICT)
                self.assertLessEqual(context.diagnostic.token_count, 500)
                self.assertEqual(context.page_summaries, (PageSummary('003.png', 'Current clue.'),))
                self.assertEqual(project.get_llm_visual_summary('001.png')['text'], old_summary)

    def test_summary_pressure_bulk_evicts_a_growing_history(self) -> None:
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_prior_context_token_budget = 1000
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(6)
        project.set_llm_visual_summary_text('001.png', 'An earlier fact.')
        project.set_llm_visual_summary_text('002.png', 'A recent event.')
        self._complete(project, '002.png')
        contexts = []

        with mock.patch(
            'ballontranslator.modules.translators.trans_llm.render_history_page',
            side_effect=lambda page, *_args: RenderedHistoryPage(page, (), 300),
        ), mock.patch(
            'ballontranslator.modules.context.translation_context.messages_token_count',
            return_value=400,
        ), mock.patch.object(
            self.translator, 'request_chat_completion',
            return_value=SimpleNamespace(content='Earlier facts.', usage=None, finish_reason='stop'),
        ) as compact, mock.patch.object(
            self.translator, '_request_translation',
            return_value='{"translations":{"1":"translated"},"page_summary":"New event."}',
        ):
            for page_key in ('003.png', '004.png', '005.png'):
                context = self._snapshot_request_context(
                    project, page_key, self.profile, summary_enabled=True,
                )
                contexts.append(context)
                self._translate(
                    ['source'], profile=self.profile, request_context=context,
                    page_key=page_key, summary_enabled=True,
                )
                self._complete(project, page_key)
                self.translator.on_page_translation_finished(project, page_key)
            following = self._snapshot_request_context(
                project, '006.png', self.profile, summary_enabled=True,
            )

        self.assertEqual([context.diagnostic.token_count for context in contexts[:2]], [700, 1000])
        self.assertTrue(all(context.memory is None for context in contexts[:2]))
        eviction = contexts[2]
        self.assertIsNotNone(eviction.memory)
        self.assertEqual(eviction.memory.covered_page_keys, ('001.png', '002.png', '003.png'))
        self.assertEqual(eviction.diagnostic.action, ContextAction.EVICT)
        self.assertEqual(eviction.diagnostic.token_count, 300)
        self.assertEqual([page.page_key for page in eviction.history], ['004.png'])
        self.assertEqual(eviction.page_summaries, ())
        self.assertEqual(following.diagnostic.action, ContextAction.GROW)
        self.assertEqual(following.diagnostic.token_count, 600)
        self.assertEqual(following.memory.text, eviction.memory.text)
        compact.assert_called_once()

    def test_memory_compacts_evicted_summary_before_recent_history(self) -> None:
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_prior_context_token_budget = 1000
        pcfg.module.llm_translate_vision = True
        pcfg.module.llm_translate_summary_memory = True
        pages = {
            key: HistoryPage(key, (key,), (f't-{key}',), f'summary-{key}')
            for key in ('001.png', '002.png', '003.png', '004.png', '005.png')
        }
        summaries = {
            key: PageSummary(key, f'summary-{key}')
            for key in pages
        }
        project = self._project(5)
        for summary in summaries.values():
            project.set_llm_visual_summary_text(
                summary.page_key,
                summary.text,
            )
        model = self.translator._text_model(self.profile)
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
                ('token_budget', 1000),
                ('memory_enabled', True),
                ('memory_signature', ''),
            ),
        )

        def rendered(page: HistoryPage, *_args: object, **_kwargs: object) -> RenderedHistoryPage:
            return RenderedHistoryPage(
                snapshot=page,
                messages=(('user', page.page_key), ('assistant', page.summary)),
                token_count=400,
            )

        self.translator._history_window = HistoryWindow(
            key=key,
            request_page_key='003.png',
            history=(rendered(pages['001.png']), rendered(pages['002.png'])),
            token_count=800,
        )
        committed_window = self.translator._history_window
        completion = SimpleNamespace(
            content='The station meeting remains unresolved.',
            usage=None,
            finish_reason='',
        )

        def compact_request(*_args: object) -> SimpleNamespace:
            self.assertIs(self.translator._history_window, committed_window)
            self.assertEqual(
                [page.page_key for page in committed_window.history],
                ['001.png', '002.png'],
            )
            return completion

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
                6000
                if str(messages[0].get('content', '')).startswith(
                    'Compacted translation memory'
                )
                else 10
            ),
        ), mock.patch.object(
            self.translator,
            'request_chat_completion',
            side_effect=compact_request,
        ) as request:
            self.translator.set_param_value('retry attempts', 1)
            with mock.patch.object(
                self.translator, 'request_chat_completion',
                side_effect=RuntimeError('compaction failed'),
            ), self.assertRaises(LLMMemoryCompactionError):
                self._snapshot_request_context(
                    project, '004.png', self.profile,
                    model=model, summary_enabled=True,
                )
            self.assertIs(self.translator._history_window, committed_window)
            self.assertIsNone(project.get_llm_compact_memory())

            context = self._snapshot_request_context(
                project,
                '004.png',
                self.profile,
                model=model,
                summary_enabled=True,
            )
            with mock.patch.object(
                self.translator, '_request_translation',
                return_value='{"translations":{"1":"translated"},"page_summary":"summary"}',
            ):
                self._translate(
                    ['source'], profile=self.profile, request_context=context,
                    summary_enabled=True,
                )
            following = self._snapshot_request_context(
                project, '005.png', self.profile,
                model=model, summary_enabled=True,
            )

        self.assertEqual(request.call_count, 1)
        self.assertEqual(context.memory.covered_page_keys, ('001.png', '002.png'))
        self.assertEqual(
            [page.page_key for page in context.history],
            ['003.png'],
        )
        self.assertLessEqual(context.diagnostic.token_count, 500)
        self.assertEqual(context.page_summaries, (summaries['004.png'],))
        self.assertEqual(following.diagnostic.action, ContextAction.GROW)
        self.assertEqual(
            [page.page_key for page in following.history],
            ['003.png', '004.png'],
        )
        self.assertEqual(following.page_summaries, (summaries['005.png'],))
        self.assertEqual(following.memory.text, context.memory.text)
        for summary in summaries.values():
            self.assertEqual(
                project.get_llm_visual_summary(summary.page_key)['text'],
                summary.text,
            )
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

    def test_memory_compacts_summary_overflow_without_history(self) -> None:
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_prior_context_token_budget = 128
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(3)
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
            project.set_llm_visual_summary_text('002.png', 'A new clue.')
            following = self._snapshot_request_context(
                project, '003.png', self.profile, summary_enabled=True,
            )

        self.assertEqual(context.history, ())
        self.assertEqual(context.diagnostic.action, ContextAction.EVICT)
        self.assertEqual(context.diagnostic.summaries_evicted, 1)
        self.assertLessEqual(context.diagnostic.token_count, int(128 * 0.50))
        self.assertEqual(
            following.page_summaries,
            (PageSummary('002.png', 'A new clue.'),),
        )
        compact.assert_called_once()
        self.assertEqual(
            project.get_llm_visual_summary('001.png')['text'], summary_text,
        )
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

    def test_last_page_finalization_compacts_remaining_summaries(self):
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(2)
        project.set_llm_visual_summary_text('001.png', 'Earlier summary.')
        project.set_llm_visual_summary_text('002.png', 'Final summary.')
        project.set_llm_compact_memory({
            'version': 1,
            'text': 'Existing memory.',
            'covered_pages': ['001.png'],
        })
        merged = MemoryCheckpoint(
            'Merged memory.',
            ('001.png', '002.png'),
            4,
        )

        with mock.patch.object(
            type(self.translator),
            'profile',
            new_callable=mock.PropertyMock,
            return_value=self.profile,
        ), mock.patch.object(
            self.translator,
            '_compact_summary_batch',
            return_value=merged,
        ) as compact:
            self.translator.on_page_translation_finished(project, '001.png')
            compact.assert_not_called()
            self._complete(project, '002.png')
            self.translator.on_page_translation_finished(project, '002.png')
            compaction_kwargs = compact.call_args.kwargs
            compact.reset_mock()
            self.translator.on_page_translation_finished(project, '002.png')
            compact.assert_not_called()

        self.assertEqual(
            compaction_kwargs['summaries'],
            (PageSummary('002.png', 'Final summary.'),),
        )
        self.assertEqual(
            compaction_kwargs['previous'].text,
            'Existing memory.',
        )
        self.assertEqual(
            project.get_llm_compact_memory(),
            {
                'version': 1,
                'text': 'Merged memory.',
                'covered_pages': ['001.png', '002.png'],
            },
        )

    def test_memory_compaction_failure_propagates_after_retries(self) -> None:
        previous = MemoryCheckpoint('old memory', ('001.png',), 1)
        retired = (
            PageSummary('002.png', 'new summary'),
        )
        self.translator.set_param_value('retry attempts', 2)
        self.translator.set_param_value('retry timeout', 0)
        for response, error in (
            (RuntimeError('provider unavailable'), 'provider unavailable'),
            (
                SimpleNamespace(content=' \n\t ', usage=None, finish_reason=''),
                'Memory compaction returned no memory text.',
            ),
        ):
            with self.subTest(error=error), mock.patch.object(
                self.translator,
                'request_chat_completion',
                side_effect=[response, response],
            ), self.assertRaisesRegex(
                LLMMemoryCompactionError,
                f'after 2 attempts: {error}',
            ):
                self.translator._compact_summary_batch(
                    previous=previous,
                    summaries=retired,
                    profile=self.profile,
                    model=self.profile.model,
                    target_language='Simplified Chinese',
                )

    def test_memory_compaction_sends_only_not_yet_covered_summaries(self) -> None:
        self.profile.model = 'selected-translation-model'
        self.profile.vision_model = 'ignored-ocr-model'
        self.profile.json_schema_response_format = True
        previous = MemoryCheckpoint('old memory', ('001.png',), 1)
        summaries = (
            PageSummary('001.png', 'covered summary'),
            PageSummary('002.png', 'new summary'),
            PageSummary('003.png', 'another summary'),
        )
        completion = SimpleNamespace(
            content='  merged memory\nwith a second line.  ',
            usage=None,
            finish_reason='',
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
                model=self.profile.model,
                target_language='Simplified Chinese',
            )

        user_payload = json.loads(
            request.call_args.args[1]['messages'][1]['content']
        )
        self.assertEqual(
            user_payload['page_summaries'],
            [
                {'page': '002.png', 'summary': 'new summary'},
                {'page': '003.png', 'summary': 'another summary'},
            ],
        )
        self.assertEqual(
            checkpoint.covered_page_keys,
            ('001.png', '002.png', '003.png'),
        )
        self.assertEqual(checkpoint.text, 'merged memory\nwith a second line.')
        self.assertEqual(
            request.call_args.args[1]['model'],
            self.profile.model,
        )
        self.assertNotIn('response_format', request.call_args.args[1])
        self.assertIn(
            'complete memory body in Simplified Chinese',
            request.call_args.args[1]['messages'][0]['content'],
        )

    def test_memory_compaction_keeps_its_actual_translation_context_size(self) -> None:
        completion = SimpleNamespace(
            content='merged memory',
            usage=None,
            finish_reason='',
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

    def test_large_saved_memory_preserves_history_and_summary_allowance(self) -> None:
        pcfg.module.llm_prior_context_token_budget = 2048
        pcfg.module.llm_translate_summary_memory = True
        for mode in (LLMTranslateContext.HISTORY, LLMTranslateContext.PAGE):
            with self.subTest(mode=mode):
                pcfg.module.llm_translate_context = mode
                project = self._project(3)
                self._complete(project, '001.png')
                for page_key in project.pages:
                    project.set_llm_visual_summary_text(page_key, f'Clue on {page_key}.')
                with mock.patch.object(
                    self.translator,
                    'request_chat_completion',
                    side_effect=AssertionError('Saved memory must not cause compaction.'),
                ):
                    before = self._snapshot_request_context(
                        project, '003.png', self.profile,
                        model='unknown-model', summary_enabled=True,
                    )
                    memory_text = 'A stable identity fact. ' * 3000
                    project.set_llm_compact_memory_text(memory_text)
                    after = self._snapshot_request_context(
                        project, '003.png', self.profile,
                        model='unknown-model', summary_enabled=True,
                    )

                self.assertGreater(after.memory.token_count, after.history_budget)
                self.assertEqual(after.memory.text, memory_text.strip())
                self.assertEqual(after.history, before.history)
                self.assertTrue(after.page_summaries)
                self.assertEqual(after.page_summaries, before.page_summaries)
                self.assertEqual(after.diagnostic.token_count, before.diagnostic.token_count)
                self.assertEqual(len(after.history), int(mode == LLMTranslateContext.HISTORY))

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

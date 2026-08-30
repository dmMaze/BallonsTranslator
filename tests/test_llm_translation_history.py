import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from _llm_translation_test_support import (
    LLMTranslationTestMixin,
    _block,
)
from ballontranslator.modules.context.errors import ContextLengthError
from ballontranslator.modules.context.glossary import GlossaryEntry
from ballontranslator.modules.context.history import (
    ContextAction,
    ContextReason,
    HistoryPage,
    HistoryWindow,
    HistoryWindowKey,
    RenderedHistoryPage,
    eligible_history_for_request,
)
from ballontranslator.modules.context.token_usage import (
    MESSAGE_TOKEN_OVERHEAD,
    messages_token_count,
)
from ballontranslator.modules.context.translation_context import (
    MemoryCheckpoint,
    PageSummary,
    RequestContext,
    recover_context_length,
)
from ballontranslator.modules.translators.llm_translation_contract import (
    assemble_translation_request,
    render_assistant_response,
    render_history_page,
)
from ballontranslator.ui.module_manager import TranslateThread
from ballontranslator.utils.config import (
    LLMGlossaryMode,
    LLMTranslateContext,
    OCRTextPostprocess,
    RunStatus,
    pcfg,
)
from ballontranslator.utils.llm_profiles import default_profile
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class LLMTranslationHistoryTest(
    LLMTranslationTestMixin,
    unittest.TestCase,
):
    def _history_for_rebuild(
        self,
        pages,
        token_budget=1000,
        model='test-model',
    ):
        pages_by_key = {page.page_key: page for page in pages}
        project = SimpleNamespace(
            pages={**{key: [] for key in pages_by_key}, 'current': []}
        )
        history, _ = eligible_history_for_request(
            window=None,
            project=project,
            page_key='current',
            previous_page=None,
            token_budget=token_budget,
            rebuild_reason=ContextReason.WINDOW_EMPTY,
            snapshot_page=pages_by_key.get,
            render_page=lambda page: render_history_page(
                page,
                model,
                self._prompt_spec(),
            ),
        )
        return history

    def _successful_request(self, project, page_key, profile=None):
        """Snapshot and advance the runtime window through the real success path."""
        profile = profile or self.profile
        context = self._snapshot_request_context(
            project,
            page_key,
            profile,
        )
        source = project.pages[page_key][0].get_text()
        translation = project.pages[page_key][0].translation or 'translated'
        response = render_assistant_response((translation,))
        with mock.patch.object(
            self.translator,
            '_request_translation',
            return_value=response,
        ):
            self._translate(
                [source],
                profile=profile,
                request_context=context,
            )
        return context

    def test_cache_key_and_wire_share_one_frozen_prompt_snapshot(self):
        project = self._project(2)
        self._complete(project, '001.png')
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        prompt_spec = self._prompt_spec(history_enabled=True)

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            context = self.translator._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                prompt_spec=prompt_spec,
                source_language=str(self.translator.lang_source),
                target_language=str(self.translator.lang_target),
                history_budget=max(
                    0,
                    int(pcfg.module.llm_prior_context_token_budget),
                ),
                glossary_path=str(pcfg.module.llm_glossary_path or ''),
                glossary_mode=pcfg.module.llm_glossary_mode,
                memory_enabled=False,
            )

        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        self.translator.set_target('English')
        messages, prompt = assemble_translation_request(
            ('current',),
            prompt_spec=prompt_spec,
            request_context=context,
        )

        settings = dict(context.window_key.settings)
        self.assertEqual(settings['system_prompt'], messages[0]['content'])
        self.assertIn('to Simplified Chinese', prompt)
        self.assertIn('to Simplified Chinese', messages[0]['content'])

    def test_saved_summary_edits_change_only_the_current_request_suffix(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(2)
        self._complete(project, '001.png')
        project.set_llm_visual_summary_text('002.png', 'First user edit.')

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ), mock.patch(
            'ballontranslator.modules.context.translation_context.messages_token_count',
            return_value=1,
        ):
            first_context = self._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                summary_enabled=True,
            )
            first_messages, _ = self._assemble_request(
                ['current'],
                self.profile,
                request_context=first_context,
                summary_enabled=True,
            )
            project.set_llm_visual_summary_text(
                '002.png',
                'Second user edit.',
            )
            second_context = self._snapshot_request_context(
                project,
                '002.png',
                self.profile,
                summary_enabled=True,
            )
            second_messages, _ = self._assemble_request(
                ['current'],
                self.profile,
                request_context=second_context,
                summary_enabled=True,
            )

        self.assertEqual(first_messages[:-1], second_messages[:-1])
        self.assertIn('First user edit.', first_messages[-1]['content'])
        self.assertIn('Second user edit.', second_messages[-1]['content'])

    def test_memory_only_summary_edits_keep_exact_history_reusable(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_translate_summary_memory = True
        project = self._project(3)
        self._complete(project, '001.png')
        project.set_llm_visual_summary_text('001.png', 'First summary edit.')

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
                ),
            )
            self._complete(project, '002.png')
            project.set_llm_visual_summary_text(
                '001.png',
                'Second summary edit.',
            )
            context = self._snapshot_request_context(
                project,
                '003.png',
                self.profile,
            )

        self.assertEqual(context.diagnostic.action, ContextAction.GROW)
        self.assertIsNone(context.diagnostic.rebuild_reason)
        self.assertTrue(all(
            not page.snapshot.summary
            for page in context.history
        ))

    def test_memory_tokens_reduce_available_recent_history_budget(self):
        pages = tuple(
            HistoryPage(str(index), (f's{index}',), (f't{index}',))
            for index in range(1, 4)
        )

        def rendered(page):
            return RenderedHistoryPage(page, (), 3)

        key = HistoryWindowKey(object(), ())
        window = HistoryWindow(
            key,
            '2',
            (rendered(pages[0]), rendered(pages[1])),
            6,
        )
        history, diagnostic = eligible_history_for_request(
            window=window,
            project=None,
            page_key='3',
            previous_page=pages[2],
            token_budget=10,
            rebuild_reason=None,
            snapshot_page=lambda _key: None,
            render_page=rendered,
            reserved_tokens=3,
        )

        self.assertEqual([page.page_key for page in history], ['2', '3'])
        self.assertEqual(diagnostic.evicted, 1)
        self.assertEqual(diagnostic.token_count, 9)

    def test_context_overflow_recovery_preserves_compacted_memory(self):
        pages = tuple(
            RenderedHistoryPage(
                HistoryPage(str(index), ('s',), ('t',)),
                (),
                4,
            )
            for index in range(2)
        )
        memory = MemoryCheckpoint('memory', ('old',), 2)
        summary = PageSummary('current', 'Current page context.')
        recovered = recover_context_length(RequestContext(
            history=pages,
            history_budget=12,
            memory=memory,
            request_page_key='current',
            page_summaries=(summary,),
            summary_token_count=1,
            current_summary_token_count=1,
        ))

        self.assertIs(recovered.memory, memory)
        self.assertEqual(recovered.page_summaries, (summary,))
        self.assertEqual(len(recovered.history), 1)
        self.assertEqual(recovered.diagnostic.token_count, 7)

    def test_context_overflow_drops_prior_summaries_before_current_page(self):
        prior = PageSummary('001.png', 'Prior context.')
        current = PageSummary('002.png', 'Current context.')
        context = RequestContext(
            history=(),
            history_budget=20,
            request_page_key='002.png',
            page_summaries=(prior, current),
            summary_token_count=10,
            current_summary_token_count=4,
        )
        recovered = recover_context_length(context)
        self.assertEqual(recovered.page_summaries, (current,))
        self.assertEqual(recovered.summary_token_count, 4)
        self.assertEqual(recovered.diagnostic.token_count, 4)
        self.assertEqual(recovered.diagnostic.summaries_evicted, 1)

        with mock.patch.object(
            self.translator,
            '_request_translation',
            side_effect=(
                ContextLengthError('maximum context length exceeded'),
                '{"1":"translated"}',
            ),
        ) as request:
            translated = self._translate(
                ['source'],
                profile=self.profile,
                request_context=context,
            )

        self.assertEqual(translated, ['translated'])
        first_prompt = request.call_args_list[0].args[1][-1]['content']
        recovered_prompt = request.call_args_list[1].args[1][-1]['content']
        self.assertIn('Prior context.', first_prompt)
        self.assertNotIn('Prior context.', recovered_prompt)
        self.assertIn('Current context.', recovered_prompt)

    def test_history_eligibility_is_page_complete_target_compatible_and_past_only(self):
        pages = {
            '001.png': [_block('one', '一'), _block('', '')],
            '002.png': [_block('two', '二')],
            '003.png': [_block('three', '三')],
            '004.png': [_block('four-a', '四'), _block('four-b', '')],
            '005.png': [_block('', '')],
            '006.png': [_block('current', '')],
            '007.png': [_block('future', '未来')],
        }
        project = SimpleNamespace(
            pages=pages,
            _image_info={
                '001.png': {
                    'finish_code': RunStatus.FIN_TRANSLATE,
                    'translation_target': '简体中文',
                },
                # Missing target metadata remains eligible for old projects.
                '002.png': {'finish_code': RunStatus.FIN_TRANSLATE},
                '003.png': {
                    'finish_code': RunStatus.FIN_TRANSLATE,
                    'translation_target': 'English',
                },
                '004.png': {'finish_code': RunStatus.FIN_TRANSLATE},
                '005.png': {'finish_code': RunStatus.FIN_TRANSLATE},
                '006.png': {'finish_code': 0},
                '007.png': {'finish_code': RunStatus.FIN_TRANSLATE},
            },
        )

        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            context = self._snapshot_request_context(
                project,
                '006.png',
                self.profile,
            )
        history = tuple(page.snapshot for page in context.history)

        self.assertEqual([page.page_key for page in history], ['001.png', '002.png'])
        self.assertEqual(history[0].sources, ('one',))
        self.assertEqual(history[0].translations, ('一',))

    def test_matching_glossary_is_rendered_for_current_page_only(self):
        glossary = (
            GlossaryEntry('Hero', '勇者', 'title'),
            GlossaryEntry('Mage', '法师'),
        )
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            history = self._history_for_rebuild(
                (HistoryPage('001.png', ('Hero arrives',), ('勇者到来',)),),
            )
        context = RequestContext(
            history=history,
            glossary=glossary,
            glossary_mode=LLMGlossaryMode.Matching,
        )
        messages, _ = self._assemble_request(
            ['Mage speaks'],
            self.profile,
            request_context=context,
        )

        self.assertEqual(
            [message['role'] for message in messages],
            ['system', 'user', 'assistant', 'user'],
        )
        self.assertIn('"source": "Hero arrives"', messages[1]['content'])
        self.assertNotIn('GLOSSARY:', messages[1]['content'])
        self.assertEqual(
            messages[2]['content'],
            '{"1":"勇者到来"}',
        )
        self.assertIn('"source":"Mage"', messages[3]['content'])
        self.assertNotIn('"source":"Hero"', messages[3]['content'])

    def test_matching_glossary_preserves_only_the_clean_history_prefix(self):
        glossary = (
            GlossaryEntry('Hero', '勇者'),
            GlossaryEntry('Mage', '法师'),
        )
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            hero_page = render_history_page(
                HistoryPage('001.png', ('Hero arrives',), ('勇者到来',)),
                'test-model',
                self._prompt_spec(),
            )
            mage_page = render_history_page(
                HistoryPage('002.png', ('Mage speaks',), ('法师说话',)),
                'test-model',
                self._prompt_spec(),
            )

        def request_messages(query, history):
            context = RequestContext(
                history=history,
                glossary=glossary,
                glossary_mode=LLMGlossaryMode.Matching,
            )
            messages, _ = self._assemble_request(
                [query],
                self.profile,
                request_context=context,
            )
            return messages

        first = request_messages('Hero arrives', ())
        second = request_messages('Mage speaks', (hero_page,))
        third = request_messages('No glossary term', (hero_page, mage_page))

        # The previous current-page prompt had a glossary suffix, whereas its
        # historical form is clean, so the exact message prefix ends before it.
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])
        self.assertTrue(first[1]['content'].startswith(second[1]['content']))

        # On the next page, the older clean pair is reusable in full; matching
        # stops again at the immediately preceding page's former glossary suffix.
        self.assertEqual(second[:3], third[:3])
        self.assertNotEqual(second[3], third[3])
        self.assertTrue(second[3]['content'].startswith(third[3]['content']))

    def test_all_glossary_uses_stable_system_message(self):
        glossary = (
            GlossaryEntry('Hero', '勇者'),
            GlossaryEntry('Mage', '法师'),
        )
        context = RequestContext(
            history=(),
            glossary=glossary,
            glossary_mode=LLMGlossaryMode.All,
        )
        messages, _ = self._assemble_request(
            ['No matching term'],
            self.profile,
            request_context=context,
        )

        self.assertEqual(
            [message['role'] for message in messages],
            ['system', 'system', 'user'],
        )
        self.assertIn('"source":"Hero"', messages[1]['content'])
        self.assertIn('"source":"Mage"', messages[1]['content'])
        self.assertNotIn('GLOSSARY:', messages[2]['content'])

    def test_rebuild_selects_newest_pages_with_growth_headroom(self):
        history = tuple(
            HistoryPage(str(index), (f'source-{index}',), (f'target-{index}',))
            for index in range(1, 4)
        )
        cases = (
            ((4, 4, 4), 10, ['3']),
            ((12, 4, 4), 8, ['2']),
            ((7, 1, 1), 10, ['3']),
        )
        for token_costs, budget, expected in cases:
            with self.subTest(token_costs=token_costs, budget=budget):
                with mock.patch(
                    'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
                    side_effect=token_costs,
                ):
                    selected = self._history_for_rebuild(
                        history,
                        budget,
                        'unknown-model',
                    )
                self.assertEqual(
                    [page.page_key for page in selected],
                    expected,
                )

    def test_rebuild_reserves_headroom_for_the_next_adjacent_page(self):
        project = self._project(7)
        for page_key in ('001.png', '002.png', '003.png', '004.png', '005.png'):
            self._complete(project, page_key)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_prior_context_token_budget = 10

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=2,
        ):
            rebuilt = self._successful_request(
                project,
                '006.png',
                self.profile,
            )
            self._complete(project, '006.png')
            adjacent = self._successful_request(
                project,
                '007.png',
                self.profile,
            )

        self.assertEqual(
            [page.page_key for page in rebuilt.history],
            ['003.png', '004.png', '005.png'],
        )
        self.assertEqual(adjacent.diagnostic.action, ContextAction.GROW)
        self.assertEqual(adjacent.history[:-1], rebuilt.history)

    def test_stateless_rebuild_stops_at_first_ordinary_overflow(self):
        history = tuple(
            HistoryPage(str(index), (f'source-{index}',), (f'target-{index}',))
            for index in range(1, 4)
        )
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            side_effect=(4, 8, 1),
        ) as token_count:
            selected = self._history_for_rebuild(
                history,
                10,
                'test-model',
            )

        self.assertEqual([page.page_key for page in selected], ['3'])
        self.assertEqual(token_count.call_count, 2)

    def test_adjacent_requests_grow_append_only(self):
        project = self._project(4)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_prior_context_token_budget = 10

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=2,
        ):
            first = self._successful_request(
                project, '001.png', self.profile,
            )
            self._complete(project, '001.png')
            second = self._successful_request(
                project, '002.png', self.profile,
            )
            self._complete(project, '002.png')
            third = self._successful_request(
                project, '003.png', self.profile,
            )

        self.assertEqual(first.diagnostic.action, ContextAction.EMPTY)
        self.assertEqual(second.diagnostic.action, ContextAction.GROW)
        self.assertEqual(third.diagnostic.action, ContextAction.GROW)
        self.assertEqual(
            [page.page_key for page in third.history],
            ['001.png', '002.png'],
        )
        self.assertEqual(third.history[:-1], second.history)
        first_messages, _ = self._assemble_request(
            ['source-1'], self.profile, request_context=first,
        )
        second_messages, _ = self._assemble_request(
            ['source-2'], self.profile, request_context=second,
        )
        third_messages, _ = self._assemble_request(
            ['source-3'], self.profile, request_context=third,
        )
        system_prompt = first_messages[0]['content']
        self.assertIn('read-only completed page examples', system_prompt)
        self.assertIn('output format.\n- Treat prior', system_prompt)
        self.assertIn('IDs are local to each pair and may repeat', system_prompt)
        self.assertIn(
            'never translate, repeat, correct, or include those earlier items',
            system_prompt,
        )
        self.assertEqual(second_messages[:len(first_messages)], first_messages)
        self.assertEqual(third_messages[:len(second_messages)], second_messages)

    def test_overflow_bulk_evicts_and_later_prefix_grows_stably(self):
        project = self._project(8)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_prior_context_token_budget = 10
        contexts = []

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=2,
        ):
            for index in range(1, 9):
                page_key = f'{index:03}.png'
                contexts.append(self._successful_request(
                    project,
                    page_key,
                    self.profile,
                ))
                self._complete(project, page_key)

        eviction = contexts[6]
        after_eviction = contexts[7]
        self.assertEqual(eviction.diagnostic.action, ContextAction.EVICT)
        self.assertEqual(eviction.diagnostic.evicted, 2)
        self.assertEqual(eviction.diagnostic.token_count, 8)
        self.assertEqual(after_eviction.diagnostic.action, ContextAction.GROW)
        self.assertEqual(after_eviction.history[:-1], eviction.history)
        eviction_messages, _ = self._assemble_request(
            ['source-7'], self.profile, request_context=eviction,
        )
        later_messages, _ = self._assemble_request(
            ['source-8'], self.profile, request_context=after_eviction,
        )
        self.assertEqual(
            later_messages[:len(eviction_messages)],
            eviction_messages,
        )

    def test_oversized_adjacent_page_is_skipped_without_splitting_window(self):
        project = self._project(3)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_prior_context_token_budget = 5

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            side_effect=(2, 7),
        ):
            self._successful_request(
                project, '001.png', self.profile,
            )
            self._complete(project, '001.png')
            second = self._successful_request(
                project, '002.png', self.profile,
            )
            self._complete(project, '002.png')
            third = self._successful_request(
                project, '003.png', self.profile,
            )

        self.assertEqual([page.page_key for page in second.history], ['001.png'])
        self.assertEqual(third.diagnostic.action, ContextAction.REUSE)
        self.assertEqual(
            third.diagnostic.rebuild_reason,
            ContextReason.OVERSIZED_PAGE,
        )
        self.assertEqual(third.history, second.history)

    def test_oversized_adjacent_page_refits_history_for_current_summary(self):
        retained = RenderedHistoryPage(
            HistoryPage('001.png', ('s1',), ('t1',)),
            (),
            4,
        )
        window = HistoryWindow(
            HistoryWindowKey(object(), ()),
            '001.png',
            (retained,),
            4,
        )

        history, diagnostic = eligible_history_for_request(
            window=window,
            project=None,
            page_key='002.png',
            previous_page=HistoryPage('001.png', ('large',), ('large',)),
            token_budget=10,
            rebuild_reason=None,
            snapshot_page=lambda _page_key: None,
            render_page=lambda page: RenderedHistoryPage(page, (), 7),
            reserved_tokens=8,
        )

        self.assertEqual(history, ())
        self.assertEqual(diagnostic.action, ContextAction.EVICT)
        self.assertEqual(diagnostic.evicted, 1)
        self.assertEqual(diagnostic.token_count, 8)

    def test_page_jumps_and_failed_previous_page_rebuild(self):
        project = self._project(5)
        for page_key in ('001.png', '002.png', '003.png'):
            self._complete(project, page_key)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            initial = self._successful_request(
                project, '004.png', self.profile,
            )
            backward = self._successful_request(
                project, '002.png', self.profile,
            )
            forward = self._successful_request(
                project, '005.png', self.profile,
            )

        self.assertEqual(initial.diagnostic.action, ContextAction.REBUILD)
        self.assertEqual(
            backward.diagnostic.rebuild_reason,
            ContextReason.NON_ADJACENT,
        )
        self.assertEqual(
            forward.diagnostic.rebuild_reason,
            ContextReason.NON_ADJACENT,
        )
        self.assertEqual(
            [page.page_key for page in forward.history],
            ['001.png', '002.png', '003.png'],
        )

        project.clear_page_progress('004.png', RunStatus.FIN_TRANSLATE)
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            self._successful_request(
                project, '004.png', self.profile,
            )
            failed_previous = self._snapshot_request_context(
                project, '005.png', self.profile,
            )
        self.assertEqual(
            failed_previous.diagnostic.rebuild_reason,
            ContextReason.PREVIOUS_INCOMPLETE,
        )

    def _primed_window(self):
        project = self._project(4)
        self.translator._history_window = None
        self.translator.set_source('日本語')
        self.translator.set_target('简体中文')
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_glossary_mode = LLMGlossaryMode.Matching
        pcfg.module.llm_prior_context_token_budget = 10
        self._complete(project, '001.png')
        self._successful_request(
            project, '002.png', self.profile,
        )
        self._complete(project, '002.png')
        self._successful_request(
            project, '003.png', self.profile,
        )
        self._complete(project, '003.png')
        return project

    def test_window_invalidates_for_project_snapshots_and_prompt_inputs(self):
        cases = (
            'project',
            'source',
            'translation',
            'target',
            'model',
            'prompt',
            'budget',
        )
        for change in cases:
            with self.subTest(change=change), mock.patch(
                'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
                return_value=1,
            ):
                project = self._primed_window()
                request_project = project
                profile = self.profile
                if change == 'project':
                    request_project = self._project(4)
                    for page_key in ('001.png', '002.png', '003.png'):
                        self._complete(request_project, page_key)
                elif change == 'source':
                    project.pages['001.png'][0].text = ['edited-source']
                elif change == 'translation':
                    project.pages['001.png'][0].translation = 'edited-target'
                elif change == 'target':
                    self.translator.set_target('English')
                elif change == 'model':
                    profile.model = 'changed-model'
                    profile.model_options.append('changed-model')
                elif change == 'prompt':
                    profile.prompt = 'Changed instructions.'
                elif change == 'budget':
                    pcfg.module.llm_prior_context_token_budget = 9

                context = self._snapshot_request_context(
                    request_project,
                    '004.png',
                    profile,
                )

                expected_reason = (
                    ContextReason.PROJECT_CHANGED if change == 'project'
                    else ContextReason.SNAPSHOT_CHANGED
                    if change in ('source', 'translation')
                    else ContextReason.SETTINGS_CHANGED
                )
                self.assertEqual(context.diagnostic.rebuild_reason, expected_reason)

            self.profile = default_profile('OpenAI')
            self.profile.api_key = 'sk-test'

    def test_glossary_changes_do_not_rebuild_clean_history(self):
        with tempfile.TemporaryDirectory() as directory:
            glossary_path = os.path.join(directory, 'terms.tsv')
            with open(glossary_path, 'w', encoding='utf-8') as glossary_file:
                glossary_file.write('Hero\t勇者\n')
            pcfg.module.llm_glossary_path = glossary_path
            pcfg.module.llm_glossary_mode = LLMGlossaryMode.Matching
            pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
            project = self._project(3)
            self._complete(project, '001.png')

            with mock.patch(
                'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
                return_value=1,
            ):
                self._successful_request(
                    project, '002.png', self.profile,
                )
                self._complete(project, '002.png')
                with open(glossary_path, 'w', encoding='utf-8') as glossary_file:
                    glossary_file.write('Hero\t勇者\nMage\t法师\n')
                pcfg.module.llm_glossary_mode = LLMGlossaryMode.All
                context = self._snapshot_request_context(
                    project, '003.png', self.profile,
                )

        self.assertIsNone(context.diagnostic.rebuild_reason)
        self.assertEqual(context.diagnostic.action, ContextAction.GROW)
        self.assertTrue(all(
            'GLOSSARY:' not in content
            for page in context.history
            for _role, content in page.messages
        ))
        messages, _ = self._assemble_request(
            ['Current page'],
            self.profile,
            request_context=context,
        )
        self.assertIn('"source":"Mage"', messages[1]['content'])
        self.assertNotIn(glossary_path, repr(self.translator._history_window.key))

    def test_runtime_window_retains_only_immutable_snapshots(self):
        project = self._project(3)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        self._complete(project, '001.png')
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            self._successful_request(
                project, '002.png', self.profile,
            )

        window = self.translator._history_window
        self.assertIs(window.key.load_identity, project.load_identity)
        self.assertEqual(
            window.history[0].snapshot,
            HistoryPage('001.png', ('source-1',), ('target-1',)),
        )
        self.assertTrue(all(
            isinstance(text, str)
            for page in window.history
            for text in page.snapshot.sources + page.snapshot.translations
        ))

        self.translator.unload_model()
        self.assertIsNone(self.translator._history_window)

    def test_reconstructed_history_matches_preprocessed_current_prompt(self):
        block = _block('Hero returns')
        captured_messages = []
        source_substitutions = [{
            'keyword': 'Hero',
            'sub': 'Champion',
            'use_reg': False,
            'case_sens': True,
        }]

        def request_translation(_profile, messages, **_usage):
            captured_messages.append(messages)
            return '{"translations":[{"id":1,"translation":"勇者归来"}]}'

        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_glossary_path = ''
        with mock.patch.object(
            pcfg,
            'pre_mt_sublist',
            source_substitutions,
        ), mock.patch.object(
            pcfg,
            'mt_sublist',
            [],
        ), mock.patch.object(
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
            side_effect=request_translation,
        ):
            self.translator.translate_textblk_lst([block])
            current_prompt = captured_messages[0][-1]['content']

            project = SimpleNamespace(
                pages={
                    '001.png': [block],
                    '002.png': [_block('Next page')],
                },
                _image_info={
                    '001.png': {
                        'finish_code': RunStatus.FIN_TRANSLATE,
                        'translation_target': '简体中文',
                    },
                    '002.png': {'finish_code': 0},
                },
            )
            pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
            with mock.patch(
                'ballontranslator.modules.translators.trans_llm.render_history_page',
                wraps=render_history_page,
            ) as render_history, mock.patch(
                'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
                return_value=1,
            ):
                context = self._snapshot_request_context(
                    project,
                    '002.png',
                    self.profile,
                )
                history_messages, _ = self._assemble_request(
                    ['Next page'],
                    self.profile,
                    request_context=context,
                )

        self.assertEqual(block.get_text(), 'Hero returns')
        self.assertIn('"source": "Champion returns"', current_prompt)
        self.assertEqual(history_messages[1]['content'], current_prompt)
        self.assertEqual(render_history.call_count, 1)

    def test_selected_request_commits_only_when_covering_page_sources(self):
        project = self._project(2)
        project.pages['002.png'].extend((
            _block('second-source', 'second-target'),
            _block('', ''),
        ))
        self._complete(project, '001.png')
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        responses = (
            render_assistant_response(('translated',)),
            render_assistant_response(
                ('translated', 'second-translated'),
            ),
        )

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
            side_effect=responses,
        ) as request:
            self.translator.translate_textblk_lst(
                project.pages['002.png'][:1],
                project=project,
                page_key='002.png',
            )
            selected_messages = request.call_args.args[1]
            self.assertIsNone(self.translator._history_window)

            self.translator.translate_textblk_lst(
                project.pages['002.png'][:2],
                project=project,
                page_key='002.png',
            )

        self.assertTrue(any(
            message['role'] == 'assistant'
            for message in selected_messages
        ))
        self.assertEqual(
            self.translator._history_window.request_page_key,
            '002.png',
        )

    def test_next_page_history_uses_fully_finalized_previous_translation(self):
        project = ProjImgTrans()
        project.pages = {
            '001.png': [_block('First source')],
            '002.png': [_block('Second source')],
        }
        project._image_info = {
            '001.png': {'finish_code': 0},
            '002.png': {'finish_code': 0},
        }
        captured_messages = []
        responses = iter(
            (
                '{"translations":[{"id":1,"translation":"未処理"}]}',
                '{"translations":[{"id":1,"translation":"第二"}]}',
            )
        )

        def request_translation(_profile, messages, **_usage):
            captured_messages.append(messages)
            return next(responses)
        result_substitutions = [{
            'keyword': '未処理',
            'sub': '完了',
            'use_reg': False,
            'case_sens': True,
        }]

        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        with mock.patch.object(
            pcfg,
            'pre_mt_sublist',
            [],
        ), mock.patch.object(
            pcfg,
            'mt_sublist',
            result_substitutions,
        ), mock.patch.object(
            pcfg,
            'let_letter_case',
            OCRTextPostprocess.NONE,
        ), mock.patch.object(
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
            side_effect=request_translation,
        ):
            thread = TranslateThread()
            thread.translator = self.translator
            self.assertTrue(
                thread._translate_page(project, '001.png')
            )
            self.assertTrue(
                thread._translate_page(project, '002.png')
            )

        self.assertEqual(project.pages['001.png'][0].translation, '完了')
        history_response = next(
            message['content']
            for message in captured_messages[1]
            if message['role'] == 'assistant'
        )
        self.assertIn('完了', history_response)
        self.assertNotIn('未処理', history_response)

    def test_token_count_uses_model_encoder_or_deterministic_fallback(self):
        messages = [{'role': 'user', 'content': 'abcdefgh你'}]
        encoding = SimpleNamespace(encode=lambda _text: [1, 2])
        with mock.patch(
            'ballontranslator.modules.context.token_usage.'
            '_token_encoding_for_model',
            return_value=encoding,
        ):
            recognized = messages_token_count(
                messages,
                'known-model',
            )
        with mock.patch(
            'ballontranslator.modules.context.token_usage.'
            '_token_encoding_for_model',
            return_value=None,
        ):
            fallback = messages_token_count(
                messages,
                'custom-model',
            )

        self.assertEqual(recognized, MESSAGE_TOKEN_OVERHEAD + 2)
        self.assertEqual(fallback, MESSAGE_TOKEN_OVERHEAD + 3)

    def test_context_overflow_evicts_whole_pages_and_commits_after_success(self):
        project = self._project(4)
        for page_key in ('001.png', '002.png', '003.png'):
            self._complete(project, page_key)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_prior_context_token_budget = 5
        pcfg.module.llm_glossary_mode = LLMGlossaryMode.All

        with tempfile.TemporaryDirectory() as directory:
            glossary_path = os.path.join(directory, 'terms.tsv')
            with open(glossary_path, 'w', encoding='utf-8') as glossary_file:
                glossary_file.write('SecretTerm\tSecretTranslation\n')
            pcfg.module.llm_glossary_path = glossary_path
            debug_lines = []
            self.translator.logger = SimpleNamespace(
                debug=debug_lines.append,
                info=lambda *_args: None,
                warning=lambda *_args: None,
                error=lambda *_args: None,
            )
            with mock.patch(
                'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
                return_value=1,
            ):
                context = self._snapshot_request_context(
                    project,
                    '004.png',
                    self.profile,
                )
            with mock.patch.object(
                self.translator,
                '_request_translation',
                side_effect=(
                    ContextLengthError('maximum context length exceeded'),
                    '{"translations":[{"id":1,"translation":"translated"}]}',
                ),
            ) as request:
                result = self._translate(
                    ['current-source'],
                    profile=self.profile,
                    request_context=context,
                )

        first_messages = request.call_args_list[0].args[1]
        recovered_messages = request.call_args_list[1].args[1]
        self.assertEqual(result, ['translated'])
        self.assertEqual(first_messages[:2], recovered_messages[:2])
        self.assertEqual(first_messages[-1], recovered_messages[-1])
        self.assertEqual(
            [page.page_key for page in self.translator._history_window.history],
            ['002.png', '003.png'],
        )
        self.assertEqual(context.history[1:], self.translator._history_window.history)
        diagnostic_text = '\n'.join(debug_lines)
        self.assertIn('action=context-recovery', diagnostic_text)
        self.assertNotIn('current-source', diagnostic_text)
        self.assertNotIn('SecretTerm', diagnostic_text)
        self.assertNotIn('SecretTranslation', diagnostic_text)

    def test_context_overflow_without_history_is_reraised(self):
        context = RequestContext(
            history=(),
            glossary=(),
            glossary_mode=LLMGlossaryMode.Matching,
            history_budget=10,
        )
        with mock.patch.object(
            self.translator,
            '_request_translation',
            side_effect=ContextLengthError('maximum context length exceeded'),
        ) as request:
            with self.assertRaisesRegex(
                ContextLengthError,
                'maximum context length exceeded',
            ):
                self._translate(
                    ['source'],
                    profile=self.profile,
                    request_context=context,
                )
        self.assertEqual(request.call_count, 1)

    def test_unrelated_request_failure_does_not_shrink_window(self):
        project = self._project(3)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        self.translator.set_param_value('retry attempts', 1)
        self._complete(project, '001.png')
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=1,
        ):
            self._successful_request(project, '002.png')
            self._complete(project, '002.png')
            committed_window = self.translator._history_window
            context = self._snapshot_request_context(
                project,
                '003.png',
                self.profile,
            )

        with mock.patch.object(
            self.translator,
            '_request_translation',
            side_effect=RuntimeError('rate limited'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'rate limited'):
                self._translate(
                    ['source'],
                    profile=self.profile,
                    request_context=context,
                )

        self.assertIs(self.translator._history_window, committed_window)

    def test_failed_eviction_preserves_committed_window_and_retry_messages(self):
        project = self._project(8)
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_prior_context_token_budget = 10
        self.translator.set_param_value('retry attempts', 1)

        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=2,
        ):
            for index in range(1, 7):
                page_key = f'{index:03}.png'
                self._successful_request(project, page_key)
                self._complete(project, page_key)

            committed_window = self.translator._history_window
            eviction_context = self._snapshot_request_context(
                project,
                '007.png',
                self.profile,
            )
            self.assertEqual(
                eviction_context.diagnostic.action,
                ContextAction.EVICT,
            )
            self.assertIs(self.translator._history_window, committed_window)

            with mock.patch.object(
                self.translator,
                '_request_translation',
                side_effect=RuntimeError('provider failed'),
            ) as failed_request:
                with self.assertRaisesRegex(RuntimeError, 'provider failed'):
                    self._translate(
                        ['source-7'],
                        profile=self.profile,
                        request_context=eviction_context,
                    )
            failed_messages = failed_request.call_args.args[1]
            self.assertIs(self.translator._history_window, committed_window)

            retry_context = self._snapshot_request_context(
                project,
                '007.png',
                self.profile,
            )
            retry_response = render_assistant_response(
                ('target-7',),
            )
            with mock.patch.object(
                self.translator,
                '_request_translation',
                return_value=retry_response,
            ) as retry_request:
                self._translate(
                    ['source-7'],
                    profile=self.profile,
                    request_context=retry_context,
                )

        self.assertEqual(retry_context.history, eviction_context.history)
        self.assertEqual(retry_request.call_args.args[1], failed_messages)
        self.assertIsNot(self.translator._history_window, committed_window)
        self.assertEqual(
            self.translator._history_window.history,
            eviction_context.history,
        )


if __name__ == '__main__':
    unittest.main()

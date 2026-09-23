import copy
import json
import unittest
from unittest import mock

from ballontranslator.modules.context.glossary import (
    GLOSSARY_MODE_ALL,
    GlossaryEntry,
)
from ballontranslator.modules.context.history import HistoryPage
from ballontranslator.modules.context.translation_context import (
    MemoryCheckpoint,
    PageSummary,
    RequestContext,
)
from ballontranslator.modules.translators.llm_translation_contract import (
    InvalidNumTranslations,
    TranslationPromptSpec,
    assemble_translation_request,
    parse_translation_response,
    render_history_page,
    translation_json_schema,
    translation_cache_messages,
    translation_system_prompt,
)


class LLMTranslationContractTest(unittest.TestCase):
    def test_cache_boundaries_survive_growth_and_leave_volatile_input_unmarked(self) -> None:
        prefix = [
            {'role': 'system', 'content': 'contract'},
            {'role': 'system', 'content': 'glossary'},
            {'role': 'system', 'content': 'memory'},
        ]
        current = {'role': 'user', 'content': [
            {'type': 'text', 'text': 'current page'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
        ]}
        previous = None
        previous_messages = None
        for count in range(5):
            messages = prefix + [
                {'role': 'user', 'content': render_history_page(
                    HistoryPage(str(page), (f'source {page}',), (f'translation {page}',)),
                    'test-model',
                ).content} for page in range(count)
            ] + [current]
            original = copy.deepcopy(messages)
            marked = translation_cache_messages(messages)
            self.assertEqual(messages, original)
            self.assertIs(marked[-1], current)
            boundaries = [i for i, message in enumerate(marked)
                          if isinstance(message['content'], list)
                          and 'prompt_cache_breakpoint' in message['content'][0]]
            self.assertLessEqual(len(boundaries), 4)
            self.assertEqual(boundaries[:2], [1, 2])
            for index in boundaries:
                self.assertIn(marked[index]['role'], ('system', 'user'))
                self.assertEqual(marked[index]['content'][0]['text'], original[index]['content'])
            if previous is not None:
                old_boundary = previous[-1]
                self.assertIn(old_boundary, boundaries)
                # Cache metadata may rotate, but all preceding text blocks stay fixed.
                for old, new in zip(previous_messages[:old_boundary + 1], marked[:old_boundary + 1]):
                    old, new = copy.deepcopy(old), copy.deepcopy(new)
                    old['content'][0].pop('prompt_cache_breakpoint', None)
                    new['content'][0].pop('prompt_cache_breakpoint', None)
                    self.assertEqual(old, new)
            previous = boundaries
            previous_messages = marked

    def test_history_records_contain_only_reference_data_and_match_budget(self) -> None:
        for summary in ('', 'Scene with a newline.\nMore context.'):
            for sources, translations in ((('心', '"quoted"'), ('heart', '译文')), ((), ())):
                with self.subTest(summary=summary, sources=sources):
                    page = HistoryPage('001.png', sources, translations, summary)
                    with mock.patch(
                        'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
                        return_value=19,
                    ) as count:
                        rendered = render_history_page(page, 'test-model')
                    expected = {'page_id': '001.png', 'translations': [
                        {'source': source, 'translation': translation}
                        for source, translation in zip(sources, translations)
                    ]}
                    if summary:
                        expected['summary'] = summary
                    self.assertEqual(json.loads(rendered.content), expected)
                    self.assertEqual(rendered.token_count, 19)
                    count.assert_called_once_with([{'role': 'user', 'content': rendered.content}], 'test-model')
        with self.assertRaises(ValueError):
            render_history_page(HistoryPage('bad', ('source',), ()), 'test-model')

    def test_disabled_features_keep_numeric_response_contract(self):
        profile_prompt = 'Keep JSON example {"x": 1}.'
        spec = TranslationPromptSpec(
            'Japanese',
            'Simplified Chinese',
            translation_system_prompt(
                profile_prompt,
                'Simplified Chinese',
            ),
            False,
        )

        messages, prompt = assemble_translation_request(
            ('心',),
            prompt_spec=spec,
        )

        self.assertEqual(
            [message['role'] for message in messages],
            ['system', 'user'],
        )
        self.assertIn('{"1":"Translated text"}', spec.system_prompt)
        self.assertNotIn('page_summary', spec.system_prompt)
        self.assertIn(
            'Additional translation instructions:\nKeep JSON example {"x": 1}.',
            spec.system_prompt,
        )
        self.assertIn('"source": "心"', prompt)

    def test_summary_contract_uses_target_language(self) -> None:
        system_prompt = translation_system_prompt(
            '',
            'Simplified Chinese',
            summary_enabled=True,
        )

        self.assertIn(
            '"page_summary":"Short factual page summary in Simplified Chinese"',
            system_prompt,
        )
        self.assertIn(
            'Write page_summary in Simplified Chinese',
            system_prompt,
        )
        self.assertNotIn('English page memory', system_prompt)

    def test_combined_context_order_and_current_suffix(self):
        spec = TranslationPromptSpec(
            'Japanese',
            'English',
            'stable system',
            True,
        )
        with mock.patch(
            'ballontranslator.modules.translators.llm_translation_contract.messages_token_count',
            return_value=7,
        ):
            history = render_history_page(
                HistoryPage(
                    '002.png',
                    ('old source',),
                    ('old target',),
                    'Old page summary.',
                ),
                'test-model',
            )
        context = RequestContext(
            history=(history,),
            glossary=(GlossaryEntry('Hero', 'Brave'),),
            glossary_mode=GLOSSARY_MODE_ALL,
            memory=MemoryCheckpoint(
                'Covers page summaries: 001.png\n\nThe hero arrived.',
                ('001.png',),
                4,
            ),
            page_summaries=(PageSummary('003.png', 'Current clue.'),),
        )
        image_part = {
            'type': 'image_url',
            'image_url': {'url': 'data:image/jpeg;base64,AA=='},
        }

        messages, prompt = assemble_translation_request(
            ('Hero speaks',),
            prompt_spec=spec,
            request_context=context,
            image_part=image_part,
        )

        self.assertEqual(
            [message['role'] for message in messages],
            ['system', 'system', 'system', 'user', 'user'],
        )
        self.assertEqual(messages[0]['content'], spec.system_prompt)
        self.assertIn('"source":"Hero"', messages[1]['content'])
        self.assertIn('Compacted translation memory', messages[2]['content'])
        self.assertEqual(json.loads(messages[3]['content']), {
            'page_id': '002.png',
            'translations': [{'source': 'old source', 'translation': 'old target'}],
            'summary': 'Old page summary.',
        })
        self.assertIn('Current clue.', prompt)
        self.assertIn('infer the natural comic reading order', prompt)
        self.assertIn('mapped to its original input ID', prompt)
        self.assertIsInstance(messages[-1]['content'], list)
        self.assertEqual(messages[-1]['content'][0], {'type': 'text', 'text': prompt})
        self.assertIs(messages[-1]['content'][1], image_part)

    def test_schema_shapes_are_exact(self):
        translation_schema = {
            'type': 'object',
            'properties': {
                '1': {'type': 'string'},
                '2': {'type': 'string'},
            },
            'required': ['1', '2'],
            'additionalProperties': False,
        }

        self.assertEqual(translation_json_schema(2), translation_schema)
        self.assertEqual(
            translation_json_schema(2, summary_enabled=True),
            {
                'type': 'object',
                'properties': {
                    'page_summary': {'type': 'string'},
                    'translations': translation_schema,
                },
                'required': ['page_summary', 'translations'],
                'additionalProperties': False,
            },
        )
        self.assertEqual(
            list(translation_json_schema(2, summary_enabled=True)['properties']),
            ['page_summary', 'translations'],
        )

    def test_parser_accepts_numeric_map_and_legacy_wrapper_list(self):
        numeric = parse_translation_response('{"2":"spirit","1":"heart"}', 2)
        legacy = parse_translation_response(
            '{"translations":['
            '{"id":1,"translation":"heart"},'
            '{"id":2,"translation":"spirit"}]}',
            2,
        )

        self.assertEqual(numeric.translations, ('heart', 'spirit'))
        self.assertEqual(legacy.translations, ('heart', 'spirit'))

    def test_array_contract_keeps_schema_and_prompt_consistent(self) -> None:
        for summary in (False, True):
            with self.subTest(summary=summary):
                schema = translation_json_schema(1, summary_enabled=summary, array_response=True)
                self.assertEqual(schema, translation_json_schema(13, summary_enabled=summary, array_response=True))
                if summary:
                    self.assertEqual(schema, translation_json_schema(0, summary_enabled=True, array_response=True))
                expected_properties = ['page_summary', 'translations'] if summary else ['translations']
                self.assertEqual(list(schema['properties']), expected_properties)
                self.assertEqual(schema['properties']['translations'], {
                    'type': 'array', 'items': {
                        'type': 'object',
                        'properties': {'id': {'type': 'integer'}, 'translation': {'type': 'string'}},
                        'required': ['id', 'translation'], 'additionalProperties': False,
                    },
                })
                prompt = translation_system_prompt('', 'English', history_enabled=True,
                                                   summary_enabled=summary, array_response=True)
                self.assertIn('"translations":[{"id":1,"translation":"Translated text"}]', prompt)
                self.assertNotIn('as keys', prompt)
                self.assertNotIn('object keys', prompt)

    def test_array_parser_rejects_duplicate_missing_extra_and_coerced_items(self) -> None:
        valid = {'id': 1, 'translation': 'heart'}
        for items in (
            [valid, valid], [], [valid, {'id': 2, 'translation': 'extra'}],
            [{'id': True, 'translation': 'heart'}],
            [{'id': 1.0, 'translation': 'heart'}], [{'id': '1', 'translation': 'heart'}],
            [{'id': 1, 'translation': 3}], [{'id': 1, 'translation': None}],
            [None], {'1': 'heart'},
        ):
            with self.subTest(items=items), self.assertRaises((InvalidNumTranslations, ValueError)):
                parse_translation_response(json.dumps({'translations': items}), 1, array_response=True)
        parsed = parse_translation_response(
            '{"translations":[{"id":2,"translation":"second"},{"id":1,"translation":"first"}]}',
            2, array_response=True,
        )
        self.assertEqual(parsed.translations, ('first', 'second'))
        self.assertEqual(parse_translation_response(
            '{"page_summary":"Scene.","translations":[]}', 0, array_response=True,
        ).page_summary, 'Scene.')

    def test_parser_normalizes_without_truncating_optional_summary(self) -> None:
        body = ' '.join(['detail'] * 501)
        summary = '  scene\n\tmemory  ' + body

        parsed = parse_translation_response(
            json.dumps({
                'page_summary': summary,
                'translations': {'1': 'heart'},
            }),
            1,
        )
        non_string = parse_translation_response(
            '{"translations":{"1":"heart"},"page_summary":null}',
            1,
        )

        self.assertEqual(
            parsed.page_summary,
            'scene memory ' + body,
        )
        self.assertEqual(non_string.page_summary, '')

    def test_empty_input_requires_only_a_usable_summary(self) -> None:
        for payload in ({}, {'translations': 'ignored'}, {'translations': {'99': []}}):
            with self.subTest(payload=payload):
                parsed = parse_translation_response(
                    json.dumps({**payload, 'page_summary': '  The train\narrives. '}),
                    0,
                )
                self.assertEqual(parsed.translations, ())
                self.assertEqual(parsed.page_summary, 'The train arrives.')
        for summary in (None, '', ' \n ', []):
            with self.subTest(summary=summary):
                with self.assertRaisesRegex(ValueError, 'no usable page_summary'):
                    parse_translation_response(json.dumps({'page_summary': summary}), 0)

        # A summary alone must not bypass translation validation on text pages.
        with self.assertRaisesRegex(ValueError, 'Unsupported translations payload'):
            parse_translation_response(
                '{"translations":"ignored","page_summary":"The train arrives."}',
                1,
            )

    def test_parser_preserves_fenced_and_prose_object_compatibility(self):
        fenced = parse_translation_response('```json\n{"1":"heart"}\n```', 1)
        prose = parse_translation_response('Answer: {"1":"heart"}.', 1)

        self.assertEqual(fenced.translations, ('heart',))
        self.assertEqual(prose.translations, ('heart',))

    def test_parser_preserves_strict_id_and_ambiguous_payload_failures(self):
        with self.assertRaisesRegex(
            InvalidNumTranslations,
            r"Expected ids 1-2, got \[1\]",
        ):
            parse_translation_response('{"1":"heart"}', 2)
        with self.assertRaises(json.JSONDecodeError):
            parse_translation_response('{"1":"heart"} {"1":"spirit"}', 1)
        with self.assertRaisesRegex(ValueError, 'Unsupported translations payload'):
            parse_translation_response(
                '{"translations":"bad","1":"heart"}',
                1,
            )


if __name__ == '__main__':
    unittest.main()

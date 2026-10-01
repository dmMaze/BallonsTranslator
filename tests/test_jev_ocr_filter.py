import json
import threading
import unittest
from unittest.mock import patch

import numpy as np
import requests

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.ocr.base import OCRBase
from ballontranslator.modules.ocr import jev_filter
from ballontranslator.modules.ocr.jev_filter import filter_ocr_texts
from ballontranslator.utils.config import ModuleConfig, json_dump_program_config, pcfg
from ballontranslator.utils.llm_profiles import default_profile, profile_by_id, store_api_key
from ballontranslator.utils.secret_store import SecretStore, is_portable_secret
from ballontranslator.utils.textblock import TextBlock


WHOLE = {'breakdown': .99, 'usable': .01, 'expression': .01, 'edit_safe': .99}
KEEP = {'breakdown': .01, 'usable': .99, 'expression': .01, 'edit_safe': .01}
MIXED = dict(WHOLE, usable=.99, edit_safe=.01)


def choice_answer(options: dict, selected: str) -> dict:
    return {'type': 'choice', 'choice': selected, 'confidence': .99,
            'probabilities': {key: .99 if key == selected else .01 / max(1, len(options) - 1)
                              for key in options}}


class JevOCRFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        for mock in (patch.dict('os.environ', {'TYPESAFE_API_KEY': 'test-key', 'OPENROUTER_API_KEY': '', 'JEV_API_PROVIDER': ''}),
                     patch.object(pcfg.module, 'ocr_jev_provider', 'typesafe'),
                     patch.object(pcfg.module, 'ocr_jev_typesafe_api_key', ''),
                     patch.object(pcfg.module, 'llm_profiles', [default_profile('OpenRouter')])):
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.object(jev_filter.requests, 'post')
        self.post = mock.start()
        self.addCleanup(mock.stop)
        self.response = self.post.return_value.__enter__.return_value
        self.response.status_code = 200
        self.response.json.side_effect = self.answers
        self.default_scores, self.scores = WHOLE, {}
        self.preserve, self.notation = .01, .01
        self.convention, self.edge_notation = .01, .01
        self.edits = {}
        self.repetition = (.01, .01)

    def answers(self) -> dict:
        payload = self.post.call_args.kwargs['json']
        state, answers = payload['state'], {}
        for key, question in payload['questions'].items():
            if 'repeat_edit' in state:
                probability, sound = self.repetition
                if key == 'choice':
                    answer = choice_answer(question['criteria'], 'shortened' if probability > .5 else 'original')
                    answer['probabilities'] = {'shortened': probability, 'original': 1 - probability}
                else:
                    answer = choice_answer(question['criteria'], 'sound' if sound > .5 else 'content')
                    answer['probabilities'] = {'sound': sound, 'content': 1 - sound, 'uncertain': 0.0}
                answers[key] = answer
                continue
            if question['type'] == 'choice':
                selected = 'cleaned' if key == 'choice' else next(iter(question['criteria']))
                answer = choice_answer(question['criteria'], selected)
                if key == 'choice':
                    probability = self.edits.get(state['edits'][0]['removed'], (.99, .01, .01))[0]
                    answer['probabilities'] = {'cleaned': probability, 'original': 1 - probability}
            else:
                if key in ('loss', 'cuts_word'):
                    score = self.edits.get(state['edits'][0]['removed'], (.99, .01, .01))[1 if key == 'loss' else 2]
                elif key.isdigit():
                    if question is jev_filter._CONVENTION:
                        score = self.convention
                    elif set(payload['questions']) == {'0', '1'}:
                        score = self.edge_notation
                    else:
                        score = self.preserve if question is jev_filter._PRESERVE else self.notation
                    if callable(score):
                        score = score(state['items'][0])
                else:
                    index, kind = key.split('_', 1)
                    score = self.scores.get(state['items'][int(index)]['text'], self.default_scores)[kind]
                answer = {'type': 'noul', 'noul': score, 'confidence': .99}
            answers[key] = answer
        return {'answers': answers}

    def test_whole_noise_and_usable_text_are_distinct(self) -> None:
        texts = ['IC' * 100, 'Ha ha ha!', '名前', 'ATK 30 → 45', 'Welcome home.']
        self.scores.update({text: KEEP for text in texts[1:]})
        self.assertEqual(filter_ocr_texts(texts), ['', *texts[1:]])
        self.assertEqual(texts[0], 'IC' * 100)

    def test_no_key_or_blank_input_does_not_send(self) -> None:
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': ''}), \
                patch.object(jev_filter.LOGGER, 'info') as log:
            self.assertEqual(filter_ocr_texts(['ICIC']), ['ICIC'])
            record = next(json.loads(c.args[1]) for c in reversed(log.call_args_list)
                          if c.args[0] == 'Jev OCR filter audit: %s')
            self.assertEqual(record['reason'], 'missing_api_key')
        with patch.object(jev_filter.LOGGER, 'info') as log:
            self.assertEqual(filter_ocr_texts(['', '  ']), ['', '  '])
            record = next(json.loads(c.args[1]) for c in reversed(log.call_args_list)
                          if c.args[0] == 'Jev OCR filter audit: %s')
            self.assertEqual(record['status'], 'skipped')
            self.assertEqual(record['reason'], 'empty_input')
            self.assertEqual(record['usage']['requests'], 0)
        self.post.assert_not_called()

    def test_usage_counts_real_requests_but_not_cached_decisions(self) -> None:
        client = jev_filter._JevFilter('endpoint', 'jev-latest', 'key', None)
        self.response.json.side_effect = None
        self.response.json.return_value = {
            'answers': {'keep': {'type': 'noul', 'noul': .99}},
            'usage': {'input_tokens': 120, 'output_tokens': 8},
        }
        with patch.object(jev_filter.LOGGER, 'info') as log:
            for _ in range(2):
                client.ask({'text': 'Hello'}, {'keep': {'type': 'noul'}})
        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(client.usage, dict(requests=1, input_tokens=120, reported_input_requests=1,
                                           cost_nano_usd=5040, estimated_cost_requests=1))
        usage_logs = [call for call in log.call_args_list if call.args[0].startswith('Jev OCR filter usage:')]
        self.assertIn('requests=1, input_tokens=%s', usage_logs[0].args[0])
        self.assertEqual(usage_logs[0].args[2:], (120, '0.000005040', 'estimated'))
        self.assertNotIn('completion', usage_logs[0].args[0])
        events = [json.loads(call.args[1]) for call in log.call_args_list
                  if call.args[0] == 'Jev OCR filter audit: %s']
        self.assertEqual([e['event'] for e in events], ['request', 'cache_hit'])
        self.assertEqual(events[0]['request_id'], events[1]['request_id'])
        self.assertEqual(events[0]['status'], 'valid')
        self.assertEqual((events[0]['cost_usd'], events[0]['cost_source']), ('0.000005040', 'estimated'))

    def test_cost_reports_estimates_zero_and_missing_usage_for_both_providers(self) -> None:
        self.default_scores = KEEP
        cases = [
            ({'input_tokens': 4475}, '0.000187950', 'estimated', 0),
            ({'input_tokens': 4475, 'cost': 0.000019992}, '0.000019992', 'reported', 0),
            ({'input_tokens': 4475, 'cost': 0}, '0.000000000', 'reported', 0),
            ({'cost': '0.000000042'}, '0.000000042', 'reported', 0),
            ({'input_tokens': 0}, '0.000000000', 'estimated', 0),
            ({'output_tokens': 999}, 'unavailable', 'none', 1),
            (None, 'unavailable', 'none', 1),
        ]
        cases += [({'cost': invalid}, 'unavailable', 'none', 1)
                  for invalid in (True, -1, 'bad', 'NaN', 'Infinity', '1e999999999', [], {})]
        cases.append(({'cost': -1, 'input_tokens': 100}, '0.000004200', 'estimated', 0))
        for provider in ('typesafe', 'openrouter'):
            for usage, cost, source, unpriced in cases:
                with self.subTest(provider=provider, usage=usage), \
                        patch.object(pcfg.module, 'ocr_jev_provider', provider), \
                        patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
                    self.response.json.side_effect = lambda: dict(self.answers(), usage=usage)
                    totals = {}
                    self.assertEqual(filter_ocr_texts(['Hello'], cleanup_totals=totals), ['Hello'])
                    summary = jev_filter.format_jev_cleanup_totals(totals)
                    self.assertIn(f'cost_usd={cost}, cost_source={source},', summary)
                    self.assertIn(f'unpriced_requests={unpriced},', summary)
                    self.assertEqual(totals['requests'], 1)
                    self.assertNotIn('output_tokens', totals)
                    self.assertNotIn('total_tokens', totals)


    def test_invalid_decision_and_http_error_still_count_reported_usage(self) -> None:
        self.response.json.side_effect = None
        self.response.json.return_value = {'answers': {}, 'usage': {'input_tokens': 10, 'output_tokens': 2}}
        for error in (None, requests.HTTPError('429')):
            with self.subTest(error=error), patch.object(jev_filter.LOGGER, 'info') as log:
                self.response.raise_for_status.side_effect = error
                totals = {}
                self.assertEqual(filter_ocr_texts(['Hello'], cleanup_totals=totals), ['Hello'])
                self.assertEqual(totals['requests'], 1)
                self.assertEqual(totals['input_tokens'], 10)
                self.assertEqual(totals['reported_input_requests'], 1)
                self.assertEqual(totals['cost_nano_usd'], 420)
                self.assertEqual(totals['estimated_cost_requests'], 1)
                self.assertNotIn('total_tokens', totals)
                events = [json.loads(call.args[1]) for call in log.call_args_list
                          if call.args[0] == 'Jev OCR filter audit: %s']
                request = next(e for e in events if e['event'] == 'request')
                self.assertEqual(request['status'], 'error')
                self.assertEqual(request['error_type'], 'HTTPError' if error else 'KeyError')
                self.assertEqual(events[-1]['blocks'][0]['reason'], 'request_error')

    def test_missing_input_usage_and_network_failure_still_count_requests(self) -> None:
        self.default_scores = KEEP
        for usage, error, reports in ((None, None, 0),
                                      ({'output_tokens': 20, 'total_tokens': 20}, None, 0),
                                      ({'input_tokens': 0, 'output_tokens': 20}, None, 1),
                                      (None, requests.Timeout(), 0)):
            with self.subTest(usage=usage, error=error):
                self.post.side_effect = error
                self.response.json.side_effect = lambda: dict(self.answers(), usage=usage)
                totals = {}
                self.assertEqual(filter_ocr_texts(['Hello'], cleanup_totals=totals), ['Hello'])
                self.assertEqual(totals['requests'], 1)
                self.assertEqual(totals['input_tokens'], 0)
                self.assertEqual(totals['reported_input_requests'], reports)
                self.assertIn(f'missing_input_usage_requests={1 - reports},',
                              jev_filter.format_jev_cleanup_totals(totals))

    def test_audit_links_source_questions_scores_and_final_block_outcomes(self) -> None:
        texts = ['ICIC', 'Hello', '  ', 'ローナ' * 3 + 'オ' * 122]
        self.scores = {'Hello': KEEP, texts[3]: dict(KEEP, expression=.7, breakdown=.3)}
        self.repetition = (.99, .96)
        totals = {}
        with patch.object(jev_filter.LOGGER, 'info') as log:
            result = filter_ocr_texts(texts, source='Chapter 51/30.jpg', cleanup_totals=totals)
        self.assertEqual(totals, dict(requests=self.post.call_count, input_tokens=0, reported_input_requests=0,
                                     calls=1, changed_calls=1, cleared_blocks=1,
                                     trimmed_blocks=1, kept_blocks=2, removed_characters=118))
        summary = next(c for c in log.call_args_list if c.args[0].startswith('Jev OCR cleanup:'))
        self.assertIn('changed_blocks=2 (cleared=1, trimmed=1)', summary.args[-1])
        self.assertIn('removed_characters=118', summary.args[-1])
        self.assertIn('changed_calls=1/1', summary.args[-1])
        events = [json.loads(call.args[1]) for call in log.call_args_list
                  if call.args[0] == 'Jev OCR filter audit: %s']
        self.assertEqual(len({e['audit_id'] for e in events}), 1)
        self.assertTrue(all(e['source'] == 'Chapter 51/30.jpg' for e in events))
        request = next(e for e in events if e['event'] == 'request')
        self.assertEqual([p['block'] for p in request['scope']], [1, 2, 4])
        self.assertEqual(request['answers']['0_breakdown']['noul'], .99)
        self.assertIn('instructions', request['request']['questions']['0_breakdown'])
        self.assertIn('criteria', request['request']['questions']['0_breakdown'])
        self.assertEqual(request['http_status'], 200)
        final = events[-1]
        self.assertEqual((final['event'], final['status']), ('finished', 'completed'))
        self.assertEqual([b['original'] for b in final['blocks']], texts)
        self.assertEqual([b['retained'] for b in final['blocks']], result)
        self.assertEqual([b['action'] for b in final['blocks']], ['cleared', 'kept', 'kept', 'trimmed'])
        self.assertEqual([b['removed_characters'] for b in final['blocks']], [4, 0, 0, 114])
        serialized = json.dumps(events)
        self.assertNotIn('test-key', serialized)
        self.assertNotIn('Authorization', serialized)

    def test_failed_and_cancelled_audits_do_not_claim_rolled_back_edits(self) -> None:
        # The first block clears before a later request fails or cancels the call.
        texts = ['ICIC', 'Hello']
        self.scores['Hello'] = MIXED
        self.response.json.side_effect = lambda: dict(self.answers(), usage={'cost': .000001})
        for error in (requests.Timeout(), LLMRequestStopped()):
            with self.subTest(error=type(error).__name__), \
                    patch.object(jev_filter._JevFilter, 'clean', side_effect=['', error]), \
                    patch.object(jev_filter.LOGGER, 'info') as log:
                totals = {}
                if isinstance(error, LLMRequestStopped):
                    with self.assertRaises(LLMRequestStopped):
                        filter_ocr_texts(texts, cleanup_totals=totals)
                else:
                    self.assertEqual(filter_ocr_texts(texts, cleanup_totals=totals), ['', 'Hello'])
                final = next(json.loads(c.args[1]) for c in reversed(log.call_args_list)
                             if c.args[0] == 'Jev OCR filter audit: %s')
                changed = 0 if isinstance(error, LLMRequestStopped) else 1
                self.assertEqual(totals, dict(requests=1, input_tokens=0, reported_input_requests=0,
                                             cost_nano_usd=1000, reported_cost_requests=1,
                                             calls=1, changed_calls=changed, cleared_blocks=changed,
                                             trimmed_blocks=0, kept_blocks=2 - changed,
                                             removed_characters=4 * changed))
                if isinstance(error, LLMRequestStopped):
                    self.assertEqual(final['status'], 'cancelled')
                    self.assertEqual([b['retained'] for b in final['blocks']], texts)
                    self.assertTrue(all(b['action'] == 'kept' for b in final['blocks']))
                else:
                    self.assertEqual(final['blocks'][1]['reason'], 'request_error')
                    self.assertEqual(final['blocks'][0]['action'], 'cleared')

    def test_cleanup_counts_blocks_once_despite_cached_requests_and_accumulates_calls(self) -> None:
        totals = {}
        with patch.object(jev_filter.LOGGER, 'info') as log:
            for _ in range(2):
                self.assertEqual(filter_ocr_texts(['ICIC', 'ICIC'], cleanup_totals=totals), ['', ''])
        events = [json.loads(c.args[1]) for c in log.call_args_list if c.args[0] == 'Jev OCR filter audit: %s']
        self.assertTrue(any(e['event'] == 'cache_hit' for e in events))
        self.assertEqual(totals, dict(requests=self.post.call_count, input_tokens=0, reported_input_requests=0,
                                     calls=2, changed_calls=2, cleared_blocks=4,
                                     trimmed_blocks=0, kept_blocks=0, removed_characters=16))
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': ''}):
            self.assertEqual(filter_ocr_texts(['Hello'], cleanup_totals=totals), ['Hello'])
        self.assertEqual(filter_ocr_texts([''], cleanup_totals=totals), [''])
        self.assertEqual(totals, dict(requests=self.post.call_count, input_tokens=0, reported_input_requests=0,
                                     calls=4, changed_calls=2, cleared_blocks=4,
                                     trimmed_blocks=0, kept_blocks=2, removed_characters=16))

    def test_every_nonempty_block_is_screened_without_pattern_gate(self) -> None:
        self.default_scores = KEEP
        texts = ['XQZ', '勇者 XQZPWV', '!@#', '□', 'おはよう']
        self.assertEqual(filter_ocr_texts(texts), texts)
        sent = [item['text'] for call in self.post.call_args_list for item in call.kwargs['json']['state']['items']]
        self.assertEqual(sent, texts)

    def test_invalid_probabilities_or_missing_answers_retain_original(self) -> None:
        text = 'IC' * 100
        for invalid in (True, None, '1', float('nan'), float('inf'), -.1, 1.1, 10 ** 500):
            with self.subTest(invalid=invalid):
                self.default_scores = dict(WHOLE, breakdown=invalid)
                self.assertEqual(filter_ocr_texts([text]), [text])
        self.response.json.side_effect = None
        for payload in ({}, {'answers': []}, {'answers': {}}, {'answers': {'0_breakdown': {'type': 'choice'}}}):
            with self.subTest(payload=payload):
                self.response.json.return_value = payload
                self.assertEqual(filter_ocr_texts([text]), [text])
        self.post.side_effect = requests.Timeout()
        self.assertEqual(filter_ocr_texts([text]), [text])

    def test_unknown_choice_and_incomplete_choice_probabilities_retain_text(self) -> None:
        self.default_scores = MIXED
        normal_answers = self.answers
        for malformed in ('unknown', 'missing_probability'):
            def invalid() -> dict:
                payload = normal_answers()
                if 'edit' in payload['answers']:
                    answer = payload['answers']['edit']
                    if malformed == 'unknown':
                        answer['choice'] = '99999:999999'
                    else:
                        answer['probabilities'].pop(next(iter(answer['probabilities'])))
                return payload
            self.response.json.side_effect = invalid
            self.assertEqual(filter_ocr_texts(['Help ICICIC']), ['Help ICICIC'])

    def test_later_failure_rolls_back_the_entire_block(self) -> None:
        text = 'IC' * 400
        with patch.object(jev_filter._JevFilter, 'clean', side_effect=['', requests.Timeout()]):
            self.assertEqual(filter_ocr_texts([text]), [text])

    def test_failure_is_isolated_to_affected_source_blocks(self) -> None:
        texts = ['ICIC', 'QXQX', 'XQXQ', 'JQJQ', 'VXVX']
        response = self.post.return_value
        def fail_last_block(*args, **kwargs):
            payload = kwargs['json']
            if '0_breakdown' in payload['questions'] and any(
                    item['text'] == texts[-1] for item in payload['state']['items']):
                raise requests.Timeout()
            return response
        self.post.side_effect = fail_last_block
        self.assertEqual(filter_ocr_texts(texts), ['', '', '', '', texts[-1]])

    def test_long_input_is_bounded_and_keeps_adjacent_context(self) -> None:
        text = 'IC' * 4000
        self.assertEqual(filter_ocr_texts([text]), [''])
        batches = [call.kwargs['json'] for call in self.post.call_args_list
                   if '0_breakdown' in call.kwargs['json']['questions']]
        pieces = [item for batch in batches for item in batch['state']['items']]
        self.assertTrue(all(len(item['text']) <= 512 for item in pieces))
        self.assertTrue(pieces[1]['before'])

    def test_whitespace_only_pieces_remain_verbatim_without_model_judgment(self) -> None:
        self.default_scores = KEEP
        text = 'Prefix' + ' ' * 1024 + 'Suffix'
        self.assertEqual(filter_ocr_texts([text]), [text])
        sent = [item['text'] for call in self.post.call_args_list for item in call.kwargs['json']['state']['items']]
        self.assertTrue(all(piece.strip() for piece in sent))

    def test_boundary_search_stays_within_api_choice_limit(self) -> None:
        client = jev_filter._JevFilter('endpoint', 'model', 'key', None)
        client.proposals('ABCDEFGHIJ' * 51, ('', ''))
        options = [q['criteria'] for call in self.post.call_args_list for q in call.kwargs['json']['questions'].values()]
        self.assertTrue(all(1 < len(choices) <= 255 for choices in options))

    def test_large_choice_payloads_use_bounded_semantic_tournaments(self) -> None:
        client = jev_filter._JevFilter('endpoint', 'model', 'key', None)
        options = {str(i): {'removed': 'QXV' * 145, 'cleaned': 'Help!'} for i in range(145)}
        def pick_last() -> dict:
            choices = self.post.call_args.kwargs['json']['questions']['cut']['criteria']
            return {'answers': {'cut': choice_answer(choices, max(choices, key=int))}}
        self.response.json.side_effect = pick_last
        self.assertEqual(client.choose({'original': 'QXV' * 145}, 'cut', 'Choose the best boundary.', options), '144')
        self.assertGreater(self.post.call_count, 1)
        for call in self.post.call_args_list:
            self.assertLessEqual(len(json.dumps(call.kwargs['json'], ensure_ascii=False).encode()), 16000)
        self.post.reset_mock()
        self.assertEqual(client.choose({}, 'cut', 'Choose.', {'only': {}}), 'only')
        self.post.assert_not_called()

    def test_best_semantic_edit_wins_over_longest_approved_span(self) -> None:
        noise = 'IC' * 30
        text = '前😀\n' + noise + '\n後'
        a, b = text.index(noise), text.index(noise) + len(noise)
        self.default_scores = KEEP
        self.scores = {text: MIXED, text[:b]: WHOLE, noise: WHOLE}
        self.edits = {text[:b]: (.66, .28, .28), noise: (.99, .07, .07)}
        with patch.object(jev_filter._JevFilter, 'proposals', return_value=[(0, b), (a, b)]):
            self.assertEqual(filter_ocr_texts([text]), ['前😀\n\n後'])

    def test_long_whole_vote_cannot_hide_useful_interior(self) -> None:
        text = 'IC' * 40 + ' Help me! ' + 'IC' * 40
        self.preserve = lambda item: .99 if 'Help' in item['text'] else .01
        with patch.object(jev_filter._JevFilter, 'choose_span', return_value=None):
            self.assertEqual(filter_ocr_texts([text]), [text])

    def test_intentional_utterance_judgment_protects_the_whole_source_block(self) -> None:
        text = 'No' + 'o' * 1100 + '!'
        self.scores[text[:512]] = dict(KEEP, expression=.9, breakdown=.2)
        # Later pieces may look like meaningless repetition in isolation.
        with patch.object(jev_filter._JevFilter, 'choose_span') as search:
            self.assertEqual(filter_ocr_texts([text, 'ICIC']), [text, ''])
        search.assert_not_called()

    def test_conventional_reaction_overrides_an_incorrect_noise_vote(self) -> None:
        self.convention = lambda item: .66 if item['text'] == 'wwwwwwww' else .01
        self.assertEqual(filter_ocr_texts(['wwwwwwww', 'ICIC']), ['wwwwwwww', ''])

    def test_inflated_cry_keeps_names_and_audible_expression(self) -> None:
        text = 'ローナ' * 3 + 'オ' * 122
        self.scores[text] = dict(KEEP, expression=.70, breakdown=.44)
        self.repetition = (.99, .96)
        self.assertEqual(filter_ocr_texts([text]), ['ローナ' * 3 + 'オ' * 8])

    def test_repeat_shortening_requires_confident_edit_and_sound_judgments(self) -> None:
        text = 'Password: ' + 'AB' * 80
        self.scores[text] = dict(KEEP, expression=.7, breakdown=.3)
        for decision in ((.84, .99), (.99, .84), (.01, .99)):
            with self.subTest(decision=decision):
                self.repetition = decision
                self.assertEqual(filter_ocr_texts([text]), [text])

    def test_normal_sounds_and_numeric_values_are_not_shortening_candidates(self) -> None:
        self.default_scores = dict(KEEP, expression=.9)
        self.repetition = (.99, .99)
        texts = ['ああああああッ！', 'ドドドドドド', 'Ha ha ha!', 'HP: 1' + '0' * 80, 'ATK 30 → 45']
        self.assertEqual(filter_ocr_texts(texts), texts)
        self.assertFalse(any('repeat_edit' in call.kwargs['json']['state'] for call in self.post.call_args_list))

    def test_repeat_shortening_is_bounded_across_source_pieces(self) -> None:
        text = 'No' + 'o' * 1100 + '!'
        self.scores[text[:512]] = dict(KEEP, expression=.9, breakdown=.2)
        self.repetition = (.99, .99)
        self.assertEqual(filter_ocr_texts([text]), ['N' + 'o' * 8 + '!'])
        edits = [call.kwargs['json']['state']['repeat_edit'] for call in self.post.call_args_list
                 if 'repeat_edit' in call.kwargs['json']['state']]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]['copies_before'], 1101)
        self.assertLess(len(edits[0]['original']), 800)

    def test_repeat_request_failure_retains_original(self) -> None:
        text = 'ローナ' * 3 + 'オ' * 122
        self.scores[text] = dict(KEEP, expression=.70, breakdown=.44)
        response = self.post.return_value
        def fail_repeat(*args, **kwargs):
            if 'repeat_edit' in kwargs['json']['state']:
                raise requests.Timeout()
            return response
        self.post.side_effect = fail_repeat
        self.assertEqual(filter_ocr_texts([text]), [text])

    def test_exact_repeat_candidate_preserves_neighboring_stats_and_delimiters(self) -> None:
        noise = 'LJQZXV' * 12
        text = 'HP: 125/200; ' + noise + ' MP: 50/80.'
        self.default_scores = KEEP
        self.scores = {text: MIXED, noise: WHOLE}
        self.assertEqual(filter_ocr_texts([text]), ['HP: 125/200;  MP: 50/80.'])

    def test_meaningful_repeat_does_not_hide_separate_nonrepeating_noise(self) -> None:
        code, noise = 'AB' * 40, 'QX7J9ZV#LJQ0VVZ#XJ8QZ#'
        text = 'Access code: ' + code + ' ' + noise
        start, end = text.index(noise), len(text)
        self.default_scores = KEEP
        self.scores = {text: MIXED, noise: WHOLE}
        with patch.object(jev_filter._JevFilter, 'choose', side_effect=[f'{start}:{end}', str(start), str(end)]):
            self.assertEqual(filter_ocr_texts([text]), ['Access code: ' + code + ' '])

    def test_later_repeat_failure_rolls_back_earlier_shortening(self) -> None:
        text = 'No' + 'o' * 80 + '! Help' + 'p' * 80 + '!'
        self.scores[text] = dict(KEEP, expression=.9)
        self.repetition = (.99, .99)
        response, repeat_calls = self.post.return_value, []
        def fail_second_repeat(*args, **kwargs):
            if 'repeat_edit' in kwargs['json']['state']:
                repeat_calls.append(1)
                if len(repeat_calls) == 2:
                    raise requests.Timeout()
            return response
        self.post.side_effect = fail_second_repeat
        self.assertEqual(filter_ocr_texts([text]), [text])
        self.assertEqual(len(repeat_calls), 2)

    def test_final_boundary_veto_does_not_fall_back_to_a_worse_edit(self) -> None:
        noise = 'LJQZXV' * 12
        text = 'HP: 125/200; ' + noise + ' MP: 50/80.'
        start, end = text.index(noise), text.index(noise) + len(noise)
        self.scores = {text: MIXED}
        self.edge_notation = .9
        with patch.object(jev_filter._JevFilter, 'proposals', return_value=[(start - 2, end), (start + 1, end)]):
            self.assertEqual(filter_ocr_texts([text]), [text])

    def test_notation_expression_and_word_boundaries_veto_deletion(self) -> None:
        text = 'Name XQXQ tail'
        a, b = 5, 9
        self.scores = {text: MIXED}
        for protection in ('notation', 'expression', 'boundary'):
            with self.subTest(protection=protection):
                self.notation = .9 if protection == 'notation' else .01
                self.scores['XQXQ'] = dict(WHOLE, expression=.9) if protection == 'expression' else WHOLE
                self.edits['XQXQ'] = (.99, .01, .9 if protection == 'boundary' else .01)
                with patch.object(jev_filter._JevFilter, 'proposals', return_value=[(a, b)]):
                    self.assertEqual(filter_ocr_texts([text]), [text])

    def test_stop_before_and_after_response_is_propagated(self) -> None:
        event = threading.Event()
        event.set()
        with self.assertRaises(LLMRequestStopped):
            filter_ocr_texts(['ICIC'], event)
        self.post.assert_not_called()
        event.clear()
        def cancel() -> dict:
            event.set()
            return dict(self.answers(), usage={'input_tokens': 10, 'output_tokens': 2})
        self.response.json.side_effect = cancel
        totals = {}
        with self.assertRaises(LLMRequestStopped):
            filter_ocr_texts(['ICIC'], event, cleanup_totals=totals)
        self.assertEqual(totals['requests'], 1)
        self.assertEqual(totals['input_tokens'], 10)
        self.assertEqual(totals['reported_input_requests'], 1)
        self.assertEqual(totals['removed_characters'], 0)
    def test_provider_selection_keeps_keys_and_endpoints_together(self) -> None:
        for provider in ('typesafe', 'openrouter'):
            with self.subTest(provider=provider), \
                    patch.object(pcfg.module, 'ocr_jev_provider', provider), \
                    patch.dict('os.environ', {
                        'JEV_API_PROVIDER': 'openrouter' if provider == 'typesafe' else 'typesafe',
                        'OPENROUTER_API_KEY': 'router-key',
                    }):
                self.assertEqual(filter_ocr_texts(['IC' * 100]), [''])
                args, kwargs = self.post.call_args
                router = provider == 'openrouter'
                self.assertEqual(args[0], 'https://openrouter.ai/api/v1/systemone' if router
                                 else 'https://api.typesafe.ai/v1/systemone')
                self.assertEqual(kwargs['headers']['Authorization'],
                                 'Bearer router-key' if router else 'Bearer test-key')
                self.assertEqual(kwargs['json']['model'], '~typesafe/jev-latest' if router else 'jev-latest')

    def test_invalid_provider_or_missing_selected_key_does_not_fallback(self) -> None:
        for provider in ('invalid', 'openrouter'):
            with patch.object(pcfg.module, 'ocr_jev_provider', provider):
                self.assertEqual(filter_ocr_texts(['IC' * 100]), ['IC' * 100])
        # The official default must not silently route text to OpenRouter.
        with patch.dict('os.environ', {'TYPESAFE_API_KEY': '', 'OPENROUTER_API_KEY': 'router-key'}):
            self.assertEqual(filter_ocr_texts(['IC' * 100]), ['IC' * 100])
        self.post.assert_not_called()

    def test_provider_config_defaults_sanitizes_and_round_trips(self) -> None:
        self.assertEqual(ModuleConfig().ocr_jev_provider, 'typesafe')
        for invalid in ('', 'unknown', None, [], {}, True):
            with self.subTest(invalid=invalid):
                config = ModuleConfig(enable_detect=False, ocr_jev_provider=invalid)
                self.assertEqual(config.ocr_jev_provider, 'typesafe')
                self.assertFalse(config.enable_detect)
        for provider in ('typesafe', 'openrouter'):
            config = ModuleConfig(ocr_jev_provider=provider)
            self.assertEqual(ModuleConfig(**config.get_saving_params()).ocr_jev_provider, provider)

    def test_saved_key_precedence_and_provider_isolation(self) -> None:
        router = profile_by_id(pcfg.module.llm_profiles, 'openrouter')
        router.name = 'My renamed router'
        unrelated = default_profile('OpenRouter')
        unrelated.id, unrelated.api_key = 'openrouter-copy', 'wrong-key'
        pcfg.module.llm_profiles.insert(0, unrelated)
        for provider in ('typesafe', 'openrouter'):
            with self.subTest(provider=provider), patch.object(router, 'api_key', 'other-key'), \
                    patch.object(pcfg.module, 'ocr_jev_provider', provider), \
                    patch.object(pcfg.module, 'ocr_jev_typesafe_api_key', 'other-key'), \
                    patch.dict('os.environ', {f'{provider.upper()}_API_KEY': 'env-key'}):
                if provider == 'typesafe':
                    pcfg.module.ocr_jev_typesafe_api_key = SecretStore().store('jev-typesafe', 'saved-key')
                else:
                    store_api_key(router, 'saved-key')
                self.assertEqual(filter_ocr_texts(['IC' * 100]), [''])
                self.assertEqual(self.post.call_args.kwargs['headers']['Authorization'], 'Bearer saved-key')
                if provider == 'typesafe':
                    pcfg.module.ocr_jev_typesafe_api_key = ''
                else:
                    store_api_key(router, '')
                self.assertEqual(filter_ocr_texts(['IC' * 100]), [''])
                self.assertEqual(self.post.call_args.kwargs['headers']['Authorization'], 'Bearer env-key')
                self.post.reset_mock()
                with patch.dict('os.environ', {f'{provider.upper()}_API_KEY': ''}):
                    self.assertEqual(filter_ocr_texts(['IC' * 100]), ['IC' * 100])
                    self.post.assert_not_called()

    def test_typesafe_key_config_round_trip_and_invalid_data(self) -> None:
        self.assertEqual(ModuleConfig().ocr_jev_typesafe_api_key, '')
        config = ModuleConfig(ocr_jev_typesafe_api_key='saved-test-secret')
        serialized = json_dump_program_config(config)
        self.assertNotIn('saved-test-secret', serialized)
        reloaded = ModuleConfig(**json.loads(serialized))
        self.assertTrue(is_portable_secret(reloaded.ocr_jev_typesafe_api_key))
        self.assertEqual(SecretStore().resolve(reloaded.ocr_jev_typesafe_api_key).value, 'saved-test-secret')
        for invalid in (None, [], {}, True, 123, {'storage': 'unknown'},
                        {'storage': 'portable_obfuscated', 'version': 1, 'value': 'abc'}):
            with self.subTest(invalid=invalid):
                config = ModuleConfig(enable_detect=False, ocr_jev_typesafe_api_key=invalid)
                self.assertEqual(config.ocr_jev_typesafe_api_key, '')
                self.assertFalse(config.enable_detect)

    def test_openrouter_reuses_reloaded_profile_and_handles_missing_profile(self) -> None:
        router = default_profile('OpenRouter')
        store_api_key(router, 'shared-router-key')
        config = ModuleConfig(ocr_jev_provider='openrouter', llm_profiles=[router],
                              ocr_jev_openrouter_api_key='obsolete-key')
        serialized = json_dump_program_config(config)
        self.assertNotIn('ocr_jev_openrouter_api_key', serialized)
        self.assertNotIn('shared-router-key', serialized)
        with patch.object(pcfg, 'module', ModuleConfig(**json.loads(serialized))):
            self.assertEqual(filter_ocr_texts(['IC' * 100]), [''])
            self.assertEqual(self.post.call_args.kwargs['headers']['Authorization'], 'Bearer shared-router-key')
            pcfg.module.llm_profiles.clear()
            self.post.reset_mock()
            self.assertEqual(filter_ocr_texts(['ICIC']), ['ICIC'])
            self.post.assert_not_called()
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'env-router-key'}):
                self.assertEqual(filter_ocr_texts(['IC' * 100]), [''])
                self.assertEqual(self.post.call_args.kwargs['headers']['Authorization'], 'Bearer env-router-key')

    def test_ocr_never_calls_jev_even_with_legacy_enabled_config(self) -> None:
        legacy = ModuleConfig(ocr_jev_filter=True, ocr_jev_provider='openrouter')
        self.assertNotIn('ocr_jev_filter', legacy.get_saving_params())
        self.assertEqual(legacy.ocr_jev_provider, 'openrouter')
        block = TextBlock()
        model = OCRBase()

        def recognize(*args, **kwargs):
            block.text = 'IC' * 100

        with patch.object(model, 'all_model_loaded', return_value=True), \
                patch.object(model, '_ocr_blk_list', side_effect=recognize), \
                patch.object(model, 'ocr_img', return_value='IC' * 100), \
                patch.object(pcfg.module, 'ocr_font_detect', False):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            self.assertEqual(model.run_ocr(image, [block])[0].get_text(), 'IC' * 100)
            self.assertEqual(model.run_ocr(image), 'IC' * 100)
        self.post.assert_not_called()


if __name__ == '__main__':
    unittest.main()

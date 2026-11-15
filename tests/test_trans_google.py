import unittest
from unittest.mock import Mock, patch

import requests

from ballontranslator.modules.translators import trans_google


class GoogleTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.translator = trans_google.TransGoogle('English', '简体中文')
        self.provider = self.translator.internal_google_translator
        self.request_patch = patch.object(self.provider.requests_session, 'request')
        self.request = self.request_patch.start()
        self.addCleanup(self.request_patch.stop)
        self.addCleanup(self.provider.requests_session.close)

    def test_all_network_failures_raise_with_actionable_message_and_cause(self) -> None:
        self.request.side_effect = requests.ConnectionError('network unavailable')
        with self.assertRaisesRegex(trans_google.TranslateError, 'Check your network/proxy') as caught:
            self.translator.translate(['first', 'second'])
        self.assertIsInstance(caught.exception.__cause__, trans_google.ProviderError)
        self.assertEqual(self.request.call_count, 2)

    def test_http_invalid_json_and_unparsable_responses_raise(self) -> None:
        responses = [
            Mock(status_code=503, reason='Unavailable', json=Mock(return_value={})),
            Mock(status_code=200, text='invalid',
                 json=Mock(side_effect=trans_google.json.JSONDecodeError('invalid', 'x', 0))),
            Mock(status_code=200, json=Mock(return_value={'unexpected': []})),
            Mock(status_code=200, json=Mock(return_value=[['   ']])),
        ]
        for response in responses:
            with self.subTest(response_index=responses.index(response)):
                self.request.return_value = response
                with self.assertRaises(trans_google.TranslateError):
                    self.translator.translate(['source'])

    def test_partial_failure_preserves_empty_item_and_warns_with_count(self) -> None:
        self.request.side_effect = [
            Mock(status_code=200, json=Mock(return_value=[['translated &amp; valid']])),
            requests.ConnectionError('network unavailable'),
        ]
        with self.assertLogs(trans_google.LOGGER, level='WARNING') as logs:
            result = self.translator.translate(['first', '', 'second', '  '])
        self.assertEqual(result, ['translated & valid', '', '', ''])
        self.assertEqual(len(logs.output), 1)
        self.assertIn('1 of 2 non-empty source items', logs.output[0])

    def test_empty_sources_do_not_count_as_failures_or_request_network(self) -> None:
        with patch.object(trans_google.LOGGER, 'warning') as warning:
            self.assertEqual(self.provider.translate(['', ' \t'], 'en')['translations'], ['', ''])
            self.assertEqual(self.translator._translate(['', ' \t']), ['', ''])
            self.assertEqual(self.translator._translate([]), [])
        warning.assert_not_called()
        self.request.assert_not_called()

    def test_empty_sources_cannot_hide_total_failure(self) -> None:
        self.request.side_effect = requests.ConnectionError('network unavailable')
        with self.assertRaises(trans_google.TranslateError):
            self.translator._translate(['', 'source', '  '])
        self.request.assert_called_once()

    def test_success_with_empty_sources_does_not_warn(self) -> None:
        self.request.return_value = Mock(status_code=200, json=Mock(return_value=[['translated']]))
        with patch.object(trans_google.LOGGER, 'warning') as warning:
            self.assertEqual(self.translator._translate(['', 'source', ' ']), ['', 'translated', ''])
        warning.assert_not_called()

    def test_length_mismatch_and_unexpected_error_raise_with_cause(self) -> None:
        for response in ({'translations': []}, {'translations': ['', '']}, None):
            with self.subTest(response=response), patch.object(
                self.provider, 'translate', return_value=response,
            ):
                with self.assertRaises(trans_google.TranslateError) as caught:
                    self.translator._translate(['first', 'second'])
                self.assertIsInstance(caught.exception.__cause__, trans_google.ProviderError)
        failure = ValueError('unexpected provider failure')
        with patch.object(self.provider, 'translate', side_effect=failure):
            with self.assertRaises(trans_google.TranslateError) as caught:
                self.translator._translate(['source'])
        self.assertIs(caught.exception.__cause__, failure)


if __name__ == '__main__':
    unittest.main()

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from _llm_translation_test_support import (
    LLMTranslationTestMixin,
    _block,
)
from ballontranslator.modules.llm_vision import EncodedChatImage
from ballontranslator.modules.translators.base import BaseTranslator
from ballontranslator.utils.config import (
    LLMGlossaryMode,
    LLMTranslateContext,
    pcfg,
)


class LLMTranslationIntegrationTest(
    LLMTranslationTestMixin,
    unittest.TestCase,
):
    def test_disabled_features_skip_request_context(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_glossary_path = ''

        self.assertIsNone(
            self._snapshot_request_context(None, None, self.profile)
        )

    def test_vision_is_a_suffix_and_keeps_selected_translation_model(self):
        self.profile.model = 'selected-translation-model'
        self.profile.vision_model = 'ignored-ocr-model'
        project = self._project(1)
        project.read_img = mock.Mock(
            return_value=np.zeros((32, 24, 3), dtype=np.uint8)
        )
        vision = self.translator._vision_request_context(
            project,
            '001.png',
            self.profile,
        )

        text_messages, _ = self._assemble_request(
            ['source-1'],
            self.profile,
        )
        vision_messages, _ = self._assemble_request(
            ['source-1'],
            self.profile,
            vision_request=vision,
        )

        self.assertEqual(vision_messages[:-1], text_messages[:-1])
        vision_prompt = vision_messages[-1]['content'][0]['text']
        self.assertTrue(
            vision_prompt.startswith(text_messages[-1]['content']),
        )
        self.assertIn('infer the natural comic reading order', vision_prompt)
        self.assertIn('mapped to its original input ID', vision_prompt)
        self.assertNotIn(
            'infer the natural comic reading order',
            text_messages[-1]['content'],
        )
        self.assertEqual(
            vision_messages[-1]['content'][1]['type'],
            'image_url',
        )
        self.assertTrue(
            vision_messages[-1]['content'][1]['image_url']['url'].startswith(
                'data:image/jpeg;base64,'
            )
        )
        self.assertEqual(
            self.translator._api_args(
                self.profile,
                vision_messages,
            )['model'],
            self.profile.model,
        )

    def test_translate_freezes_all_context_settings_before_image_work(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.HISTORY
        pcfg.module.llm_prior_context_token_budget = 321
        pcfg.module.llm_glossary_path = 'before.tsv'
        pcfg.module.llm_glossary_mode = LLMGlossaryMode.Matching
        pcfg.module.llm_translate_vision = True
        pcfg.module.llm_translate_summary_memory = True
        vision = EncodedChatImage(
            'data:image/jpeg;base64,AA==',
            'auto',
        )

        def prepare_vision(*_args, **_kwargs):
            pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
            pcfg.module.llm_prior_context_token_budget = 1
            pcfg.module.llm_glossary_path = 'after.tsv'
            pcfg.module.llm_glossary_mode = LLMGlossaryMode.All
            pcfg.module.llm_translate_vision = False
            pcfg.module.llm_translate_summary_memory = False
            return vision

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
            '_vision_request_context',
            side_effect=prepare_vision,
        ), mock.patch.object(
            self.translator,
            '_snapshot_request_context',
            return_value=None,
        ) as snapshot, mock.patch.object(
            self.translator,
            '_request_translation',
            return_value='{"1":"translated"}',
        ) as request:
            result = self.translator.translate(
                ['source'],
                project=SimpleNamespace(
                    get_llm_visual_summary=lambda _page_key: None,
                ),
                page_key='001.png',
            )

        self.assertEqual(result, ['translated'])
        kwargs = snapshot.call_args.kwargs
        self.assertTrue(kwargs['prompt_spec'].history_enabled)
        self.assertTrue(kwargs['prompt_spec'].summary_enabled)
        self.assertEqual(kwargs['history_budget'], 321)
        self.assertEqual(kwargs['glossary_path'], 'before.tsv')
        self.assertEqual(kwargs['glossary_mode'], LLMGlossaryMode.Matching)
        self.assertTrue(kwargs['memory_enabled'])
        self.assertIn(
            'Treat prior user/assistant pairs as read-only',
            request.call_args.args[1][0]['content'],
        )

    def test_vision_jpeg_preserves_project_rgb_channel_order(self):
        project = self._project(1)
        project.read_img = mock.Mock(
            return_value=np.array([[[10, 20, 30]]], dtype=np.uint8)
        )
        encoded = np.frombuffer(b'jpeg', dtype=np.uint8)

        with mock.patch(
            'ballontranslator.modules.llm_vision.cv2.imencode',
            return_value=(True, encoded),
        ) as imencode:
            self.translator._vision_request_context(
                project,
                '001.png',
                self.profile,
            )

        encoded_image = imencode.call_args.args[1]
        self.assertEqual(encoded_image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(
            imencode.call_args.args[2],
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )

    def test_vision_scales_long_side_before_explicit_quality_encoding(self):
        project = self._project(1)
        source = np.zeros((2000, 1000, 3), dtype=np.uint8)
        resized = np.zeros((1536, 768, 3), dtype=np.uint8)
        project.read_img = mock.Mock(return_value=source)
        encoded = np.frombuffer(b'jpeg', dtype=np.uint8)

        with mock.patch(
            'ballontranslator.modules.translators.trans_llm.cv2.resize',
            return_value=resized,
        ) as resize, mock.patch(
            'ballontranslator.modules.llm_vision.cv2.imencode',
            return_value=(True, encoded),
        ) as imencode:
            self.translator._vision_request_context(
                project,
                '001.png',
                self.profile,
            )

        resize.assert_called_once_with(
            source,
            (768, 1536),
            interpolation=cv2.INTER_AREA,
        )
        self.assertEqual(imencode.call_args.args[1].shape, (1536, 768, 3))
        self.assertEqual(
            imencode.call_args.args[2],
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )

    def test_vision_image_is_encoded_once_across_translation_retries(self):
        project = self._project(1)
        project.pages['001.png'][0].translation = ''
        project.read_img = mock.Mock(
            return_value=np.zeros((32, 24, 3), dtype=np.uint8)
        )
        pcfg.module.llm_translate_vision = True
        self.translator.set_param_value('retry timeout', 0)
        debug_messages = []
        self.translator.logger = SimpleNamespace(
            debug=debug_messages.append,
            warning=lambda *_args: None,
            error=lambda *_args: None,
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
            side_effect=('not json', '{"1":"translated"}'),
        ):
            self.translator.translate_textblk_lst(
                project.pages['001.png'],
                project=project,
                page_key='001.png',
                full_page=True,
            )

        project.read_img.assert_called_once_with('001.png')
        self.assertIn(
            "LLM invalid translation response: page=001.png, attempt=1, "
            "chars=8, content='not json'",
            debug_messages,
        )

    def test_retry_reuses_same_messages_and_raises_final_failure(self):
        self.translator.set_param_value('retry attempts', 2)
        self.translator.set_param_value('retry timeout', 0)
        with mock.patch.object(
            self.translator,
            '_request_translation',
            side_effect=[RuntimeError('first'), RuntimeError('final')],
        ) as request:
            with self.assertRaisesRegex(RuntimeError, 'final'):
                self._translate(
                    ['source'],
                    profile=self.profile,
                    page_key='001.png',
                )

        first_messages = request.call_args_list[0].args[1]
        second_messages = request.call_args_list[1].args[1]
        self.assertIs(first_messages, second_messages)
        self.assertEqual(first_messages, second_messages)
        self.assertEqual(
            [call.kwargs for call in request.call_args_list],
            [
                {
                    'expected_translations': 1,
                    'usage_page_key': '001.png',
                    'usage_attempt': 1,
                },
                {
                    'expected_translations': 1,
                    'usage_page_key': '001.png',
                    'usage_attempt': 2,
                },
            ],
        )

    def test_build_copy_prompt_includes_glossary_but_not_project_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'terms.txt')
            with open(path, 'w', encoding='utf-8') as glossary_file:
                glossary_file.write('Hero->勇者\nMage->法师\n')
            pcfg.module.llm_glossary_path = path
            pcfg.module.llm_glossary_mode = LLMGlossaryMode.Matching

            prompt = self.translator.build_copy_prompt(['The Hero returns'])

        self.assertIn('"source":"Hero"', prompt)
        self.assertNotIn('"source":"Mage"', prompt)
        self.assertNotIn('"translations"', prompt)

    def test_textblock_boundary_forwards_project_and_page_unchanged(self):
        project = object()
        translator = SimpleNamespace(
            lang_source='日本語',
            lang_target='简体中文',
            cht_require_convert=False,
            translate=mock.Mock(return_value=['translated']),
        )
        block = _block('source')

        with mock.patch.object(
            pcfg,
            'pre_mt_sublist',
            [],
        ), mock.patch.object(
            pcfg,
            'mt_sublist',
            [],
        ):
            BaseTranslator.translate_textblk_lst(
                translator,
                [block],
                project=project,
                page_key='001.png',
            )

        translator.translate.assert_called_once_with(
            ['source'],
            project=project,
            page_key='001.png',
            commit_history_window=False,
        )
        self.assertEqual(block.translation, 'translated')


if __name__ == '__main__':
    unittest.main()

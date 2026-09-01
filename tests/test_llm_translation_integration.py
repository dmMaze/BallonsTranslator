import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from _llm_translation_test_support import (
    LLMTranslationTestMixin,
)
from ballontranslator.utils.config import (
    LLMGlossaryMode,
    pcfg,
)


class LLMTranslationIntegrationTest(
    LLMTranslationTestMixin,
    unittest.TestCase,
):
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
        ) as request:
            self.translator.translate_textblk_lst(
                project.pages['001.png'],
                project=project,
                page_key='001.png',
                full_page=True,
            )

        project.read_img.assert_called_once_with('001.png')
        self.assertIs(
            request.call_args_list[0].args[1],
            request.call_args_list[1].args[1],
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

if __name__ == '__main__':
    unittest.main()

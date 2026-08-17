import json
import tempfile
import unittest
from unittest.mock import patch

from ballontranslator.utils.config import (
    LLMGlossaryMode,
    LLMTranslateContext,
    ModuleConfig,
    ProgramConfig,
    RunStatus,
    json_dump_program_config,
)
from ballontranslator.utils.llm_profiles import LLMProfile
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


def _module_config(**kwargs):
    kwargs.setdefault('llm_profiles', [LLMProfile(id='test-profile')])
    return ModuleConfig(**kwargs)


class LLMContextConfigTest(unittest.TestCase):

    def test_legacy_translate_context_is_migrated_without_manual_cleanup(self):
        with tempfile.NamedTemporaryFile('w+', encoding='utf8') as temp:
            json.dump({'module': {'translate_by_textblock': True}}, temp)
            temp.flush()
            loaded = ProgramConfig.load(temp.name)

        self.assertEqual(loaded.module.translate_context, 'textblock')
        self.assertFalse(hasattr(loaded.module, 'translate_by_textblock'))

    def test_llm_context_defaults_and_invalid_values_are_safe(self):
        with tempfile.NamedTemporaryFile('w+', encoding='utf8') as temp:
            json.dump({'module': {}}, temp)
            temp.flush()
            loaded = ProgramConfig.load(temp.name)

        self.assertEqual(
            (
                loaded.module.llm_translate_context,
                loaded.module.llm_prior_context_token_budget,
                loaded.module.llm_glossary_path,
                loaded.module.llm_glossary_mode,
            ),
            (LLMTranslateContext.PAGE, 4096, '', LLMGlossaryMode.Matching),
        )

        invalid_cases = (
            ('llm_glossary_mode', (None, '', 'everything'), LLMGlossaryMode.Matching),
            ('llm_prior_context_token_budget', (0, -1, False, '4096'), 4096),
            (
                'llm_translate_context',
                (None, '', True, 'textblock', [], {}),
                LLMTranslateContext.PAGE,
            ),
            ('llm_glossary_path', (None, False, 1, [], {}), ''),
        )
        for field, values, expected in invalid_cases:
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assertEqual(
                        getattr(_module_config(**{field: value}), field),
                        expected,
                    )

    def test_llm_context_settings_roundtrip_directly_under_module(self):
        cfg = ProgramConfig(module=_module_config(
            llm_translate_context=LLMTranslateContext.HISTORY,
            llm_prior_context_token_budget=2048,
            llm_glossary_path='glossaries/terms.tsv',
            llm_glossary_mode=LLMGlossaryMode.All,
        ))
        raw = json.loads(json_dump_program_config(cfg))

        self.assertEqual(
            {
                key: raw['module'][key]
                for key in (
                    'llm_translate_context',
                    'llm_prior_context_token_budget',
                    'llm_glossary_path',
                    'llm_glossary_mode',
                )
            },
            {
                'llm_translate_context': LLMTranslateContext.HISTORY,
                'llm_prior_context_token_budget': 2048,
                'llm_glossary_path': 'glossaries/terms.tsv',
                'llm_glossary_mode': LLMGlossaryMode.All,
            },
        )

        with tempfile.NamedTemporaryFile('w+', encoding='utf8') as temp:
            json.dump(raw, temp)
            temp.flush()
            loaded = ProgramConfig.load(temp.name)

        self.assertEqual(
            loaded.module.llm_translate_context,
            LLMTranslateContext.HISTORY,
        )
        self.assertEqual(loaded.module.llm_prior_context_token_budget, 2048)
        self.assertEqual(
            loaded.module.llm_glossary_path,
            'glossaries/terms.tsv',
        )
        self.assertEqual(loaded.module.llm_glossary_mode, LLMGlossaryMode.All)


class ProjectLoadIdentityTest(unittest.TestCase):

    def test_identity_is_stable_until_project_contents_are_replaced(self):
        project = ProjImgTrans()
        initial_identity = project.load_identity

        self.assertIs(initial_identity, project.load_identity)
        project.directory = '/unused'
        with patch(
            'ballontranslator.utils.proj_imgtrans.find_all_imgs',
            return_value=[],
        ):
            project.load_from_dict({'pages': {}, 'image_info': {}})
            first_load_identity = project.load_identity
            project.load_from_dict({'pages': {}, 'image_info': {}})

        self.assertIsNot(first_load_identity, initial_identity)
        self.assertIsNot(project.load_identity, first_load_identity)
        self.assertNotIn('load_identity', project.to_dict())
        self.assertNotIn('_load_identity', project.to_dict())

    def test_reloading_same_path_and_new_project_replace_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans(directory)
            new_project_identity = project.load_identity
            project.load(directory)
            first_reload_identity = project.load_identity
            project.load(directory)

            self.assertIsNot(first_reload_identity, new_project_identity)
            self.assertIsNot(project.load_identity, first_reload_identity)

            before_new_project = project.load_identity
            project.new_project()
            self.assertIsNot(project.load_identity, before_new_project)


class ProjectTranslationTargetTest(unittest.TestCase):

    @staticmethod
    def _project(*page_names):
        project = ProjImgTrans()
        project.pages = {
            page_name: [TextBlock()]
            for page_name in page_names
        }
        project._image_info = {
            page_name: {'finish_code': 0}
            for page_name in page_names
        }
        return project

    def test_mark_and_progress_updates_manage_translation_target(self):
        project = self._project('001.png')
        info = project._image_info['001.png']
        info['finish_code'] = RunStatus.FIN_DET

        project.mark_translation_finished('001.png', '简体中文')

        self.assertEqual(
            info['finish_code'],
            RunStatus.FIN_DET | RunStatus.FIN_TRANSLATE,
        )
        self.assertEqual(info['translation_target'], '简体中文')
        self.assertEqual(
            project.to_dict()['image_info']['001.png']['translation_target'],
            '简体中文',
        )

        project.clear_page_progress('001.png', RunStatus.FIN_DET)
        self.assertEqual(info['translation_target'], '简体中文')

        project.set_page_progress(
            '001.png',
            RunStatus.FIN_OCR | RunStatus.FIN_TRANSLATE,
        )
        self.assertEqual(info['translation_target'], '简体中文')

        project.set_page_progress('001.png', RunStatus.FIN_OCR)
        self.assertNotIn('translation_target', info)

        project.mark_translation_finished('001.png', 'English')
        project.clear_page_progress('001.png', RunStatus.FIN_TRANSLATE)
        self.assertNotIn('translation_target', info)

    def test_load_from_dict_preserves_target_and_legacy_absence(self):
        for expected_target, image_info in (
            ('English', {
                'finish_code': RunStatus.FIN_TRANSLATE,
                'translation_target': 'English',
            }),
            (None, {'finish_code': RunStatus.FIN_TRANSLATE}),
        ):
            with self.subTest(target=expected_target):
                project = ProjImgTrans()
                project.directory = '/unused'
                raw = {
                    'pages': {'001.png': []},
                    'image_info': {'001.png': image_info},
                }
                with patch(
                    'ballontranslator.utils.proj_imgtrans.find_all_imgs',
                    return_value=['001.png'],
                ), patch.object(project, 'set_current_img'):
                    project.load_from_dict(raw)

                loaded_info = project._image_info['001.png']
                if expected_target is None:
                    self.assertNotIn('translation_target', loaded_info)
                else:
                    self.assertEqual(
                        loaded_info['translation_target'],
                        expected_target,
                    )

    def test_import_updates_completion_per_page(self):
        imported_pages = [
            {'page_name': '001.png', 'blk_list': ['one']},
            {'page_name': '002.png', 'blk_list': ['two']},
        ]
        project = self._project('001.png', '002.png')

        with patch(
            'ballontranslator.utils.proj_imgtrans.parse_txt_translation',
            return_value=imported_pages,
        ):
            all_matched, _ = project.load_translation_from_txt(
                'translation.md',
                target_language='English',
            )

        self.assertTrue(all_matched)
        for info in project._image_info.values():
            self.assertTrue(info['finish_code'] & RunStatus.FIN_TRANSLATE)
            self.assertEqual(info['translation_target'], 'English')

        legacy_project = self._project('001.png', '002.png')
        for info in legacy_project._image_info.values():
            info['translation_target'] = 'stale-target'
        with patch(
            'ballontranslator.utils.proj_imgtrans.parse_txt_translation',
            return_value=imported_pages,
        ):
            all_matched, _ = legacy_project.load_translation_from_txt(
                'translation.md',
            )

        self.assertTrue(all_matched)
        for info in legacy_project._image_info.values():
            self.assertTrue(info['finish_code'] & RunStatus.FIN_TRANSLATE)
            self.assertNotIn('translation_target', info)

        partial_project = self._project('001.png', '002.png', '003.png')
        for page_name in partial_project.pages:
            partial_project.mark_translation_finished(page_name, 'stale-target')
        partial_pages = [
            {'page_name': '001.png', 'blk_list': ['one']},
            {'page_name': '002.png', 'blk_list': ['two', 'extra']},
        ]
        with patch(
            'ballontranslator.utils.proj_imgtrans.parse_txt_translation',
            return_value=partial_pages,
        ):
            all_matched, _ = partial_project.load_translation_from_txt(
                'partial.md',
                target_language='English',
            )

        self.assertFalse(all_matched)
        imported_info = partial_project._image_info['001.png']
        self.assertTrue(imported_info['finish_code'] & RunStatus.FIN_TRANSLATE)
        self.assertEqual(imported_info['translation_target'], 'English')

        malformed_info = partial_project._image_info['002.png']
        self.assertFalse(malformed_info['finish_code'] & RunStatus.FIN_TRANSLATE)
        self.assertNotIn('translation_target', malformed_info)

        missing_info = partial_project._image_info['003.png']
        self.assertTrue(missing_info['finish_code'] & RunStatus.FIN_TRANSLATE)
        self.assertEqual(missing_info['translation_target'], 'stale-target')

if __name__ == '__main__':
    unittest.main()

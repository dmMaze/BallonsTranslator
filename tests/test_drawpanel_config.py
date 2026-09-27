import json
import tempfile
import unittest
from pathlib import Path

from ballontranslator.utils.config import DrawPanelConfig, ProgramConfig, json_dump_program_config


class DrawPanelConfigTest(unittest.TestCase):
    def load_config(self, payload: dict) -> ProgramConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            return ProgramConfig.load(str(path))

    def test_old_drawing_settings_use_defaults_independently_of_run(self) -> None:
        config = self.load_config({
            'module': {'inpainter': 'LLMInpaint', 'inpaint_llm_id': 'codex'},
            'drawpanel': {'pentool_width': 12.5, 'magicwand_tolerance': 77},
        })
        drawing = config.drawpanel
        self.assertEqual(drawing.inpainter, 'lama_large_512px')
        self.assertEqual(drawing.inpaint_llm_id, '')
        self.assertEqual(drawing.inpaint_llm_model, '')
        self.assertEqual(drawing.inpaint_prompt_override, '')
        self.assertTrue(drawing.rectool_use_mask)
        self.assertEqual(drawing.pentool_width, 12.5)
        self.assertEqual(drawing.magicwand_tolerance, 77)
        self.assertEqual(config.module.inpainter, 'LLMInpaint')
        self.assertEqual(config.module.inpaint_llm_id, 'codex')

    def test_draw_inpainter_settings_and_raw_prompt_survive_restart(self) -> None:
        settings = {
            'inpainter': 'LLMInpaint',
            'inpaint_llm_id': 'codex',
            'inpaint_llm_model': 'gpt-image-2',
            'inpaint_prompt_override': '\n  Remove only the marked lettering.\n\n ',
            'rectool_use_mask': False,
        }
        config = self.load_config({
            'module': {'inpainter': 'lama_mpe', 'inpaint_llm_id': 'openai'},
            'drawpanel': settings,
        })
        saved = json.loads(json_dump_program_config(config))
        self.assertEqual({key: saved['drawpanel'][key] for key in settings}, settings)
        restarted = self.load_config(saved)
        self.assertEqual({key: getattr(restarted.drawpanel, key) for key in settings}, settings)
        self.assertEqual(restarted.module.inpainter, 'lama_mpe')
        self.assertEqual(restarted.module.inpaint_llm_id, 'openai')

    def test_malformed_fields_default_independently_and_preserve_other_settings(self) -> None:
        settings = {
            'inpainter': 'LLMInpaint', 'inpaint_llm_id': 'codex',
            'inpaint_llm_model': 'chosen-model', 'inpaint_prompt_override': ' Keep this prompt. ',
            'pentool_width': 12.5,
        }
        for field in ('inpainter', 'inpaint_llm_id', 'inpaint_llm_model', 'inpaint_prompt_override'):
            for invalid in (None, True, 42, {}, []):
                with self.subTest(field=field, invalid=invalid), self.assertLogs('BallonTranslator', level='WARNING'):
                    loaded = self.load_config({'drawpanel': {**settings, field: invalid}}).drawpanel
                expected = {**settings, field: 'lama_large_512px' if field == 'inpainter' else ''}
                self.assertEqual({key: getattr(loaded, key) for key in settings}, expected)

    def test_blank_inpainter_defaults_and_whitespace_prompt_is_preserved(self) -> None:
        for name in ('', ' \t\n'):
            with self.subTest(name=name), self.assertLogs('BallonTranslator', level='WARNING'):
                drawing = DrawPanelConfig(inpainter=name, inpaint_prompt_override=' \n\t ')
            self.assertEqual(drawing.inpainter, 'lama_large_512px')
            self.assertEqual(drawing.inpaint_prompt_override, ' \n\t ')
        # Module membership belongs to lazy registry validation, not config loading.
        self.assertEqual(DrawPanelConfig(inpainter='custom-inpainter').inpainter, 'custom-inpainter')

    def test_invalid_use_mask_defaults_without_losing_other_settings(self) -> None:
        for value in (None, 0, 'false', [], {}):
            with self.subTest(value=value), self.assertLogs('BallonTranslator', level='WARNING'):
                drawing = self.load_config({'drawpanel': {
                    'rectool_use_mask': value, 'pentool_width': 12.5,
                }}).drawpanel
            self.assertTrue(drawing.rectool_use_mask)
            self.assertEqual(drawing.pentool_width, 12.5)


if __name__ == '__main__':
    unittest.main()

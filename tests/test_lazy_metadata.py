import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ballontranslator.modules.lazy_registry import _scan_file, validate_lazy_module_specs
from ballontranslator.utils.registry import ModuleSpec


class LazyMetadataTests(unittest.TestCase):

    def scan_source(self, source: str, module_type: str):
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf8') as f:
            f.write(source)
            path = f.name
        try:
            return _scan_file(path, module_type)
        finally:
            os.unlink(path)

    def test_translator_lang_map_update_is_lazy_metadata(self):
        specs = self.scan_source(
            '''
@register_translator("demo")
class DemoTranslator:
    def _setup_translator(self):
        self.lang_map.update({"日本語": "ja", "English": "en"})
''',
            'translator',
        )

        self.assertEqual(specs[0].supported_src_list, ['日本語', 'English'])
        self.assertEqual(specs[0].supported_tgt_list, ['日本語', 'English'])
        self.assertEqual(validate_lazy_module_specs(specs), [])

    def test_dynamic_translator_lang_map_is_reported(self):
        specs = self.scan_source(
            '''
@register_translator("demo")
class DemoTranslator:
    def _setup_translator(self):
        self.lang_map.update(load_remote_languages())
''',
            'translator',
        )

        warnings = validate_lazy_module_specs(specs)
        self.assertTrue(any('unsupported lazy lang_map.update call' in warning for warning in warnings))
        self.assertTrue(any('translator has no lazy supported language metadata' in warning for warning in warnings))

    def test_dynamic_selector_options_are_reported(self):
        specs = self.scan_source(
            '''
@register_OCR("demo")
class DemoOCR:
    params = {
        "model": {"type": "selector", "options": load_models(), "value": "default"}
    }
''',
            'ocr',
        )

        warnings = validate_lazy_module_specs(specs)
        self.assertEqual(specs[0].params, None)
        self.assertTrue(any('params could not be evaluated lazily' in warning for warning in warnings))

    def test_safe_selector_helper_output_is_preserved(self):
        source = '''
MODEL_DIR = "data/models"
PREFIXES = ("demo",)

@register_textdetectors("demo")
class DemoDetector:
    params = {
        "model path": {
            "type": "selector",
            "options": find_model_paths(MODEL_DIR, PREFIXES),
            "value": "data/models/demo.pt",
            "editable": True,
        }
    }
'''
        with mock.patch('ballontranslator.modules.lazy_registry.os.listdir', return_value=['demo_a.pt', 'other.pt']):
            specs = self.scan_source(source, 'textdetector')

        self.assertEqual(specs[0].params['model path']['options'], ['data/models/demo_a.pt'])
        self.assertEqual(validate_lazy_module_specs(specs), [])

    def test_current_lazy_registry_metadata_is_complete(self):
        from ballontranslator.modules import GET_VALID_TEXTDETECTORS, INPAINTERS, OCR, TEXTDETECTORS, TRANSLATORS

        registries = [TRANSLATORS, TEXTDETECTORS, OCR, INPAINTERS]
        specs = [registry.get_spec(key) for registry in registries for key in registry.module_dict]

        self.assertEqual(validate_lazy_module_specs([spec for spec in specs if spec is not None]), [])
        for spec in specs:
            if spec is not None and spec.module_type == 'translator':
                self.assertTrue(spec.supported_src_list, spec.key)
                self.assertTrue(spec.supported_tgt_list, spec.key)

        self.assertIsInstance(TRANSLATORS.get('Papago'), ModuleSpec)
        self.assertIsNone(TRANSLATORS.get('Papago').resolved_class)
        self.assertIsInstance(TEXTDETECTORS.get('ysgyolo'), ModuleSpec)
        self.assertIsNone(TEXTDETECTORS.get('ysgyolo').resolved_class)

        ysgyolo_paths = sorted(Path('data/models').glob('ysgyolo*')) + sorted(Path('data/models').glob('ultralyticsyolo*'))
        if ysgyolo_paths:
            ysgyolo = TEXTDETECTORS.get_spec('ysgyolo')
            self.assertTrue(ysgyolo.params['model path']['options'])
        self.assertIn('ysgyolo', GET_VALID_TEXTDETECTORS())


if __name__ == '__main__':
    unittest.main()

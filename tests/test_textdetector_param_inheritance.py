import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from ballontranslator.modules import (
    TEXTDETECTORS,
    TEXTDETECTOR_PARAM_INHERITANCE,
)
from ballontranslator.modules.base import merge_config_module_params
from ballontranslator.modules.lazy_registry import _find_model_paths
from ballontranslator.modules.textdetector.detector_ysg import (
    _YSGYOLO_PARAMS,
    find_model_paths,
)
from ballontranslator.utils.config import (
    ProgramConfig,
    migrate_legacy_ysg_ctd_params,
)


class FakeSpec:
    def __init__(self, params):
        self.params = params

    def params_copy(self):
        return deepcopy(self.params)


def _param(value, **extra):
    return {'type': 'line_editor', 'value': value, **extra}


BASE_PARAMS = {
    'model path': {
        'type': 'selector',
        'options': ['data/models/ysgyolo_yolo26_2.0.pt'],
        'value': 'data/models/ysgyolo_yolo26_2.0.pt',
        'editable': True,
        'path_selector': True,
    },
    'confidence threshold': _param(0.3),
    'label': {
        'type': 'check_group',
        'value': {'balloon': True, 'qipao': True, 'other': False},
    },
    'mask dilate size': _param(2),
}
DERIVED_PARAMS = {
    **deepcopy(BASE_PARAMS),
    'ctd detect size': {'type': 'selector', 'options': [1024, 1280], 'value': 1280},
}


class ParamInheritanceTest(unittest.TestCase):
    def merge(self, config, inherit=True):
        registry = {
            'ysgyolo': FakeSpec(BASE_PARAMS),
            'ysg_ctd': FakeSpec(DERIVED_PARAMS),
        }
        return merge_config_module_params(
            config,
            ['ysgyolo', 'ysg_ctd'],
            registry.get,
            inherit_from={'ysg_ctd': 'ysgyolo'} if inherit else None,
        )

    def saved_base(self):
        return {
            'model path': 'data/models/ysgyolo_yolo26_2.0.pt',
            'confidence threshold': 0.2,
            'label': {'balloon': True, 'qipao': False, 'other': False},
            'mask dilate size': 2,
        }

    def test_new_derived_section_inherits_shared_params(self):
        config = {'ysgyolo': self.saved_base()}

        self.merge(config)

        derived = config['ysg_ctd']
        self.assertEqual(derived['confidence threshold']['value'], 0.2)
        self.assertEqual(derived['model path']['value'], 'data/models/ysgyolo_yolo26_2.0.pt')
        self.assertEqual(
            derived['label']['value'],
            {'balloon': True, 'qipao': False, 'other': False},
        )
        # Hybrid-only params keep their own defaults.
        self.assertEqual(derived['ctd detect size']['value'], 1280)

    def test_without_inheritance_new_section_keeps_template_defaults(self):
        config = {'ysgyolo': self.saved_base()}

        self.merge(config, inherit=False)

        self.assertEqual(config['ysg_ctd']['confidence threshold']['value'], 0.3)

    def test_default_valued_params_are_adopted_but_custom_values_survive(self):
        config = {
            'ysgyolo': self.saved_base(),
            'ysg_ctd': {
                'model path': 'data/models/custom.pt',
                'confidence threshold': 0.3,  # still the template default
                'label': {'balloon': True, 'qipao': True, 'other': False},
                'mask dilate size': 5,  # user customized
                'ctd detect size': 1024,
            },
        }

        self.merge(config)

        derived = config['ysg_ctd']
        self.assertEqual(derived['model path']['value'], 'data/models/custom.pt')
        self.assertEqual(derived['confidence threshold']['value'], 0.2)
        self.assertEqual(derived['mask dilate size']['value'], 5)
        self.assertEqual(derived['ctd detect size']['value'], 1024)

    def test_lazy_registry_specs_inherit_in_real_merge(self):
        config = {
            'ysgyolo': {
                'model path': 'data/models/ysgyolo_yolo26_2.0.pt',
                'confidence threshold': 0.2,
            },
        }

        merge_config_module_params(
            config,
            ['ysgyolo', 'ysg_ctd'],
            TEXTDETECTORS.get,
            inherit_from=TEXTDETECTOR_PARAM_INHERITANCE,
        )

        derived = config['ysg_ctd']
        self.assertEqual(derived['confidence threshold']['value'], 0.2)
        self.assertEqual(derived['model path']['value'], 'data/models/ysgyolo_yolo26_2.0.pt')
        self.assertIn('ctd detect size', derived)
        self.assertNotIn('ctd detect size', config['ysgyolo'])


class LegacyModelPathMigrationTest(unittest.TestCase):
    def test_helper_rewrites_stale_default_from_ysgyolo(self):
        params = {
            'ysgyolo': {'model path': 'data/models/ysgyolo_yolo26_2.0.pt'},
            'ysg_ctd': {'model path': 'data/models/ysgyolo_1.2_OS1.0.pt'},
        }

        migrate_legacy_ysg_ctd_params(params)

        self.assertEqual(
            params['ysg_ctd']['model path'], 'data/models/ysgyolo_yolo26_2.0.pt')

    def test_helper_leaves_custom_path_and_missing_sections_alone(self):
        custom = {'ysg_ctd': {'model path': 'data/my_model.pt'},
                  'ysgyolo': {'model path': 'data/models/ysgyolo_yolo26_2.0.pt'}}
        migrate_legacy_ysg_ctd_params(custom)
        self.assertEqual(custom['ysg_ctd']['model path'], 'data/my_model.pt')

        empty = {'ysg_ctd': {'model path': 'data/models/ysgyolo_1.2_OS1.0.pt'}}
        migrate_legacy_ysg_ctd_params(empty)
        self.assertEqual(
            empty['ysg_ctd']['model path'], 'data/models/ysgyolo_1.2_OS1.0.pt')

    def test_program_config_load_migrates_stale_section(self):
        payload = {
            'module': {
                'textdetector': 'ysg_ctd',
                'textdetector_params': {
                    'ysgyolo': {'model path': 'data/models/ysgyolo_yolo26_2.0.pt'},
                    'ysg_ctd': {'model path': 'data/models/ysgyolo_1.2_OS1.0.pt'},
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            config = ProgramConfig.load(str(path))

        self.assertEqual(
            config.module.textdetector_params['ysg_ctd']['model path'],
            'data/models/ysgyolo_yolo26_2.0.pt',
        )


class ModelPathOrderingTest(unittest.TestCase):
    def test_default_checkpoints_come_first_for_runtime_and_lazy_scan(self):
        expected = [
            'data/models/ysgyolo_yolo26_2.0.pt',
            'data/models/ysgyolo_yolo26OBB_2.0.pt',
        ]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'ysgyolo_demo.pt').write_text('', encoding='utf8')
            runtime_paths = find_model_paths(directory, ('ysgyolo',))
            lazy_paths = _find_model_paths(directory, ('ysgyolo',))

        self.assertEqual(runtime_paths[:2], expected)
        self.assertEqual(lazy_paths, runtime_paths)

    def test_template_default_points_at_primary_checkpoint(self):
        self.assertEqual(
            _YSGYOLO_PARAMS['model path']['value'],
            'data/models/ysgyolo_yolo26_2.0.pt',
        )


if __name__ == '__main__':
    unittest.main()

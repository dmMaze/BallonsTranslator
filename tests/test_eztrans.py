import sys
import unittest
from unittest.mock import MagicMock

from ballontranslator.modules.package_import_names import PACKAGE_IMPORT_NAMES
from ballontranslator.utils.py_package_manager import (
    DEFAULT_PACKAGE_IMPORT_NAMES,
    PyPackageManager,
)
from ballontranslator.utils.textblock import TextBlock


@unittest.skipUnless(sys.platform == 'win32', 'ezTrans is only registered on Windows')
class EzTransTranslatorTest(unittest.TestCase):

    def test_msl_loadlib_package_import_name_targets_subpackage(self):
        # msl-loadlib installs `msl` as a PEP 420 namespace directory without
        # `msl/__init__.py`, so `PyPackageManager._import_available('msl')`
        # rejects it unless mapped to `msl.loadlib`.
        self.assertEqual(PACKAGE_IMPORT_NAMES.get('msl-loadlib'), ['msl.loadlib'])
        self.assertEqual(DEFAULT_PACKAGE_IMPORT_NAMES.get('msl-loadlib'), ['msl.loadlib'])
        mgr = PyPackageManager(package_import_names=PACKAGE_IMPORT_NAMES)
        self.assertEqual(mgr.import_names_for_requirement('msl-loadlib'), ['msl.loadlib'])

    def test_fullwidth_to_halfwidth_supports_str_and_list(self):
        from ballontranslator.modules.translators.trans_eztrans import fullwidth_to_halfwidth

        self.assertEqual(
            fullwidth_to_halfwidth('こんにちは！　ＡＢＣ１２３'),
            'こんにちは! ABC123',
        )
        self.assertEqual(
            fullwidth_to_halfwidth(['こんにちは！', 'テストです。　１２３']),
            ['こんにちは!', 'テストです。 123'],
        )

    def test_eztrans_2x2_matrix_and_passthrough_with_list_inputs(self):
        from ballontranslator.modules.translators.trans_eztrans import ezTransTranslator

        translator = ezTransTranslator.__new__(ezTransTranslator)
        translator.lang_map = {'한국어': 'k', '日本語': 'j'}
        translator.lang_source = '한국어'
        translator.lang_target = '日本語'
        translator.concate_text = False
        translator.textblk_break = '\n##\n'
        translator.j2k_engine = MagicMock()
        translator.j2k_engine.translate.side_effect = lambda xs: [f'KO:{x}' for x in xs]
        translator.k2j_engine = MagicMock()
        translator.k2j_engine.translate.side_effect = lambda xs: [f'ＪＡ：{x}！' for x in xs]

        # 1. Korean -> Japanese (converts fullwidth ASCII/punctuation to halfwidth on List[str])
        out_k2j = translator._translate(['안녕하세요', '테스트'])
        self.assertEqual(out_k2j, ['JA:안녕하세요!', 'JA:테스트!'])

        # 2. Japanese -> Korean
        translator.lang_source = '日本語'
        translator.lang_target = '한국어'
        out_j2k = translator._translate(['こんにちは', 'テスト'])
        self.assertEqual(out_j2k, ['KO:こんにちは', 'KO:テスト'])

        # 3. Pass-through (Japanese -> Japanese & Korean -> Korean)
        translator.lang_source = '日本語'
        translator.lang_target = '日本語'
        self.assertEqual(translator._translate(['パススルー1', 'パススルー2']), ['パススルー1', 'パススルー2'])

        translator.lang_source = '한국어'
        translator.lang_target = '한국어'
        self.assertEqual(translator._translate(['패스스루1', '패스스루2']), ['패스스루1', '패스스루2'])


if __name__ == '__main__':
    unittest.main()

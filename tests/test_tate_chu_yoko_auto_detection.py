import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtGui import QTextDocument
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.annotations import (
    apply_auto_text_combine_upright,
    text_combine_upright_ranges,
)
from ballontranslator.ui.text_engine.pipeline_formatting import (
    apply_auto_tate_chu_yoko,
)
from ballontranslator.utils.config import AutoTateChuYokoConfig
from ballontranslator.utils.textblock import TextBlock


class AutoTateChuYokoDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _ranges(text: str, allowed: str, max_length: int = 3):
        document = QTextDocument(text)
        changed = apply_auto_text_combine_upright(
            document, frozenset(allowed), max_length
        )
        return changed, text_combine_upright_ranges(document.firstBlock())

    def test_full_width_list_matches_ascii_punctuation(self):
        changed, ranges = self._ranges('干什么…!?', '！？')
        self.assertTrue(changed)
        self.assertEqual([(start, length) for start, length, _id in ranges], [(4, 2)])

    def test_ascii_list_matches_full_width_punctuation(self):
        changed, ranges = self._ranges('干什么…！？', '!?')
        self.assertTrue(changed)
        self.assertEqual([(start, length) for start, length, _id in ranges], [(4, 2)])

    def test_mixed_width_run_is_one_group(self):
        changed, ranges = self._ranges('干什么…！?', '！？')
        self.assertTrue(changed)
        self.assertEqual([(start, length) for start, length, _id in ranges], [(4, 2)])

    def test_width_matching_does_not_reach_unrelated_punctuation(self):
        changed, ranges = self._ranges('a.b', '。')
        self.assertFalse(changed)
        self.assertEqual(ranges, ())

    def test_digits_keep_their_run_length_limit(self):
        changed, ranges = self._ranges('12', '！？', max_length=1)
        self.assertFalse(changed)
        self.assertEqual(ranges, ())

    def test_project_apply_combines_pasted_ascii_punctuation(self):
        block = TextBlock()
        block.vertical = True
        block.translation = '你……你到底要干什么…!?'
        settings = AutoTateChuYokoConfig(
            enabled=True,
            max_length=3,
            include_numbers=True,
            include_letters=True,
            additional_chars='！？',
        )

        changed = apply_auto_tate_chu_yoko([block], settings)

        self.assertEqual(changed, 1)
        self.assertIn('text-combine-upright: all', block.rich_text)
        self.assertIn('>!?<', block.rich_text)


if __name__ == '__main__':
    unittest.main()

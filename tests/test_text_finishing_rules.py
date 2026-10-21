"""Finishing rules: whole Latin words, single ellipsis, grouped punctuation."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.pipeline_formatting import (
    apply_auto_tate_chu_yoko,
    collapse_ellipsis_runs,
    normalize_translation_punctuation,
)
from ballontranslator.ui.text_engine.vertical_layout import (
    _is_non_fullwidth_roman,
    _roman_word_ends,
)
from ballontranslator.utils.config import AutoTateChuYokoConfig
from ballontranslator.utils.fontformat import TextAlignment
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


def make_item(text, bounds, rotation=True, font_size=30) -> TextBlkItem:
    block = TextBlock(list(bounds), text_layout_version=TEXT_LAYOUT_VERSION)
    block._bounding_rect = list(bounds)
    block.vertical = True
    block.alignment = TextAlignment.Left
    block.translation = text
    block.fontformat.font_size = font_size
    block.fontformat.letter_spacing = 1.0
    block.fontformat.roman_word_rotation = rotation
    return TextBlkItem(block, 0)


def letter_columns(item, latin_only=False) -> list:
    layout = item.layout
    block = item.document().firstBlock().layout()
    text = item.toPlainText()
    columns = []
    for index in range(block.lineCount()):
        start = block.lineAt(index).textStart()
        char = text[start]
        if char.isalpha() and _is_non_fullwidth_roman(char):
            if latin_only and not char.isascii():
                continue
            columns.append(round(block.lineAt(index).x(), 1))
    return columns


class RomanWordEndsTest(unittest.TestCase):

    def test_maps_only_multi_letter_words(self) -> None:
        self.assertEqual(_roman_word_ends('a Master b'), {2: 8})
        self.assertEqual(_roman_word_ends('漢字のみ'), {})
        self.assertEqual(_roman_word_ends('AB cd'), {0: 2, 3: 5})


class WholeWordColumnsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_word_moves_to_the_next_column_instead_of_splitting(self) -> None:
        """Two CJK cells leave 140 of 200; the six-letter word needs 180."""
        item = make_item('天地master', (0, 0, 200, 200))
        columns = letter_columns(item)
        self.assertTrue(columns)
        self.assertEqual(len(set(columns)), 1, columns)
        # The word starts a fresh column rather than trailing the CJK cells.
        block = item.document().firstBlock().layout()
        text = item.toPlainText()
        first = next(
            index for index in range(block.lineCount())
            if text[block.lineAt(index).textStart()] == 'm'
        )
        self.assertAlmostEqual(block.lineAt(first).y(), 0.0, delta=0.5)

    def test_word_taller_than_the_column_overflows_whole(self) -> None:
        """Compression stops at 92%; beyond that the word overflows intact.

        This is the reported case: a small box where six letters cannot be
        squeezed in even with tight tracking, so the word must stay whole.
        """
        item = make_item('master', (0, 0, 90, 120), font_size=59.5)
        columns = letter_columns(item, latin_only=True)
        self.assertEqual(len(columns), 6)
        self.assertEqual(len(set(columns)), 1, columns)

    def test_word_fits_a_fresh_column_after_moving(self) -> None:
        item = make_item('天地master', (0, 0, 200, 200))
        columns = letter_columns(item, latin_only=True)
        self.assertEqual(len(set(columns)), 1, columns)


class EllipsisNormalizationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_collapses_a_run_of_ellipsis_marks(self) -> None:
        self.assertEqual(collapse_ellipsis_runs('等等……'), '等等…')
        self.assertEqual(collapse_ellipsis_runs('……啊'), '…啊')
        self.assertEqual(collapse_ellipsis_runs('...'), '...')
        self.assertEqual(collapse_ellipsis_runs('…'), '…')

    def test_updates_translation_and_rich_text(self) -> None:
        block = TextBlock(translation='等等……')
        block.rich_text = '<p>等等……</p>'
        self.assertEqual(normalize_translation_punctuation([block]), 1)
        self.assertEqual(block.translation, '等等…')
        self.assertEqual(block.rich_text, '<p>等等…</p>')
        self.assertEqual(normalize_translation_punctuation([block]), 0)


class RepeatedPunctuationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _block() -> TextBlock:
        bounds = [0, 0, 200, 300]
        block = TextBlock(list(bounds), text_layout_version=TEXT_LAYOUT_VERSION)
        block._bounding_rect = list(bounds)
        block.vertical = True
        block.alignment = TextAlignment.Left
        block.translation = '真的！！！'
        block.fontformat.font_size = 30
        block.fontformat.letter_spacing = 1.0
        block.fontformat.vertical = True
        return block

    def _settings(self) -> AutoTateChuYokoConfig:
        return AutoTateChuYokoConfig(
            enabled=True,
            max_length=4,
            include_numbers=True,
            include_letters=False,
            additional_chars='！？',
        )

    def test_repeated_marks_become_one_combined_run(self) -> None:
        block = self._block()
        apply_auto_tate_chu_yoko([block], self._settings())

        item = TextBlkItem(block, 0)
        ranges = list(item.layout.text_combine_ranges[0])
        self.assertEqual(len(ranges), 1)
        start, length = ranges[0][0], ranges[0][1]
        self.assertEqual(item.toPlainText()[start:start + length], '！！！')

    def test_disabled_settings_change_nothing(self) -> None:
        block = self._block()
        settings = self._settings()
        settings.enabled = False
        self.assertEqual(apply_auto_tate_chu_yoko([block], settings), 0)


if __name__ == '__main__':
    unittest.main()

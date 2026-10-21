"""Compact rotated Roman words in vertical layout.

The option changes only Latin letters: punctuation, digits, and fullwidth
forms keep whichever orientation the standard Roman setting selects.
"""

import os
import unittest
import warnings


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.vertical_layout import (
    VerticalTextDocumentLayout,
)
from ballontranslator.utils.fontformat import FontFormat, TextAlignment
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


BOUNDS = (0, 0, 240, 400)
TEXT = '怎么了，Master，你'
FONT_SIZE = 30


class RomanWordRotationLayoutTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _make_item(
        self,
        *,
        standard_roman: bool = True,
        word_rotation: bool = False,
        vertical: bool = True,
        text: str = TEXT,
        bounds=BOUNDS,
    ) -> TextBlkItem:
        block = TextBlock(list(bounds), text_layout_version=TEXT_LAYOUT_VERSION)
        block._bounding_rect = list(bounds)
        block.vertical = vertical
        block.alignment = TextAlignment.Center
        block.translation = text
        block.fontformat.font_size = FONT_SIZE
        block.fontformat.letter_spacing = 1.15
        block.fontformat.standard_vertical_roman_alignment = standard_roman
        block.fontformat.roman_word_rotation = word_rotation
        return TextBlkItem(block, 0)

    @staticmethod
    def _advances(item: TextBlkItem) -> dict:
        """Map each character to its settled vertical advance."""
        layout = item.layout
        text = item.toPlainText()
        block = item.document().firstBlock().layout()
        result = {}
        for index in range(block.lineCount()):
            start = block.lineAt(index).textStart()
            top, bottom = layout.y_offset_lst[0][index]
            result.setdefault(
                text[start], round(float(bottom) - float(top), 2)
            )
        return result

    def test_option_rotates_latin_letters_only(self) -> None:
        item = self._make_item(word_rotation=True)
        layout = item.layout
        for letter in 'Master':
            with self.subTest(letter=letter):
                self.assertTrue(layout.needs_vertical_rotation(letter))
        for untouched in ('了', '，', '你', '2', '!', 'Ａ'):
            with self.subTest(char=untouched):
                self.assertFalse(layout.needs_vertical_rotation(untouched))

    def test_disabled_option_keeps_letters_upright(self) -> None:
        layout = self._make_item().layout
        for letter in 'Master':
            with self.subTest(letter=letter):
                self.assertFalse(layout.needs_vertical_rotation(letter))
                self.assertTrue(layout.centers_vertical_glyph(letter))

    def test_enabled_option_stops_centering_rotated_letters(self) -> None:
        layout = self._make_item(word_rotation=True).layout
        self.assertFalse(layout.centers_vertical_glyph('M'))

    def test_letters_take_the_rotated_branch_advance(self) -> None:
        """Rotation and the compressed advance must arrive together.

        With the option the letters must match exactly what the fully rotated
        mode produces, so a font-dependent measurement is never asserted
        directly.
        """
        enabled = self._advances(self._make_item(word_rotation=True))
        reference = self._advances(
            self._make_item(standard_roman=False)
        )
        for letter in 'Master':
            with self.subTest(letter=letter):
                self.assertEqual(enabled[letter], reference[letter])

    def test_punctuation_and_cjk_keep_their_advance(self) -> None:
        baseline = self._advances(self._make_item())
        enabled = self._advances(self._make_item(word_rotation=True))
        for char in '怎么了，你':
            with self.subTest(char=char):
                self.assertEqual(enabled[char], baseline[char])

    def test_word_never_grows_and_letters_shrink_when_measurable(self) -> None:
        def content_height(item):
            layout = item.layout
            return max(
                float(layout.y_offset_lst[0][index][1])
                for index in range(
                    item.document().firstBlock().layout().lineCount()
                )
            )

        upright = content_height(self._make_item())
        rotated = content_height(self._make_item(word_rotation=True))
        self.assertLessEqual(rotated, upright + 1e-6)

    def test_letter_rotation_survives_row_and_column_changes(self) -> None:
        """Narrow boxes force new columns; every letter stays rotated."""
        item = self._make_item(word_rotation=True, bounds=(0, 0, 120, 140))
        layout = item.layout
        block = item.document().firstBlock().layout()
        text = item.toPlainText()
        letters = 0
        for index in range(block.lineCount()):
            char = text[block.lineAt(index).textStart()]
            if char in 'Master':
                letters += 1
                self.assertTrue(layout.needs_vertical_rotation(char))
        self.assertEqual(letters, len(set('Master')))

    def test_horizontal_layout_is_inert(self) -> None:
        off = self._make_item(vertical=False)
        on = self._make_item(vertical=False, word_rotation=True)
        self.assertIsNot(on.layout, off.layout)
        self.assertFalse(
            isinstance(on.layout, VerticalTextDocumentLayout)
        )
        off_lines = [
            (round(off.document().firstBlock().layout().lineAt(i).y(), 3),
             off.document().firstBlock().layout().lineAt(i).width())
            for i in range(off.document().firstBlock().layout().lineCount())
        ]
        on_lines = [
            (round(on.document().firstBlock().layout().lineAt(i).y(), 3),
             on.document().firstBlock().layout().lineAt(i).width())
            for i in range(on.document().firstBlock().layout().lineCount())
        ]
        self.assertEqual(off_lines, on_lines)

    def test_live_toggle_matches_fresh_construction(self) -> None:
        live = self._make_item()
        live.setRomanWordRotation(True)
        fresh = self._make_item(word_rotation=True)
        self.assertEqual(
            self._advances(live), self._advances(fresh)
        )

    def test_toggling_to_the_same_value_is_a_noop(self) -> None:
        item = self._make_item(word_rotation=True)
        before = self._advances(item)
        item.setRomanWordRotation(True)
        self.assertEqual(self._advances(item), before)

    def test_rotation_composes_with_the_vertical_anchor(self) -> None:
        """Both options settle in the same relayout without overriding it."""
        from ballontranslator.utils.fontformat import TextVerticalAlignment

        item = self._make_item(
            word_rotation=True, bounds=(0, 0, 300, 400)
        )
        item.fontformat.vertical_alignment = TextVerticalAlignment.Center
        item.layout.reLayout()
        layout = item.layout
        lines = [
            layout.y_offset_lst[0][index][0]
            for index in range(
                item.document().firstBlock().layout().lineCount()
            )
        ]
        self.assertGreater(min(lines), 1.0)
        slack = max(0.0, layout.available_height - layout.shrink_height)
        self.assertAlmostEqual(min(lines), slack / 2, delta=0.5)
        for letter in 'Master':
            self.assertTrue(layout.needs_vertical_rotation(letter))

    def test_column_cache_key_separates_both_rotation_values(self) -> None:
        """Toggling must not replay a column measured upright.

        Vertical layout caches plain-column geometry under
        ``_plain_column_context``. Two configurations that settle different
        advances must therefore produce different keys; otherwise enabling the
        option on an existing item keeps the old one-emoji-per-cell geometry.
        """
        upright = self._make_item()
        rotated = self._make_item(word_rotation=True)
        self.assertNotEqual(
            upright.layout._plain_column_context,
            rotated.layout._plain_column_context,
        )

        before = upright.layout._plain_column_context
        upright.setRomanWordRotation(True)
        self.assertNotEqual(upright.layout._plain_column_context, before)


class RomanWordRotationCommandTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_command_is_registered(self) -> None:
        from ballontranslator.ui.text_engine.formatting.commands import (
            handle_ffmt_change,
        )

        self.assertIn('roman_word_rotation', handle_ffmt_change)

    def test_command_applies_as_one_undo_step(self) -> None:
        from unittest.mock import patch

        from ballontranslator.ui.text_engine.formatting import commands

        item = RomanWordRotationLayoutTest()._make_item()
        pushed = []

        class _Canvas:
            def push_undo_command(self, command):
                pushed.append(command)
                command.redo()

        with patch.object(commands.SW, 'canvas', _Canvas(), create=True):
            commands.ffmt_change_roman_word_rotation(
                'roman_word_rotation',
                True,
                FontFormat(),
                False,
                [item],
                set_focus=False,
            )

        self.assertEqual(len(pushed), 1)
        self.assertTrue(item.fontformat.roman_word_rotation)
        pushed[0].undo()
        self.assertFalse(item.fontformat.roman_word_rotation)


class RomanWordRotationPersistenceTest(unittest.TestCase):

    def test_default_is_disabled_and_round_trips(self) -> None:
        self.assertFalse(FontFormat().roman_word_rotation)
        format_ = FontFormat()
        format_.roman_word_rotation = True
        self.assertTrue(
            format_.to_serializable_dict()['roman_word_rotation']
        )
        restored = FontFormat(**format_.to_serializable_dict())
        self.assertTrue(restored.roman_word_rotation)

    def test_invalid_value_falls_back_to_disabled(self) -> None:
        with self.assertLogs('BallonTranslator', level='WARNING'):
            format_ = FontFormat(roman_word_rotation='yes')
        self.assertFalse(format_.roman_word_rotation)


if __name__ == '__main__':
    unittest.main()

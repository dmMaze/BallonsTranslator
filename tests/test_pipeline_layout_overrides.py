"""Batch RUN copies the settled-layout options according to their flags."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.pipeline_formatting import (
    apply_layout_overrides,
)
from ballontranslator.utils.fontformat import FontFormat, TextVerticalAlignment
from ballontranslator.utils.textblock import TextBlock


class LayoutOverridesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _blocks() -> list:
        blocks = []
        for index in range(3):
            block = TextBlock(translation=f'page {index}')
            block.fontformat.font_size = 30 + index
            block.fontformat.line_spacing = 1.2
            blocks.append(block)
        return blocks

    def test_flags_off_keep_each_block_value(self) -> None:
        blocks = self._blocks()
        global_format = FontFormat()
        global_format.roman_word_rotation = True
        global_format.vertical_alignment = TextVerticalAlignment.Center

        changed = apply_layout_overrides(blocks, global_format)

        self.assertEqual(changed, 0)
        for block in blocks:
            self.assertFalse(block.fontformat.roman_word_rotation)
            self.assertIs(
                block.fontformat.vertical_alignment, TextVerticalAlignment.Auto
            )

    def test_rotation_flag_copies_the_global_default(self) -> None:
        blocks = self._blocks()
        global_format = FontFormat()
        global_format.roman_word_rotation = True

        changed = apply_layout_overrides(
            blocks, global_format, roman_word_rotation=True
        )

        self.assertEqual(changed, 3)
        for block in blocks:
            self.assertTrue(block.fontformat.roman_word_rotation)
        # The undo/redo contract relies on the second pass being a no-op.
        self.assertEqual(
            apply_layout_overrides(
                blocks, global_format, roman_word_rotation=True
            ),
            0,
        )

    def test_vertical_alignment_flag_copies_the_global_default(self) -> None:
        blocks = self._blocks()
        global_format = FontFormat()
        global_format.vertical_alignment = TextVerticalAlignment.Center

        changed = apply_layout_overrides(
            blocks, global_format, vertical_alignment=True
        )

        self.assertEqual(changed, 3)
        for block in blocks:
            self.assertIs(
                block.fontformat.vertical_alignment, TextVerticalAlignment.Center
            )

    def test_flag_can_turn_the_option_off_again(self) -> None:
        """'Use the global setting' means the global value, not just enabling."""
        blocks = self._blocks()
        for block in blocks:
            block.fontformat.roman_word_rotation = True
        global_format = FontFormat()
        global_format.roman_word_rotation = False

        changed = apply_layout_overrides(
            blocks, global_format, roman_word_rotation=True
        )

        self.assertEqual(changed, 3)
        for block in blocks:
            self.assertFalse(block.fontformat.roman_word_rotation)

    def test_only_the_selected_option_is_written(self) -> None:
        blocks = self._blocks()
        global_format = FontFormat()
        global_format.roman_word_rotation = True
        global_format.vertical_alignment = TextVerticalAlignment.Bottom
        sizes = [block.fontformat.font_size for block in blocks]

        apply_layout_overrides(blocks, global_format, roman_word_rotation=True)

        for block, size in zip(blocks, sizes):
            self.assertAlmostEqual(block.fontformat.font_size, size)
            self.assertTrue(block.fontformat.roman_word_rotation)
            self.assertIs(
                block.fontformat.vertical_alignment, TextVerticalAlignment.Auto
            )


if __name__ == '__main__':
    unittest.main()

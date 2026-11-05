"""Applying a whole text style must not move or clear inline annotations.

A style preset is a ``FontFormat``: it owns font, colour, spacing, effects,
alignment, and transform, and it replaces every fragment format in the block.
Tate-chu-yoko, Ruby, and emphasis are per-character annotations the user placed
by hand, so a style apply keeps them on the characters they belong to — it
neither clears them nor copies the first character's annotation over the block.

The reported failure: a vertical block whose ``!?`` was combined lost its
Tate-chu-yoko as soon as a preset was applied from the style list.
"""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtGui import QTextCharFormat, QTextCursor
from qtpy.QtWidgets import QApplication, QTextEdit

from ballontranslator.ui.text_engine.annotations import (
    AnnotationProperty,
    apply_emphasis,
    apply_ruby,
    apply_text_combine_upright,
    emphasis_ranges,
    emphasis_values,
    inline_annotation_state,
    restore_inline_annotations,
    ruby_containers,
    text_combine_upright_ranges,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.editing.commands import ApplyFontformatCommand
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock

TEXT = '你……你到底要干什么…!?'


def _shape(state) -> tuple:
    """Describe an annotation state without its runtime-only identities."""
    return (
        tuple((start, length) for start, length, _id in state.text_combine),
        tuple(
            (
                container.start,
                container.length,
                container.ruby_type,
                container.position,
                tuple((unit.start, unit.length, unit.text) for unit in container.units),
            )
            for container in state.ruby
        ),
        tuple(
            (entry.start, entry.length, entry.style, entry.position)
            for entry in state.emphasis
        ),
    )


class TextStyleAnnotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(text: str = TEXT, vertical: bool = True) -> TextBlkItem:
        bounds = (0, 0, 200, 400)
        block = TextBlock(list(bounds))
        block._bounding_rect = list(bounds)
        block.vertical = vertical
        block.translation = text
        item = TextBlkItem(block, 0)
        item.fontformat.vertical = vertical
        return item

    @staticmethod
    def _select(item: TextBlkItem, start: int, end: int) -> QTextCursor:
        cursor = QTextCursor(item.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        return cursor

    @classmethod
    def _combine(cls, item: TextBlkItem, start: int, end: int) -> None:
        apply_text_combine_upright(cls._select(item, start, end), True)

    @staticmethod
    def _style(font_size: float = 50.0) -> FontFormat:
        """A preset as the style list stores it: no annotation fields."""
        style = FontFormat()
        style.font_size = font_size
        style.letter_spacing = 1.3
        style.frgb = [200, 30, 30]
        return style

    def test_style_apply_keeps_tate_chu_yoko_group(self):
        item = self._make_item()
        self._combine(item, 9, 11)
        before = text_combine_upright_ranges(item.document().firstBlock())
        self.assertEqual([entry[:2] for entry in before], [(9, 2)])

        item.set_fontformat(self._style(), set_char_format=True)

        after = text_combine_upright_ranges(item.document().firstBlock())
        self.assertEqual(after, before)
        self.assertIn(
            'text-combine-upright',
            to_rich_text_html(item.document()),
        )
        item.deleteLater()

    def test_style_apply_keeps_ruby_container_identity(self):
        item = self._make_item('又要去了')
        apply_ruby(self._select(item, 2, 3), 'mono', 'か', 'over')
        before = ruby_containers(item.document())
        self.assertEqual(len(before), 1)

        item.set_fontformat(self._style(), set_char_format=True)

        after = ruby_containers(item.document())
        self.assertEqual(after, before)
        item.deleteLater()

    def test_style_apply_keeps_emphasis_runs(self):
        item = self._make_item('又要去了')
        apply_emphasis(self._select(item, 1, 2), 'filled dot', 'over right')
        before = emphasis_ranges(item.document())
        self.assertEqual([(entry.start, entry.length) for entry in before], [(1, 1)])

        item.set_fontformat(self._style(), set_char_format=True)

        self.assertEqual(emphasis_ranges(item.document()), before)
        item.deleteLater()

    def test_style_apply_does_not_spread_the_first_character_annotation(self):
        item = self._make_item('又要去了！！')
        # The combined run starts the block, so the replacing format is built
        # from an annotated character.
        self._combine(item, 0, 2)
        apply_emphasis(self._select(item, 0, 1), 'open circle', 'under left')

        item.set_fontformat(self._style(), set_char_format=True)

        block = item.document().firstBlock()
        self.assertEqual(
            [entry[:2] for entry in text_combine_upright_ranges(block)],
            [(0, 2)],
        )
        self.assertEqual(
            [(entry.start, entry.length) for entry in emphasis_ranges(item.document())],
            [(0, 1)],
        )
        cursor = QTextCursor(item.document())
        cursor.setPosition(5)
        self.assertEqual(emphasis_values(cursor.charFormat())[0], 'none')
        item.deleteLater()

    def test_style_apply_leaves_a_plain_block_plain(self):
        item = self._make_item('又要去了')

        item.set_fontformat(self._style(), set_char_format=True)

        block = item.document().firstBlock()
        self.assertEqual(text_combine_upright_ranges(block), ())
        self.assertEqual(ruby_containers(item.document()), ())
        self.assertEqual(emphasis_ranges(item.document()), ())
        self.assertNotIn('text-combine-upright', to_rich_text_html(item.document()))
        item.deleteLater()

    def test_style_apply_still_owns_font_colour_and_spacing(self):
        item = self._make_item('又要去了！！')
        cursor = self._select(item, 0, 2)
        override = QTextCharFormat()
        override.setFontPointSize(11.0)
        cursor.setCharFormat(override)
        self._combine(item, 4, 6)

        item.set_fontformat(self._style(font_size=50.0), set_char_format=True)

        document = item.document()
        block = document.firstBlock()
        sizes = set()
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.length() > 0:
                sizes.add(round(fragment.charFormat().font().pointSizeF(), 3))
            iterator += 1
        self.assertEqual(sizes, {round(item.fontformat.size_pt, 3)})
        self.assertEqual(
            [entry[:2] for entry in text_combine_upright_ranges(block)],
            [(4, 2)],
        )
        item.deleteLater()

    def test_annotations_survive_a_round_trip_through_saved_rich_text(self):
        source = self._make_item()
        self._combine(source, 9, 11)
        apply_ruby(self._select(source, 2, 3), 'mono', 'か', 'over')
        apply_emphasis(self._select(source, 4, 6), 'filled sesame', 'over right')
        html = to_rich_text_html(source.document())

        item = self._make_item('')
        item.load_rich_text_html(html)
        # Loading allocates fresh runtime group and container identities, so
        # the shape is compared before the style takes over the formatting.
        self.assertEqual(
            _shape(inline_annotation_state(item.document())),
            _shape(inline_annotation_state(source.document())),
        )
        loaded = inline_annotation_state(item.document())

        item.set_fontformat(self._style(), set_char_format=True)

        self.assertEqual(inline_annotation_state(item.document()), loaded)
        item.deleteLater()
        source.deleteLater()

    def test_style_command_round_trip_keeps_the_group(self):
        item = self._make_item()
        self._combine(item, 9, 11)
        before = text_combine_upright_ranges(item.document().firstBlock())
        edit = QTextEdit()
        self.addCleanup(edit.deleteLater)
        command = ApplyFontformatCommand([item], [edit], self._style())

        command.redo()
        self.assertEqual(
            text_combine_upright_ranges(item.document().firstBlock()),
            before,
        )

        command.undo()
        self.assertEqual(
            [entry[:2] for entry in
             text_combine_upright_ranges(item.document().firstBlock())],
            [(9, 2)],
        )
        item.deleteLater()

    def test_state_restoration_ignores_annotated_foreign_properties(self):
        item = self._make_item('又要去了')
        document = item.document()
        cursor = self._select(item, 0, 1)
        fmt = QTextCharFormat()
        fmt.setProperty(AnnotationProperty.LETTER_SPACING, 1.5)
        cursor.mergeCharFormat(fmt)

        state = inline_annotation_state(document)
        self.assertTrue(state.empty)
        self.assertFalse(restore_inline_annotations(document, state))
        cursor = self._select(item, 0, 1)
        self.assertAlmostEqual(
            cursor.charFormat().property(AnnotationProperty.LETTER_SPACING),
            1.5,
        )
        item.deleteLater()


if __name__ == '__main__':
    unittest.main()

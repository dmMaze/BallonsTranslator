import os
import unittest
from unittest.mock import patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.misc import doc_replace, pixmap2ndarray
from ballontranslator.ui.text_engine.annotations import (
    LETTER_SPACING_ATTRIBUTE,
    TEXT_COMBINE_ID_ATTRIBUTE,
    apply_emphasis,
    apply_letter_spacing,
    apply_text_combine_upright,
    create_rich_text_mime,
    emphasis_values,
    insert_rich_text_mime,
    letter_spacing_value,
    load_rich_text_html,
    text_combine_upright_ranges,
    text_combine_upright_values,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.formatting.advanced import (
    TateChuYokoGroup,
    TextEmphasisGroup,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.emphasis import emphasis_ink_bounds
from ballontranslator.ui.text_engine.rendering.indexing import _grapheme_ranges
from ballontranslator.ui.text_engine.rendering.tate_chu_yoko import (
    tate_chu_yoko_ink_bounds,
    tate_chu_yoko_natural_bounds,
)
from ballontranslator.utils.textblock import TextBlock


def _format_at(document: QTextDocument, start: int, length: int = 1):
    cursor = QTextCursor(document)
    cursor.setPosition(start)
    cursor.setPosition(start + length, QTextCursor.MoveMode.KeepAnchor)
    return cursor.charFormat()


class RichTextAnnotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(
        vertical: bool = False,
        *,
        bounds=(0, 0, 600, 300),
        text: str = '強調 test',
    ) -> TextBlkItem:
        block = TextBlock(list(bounds))
        block._bounding_rect = list(bounds)
        block.vertical = vertical
        block.translation = text
        return TextBlkItem(block, 0)

    def test_old_qt_html_loads_without_extensions_or_format_loss(self):
        source = QTextDocument()
        source.setPlainText('old rich text')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        cursor.mergeCharFormat(char_format)
        old_html = source.toHtml()

        restored = QTextDocument()
        load_rich_text_html(restored, old_html)

        self.assertEqual(restored.toPlainText(), 'old rich text')
        self.assertTrue(_format_at(restored, 0, 3).font().bold())
        self.assertNotIn(
            'text-emphasis-style',
            to_rich_text_html(restored),
        )

    def test_old_qt_html_skips_extension_parser_and_keeps_spacing_fallback(self):
        source = QTextDocument()
        source.setPlainText('old rich text')
        restored = QTextDocument()

        with patch(
            'ballontranslator.ui.text_engine.annotations.'
            '_inline_extension_ranges_from_html'
        ) as parse_extensions:
            load_rich_text_html(
                restored,
                source.toHtml(),
                letter_spacing_fallback=1.25,
            )

        parse_extensions.assert_not_called()
        self.assertEqual(restored.toPlainText(), 'old rich text')
        self.assertEqual(letter_spacing_value(_format_at(restored, 0)), 1.25)

    def test_emphasis_inline_round_trip_keeps_fragment_style(self):
        source = QTextDocument()
        source.setPlainText('A𠮷B')
        cursor = QTextCursor(source)
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        char_format.setFontItalic(True)
        char_format.setForeground(QColor('#c02040'))
        cursor.mergeCharFormat(char_format)
        apply_emphasis(cursor, 'filled sesame', 'under left')

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)
        restored_format = _format_at(restored, 1, 2)
        legacy_reader = QTextDocument()
        legacy_reader.setHtml(html)

        self.assertTrue(html.startswith('<!DOCTYPE html>'))
        self.assertIn('text-emphasis-style: filled sesame', html)
        self.assertIn('text-emphasis-position: under left', html)
        self.assertNotIn('ballontranslator-rich-text', html)
        self.assertEqual(restored.toPlainText(), 'A𠮷B')
        self.assertEqual(legacy_reader.toPlainText(), 'A𠮷B')
        self.assertTrue(_format_at(legacy_reader, 1, 2).font().bold())
        self.assertEqual(
            emphasis_values(_format_at(legacy_reader, 1, 2))[0],
            'none',
        )
        self.assertEqual(
            emphasis_values(restored_format),
            ('filled sesame', 'under left'),
        )
        self.assertTrue(restored_format.font().bold())
        self.assertTrue(restored_format.font().italic())
        self.assertEqual(restored_format.foreground().color(), QColor('#c02040'))

    def test_invalid_inline_extension_drops_only_the_annotation(self):
        source = QTextDocument()
        source.setPlainText('safe text')
        html = source.toHtml().replace(
            'safe text',
            '<span style="text-emphasis-style: sparks; '
            'text-emphasis-position: over right;">safe</span> text',
        )
        restored = QTextDocument()

        load_rich_text_html(restored, html)

        self.assertEqual(restored.toPlainText(), 'safe text')
        self.assertEqual(
            emphasis_values(_format_at(restored, 0)),
            ('none', 'over right'),
        )

    def test_invalid_letter_spacing_keeps_legacy_fallback(self):
        source = QTextDocument()
        source.setPlainText('safe text')
        html = source.toHtml().replace(
            'safe text',
            f'<span {LETTER_SPACING_ATTRIBUTE}="wide">safe</span> text',
        )
        restored = QTextDocument()

        load_rich_text_html(
            restored,
            html,
            letter_spacing_fallback=1.25,
        )

        self.assertEqual(restored.toPlainText(), 'safe text')
        self.assertEqual(
            letter_spacing_value(_format_at(restored, 0)),
            1.25,
        )

    def test_selection_and_insertion_format_are_independent(self):
        document = QTextDocument()
        document.setPlainText('ABC')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_emphasis(cursor, 'filled dot', 'over right')

        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        apply_emphasis(cursor, 'open circle', 'under left')
        cursor.insertText('D')

        self.assertEqual(
            emphasis_values(_format_at(document, 0))[0], 'none'
        )
        self.assertEqual(
            emphasis_values(_format_at(document, 1))[0], 'filled dot'
        )
        self.assertEqual(
            emphasis_values(_format_at(document, 2))[0], 'none'
        )
        self.assertEqual(
            emphasis_values(_format_at(document, 3)),
            ('open circle', 'under left'),
        )

    def test_letter_spacing_selection_and_insertion_are_independent(self):
        document = QTextDocument()
        document.setPlainText('ABC')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_letter_spacing(cursor, 1.4, vertical=False)

        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        apply_letter_spacing(cursor, 0.8, vertical=False)
        cursor.insertText('D')

        self.assertEqual(letter_spacing_value(_format_at(document, 0)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(document, 1)), 1.4)
        self.assertEqual(letter_spacing_value(_format_at(document, 2)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(document, 3)), 0.8)

    def test_letter_spacing_inline_html_round_trip(self):
        source = QTextDocument()
        source.setPlainText('A𠮷&B\nCD')
        cursor = QTextCursor(source)
        cursor.setPosition(1)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        apply_letter_spacing(cursor, 1.35, vertical=False)

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)
        restored_with_conflicting_fallback = QTextDocument()
        load_rich_text_html(
            restored_with_conflicting_fallback,
            html,
            letter_spacing_fallback=2.0,
        )
        legacy_reader = QTextDocument()
        legacy_reader.setHtml(html)

        self.assertIn('style="letter-spacing: 0.35em;"', html)
        self.assertIn(
            f'{LETTER_SPACING_ATTRIBUTE}="1.35"',
            html,
        )
        self.assertEqual(restored.toPlainText(), 'A𠮷&B\nCD')
        self.assertEqual(letter_spacing_value(_format_at(restored, 0)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(restored, 1, 2)), 1.35)
        self.assertEqual(letter_spacing_value(_format_at(restored, 3)), 1.35)
        self.assertEqual(letter_spacing_value(_format_at(restored, 4)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(restored, 6)), 1.0)
        self.assertEqual(
            letter_spacing_value(
                _format_at(restored_with_conflicting_fallback, 0)
            ),
            1.0,
        )
        self.assertEqual(
            letter_spacing_value(
                _format_at(restored_with_conflicting_fallback, 1)
            ),
            1.35,
        )
        self.assertEqual(legacy_reader.toPlainText(), 'A𠮷&B\nCD')
        self.assertEqual(
            _format_at(legacy_reader, 1, 2).font().letterSpacing(),
            135.0,
        )

    def test_old_item_spacing_migrates_to_inline_html(self):
        source = QTextDocument()
        source.setPlainText('legacy')
        old_html = source.toHtml()

        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                block = TextBlock([0, 0, 300, 300])
                block._bounding_rect = [0, 0, 300, 300]
                block.translation = 'legacy'
                block.rich_text = old_html
                block.fontformat.vertical = vertical
                block.fontformat.letter_spacing = 1.35
                item = TextBlkItem(block, 0)

                for position in range(len('legacy')):
                    self.assertEqual(
                        letter_spacing_value(
                            _format_at(item.document(), position)
                        ),
                        1.35,
                    )
                migrated_html = item.toHtml()
                self.assertIn('style="letter-spacing: 0.35em;"', migrated_html)
                self.assertIn(
                    f'{LETTER_SPACING_ATTRIBUTE}="1.35"',
                    migrated_html,
                )
                restored = QTextDocument()
                load_rich_text_html(restored, migrated_html)
                self.assertEqual(
                    letter_spacing_value(_format_at(restored, 0)),
                    1.35,
                )

    def test_item_letter_spacing_uses_selection_then_insertion_format(self):
        item = self._make_item(False, text='ABC')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(1.5)

        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 0)),
            1.15,
        )
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            1.5,
        )
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 2)),
            1.15,
        )
        self.assertEqual(item.fontformat.letter_spacing, 1.15)

        cursor = item.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.setLetterSpacing(0.8)
        cursor = item.textCursor()
        cursor.insertText('D')
        item.setTextCursor(cursor)

        self.assertEqual(item.toPlainText(), 'ABCD')
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 3)),
            0.8,
        )
        self.assertEqual(item.fontformat.letter_spacing, 1.15)

    def test_nonediting_letter_spacing_updates_the_item_default(self):
        item = self._make_item(False, text='ABC')

        item.setLetterSpacing(1.6)

        self.assertEqual(item.fontformat.letter_spacing, 1.6)
        for position in range(3):
            self.assertEqual(
                letter_spacing_value(_format_at(item.document(), position)),
                1.6,
            )

    def test_vertical_letter_spacing_is_per_character_and_survives_switch(self):
        item = self._make_item(False, text='甲乙丙')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(2.0)

        item.setVertical(True)
        heights = [bottom - top for top, bottom in item.layout.y_offset_lst[0]]
        self.assertGreater(heights[1], heights[0])
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            2.0,
        )
        self.assertEqual(
            _format_at(item.document(), 1).font().letterSpacing(),
            100.0,
        )

        item.setVertical(False)
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            2.0,
        )
        self.assertEqual(
            _format_at(item.document(), 1).font().letterSpacing(),
            200.0,
        )

    def test_spacing_insertion_format_survives_writing_mode_switch(self):
        item = self._make_item(False, text='ABC')
        item.startEdit()
        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.setLetterSpacing(0.75)

        item.setVertical(True)
        cursor = item.textCursor()
        cursor.insertText('D')
        item.setTextCursor(cursor)

        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 3)),
            0.75,
        )
        self.assertEqual(
            _format_at(item.document(), 3).font().letterSpacing(),
            100.0,
        )

    def test_nonediting_item_applies_to_document_and_restores_cursor(self):
        item = self._make_item()
        cursor = item.textCursor()
        cursor.setPosition(2)
        item.setTextCursor(cursor)

        item.setEmphasis('filled circle', 'over right')

        self.assertEqual(item.textCursor().position(), 2)
        self.assertFalse(item.textCursor().hasSelection())
        self.assertEqual(
            emphasis_values(_format_at(item.document(), 0, 2)),
            ('filled circle', 'over right'),
        )
        self.assertEqual(
            emphasis_values(_format_at(item.document(), 2, 1)),
            ('filled circle', 'over right'),
        )

    def test_custom_clipboard_round_trip_preserves_annotations(self):
        source = QTextDocument()
        source.setPlainText('copy me')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        apply_emphasis(cursor, 'open sesame', 'over right')
        apply_text_combine_upright(cursor, True)
        apply_letter_spacing(cursor, 1.4, vertical=False)

        mime = create_rich_text_mime(cursor)
        target = QTextDocument()
        inserted = insert_rich_text_mime(QTextCursor(target), mime)

        self.assertTrue(inserted)
        self.assertIn(LETTER_SPACING_ATTRIBUTE, mime.html())
        self.assertIn('text-emphasis-style: open sesame', mime.html())
        self.assertIn('text-combine-upright: all', mime.html())
        self.assertEqual(target.toPlainText(), 'copy')
        self.assertEqual(
            emphasis_values(_format_at(target, 0, 4)),
            ('open sesame', 'over right'),
        )
        self.assertEqual(
            letter_spacing_value(_format_at(target, 0, 4)),
            1.4,
        )
        self.assertEqual(
            text_combine_upright_values(_format_at(target, 0, 4))[0],
            'all',
        )

    def test_text_combine_inline_round_trip_keeps_qt_html_readable(self):
        source = QTextDocument()
        source.setPlainText('A12B')
        cursor = QTextCursor(source)
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        char_format.setFontItalic(True)
        char_format.setForeground(QColor('#2070c0'))
        cursor.mergeCharFormat(char_format)
        apply_text_combine_upright(cursor, True)
        _value, source_group_id = text_combine_upright_values(
            _format_at(source, 1, 2)
        )

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)
        legacy_reader = QTextDocument()
        legacy_reader.setHtml(html)

        self.assertIn('text-combine-upright: all', html)
        self.assertIn(TEXT_COMBINE_ID_ATTRIBUTE, html)
        self.assertEqual(restored.toPlainText(), 'A12B')
        self.assertEqual(legacy_reader.toPlainText(), 'A12B')
        self.assertEqual(
            text_combine_upright_values(_format_at(legacy_reader, 1, 2))[0],
            'none',
        )
        value, group_id = text_combine_upright_values(
            _format_at(restored, 1, 2)
        )
        self.assertEqual(value, 'all')
        self.assertEqual(group_id, source_group_id)
        restored_format = _format_at(restored, 1, 2)
        self.assertTrue(restored_format.font().bold())
        self.assertTrue(restored_format.font().italic())
        self.assertEqual(
            restored_format.foreground().color(), QColor('#2070c0')
        )

    def test_text_combine_selection_and_insertion_format_group_runs(self):
        document = QTextDocument()
        document.setPlainText('ABC')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_text_combine_upright(cursor, True)

        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        apply_text_combine_upright(cursor, True)
        cursor.insertText('1')
        cursor.insertText('2')
        apply_text_combine_upright(cursor, False)
        cursor.insertText('3')

        ranges = text_combine_upright_ranges(document.firstBlock())
        self.assertEqual([(start, length) for start, length, _id in ranges], [
            (1, 1),
            (3, 2),
        ])
        self.assertNotEqual(ranges[0][2], ranges[1][2])
        self.assertEqual(
            text_combine_upright_values(_format_at(document, 5))[0],
            'none',
        )

    def test_adjacent_text_combine_runs_and_pastes_keep_boundaries(self):
        source = QTextDocument()
        source.setPlainText('1234')
        cursor = QTextCursor(source)
        for start, end in ((0, 2), (2, 4)):
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            apply_text_combine_upright(cursor, True)

        source_ranges = text_combine_upright_ranges(source.firstBlock())
        self.assertEqual([length for _start, length, _id in source_ranges], [2, 2])
        self.assertNotEqual(source_ranges[0][2], source_ranges[1][2])

        item = self._make_item(True, text='1234')
        item.startEdit()
        item_cursor = item.textCursor()
        for start, end in ((0, 2), (2, 4)):
            item_cursor.setPosition(start)
            item_cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            item.setTextCursor(item_cursor)
            item.setTateChuYoko(True)
        item_layout = item.document().firstBlock().layout()
        self.assertEqual(item_layout.lineCount(), 2)
        self.assertEqual(
            [item_layout.lineAt(index).textLength() for index in range(2)],
            [2, 2],
        )

        cursor.select(QTextCursor.SelectionType.Document)
        mime = create_rich_text_mime(cursor)
        target = QTextDocument()
        target_cursor = QTextCursor(target)
        self.assertTrue(insert_rich_text_mime(target_cursor, mime))
        target_cursor.movePosition(QTextCursor.MoveOperation.End)
        self.assertTrue(insert_rich_text_mime(target_cursor, mime))

        pasted_ranges = text_combine_upright_ranges(target.firstBlock())
        self.assertEqual(
            [length for _start, length, _id in pasted_ranges],
            [2, 2, 2, 2],
        )
        self.assertEqual(len({group_id for *_range, group_id in pasted_ranges}), 4)

    def test_textblock_rich_text_round_trip_uses_production_item_boundary(self):
        source = self._make_item()
        source.startEdit()
        cursor = source.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        source.setTextCursor(cursor)
        source.setEmphasis('open triangle', 'under left')
        cursor.setPosition(3)
        cursor.setPosition(7, QTextCursor.MoveMode.KeepAnchor)
        source.setTextCursor(cursor)
        source.setTateChuYoko(True)

        block = TextBlock([0, 0, 600, 300])
        block._bounding_rect = [0, 0, 600, 300]
        block.translation = source.toPlainText()
        block.rich_text = source.toHtml()
        restored = TextBlkItem(block, 1)

        self.assertEqual(restored.toPlainText(), source.toPlainText())
        self.assertEqual(
            emphasis_values(_format_at(restored.document(), 0, 2)),
            ('open triangle', 'under left'),
        )
        self.assertEqual(
            text_combine_upright_values(
                _format_at(restored.document(), 3, 4)
            )[0],
            'all',
        )

    def test_text_combine_uses_one_natural_width_vertical_cell(self):
        item = self._make_item(True, text='年12月')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.setEmphasis('filled sesame', 'over right')

        block = item.document().firstBlock()
        text_layout = block.layout()
        self.assertEqual(text_layout.lineCount(), 3)
        line = text_layout.lineAt(1)
        self.assertEqual((line.textStart(), line.textLength()), (1, 2))
        cell = item.layout.tate_chu_yoko_cell_rect(block, 1)
        self.assertIsNotNone(cell)
        natural_bounds = tate_chu_yoko_natural_bounds(line)
        self.assertGreaterEqual(cell.width(), natural_bounds.width())
        ink = tate_chu_yoko_ink_bounds(line, cell)
        self.assertTrue(cell.adjusted(-0.01, -0.01, 0.01, 0.01).contains(ink))
        _line, offset, orientation = item.layout.vertical_line_placement(
            block, 1
        )
        self.assertAlmostEqual(orientation.m11(), 1.0)
        self.assertAlmostEqual(orientation.m22(), 1.0)
        self.assertAlmostEqual(orientation.m12(), 0.0)
        self.assertAlmostEqual(orientation.m21(), 0.0)
        mark_ink = emphasis_ink_bounds(
            block,
            line,
            vertical=True,
            offset=offset,
            orientation=orientation,
        )
        self.assertTrue(
            item.boundingRect().adjusted(
                -0.01, -0.01, 0.01, 0.01
            ).contains(mark_ink)
        )

        caret = item.layout.source_cursor_rect(2)
        self.assertAlmostEqual(caret.width(), 2.0)
        self.assertGreater(caret.height(), caret.width())
        positions = [
            item.layout.hitTest(
                QPointF(x, cell.center().y()),
                Qt.HitTestAccuracy.FuzzyHit,
            )
            for x in (
                cell.left() + 0.25,
                cell.center().x(),
                cell.right() - 0.25,
            )
        ]
        self.assertEqual(positions, [1, 2, 3])

    def test_text_combine_cursor_blink_invalidates_mapped_rect(self):
        item = self._make_item(True, text='年12月')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)

        updates = []
        item.layout.update.connect(
            lambda *args: updates.append(
                QRectF(args[0]) if args else QRectF(item.boundingRect())
            )
        )
        image = QImage(
            item.boundingRect().size().toSize(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        context = QAbstractTextDocumentLayout.PaintContext()
        try:
            context.cursorPosition = 2
            item.layout.draw(painter, context)
            caret = item.layout.source_cursor_rect(2)
            updates.clear()

            context.cursorPosition = -1
            item.layout.draw(painter, context)
        finally:
            painter.end()

        self.assertTrue(any(rect.contains(caret) for rect in updates))

        supplementary = self._make_item(True, text='A𠮷1B')
        supplementary.startEdit()
        cursor = supplementary.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        supplementary.setTextCursor(cursor)
        supplementary.setTateChuYoko(True)
        supplementary_layout = supplementary.document().firstBlock().layout()
        self.assertEqual(supplementary_layout.lineCount(), 3)
        self.assertEqual(supplementary_layout.lineAt(1).textLength(), 3)

        spaced = self._make_item(True, text='年1 2月')
        spaced.startEdit()
        cursor = spaced.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        spaced.setTextCursor(cursor)
        spaced.setTateChuYoko(True)
        spaced_layout = spaced.document().firstBlock().layout()
        self.assertEqual(spaced_layout.lineCount(), 3)
        self.assertEqual(spaced_layout.lineAt(1).textLength(), 3)

        mixed_size = self._make_item(True, text='A12B')
        mixed_size.startEdit()
        cursor = mixed_size.textCursor()
        cursor.setPosition(2)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        large = QTextCharFormat()
        large.setFontPointSize(72.0)
        cursor.mergeCharFormat(large)
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        mixed_size.setTextCursor(cursor)
        mixed_size.setTateChuYoko(True)
        mixed_block = mixed_size.document().firstBlock()
        mixed_line = mixed_block.layout().lineAt(1)
        mixed_cell = mixed_size.layout.tate_chu_yoko_cell_rect(
            mixed_block, 1
        )
        mixed_ink = tate_chu_yoko_ink_bounds(mixed_line, mixed_cell)
        self.assertTrue(
            mixed_cell.adjusted(-0.01, -0.01, 0.01, 0.01).contains(
                mixed_ink
            )
        )

        partial_mark = self._make_item(True, text='12')
        partial_mark.startEdit()
        cursor = partial_mark.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        partial_mark.setTextCursor(cursor)
        partial_mark.setTateChuYoko(True)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        partial_mark.setTextCursor(cursor)
        partial_mark.setEmphasis('filled dot', 'over right')
        partial_block = partial_mark.document().firstBlock()
        partial_line, offset, orientation = (
            partial_mark.layout.vertical_line_placement(partial_block, 0)
        )
        self.assertFalse(
            emphasis_ink_bounds(
                partial_block,
                partial_line,
                vertical=True,
                offset=offset,
                orientation=orientation,
            ).isEmpty()
        )

    def test_text_combine_overhang_does_not_move_columns_or_border(self):
        item = self._make_item(
            True,
            bounds=(100, 20, 100, 90),
            text='甲12乙丙丁戊',
        )
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        self.app.processEvents()

        def line_x_positions() -> list[float]:
            text_layout = item.document().firstBlock().layout()
            return [
                text_layout.lineAt(index).x()
                for index in range(text_layout.lineCount())
            ]

        logical_rect = item.rect()
        layout_width = item.layout.max_width
        column_positions = line_x_positions()
        old_paint_width = item.boundingRect().width()
        cursor = item.textCursor()
        cursor.setPosition(2)
        cursor.insertText('3456')
        item.setTextCursor(cursor)
        self.app.processEvents()

        block = item.document().firstBlock()
        line = block.layout().lineForTextPosition(1)
        cell = item.layout.tate_chu_yoko_cell_rect(
            block, line.lineNumber()
        )
        ink = tate_chu_yoko_ink_bounds(line, cell)
        self.assertEqual(item.rect(), logical_rect)
        self.assertAlmostEqual(item.layout.max_width, layout_width)
        self.assertEqual(line_x_positions(), column_positions)
        self.assertGreater(item.boundingRect().width(), old_paint_width)
        self.assertTrue(item.boundingRect().contains(ink))
        left_hit = QPointF(cell.left() + 0.01, cell.center().y())
        right_hit = QPointF(cell.right() - 0.01, cell.center().y())
        self.assertTrue(item.shape().contains(left_hit))
        self.assertTrue(item.shape().contains(right_hit))
        self.assertEqual(
            item.layout.hitTest(
                left_hit,
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            line.textStart(),
        )
        self.assertEqual(
            item.layout.hitTest(
                right_hit,
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            line.textStart() + line.textLength(),
        )

    def test_text_combine_is_persistent_but_visually_inert_horizontally(self):
        plain = self._make_item(False, text='A12B')
        combined = self._make_item(False, text='A12B')
        combined.startEdit()
        cursor = combined.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        combined.setTextCursor(cursor)
        combined.setTateChuYoko(True)

        plain_line = plain.document().firstBlock().layout().lineAt(0)
        combined_line = combined.document().firstBlock().layout().lineAt(0)
        self.assertEqual(combined.toPlainText(), plain.toPlainText())
        self.assertEqual(combined_line.textLength(), plain_line.textLength())
        self.assertAlmostEqual(
            combined_line.naturalTextWidth(), plain_line.naturalTextWidth()
        )
        self.assertEqual(
            text_combine_upright_values(
                _format_at(combined.document(), 1, 2)
            )[0],
            'all',
        )
        combined.setVertical(True)
        vertical_layout = combined.document().firstBlock().layout()
        self.assertEqual(vertical_layout.lineCount(), 3)
        self.assertEqual(vertical_layout.lineAt(1).textLength(), 2)

    def test_wrapped_text_combine_reserves_its_own_visible_column_width(self):
        item = self._make_item(
            True,
            bounds=(0, 0, 180, 55),
            text='年年12',
        )
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(2)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontPointSize(24.0)
        cursor.mergeCharFormat(char_format)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.setEmphasis('filled sesame', 'over right')

        block = item.document().firstBlock()
        line = block.layout().lineAt(2)
        record = item.layout.per_char_records[0][2]
        cell = item.layout.tate_chu_yoko_cell_rect(block, 2)
        self.assertLess(line.x(), block.layout().lineAt(0).x())
        self.assertGreater(record['line_width'], record['text_combine_width'])
        self.assertGreaterEqual(cell.left(), item.layout.layout_left - 0.01)
        self.assertLessEqual(
            cell.right(), line.x() + record['line_width'] + 0.01
        )

    def test_text_combine_renders_with_styles_effects_and_glyph_slant(self):
        block = TextBlock([0, 0, 180, 160])
        block._bounding_rect = [0, 0, 180, 160]
        block.vertical = True
        block.translation = '年12月'
        block.fontformat.glyph_slant_angle = 12.0
        block.fontformat.stroke_width = 0.08
        block.fontformat.shadow_radius = 0.06
        block.fontformat.shadow_strength = 0.7
        block.fontformat.shadow_offset = [0.05, 0.04]
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)

        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        char_format.setFontItalic(True)
        char_format.setForeground(QColor('#df4050'))
        cursor.mergeCharFormat(char_format)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.setEmphasis('filled sesame', 'over right')
        item.endEdit(keep_focus=False)
        self.app.processEvents()

        renderer = item.geometry_controller.layout_renderer
        self.assertIsNotNone(renderer)
        ink_bounds = renderer.ink_bounds()
        self.assertFalse(ink_bounds.isEmpty())
        self.assertTrue(
            item.boundingRect().adjusted(-0.01, -0.01, 0.01, 0.01).contains(
                ink_bounds
            )
        )
        self.assertGreater(item.effect_renderer.padding(), 0.0)

        image = QImage(
            260,
            220,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        try:
            scene.render(
                painter,
                QRectF(0, 0, 260, 220),
                scene.itemsBoundingRect(),
            )
        finally:
            painter.end()
        byte_count = (
            image.sizeInBytes()
            if hasattr(image, 'sizeInBytes')
            else image.byteCount()
        )
        pixels = bytes(image.bits().asstring(byte_count))
        self.assertTrue(any(pixels[3::4]))

    def test_annotation_effect_cache_stays_at_one_x(self):
        for effect in ('stroke', 'shadow'):
            with self.subTest(effect=effect):
                block = TextBlock([0, 0, 140, 140])
                block._bounding_rect = [0, 0, 140, 140]
                block.vertical = True
                block.translation = '天天'
                block.fontformat.font_size = 48
                if effect == 'stroke':
                    block.fontformat.stroke_width = 0.2
                else:
                    block.fontformat.shadow_radius = 0.04
                    block.fontformat.shadow_strength = 0.8
                    block.fontformat.shadow_offset = [0.04, 0.04]
                item = TextBlkItem(block, 0)
                scene = QGraphicsScene()
                scene.addItem(item)

                item.startEdit()
                cursor = item.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                item.setTextCursor(cursor)
                item.setTateChuYoko(True)
                item.setEmphasis('filled dot', 'over right')
                item.endEdit(keep_focus=False)
                self.app.processEvents()

                source = scene.itemsBoundingRect()
                image = QImage(
                    max(1, round(source.width() * 4)),
                    max(1, round(source.height() * 4)),
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                try:
                    scene.render(painter, QRectF(image.rect()), source)
                finally:
                    painter.end()

                renderer = item.effect_renderer
                self.assertEqual(renderer.background_pixmap_scale, 1.0)
                self.assertEqual(
                    renderer.background_pixmap.devicePixelRatioF(), 1.0
                )
                alpha = pixmap2ndarray(
                    renderer.background_pixmap, keep_alpha=True
                )[..., 3]
                self.assertTrue(((alpha > 0) & (alpha < 255)).any())
                scene.removeItem(item)

    def test_nonediting_effect_change_invalidates_scene_cache(self):
        for effect in ('stroke width', 'shadow color'):
            with self.subTest(effect=effect):
                block = TextBlock([0, 0, 140, 140])
                block._bounding_rect = [0, 0, 140, 140]
                block.translation = 'Effect'
                if effect == 'stroke width':
                    block.fontformat.stroke_width = 0.05
                else:
                    block.fontformat.shadow_radius = 0.1
                    block.fontformat.shadow_strength = 0.8
                item = TextBlkItem(block, 0)
                item.setSelected(True)
                scene = QGraphicsScene()
                scene.addItem(item)
                self.app.processEvents()

                changed_regions = []
                scene.changed.connect(changed_regions.extend)
                old_cache_key = item.effect_renderer.background_pixmap.cacheKey()
                if effect == 'stroke width':
                    item.setStrokeWidth(0.2)
                else:
                    shadow = item.fontformat.deepcopy()
                    shadow.shadow_color = [255, 0, 0]
                    item.setShadow(shadow)
                self.app.processEvents()

                self.assertFalse(item.isEditing())
                self.assertTrue(item.isSelected())
                self.assertNotEqual(
                    item.effect_renderer.background_pixmap.cacheKey(),
                    old_cache_key,
                )
                self.assertTrue(changed_regions)
                scene.removeItem(item)

    def test_tate_chu_yoko_stroke_has_no_small_glyph_cavity(self):
        block = TextBlock([0, 0, 140, 140])
        block._bounding_rect = [0, 0, 140, 140]
        block.vertical = True
        block.translation = '!'
        block.fontformat.font_size = 48
        block.fontformat.stroke_width = 0.4
        item = TextBlkItem(block, 0)

        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.endEdit(keep_focus=False)
        self.app.processEvents()

        item.effect_renderer._repaint_neutral_background()
        alpha = pixmap2ndarray(
            item.effect_renderer.background_pixmap, keep_alpha=True
        )[..., 3]
        occupied_y, occupied_x = alpha.nonzero()
        center_x = int(round((occupied_x.min() + occupied_x.max()) / 2))
        center_column = alpha[:, center_x]
        occupied_column = center_column.nonzero()[0]

        self.assertGreater(occupied_column.size, 0)
        self.assertTrue(
            (
                center_column[
                    occupied_column[0]:occupied_column[-1] + 1
                ]
                > 0
            ).all()
        )

    def test_document_replace_keeps_annotation_attached_to_replacement(self):
        source = QTextDocument()
        source.setPlainText('foo bar')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        apply_emphasis(cursor, 'filled dot', 'over right')
        apply_text_combine_upright(cursor, True)

        edited = QTextDocument()
        load_rich_text_html(edited, to_rich_text_html(source))
        doc_replace(edited, [[0, 3]], 'longer')
        restored = QTextDocument()
        load_rich_text_html(restored, to_rich_text_html(edited))

        self.assertEqual(restored.toPlainText(), 'longer bar')
        self.assertEqual(
            emphasis_values(_format_at(restored, 0, 6)),
            ('filled dot', 'over right'),
        )
        self.assertEqual(
            emphasis_values(_format_at(restored, 6)),
            ('none', 'over right'),
        )
        self.assertEqual(
            text_combine_upright_values(_format_at(restored, 0, 6))[0],
            'all',
        )

    def test_grapheme_ranges_match_qt_utf16_positions(self):
        self.assertEqual(
            _grapheme_ranges('A𠮷e\u0301👩\u200d👩\u200d👧\u200d👦'),
            ((0, 1), (1, 3), (3, 5), (5, 16)),
        )

    def test_vertical_emphasis_keeps_supplementary_layout_records_aligned(self):
        item = self._make_item(True, text='A𠮷B')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setEmphasis('filled dot', 'over right')

        text_layout = item.document().firstBlock().layout()
        self.assertEqual(
            [
                text_layout.lineAt(index).textLength()
                for index in range(text_layout.lineCount())
            ],
            [1, 2, 1],
        )
        self.assertEqual(
            len(item.layout.line_spaces_lst[0]),
            text_layout.lineCount(),
        )
        self.assertEqual(
            len(item.layout.y_offset_lst[0]),
            text_layout.lineCount(),
        )

    def test_emphasis_adds_css_like_line_and_column_leading(self):
        horizontal_plain = self._make_item(False)
        horizontal_marked = self._make_item(False)
        vertical_plain = self._make_item(True)
        vertical_marked = self._make_item(True)

        for item in (horizontal_marked, vertical_marked):
            cursor = item.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            item.setTextCursor(cursor)
            item.startEdit()
            item.setEmphasis('filled sesame', 'over right')

        plain_y = horizontal_plain.document().firstBlock().layout().lineAt(0).y()
        marked_y = horizontal_marked.document().firstBlock().layout().lineAt(0).y()
        self.assertGreater(marked_y, plain_y)
        self.assertGreater(
            vertical_marked.layout.shrink_width,
            vertical_plain.layout.shrink_width,
        )

    def test_effect_outline_does_not_reflow_emphasis_layout(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._make_item(vertical, text='A強調B')
                item.startEdit()
                cursor = item.textCursor()
                cursor.setPosition(1)
                cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
                item.setTextCursor(cursor)
                item.setEmphasis('filled dot', 'over right')

                text_layout = item.document().firstBlock().layout()
                before = tuple(
                    text_layout.lineAt(index).position()
                    for index in range(text_layout.lineCount())
                )

                # The neutral effect path adds this outline to its temporary
                # clone. It must affect ink only, never line placement.
                cursor = QTextCursor(item.document())
                cursor.select(QTextCursor.SelectionType.Document)
                outline = QTextCharFormat()
                outline.setTextOutline(QPen(QColor('black'), 12.0))
                cursor.mergeCharFormat(outline)

                text_layout = item.document().firstBlock().layout()
                after = tuple(
                    text_layout.lineAt(index).position()
                    for index in range(text_layout.lineCount())
                )
                self.assertEqual(len(after), len(before))
                for actual, expected in zip(after, before):
                    self.assertAlmostEqual(actual.x(), expected.x())
                    self.assertAlmostEqual(actual.y(), expected.y())

    def test_wrapped_vertical_columns_keep_mark_space_and_render(self):
        text = '強調文字列強調文字列'
        plain = self._make_item(True, bounds=(0, 0, 180, 90), text=text)
        marked = self._make_item(True, bounds=(0, 0, 180, 90), text=text)
        marked.startEdit()
        cursor = marked.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        marked.setTextCursor(cursor)
        marked.setEmphasis('filled sesame', 'over right')

        def column_positions(item: TextBlkItem):
            layout = item.document().firstBlock().layout()
            return sorted(
                {
                    round(layout.lineAt(index).x(), 3)
                    for index in range(layout.lineCount())
                }
            )

        plain_columns = column_positions(plain)
        marked_columns = column_positions(marked)
        self.assertGreaterEqual(len(marked_columns), 2)
        self.assertGreater(
            marked_columns[1] - marked_columns[0],
            plain_columns[1] - plain_columns[0],
        )

        image = QImage(400, 200, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        try:
            marked.document().drawContents(painter)
        finally:
            painter.end()
        self.assertFalse(image.isNull())

    def test_document_undo_redo_restores_emphasis(self):
        item = self._make_item(False)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setEmphasis('filled dot', 'over right')

        item.document().undo()
        self.assertEqual(emphasis_values(_format_at(item.document(), 0))[0], 'none')
        item.document().redo()
        self.assertEqual(
            emphasis_values(_format_at(item.document(), 0))[0],
            'filled dot',
        )

    def test_document_undo_redo_restores_text_combine(self):
        item = self._make_item(True, text='12')
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)

        item.document().undo()
        self.assertEqual(
            text_combine_upright_values(_format_at(item.document(), 0))[0],
            'none',
        )
        item.document().redo()
        self.assertEqual(
            text_combine_upright_values(_format_at(item.document(), 0))[0],
            'all',
        )

    def test_document_undo_redo_restores_letter_spacing(self):
        item = self._make_item(False, text='ABC')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(1.8)

        item.document().undo()
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            1.15,
        )
        item.document().redo()
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            1.8,
        )

    def test_panel_exposes_stable_values_and_emits_one_edit(self):
        group = TextEmphasisGroup()
        edits = []
        group.emphasis_changed.connect(
            lambda style, position: edits.append((style, position))
        )
        group.set_values('open circle', 'under left')

        group._on_value_changed(group.style_combobox.currentIndex())

        self.assertEqual(edits, [('open circle', 'under left')])

    def test_tate_chu_yoko_panel_exposes_one_boolean_edit(self):
        group = TateChuYokoGroup()
        edits = []
        group.enabled_changed.connect(edits.append)

        group.set_enabled(True)
        group.enable_checker.checkStateChanged.emit(True)

        self.assertTrue(group.enable_checker.isChecked())
        self.assertEqual(edits, [True])


if __name__ == '__main__':
    unittest.main()

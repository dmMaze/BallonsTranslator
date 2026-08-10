import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.misc import doc_replace
from ballontranslator.ui.text_engine.annotations import (
    RICH_TEXT_METADATA_NAME,
    apply_emphasis,
    create_rich_text_mime,
    emphasis_values,
    insert_rich_text_mime,
    load_rich_text_html,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.formatting.advanced import TextEmphasisGroup
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.indexing import _grapheme_ranges
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

    def test_old_qt_html_loads_without_metadata_or_format_loss(self):
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
        self.assertNotIn(RICH_TEXT_METADATA_NAME, to_rich_text_html(restored))

    def test_metadata_round_trip_uses_qt_positions_and_keeps_fragment_style(self):
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

        self.assertIn(RICH_TEXT_METADATA_NAME, html)
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

    def test_invalid_metadata_drops_only_the_annotation(self):
        source = QTextDocument()
        source.setPlainText('safe text')
        html = source.toHtml().replace(
            '</head>',
            '<meta name="ballontranslator-rich-text" '
            'content="{&quot;version&quot;:1,&quot;annotations&quot;:['
            '{&quot;kind&quot;:&quot;emphasis&quot;,&quot;start&quot;:999,'
            '&quot;length&quot;:1,&quot;style&quot;:&quot;filled dot&quot;,'
            '&quot;position&quot;:&quot;over right&quot;}]}" /></head>',
        )
        restored = QTextDocument()

        load_rich_text_html(restored, html)

        self.assertEqual(restored.toPlainText(), 'safe text')
        self.assertEqual(
            emphasis_values(_format_at(restored, 0)),
            ('none', 'over right'),
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

        mime = create_rich_text_mime(cursor)
        target = QTextDocument()
        inserted = insert_rich_text_mime(QTextCursor(target), mime)

        self.assertTrue(inserted)
        self.assertEqual(target.toPlainText(), 'copy')
        self.assertEqual(
            emphasis_values(_format_at(target, 0, 4)),
            ('open sesame', 'over right'),
        )

    def test_textblock_rich_text_round_trip_uses_production_item_boundary(self):
        source = self._make_item()
        source.startEdit()
        cursor = source.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        source.setTextCursor(cursor)
        source.setEmphasis('open triangle', 'under left')

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

    def test_document_replace_keeps_annotation_attached_to_replacement(self):
        source = QTextDocument()
        source.setPlainText('foo bar')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        apply_emphasis(cursor, 'filled dot', 'over right')

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

    def test_grapheme_ranges_match_qt_utf16_positions(self):
        self.assertEqual(
            _grapheme_ranges('A𠮷e\u0301👩\u200d👩\u200d👧\u200d👦'),
            ((0, 1), (1, 3), (3, 5), (5, 16)),
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

    def test_panel_exposes_stable_values_and_emits_one_edit(self):
        group = TextEmphasisGroup()
        edits = []
        group.emphasis_changed.connect(
            lambda style, position: edits.append((style, position))
        )
        group.set_values('open circle', 'under left')

        group._on_value_changed(group.style_combobox.currentIndex())

        self.assertEqual(edits, [('open circle', 'under left')])


if __name__ == '__main__':
    unittest.main()

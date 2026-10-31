import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtGui import QTextCursor, QTextDocument
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.annotations import (
    apply_text_combine_upright,
    prepare_tate_chu_yoko_insertion,
    text_combine_upright_ranges,
    text_combine_upright_values,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TextBlock


OLD = '到底是哪方面的熟人呢！'
PASTED = '全新的文本内容全新的文本内容'


class TateChuYokoInsertionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(text: str = OLD) -> TextBlkItem:
        bounds = (0, 0, 200, 400)
        block = TextBlock(list(bounds))
        block._bounding_rect = list(bounds)
        block.vertical = True
        block.translation = text
        item = TextBlkItem(block, 0)
        item.fontformat.vertical = True
        return item

    @staticmethod
    def _combine_tail(item: TextBlkItem, length: int = 2) -> None:
        """Combine the last characters, as automatic Tate-chu-yoko does."""
        document = item.document()
        cursor = QTextCursor(document)
        text_length = len(document.toPlainText())
        cursor.setPosition(text_length - length)
        cursor.setPosition(text_length, QTextCursor.MoveMode.KeepAnchor)
        apply_text_combine_upright(cursor, True)

    @staticmethod
    def _select_all(item: TextBlkItem) -> None:
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)

    def test_select_all_paste_does_not_combine_the_whole_paragraph(self):
        item = self._make_item()
        self._combine_tail(item)
        self.assertTrue(text_combine_upright_ranges(item.document().firstBlock()))

        self._select_all(item)
        item.insert_plain_text_at_cursor(PASTED)

        block = item.document().firstBlock()
        self.assertEqual(item.document().toPlainText(), PASTED)
        self.assertEqual(text_combine_upright_ranges(block), ())
        self.assertNotIn('text-combine-upright', to_rich_text_html(item.document()))
        item.deleteLater()

    def test_long_paste_at_group_end_keeps_the_existing_group(self):
        item = self._make_item()
        self._combine_tail(item)
        before = text_combine_upright_ranges(item.document().firstBlock())

        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.insert_plain_text_at_cursor(PASTED)

        block = item.document().firstBlock()
        self.assertEqual(text_combine_upright_ranges(block), before)
        self.assertEqual(block.text(), OLD + PASTED)
        item.deleteLater()

    def test_single_character_insertions_still_build_a_group(self):
        item = self._make_item(text='')
        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)

        item.insert_plain_text_at_cursor('1')
        item.insert_plain_text_at_cursor('2')

        ranges = text_combine_upright_ranges(item.document().firstBlock())
        self.assertEqual(item.document().toPlainText(), '12')
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0][:2], (0, 2))
        item.deleteLater()

    def test_selection_replacement_with_one_character_is_plain(self):
        item = self._make_item()
        self._combine_tail(item)

        self._select_all(item)
        item.insert_plain_text_at_cursor('X')

        block = item.document().firstBlock()
        self.assertEqual(block.text(), 'X')
        self.assertEqual(text_combine_upright_ranges(block), ())
        item.deleteLater()

    def test_guard_ignores_documents_without_a_group(self):
        document = QTextDocument('plain')
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        prepare_tate_chu_yoko_insertion(cursor, 'replacement')
        cursor.insertText('replacement')
        self.assertEqual(document.toPlainText(), 'replacement')
        self.assertEqual(
            text_combine_upright_values(cursor.charFormat())[0],
            'none',
        )


if __name__ == '__main__':
    unittest.main()

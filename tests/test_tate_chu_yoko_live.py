import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtGui import QTextCursor, QTextDocument
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.annotations import (
    apply_text_combine_upright,
    create_rich_text_mime,
    text_combine_upright_ranges,
)
from ballontranslator.ui.text_engine.editing.commands import propagate_user_edit
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.config import AutoTateChuYokoConfig, pcfg
from ballontranslator.utils.textblock import TextBlock


def _settings(**kwargs) -> AutoTateChuYokoConfig:
    values = dict(
        enabled=True,
        max_length=4,
        include_numbers=True,
        include_letters=False,
        additional_chars='！？',
    )
    values.update(kwargs)
    return AutoTateChuYokoConfig(**values)


class LiveAutoTateChuYokoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        patcher = patch.object(pcfg, 'auto_tate_chu_yoko', _settings())
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _make_item(text: str = '', vertical: bool = True) -> TextBlkItem:
        bounds = (0, 0, 200, 400)
        block = TextBlock(list(bounds))
        block._bounding_rect = list(bounds)
        block.vertical = vertical
        block.translation = text
        item = TextBlkItem(block, 0)
        item.fontformat.vertical = vertical
        return item

    @staticmethod
    def _caret_to_end(item: TextBlkItem) -> None:
        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)

    def test_paste_combines_ascii_punctuation_immediately(self):
        item = self._make_item()
        self._caret_to_end(item)
        item.insert_plain_text_at_cursor('干什么…!?')

        ranges = text_combine_upright_ranges(item.document().firstBlock())
        self.assertEqual(
            [(start, length) for start, length, _id in ranges],
            [(4, 2)],
        )
        item.deleteLater()

    def test_typing_digits_extends_one_group(self):
        item = self._make_item()
        self._caret_to_end(item)
        item.insert_plain_text_at_cursor('1')
        first = text_combine_upright_ranges(item.document().firstBlock())
        item.insert_plain_text_at_cursor('2')
        second = text_combine_upright_ranges(item.document().firstBlock())

        self.assertEqual([entry[:2] for entry in first], [(0, 1)])
        self.assertEqual([entry[:2] for entry in second], [(0, 2)])
        self.assertEqual(first[0][2], second[0][2])
        item.deleteLater()

    def test_run_growing_past_the_limit_is_released(self):
        with patch.object(pcfg, 'auto_tate_chu_yoko', _settings(max_length=3)):
            item = self._make_item()
            self._caret_to_end(item)
            for character in '123':
                item.insert_plain_text_at_cursor(character)
            combined = text_combine_upright_ranges(item.document().firstBlock())
            item.insert_plain_text_at_cursor('4')
            released = text_combine_upright_ranges(item.document().firstBlock())

        self.assertEqual([entry[:2] for entry in combined], [(0, 3)])
        self.assertEqual(released, ())
        item.deleteLater()

    def test_group_of_unconfigured_characters_survives(self):
        item = self._make_item(text='汉字')
        document = item.document()
        cursor = QTextCursor(document)
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_text_combine_upright(cursor, True)
        manual_id = text_combine_upright_ranges(document.firstBlock())[0][2]

        self._caret_to_end(item)
        item.insert_plain_text_at_cursor('12')

        ranges = text_combine_upright_ranges(document.firstBlock())
        self.assertEqual([entry[:2] for entry in ranges], [(0, 2), (2, 2)])
        self.assertEqual(ranges[0][2], manual_id)
        item.deleteLater()

    def test_disabled_setting_leaves_inserted_text_plain(self):
        with patch.object(pcfg, 'auto_tate_chu_yoko', _settings(enabled=False)):
            item = self._make_item()
            self._caret_to_end(item)
            item.insert_plain_text_at_cursor('12')
            ranges = text_combine_upright_ranges(item.document().firstBlock())

        self.assertEqual(ranges, ())
        item.deleteLater()

    def test_horizontal_item_leaves_inserted_text_plain(self):
        item = self._make_item(vertical=False)
        self._caret_to_end(item)
        item.insert_plain_text_at_cursor('12')
        ranges = text_combine_upright_ranges(item.document().firstBlock())

        self.assertEqual(ranges, ())
        item.deleteLater()

    def test_multi_paragraph_paste_combines_every_run(self):
        item = self._make_item()
        self._caret_to_end(item)
        item.insert_plain_text_at_cursor('哈哈\n真是壮观啊!!!\n活该!!!')

        blocks = []
        block = item.document().firstBlock()
        while block.isValid():
            blocks.append(
                [entry[:2] for entry in text_combine_upright_ranges(block)]
            )
            block = block.next()

        self.assertEqual(blocks, [[], [(5, 3)], [(2, 3)]])
        item.deleteLater()

    def test_rich_text_paste_combines_the_run(self):
        source = QTextDocument('为…什么!?')
        source_cursor = QTextCursor(source)
        source_cursor.select(QTextCursor.SelectionType.Document)
        mime = create_rich_text_mime(source_cursor)

        item = self._make_item(text='旧的文本')
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        self.assertTrue(item.insert_from_mime_data(mime))

        ranges = text_combine_upright_ranges(item.document().firstBlock())
        self.assertEqual([entry[:2] for entry in ranges], [(4, 2)])
        item.deleteLater()

    def test_paired_editor_replay_combines_the_run(self):
        item = self._make_item()
        self._caret_to_end(item)
        propagate_user_edit(item, item.document().characterCount() - 1, 0, '!?', False)

        ranges = text_combine_upright_ranges(item.document().firstBlock())
        self.assertEqual([entry[:2] for entry in ranges], [(0, 2)])
        item.deleteLater()


if __name__ == '__main__':
    unittest.main()

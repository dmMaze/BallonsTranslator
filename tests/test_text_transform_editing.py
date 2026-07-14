import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QInputMethodEvent, QPolygonF, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.textedit_commands import (
    propagate_user_edit,
    utf16_code_unit_length,
)
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


_APP = QApplication.instance() or QApplication([])


def make_item(text='A\U0001f600e\u0301\ufe0f\n\u6f22\u5b57', *, vertical=False):
    block = TextBlock(
        xyxy=[20, 30, 240, 150],
        _bounding_rect=[20, 30, 220, 120],
        translation=text,
        fontformat=FontFormat(
            vertical=vertical,
            horizontal_scale=1.6,
            vertical_scale=0.7,
            slant_angle=13.0,
        ),
    )
    item = TextBlkItem(block)
    scene = QGraphicsScene()
    scene.addItem(item)
    return item, block, scene


def rect_polygon(rect):
    return QPolygonF(
        [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
    )


class TextTransformEditingTest(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected, places=5):
        self.assertAlmostEqual(actual.x(), expected.x(), places=places)
        self.assertAlmostEqual(actual.y(), expected.y(), places=places)

    def test_preview_and_commit_preserve_cursor_html_revision_and_layout(self):
        item, block, _scene = make_item()
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(7)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        before_cursor = (cursor.position(), cursor.anchor())
        before_html = item.document().toHtml()
        before_revision = item.document().revision()
        before_size = item.documentSize()
        before_layout = item.document().documentLayout()
        before_rect = item.absBoundingRect(qrect=True)
        before_pos = QPointF(item.pos())

        self.assertTrue(item.set_text_transform(2.0, 0.5, -8.0, preview=True))
        self.assertEqual(block.fontformat.text_transform, (1.6, 0.7, 13.0))
        self.assertTrue(item.set_text_transform(2.0, 0.5, -8.0))

        cursor = item.textCursor()
        self.assertEqual((cursor.position(), cursor.anchor()), before_cursor)
        self.assertEqual(item.document().toHtml(), before_html)
        self.assertEqual(item.document().revision(), before_revision)
        self.assertEqual(item.documentSize(), before_size)
        self.assertIs(item.document().documentLayout(), before_layout)
        self.assertEqual(item.absBoundingRect(qrect=True), before_rect)
        self.assertEqual(item.pos(), before_pos)

    def test_direction_switch_is_layout_only_and_preserves_selection_direction(self):
        item, block, _scene = make_item()
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(7)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        before_cursor = (cursor.position(), cursor.anchor())
        before_html = item.document().toHtml()
        before_revision = item.document().revision()
        before_transform = item.transform()
        before_tuple = block.fontformat.text_transform

        item.setVertical(True)
        self.assertTrue(block.fontformat.vertical)
        self.assertEqual((item.textCursor().position(), item.textCursor().anchor()), before_cursor)
        self.assertEqual(block.fontformat.text_transform, before_tuple)
        self.assertEqual(item.transform(), before_transform)
        self.assertEqual(item.document().toHtml(), before_html)
        self.assertEqual(item.document().revision(), before_revision)

        item.setVertical(False)
        self.assertFalse(block.fontformat.vertical)
        self.assertEqual((item.textCursor().position(), item.textCursor().anchor()), before_cursor)
        self.assertEqual(block.fontformat.text_transform, before_tuple)
        self.assertEqual(item.document().toHtml(), before_html)
        self.assertEqual(item.document().revision(), before_revision)

    def test_cursor_local_scene_round_trip_uses_item_boundary(self):
        item, _block, _scene = make_item('Latin \u6f22\u5b57')
        item.setRotation(21.0)
        item.startEdit()
        cursor = item.textCursor()
        original_position = 4
        cursor.setPosition(original_position)
        item.setTextCursor(cursor)

        query = Qt.InputMethodQuery.ImCursorRectangle
        local_rect = item.inputMethodQuery(query)
        self.assertIsInstance(local_rect, QRectF)
        local_point = local_rect.center()
        scene_point = item.mapToScene(local_point)
        self.assertPointAlmostEqual(item.mapFromScene(scene_point), local_point)

        scene_polygon = QPolygonF(
            [item.mapToScene(point) for point in rect_polygon(local_rect)]
        )
        self.assertEqual(len(scene_polygon), 4)
        self.assertGreater(scene_polygon.boundingRect().height(), 0)

        round_trip_local = item.mapFromScene(scene_point)
        hit = item.layout.hitTest(round_trip_local, None)
        self.assertEqual(hit, original_position)
        self.assertEqual(item.textCursor().position(), original_position)

    def test_ime_preedit_commit_and_cursor_query_survive_transform_update(self):
        item, _block, _scene = make_item('IME ')
        item.startEdit()
        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)

        query = Qt.InputMethodQuery.ImCursorRectangle
        preedit = QInputMethodEvent('\ud55c', [])
        item.inputMethodEvent(preedit)
        self.assertTrue(item.pre_editing)
        self.assertEqual(item.toPlainText(), 'IME ')

        logical_before = item.inputMethodQuery(query)
        scene_before = item.mapToScene(logical_before.center())

        item.set_text_transform(0.5, 2.0, -17.0, preview=True)
        logical_after = item.inputMethodQuery(query)
        scene_after = item.mapToScene(logical_after.center())
        self.assertEqual(logical_after, logical_before)
        self.assertNotEqual(scene_after, scene_before)

        commit = QInputMethodEvent('', [])
        commit.setCommitString('\ud55c')
        item.inputMethodEvent(commit)
        self.assertFalse(item.pre_editing)
        self.assertEqual(item.toPlainText(), 'IME \ud55c')
        self.assertEqual(
            item.textCursor().position(), utf16_code_unit_length('IME \ud55c')
        )

    def test_utf16_edit_propagation_handles_emoji_variation_and_combining(self):
        inserted = '\U0001f600\ufe0fe\u0301'
        self.assertEqual(utf16_code_unit_length(inserted), 5)

        source, _source_block, _source_scene = make_item('A' + inserted + 'B')
        target, _target_block, _target_scene = make_item('AB')
        propagate_user_edit(source, target, 1, inserted)
        self.assertEqual(target.toPlainText(), 'A' + inserted + 'B')


if __name__ == '__main__':
    unittest.main()

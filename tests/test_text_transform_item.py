import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene
try:
    from qtpy.QtWidgets import QUndoStack
except ImportError:
    from qtpy.QtGui import QUndoStack

from ballontranslator.ui.scenetext_manager import (
    PasteBlkItemsCommand,
    SceneTextManager,
)
from ballontranslator.ui.text_transform import text_transform_matrix
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


_APP = QApplication.instance() or QApplication([])


class TrackingTextBlkItem(TextBlkItem):
    def __init__(self, *args, **kwargs):
        self.transform_calls = 0
        self.micro_focus_calls = 0
        super().__init__(*args, **kwargs)

    def setTransform(self, matrix, combine=False):
        self.transform_calls += 1
        return super().setTransform(matrix, combine)

    def updateMicroFocus(self):
        self.micro_focus_calls += 1
        return super().updateMicroFocus()


def make_item(transform=(1.5, 0.75, 12.0)):
    fontformat = FontFormat(
        horizontal_scale=transform[0],
        vertical_scale=transform[1],
        slant_angle=transform[2],
    )
    block = TextBlock(
        xyxy=[10, 20, 110, 70],
        _bounding_rect=[10, 20, 100, 50],
        translation='x',
        fontformat=fontformat,
    )
    return TrackingTextBlkItem(block), block


class _FakeClipboard:
    def __init__(self):
        self.text = None

    def setText(self, text, _mode):
        self.text = text


class _FakePairWidget:
    def __init__(self, idx):
        self.idx = idx

    def updateIndex(self, idx):
        self.idx = idx


class _FakeFormatPanel:
    def __init__(self):
        self.calls = []

    def set_textblk_item(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _CopyPasteCanvas:
    def __init__(self):
        self.scene = QGraphicsScene()
        self.clipboard_blks = []
        self.scale_factor = 2.0
        self.block_selection_signal = False
        self.selected = []
        self.undo_stack = QUndoStack()
        self.last_command = None

    def selected_text_items(self):
        return list(self.selected)

    def text_change_unsaved(self):
        return False

    def clearSelection(self):
        self.scene.clearSelection()
        self.selected = []

    def push_undo_command(self, command):
        self.last_command = command
        self.undo_stack.push(command)

    def removeItem(self, item):
        self.scene.removeItem(item)


class _CopyPasteManager:
    """Small collaborator shell for the real manager copy/paste methods."""

    def __init__(self, source_item):
        self.canvas = _CopyPasteCanvas()
        self.app_clipborad = _FakeClipboard()
        self.formatpanel = _FakeFormatPanel()
        self.textblk_item_list = [source_item]
        self.pairwidget_list = [_FakePairWidget(0)]
        self.selection_syncs = 0
        self.canvas.scene.addItem(source_item)
        self.canvas.selected = [source_item]

    def addTextBlock(self, block):
        item = TextBlkItem(block, len(self.textblk_item_list))
        pair = _FakePairWidget(len(self.pairwidget_list))
        self.textblk_item_list.append(item)
        self.pairwidget_list.append(pair)
        self.canvas.scene.addItem(item)
        return item

    def _reindex(self):
        for idx, (item, pair) in enumerate(
            zip(self.textblk_item_list, self.pairwidget_list)
        ):
            item.idx = idx
            pair.updateIndex(idx)

    def deleteTextblkItemList(self, items, pairs):
        for item, pair in zip(items, pairs):
            self.canvas.removeItem(item)
            self.textblk_item_list.remove(item)
            self.pairwidget_list.remove(pair)
        self._reindex()

    def recoverTextblkItemList(self, items, pairs):
        for item, pair in zip(items, pairs):
            self.textblk_item_list.insert(item.idx, item)
            self.pairwidget_list.insert(pair.idx, pair)
            self.canvas.scene.addItem(item)
        self._reindex()

    def on_incanvas_selection_changed(self):
        self.selection_syncs += 1


class TextTransformItemTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual.x(), expected.x())
        self.assertAlmostEqual(actual.y(), expected.y())

    def assertRectAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual.x(), expected.x())
        self.assertAlmostEqual(actual.y(), expected.y())
        self.assertAlmostEqual(actual.width(), expected.width())
        self.assertAlmostEqual(actual.height(), expected.height())

    def test_load_uses_unpadded_local_pivot_and_explicit_visual_geometry(self):
        item, block = make_item()
        scene = QGraphicsScene()
        scene.addItem(item)

        logical = item.logical_unpadded_rect()
        self.assertRectAlmostEqual(logical, QRectF(0, 0, 100, 50))
        self.assertRectAlmostEqual(
            item.absBoundingRect(qrect=True), QRectF(10, 20, 100, 50)
        )
        self.assertPointAlmostEqual(item.transformOriginPoint(), logical.center())
        self.assertEqual(
            item.transform(),
            text_transform_matrix(*block.fontformat.text_transform, logical.center()),
        )

        polygon = item.visual_polygon_in_scene()
        self.assertEqual(len(polygon), 4)
        self.assertRectAlmostEqual(item.visual_bounds_in_scene(), polygon.boundingRect())

        scene_pivot = item.mapToScene(item.transformOriginPoint())
        item.setPadding(10)
        self.assertRectAlmostEqual(
            item.logical_unpadded_rect(), QRectF(10, 10, 100, 50)
        )
        self.assertRectAlmostEqual(
            item.absBoundingRect(qrect=True), QRectF(10, 20, 100, 50)
        )
        self.assertPointAlmostEqual(
            item.mapToScene(item.transformOriginPoint()), scene_pivot
        )
        for actual, expected in zip(item.visual_polygon_in_scene(), polygon):
            self.assertPointAlmostEqual(actual, expected)

    def test_preview_commit_and_noop_do_not_touch_document_geometry(self):
        item, block = make_item()
        original_tuple = block.fontformat.text_transform
        original_html = item.document().toHtml()
        original_revision = item.document().revision()
        original_size = item.documentSize()

        item.transform_calls = 0
        self.assertTrue(item.set_text_transform(horizontal_scale=2.0, preview=True))
        self.assertEqual(block.fontformat.text_transform, original_tuple)
        self.assertEqual(
            item.transform(),
            text_transform_matrix(2.0, 0.75, 12.0, item.logical_unpadded_rect().center()),
        )

        calls_after_preview = item.transform_calls
        self.assertFalse(item.set_text_transform(horizontal_scale=2.0, preview=True))
        self.assertEqual(item.transform_calls, calls_after_preview)
        self.assertTrue(item.clear_text_transform_preview())
        self.assertEqual(block.fontformat.text_transform, original_tuple)

        self.assertTrue(item.set_text_transform(2.0, 0.5, -5.0))
        self.assertEqual(block.fontformat.text_transform, (2.0, 0.5, -5.0))
        calls_after_commit = item.transform_calls
        self.assertFalse(item.set_text_transform(2.0, 0.5, -5.0))
        self.assertEqual(item.transform_calls, calls_after_commit)

        self.assertEqual(item.document().toHtml(), original_html)
        self.assertEqual(item.document().revision(), original_revision)
        self.assertEqual(item.documentSize(), original_size)

    def test_rotation_is_separate_and_cache_tracks_visual_transform(self):
        item, _ = make_item(transform=(1.0, 1.0, 0.0))
        self.assertTrue(item.transform().isIdentity())
        self.assertEqual(
            item.cacheMode(), QGraphicsItem.CacheMode.DeviceCoordinateCache
        )

        item.setRotation(17)
        self.assertTrue(item.transform().isIdentity())
        self.assertEqual(item.rotation(), 17)
        self.assertEqual(item.cacheMode(), QGraphicsItem.CacheMode.NoCache)

        item.setRotation(0)
        self.assertEqual(
            item.cacheMode(), QGraphicsItem.CacheMode.DeviceCoordinateCache
        )
        self.assertTrue(item.set_text_transform(horizontal_scale=1.25))
        self.assertEqual(item.cacheMode(), QGraphicsItem.CacheMode.NoCache)

        scene = QGraphicsScene()
        scene.addItem(item)
        item.startEdit()
        item.micro_focus_calls = 0
        self.assertTrue(item.set_text_transform(slant_angle=8, preview=True))
        self.assertGreaterEqual(item.micro_focus_calls, 1)
        item.endEdit()
        self.assertEqual(item.cacheMode(), QGraphicsItem.CacheMode.NoCache)

    def test_manager_copy_paste_deep_copies_transform_and_command_keeps_it(self):
        original_transform = (1.65, 0.55, -17.0)
        source, _ = make_item(transform=original_transform)
        manager = _CopyPasteManager(source)

        # Exercise the production manager method itself. Its first deepcopy is
        # the clipboard ownership boundary.
        SceneTextManager.onCopyBlkItems(manager)
        self.assertEqual(len(manager.canvas.clipboard_blks), 1)
        clipboard_block = manager.canvas.clipboard_blks[0]
        self.assertEqual(
            clipboard_block.fontformat.text_transform,
            original_transform,
        )
        self.assertIsNot(clipboard_block, source.blk)
        self.assertIsNot(clipboard_block.fontformat, source.blk.fontformat)

        source.set_text_transform(horizontal_scale=0.8)
        source.set_text_transform(vertical_scale=1.9)
        source.set_text_transform(slant_angle=23.0)
        self.assertEqual(source.fontformat.text_transform, (0.8, 1.9, 23.0))
        self.assertEqual(
            clipboard_block.fontformat.text_transform,
            original_transform,
        )

        # onPasteBlkItems performs the second deepcopy and pushes the real
        # PasteBlkItemsCommand through the canvas undo stack.
        SceneTextManager.onPasteBlkItems(manager, QPointF(300.0, 200.0))
        self.assertEqual(manager.canvas.undo_stack.count(), 1)
        self.assertIsInstance(manager.canvas.last_command, PasteBlkItemsCommand)
        self.assertEqual(manager.canvas.last_command.childCount(), 0)
        self.assertEqual(len(manager.textblk_item_list), 2)

        pasted = manager.textblk_item_list[1]
        pasted_pair = manager.pairwidget_list[1]
        self.assertEqual(pasted.fontformat.text_transform, original_transform)
        self.assertEqual(pasted.blk.fontformat.text_transform, original_transform)
        self.assertIsNot(pasted.blk, clipboard_block)
        self.assertIsNot(pasted.fontformat, clipboard_block.fontformat)
        self.assertEqual(
            pasted.transform(),
            text_transform_matrix(
                *original_transform,
                pasted.logical_unpadded_rect().center(),
            ),
        )
        self.assertEqual(manager.formatpanel.calls[-1], ((pasted,), {}))

        # Neither later source edits nor clipboard edits may alias the pasted
        # item's three independent canonical fields.
        source.set_text_transform(0.95, 0.85, -4.0)
        self.assertEqual(
            clipboard_block.fontformat.text_transform,
            original_transform,
        )
        self.assertEqual(pasted.fontformat.text_transform, original_transform)
        clipboard_block.fontformat.horizontal_scale = 3.25
        clipboard_block.fontformat.vertical_scale = 0.25
        clipboard_block.fontformat.slant_angle = 31.0
        self.assertEqual(pasted.fontformat.text_transform, original_transform)

        manager.canvas.undo_stack.undo()
        self.assertNotIn(pasted, manager.textblk_item_list)
        self.assertNotIn(pasted_pair, manager.pairwidget_list)
        self.assertEqual(pasted.fontformat.text_transform, original_transform)

        source.set_text_transform(1.1, 1.2, 7.0)
        manager.canvas.undo_stack.redo()
        self.assertIs(manager.textblk_item_list[1], pasted)
        self.assertIs(manager.pairwidget_list[1], pasted_pair)
        self.assertEqual(pasted.fontformat.text_transform, original_transform)
        self.assertEqual(
            pasted.transform(),
            text_transform_matrix(
                *original_transform,
                pasted.logical_unpadded_rect().center(),
            ),
        )


if __name__ == '__main__':
    unittest.main()

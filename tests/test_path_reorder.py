import os
import unittest
from types import MethodType, SimpleNamespace
from typing import List, Tuple


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QEvent, QPointF, QRectF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QGraphicsRectItem

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TextBlock


class _Item:
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def refresh_order_badge(self) -> None:
        pass

    def update(self) -> None:
        pass


class _Pair:
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def updateIndex(self, idx: int) -> None:
        self.idx = idx

    def hide(self) -> None:
        pass

    def show(self) -> None:
        pass

    def height(self) -> int:
        return 1


class _TextEditList:
    checked_list = []

    def insertPairWidget(self, _pair: _Pair, _idx: int) -> None:
        pass

    def ensureWidgetVisible(self, *_args, **_kwargs) -> None:
        pass


class PathReorderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _make_canvas(self, rects) -> Tuple[Canvas, List[TextBlkItem]]:
        canvas = Canvas()
        canvas.editor_index = 1
        items = []
        for idx, (x, y, width, height) in enumerate(rects):
            block = TextBlock(
                [x, y, x + width, y + height],
                _bounding_rect=[x, y, width, height],
                translation=str(idx),
            )
            item = TextBlkItem(block, idx)
            canvas.attach_text_item(item)
            items.append(item)
        self.addCleanup(self._dispose_canvas, canvas, items)
        return canvas, items

    def _dispose_canvas(
        self,
        canvas: Canvas,
        items: List[TextBlkItem],
    ) -> None:
        canvas.cancel_path_reorder()
        for item in items:
            item.geometry_controller.release_render_resources()
            if item.scene() is canvas:
                canvas.removeItem(item)
        canvas.gv.close()
        self.app.processEvents()

    def test_fast_reverse_stroke_collects_blocks_in_travel_order(self) -> None:
        canvas, items = self._make_canvas(
            [(0, 20, 80, 60), (120, 20, 80, 60), (240, 20, 80, 60)]
        )
        canvas.setSceneRect(QRectF(-20, 0, 380, 120))
        canvas.gv.resize(420, 180)
        canvas.gv.show()
        canvas.gv.fitInView(
            canvas.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.app.processEvents()
        finished = []
        canvas.path_reorder_finished.connect(finished.append)
        canvas.set_order_badges_visible(False)

        self.assertTrue(canvas.start_path_reorder())
        start = canvas.gv.mapFromScene(QPointF(340, 50))
        end = canvas.gv.mapFromScene(QPointF(-20, 50))
        QTest.mousePress(
            canvas.gv.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(end),
            QPointF(end),
            QPointF(canvas.gv.viewport().mapToGlobal(end)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(canvas.gv.viewport(), move)

        self.assertEqual(
            [item.idx for item in canvas._path_reorder_touched],
            [2, 1, 0],
        )
        self.assertEqual(
            [item.order_number() for item in items],
            [3, 2, 1],
        )
        self.assertTrue(
            all(item._order_badge_item.isVisible() for item in items)
        )
        QTest.mouseRelease(
            canvas.gv.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        self.app.processEvents()
        self.assertEqual(finished, [[2, 1, 0]])
        self.assertTrue(
            all(not item._order_badge_item.isVisible() for item in items)
        )

    def test_overlapping_blocks_are_ordered_by_first_contact(self) -> None:
        canvas, _items = self._make_canvas(
            [(0, 20, 300, 60), (100, 20, 20, 60)]
        )
        self.assertTrue(canvas.start_path_reorder())
        canvas._start_path_reorder_stroke(QPointF(-40, 50))
        canvas._extend_path_reorder_stroke(QPointF(340, 50))
        self.assertEqual(
            [item.idx for item in canvas._path_reorder_touched],
            [0, 1],
        )

    def test_history_mutation_cancels_path_preview(self) -> None:
        canvas, items = self._make_canvas(
            [(0, 20, 80, 60), (120, 20, 80, 60)]
        )
        self.assertTrue(canvas.start_path_reorder())
        canvas.push_text_command(None, update_pushed_step=False)
        self.assertFalse(canvas.path_reorder_active)
        self.assertTrue(
            all(item._order_number_override is None for item in items)
        )

    def test_other_canvas_modes_cancel_path_preview(self) -> None:
        canvas, _items = self._make_canvas(
            [(0, 20, 80, 60), (120, 20, 80, 60)]
        )
        self.assertTrue(canvas.start_path_reorder())
        canvas.setTextBlockMode(True)
        self.assertFalse(canvas.path_reorder_active)

        self.assertTrue(canvas.start_path_reorder())
        canvas.setPaintMode(True)
        self.assertFalse(canvas.path_reorder_active)

        canvas.imgtrans_proj = SimpleNamespace(img_valid=True)
        canvas.gv.show()
        self.app.processEvents()
        self.assertTrue(canvas.start_path_reorder())
        canvas.scaleImage(1.1)
        self.assertFalse(canvas.path_reorder_active)

    def test_active_text_editor_blocks_direct_path_start(self) -> None:
        canvas, items = self._make_canvas(
            [(0, 20, 80, 60), (120, 20, 80, 60)]
        )
        editing_item = items[0]
        editing_item.startEdit()
        canvas.editing_textblkitem = editing_item

        self.assertFalse(canvas.start_path_reorder())
        self.assertTrue(editing_item.isEditing())
        self.assertFalse(canvas.path_reorder_active)
        editing_item.endEdit(keep_focus=False)

    def test_cancel_restores_hand_cursor(self) -> None:
        canvas, _items = self._make_canvas(
            [(0, 20, 80, 60), (120, 20, 80, 60)]
        )
        self.assertTrue(canvas.start_path_reorder())
        self.assertEqual(
            canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.CrossCursor,
        )

        canvas.cancel_path_reorder()
        self.assertEqual(
            canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.OpenHandCursor,
        )

    def test_n_toggles_badges_without_intercepting_text_editing(self) -> None:
        canvas, items = self._make_canvas(
            [(40, 40, 80, 60), (160, 40, 80, 60)]
        )
        canvas.setSceneRect(QRectF(0, 0, 280, 140))
        canvas.gv.resize(360, 220)
        canvas.gv.show()
        canvas.gv.setFocus()
        self.app.processEvents()

        self.assertTrue(canvas.order_badges_visible)
        self.assertTrue(
            all(item._order_badge_item.isVisible() for item in items)
        )
        QTest.keyClick(canvas.gv.viewport(), Qt.Key.Key_N)
        self.app.processEvents()
        self.assertFalse(canvas.order_badges_visible)
        self.assertTrue(
            all(not item._order_badge_item.isVisible() for item in items)
        )
        QTest.keyClick(canvas.gv.viewport(), Qt.Key.Key_N)
        self.app.processEvents()
        self.assertTrue(canvas.order_badges_visible)
        self.assertTrue(
            all(item._order_badge_item.isVisible() for item in items)
        )

        editing_item = items[0]
        before = editing_item.toPlainText()
        editing_item.startEdit()
        canvas.editing_textblkitem = editing_item
        QTest.keyClick(canvas.gv.viewport(), Qt.Key.Key_N)
        self.app.processEvents()
        self.assertEqual(editing_item.toPlainText(), 'n' + before)
        self.assertTrue(canvas.order_badges_visible)
        self.assertFalse(editing_item._order_badge_item.isVisible())

        editing_item.endEdit(keep_focus=False)
        canvas.editing_textblkitem = None
        self.assertTrue(editing_item._order_badge_item.isVisible())
        QTest.keyClick(canvas.gv.viewport(), Qt.Key.Key_N)
        self.app.processEvents()
        self.assertFalse(canvas.order_badges_visible)
        self.assertTrue(
            all(not item._order_badge_item.isVisible() for item in items)
        )

    def test_badge_has_its_own_bounds_above_the_text_item(self) -> None:
        canvas, items = self._make_canvas(
            [(40, 40, 80, 60), (160, 40, 80, 60)]
        )
        item = items[0]
        canvas.set_order_badges_visible(True)
        badge = item._order_badge_item
        outline = item.geometry_controller.visual_outline_in_item()

        self.assertEqual(
            badge.mapToScene(QPointF()),
            item.mapToScene(outline.boundingRect().topLeft()),
        )
        self.assertEqual(badge.boundingRect().bottom(), 0)
        self.assertLess(badge.boundingRect().top(), 0)
        badge_center = badge.mapToScene(badge.boundingRect().center())
        self.assertFalse(outline.contains(item.mapFromScene(badge_center)))
        self.assertIn(badge, canvas.items(badge_center))

        item.idx = 11
        item.refresh_order_badge()
        self.assertEqual(badge._text, '12')

        item.set_ui_guide_suppressed(True)
        self.assertFalse(badge.isVisible())
        item.set_ui_guide_suppressed(False)
        self.assertTrue(badge.isVisible())

        item.setPos(item.pos() + QPointF(15, 10))
        self.assertEqual(
            badge.mapToScene(QPointF()),
            item.mapToScene(outline.boundingRect().topLeft()),
        )

        cover = QGraphicsRectItem(badge.sceneBoundingRect())
        cover.setParentItem(canvas.textLayer)
        hits = canvas.items(badge.mapToScene(badge.boundingRect().center()))
        self.assertLess(hits.index(badge), hits.index(cover))

        canvas.removeItem(item)
        self.assertIsNone(badge.scene())
        self.assertIs(badge.parentItem(), item)
        canvas.attach_text_item(item)
        self.assertIs(badge.parentItem(), canvas.orderBadgeLayer)
        self.assertTrue(badge.isVisible())

    def test_reorder_uses_existing_canvas_undo_command(self) -> None:
        items = [_Item(idx) for idx in range(4)]
        pairs = [_Pair(idx) for idx in range(4)]
        pushed = []

        def push(command) -> None:
            pushed.append(command)
            command.redo()

        manager = SimpleNamespace(
            canvas=SimpleNamespace(push_undo_command=push),
            textblk_item_list=items.copy(),
            pairwidget_list=pairs.copy(),
            textEditList=_TextEditList(),
        )
        for method_name in (
            "on_path_reorder_finished",
            "on_rearrange_blks",
            "updateTextBlkItemIdx",
        ):
            setattr(
                manager,
                method_name,
                MethodType(getattr(SceneTextManager, method_name), manager),
            )

        manager.on_path_reorder_finished([2, 0])

        self.assertEqual(
            manager.textblk_item_list,
            [items[2], items[0], items[1], items[3]],
        )
        self.assertEqual(
            [item.idx for item in manager.textblk_item_list],
            [0, 1, 2, 3],
        )
        self.assertEqual(len(pushed), 1)

        pushed[0].undo()
        self.assertEqual(manager.textblk_item_list, items)
        self.assertEqual([item.idx for item in items], [0, 1, 2, 3])

        pushed[0].redo()
        self.assertEqual(
            manager.textblk_item_list,
            [items[2], items[0], items[1], items[3]],
        )
        self.assertEqual(
            [item.idx for item in manager.textblk_item_list],
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()

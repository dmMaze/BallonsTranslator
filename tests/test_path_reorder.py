import os
import unittest
from types import MethodType, SimpleNamespace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QPointF
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TextBlock


class _Item:
    def __init__(self, idx: int) -> None:
        self.idx = idx

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

    def test_fast_reverse_stroke_collects_blocks_in_travel_order(self) -> None:
        canvas = Canvas()
        canvas.editor_index = 1
        items = []
        try:
            for idx, x in enumerate((0, 120, 240)):
                block = TextBlock(
                    [x, 20, x + 80, 80],
                    _bounding_rect=[x, 20, 80, 60],
                    translation=str(idx),
                )
                item = TextBlkItem(block, idx)
                item.setParentItem(canvas.textLayer)
                items.append(item)

            self.assertTrue(canvas.start_path_reorder())
            canvas._start_path_reorder_stroke(QPointF(340, 50))
            canvas._extend_path_reorder_stroke(QPointF(-20, 50))

            self.assertEqual(
                [item.idx for item in canvas._path_reorder_touched],
                [2, 1, 0],
            )
            self.assertEqual(
                [item.order_number() for item in items],
                [3, 2, 1],
            )
        finally:
            canvas.cancel_path_reorder()
            for item in items:
                item.geometry_controller.release_render_resources()
                if item.scene() is canvas:
                    canvas.removeItem(item)
            canvas.gv.close()
            self.app.processEvents()

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


if __name__ == "__main__":
    unittest.main()

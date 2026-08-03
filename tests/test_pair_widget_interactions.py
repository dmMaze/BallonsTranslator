import os
import unittest
from types import MethodType, SimpleNamespace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.editing.widgets import (
    TextEditListScrollArea,
    TransPairWidget,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TextBlock


class _ManagerHarness:
    pass


class PairWidgetInteractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.canvas = Canvas()
        self.canvas.editor_index = 1
        self.edit_list = TextEditListScrollArea()
        self.edit_list.pairwidget_list = []
        self.items = []
        self.pairs = []
        for idx in range(4):
            block = TextBlock(
                [idx * 120, 0, idx * 120 + 100, 60],
                _bounding_rect=[idx * 120, 0, 100, 60],
                translation=f"item {idx}",
            )
            item = TextBlkItem(block, idx)
            item.setParentItem(self.canvas.textLayer)
            pair = TransPairWidget(idx, False)
            self.items.append(item)
            self.pairs.append(pair)
            self.edit_list.pairwidget_list.append(pair)
            self.edit_list.addPairWidget(pair)

        self.panel_updates = []
        formatpanel = SimpleNamespace(
            set_textblk_item=lambda *args, **kwargs: self.panel_updates.append(
                (args, kwargs)
            )
        )
        self.manager = _ManagerHarness()
        self.manager.app = self.app
        self.manager.canvas = self.canvas
        self.manager.textEditList = self.edit_list
        self.manager.textblk_item_list = self.items
        self.manager.pairwidget_list = self.pairs
        self.manager.txtblkShapeControl = self.canvas.txtblkShapeControl
        self.manager.formatpanel = formatpanel
        for name in (
            "_update_selection_panels",
            "editingTextItem",
            "is_editting",
            "on_incanvas_selection_changed",
            "onTextBlkItemEndEdit",
            "on_transwidget_focus_in",
            "on_transwidget_selection_changed",
        ):
            setattr(
                self.manager,
                name,
                MethodType(getattr(SceneTextManager, name), self.manager),
            )

        self.canvas.incanvas_selection_changed.connect(
            self.manager.on_incanvas_selection_changed
        )
        self.edit_list.selection_changed.connect(
            self.manager.on_transwidget_selection_changed
        )
        for item, pair in zip(self.items, self.pairs):
            item.end_edit.connect(self.manager.onTextBlkItemEndEdit)
            pair.e_source.focus_in.connect(self.manager.on_transwidget_focus_in)
            pair.e_trans.focus_in.connect(self.manager.on_transwidget_focus_in)

    def tearDown(self) -> None:
        self.canvas.block_selection_signal = True
        for item in self.items:
            item.geometry_controller.release_render_resources()
            if item.scene() is self.canvas:
                self.canvas.removeItem(item)
        self.edit_list.deleteLater()
        self.canvas.gv.close()
        self.app.processEvents()

    def test_shift_selection_keeps_drag_anchor_and_reorders_selected_pairs(self) -> None:
        self.edit_list.on_widget_checkstate_changed(
            self.pairs[1], shift_pressed=False, ctrl_pressed=False
        )
        self.assertIs(self.edit_list.sel_anchor_widget, self.pairs[1])
        self.assertEqual(
            [item.idx for item in self.canvas.selected_text_items()], [1]
        )

        # A redundant canvas sync must not discard the user's range/drag anchor.
        self.edit_list.set_selected_list([1])
        self.assertIs(self.edit_list.sel_anchor_widget, self.pairs[1])
        self.edit_list.on_widget_checkstate_changed(
            self.pairs[3], shift_pressed=True, ctrl_pressed=False
        )
        self.assertEqual(
            [pair.idx for pair in self.edit_list.checked_list], [1, 2, 3]
        )
        self.assertIs(self.edit_list.sel_anchor_widget, self.pairs[1])

        rearrangements = []
        self.edit_list.rearrange_blks.connect(rearrangements.append)
        self.edit_list.drag_to_pos = 0
        self.edit_list.on_pw_dropped()
        self.assertEqual(rearrangements, [([1, 2, 3, 0], [0, 1, 2, 3])])

    def test_selecting_other_pair_ends_edit_and_replaces_canvas_selection(self) -> None:
        self.edit_list.set_selected_list([0])
        self.items[0].setSelected(True)
        self.canvas.txtblkShapeControl.setBlkItem(self.items[0])
        self.canvas.editing_textblkitem = self.items[0]
        self.items[0].startEdit()
        self.canvas.txtblkShapeControl.startEditing()

        self.edit_list.on_widget_checkstate_changed(
            self.pairs[1], shift_pressed=False, ctrl_pressed=False
        )

        self.assertFalse(self.items[0].isEditing())
        self.assertIsNone(self.canvas.editing_textblkitem)
        self.assertEqual(
            [item.idx for item in self.canvas.selected_text_items()], [1]
        )
        self.assertEqual(
            [pair.idx for pair in self.edit_list.checked_list], [1]
        )


if __name__ == "__main__":
    unittest.main()

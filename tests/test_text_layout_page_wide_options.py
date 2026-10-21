"""Page-wide routing for the two layout options in the format panel.

`FontFormatPanel` needs the whole main window to construct, so these tests
borrow the real method objects and drive them against a real scene of real
``TextBlkItem`` instances. What is exercised is the routing decision, which is
where the previous silent failure lived.
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.text_engine.formatting import commands
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import FontFormat, TextVerticalAlignment
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


BOUNDS = (0, 0, 200, 300)


def make_item(index: int, vertical: bool = True) -> TextBlkItem:
    block = TextBlock(list(BOUNDS), text_layout_version=TEXT_LAYOUT_VERSION)
    block._bounding_rect = list(BOUNDS)
    block.vertical = vertical
    block.translation = '怎么了，master'
    block.fontformat.font_size = 30
    block.fontformat.roman_word_rotation = False
    block.fontformat.vertical_alignment = TextVerticalAlignment.Auto
    return TextBlkItem(block, index)


class _View:
    def __init__(self, scene):
        self._scene = scene

    def scene(self):
        return self._scene

    def hasFocus(self) -> bool:
        # The command restores canvas focus after a local change; reporting
        # focus keeps that path a no-op without a real view widget.
        return True

    def setFocus(self) -> None:
        pass


class _Canvas:
    def __init__(self, view):
        self.gv = view
        self.pushed = []

    def push_undo_command(self, command):
        self.pushed.append(command)
        command.redo()

    def selected_text_items(self):
        return []


class _PanelShim:
    """Real ``FontFormatPanel`` methods, minimal state."""

    PAGE_WIDE_PARAMS = FontFormatPanel.PAGE_WIDE_PARAMS
    page_text_items = FontFormatPanel.page_text_items
    _apply_page_wide_option = FontFormatPanel._apply_page_wide_option
    on_param_changed = FontFormatPanel.on_param_changed

    def __init__(self, global_format: FontFormat, is_global: bool) -> None:
        self.global_format = global_format
        self.textblk_item = None
        self._is_global = is_global
        self.labels_updated = 0

    def global_mode(self) -> bool:
        return self._is_global

    def update_text_style_label(self) -> None:
        self.labels_updated += 1


class PageWideOptionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.scene = QGraphicsScene()
        # Kept on the instance: a scene collected mid-test destroys its items.
        self.empty_scene = QGraphicsScene()
        self.items = [make_item(0), make_item(1, vertical=False)]
        for item in self.items:
            self.scene.addItem(item)
        self.canvas = _Canvas(_View(self.scene))
        self._canvas_patch = patch.object(
            commands.SW, 'canvas', self.canvas, create=True
        )
        self._canvas_patch.start()
        self._active_patch = patch.object(
            C, 'active_format', FontFormat(), create=True
        )
        self._active_patch.start()

    def tearDown(self) -> None:
        self._active_patch.stop()
        self._canvas_patch.stop()
        self.empty_scene.clear()
        self.scene.clear()

    def test_page_text_items_lists_the_whole_page_in_order(self) -> None:
        panel = _PanelShim(FontFormat(), is_global=False)
        self.assertEqual(
            [item.idx for item in panel.page_text_items()], [0, 1]
        )

    def test_page_items_come_from_the_scene_manager_not_the_scene(self) -> None:
        """The app's canvas scene holds no TextBlkItem; the manager owns them.

        This is the regression for the page-wide change doing nothing: the
        scene contained only block handles, so a scene walk found no targets.
        """
        from ballontranslator.ui import shared_widget as SW

        class _Manager:
            def __init__(self, items):
                self.textblk_item_list = items

        # The app's canvas scene exposes no TextBlkItem (only block handles),
        # so point the view at an empty scene and keep the items alive.
        self.canvas.gv = _View(self.empty_scene)
        with patch.object(
            SW, 'st_manager', _Manager(self.items), create=True
        ):
            panel = _PanelShim(FontFormat(), is_global=False)
            self.assertEqual(
                [item.idx for item in panel.page_text_items()], [0, 1]
            )
            panel.on_param_changed('roman_word_rotation', True)
        for item in self.items:
            with self.subTest(idx=item.idx):
                self.assertTrue(item.fontformat.roman_word_rotation)

    def test_page_items_fall_back_to_the_canvas_text_layer(self) -> None:
        from ballontranslator.ui import shared_widget as SW

        # Detach from the original scene first: reparenting across scenes
        # corrupts Qt's item bookkeeping.
        for item in self.items:
            self.scene.removeItem(item)
        layer = self.empty_scene.addRect(0, 0, 1, 1)
        for item in self.items:
            item.setParentItem(layer)
        self.canvas.textLayer = layer
        with patch.object(SW, 'st_manager', None, create=True):
            panel = _PanelShim(FontFormat(), is_global=False)
            self.assertEqual(
                [item.idx for item in panel.page_text_items()], [0, 1]
            )

    def test_unbound_rotation_reaches_every_page_block(self) -> None:
        panel = _PanelShim(FontFormat(), is_global=False)
        panel.on_param_changed('roman_word_rotation', True)
        for item in self.items:
            with self.subTest(idx=item.idx):
                self.assertTrue(item.fontformat.roman_word_rotation)
        self.assertEqual(len(self.canvas.pushed), 1)

    def test_unbound_anchor_reaches_every_page_block(self) -> None:
        panel = _PanelShim(FontFormat(), is_global=False)
        panel.on_param_changed(
            'vertical_alignment', TextVerticalAlignment.Center
        )
        for item in self.items:
            with self.subTest(idx=item.idx):
                self.assertIs(
                    item.fontformat.vertical_alignment,
                    TextVerticalAlignment.Center,
                )

    def test_global_mode_updates_default_and_still_reaches_the_page(self) -> None:
        default = FontFormat()
        panel = _PanelShim(default, is_global=True)
        panel.on_param_changed('roman_word_rotation', True)
        self.assertTrue(default.roman_word_rotation)
        for item in self.items:
            with self.subTest(idx=item.idx):
                self.assertTrue(item.fontformat.roman_word_rotation)

    def test_explicit_selection_is_not_widened_to_the_page(self) -> None:
        panel = _PanelShim(FontFormat(), is_global=False)
        panel.textblk_item = self.items[0]
        panel.on_param_changed('roman_word_rotation', True)
        self.assertTrue(self.items[0].fontformat.roman_word_rotation)
        self.assertFalse(self.items[1].fontformat.roman_word_rotation)

    def test_other_params_keep_the_existing_binding(self) -> None:
        """A typography parameter must not silently become page-wide."""
        panel = _PanelShim(FontFormat(), is_global=True)
        panel.on_param_changed('line_spacing', 1.5)
        self.assertAlmostEqual(panel.global_format.line_spacing, 1.5)
        for item in self.items:
            with self.subTest(idx=item.idx):
                self.assertNotAlmostEqual(item.fontformat.line_spacing, 1.5)


if __name__ == '__main__':
    unittest.main()

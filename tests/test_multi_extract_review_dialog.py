import unittest

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.multi_extract_review_dialog import (
    ALL_FILTER,
    TEXT_FILTER,
    WEAK_FILTER,
    MultiExtractReviewDialog,
    prefixed,
)
from ballontranslator.ui.text_engine.editing.commands import (
    MultiExtractResolveCommand,
)
from ballontranslator.utils.multi_extract import has_conflict
from ballontranslator.utils.textblock import TextBlock

APP = None


def get_app():
    global APP
    if APP is None:
        APP = QApplication.instance() or QApplication([])
    return APP


def make_block(text, candidates=(), support=(0, 1)):
    block = TextBlock([0, 0, 120, 60])
    block._bounding_rect = [0, 0, 120, 60]
    block.text = [text]
    if candidates or len(support) < 2:
        block.multi_extract = {
            'chosen': 0,
            'support': list(support),
            'candidates': [
                {'pass': index + 1, 'text': [candidate]}
                for index, candidate in enumerate(candidates)
            ],
        }
    return block


class FakeProject:
    """Minimal project: one page per entry, no real images."""

    def __init__(self, pages):
        self.pages = pages

    def read_img(self, page):
        raise FileNotFoundError(page)


class ReviewDialogTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def _dialog(self, pages, callback=None):
        dialog = MultiExtractReviewDialog(FakeProject(pages), apply_callback=callback)
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_lists_only_blocks_with_a_payload(self):
        pages = {
            'a.jpg': [make_block('あ'), make_block('い', candidates=['い1'])],
            'b.jpg': [make_block('う', candidates=['う1'])],
        }

        dialog = self._dialog(pages)

        self.assertEqual(dialog.list_widget.count(), 2)
        self.assertIn('2 pending', dialog.summary_label.text())

    def test_text_filter_hides_single_pass_regions(self):
        pages = {
            'a.jpg': [
                make_block('あ', candidates=['あ1']),
                make_block('い', support=(1,)),
            ],
        }

        dialog = self._dialog(pages)
        dialog.filter_combo.setCurrentIndex(
            dialog.filter_combo.findData(TEXT_FILTER)
        )

        self.assertEqual(dialog.list_widget.count(), 1)
        page, block = dialog.current_entry()
        self.assertEqual(block.get_text(), 'あ')

    def test_weak_filter_shows_only_single_pass_regions(self):
        pages = {
            'a.jpg': [
                make_block('あ', candidates=['あ1']),
                make_block('い', support=(1,)),
            ],
        }

        dialog = self._dialog(pages)
        dialog.filter_combo.setCurrentIndex(
            dialog.filter_combo.findData(WEAK_FILTER)
        )

        self.assertEqual(dialog.list_widget.count(), 1)
        _, block = dialog.current_entry()
        self.assertEqual(block.get_text(), 'い')

    def test_all_filter_keeps_both_kinds(self):
        pages = {
            'a.jpg': [
                make_block('あ', candidates=['あ1']),
                make_block('い', support=(1,)),
            ],
        }

        dialog = self._dialog(pages)

        self.assertEqual(dialog.filter_combo.currentData(), ALL_FILTER)
        self.assertEqual(dialog.list_widget.count(), 2)

    def test_single_pass_region_cannot_be_applied(self):
        pages = {'a.jpg': [make_block('い', support=(1,))]}

        dialog = self._dialog(pages)

        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.apply_current()  # must be a no-op
        self.assertTrue(has_conflict(pages['a.jpg'][0]))

    def test_applying_a_candidate_calls_back_and_clears_the_conflict(self):
        calls = []
        block = make_block('そういえば', candidates=['そういえは'])
        dialog = self._dialog({'a.jpg': [block]})

        def apply(page, blk, index):
            calls.append((page, index))
            MultiExtractResolveCommand(blk, index)

        dialog._apply_callback = apply
        dialog._radio_buttons[1].setChecked(True)

        dialog.apply_current()

        self.assertEqual(calls, [('a.jpg', 1)])
        self.assertFalse(has_conflict(block))
        self.assertEqual(block.get_text(), 'そういえは')

    def test_reviewing_the_current_reading_also_clears_the_conflict(self):
        block = make_block('そういえば', candidates=['そういえは'])
        dialog = self._dialog({'a.jpg': [block]})
        dialog._apply_callback = None

        dialog.apply_current()

        self.assertFalse(has_conflict(block))
        self.assertEqual(block.get_text(), 'そういえば')

    def test_empty_project_shows_the_empty_state(self):
        dialog = self._dialog({'a.jpg': [make_block('あ')]})

        self.assertEqual(dialog.list_widget.count(), 0)
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIn('Nothing left', dialog.hint_label.text())

    def test_applying_the_last_conflict_refreshes_the_list(self):
        block = make_block('あ', candidates=['あ1'])
        dialog = self._dialog({'a.jpg': [block]})

        dialog.apply_current()

        self.assertEqual(dialog.list_widget.count(), 0)
        self.assertEqual(dialog.list_widget.currentRow(), -1)

    def test_digit_keys_select_a_reading(self):
        block = make_block('あ', candidates=['あ1'])
        dialog = self._dialog({'a.jpg': [block]})
        self.assertEqual(dialog.chosen_row(), 0)

        dialog.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_2, Qt.KeyboardModifier.NoModifier)
        )

        self.assertEqual(dialog.chosen_row(), 1)


class PrefixedTest(unittest.TestCase):

    def test_short_text_is_kept(self):
        self.assertEqual(prefixed('そういえば'), 'そういえば')

    def test_long_text_is_truncated_with_an_ellipsis(self):
        truncated = prefixed('あ' * 40)

        self.assertTrue(truncated.endswith('…'))
        self.assertEqual(len(truncated), 29)


class ResolveCommandTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_undo_restores_text_and_badge(self):
        block = make_block('そういえば', candidates=['そういえは'])
        command = MultiExtractResolveCommand(block, 1)
        self.assertEqual(block.get_text(), 'そういえは')

        command.undo()
        self.assertEqual(block.get_text(), 'そういえば')
        self.assertTrue(has_conflict(block))

        command.redo()
        command.redo()  # the first redo is the push that already happened
        self.assertEqual(block.get_text(), 'そういえは')
        self.assertFalse(has_conflict(block))

    def test_refresh_callback_runs_on_every_state_change(self):
        refreshed = []
        block = make_block('あ', candidates=['あ1'])
        command = MultiExtractResolveCommand(
            block, 1, refresh=lambda blk: refreshed.append(blk.get_text())
        )

        command.undo()
        command.redo()  # skipped: the stack already pushed this command

        self.assertEqual(refreshed, ['あ'])


if __name__ == '__main__':
    unittest.main()

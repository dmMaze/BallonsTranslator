import os
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QMainWindow

from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.mainwindowbars import LeftBar, TitleBar
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.utils.multi_extract import has_conflict
from ballontranslator.utils.textblock import TextBlock

_APP = None


def qapp() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv[:1])
    return _APP


class FakeAction:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def isEnabled(self):
        return self.enabled


def make_block(text='そういえば', candidate='そういえは'):
    block = TextBlock([0, 0, 120, 60])
    block._bounding_rect = [0, 0, 120, 60]
    block.text = [text]
    block.multi_extract = {
        'chosen': 0,
        'support': [0, 1],
        'candidates': [{'pass': 1, 'text': [candidate]}],
    }
    return block


class ReviewActionStateTest(unittest.TestCase):

    def _host(self, pages):
        return SimpleNamespace(
            imgtrans_proj=SimpleNamespace(pages=pages),
            titleBar=SimpleNamespace(reviewMultiExtractAction=FakeAction()),
        )

    def test_action_is_disabled_without_conflicts(self):
        host = self._host({'a.jpg': [make_block()]})
        host.imgtrans_proj.pages['a.jpg'][0].multi_extract = {}

        MainWindow._refresh_multi_extract_review_action(host)

        self.assertFalse(host.titleBar.reviewMultiExtractAction.isEnabled())

    def test_action_is_enabled_with_conflicts(self):
        host = self._host({'a.jpg': [make_block()]})

        MainWindow._refresh_multi_extract_review_action(host)

        self.assertTrue(host.titleBar.reviewMultiExtractAction.isEnabled())

    def test_missing_action_is_tolerated(self):
        host = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(pages={'a.jpg': [make_block()]}),
            titleBar=SimpleNamespace(),
        )

        MainWindow._refresh_multi_extract_review_action(host)


class ApplyReadingTest(unittest.TestCase):

    def _host(self):
        host = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(current_img='a.jpg', pages={}),
            canvas=SimpleNamespace(
                textEditMode=lambda: False,
                saved=[], 
                setProjSaveState=lambda value: host.canvas.saved.append(value),
                update=lambda: None,
            ),
            st_manager=SimpleNamespace(
                textblk_item_list=[],
                pairwidget_list=[],
            ),
            titleBar=SimpleNamespace(reviewMultiExtractAction=FakeAction()),
        )
        # Bound helpers the method under test calls on itself.
        host._refresh_resolved_block = (
            lambda page, blk: MainWindow._refresh_resolved_block(host, page, blk)
        )
        host._refresh_multi_extract_review_action = (
            lambda: MainWindow._refresh_multi_extract_review_action(host)
        )
        return host

    def test_applying_outside_text_mode_writes_text_and_marks_the_project(self):
        block = make_block()
        host = self._host()
        host.imgtrans_proj.pages['a.jpg'] = [block]

        MainWindow._apply_multi_extract_reading(host, 'a.jpg', block, 1)

        self.assertEqual(block.get_text(), 'そういえは')
        self.assertFalse(has_conflict(block))
        self.assertEqual(host.canvas.saved, [True])
        self.assertFalse(host.titleBar.reviewMultiExtractAction.isEnabled())

    def test_refreshing_a_block_on_another_page_is_a_no_op(self):
        block = make_block()
        host = self._host()
        host.imgtrans_proj.current_img = 'b.jpg'
        host.imgtrans_proj.pages['a.jpg'] = [block]

        MainWindow._refresh_resolved_block(host, 'a.jpg', block)

        self.assertEqual(host.canvas.saved, [])

    def test_refreshing_missing_block_does_not_raise(self):
        host = self._host()
        host.imgtrans_proj.pages['a.jpg'] = []

        MainWindow._refresh_resolved_block(host, 'a.jpg', make_block())


class ReviewMenuOwnershipTest(unittest.TestCase):
    """The review entry is wired to the bar that actually owns the Edit menu.

    SetupUi connects this signal through a parent window attribute, so a bar
    swap compiles and imports fine and only fails when the window is built.
    """

    def _bar(self, kind):
        app = qapp()
        host = QMainWindow()
        bar = kind(host)
        app.processEvents()
        self.addCleanup(host.deleteLater)
        self.addCleanup(bar.deleteLater)
        return bar

    def test_title_bar_exposes_the_review_trigger(self):
        bar = self._bar(TitleBar)

        self.assertTrue(hasattr(bar, 'review_multi_extract_trigger'))
        self.assertTrue(
            any(
                action is bar.reviewMultiExtractAction
                for action in bar.editToolBtn.menu().actions()
            )
        )

    def test_left_bar_does_not_expose_the_review_trigger(self):
        bar = self._bar(LeftBar)

        self.assertFalse(hasattr(bar, 'review_multi_extract_trigger'))


class ReviewRequestForwardingTest(unittest.TestCase):
    """A badge click is forwarded only after the scene unwinds its press."""

    class _Signal:
        def __init__(self):
            self.emitted = 0

        def emit(self):
            self.emitted += 1

    def test_forwarding_waits_for_the_next_event_loop_pass(self):
        qapp()
        signal = self._Signal()
        manager = SimpleNamespace(review_multi_extract_requested=signal)

        SceneTextManager.on_review_multi_extract_requested(manager)

        self.assertEqual(signal.emitted, 0)
        qapp().processEvents()
        self.assertEqual(signal.emitted, 1)


if __name__ == '__main__':
    unittest.main()

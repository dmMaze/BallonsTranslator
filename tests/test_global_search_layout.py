import os
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.global_search_widget import GlobalSearchWidget

# The application must outlive every widget; a local-only reference lets PyQt
# destroy the QApplication (and the whole widget tree) when the helper returns.
_APP = None


def qapp() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv[:1])
    return _APP


class GlobalSearchEditorWidthTest(unittest.TestCase):
    """The replace row must line up with the find row at any panel width."""

    def _build(self, width: int) -> GlobalSearchWidget:
        app = qapp()
        widget = GlobalSearchWidget()
        widget.resize(width, 600)
        widget.show()
        app.processEvents()
        self.addCleanup(self._destroy, widget)
        return widget

    @staticmethod
    def _destroy(widget: GlobalSearchWidget) -> None:
        widget.close()
        widget.deleteLater()
        qapp().processEvents()

    def test_replace_editor_matches_find_editor_width(self):
        for width in (320, 520, 760):
            with self.subTest(width=width):
                widget = self._build(width)
                self.assertEqual(
                    widget.search_editor.width(),
                    widget.replace_editor.width(),
                )
                self.assertEqual(
                    widget.search_editor.geometry().right(),
                    widget.replace_editor.geometry().right(),
                )

    def test_replace_editor_stays_below_find_editor(self):
        widget = self._build(520)
        self.assertGreater(
            widget.replace_editor.geometry().top(),
            widget.search_editor.geometry().top(),
        )

    def test_range_selector_shares_replace_row(self):
        widget = self._build(520)
        replace_box = widget.replace_editor.geometry()
        for control in (widget.range_label, widget.range_combobox):
            box = control.geometry()
            self.assertTrue(
                replace_box.top() <= box.center().y() <= replace_box.bottom(),
                f'{control.objectName() or type(control).__name__} is not on the replace row',
            )


if __name__ == '__main__':
    unittest.main()

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QRect
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from ballontranslator.ui.adaptive_wrap_layout import AdaptiveWrapLayout
from ballontranslator.ui.text_engine.formatting.advanced import (
    TextShadowGroup,
    _atomic_unit,
)


class AdaptiveWrapLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_expanding_units_fill_available_row_width(self):
        host = QWidget()
        layout = AdaptiveWrapLayout(
            host, horizontal_spacing=5, vertical_spacing=5
        )
        first = _atomic_unit(host, QLabel('First'))
        second = _atomic_unit(host, QLabel('Second'))
        layout.addWidget(first)
        layout.addWidget(second)

        layout.setGeometry(QRect(0, 0, 300, 40))

        self.assertEqual(second.geometry().right(), layout.contentsRect().right())

    def test_advanced_format_editors_expand_across_semantic_rows(self):
        shadow = TextShadowGroup(lambda *_args: None, 'Shadow')
        shadow.resize(600, 120)
        shadow.show()
        self.app.processEvents()

        self.assertEqual(
            shadow.offset_unit.geometry().right(),
            shadow.offset_layout.contentsRect().right(),
        )
        self.assertEqual(
            shadow.radius_unit.geometry().right(),
            shadow.detail_layout.contentsRect().right(),
        )


if __name__ == '__main__':
    unittest.main()

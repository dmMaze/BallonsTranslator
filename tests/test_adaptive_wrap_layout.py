import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QRect
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from ballontranslator.ui.adaptive_wrap_layout import AdaptiveWrapLayout
from ballontranslator.ui.text_engine.formatting.advanced import (
    TextGradientGroup,
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

    def test_gradient_toggle_publishes_one_persistent_change(self):
        changes = []
        gradient = TextGradientGroup(
            lambda name, value: changes.append((name, value))
        )

        gradient.enable_checker.setChecked(True)

        self.assertEqual(changes, [('gradient_enabled', True)])


if __name__ == '__main__':
    unittest.main()

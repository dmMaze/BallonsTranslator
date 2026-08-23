import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPoint, QRect, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from ballontranslator.ui.adaptive_wrap_layout import AdaptiveWrapLayout
from ballontranslator.ui.text_engine.formatting.advanced import (
    TextGradientGroup,
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

    def test_shadow_strength_is_a_direct_persistent_control(self):
        changes = []
        shadow = TextShadowGroup(
            lambda name, value: changes.append((name, value)),
            'Shadow',
        )

        shadow.strength_box.param_changed.emit('shadow_strength', 0.7)

        self.assertEqual(changes, [('shadow_strength', 0.7)])
        self.assertFalse(hasattr(shadow, 'enable_checker'))

    def test_size_control_click_is_not_a_formatting_change(self):
        changes = []
        shadow = TextShadowGroup(
            lambda name, value: changes.append((name, value)),
            'Shadow',
        )

        QTest.mouseClick(
            shadow.strength_label, Qt.MouseButton.LeftButton
        )

        self.assertEqual(changes, [])

    def test_shadow_strength_double_click_restores_model_default(self):
        changes = []
        shadow = TextShadowGroup(
            lambda name, value: changes.append((name, value)),
            'Shadow',
        )
        shadow.strength_box.setValue(0.4)

        QTest.mouseDClick(
            shadow.strength_label,
            Qt.MouseButton.LeftButton,
            pos=QPoint(2, 2),
        )

        self.assertEqual(shadow.strength_box.value(), 1.0)
        self.assertEqual(changes, [('shadow_strength', 1.0)])

        changes.clear()
        QTest.mouseDClick(
            shadow.strength_label,
            Qt.MouseButton.LeftButton,
            pos=QPoint(2, 2),
        )
        self.assertEqual(changes, [])

    def test_gradient_toggle_publishes_one_persistent_change(self):
        changes = []
        gradient = TextGradientGroup(
            lambda name, value: changes.append((name, value))
        )

        gradient.enable_checker.setChecked(True)

        self.assertEqual(changes, [('gradient_enabled', True)])


if __name__ == '__main__':
    unittest.main()

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import (
    QCoreApplication,
    QEvent,
    QMimeData,
    QPoint,
    QPointF,
    QRect,
    Qt,
)
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.formatting import presets
from ballontranslator.utils import config, shared
from ballontranslator.utils.fontformat import FontFormat


class TextStylePresetReorderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.original_styles = list(config.text_styles)
        self.panels: list[presets.TextStylePresetPanel] = []
        config.text_styles.clear()

    def tearDown(self) -> None:
        for panel in self.panels:
            panel.view_widget.close()
            panel.view_widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        config.text_styles[:] = self.original_styles

    def _make_panel(self, *style_names: str) -> presets.TextStylePresetPanel:
        config.text_styles[:] = [
            FontFormat(_style_name=style_name) for style_name in style_names
        ]
        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = presets.TextStylePresetPanel(
                'Text Style', 'text_style', 'expand_tstyle_panel'
            )
        panel.initStyles(config.text_styles)
        self.panels.append(panel)
        return panel

    @staticmethod
    def _labels(
        panel: presets.TextStylePresetPanel,
    ) -> list[presets.TextStyleLabel]:
        return [panel.flayout.itemAt(index).widget() for index in range(panel.count())]

    def test_reordering_moves_widget_and_saved_style_together(self) -> None:
        panel = self._make_panel('First', 'Second', 'Third')
        panel.flayout.setGeometry(QRect(0, 0, 2000, 200))
        first_label, _, third_label = self._labels(panel)
        self.assertTrue(panel.scrollContent.acceptDrops())
        self.assertFalse(first_label.acceptDrops())
        self.assertFalse(first_label.stylelabel.acceptDrops())
        QTest.mouseClick(third_label, Qt.MouseButton.LeftButton)
        self.assertIs(panel.active_text_style_label, third_label)
        third_item = panel.flayout.itemAt(2)
        mime_data = QMimeData()
        mime_data.setData(presets.TextStyleLabel.text_style_mime_type, b'')
        drop_event = Mock()
        drop_event.type.return_value = QEvent.Type.Drop
        drop_event.source.return_value = third_label
        drop_event.mimeData.return_value = mime_data
        drop_event.position.return_value = QPointF(
            first_label.geometry().left(), first_label.geometry().center().y()
        )

        with patch.object(presets, 'save_text_styles') as save_styles:
            handled = panel.eventFilter(panel.scrollContent, drop_event)

        expected_names = ['Third', 'First', 'Second']
        self.assertEqual(
            [style._style_name for style in config.text_styles], expected_names
        )
        self.assertEqual(
            [label.fontfmt._style_name for label in self._labels(panel)],
            expected_names,
        )
        self.assertIs(panel.flayout.itemAt(0), third_item)
        self.assertTrue(handled)
        drop_event.acceptProposedAction.assert_called_once_with()
        save_styles.assert_called_once_with()

    def test_drop_in_row_gap_uses_horizontal_insertion_point(self) -> None:
        panel = self._make_panel(*(f'Style {index}' for index in range(6)))
        panel.flayout.setGeometry(QRect(0, 0, 2000, 200))
        labels = self._labels(panel)
        first_geometry = labels[0].geometry()
        second_geometry = labels[1].geometry()
        gap_position = QPoint(
            (first_geometry.right() + second_geometry.left()) // 2,
            first_geometry.center().y(),
        )

        with patch.object(presets, 'save_text_styles') as save_styles:
            panel._reorderStyleAtPosition(labels[-1], gap_position)

        expected_names = [
            'Style 0',
            'Style 5',
            'Style 1',
            'Style 2',
            'Style 3',
            'Style 4',
        ]
        self.assertEqual(
            [style._style_name for style in config.text_styles], expected_names
        )
        self.assertEqual(
            [label.fontfmt._style_name for label in self._labels(panel)],
            expected_names,
        )
        save_styles.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()

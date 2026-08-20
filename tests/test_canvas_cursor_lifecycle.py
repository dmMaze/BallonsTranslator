import os
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsRectItem,
    QHBoxLayout,
    QStackedWidget,
    QWidget,
)
from qtpy.QtTest import QTest

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel


def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()


class CanvasCursorLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canvas = Canvas()
        cls.panel = DrawingPanel(cls.canvas)
        cls.text_panel = QWidget()
        cls.stack = QStackedWidget()
        cls.stack.addWidget(cls.panel)
        cls.stack.addWidget(cls.text_panel)

        cls.window = QWidget()
        layout = QHBoxLayout(cls.window)
        layout.addWidget(cls.canvas.gv)
        layout.addWidget(cls.stack)
        cls.window.resize(700, 400)
        cls.canvas.setSceneRect(0, 0, 300, 300)
        cls.canvas.baseLayer.setRect(QRectF(0, 0, 300, 300))

        cls.text_cursor_item = QGraphicsRectItem(
            QRectF(80, 80, 120, 120),
            cls.canvas.textLayer,
        )
        cls.text_cursor_item.setAcceptHoverEvents(True)
        cls.text_cursor_item.setCursor(Qt.CursorShape.IBeamCursor)

        cls.window.show()
        _APP.processEvents()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.window.close()
        _APP.processEvents()

    def setUp(self) -> None:
        self.canvas.clearToolStates()
        self.canvas.clear_states()
        self.stack.setCurrentIndex(0)
        self.canvas.editor_index = 0
        self.canvas.textLayer.show()
        self.panel.penTool.setChecked(True)
        self.panel.on_use_pentool()
        _APP.processEvents()

    def _move_to_scene(self, x: float, y: float) -> None:
        QTest.mouseMove(
            self.canvas.gv.viewport(),
            self.canvas.gv.mapFromScene(QPointF(x, y)),
        )
        _APP.processEvents()

    def test_drawing_cursor_survives_hidden_text_item_hover_leave(self) -> None:
        self.stack.setCurrentIndex(1)
        self.canvas.editor_index = 1
        self.canvas.setPaintMode(False)
        self.canvas.textLayer.show()
        self._move_to_scene(250, 250)
        self._move_to_scene(120, 120)
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.IBeamCursor,
        )

        self.stack.setCurrentIndex(0)
        self.canvas.editor_index = 0
        self.canvas.setPaintMode(True)
        self.canvas.textLayer.hide()
        self._move_to_scene(250, 250)

        cursor = self.canvas.gv.viewport().cursor()
        self.assertEqual(cursor.shape(), Qt.CursorShape.BitmapCursor)
        self.assertFalse(cursor.pixmap().isNull())

        self.panel.on_use_handtool()
        self._move_to_scene(250, 250)
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.OpenHandCursor,
        )

    def test_text_block_creation_crosshair_is_released(self) -> None:
        self.stack.setCurrentIndex(1)
        self.canvas.editor_index = 1
        self.canvas.setPaintMode(False)
        self.canvas.textLayer.show()
        self._move_to_scene(250, 250)

        self.canvas.startCreateTextblock(QPointF(40, 40))
        self.canvas.txtblkShapeControl.hide()
        self.canvas.gv.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        self._move_to_scene(251, 250)
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.CrossCursor,
        )

        self.canvas.endCreateTextblock()
        self.assertNotEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.CrossCursor,
        )

        self.canvas.startCreateTextblock(QPointF(40, 40))
        self.canvas.editor_index = 0
        self.canvas.setPaintMode(True)
        self.assertNotEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.CrossCursor,
        )
        self.assertFalse(self.canvas.txtblkShapeControl.isVisible())

    def test_rect_and_scale_cursor_return_to_the_active_tool(self) -> None:
        self._move_to_scene(250, 250)
        self.panel.on_use_recttool()
        rect_cursor = QCursor(self.canvas.gv.viewport().cursor())
        self.canvas.startCreateTextblock(QPointF(40, 40), hide_control=True)
        self.canvas.clear_states()
        self.assertEqual(self.canvas.gv.viewport().cursor(), rect_cursor)

        self.panel.on_use_pentool()
        self.panel.setPenToolWidth(30)
        self._move_to_scene(100, 100)
        self.panel.on_begin_scale_tool(QPointF(100, 100))
        self._move_to_scene(101, 100)
        scale_cursor = QCursor(self.canvas.gv.viewport().cursor())
        self.assertIsNotNone(self.panel.scale_tool_pos)
        self.assertEqual(scale_cursor, self.panel.scale_circle.cursor())

        self.canvas.on_activation_changed()
        self._move_to_scene(100, 100)
        self.assertIsNone(self.panel.scale_tool_pos)
        self.assertNotEqual(self.canvas.gv.viewport().cursor(), scale_cursor)

    def test_inpaint_cursor_uses_the_same_scene_owner(self) -> None:
        self._move_to_scene(250, 250)
        self.panel.on_use_pentool()
        pen_cursor = QCursor(self.canvas.gv.viewport().cursor())

        self.panel.on_use_inpainttool()
        inpaint_cursor = QCursor(self.canvas.gv.viewport().cursor())
        self.assertEqual(inpaint_cursor.shape(), Qt.CursorShape.BitmapCursor)
        self.assertNotEqual(inpaint_cursor, pen_cursor)

        self._move_to_scene(120, 120)
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.IBeamCursor,
        )
        self.canvas.textLayer.hide()
        self._move_to_scene(250, 250)
        self.assertEqual(self.canvas.gv.viewport().cursor(), inpaint_cursor)

    def test_native_and_item_cursors_return_after_canvas_cursor_release(self) -> None:
        self.panel.on_use_handtool()
        self.canvas.setPaintMode(False)
        self.canvas.setPaintMode(False)
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.OpenHandCursor,
        )

        self.stack.setCurrentIndex(1)
        self.canvas.editor_index = 1
        self.canvas.textLayer.show()
        self.canvas.startCreateTextblock(QPointF(40, 40))
        self.canvas.clear_states()
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.OpenHandCursor,
        )

        self._move_to_scene(250, 250)
        self._move_to_scene(120, 120)
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.IBeamCursor,
        )

    def test_hidden_drawing_tool_cannot_claim_text_cursor(self) -> None:
        self.stack.setCurrentIndex(1)
        self.canvas.editor_index = 1
        self.canvas.setPaintMode(False)
        self.canvas.textLayer.show()
        self._move_to_scene(250, 250)
        self._move_to_scene(120, 120)

        self.panel.setPenCursor()
        self._move_to_scene(120, 120)
        self.assertEqual(
            self.canvas.gv.viewport().cursor().shape(),
            Qt.CursorShape.IBeamCursor,
        )


if __name__ == '__main__':
    unittest.main()

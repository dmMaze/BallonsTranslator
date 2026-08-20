import numpy as np
from typing import Callable, List, Optional, Union
import os

from qtpy.QtWidgets import QApplication, QSlider, QMenu, QGraphicsScene, QGraphicsSceneDragDropEvent , QGraphicsView, QGraphicsSceneDragDropEvent, QGraphicsRectItem, QGraphicsItem, QScrollBar, QGraphicsPixmapItem, QGraphicsSceneMouseEvent, QGraphicsSceneContextMenuEvent, QRubberBand
from qtpy.QtCore import Qt, QDateTime, QRectF, QPointF, QPoint, Signal, QSize, QSizeF, QEvent, QTimer
from qtpy.QtGui import QKeySequence, QPixmap, QImage, QHideEvent, QKeyEvent, QWheelEvent, QResizeEvent, QPainter, QPen, QPainterPath, QCursor, QNativeGestureEvent
from qtpy.QtWidgets import QGraphicsPathItem
from qtpy.QtCore import QLineF
from qtpy.QtGui import QColor, QPainterPathStroker

try:
    from qtpy.QtWidgets import QUndoStack, QUndoCommand
except:
    from qtpy.QtGui import QUndoStack, QUndoCommand

from .misc import ndarray2pixmap, QKEY, QNUMERIC_KEYS, ARROWKEY2DIRECTION
from .text_engine.item import TextBlkItem, TextBlock
from .text_engine.shape_control import (
    CONTROL_ITEM_DATA_KEY,
    TextBlkShapeControl,
)
from .text_engine.transforms.grid_control import TextGridTransformControl
from .text_engine.transforms.projective_control import TextProjectiveTransformControl
from .custom_widget import ScrollBar, FadeLabel
from .image_edit import ImageEditMode, DrawingLayer, StrokeImgItem
from .page_search_widget import PageSearchWidget
from .text_engine.editing.commands import MoveByKeyCommand
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans

CANVAS_SCALE_MAX = 10.0
CANVAS_SCALE_MIN = 0.01
CANVAS_SCALE_SPEED = 0.1


def _segment_rect_entry(
    start: QPointF,
    end: QPointF,
    rect: QRectF,
    padding: float = 0.0,
) -> float:
    """Return where a segment first enters a padded rectangle.

    >>> _segment_rect_entry(
    ...     QPointF(0, 0), QPointF(100, 0), QRectF(20, -5, 20, 10)
    ... )
    0.2
    """
    padding = max(0.0, padding)
    rect = rect.adjusted(-padding, -padding, padding, padding)
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    entry = 0.0
    exit_ = 1.0
    for origin, delta, lower, upper in (
        (start.x(), dx, rect.left(), rect.right()),
        (start.y(), dy, rect.top(), rect.bottom()),
    ):
        if abs(delta) < 1e-9:
            if origin < lower or origin > upper:
                return 1.0
            continue
        near = (lower - origin) / delta
        far = (upper - origin) / delta
        if near > far:
            near, far = far, near
        entry = max(entry, near)
        exit_ = min(exit_, far)
        if entry > exit_:
            return 1.0
    return min(max(entry, 0.0), 1.0)


class CustomGV(QGraphicsView):
    ctrl_pressed = False
    scale_up_signal = Signal()
    scale_down_signal = Signal()
    scale_with_value = Signal(float)
    view_resized = Signal()
    hide_canvas = Signal()
    ctrl_released = Signal()
    canvas: Optional["Canvas"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, self, fadeout=True)
        self.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, self, fadeout=True)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event : QWheelEvent) -> None:
        # qgraphicsview always scroll content according to wheelevent
        # which is not desired when scaling img
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_up_signal.emit()
            else:
                self.scale_down_signal.emit()
            return
        return super().wheelEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == QKEY.Key_Control:
            self.ctrl_pressed = False
            self.ctrl_released.emit()
        return super().keyReleaseEvent(event)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        key = e.key()
        if key == QKEY.Key_Control:
            self.ctrl_pressed = True

        if self.canvas is not None and self.canvas.path_reorder_active:
            return super().keyPressEvent(e)

        modifiers = e.modifiers()
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == QKEY.Key_V:
                # self.ctrlv_pressed.emit(e)
                if self.canvas.handle_ctrlv():
                    e.accept()
                    return
            if key == QKEY.Key_C:
                if self.canvas.handle_ctrlc():
                    e.accept()
                    return
                
        elif modifiers & Qt.KeyboardModifier.ControlModifier and modifiers & Qt.KeyboardModifier.ShiftModifier:
            if key == QKEY.Key_C:
                self.canvas.copy_src_signal.emit()
                e.accept()
                return
            elif key == QKEY.Key_V:
                self.canvas.paste_src_signal.emit()
                e.accept()
                return
            elif key == QKEY.Key_D:
                self.canvas.delete_textblks.emit(1)
                e.accept()
                return

        return super().keyPressEvent(e)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.view_resized.emit()

    def hideEvent(self, event: QHideEvent) -> None:
        self.hide_canvas.emit()
        return super().hideEvent(event)

    def event(self, e: QEvent) -> bool:
        if isinstance(e, QNativeGestureEvent):
            if e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.scale_with_value.emit(e.value() + 1)
                e.setAccepted(True)

        return super().event(e)
    
    def dragMoveEvent(self, e: QGraphicsSceneDragDropEvent) -> None:
        super().dragMoveEvent(e)
        if e.mimeData().hasUrls():
            # issue #908, https://stackoverflow.com/questions/4177720/accepting-drops-on-a-qgraphicsscene
            e.setAccepted(True)


class Canvas(QGraphicsScene):

    scalefactor_changed = Signal()
    end_create_textblock = Signal(QRectF)
    paste2selected_textitems = Signal()
    end_create_rect = Signal(QRectF, int)
    finish_painting = Signal(StrokeImgItem)
    finish_erasing = Signal(StrokeImgItem)
    delete_textblks = Signal(int)
    copy_textblks = Signal()
    paste_textblks = Signal(QPointF)
    copy_src_signal = Signal()
    paste_src_signal = Signal()

    format_textblks = Signal()
    layout_textblks = Signal()
    reset_angle = Signal()
    squeeze_blk = Signal()

    run_blktrans = Signal(int)

    begin_scale_tool = Signal(QPointF)
    scale_tool = Signal(QPointF)
    end_scale_tool = Signal()
    canvas_undostack_changed = Signal()
    
    imgtrans_proj: ProjImgTrans = None
    painting_pen = QPen()
    painting_shape = 0
    erasing_pen = QPen()
    image_edit_mode = ImageEditMode.NONE

    projstate_unsaved = False
    proj_savestate_changed = Signal(bool)
    textstack_changed = Signal()
    drop_open_folder = Signal(str)
    context_menu_requested = Signal(QPoint, bool)
    incanvas_selection_changed = Signal()
    switch_text_item = Signal(int, QKeyEvent)
    path_reorder_finished = Signal(object)
    path_reorder_mode_changed = Signal(bool)
    projective_scale_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.
        self.text_transparency = 0
        self.textblock_mode = False
        self.order_badges_visible = True
        self.creating_textblock = False
        self._text_creation_cursor_active = False
        self.create_block_origin: QPointF = None
        self.editing_textblkitem: TextBlkItem = None
        self._path_reorder_active = False
        self._path_reorder_drawing = False
        self._path_reorder_path = QPainterPath()
        self._path_reorder_path_item: Optional[QGraphicsPathItem] = None
        self._path_reorder_items: List[TextBlkItem] = []
        self._path_reorder_touched: List[TextBlkItem] = []
        self._path_reorder_last_pos: Optional[QPointF] = None

        self.gv = CustomGV(self)
        self.gv.scale_down_signal.connect(self.scaleDown)
        self.gv.scale_up_signal.connect(self.scaleUp)
        self.gv.scale_with_value.connect(self.scaleBy)
        self.gv.view_resized.connect(self.onViewResized)
        self.gv.hide_canvas.connect(self.on_hide_canvas)
        self.gv.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.gv.canvas = self
        self.gv.setAcceptDrops(True)
        self.gv.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.gv.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.context_menu_requested.connect(self.on_create_contextmenu)
        
        if not shared.FLAG_QT6:
            # mitigate https://bugreports.qt.io/browse/QTBUG-93417
            # produce blurred result, saving imgs remain unaffected
            self.gv.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.search_widget = PageSearchWidget(self.gv)
        self.search_widget.hide()
        
        self.ctrl_relesed = self.gv.ctrl_released
        self.vscroll_bar = self.gv.verticalScrollBar()
        self.hscroll_bar = self.gv.horizontalScrollBar()
        # self.default_cursor = self.gv.cursor()
        # Grid handle selection reuses this scene-owned gesture so there is one
        # mouse lifecycle and one rubber-band visual to cancel or finish.
        self.rubber_band = self.addWidget(QRubberBand(QRubberBand.Shape.Rectangle))
        self.rubber_band.hide()
        self.rubber_band.setZValue(100)
        self.rubber_band_origin = None
        self.rubber_band_modifiers = Qt.KeyboardModifier.NoModifier
        self._rubber_band_button = Qt.MouseButton.NoButton
        self._rubber_band_target = None
        self._rubber_band_update = None
        self._rubber_band_finish = None

        self.draw_undo_stack = QUndoStack(self)
        self.text_undo_stack = QUndoStack(self)
        self.saved_drawundo_step = 0
        self.saved_textundo_step = 0

        self.scaleFactorLabel = FadeLabel(self.gv)
        self.scaleFactorLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scaleFactorLabel.setText('100%')
        self.scaleFactorLabel.gv = self.gv

        self.txtblkShapeControl = TextBlkShapeControl(self.gv)
        self.txtblkGridControl = TextGridTransformControl()
        self.txtblkProjectiveControl = TextProjectiveTransformControl()
        
        self.baseLayer = QGraphicsRectItem()
        pen = QPen()
        pen.setColor(Qt.GlobalColor.transparent)
        self.baseLayer.setPen(pen)

        self.inpaintLayer = QGraphicsPixmapItem()
        self.inpaintLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.drawingLayer = DrawingLayer()
        self.drawingLayer.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.textLayer = QGraphicsPixmapItem()
        self.orderBadgeLayer = QGraphicsRectItem()
        self.orderBadgeLayer.setZValue(100.0)

        self.inpaintLayer.setAcceptDrops(True)
        self.drawingLayer.setAcceptDrops(True)
        self.textLayer.setAcceptDrops(True)
        self.baseLayer.setAcceptDrops(True)
        
        self.base_pixmap: QPixmap = None

        self.addItem(self.baseLayer)
        self.inpaintLayer.setParentItem(self.baseLayer)
        self.drawingLayer.setParentItem(self.baseLayer)
        self.textLayer.setParentItem(self.baseLayer)
        self.orderBadgeLayer.setParentItem(self.textLayer)
        self.txtblkShapeControl.setParentItem(self.baseLayer)
        self.txtblkGridControl.setParentItem(self.baseLayer)
        self.txtblkProjectiveControl.setParentItem(self.baseLayer)
        self._text_shape_refresh_timer = QTimer(self.gv)
        self._text_shape_refresh_timer.setSingleShot(True)
        self._text_shape_refresh_timer.timeout.connect(
            self._refresh_text_shape_control_now
        )
        self.hscroll_bar.valueChanged.connect(self.refresh_text_shape_control)
        self.vscroll_bar.valueChanged.connect(self.refresh_text_shape_control)

        self.scalefactor_changed.connect(self.onScaleFactorChanged)
        self.selectionChanged.connect(self.on_selection_changed)     

        self.stroke_img_item: StrokeImgItem = None
        self.erase_img_key = None

        self.editor_index = 0 # 0: drawing 1: text editor
        self.mid_btn_pressed = False
        self.pan_initial_pos = QPoint(0, 0)

        self.saved_textundo_step = 0
        self.saved_drawundo_step = 0
        self.num_pushed_textstep = 0
        self.num_pushed_drawstep = 0

        self.clipboard_blks: List[TextBlock] = []

        self.drop_folder: str = None
        self.block_selection_signal = False
        
        im_rect = QRectF(0, 0, shared.SCREEN_W, shared.SCREEN_H)
        self.baseLayer.setRect(im_rect)

        self.textlayer_trans_slider: QSlider = None
        self.originallayer_trans_slider: QSlider = None

    def on_switch_item(
        self,
        switch_delta: int,
        key_event: Optional[QKeyEvent] = None,
    ) -> None:
        if self.textEditMode():
            self.switch_text_item.emit(switch_delta, key_event)

    def img_window_size(self) -> QSize:
        if self.imgtrans_proj.inpainted_valid:
            return self.inpaintLayer.pixmap().size()
        return self.baseLayer.rect().size().toSize()

    def dragEnterEvent(self, e: QGraphicsSceneDragDropEvent):
        
        self.drop_folder = None
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            ufolder = None
            for url in urls:
                furl = url.toLocalFile()
                if os.path.isdir(furl):
                    ufolder = furl
                    break
            if ufolder is not None:
                e.acceptProposedAction()
                self.drop_folder = ufolder

    def dropEvent(self, event) -> None:
        if self.drop_folder is not None:
            self.drop_open_folder.emit(self.drop_folder)
            self.drop_folder = None
        return super().dropEvent(event)

    def textEditMode(self) -> bool:
        return self.editor_index == 1

    def drawMode(self) -> bool:
        return self.editor_index == 0

    def set_canvas_cursor(
        self,
        cursor: Union[QCursor, Qt.CursorShape],
    ) -> None:
        # Keep tool cursors in the scene hierarchy so child text and control
        # items can temporarily override them through Qt's native cursor rules.
        self.baseLayer.setCursor(cursor)

    def clear_canvas_cursor(self) -> None:
        if self.baseLayer.hasCursor():
            self.baseLayer.unsetCursor()

    def _restore_viewport_cursor(self) -> None:
        if self.gv.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.gv.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.gv.viewport().unsetCursor()

    def _clear_text_creation_cursor(self) -> None:
        self._text_creation_cursor_active = False
        self._restore_viewport_cursor()

    def scaleUp(self) -> None:
        self.scaleImage(1 + CANVAS_SCALE_SPEED)

    def scaleDown(self) -> None:
        self.scaleImage(1 - CANVAS_SCALE_SPEED)

    def scaleBy(self, value: float) -> None:
        self.scaleImage(value)

    def refresh_text_shape_control(self, *_args: object) -> None:
        # One view change can emit resize and both scrollbar signals.
        self._text_shape_refresh_timer.start(0)

    def _refresh_text_shape_control_now(self) -> None:
        self.txtblkShapeControl.requestGeometryRefresh()
        self.txtblkGridControl.requestGeometryRefresh()
        self.txtblkProjectiveControl.requestGeometryRefresh()

    def bind_text_grid_control(
        self,
        item: TextBlkItem,
        stack_index: int,
        **callbacks: Callable[..., None],
    ) -> None:
        if self._rubber_band_target == 'grid':
            self.hide_rubber_band()
        self.txtblkProjectiveControl.clear()
        if self.txtblkShapeControl.blk_item is not item:
            self.txtblkShapeControl.setBlkItem(item)
        self.txtblkGridControl.bind(
            item,
            stack_index,
            **callbacks,
        )
        self.txtblkShapeControl.hide()

    def bind_text_projective_control(
        self,
        item: TextBlkItem,
        stack_index: int,
        **callbacks: Callable[..., None],
    ) -> None:
        if self._rubber_band_target == 'grid':
            self.hide_rubber_band()
        self.txtblkGridControl.clear()
        if self.txtblkShapeControl.blk_item is not item:
            self.txtblkShapeControl.setBlkItem(item)
        self.txtblkProjectiveControl.bind(
            item,
            stack_index,
            **callbacks,
        )
        self.txtblkShapeControl.hide()

    def _restore_shape_after_transform_control(
        self,
        had_binding: bool,
    ) -> None:
        if not had_binding:
            return
        selected = self.selected_text_items()
        if len(selected) == 1:
            self.txtblkShapeControl.setBlkItem(selected[0])

    def clear_text_transform_controls(self) -> None:
        if self._rubber_band_target == 'grid':
            self.hide_rubber_band()
        had_binding = (
            self.txtblkGridControl.item is not None
            or self.txtblkProjectiveControl.item is not None
        )
        self.txtblkGridControl.clear()
        self.txtblkProjectiveControl.clear()
        self._restore_shape_after_transform_control(had_binding)

    def active_text_transform_control(
        self,
    ) -> Optional[
        Union[TextGridTransformControl, TextProjectiveTransformControl]
    ]:
        for control in (
            self.txtblkGridControl,
            self.txtblkProjectiveControl,
        ):
            if control.item is not None and control.isVisible():
                return control
        return None

    def active_transform_control_item(self) -> Optional[TextBlkItem]:
        control = self.active_text_transform_control()
        return None if control is None else control.item

    def _set_scene_scale(
        self,
        scale: float,
        refresh_control: bool = True,
    ) -> None:
        self.scale_factor = scale
        self.baseLayer.setScale(scale)
        self.setSceneRect(0, 0, self.baseLayer.sceneBoundingRect().width(), self.baseLayer.sceneBoundingRect().height())
        if refresh_control:
            self.refresh_text_shape_control()

    def render_result_img(self) -> QImage:
        self.inpaintLayer.hide()
        tlayer_opacity_before = self.textLayer.opacity()
        tlayer_visible = self.textLayer.isVisible()
        if tlayer_opacity_before != 1:
            self.textLayer.setOpacity(1)
        if not tlayer_visible:
            self.textLayer.show()
        scale_before = self.scale_factor
        if scale_before != 1:
            hb_pos = self.hscroll_bar.value()
            vb_pos = self.vscroll_bar.value()
            self._set_scene_scale(1.0, refresh_control=False)

        scene_items = self.items()
        control_visibility = {
            item: item.isVisible()
            for item in scene_items
            if bool(item.data(CONTROL_ITEM_DATA_KEY))
        }
        text_items = [
            item
            for item in scene_items
            if isinstance(item, TextBlkItem)
        ]
        export_effect_items = [
            item
            for item in text_items
            if not item._text_transform_is_neutral()
        ]
        enabled_export_effect_items = []
        painter = None
        try:
            for item in text_items:
                item.set_ui_guide_suppressed(True)
            if self.textEditMode() and self.txtblkShapeControl.blk_item is not None:
                blk_item = self.txtblkShapeControl.blk_item
                if blk_item.isEditing():
                    blk_item.endEdit(keep_focus=False)

            for item in control_visibility:
                item.hide()

            for item in export_effect_items:
                enabled_export_effect_items.append(item)
                item.set_export_effect_render(True)

            result = ndarray2pixmap(
                self.imgtrans_proj.inpainted_array, return_qimg=True
            )
            canvas_sz = self.img_window_size()
            painter = QPainter(result)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = QRectF(0, 0, canvas_sz.width(), canvas_sz.height())
            # Explicit source/target rectangles avoid the blurred #320 path.
            self.render(painter, rect, rect)
            for item in enabled_export_effect_items:
                if item.export_effect_error is not None:
                    raise item.export_effect_error
            return result
        finally:
            if painter is not None and painter.isActive():
                painter.end()
            for item in enabled_export_effect_items:
                item.set_export_effect_render(False)
            for item in text_items:
                if item.scene() is self:
                    item.set_ui_guide_suppressed(False)
            if scale_before != 1:
                self._set_scene_scale(scale_before, refresh_control=False)
                if self.hscroll_bar.value() != hb_pos:
                    self.hscroll_bar.setValue(hb_pos)
                if self.vscroll_bar.value() != vb_pos:
                    self.vscroll_bar.setValue(vb_pos)
            if tlayer_opacity_before != 1:
                self.textLayer.setOpacity(tlayer_opacity_before)
            if not tlayer_visible:
                self.textLayer.hide()
            self.inpaintLayer.show()
            for item, was_visible in control_visibility.items():
                if item.scene() is self:
                    item.setVisible(was_visible)
            self.refresh_text_shape_control()
    
    def updateLayers(self):
        
        if not self.imgtrans_proj.img_valid:
            return
        
        inpainted_as_base = self.imgtrans_proj.inpainted_valid
        
        if inpainted_as_base:
            self.base_pixmap = ndarray2pixmap(self.imgtrans_proj.inpainted_array)

        pixmap = self.base_pixmap.copy()
        painter = QPainter(pixmap)
        origin = QPoint(0, 0)

        if self.imgtrans_proj.img_valid and pcfg.original_transparency > 0:
            painter.setOpacity(pcfg.original_transparency)
            if inpainted_as_base:
                painter.drawPixmap(origin, ndarray2pixmap(self.imgtrans_proj.img_array))
            else:
                painter.drawPixmap(origin, pixmap)

        if self.imgtrans_proj.mask_valid and pcfg.mask_transparency > 0 and not self.textEditMode():
            painter.setOpacity(pcfg.mask_transparency)
            painter.drawPixmap(origin, ndarray2pixmap(self.imgtrans_proj.mask_array))

        painter.end()
        self.inpaintLayer.setPixmap(pixmap)

    def setMaskTransparency(self, transparency: float):
        pcfg.mask_transparency = transparency
        self.updateLayers()

    def setOriginalTransparency(self, transparency: float):
        pcfg.original_transparency = transparency
        self.updateLayers()

    def setTextLayerTransparency(self, transparency: float):
        self.textLayer.setOpacity(transparency)
        self.text_transparency = transparency

    def adjustScrollBar(self, scrollBar: QScrollBar, factor: float):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))

    def scaleImage(self, factor: float) -> None:
        if not self.gv.isVisible() or not self.imgtrans_proj.img_valid:
            return
        s_f = np.clip(
            self.scale_factor * factor,
            CANVAS_SCALE_MIN,
            CANVAS_SCALE_MAX,
        )

        scale_changed = self.scale_factor != s_f
        if scale_changed:
            # The guide is stored in scene coordinates while text items scale.
            self.cancel_path_reorder()
        self.scale_factor = s_f
        self.baseLayer.setScale(self.scale_factor)

        if scale_changed:
            self.adjustScrollBar(self.gv.horizontalScrollBar(), factor)
            self.adjustScrollBar(self.gv.verticalScrollBar(), factor)
            self.scalefactor_changed.emit()
        self.setSceneRect(
            0,
            0,
            self.baseLayer.sceneBoundingRect().width(),
            self.baseLayer.sceneBoundingRect().height(),
        )
        self.refresh_text_shape_control()

    def onViewResized(self) -> None:
        gv_w, gv_h = self.gv.geometry().width(), self.gv.geometry().height()

        x = gv_w - self.scaleFactorLabel.width()
        y = gv_h - self.scaleFactorLabel.height()
        pos_new = (QPointF(x, y) / 2).toPoint()
        if self.scaleFactorLabel.pos() != pos_new:
            self.scaleFactorLabel.move(pos_new)
        
        x = gv_w - self.search_widget.width()
        pos = self.search_widget.pos()
        pos.setX(x-30)
        self.search_widget.move(pos)
        self.refresh_text_shape_control()
        self.gv.viewport().update()
        
    def onScaleFactorChanged(self) -> None:
        self.scaleFactorLabel.setText(f'{self.scale_factor*100:2.0f}%')
        self.scaleFactorLabel.raise_()
        self.scaleFactorLabel.startFadeAnimation()

    def on_selection_changed(self) -> None:
        if self.txtblkShapeControl.isVisible():
            blk_item = self.txtblkShapeControl.blk_item
            if blk_item is not None and blk_item.isEditing():
                blk_item.endEdit()
        if self.block_selection_signal:
            return
        if self.hasFocus():
            self.incanvas_selection_changed.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()

        if self._path_reorder_active:
            if key == QKEY.Key_Escape:
                self.cancel_path_reorder()
            event.accept()
            return

        modifiers = event.modifiers()
        if self.handle_transform_modal_shortcut(key, modifiers):
            event.accept()
            return
        if (modifiers == Qt.KeyboardModifier.AltModifier) and \
            not key == QKEY.Key_Alt and \
                self.editing_textblkitem is None:
            if key in {QKEY.Key_W, QKEY.Key_A, QKEY.Key_Left, QKEY.Key_Up}:
                self.on_switch_item(-1, event)
                return
            elif key in {QKEY.Key_S, QKEY.Key_D, QKEY.Key_Right, QKEY.Key_Down}:
                self.on_switch_item(1, event)
                return

        if self.editing_textblkitem is not None:
            return super().keyPressEvent(event)
        # Single-letter canvas shortcuts must remain below live text editing.
        elif (
            modifiers == Qt.KeyboardModifier.NoModifier
            and key == QKEY.Key_S
            and self.start_projective_scale()
        ):
            event.accept()
            return
        elif (
            modifiers == Qt.KeyboardModifier.NoModifier
            and key == QKEY.Key_N
        ):
            self.set_order_badges_visible(not self.order_badges_visible)
            event.accept()
            return
        elif key in ARROWKEY2DIRECTION:
            sel_blkitems = self.selected_text_items()
            if len(sel_blkitems) > 0:
                direction = ARROWKEY2DIRECTION[key]
                cmd = MoveByKeyCommand(sel_blkitems, direction)
                self.push_undo_command(cmd)
                event.setAccepted(True)
                return
        elif key in QNUMERIC_KEYS:
            value = QNUMERIC_KEYS[key]
            self.set_active_layer_transparency(value * 10)
        return super().keyPressEvent(event)

    def handle_transform_modal_shortcut(
        self,
        key,
        modifiers=Qt.KeyboardModifier.NoModifier,
    ) -> bool:
        if (
            self.editing_textblkitem is not None
            or self.rubber_band_origin is not None
        ):
            return False
        control = self.active_text_transform_control()
        return (
            control is not None
            and not control.item.isEditing()
            and control.handle_shortcut(key, modifiers)
        )

    def start_projective_scale(self) -> bool:
        if (
            not self.textEditMode()
            or self.editing_textblkitem is not None
        ):
            return False
        selected = self.selected_text_items()
        if len(selected) != 1 or selected[0].isEditing():
            return False
        item = selected[0]
        self.projective_scale_requested.emit(item)
        control = self.txtblkProjectiveControl
        return (
            control.item is item
            and control.handle_shortcut(QKEY.Key_S)
        )
    
    def set_active_layer_transparency(self, value: int):
        if self.textEditMode():
            opacity = self.textLayer.opacity() * 100
            if value == 0 and opacity == 0:
                value = 100
            self.textlayer_trans_slider.setValue(value)
            self.originallayer_trans_slider.setValue(100 - value)
            self.updateLayers()

    def addStrokeImageItem(self, pos: QPointF, pen: QPen, erasing: bool = False):
        if self.stroke_img_item is not None:
            self.stroke_img_item.startNewPoint(pos)
        else:
            self.stroke_img_item = StrokeImgItem(pen, pos, self.img_window_size(), shape=self.painting_shape)
            if not erasing:
                self.stroke_img_item.setParentItem(self.baseLayer)
            else:
                self.erase_img_key = str(QDateTime.currentMSecsSinceEpoch())
                compose_mode = QPainter.CompositionMode.CompositionMode_DestinationOut
                self.drawingLayer.addQImage(0, 0, self.stroke_img_item._img, compose_mode, self.erase_img_key)

    def startCreateTextblock(
        self,
        pos: QPointF,
        hide_control: bool = False,
    ) -> None:
        pos = pos / self.scale_factor
        self.creating_textblock = True
        self._text_creation_cursor_active = self.textEditMode()
        self.create_block_origin = pos
        if self._text_creation_cursor_active:
            self.gv.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.txtblkShapeControl.setBlkItem(None)
        self.txtblkShapeControl.setPos(0, 0)
        self.txtblkShapeControl.setRotation(0)
        self.txtblkShapeControl.setRect(QRectF(pos, QSizeF(1, 1)))
        if hide_control:
            self.txtblkShapeControl.hideControls()
        self.txtblkShapeControl.show()

    def endCreateTextblock(self, btn: int = 0) -> bool:
        self.creating_textblock = False
        if self._text_creation_cursor_active:
            self._clear_text_creation_cursor()
        self.txtblkShapeControl.hide()
        textblk_created = False
        rect = self.txtblkShapeControl.rect()
        if self.creating_normal_rect:
            self.end_create_rect.emit(rect, btn)
            self.txtblkShapeControl.showControls()
        else:
            if rect.width() > 1 and rect.height() > 1:
                self.end_create_textblock.emit(rect)
                textblk_created = True
        return textblk_created

    @property
    def path_reorder_active(self) -> bool:
        return self._path_reorder_active

    def set_order_badges_visible(self, visible: bool) -> None:
        """Set the canvas-only reading-order badge visibility."""
        visible = bool(visible)
        if self.order_badges_visible == visible:
            return
        self.order_badges_visible = visible
        for item in self.textLayer.childItems():
            if isinstance(item, TextBlkItem):
                item.set_order_badge_visible(visible)

    def attach_text_item(self, item: TextBlkItem) -> None:
        """Attach a text item and its badge to their canvas-owned layers."""
        item.setParentItem(self.textLayer)
        item.set_order_badge_layer(self.orderBadgeLayer)
        item.set_order_badge_visible(self.order_badges_visible)

    def start_path_reorder(self) -> bool:
        """Start one path gesture that defines a new page reading order."""
        if self._path_reorder_active:
            return True
        if (
            self.editing_textblkitem is not None
            and self.editing_textblkitem.isEditing()
        ):
            return False
        items = sorted(
            (
                item
                for item in self.textLayer.childItems()
                if isinstance(item, TextBlkItem)
            ),
            key=lambda item: item.idx,
        )
        if len(items) < 2:
            return False

        self.clear_states()
        self.clear_text_transform_controls()
        self.txtblkShapeControl.setBlkItem(None)
        self._path_reorder_active = True
        self._path_reorder_drawing = False
        self._path_reorder_path = QPainterPath()
        self._path_reorder_items = items
        self._path_reorder_touched = []
        self._path_reorder_last_pos = None
        self._preview_path_reorder()

        pen = QPen(QColor(30, 147, 229, 180), 3)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        path_item = QGraphicsPathItem()
        path_item.setData(CONTROL_ITEM_DATA_KEY, True)
        path_item.setPen(pen)
        path_item.setZValue(200)
        self.addItem(path_item)
        self._path_reorder_path_item = path_item
        self.gv.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.gv.setFocus()
        self.path_reorder_mode_changed.emit(True)
        return True

    def cancel_path_reorder(self) -> None:
        """Discard the transient path and order preview."""
        if not self._path_reorder_active:
            return
        self._path_reorder_active = False
        self._path_reorder_drawing = False
        self._path_reorder_path = QPainterPath()
        self._path_reorder_last_pos = None
        for item in self._path_reorder_items:
            item.set_order_number_override(None)
        self._path_reorder_items = []
        self._path_reorder_touched = []

        path_item = self._path_reorder_path_item
        self._path_reorder_path_item = None
        if path_item is not None and path_item.scene() is self:
            super().removeItem(path_item)
        self._restore_viewport_cursor()
        self.path_reorder_mode_changed.emit(False)

    def _path_reorder_brush_width(self) -> float:
        origin = self.gv.mapToScene(QPoint(0, 0))
        edge = self.gv.mapToScene(QPoint(24, 0))
        return max(1.0, QLineF(origin, edge).length())

    def _preview_path_reorder(self) -> None:
        touched = set(self._path_reorder_touched)
        order = self._path_reorder_touched + [
            item for item in self._path_reorder_items if item not in touched
        ]
        for order_number, item in enumerate(order, 1):
            override = (
                order_number
                if not item.order_badge_visible
                or order_number != item.idx + 1
                else None
            )
            item.set_order_number_override(override)

    def _collect_path_reorder_hits(
        self,
        start: QPointF,
        end: QPointF,
    ) -> None:
        """Collect intersected items in travel order, including fast drags."""
        segment = QPainterPath(start)
        segment.lineTo(end)
        brush_width = self._path_reorder_brush_width()
        stroker = QPainterPathStroker()
        stroker.setWidth(brush_width)
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        hit_area = stroker.createStroke(segment)
        if start == end:
            radius = brush_width / 2
            hit_area.addEllipse(start, radius, radius)

        candidates = []
        touched = set(self._path_reorder_touched)
        for item in self.items(
            hit_area,
            Qt.ItemSelectionMode.IntersectsItemShape,
            Qt.SortOrder.DescendingOrder,
        ):
            if not isinstance(item, TextBlkItem) or item in touched:
                continue
            entry = _segment_rect_entry(
                start,
                end,
                item.sceneBoundingRect(),
                brush_width / 2,
            )
            candidates.append((entry, item.idx, item))

        for _distance, _idx, item in sorted(candidates, key=lambda hit: hit[:2]):
            self._path_reorder_touched.append(item)
        if candidates:
            self._preview_path_reorder()

    def _start_path_reorder_stroke(self, scene_pos: QPointF) -> None:
        self._path_reorder_drawing = True
        self._path_reorder_path = QPainterPath(scene_pos)
        self._path_reorder_last_pos = QPointF(scene_pos)
        self._collect_path_reorder_hits(scene_pos, scene_pos)
        if self._path_reorder_path_item is not None:
            self._path_reorder_path_item.setPath(self._path_reorder_path)

    def _extend_path_reorder_stroke(self, scene_pos: QPointF) -> None:
        last_pos = self._path_reorder_last_pos
        if last_pos is None:
            return
        self._path_reorder_path.lineTo(scene_pos)
        self._collect_path_reorder_hits(last_pos, scene_pos)
        self._path_reorder_last_pos = QPointF(scene_pos)
        if self._path_reorder_path_item is not None:
            self._path_reorder_path_item.setPath(self._path_reorder_path)

    def _finish_path_reorder_stroke(self) -> None:
        touched_ids = [item.idx for item in self._path_reorder_touched]
        self.cancel_path_reorder()
        if len(touched_ids) >= 2:
            self.path_reorder_finished.emit(touched_ids)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._path_reorder_drawing:
            self._extend_path_reorder_stroke(event.scenePos())
            event.accept()
            return
        control = self.active_text_transform_control()
        if control is not None and control.handle_modal_mouse_move(event):
            return
        if self._update_rubber_band(event.scenePos()):
            event.accept()
            return
        if self.mid_btn_pressed:
            new_pos = event.screenPos()
            delta_pos = new_pos - self.pan_initial_pos
            self.pan_initial_pos = new_pos
            self.hscroll_bar.setValue(int(self.hscroll_bar.value() - delta_pos.x()))
            self.vscroll_bar.setValue(int(self.vscroll_bar.value() - delta_pos.y()))
            
        elif self.creating_textblock:
            self.txtblkShapeControl.setRect(QRectF(self.create_block_origin, event.scenePos() / self.scale_factor).normalized())
        
        elif self.stroke_img_item is not None:
            if self.stroke_img_item.is_painting:
                pos = self.inpaintLayer.mapFromScene(event.scenePos())
                if self.erase_img_key is None:
                    # painting
                    self.stroke_img_item.lineTo(pos)
                else:
                    rect = self.stroke_img_item.lineTo(pos, update=False)
                    if rect is not None:
                        self.drawingLayer.update(rect)
        
        elif self.scale_tool_mode:
            self.scale_tool.emit(event.scenePos())
        
        result = super().mouseMoveEvent(event)
        if self._text_creation_cursor_active:
            # Creation is a modal drag, so it overrides child text cursors.
            self.gv.viewport().setCursor(Qt.CursorShape.CrossCursor)
        return result
    
    @property
    def scale_tool_mode(self):
        return self.drawMode() and self.gv.isVisible() and QApplication.keyboardModifiers() == Qt.KeyboardModifier.AltModifier

    def clearToolStates(self):
        self.cancel_path_reorder()
        self.end_scale_tool.emit()

    def selected_text_items(self, sort: bool = True) -> List[TextBlkItem]:
        sel_textitems = []
        selitems = self.selectedItems()
        for sel in selitems:
            if isinstance(sel, TextBlkItem):
                sel_textitems.append(sel)
        if sort:
            sel_textitems.sort(key = lambda x : x.idx)
        return sel_textitems

    def handle_ctrlv(self) -> bool:
        if not self.textEditMode():
            return False        
        if self.editing_textblkitem is not None and self.editing_textblkitem.isEditing():
            return False
        self.on_paste()
        return True

    def handle_ctrlc(self):
        if not self.textEditMode():
            return False        
        if self.editing_textblkitem is not None and self.editing_textblkitem.isEditing():
            return False
        self.on_copy()
        return True

    def scene_cursor_pos(self):
        origin = self.gv.mapFromGlobal(QCursor.pos())
        return self.gv.mapToScene(origin)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()
        if self._path_reorder_active:
            if btn == Qt.MouseButton.LeftButton:
                self._start_path_reorder_stroke(event.scenePos())
                event.accept()
                return
            if btn == Qt.MouseButton.RightButton:
                self.cancel_path_reorder()
                event.accept()
                return
        control = self.active_text_transform_control()
        if control is not None and control.handle_modal_mouse_press(event):
            return
        if (
            btn in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
            )
            and self._is_grid_rubber_origin(
                event.scenePos(),
                allow_handles=btn == Qt.MouseButton.RightButton,
            )
            and self._begin_rubber_band(
                event.scenePos(),
                event.modifiers(),
                btn,
                target='grid',
                on_finish=(
                    self.txtblkGridControl.select_handles_in_scene_rect
                ),
            )
        ):
            event.accept()
            return
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = True
            self.pan_initial_pos = event.screenPos()
            return
        
        if self.imgtrans_proj.img_valid:
            if self.textblock_mode and len(self.selectedItems()) == 0 and self.textEditMode():
                if btn == Qt.MouseButton.RightButton:
                    return self.startCreateTextblock(event.scenePos())
            elif self.creating_normal_rect:
                if btn == Qt.MouseButton.RightButton or btn == Qt.MouseButton.LeftButton:
                    return self.startCreateTextblock(event.scenePos(), hide_control=True)

            elif btn == Qt.MouseButton.LeftButton:
                # user is drawing using the pen/inpainting tool
                if self.scale_tool_mode:
                    self.begin_scale_tool.emit(event.scenePos())
                elif self.painting:
                    self.addStrokeImageItem(self.inpaintLayer.mapFromScene(event.scenePos()), self.painting_pen)

            elif btn == Qt.MouseButton.RightButton:
                # user is drawing using eraser
                if self.painting:
                    erasing = self.image_edit_mode == ImageEditMode.PenTool
                    self.addStrokeImageItem(self.inpaintLayer.mapFromScene(event.scenePos()), self.erasing_pen, erasing)
                else:   # rubber band selection
                    self._begin_rubber_band(
                        event.scenePos(),
                        event.modifiers(),
                        btn,
                        target='scene',
                        on_update=self._select_scene_items_in_rect,
                    )

        if btn == Qt.MouseButton.LeftButton and self.txtblkShapeControl.isVisible():
            items_at = self.items(event.scenePos())
            # Transform controllers own their outside-click gestures and
            # lifecycle; this path only dismisses the ordinary shape frame.
            if not any(
                isinstance(item, TextBlkItem)
                or item.data(CONTROL_ITEM_DATA_KEY)
                for item in items_at
            ):
                self.txtblkShapeControl.setBlkItem(None)

        return super().mousePressEvent(event)

    @property
    def creating_normal_rect(self):
        return self.image_edit_mode == ImageEditMode.RectTool and self.editor_index == 0

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()
        if self._path_reorder_drawing and btn == Qt.MouseButton.LeftButton:
            self._finish_path_reorder_stroke()
            event.accept()
            return
        control = self.active_text_transform_control()
        if control is not None and control.handle_modal_mouse_release(event):
            return
        rubber_target = self._rubber_band_target
        if self._finish_rubber_band(event.scenePos(), btn) and (
            rubber_target == 'grid'
        ):
            event.accept()
            return

        self.hide_rubber_band()

        Qt.MouseButton.LeftButton
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = False
        textblk_created = False
        if self.creating_textblock:
            tgt = 0 if btn == Qt.MouseButton.LeftButton else 1
            textblk_created = self.endCreateTextblock(btn=tgt)
        if btn == Qt.MouseButton.RightButton:
            if self.stroke_img_item is not None:
                self.finish_erasing.emit(self.stroke_img_item)
            if self.textEditMode() and not textblk_created:
                self.context_menu_requested.emit(event.screenPos(), False)
        if btn == Qt.MouseButton.LeftButton:
            if self.stroke_img_item is not None:
                self.finish_painting.emit(self.stroke_img_item)
            elif self.scale_tool_mode:
                self.end_scale_tool.emit()
        return super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._rubber_band_target == 'grid':
            self.hide_rubber_band()
        return super().mouseDoubleClickEvent(event)

    def _is_grid_rubber_origin(
        self,
        scene_pos: QPointF,
        *,
        allow_handles: bool = False,
    ) -> bool:
        control = self.txtblkGridControl
        if control.item is None or not control.isVisible():
            return False
        for item in self.items(scene_pos):
            if item in control.handles:
                if allow_handles:
                    continue
                return False
            if isinstance(item, TextBlkItem) and item is not control.item:
                return False
            if item.data(CONTROL_ITEM_DATA_KEY) and item is not control:
                return False
        return True

    def _begin_rubber_band(
        self,
        scene_pos,
        modifiers,
        button,
        *,
        target,
        on_update=None,
        on_finish=None,
    ) -> bool:
        if self.rubber_band_origin is not None:
            return False
        self.rubber_band_origin = QPointF(scene_pos)
        self.rubber_band_modifiers = modifiers
        self._rubber_band_button = button
        self._rubber_band_target = target
        self._rubber_band_update = on_update
        self._rubber_band_finish = on_finish
        self.rubber_band.setGeometry(QRectF(scene_pos, scene_pos))
        self.rubber_band.show()
        return True

    def _update_rubber_band(self, scene_pos) -> bool:
        if self.rubber_band_origin is None:
            return False
        rect = QRectF(self.rubber_band_origin, scene_pos).normalized()
        self.rubber_band.setGeometry(rect)
        if self._rubber_band_update is not None:
            self._rubber_band_update(rect, self.rubber_band_modifiers)
        return True

    def _finish_rubber_band(self, scene_pos, button) -> bool:
        if (
            self.rubber_band_origin is None
            or button != self._rubber_band_button
        ):
            return False
        rect = QRectF(self.rubber_band_origin, scene_pos).normalized()
        finish = self._rubber_band_finish
        modifiers = self.rubber_band_modifiers
        self.hide_rubber_band()
        if finish is not None:
            finish(rect, modifiers)
        return True

    def _select_scene_items_in_rect(self, rect, _modifiers) -> None:
        path = QPainterPath(rect.topLeft())
        path.addRect(rect)
        if shared.FLAG_QT6:
            self.setSelectionArea(
                path,
                deviceTransform=self.gv.viewportTransform(),
            )
        else:
            self.setSelectionArea(
                path,
                Qt.ItemSelectionMode.IntersectsItemBoundingRect,
                self.gv.viewportTransform(),
            )

    def updateCanvas(self) -> None:
        self.cancel_path_reorder()
        self.editing_textblkitem = None
        if self.stroke_img_item is not None:
            self.removeItem(self.stroke_img_item)
        self.erase_img_key = None
        self.txtblkShapeControl.setBlkItem(None)
        self.mid_btn_pressed = False
        self.search_widget.reInitialize()

        self.clearSelection()
        self.setProjSaveState(False)
        self.updateLayers()

        if self.base_pixmap is not None:
            pixmap = self.base_pixmap.copy()
            pixmap.fill(Qt.GlobalColor.transparent)
            self.textLayer.setPixmap(pixmap)

            im_rect = pixmap.rect()
            self.baseLayer.setRect(QRectF(im_rect))
            if im_rect != self.sceneRect():
                self.setSceneRect(0, 0, im_rect.width(), im_rect.height())
            self.scaleImage(1)

        self.setDrawingLayer()


    def setDrawingLayer(self, img: Union[QPixmap, np.ndarray] = None):
        
        self.drawingLayer.clearAllDrawings()

        if not self.imgtrans_proj.img_valid:
            return
        if img is None:
            drawing_map = self.inpaintLayer.pixmap().copy()
            drawing_map.fill(Qt.GlobalColor.transparent)
        elif not isinstance(img, QPixmap):
            drawing_map = ndarray2pixmap(img)
        else:
            drawing_map = img
        self.drawingLayer.setPixmap(drawing_map)

    def setPaintMode(self, painting: bool) -> None:
        self.cancel_path_reorder()
        if self.creating_textblock:
            self.clear_states()
        if painting:
            self.editing_textblkitem = None
            self.textblock_mode = False
        else:
            self.clear_canvas_cursor()
            self.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.image_edit_mode = ImageEditMode.NONE

    @property
    def painting(self):
        return self.image_edit_mode == ImageEditMode.PenTool or self.image_edit_mode == ImageEditMode.InpaintTool

    def setMaskTransparencyBySlider(self, slider_value: int):
        self.setMaskTransparency(slider_value / 100)

    def setOriginalTransparencyBySlider(self, slider_value: int):
        self.setOriginalTransparency(slider_value / 100)

    def setTextLayerTransparencyBySlider(self, slider_value: int):
        self.setTextLayerTransparency(slider_value / 100)

    def setTextBlockMode(self, mode: bool) -> None:
        self.cancel_path_reorder()
        self.textblock_mode = mode

    def on_create_contextmenu(
        self,
        pos: QPoint,
        is_textpanel: bool,
    ) -> None:
        if self.textEditMode() and not self.creating_textblock:
            editing_item = self.editing_textblkitem
            # The view disables native context menus and routes right releases here.
            if (
                not is_textpanel
                and editing_item is not None
                and editing_item.isEditing()
            ):
                editing_item.show_editing_context_menu(pos, self.gv)
                return

            menu = QMenu(self.gv)
            copy_act = menu.addAction(self.tr("Copy"))
            copy_act.setShortcut(QKeySequence.StandardKey.Copy)
            paste_act = menu.addAction(self.tr("Paste"))
            paste_act.setShortcut(QKeySequence.StandardKey.Paste)
            delete_act = menu.addAction(self.tr("Delete"))
            delete_act.setShortcut(QKeySequence("Ctrl+D"))
            copy_src_act = menu.addAction(self.tr("Copy source text"))
            copy_src_act.setShortcut(QKeySequence("Ctrl+Shift+C"))
            paste_src_act = menu.addAction(self.tr("Paste source text"))
            paste_src_act.setShortcut(QKeySequence("Ctrl+Shift+V"))
            delete_recover_act = menu.addAction(self.tr("Delete and Recover removed text"))
            delete_recover_act.setShortcut(QKeySequence("Ctrl+Shift+D"))

            menu.addSeparator()

            format_act = menu.addAction(self.tr("Apply font formatting"))
            layout_act = menu.addAction(self.tr("Auto layout"))
            angle_act = menu.addAction(self.tr("Reset Angle"))
            squeeze_act = menu.addAction(self.tr("Squeeze"))
            menu.addSeparator()
            translate_act = menu.addAction(self.tr("translate"))
            ocr_act = menu.addAction(self.tr("OCR"))
            ocr_translate_act = menu.addAction(self.tr("OCR and translate"))
            ocr_translate_inpaint_act = menu.addAction(self.tr("OCR, translate and inpaint"))
            inpaint_act = menu.addAction(self.tr("inpaint"))

            rst = menu.exec(pos)
            
            if rst == delete_act:
                self.delete_textblks.emit(0)
            elif rst == delete_recover_act:
                self.delete_textblks.emit(1)
            elif rst == copy_act:
                self.on_copy()
            elif rst == paste_act:
                self.on_paste()
            elif rst == copy_src_act:
                self.copy_src_signal.emit()
            elif rst == paste_src_act:
                self.paste_src_signal.emit()
            elif rst == format_act:
                self.format_textblks.emit()
            elif rst == layout_act:
                self.layout_textblks.emit()
            elif rst == angle_act:
                self.reset_angle.emit()
            elif rst == squeeze_act:
                self.squeeze_blk.emit()
            elif rst == translate_act:
                self.run_blktrans.emit(-1)
            elif rst == ocr_act:
                self.run_blktrans.emit(0)
            elif rst == ocr_translate_act:
                self.run_blktrans.emit(1)
            elif rst == ocr_translate_inpaint_act:
                self.run_blktrans.emit(2)
            elif rst == inpaint_act:
                self.run_blktrans.emit(3)

    @property
    def have_selected_blkitem(self):
        return len(self.selected_text_items()) > 0

    def on_paste(self, p: QPointF = None):
        if self.textEditMode():
            if p is None:
                p = self.scene_cursor_pos()
            if self.have_selected_blkitem:
                self.paste2selected_textitems.emit()
            else:
                self.paste_textblks.emit(p)

    def on_copy(self):
        if self.textEditMode():
            if self.have_selected_blkitem:
                self.copy_textblks.emit()

    def hide_rubber_band(self):
        self.rubber_band.hide()
        self.rubber_band_origin = None
        self.rubber_band_modifiers = Qt.KeyboardModifier.NoModifier
        self._rubber_band_button = Qt.MouseButton.NoButton
        self._rubber_band_target = None
        self._rubber_band_update = None
        self._rubber_band_finish = None
    
    def on_hide_canvas(self):
        self.clear_states()

    def on_activation_changed(self) -> None:
        self.clearToolStates()
        self.clear_states()
        for textitem in self.selected_text_items():
            if textitem.isEditing():
                self.editing_textblkitem = textitem

    def clear_states(self) -> None:
        self.hide_rubber_band()
        if self._text_creation_cursor_active:
            self._clear_text_creation_cursor()
        if self.creating_textblock:
            self.txtblkShapeControl.hide()
            self.txtblkShapeControl.showControls()
        self.creating_textblock = False
        self.create_block_origin = None
        self.editing_textblkitem = None
        self.gv.ctrl_pressed = False
        if self.stroke_img_item is not None:
            self.removeItem(self.stroke_img_item)

    def setProjSaveState(self, un_saved: bool):
        if un_saved == self.projstate_unsaved:
            return
        else:
            self.projstate_unsaved = un_saved
            self.proj_savestate_changed.emit(un_saved)

    def removeItem(self, item: QGraphicsItem) -> None:
        self.block_selection_signal = True
        if isinstance(item, TextBlkItem):
            # Rejoin the badge to its owner before both leave the scene.
            item.set_order_badge_layer(None)
        if isinstance(item, StrokeImgItem):
            # A stroke paints into its QImage until mouse release. Cleanup can
            # also happen first through activation, hiding, or a page change.
            item.finishPainting()
            if item is self.stroke_img_item and self.erase_img_key is not None:
                self.drawingLayer.removeQImage(self.erase_img_key)
        if item.scene() is self:
            super().removeItem(item)
        if isinstance(item, StrokeImgItem):
            item.setParentItem(None)
            self.stroke_img_item = None
            self.erase_img_key = None
        self.block_selection_signal = False

    def get_active_undostack(self) -> QUndoStack:
        if self.textEditMode():
            return self.text_undo_stack
        elif self.drawMode():
            return self.draw_undo_stack
        return None

    def push_undo_command(self, command: QUndoCommand, update_pushed_step=True):
        if self.textEditMode():
            self.push_text_command(command, update_pushed_step)
        elif self.drawMode():
            self.push_draw_command(command, update_pushed_step)
        else:
            return

    def push_draw_command(self, command: QUndoCommand, update_pushed_step=True):
        self.cancel_path_reorder()
        if command is not None:
            self.draw_undo_stack.push(command)
        if update_pushed_step:
            self.num_pushed_drawstep += 1
            self.on_drawstack_changed()

    def push_text_command(self, command: QUndoCommand, update_pushed_step=True):
        self.cancel_path_reorder()
        if command is not None:
            self.text_undo_stack.push(command)
        if update_pushed_step:
            self.num_pushed_textstep += 1
            self.on_textstack_changed()

    def on_drawstack_changed(self):
        if self.num_pushed_drawstep != self.saved_drawundo_step or self.num_pushed_textstep != self.saved_textundo_step:
            self.setProjSaveState(True)
        else:
            self.setProjSaveState(False)

    def on_textstack_changed(self):
        if self.num_pushed_textstep != self.saved_textundo_step or self.num_pushed_drawstep != self.saved_drawundo_step:
            self.setProjSaveState(True)
        else:
            self.setProjSaveState(False)
        self.textstack_changed.emit()

    def redo_textedit(self):
        self.cancel_path_reorder()
        self.num_pushed_textstep += 1
        self.text_undo_stack.redo()

    def undo_textedit(self):
        self.cancel_path_reorder()
        if self.num_pushed_textstep > 0:
            self.num_pushed_textstep -= 1
        self.text_undo_stack.undo()

    def redo(self):
        self.cancel_path_reorder()
        if self.textEditMode():
            undo_stack = self.text_undo_stack
            self.num_pushed_textstep += 1
            self.on_textstack_changed()
        elif self.drawMode():
            undo_stack = self.draw_undo_stack
            self.num_pushed_drawstep += 1
            self.on_drawstack_changed()
        else:
            return
        if undo_stack is not None:
            undo_stack.redo()

    def undo(self):
        self.cancel_path_reorder()
        if self.textEditMode():
            undo_stack = self.text_undo_stack
            if self.num_pushed_textstep > 0:
                self.num_pushed_textstep -= 1
            self.on_textstack_changed()
        elif self.drawMode():
            undo_stack = self.draw_undo_stack
            if self.num_pushed_drawstep > 0:
                self.num_pushed_drawstep -= 1
            self.on_drawstack_changed()
        else:
            return
        if undo_stack is not None:
            undo_stack.undo()

    def clear_undostack(self, update_saved_step=False):
        if update_saved_step:
            self.saved_drawundo_step = 0
            self.saved_textundo_step = 0
            self.num_pushed_textstep = 0
            self.num_pushed_drawstep = 0
        self.draw_undo_stack.clear()
        self.text_undo_stack.clear()

    def clear_text_stack(self):
        self.num_pushed_textstep = 0
        self.text_undo_stack.clear()

    def clear_draw_stack(self):
        self.num_pushed_drawstep = 0
        self.draw_undo_stack.clear()

    def update_saved_undostep(self):
        self.saved_drawundo_step = self.num_pushed_drawstep
        self.saved_textundo_step = self.num_pushed_textstep

    def text_change_unsaved(self) -> bool:
        return self.saved_textundo_step != self.num_pushed_textstep

    def draw_change_unsaved(self) -> bool:
        return self.saved_drawundo_step != self.num_pushed_drawstep

    def prepareClose(self):
        self.blockSignals(True)
        self.text_undo_stack.blockSignals(True)
        self.draw_undo_stack.blockSignals(True)

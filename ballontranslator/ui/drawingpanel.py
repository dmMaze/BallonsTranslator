from qtpy.QtCore import Signal, Qt, QPointF, QSize, QLineF, QDateTime, QRectF, QPoint
from qtpy.QtWidgets import QPushButton, QComboBox, QSizePolicy, QBoxLayout, QCheckBox, QHBoxLayout, QGraphicsView, QStackedWidget, QVBoxLayout, QLabel, QGraphicsPixmapItem, QGraphicsEllipseItem
from qtpy.QtGui import QPen, QColor, QCursor, QPainter, QPixmap, QBrush, QFontMetrics, QImage

try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from typing import Union, Tuple, List
import numpy as np
import cv2

from utils.imgproc_utils import enlarge_window
from utils.textblock_mask import canny_flood, connected_canny_flood
from utils.logger import logger

from .dl_manager import DLManager
from .image_edit import ImageEditMode, PixmapItem, DrawingLayer, StrokeImgItem
from .configpanel import InpaintConfigPanel
from .stylewidgets import Widget, SeparatorWidget, ColorPicker, PaintQSlider
from .canvas import Canvas
from .misc import DrawPanelConfig, ndarray2pixmap, pixmap2ndarray
from .constants import CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT

INPAINT_BRUSH_COLOR = QColor(127, 0, 127, 127)
MAX_PEN_SIZE = 1000
MIN_PEN_SIZE = 1
TOOLNAME_POINT_SIZE = 13

class DrawToolCheckBox(QCheckBox):
    checked = Signal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateChanged.connect(self.on_state_changed)

    def mousePressEvent(self, event) -> None:
        if self.isChecked():
            return
        return super().mousePressEvent(event)

    def on_state_changed(self, state: int) -> None:
        if self.isChecked():
            self.checked.emit()

class ToolNameLabel(QLabel):
    def __init__(self, fix_width=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setPointSizeF(TOOLNAME_POINT_SIZE)
        fmt = QFontMetrics(font)
        
        if fix_width is not None:
            self.setFixedWidth(fix_width)
            text_width = fmt.width(self.text())
            if text_width > fix_width * 0.95:
                font_size = TOOLNAME_POINT_SIZE * fix_width * 0.95 / text_width
                font.setPointSizeF(font_size)
        self.setFont(font)
            

class InpaintPanel(Widget):

    thicknessChanged = Signal(int)

    def __init__(self, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.inpainter_panel = inpainter_panel
        self.thicknessSlider = PaintQSlider(self.tr('pen thickness ') + 'value px')
        self.thicknessSlider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        
        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thicknessSlider)
        thickness_layout.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(thickness_layout)
        layout.setSpacing(14)
        self.vlayout = layout

    def on_thickness_changed(self):
        if self.thicknessSlider.hasFocus():
            self.thicknessChanged.emit(self.thicknessSlider.value())

    def showEvent(self, e) -> None:
        self.inpainter_panel.needInpaintChecker.setVisible(False)
        self.vlayout.addWidget(self.inpainter_panel)
        super().showEvent(e)


    def hideEvent(self, e) -> None:
        self.vlayout.removeWidget(self.inpainter_panel)
        self.inpainter_panel.needInpaintChecker.setVisible(True)
        return super().hideEvent(e)


class PenConfigPanel(Widget):
    thicknessChanged = Signal(int)
    colorChanged = Signal(list)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thicknessSlider = PaintQSlider(self.tr('pen thickness ') + 'value px')
        self.thicknessSlider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.alphaSlider = PaintQSlider(self.tr('alpha value'))
        self.alphaSlider.setRange(0, 255)
        self.alphaSlider.valueChanged.connect(self.on_alpha_changed)

        self.colorPicker = ColorPicker()
        self.colorPicker.colorChanged.connect(self.on_color_changed)
        
        color_label = ToolNameLabel(None, self.tr('Color'))
        alpha_label = ToolNameLabel(None, self.tr('Alpha'))
        color_layout = QHBoxLayout()
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.colorPicker)
        color_layout.addWidget(alpha_label)
        color_layout.addWidget(self.alphaSlider)
        
        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thicknessSlider)
        thickness_layout.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(color_layout)
        layout.addLayout(thickness_layout)
        layout.setSpacing(20)

    def on_thickness_changed(self):
        if self.thicknessSlider.hasFocus():
            self.thicknessChanged.emit(self.thicknessSlider.value())

    def on_alpha_changed(self):
        if self.alphaSlider.hasFocus():
            alpha = self.alphaSlider.value()
            color = self.colorPicker.rgba()
            color[-1] = alpha
            self.colorPicker.setPickerColor(color)
            self.colorChanged.emit(color)

    def on_color_changed(self):
        color = self.colorPicker.rgba()
        self.colorChanged.emit(color)


class RectPanel(Widget):
    method_changed = Signal(int)
    delete_btn_clicked = Signal()
    inpaint_btn_clicked = Signal()
    def __init__(self, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inpainter_panel = inpainter_panel
        self.methodComboBox = QComboBox()
        self.methodComboBox.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.methodComboBox.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.methodComboBox.addItems([self.tr('method 1'), self.tr('method 2')])
        self.autoChecker = QCheckBox(self.tr("Auto"))
        self.autoChecker.setToolTip(self.tr("run inpainting automatically."))
        self.autoChecker.stateChanged.connect(self.on_auto_changed)
        self.inpaint_btn = QPushButton(self.tr("Inpaint"))
        self.inpaint_btn.clicked.connect(self.inpaint_btn_clicked)
        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.clicked.connect(self.delete_btn_clicked)
        self.btnlayout = QHBoxLayout()
        self.btnlayout.addWidget(self.inpaint_btn)
        self.btnlayout.addWidget(self.delete_btn)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.methodComboBox)
        hlayout.addWidget(self.autoChecker)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(hlayout)
        layout.addLayout(self.btnlayout)
        layout.setSpacing(14)
        self.vlayout = layout

    def showEvent(self, e) -> None:
        self.inpainter_panel.needInpaintChecker.setVisible(False)
        self.vlayout.addWidget(self.inpainter_panel)
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        self.vlayout.removeWidget(self.inpainter_panel)
        self.inpainter_panel.needInpaintChecker.setVisible(True)
        return super().hideEvent(e)

    def get_maskseg_method(self):
        if self.methodComboBox.currentIndex() == 0:
            return canny_flood
        else:
            return connected_canny_flood

    def on_auto_changed(self):
        if self.autoChecker.isChecked():
            self.inpaint_btn.hide()
            self.delete_btn.hide()
        else:
            self.inpaint_btn.show()
            self.delete_btn.show()

    def auto(self) -> bool:
        return self.autoChecker.isChecked()


class DrawingPanel(Widget):

    scale_tool_pos: QPointF = None

    def __init__(self, canvas: Canvas, inpainter_panel: InpaintConfigPanel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dl_manager: DLManager = None
        self.canvas = canvas
        self.inpaint_stroke: StrokeImgItem = None
        self.rect_inpaint_dict: dict = None

        border_pen = QPen(INPAINT_BRUSH_COLOR, 3, Qt.PenStyle.DashLine)
        self.inpaint_mask_item: PixmapItem = PixmapItem(border_pen)
        self.scale_circle = QGraphicsEllipseItem()
        
        canvas.finish_painting.connect(self.on_finish_painting)
        canvas.finish_erasing.connect(self.on_finish_erasing)
        canvas.ctrl_relesed.connect(self.on_canvasctrl_released)
        canvas.begin_scale_tool.connect(self.on_begin_scale_tool)
        canvas.scale_tool.connect(self.on_scale_tool)
        canvas.end_scale_tool.connect(self.on_end_scale_tool)
        canvas.scalefactor_changed.connect(self.on_canvas_scalefactor_changed)
        canvas.end_create_rect.connect(self.on_end_create_rect)

        self.currentTool: DrawToolCheckBox = None
        self.handTool = DrawToolCheckBox()
        self.handTool.setObjectName("DrawHandTool")
        self.handTool.checked.connect(self.on_use_handtool)
        self.handTool.stateChanged.connect(self.on_handchecker_changed)
        self.inpaintTool = DrawToolCheckBox()
        self.inpaintTool.setObjectName("DrawInpaintTool")
        self.inpaintTool.checked.connect(self.on_use_inpainttool)
        self.inpaintConfigPanel = InpaintPanel(inpainter_panel)
        self.inpaintConfigPanel.thicknessChanged.connect(self.setInpaintToolWidth)

        self.rectTool = DrawToolCheckBox()
        self.rectTool.setObjectName("DrawRectTool")
        self.rectTool.checked.connect(self.on_use_rect_tool)
        self.rectTool.stateChanged.connect(self.on_rectchecker_changed)
        self.rectPanel = RectPanel(inpainter_panel)
        self.rectPanel.inpaint_btn_clicked.connect(self.on_rect_inpaintbtn_clicked)
        self.rectPanel.delete_btn_clicked.connect(self.on_rect_deletebtn_clicked)
        
        self.penTool = DrawToolCheckBox()
        self.penTool.setObjectName("DrawPenTool")
        self.penTool.checked.connect(self.on_use_pentool)
        self.penConfigPanel = PenConfigPanel()
        self.penConfigPanel.thicknessChanged.connect(self.setPenToolWidth)
        self.penConfigPanel.colorChanged.connect(self.setPenToolColor)

        toolboxlayout = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        toolboxlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        toolboxlayout.addWidget(self.handTool)
        toolboxlayout.addWidget(self.inpaintTool)
        toolboxlayout.addWidget(self.penTool)
        toolboxlayout.addWidget(self.rectTool)

        self.canvas.painting_pen = self.pentool_pen = \
            QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.canvas.erasing_pen = self.erasing_pen = QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.inpaint_pen = QPen(INPAINT_BRUSH_COLOR, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        
        self.setPenToolWidth(10)
        self.setPenToolColor([0, 0, 0, 127])

        self.toolConfigStackwidget = QStackedWidget()
        self.toolConfigStackwidget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.toolConfigStackwidget.addWidget(self.inpaintConfigPanel)
        self.toolConfigStackwidget.addWidget(self.penConfigPanel)
        self.toolConfigStackwidget.addWidget(self.rectPanel)

        self.maskTransperancySlider = PaintQSlider(' value%')
        self.maskTransperancySlider.valueChanged.connect(self.canvas.setMaskTransparencyBySlider)
        masklayout = QHBoxLayout()
        masklayout.addWidget(ToolNameLabel(220, self.tr('Mask Transparency')))
        masklayout.addWidget(self.maskTransperancySlider)

        layout = QVBoxLayout(self)
        layout.addLayout(toolboxlayout)
        layout.addWidget(SeparatorWidget())
        layout.addWidget(self.toolConfigStackwidget)
        layout.addWidget(SeparatorWidget())
        layout.addLayout(masklayout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def initDLModule(self, dl_manager: DLManager):
        self.dl_manager = dl_manager
        dl_manager.canvas_inpaint_finished.connect(self.on_inpaint_finished)
        dl_manager.inpaint_thread.exception_occurred.connect(self.on_inpaint_failed)

    def setInpaintToolWidth(self, width):
        self.inpaint_pen.setWidth(width)
        if self.isVisible():
            self.setInpaintCursor()

    def setPenToolWidth(self, width):
        self.pentool_pen.setWidthF(width)
        self.erasing_pen.setWidthF(width)
        if self.isVisible():
            self.setPenCursor()

    def setPenToolColor(self, color: Union[QColor, Tuple, List]):
        if not isinstance(color, QColor):
            color = QColor(*color)
        self.pentool_pen.setColor(color)
        if self.isVisible():
            self.setPenCursor()

    def on_use_handtool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.handTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.handTool
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.canvas.image_edit_mode = ImageEditMode.HandTool

    def on_use_inpainttool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.inpaintTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.inpaintTool
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool
        self.canvas.painting_pen = self.inpaint_pen
        self.canvas.erasing_pen = self.inpaint_pen
        self.toolConfigStackwidget.setCurrentWidget(self.inpaintConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setInpaintCursor()

    def on_use_pentool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.penTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.penTool
        self.canvas.painting_pen = self.pentool_pen
        self.canvas.erasing_pen = self.erasing_pen
        self.canvas.image_edit_mode = ImageEditMode.PenTool
        self.toolConfigStackwidget.setCurrentWidget(self.penConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setPenCursor()

    def on_use_rect_tool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.rectTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.rectTool
        self.toolConfigStackwidget.setCurrentWidget(self.rectPanel)
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.canvas.image_edit_mode = ImageEditMode.RectTool
        self.setCrossCursor()

    def get_config(self) -> DrawPanelConfig:
        config = DrawPanelConfig()
        pc = self.pentool_pen.color()
        config.pentool_color = [pc.red(), pc.green(), pc.blue(), pc.alpha()]
        config.pentool_width = self.pentool_pen.widthF()
        config.inpainter_width = self.inpaint_pen.widthF()
        if self.currentTool == self.handTool:
            config.current_tool = ImageEditMode.HandTool
        elif self.currentTool == self.inpaintTool:
            config.current_tool = ImageEditMode.InpaintTool
        elif self.currentTool == self.penTool:
            config.current_tool = ImageEditMode.PenTool
        elif self.currentTool == self.rectTool:
            config.current_tool = ImageEditMode.RectTool
        config.rectool_auto = self.rectPanel.autoChecker.isChecked()
        config.rectool_method = self.rectPanel.methodComboBox.currentIndex()
        return config

    def set_config(self, config: DrawPanelConfig):
        self.setPenToolWidth(config.pentool_width)
        self.penConfigPanel.thicknessSlider.setValue(config.pentool_width)
        self.setInpaintToolWidth(config.inpainter_width)
        self.inpaintConfigPanel.thicknessSlider.setValue(config.inpainter_width)
        self.setPenToolColor(config.pentool_color)
        self.penConfigPanel.colorPicker.setPickerColor(config.pentool_color)

        self.rectPanel.autoChecker.setChecked(config.rectool_auto)
        self.rectPanel.methodComboBox.setCurrentIndex(config.rectool_method)
        if config.current_tool == ImageEditMode.HandTool:
            self.handTool.setChecked(True)
        elif config.current_tool == ImageEditMode.InpaintTool:
            self.inpaintTool.setChecked(True)
        elif config.current_tool == ImageEditMode.PenTool:
            self.penTool.setChecked(True)
        elif config.current_tool == ImageEditMode.RectTool:
            self.rectTool.setChecked(True)

    def get_pen_cursor(self, pen_color: QColor = None, pen_size = None, draw_circle=True) -> QCursor:
        cross_size = 31
        cross_len = cross_size // 4
        thickness = 3
        if pen_color is None:
            pen_color = self.pentool_pen.color()
        if pen_size is None:
            pen_size = self.pentool_pen.width()
        pen_size *= self.canvas.scale_factor
        map_size = max(cross_size+7, pen_size)
        cursor_center = map_size // 2
        pen_radius = pen_size // 2
        pen_color.setAlpha(127)
        pen = QPen(pen_color, thickness, Qt.PenStyle.DotLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        pen.setDashPattern([3, 6])
        if pen_size < 20:
            pen.setStyle(Qt.PenStyle.SolidLine)

        cur_pixmap = QPixmap(QSize(map_size, map_size))
        cur_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(cur_pixmap)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if draw_circle:
            painter.drawEllipse(cursor_center-pen_radius + thickness, 
                                cursor_center-pen_radius + thickness, 
                                pen_size - 2*thickness, 
                                pen_size - 2*thickness)
        cross_left = (map_size  - 1 - cross_size) // 2 
        cross_right = map_size - cross_left

        pen = QPen(Qt.GlobalColor.white, 5, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        cross_hline0 = QLineF(cross_left, cursor_center, cross_left+cross_len, cursor_center)
        cross_hline1 = QLineF(cross_right-cross_len, cursor_center, cross_right, cursor_center)
        cross_vline0 = QLineF(cursor_center, cross_left, cursor_center, cross_left+cross_len)
        cross_vline1 = QLineF(cursor_center, cross_right-cross_len, cursor_center, cross_right)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        pen.setWidth(3)
        pen.setColor(Qt.GlobalColor.black)
        painter.setPen(pen)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        painter.end()
        return QCursor(cur_pixmap)

    def on_incre_pensize(self):
        self.scalePen(1.1)

    def on_decre_pensize(self):
        self.scalePen(0.9)
        pass

    def scalePen(self, scale_factor):
        if self.currentTool == self.penTool:
            val = self.pentool_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            self.penConfigPanel.thicknessSlider.setValue(new_val)
            self.setPenToolWidth(self.penConfigPanel.thicknessSlider.value())

        elif self.currentTool == self.inpaintTool:
            val = self.inpaint_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            self.inpaintConfigPanel.thicknessSlider.setValue(new_val)
            self.setInpaintToolWidth(self.inpaintConfigPanel.thicknessSlider.value())

    def showEvent(self, event) -> None:
        if self.currentTool is not None:
            self.currentTool.setChecked(False)
            self.currentTool.setChecked(True)
        return super().showEvent(event)

    def on_finish_painting(self, stroke_item: StrokeImgItem):
        stroke_item.finishPainting()
        if not self.canvas.imgtrans_proj.img_valid:
            self.canvas.removeItem(stroke_item)
            return
        if self.currentTool == self.penTool:
            rect, _, qimg = stroke_item.clip()
            if rect is not None:
                self.canvas.undoStack.push(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg))
            self.canvas.removeItem(stroke_item)
        elif self.currentTool == self.inpaintTool:
            self.inpaint_stroke = stroke_item
            if self.canvas.gv.ctrl_pressed:
                return
            else:
                self.runInpaint()

    def on_finish_erasing(self, stroke_item: StrokeImgItem):
        stroke_item.finishPainting()
        # inpainted-erasing logic is essentially the same as inpainting
        if self.currentTool == self.inpaintTool:
            rect, mask, _ = stroke_item.clip(mask_only=True)
            if mask is None:
                self.canvas.removeItem(stroke_item)
                return
            mask = 255 - mask
            mask_h, mask_w = mask.shape[:2]
            mask_x, mask_y = rect[0], rect[1]
            inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
            origin = self.canvas.imgtrans_proj.img_array
            origin = origin[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpainted = self.canvas.imgtrans_proj.inpainted_array
            inpainted = inpainted[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpaint_mask = self.canvas.imgtrans_proj.mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            # no inpainted need to be erased
            if inpaint_mask.sum() == 0:
                return
            mask = cv2.bitwise_and(mask, inpaint_mask)
            inpaint_mask = np.zeros_like(inpainted)
            inpaint_mask[mask > 0] = 1
            erased_img = inpaint_mask * inpainted + (1 - inpaint_mask) * origin
            self.canvas.undoStack.push(InpaintUndoCommand(self.canvas, erased_img, mask, inpaint_rect))
            self.canvas.removeItem(stroke_item)

        elif self.currentTool == self.penTool:
            rect, _, qimg = stroke_item.clip()
            if self.canvas.erase_img_key is not None:
                self.canvas.drawingLayer.removeQImage(self.canvas.erase_img_key)
                self.canvas.erase_img_key = None
                self.canvas.stroke_img_item = None
            if rect is not None:
                self.canvas.undoStack.push(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg, True))
        

    def runInpaint(self, inpaint_dict=None):

        if inpaint_dict is None:
            if self.inpaint_stroke is None:
                return
            elif self.inpaint_stroke.parentItem() is None:
                logger.warning("inpainting goes wrong")
                self.clearInpaintItems()
                return
                
            rect, mask, _ = self.inpaint_stroke.clip(mask_only=True)
            if mask is None:
                self.clearInpaintItems()
                return
            # we need to enlarge the mask window a bit to get better results
            mask_h, mask_w = mask.shape[:2]
            mask_x, mask_y = rect[0], rect[1]
            img = self.canvas.imgtrans_proj.inpainted_array
            inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
            rect_enlarged = enlarge_window(inpaint_rect, img.shape[1], img.shape[0])
            top = mask_y - rect_enlarged[1]
            bottom = rect_enlarged[3] - inpaint_rect[3]
            left = mask_x - rect_enlarged[0]
            right = rect_enlarged[2] - inpaint_rect[2]

            # print('inpaint_rect: ', inpaint_rect, 'enlarged: ', rect_enlarged, 'ltrb: ', left, top, right, bottom)
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            inpaint_rect = rect_enlarged
            img = img[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpaint_dict = {'img': img, 'mask': mask, 'inpaint_rect': inpaint_rect}

        self.canvas.image_edit_mode = ImageEditMode.NONE
        self.dl_manager.canvas_inpaint(inpaint_dict)

    def on_inpaint_finished(self, inpaint_dict):
        inpainted = inpaint_dict['inpainted']
        inpaint_rect = inpaint_dict['inpaint_rect']
        mask_array = self.canvas.imgtrans_proj.mask_array
        mask = cv2.bitwise_or(inpaint_dict['mask'], mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]])
        self.canvas.undoStack.push(InpaintUndoCommand(self.canvas, inpainted, mask, inpaint_rect))
        self.clearInpaintItems()

    def on_inpaint_failed(self):
        if self.currentTool == self.inpaintTool and self.inpaint_stroke is not None:
            self.clearInpaintItems()

    def on_canvasctrl_released(self):
        if self.isVisible() and self.currentTool == self.inpaintTool:
            self.runInpaint()

    def on_begin_scale_tool(self, pos: QPointF):
        
        if self.currentTool == self.penTool:
            circle_pen = QPen(self.pentool_pen)
        elif self.currentTool == self.inpaintTool:
            circle_pen = QPen(self.inpaint_pen)
        else:
            return
        pen_radius = circle_pen.widthF() / 2 * self.canvas.scale_factor
        
        r, g, b, a = circle_pen.color().getRgb()

        circle_pen.setWidth(3)
        circle_pen.setStyle(Qt.PenStyle.DashLine)
        circle_pen.setDashPattern([3, 6])
        self.scale_circle.setPen(circle_pen)
        self.scale_circle.setBrush(QBrush(QColor(r, g, b, 127)))
        self.scale_circle.setPos(pos - QPointF(pen_radius, pen_radius))
        pen_size = 2 * pen_radius
        self.scale_circle.setRect(0, 0, pen_size, pen_size)
        self.scale_tool_pos = pos - QPointF(pen_size, pen_size)
        self.canvas.addItem(self.scale_circle)
        self.setCrossCursor()
        
    def setCrossCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor(draw_circle=False))

    def on_scale_tool(self, pos: QPointF):
        if self.scale_tool_pos is None:
            return
        radius = pos.x() - self.scale_tool_pos.x()
        radius = max(min(radius, MAX_PEN_SIZE * self.canvas.scale_factor), MIN_PEN_SIZE * self.canvas.scale_factor)
        self.scale_circle.setRect(0, 0, radius, radius)

    def on_end_scale_tool(self):
        circle_size = self.scale_circle.rect().width() / self.canvas.scale_factor
        self.scale_tool_pos = None
        self.canvas.removeItem(self.scale_circle)

        if self.currentTool == self.penTool:
            self.setPenToolWidth(circle_size)
            self.penConfigPanel.thicknessSlider.setValue(circle_size)
            self.setPenCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintToolWidth(circle_size)
            self.inpaintConfigPanel.thicknessSlider.setValue(circle_size)
            self.setInpaintCursor()

    def on_canvas_scalefactor_changed(self):
        if not self.isVisible():
            return
        if self.currentTool == self.penTool:
            self.setPenCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintCursor()

    def setPenCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor())

    def setInpaintCursor(self):
        self.canvas.gv.setCursor(self.get_pen_cursor(INPAINT_BRUSH_COLOR, self.inpaint_pen.width()))

    def on_handchecker_changed(self):
        if self.handTool.isChecked():
            self.toolConfigStackwidget.hide()
        else:
            self.toolConfigStackwidget.show()

    def on_end_create_rect(self, rect: QRectF, mode: int):
        if self.currentTool == self.rectTool:
            self.canvas.image_edit_mode = ImageEditMode.NONE
            img = self.canvas.imgtrans_proj.inpainted_array
            im_h, im_w = img.shape[:2]

            xyxy = [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]
            xyxy = np.array(xyxy)
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, im_w - 1)
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, im_h - 1)
            x1, y1, x2, y2 = xyxy.astype(np.int64)
            if y2 - y1 < 2 or x2 - x1 < 2:
                self.canvas.image_edit_mode = ImageEditMode.RectTool
                return
            if mode == 0:
                im = np.copy(img[y1: y2, x1: x2])
                maskseg_method = self.rectPanel.get_maskseg_method()
                mask, ballon_mask, bub_dict = maskseg_method(im)
                bground_bgr = bub_dict['bground_bgr']
                need_inpaint = bub_dict['need_inpaint']

                inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2]}
                inpaint_dict['need_inpaint'] = need_inpaint
                inpaint_dict['bground_bgr'] = bground_bgr
                inpaint_dict['ballon_mask'] = ballon_mask
                user_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                user_mask[:, :, [0, 2, 3]] = (mask[:, :, np.newaxis] / 2).astype(np.uint8)
                self.inpaint_mask_item.setPixmap(ndarray2pixmap(user_mask))
                self.inpaint_mask_item.setParentItem(self.canvas.baseLayer)
                self.inpaint_mask_item.setPos(x1, y1)
                if self.rectPanel.auto():
                    self.inpaintRect(inpaint_dict)
                else:
                    self.rect_inpaint_dict = inpaint_dict
            else:
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                erased = self.canvas.imgtrans_proj.img_array[y1: y2, x1: x2]
                self.canvas.undoStack.push(InpaintUndoCommand(self.canvas, erased, mask, [x1, y1, x2, y2]))
                self.canvas.image_edit_mode = ImageEditMode.RectTool
            self.setCrossCursor()

    def inpaintRect(self, inpaint_dict):
        img = inpaint_dict['img']
        mask = inpaint_dict['mask']
        need_inpaint = inpaint_dict['need_inpaint']
        bground_bgr = inpaint_dict['bground_bgr']
        ballon_mask = inpaint_dict['ballon_mask']
        if not need_inpaint and self.dl_manager.dl_config.check_need_inpaint:
            img[np.where(ballon_mask > 0)] = bground_bgr
            self.canvas.undoStack.push(InpaintUndoCommand(self.canvas, img, mask, inpaint_dict['inpaint_rect']))
            self.clearInpaintItems()
        else:
            self.runInpaint(inpaint_dict=inpaint_dict)

    def on_rect_inpaintbtn_clicked(self):
        if self.rect_inpaint_dict is not None:
            self.inpaintRect(self.rect_inpaint_dict)

    def on_rect_deletebtn_clicked(self):
        self.clearInpaintItems()

    def on_rectchecker_changed(self):
        if not self.rectTool.isChecked():
            self.clearInpaintItems()

    def hideEvent(self, e) -> None:
        self.clearInpaintItems()
        return super().hideEvent(e)

    def clearInpaintItems(self):

        self.rect_inpaint_dict = None
        if self.inpaint_mask_item is not None:
            if self.inpaint_mask_item.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_mask_item)
            if self.rectTool.isChecked():
                self.canvas.image_edit_mode = ImageEditMode.RectTool    
            
        if self.inpaint_stroke is not None:
            if self.inpaint_stroke.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_stroke)
            self.inpaint_stroke = None
            if self.inpaintTool.isChecked():
                self.canvas.image_edit_mode = ImageEditMode.InpaintTool

class StrokeItemUndoCommand(QUndoCommand):
    def __init__(self, target_layer: DrawingLayer, rect: Tuple[int], qimg: QImage, erasing=False):
        super().__init__()
        self.qimg = qimg
        self.x = rect[0]
        self.y = rect[1]
        self.target_layer = target_layer
        self.key = str(QDateTime.currentMSecsSinceEpoch())
        if erasing:
            self.compose_mode = QPainter.CompositionMode.CompositionMode_DestinationOut
        else:
            self.compose_mode = QPainter.CompositionMode.CompositionMode_SourceOver
        
    def undo(self):
        if self.qimg is not None:
            self.target_layer.removeQImage(self.key)
            self.target_layer.update()
        # self.stroke_pixmap.hide()

    def redo(self):
        if self.qimg is not None:
            self.target_layer.addQImage(self.x, self.y, self.qimg, self.compose_mode, self.key)
            self.target_layer.scene().update()


class InpaintUndoCommand(QUndoCommand):
    def __init__(self, canvas: Canvas, inpainted: np.ndarray, mask: np.ndarray, inpaint_rect: List[int]):
        super().__init__()
        self.canvas = canvas
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        self.undo_img = np.copy(img_view)
        self.undo_mask = np.copy(mask_view)
        self.redo_img = inpainted
        self.redo_mask = mask
        self.inpaint_rect = inpaint_rect

    def redo(self) -> None:
        inpaint_rect = self.inpaint_rect
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        img_view[:] = self.redo_img
        mask_view[:] = self.redo_mask
        self.canvas.setInpaintLayer()
        self.canvas.setMaskLayer()

    def undo(self) -> None:
        inpaint_rect = self.inpaint_rect
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        img_view[:] = self.undo_img
        mask_view[:] = self.undo_mask
        self.canvas.setInpaintLayer()
        self.canvas.setMaskLayer()


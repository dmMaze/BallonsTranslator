import math

import numpy as np
import copy
from qtpy.QtWidgets import QGraphicsPixmapItem, QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QLabel, QStyleOptionGraphicsItem, QGraphicsSceneMouseEvent, QGraphicsRectItem
from qtpy.QtCore import Qt, QRect, QRectF, QPointF, QPoint
from qtpy.QtGui import QPainter, QPen, QColor

from utils.imgproc_utils import xywh2xyxypoly, rotate_polygons
from .cursor import rotateCursorList, resizeCursorList
from .textitem import TextBlkItem

CBEDGE_WIDTH = 30

VISUALIZE_HITBOX = False
ctrlidx_to_hitbox = np.array([
    [-0.75, -0.75, 0.75, 0.75],
    [-0.5, -0.75, 1, 0.75],
    [0., -0.75, 0.75, 0.75],
    [0., -0.5, 0.75, 1],
    [0., 0., 0.75, 0.75],
    [-0.5, 0., 1, 0.75],
    [-0.75, 0., 0.75, 0.75],
    [-0.75, -0.5, 0.75, 1]
], dtype=np.float32)

ctrlidx_to_visiblebox = np.array([
    [0.25, 0.25],
    [0.25, 0.25],
    [0., 0.25],
    [0., 0.25],
    [0., 0.],
    [0.25, 0.],
    [0.25, 0.],
    [0.25, 0.25]
], dtype=np.float32)

class ControlBlockItem(QGraphicsRectItem):
    DRAG_NONE = 0
    DRAG_RESHAPE = 1
    DRAG_ROTATE = 2
    DRAG_WARP = 3
    CURSOR_IDX = -1
    def __init__(self, parent, idx: int):
        super().__init__(parent)
        self.idx = idx
        self.ctrl: TextBlkShapeControl = parent
        self.edge_width = 0
        self.drag_mode = self.DRAG_NONE
        self.setAcceptHoverEvents(True)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(3 if (self.idx % 2 == 0) else 2)
        self.updateEdgeWidth(CBEDGE_WIDTH)

    def updateEdgeWidth(self, edge_width: float):
        self.edge_width = edge_width
        self.visible_len = self.edge_width / 2
        self.block_shift_value = self.edge_width * 0.75
        self.pen_width = edge_width / CBEDGE_WIDTH * 2 
        offset = self.edge_width * ctrlidx_to_visiblebox[self.idx]
        self.visible_rect = QRectF(offset[0], offset[1], self.visible_len, self.visible_len)
        hitbox = ctrlidx_to_hitbox[self.idx]
        w = hitbox[2] * self.edge_width
        h = hitbox[3] * self.edge_width
        self.setRect(0, 0, w, h)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        rect = QRectF(self.visible_rect)
        rect.setTopLeft(self.boundingRect().topLeft()+rect.topLeft())
        painter.setPen(QPen(QColor(75, 75, 75), self.pen_width, Qt.PenStyle.SolidLine, Qt.SquareCap))
        painter.fillRect(rect, QColor(200, 200, 200, 125))
        painter.drawRect(rect)
        if VISUALIZE_HITBOX:
            painter.setPen(QPen(QColor(75, 125, 0), self.pen_width, Qt.PenStyle.SolidLine, Qt.SquareCap))
            painter.drawRect(self.boundingRect())

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:        
        return super().hoverEnterEvent(event)

    # def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
    #     self.drag_mode = self.DRAG_NONE
    #     self.CURSOR_IDX = -1
    #     return super().hoverLeaveEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        angle = self.ctrl.rotation() + 45 * self.idx
        idx = self.get_angle_idx(angle)
        if self.visible_rect.contains(event.pos()):
            self.setCursor(resizeCursorList[idx % 4])
        else:
            self.setCursor(rotateCursorList[idx])
        self.CURSOR_IDX = idx
        return super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        if self.drag_mode == self.DRAG_NONE:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        return super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.ctrl.ctrlblockPressed()
        if event.button() == Qt.MouseButton.LeftButton and self.ctrl.blk_item is not None:
            blk_item = self.ctrl.blk_item
            blk_item.setSelected(True)
            if self.ctrl.warp_editing and getattr(blk_item.blk, 'warp_mode', 'none') in {'quad', 'mesh'}:
                self.drag_mode = self.DRAG_WARP
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                self.ctrl.beginWarpEdit()
                event.accept()
                return
            if self.visible_rect.contains(event.pos()):
                self.ctrl.reshaping = True
                self.drag_mode = self.DRAG_RESHAPE
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
                blk_item.startReshape()
            else:
                self.drag_mode = self.DRAG_ROTATE
                self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                preview = self.ctrl.previewPixmap

                preview.setPixmap(blk_item.toPixmap().copy(blk_item.unpadRect(blk_item.boundingRect()).toRect()))
                preview.setOpacity(0.7)
                preview.setVisible(True)
                rotate_vec = event.scenePos() - self.ctrl.sceneBoundingRect().center()
                self.updateAngleLabelPos()
                rotation = np.rad2deg(math.atan2(rotate_vec.y(), rotate_vec.x()))
                self.rotate_start = - rotation + self.ctrl.rotation() 
        event.accept()

    def updateAngleLabelPos(self):
        angleLabel = self.ctrl.angleLabel
        sp = self.scenePos()
        gv = angleLabel.parent()
        pos = gv.mapFromScene(sp)
        x = max(min(pos.x(), gv.width() - angleLabel.width()), 0)
        y = max(min(pos.y(), gv.height() - angleLabel.height()), 0)
        angleLabel.move(QPoint(x, y))
        angleLabel.setText("{:.1f}°".format(self.ctrl.rotation()))
        if not angleLabel.isVisible():
            angleLabel.setVisible(True)
            angleLabel.raise_()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)
        blk_item = self.ctrl.blk_item
        if blk_item is None:
            return
        if self.drag_mode == self.DRAG_WARP:
            lp = self.ctrl.mapFromScene(event.scenePos())
            self.ctrl.updateWarpFromLocal(self.idx, lp)
            return
        if self.drag_mode == self.DRAG_RESHAPE:    
            block_group = self.ctrl.ctrlblock_group
            crect = self.ctrl.rect()
            pos_x, pos_y = 0, 0
            opposite_block = block_group[(self.idx + 4) % 8 ]
            oppo_pos = opposite_block.pos()
            if self.idx % 2 == 0:
                if self.idx == 0:
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setX(pos_x + self.block_shift_value)
                    crect.setY(pos_y + self.block_shift_value)
                elif self.idx == 2:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setWidth(pos_x - oppo_pos.x() - self.block_shift_value)
                    crect.setY(pos_y+self.block_shift_value)
                elif self.idx == 4:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setWidth(pos_x-oppo_pos.x() - self.block_shift_value)
                    crect.setHeight(pos_y-oppo_pos.y() - self.block_shift_value)
                else:   # idx == 6
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setX(pos_x+self.block_shift_value)
                    crect.setHeight(pos_y-oppo_pos.y() - self.block_shift_value)
            else:
                if self.idx == 1:
                    pos_y = min(self.pos().y(), oppo_pos.y())
                    crect.setY(pos_y+self.block_shift_value)
                elif self.idx == 3:
                    pos_x = max(self.pos().x(), oppo_pos.x())
                    crect.setWidth(pos_x-oppo_pos.x() - self.block_shift_value)
                elif self.idx == 5:
                    pos_y = max(self.pos().y(), oppo_pos.y())
                    crect.setHeight(pos_y-oppo_pos.y() - self.block_shift_value)
                else:   # idx == 7
                    pos_x = min(self.pos().x(), oppo_pos.x())
                    crect.setX(pos_x+self.block_shift_value)
            
            self.ctrl.setRect(crect)
            scale = self.ctrl.current_scale
            new_center = self.ctrl.sceneBoundingRect().center()
            new_xy = QPointF(new_center.x() / scale - crect.width() / 2, new_center.y() / scale - crect.height() / 2)
            rect = QRectF(new_xy.x(), new_xy.y(), crect.width(), crect.height())
            blk_item.setRect(rect)

        elif self.drag_mode == self.DRAG_ROTATE:   # rotating
            rotate_vec = event.scenePos() - self.ctrl.sceneBoundingRect().center()
            rotation = np.rad2deg(math.atan2(rotate_vec.y(), rotate_vec.x()))
            self.ctrl.setAngle((rotation+self.rotate_start))
            # angle = self.ctrl.rotation()
            angle = self.ctrl.rotation() + 45 * self.idx
            idx = self.get_angle_idx(angle)
            if self.CURSOR_IDX != idx:
                self.setCursor(rotateCursorList[idx])
                self.CURSOR_IDX = idx
            self.updateAngleLabelPos()

    def get_angle_idx(self, angle) -> int:
        idx = int((angle + 22.5) % 360 / 45)
        return idx
    
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.ctrl.reshaping = False
            if self.drag_mode == self.DRAG_RESHAPE:
                self.ctrl.blk_item.endReshape()
            if self.drag_mode == self.DRAG_ROTATE:
                self.ctrl.blk_item.rotated.emit(self.ctrl.rotation())
            if self.drag_mode == self.DRAG_WARP:
                self.ctrl.endWarpEdit()
            self.drag_mode = self.DRAG_NONE
            
            self.ctrl.previewPixmap.setVisible(False)
            self.ctrl.angleLabel.setVisible(False)
            self.ctrl.blk_item.update()
            self.ctrl.updateBoundingRect()
            return super().mouseReleaseEvent(event)

class CenterWarpBlockItem(QGraphicsRectItem):
    def __init__(self, parent):
        super().__init__(parent)
        self.ctrl: TextBlkShapeControl = parent
        self.edge_width = 0.0
        self.dragging = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.updateEdgeWidth(CBEDGE_WIDTH)

    def updateEdgeWidth(self, edge_width: float):
        self.edge_width = float(edge_width)
        self.visible_len = self.edge_width / 2.0
        self.pen_width = edge_width / CBEDGE_WIDTH * 2
        self.visible_rect = QRectF(self.edge_width / 4.0, self.edge_width / 4.0, self.visible_len, self.visible_len)
        self.setRect(0, 0, self.edge_width, self.edge_width)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        rect = QRectF(self.visible_rect)
        rect.setTopLeft(self.boundingRect().topLeft() + rect.topLeft())
        painter.setPen(QPen(QColor(75, 75, 75), self.pen_width, Qt.PenStyle.SolidLine, Qt.SquareCap))
        painter.fillRect(rect, QColor(200, 200, 200, 125))
        painter.drawRect(rect)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.ctrl.ctrlblockPressed()
        if event.button() == Qt.MouseButton.LeftButton and self.ctrl.blk_item is not None and self.ctrl.warp_editing:
            self.dragging = True
            self.ctrl.beginRiseFallEdit()
            lp = self.ctrl.mapFromScene(event.scenePos())
            self.ctrl.beginRiseFallDrag(lp)
            event.accept()
            return
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.dragging:
            lp = self.ctrl.mapFromScene(event.scenePos())
            self.ctrl.updateRiseFallFromLocal(lp)
            event.accept()
            return
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.dragging = False
            self.ctrl.endWarpEdit()
            event.accept()
            return
        return super().mouseReleaseEvent(event)

class TextBlkShapeControl(QGraphicsRectItem):
    blk_item : TextBlkItem = None 
    ctrl_block: ControlBlockItem = None
    reshaping: bool = False
    
    def __init__(self, parent) -> None:
        super().__init__()
        self.gv = parent
        self.warp_editing = False
        self._controls_hidden = False
        self._warp_before = None
        self._rise_fall_base = None
        self._rise_fall_peak_u = 0.5
        self._rise_fall_amp = 0.0
        self._rise_fall_drag_start_local = None
        self._rise_fall_drag_start_u = 0.5
        self._rise_fall_drag_start_amp = 0.0
        self.ctrlblock_group = [
            ControlBlockItem(self, idx) for idx in range(8)
        ]
        self.center_warp_ctrl = CenterWarpBlockItem(self)
        self.center_warp_ctrl.hide()
        
        self.previewPixmap = QGraphicsPixmapItem(self)
        self.previewPixmap.setVisible(False)
        pen = QPen(QColor(69, 71, 87), 2, Qt.PenStyle.SolidLine)
        pen.setDashPattern([7, 14])
        self.setPen(pen)
        self.setVisible(False)

        self.angleLabel = QLabel(parent)
        self.angleLabel.setText("{:.1f}°".format(self.rotation()))
        self.angleLabel.setObjectName("angleLabel")
        self.angleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.angleLabel.setHidden(True)

        self.current_scale = 1.
        self.need_rescale = False
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def setBlkItem(self, blk_item: TextBlkItem):
        if self.blk_item == blk_item and self.isVisible():
            return
        if self.blk_item is not None:
            self.blk_item.under_ctrl = False
            if self.blk_item.isEditing():
                self.blk_item.endEdit()
            self.blk_item.update()
            
        self.blk_item = blk_item
        if blk_item is None:
            self.hide()
            return
        blk_item.under_ctrl = True
        blk_item.update()
        self.updateBoundingRect()
        self.show()

    def setWarpEditing(self, enabled: bool):
        self.warp_editing = bool(enabled)
        if self.blk_item is not None:
            if self.warp_editing:
                blk = self.blk_item.blk
                if getattr(blk, 'warp_mode', 'none') == 'none':
                    blk.warp_mode = 'quad'
                    blk.warp_quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            self.updateBoundingRect()
            self.updateControlBlocks()

    def updateBoundingRect(self):
        if self.blk_item is None:
            return
        abr = self.blk_item.absBoundingRect(qrect=True)
        br = QRectF(0, 0, abr.width(), abr.height())
        self.setRect(br)
        self.blk_item.setCenterTransform()
        self.setTransformOriginPoint(self.blk_item.transformOriginPoint())
        self.setPos(abr.x(), abr.y())
        self.setAngle(self.blk_item.angle)

    def setRect(self, *args): 
        super().setRect(*args)
        self.updateControlBlocks()

    def updateControlBlocks(self):
        b_rect = self.rect()
        b_rect = [b_rect.x(), b_rect.y(), b_rect.width(), b_rect.height()]
        corner_pnts = xywh2xyxypoly(np.array([b_rect])).reshape(-1, 2)
        center_pt = corner_pnts.mean(axis=0)
        mesh_edge_pnts = None
        if self.warp_editing and self.blk_item is not None:
            blk = self.blk_item.blk
            if getattr(blk, 'warp_mode', 'none') == 'quad':
                if getattr(blk, 'warp_quad', None) is None:
                    blk.warp_quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                quad = blk.warp_quad
                w = max(1.0, float(b_rect[2]))
                h = max(1.0, float(b_rect[3]))
                corner_pnts = np.array([[quad[0][0] * w, quad[0][1] * h],
                                        [quad[1][0] * w, quad[1][1] * h],
                                        [quad[2][0] * w, quad[2][1] * h],
                                        [quad[3][0] * w, quad[3][1] * h]], dtype=np.float32)
                center_pt = corner_pnts.mean(axis=0)
            elif getattr(blk, 'warp_mode', 'none') == 'mesh':
                w = max(1.0, float(b_rect[2]))
                h = max(1.0, float(b_rect[3]))
                mesh_size = getattr(blk, 'warp_mesh_size', None)
                mesh = getattr(blk, 'warp_mesh', None)
                if mesh_size is not None and mesh is not None and len(mesh_size) == 2:
                    nx, ny = int(mesh_size[0]), int(mesh_size[1])
                    if nx >= 2 and ny >= 2 and len(mesh) == nx * ny:
                        def sample_mesh(uu: float, vv: float) -> np.ndarray:
                            uu = max(0.0, min(1.0, float(uu)))
                            vv = max(0.0, min(1.0, float(vv)))
                            fu = uu * (nx - 1)
                            fv = vv * (ny - 1)
                            i0 = int(math.floor(fu))
                            j0 = int(math.floor(fv))
                            i1 = min(i0 + 1, nx - 1)
                            j1 = min(j0 + 1, ny - 1)
                            tu = fu - i0
                            tv = fv - j0

                            def mxy(ii: int, jj: int):
                                pt = mesh[jj * nx + ii]
                                return float(pt[0]), float(pt[1])

                            x00, y00 = mxy(i0, j0)
                            x10, y10 = mxy(i1, j0)
                            x01, y01 = mxy(i0, j1)
                            x11, y11 = mxy(i1, j1)

                            x0 = (1.0 - tu) * x00 + tu * x10
                            y0 = (1.0 - tu) * y00 + tu * y10
                            x1 = (1.0 - tu) * x01 + tu * x11
                            y1 = (1.0 - tu) * y01 + tu * y11
                            x = (1.0 - tv) * x0 + tv * x1
                            y = (1.0 - tv) * y0 + tv * y1
                            return np.array([x * w, y * h], dtype=np.float32)

                        corner_pnts = np.array([sample_mesh(0.0, 0.0),
                                                sample_mesh(1.0, 0.0),
                                                sample_mesh(1.0, 1.0),
                                                sample_mesh(0.0, 1.0)], dtype=np.float32)
                        center_pt = sample_mesh(0.5, 0.5)
                        mesh_edge_pnts = np.array([sample_mesh(0.5, 0.0),
                                                   sample_mesh(1.0, 0.5),
                                                   sample_mesh(0.5, 1.0),
                                                   sample_mesh(0.0, 0.5)], dtype=np.float32)
        edge_pnts = mesh_edge_pnts if mesh_edge_pnts is not None else (corner_pnts[[1, 2, 3, 0]] + corner_pnts) / 2
        pnts = [edge_pnts, corner_pnts]
        for ii, ctrlblock in enumerate(self.ctrlblock_group):
            if self._controls_hidden:
                ctrlblock.hide()
            else:
                ctrlblock.show()
            is_corner = not ii % 2
            idx = ii // 2
            hitbox_xy = ctrlidx_to_hitbox[ii][:2]
            pos = pnts[is_corner][idx] + hitbox_xy * ctrlblock.edge_width
            ctrlblock.setPos(pos[0], pos[1])

        if self.center_warp_ctrl is not None:
            if self._controls_hidden:
                self.center_warp_ctrl.hide()
            elif self.warp_editing and self.blk_item is not None:
                self.center_warp_ctrl.show()
                blk = self.blk_item.blk
                w = max(1.0, float(b_rect[2]))
                h = max(1.0, float(b_rect[3]))
                peak_u = float(getattr(blk, 'warp_rise_fall_u', 0.5) or 0.5)
                peak_u = max(0.0, min(1.0, peak_u))
                amp = float(getattr(blk, 'warp_rise_fall_amp', 0.0) or 0.0)
                amp = max(-0.35, min(0.35, amp))
                xh = peak_u * w
                yh = h / 2.0 - (amp / 1.2) * h
                xh = max(0.0, min(w, xh))
                yh = max(0.0, min(h, yh))
                cx = xh - self.center_warp_ctrl.edge_width / 2.0
                cy = yh - self.center_warp_ctrl.edge_width / 2.0
                self.center_warp_ctrl.setPos(cx, cy)
            else:
                self.center_warp_ctrl.hide()

    def beginWarpEdit(self):
        if self.blk_item is None:
            self._warp_before = None
            return
        blk = self.blk_item.blk
        self._warp_before = (getattr(blk, 'warp_mode', 'none'), copy.deepcopy(getattr(blk, 'warp_quad', None)),
                             copy.deepcopy(getattr(blk, 'warp_mesh_size', None)), copy.deepcopy(getattr(blk, 'warp_mesh', None)),
                             float(getattr(blk, 'warp_rise_fall_u', 0.5) or 0.5),
                             float(getattr(blk, 'warp_rise_fall_amp', 0.0) or 0.0))

    def _suggest_mesh_size(self, target_cell_px: float = 64.0, min_nx: int = 7, min_ny: int = 5, max_nx: int = 25, max_ny: int = 15):
        w = float(self.rect().width())
        h = float(self.rect().height())
        target_cell_px = max(8.0, float(target_cell_px))
        nx = int(max(min_nx, min(max_nx, round(w / target_cell_px) + 1)))
        ny = int(max(min_ny, min(max_ny, round(h / target_cell_px) + 1)))
        nx = max(2, nx)
        ny = max(2, ny)
        return nx, ny

    def _ensure_mesh_for_edit(self, nx: int = 5, ny: int = 3):
        if self.blk_item is None:
            return
        blk = self.blk_item.blk
        mode = getattr(blk, 'warp_mode', 'none') or 'none'
        quad = getattr(blk, 'warp_quad', None)
        mesh = getattr(blk, 'warp_mesh', None)
        mesh_size = getattr(blk, 'warp_mesh_size', None)

        if mode == 'mesh' and mesh is not None and mesh_size is not None and len(mesh_size) == 2:
            try:
                mx, my = int(mesh_size[0]), int(mesh_size[1])
            except Exception:
                mx, my = 0, 0
            if mx >= 2 and my >= 2 and len(mesh) == mx * my:
                return

        if quad is None or len(quad) != 4:
            quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

        p00 = np.array([float(quad[0][0]), float(quad[0][1])], dtype=np.float64)
        p10 = np.array([float(quad[1][0]), float(quad[1][1])], dtype=np.float64)
        p11 = np.array([float(quad[2][0]), float(quad[2][1])], dtype=np.float64)
        p01 = np.array([float(quad[3][0]), float(quad[3][1])], dtype=np.float64)

        src = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0]], dtype=np.float64)
        dst = np.array([p00, p10, p11, p01], dtype=np.float64)

        A = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
            A.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
        A = np.asarray(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1, :].reshape(3, 3)
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        def map_uv(u: float, v: float):
            p = H @ np.array([float(u), float(v), 1.0], dtype=np.float64)
            w = float(p[2])
            if abs(w) < 1e-12:
                return 0.0, 0.0
            x = float(p[0]) / w
            y = float(p[1]) / w
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            return x, y

        out_mesh = []
        for j in range(ny):
            v = j / (ny - 1)
            for i in range(nx):
                u = i / (nx - 1)
                x, y = map_uv(u, v)
                out_mesh.append([x, y])

        blk.warp_mode = 'mesh'
        blk.warp_quad = None
        blk.warp_mesh_size = [int(nx), int(ny)]
        blk.warp_mesh = out_mesh

    def beginRiseFallEdit(self):
        self.beginWarpEdit()
        if self.blk_item is None:
            self._rise_fall_base = None
            self._rise_fall_drag_start_local = None
            return
        blk = self.blk_item.blk
        self._rise_fall_peak_u = float(getattr(blk, 'warp_rise_fall_u', 0.5) or 0.5)
        self._rise_fall_amp = float(getattr(blk, 'warp_rise_fall_amp', 0.0) or 0.0)
        nx, ny = self._suggest_mesh_size()
        self._ensure_mesh_for_edit(nx, ny)
        self._rise_fall_base = (copy.deepcopy(getattr(blk, 'warp_mesh_size', None)),
                                copy.deepcopy(getattr(blk, 'warp_mesh', None)))

    def beginRiseFallDrag(self, local_pos: QPointF):
        if self.blk_item is None:
            self._rise_fall_drag_start_local = None
            return
        blk = self.blk_item.blk
        self._rise_fall_drag_start_local = QPointF(local_pos)
        self._rise_fall_drag_start_u = float(getattr(blk, 'warp_rise_fall_u', 0.5) or 0.5)
        self._rise_fall_drag_start_amp = float(getattr(blk, 'warp_rise_fall_amp', 0.0) or 0.0)

    def updateWarpFromLocal(self, ctrl_idx: int, local_pos: QPointF):
        if self.blk_item is None:
            return
        blk = self.blk_item.blk
        mode = getattr(blk, 'warp_mode', 'none') or 'none'
        if mode == 'mesh':
            self._updateMeshFromHandle(ctrl_idx, local_pos)
            return
        if mode != 'quad' or getattr(blk, 'warp_quad', None) is None:
            blk.warp_mode = 'quad'
            blk.warp_quad = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

        if ctrl_idx in {1, 3, 5, 7}:
            nx, ny = self._suggest_mesh_size()
            self._ensure_mesh_for_edit(nx, ny)
            self._updateMeshFromHandle(ctrl_idx, local_pos)
            return

        w = max(1.0, float(self.rect().width()))
        h = max(1.0, float(self.rect().height()))
        x = max(0.0, min(w, float(local_pos.x())))
        y = max(0.0, min(h, float(local_pos.y())))

        quad = blk.warp_quad
        if ctrl_idx in {0, 2, 4, 6}:
            corner_map = {0: 0, 2: 1, 4: 2, 6: 3}
            quad[corner_map[ctrl_idx]] = [x / w, y / h]
        elif ctrl_idx in {1, 3, 5, 7}:
            edge_map = {1: (0, 1), 3: (1, 2), 5: (2, 3), 7: (3, 0)}
            ia, ib = edge_map[ctrl_idx]

            ax, ay = float(quad[ia][0]) * w, float(quad[ia][1]) * h
            bx, by = float(quad[ib][0]) * w, float(quad[ib][1]) * h
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            dx, dy = x - mx, y - my

            ax = max(0.0, min(w, ax + dx))
            ay = max(0.0, min(h, ay + dy))
            bx = max(0.0, min(w, bx + dx))
            by = max(0.0, min(h, by + dy))

            quad[ia] = [ax / w, ay / h]
            quad[ib] = [bx / w, by / h]
        else:
            return

        self.blk_item.update()
        self.updateControlBlocks()

    def updateRiseFallFromLocal(self, local_pos: QPointF):
        if self.blk_item is None:
            return
        blk = self.blk_item.blk

        w = max(1.0, float(self.rect().width()))
        h = max(1.0, float(self.rect().height()))
        if self._rise_fall_drag_start_local is not None:
            dx = float(local_pos.x() - self._rise_fall_drag_start_local.x())
            dy = float(local_pos.y() - self._rise_fall_drag_start_local.y())
            peak_u = float(self._rise_fall_drag_start_u) + dx / w
            amp = float(self._rise_fall_drag_start_amp) - (dy / h * 1.2)
        else:
            x = max(0.0, min(w, float(local_pos.x())))
            y = max(0.0, min(h, float(local_pos.y())))
            amp = (h / 2.0 - y) / h * 1.2
            peak_u = x / w

        amp = max(-0.35, min(0.35, float(amp)))
        peak_u = max(0.05, min(0.95, float(peak_u)))
        self._rise_fall_peak_u = peak_u
        blk.warp_rise_fall_u = peak_u
        self._rise_fall_amp = amp
        blk.warp_rise_fall_amp = amp

        if self._rise_fall_base is None:
            nx, ny = self._suggest_mesh_size()
            self._ensure_mesh_for_edit(nx, ny)
            self._rise_fall_base = (copy.deepcopy(getattr(blk, 'warp_mesh_size', None)),
                                    copy.deepcopy(getattr(blk, 'warp_mesh', None)))

        mesh_size, base_mesh = self._rise_fall_base
        if mesh_size is None or base_mesh is None or len(mesh_size) != 2:
            return
        nx, ny = int(mesh_size[0]), int(mesh_size[1])
        if nx < 2 or ny < 2 or len(base_mesh) != nx * ny:
            return

        out_mesh = []
        for j in range(ny):
            for i in range(nx):
                u = i / (nx - 1)
                x0, y0 = base_mesh[j * nx + i]
                if u <= peak_u:
                    uu = 0.5 * (u / peak_u) if peak_u > 1e-6 else 0.0
                else:
                    uu = 0.5 + 0.5 * ((u - peak_u) / (1.0 - peak_u)) if (1.0 - peak_u) > 1e-6 else 1.0
                dy = -amp * math.sin(math.pi * float(uu))
                yy = max(0.0, min(1.0, float(y0) + dy))
                out_mesh.append([float(x0), yy])

        blk.warp_mode = 'mesh'
        blk.warp_quad = None
        blk.warp_mesh_size = [nx, ny]
        blk.warp_mesh = out_mesh
        self.blk_item.update()
        self.updateControlBlocks()

    def _updateMeshFromHandle(self, ctrl_idx: int, local_pos: QPointF):
        if self.blk_item is None:
            return
        blk = self.blk_item.blk
        mesh_size = getattr(blk, 'warp_mesh_size', None)
        mesh = getattr(blk, 'warp_mesh', None)
        if mesh_size is None or mesh is None or len(mesh_size) != 2:
            return
        nx, ny = int(mesh_size[0]), int(mesh_size[1])
        if nx < 2 or ny < 2 or len(mesh) != nx * ny:
            return

        w = max(1.0, float(self.rect().width()))
        h = max(1.0, float(self.rect().height()))
        x = max(0.0, min(w, float(local_pos.x())))
        y = max(0.0, min(h, float(local_pos.y())))
        dxn = x / w
        dyn = y / h

        def clamp01(v: float) -> float:
            return max(0.0, min(1.0, v))

        p00 = mesh[0]
        p10 = mesh[nx - 1]
        p11 = mesh[(ny - 1) * nx + (nx - 1)]
        p01 = mesh[(ny - 1) * nx]

        if ctrl_idx in {0, 2, 4, 6}:
            corner_map = {0: 0, 2: 1, 4: 2, 6: 3}
            ci = corner_map[ctrl_idx]
            corners = [p00, p10, p11, p01]
            ox, oy = float(corners[ci][0]), float(corners[ci][1])
            ddx = dxn - ox
            ddy = dyn - oy
            out_mesh = []
            for j in range(ny):
                v = j / (ny - 1)
                for i in range(nx):
                    u = i / (nx - 1)
                    wx = (1 - u) * (1 - v)
                    wy = u * (1 - v)
                    wz = u * v
                    ww = (1 - u) * v
                    wcorner = [wx, wy, wz, ww][ci]
                    x0, y0 = mesh[j * nx + i]
                    out_mesh.append([clamp01(float(x0) + ddx * wcorner), clamp01(float(y0) + ddy * wcorner)])
            blk.warp_mesh = out_mesh
            self._rise_fall_base = None
            self.blk_item.update()
            self.updateControlBlocks()
            return

        if ctrl_idx in {1, 3, 5, 7}:
            mi = int(round((nx - 1) / 2))
            mj = int(round((ny - 1) / 2))
            if ctrl_idx == 1:
                hx, hy = mesh[0 * nx + mi]
            elif ctrl_idx == 5:
                hx, hy = mesh[(ny - 1) * nx + mi]
            elif ctrl_idx == 3:
                hx, hy = mesh[mj * nx + (nx - 1)]
            else:
                hx, hy = mesh[mj * nx + 0]
            ddx = dxn - float(hx)
            ddy = dyn - float(hy)

            out_mesh = []
            for j in range(ny):
                v = j / (ny - 1)
                for i in range(nx):
                    u = i / (nx - 1)
                    edge_u = 4.0 * u * (1.0 - u)
                    edge_v = 4.0 * v * (1.0 - v)
                    if ctrl_idx == 1:
                        weight = (1.0 - v) * edge_u
                    elif ctrl_idx == 5:
                        weight = v * edge_u
                    elif ctrl_idx == 3:
                        weight = u * edge_v
                    else:
                        weight = (1.0 - u) * edge_v
                    x0, y0 = mesh[j * nx + i]
                    out_mesh.append([clamp01(float(x0) + ddx * weight), clamp01(float(y0) + ddy * weight)])

            blk.warp_mesh = out_mesh
            self._rise_fall_base = None
            self.blk_item.update()
            self.updateControlBlocks()
            return

    def endWarpEdit(self):
        if self.blk_item is None or self._warp_before is None:
            self._warp_before = None
            self._rise_fall_base = None
            self._rise_fall_drag_start_local = None
            return
        blk = self.blk_item.blk
        before = self._warp_before
        after = (getattr(blk, 'warp_mode', 'none'), copy.deepcopy(getattr(blk, 'warp_quad', None)),
                 copy.deepcopy(getattr(blk, 'warp_mesh_size', None)), copy.deepcopy(getattr(blk, 'warp_mesh', None)),
                 float(getattr(blk, 'warp_rise_fall_u', 0.5) or 0.5),
                 float(getattr(blk, 'warp_rise_fall_amp', 0.0) or 0.0))
        self._warp_before = None
        self._rise_fall_base = None
        self._rise_fall_drag_start_local = None
        if before != after:
            self.blk_item.warped.emit(before, after)

    def setAngle(self, angle: int) -> None:
        center = self.boundingRect().center()
        self.setTransformOriginPoint(center)
        self.setRotation(angle)

    def ctrlblockPressed(self):
        self.scene().clearSelection()
        if self.blk_item is not None:
            self.blk_item.endEdit()

    def paint(self, painter: QPainter, option: 'QStyleOptionGraphicsItem', widget = ...) -> None:
        painter.setCompositionMode(QPainter.CompositionMode.RasterOp_NotDestination)
        super().paint(painter, option, widget)

    def hideControls(self):
        self._controls_hidden = True
        for ctrl in self.ctrlblock_group:
            ctrl.hide()
        if self.center_warp_ctrl is not None:
            self.center_warp_ctrl.hide()

    def showControls(self):
        self._controls_hidden = False
        self.updateControlBlocks()

    def updateScale(self, scale: float):
        if not self.isVisible():
            if scale != self.current_scale:
                self.need_rescale = True
                self.current_scale = scale
            return

        self.current_scale = scale
        scale = 1 / scale
        pen = self.pen()
        pen.setWidthF(2 * scale)
        self.setPen(pen)
        for ctrl in self.ctrlblock_group:
            ctrl.updateEdgeWidth(CBEDGE_WIDTH * scale)
        if self.center_warp_ctrl is not None:
            self.center_warp_ctrl.updateEdgeWidth(CBEDGE_WIDTH * scale)

    def show(self) -> None:
        super().show()
        if self.need_rescale:
            self.updateScale(self.current_scale)
            self.need_rescale = False
        self.setZValue(1)

    def startEditing(self):
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.hideControls()

    def endEditing(self):
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        if self.isVisible():
            self.showControls()

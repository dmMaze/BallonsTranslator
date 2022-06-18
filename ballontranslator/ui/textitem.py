import os.path as osp
import math
import numpy as np
from PyQt5.QtWidgets import QTextEdit, QGraphicsPixmapItem, QGraphicsDropShadowEffect, QGraphicsView, QGraphicsPixmapItem, QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QLabel, QSizePolicy, QScrollBar, QListView, QGraphicsSceneWheelEvent, QGraphicsTextItem, QStyleOptionGraphicsItem, QStyle, QGraphicsSceneMouseEvent
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF, QPoint, pyqtSignal, QSizeF
from PyQt5.QtGui import QTextCursor, QPixmap, QPainterPath, QTextDocument, QMouseEvent, QKeyEvent, QWheelEvent, QBrush, QFocusEvent, QPainter, QTextFrame, QTransform, QTextBlock, QAbstractTextDocumentLayout, QTextLayout, QFont, QFontMetrics, QTextOption, QTextLine, QPen, QColor, QTextFormat, QTextCursor, QTextCharFormat, QTextDocument, QTextBlockFormat
from dl.textdetector.textblock import TextBlock
from utils.imgproc_utils import xywh2xyxypoly, rotate_polygons
from typing import List, Union, Tuple
from .misc import FontFormat, px2pt, pt2px, td_pattern, table_pattern
from .textlayout import VerticalTextDocumentLayout, HorizontalTextDocumentLayout


TEXTRECT_SHOW_COLOR = QColor(30, 147, 229, 170)
TEXTRECT_SELECTED_COLOR = QColor(248, 64, 147, 170)

class TextBlkItem(QGraphicsTextItem):
    begin_edit = pyqtSignal(int)
    end_edit = pyqtSignal(int)
    hover_enter = pyqtSignal(int)
    hover_leave = pyqtSignal(int)
    hover_move = pyqtSignal(int)
    moved = pyqtSignal()
    moving = pyqtSignal(QGraphicsTextItem)
    rotated = pyqtSignal(float)
    reshaped = pyqtSignal(QGraphicsTextItem)
    content_changed = pyqtSignal(QGraphicsTextItem)
    leftbutton_pressed = pyqtSignal(int)
    doc_size_changed = pyqtSignal(int)
    def __init__(self, blk: TextBlock = None, idx: int = 0, set_format=True, show_rect=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blk = None
        self.repainting = False
        self.reshaping = False
        self.under_ctrl = False
        self.draw_rect = show_rect
        self._display_rect: QRectF = QRectF(0, 0, 1, 1)
        self.stroke_width = 0
        self.idx = idx
        self.line_spacing: float = 1.
        self.background_pixmap: QPixmap = None
        self.stroke_color = QColor(0, 0, 0)
        self.bound_checking = False # not used
        self.oldPos = QPointF()
        self.oldRect = QRectF()

        self.setVertical(False, force=True)
        self.initTextBlock(blk, set_format=set_format)
        self.setBoundingRegionGranularity(0)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        
    def is_editting(self):
        return self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction

    def documentContentChanged(self):
        if self.hasFocus():   
            self.content_changed.emit(self)
        sw = self.stroke_width * self.document().defaultFont().pointSizeF()
        if sw != 0 and not self.repainting:
            self.repaint_background()
        self.update()

    def repaint_background(self):
        
        self.repainting = True
        doc = self.document()
        sw = self.stroke_width * doc.defaultFont().pointSizeF()
        format = QTextCharFormat()
        format.setTextOutline (QPen(self.stroke_color, sw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.Document)
        old_fmt = cursor.charFormat()
        cursor.mergeCharFormat(format)

        size = self.boundingRect().size()
        self.background_pixmap = QPixmap(size.toSize())
        self.background_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(self.background_pixmap)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.device()
        doc.drawContents(painter)
        painter.end()
        
        cursor.setCharFormat(old_fmt)
        self.repainting = False
        
    def docSizeChanged(self):
        self.setCenterTransform()
        self.doc_size_changed.emit(self.idx)

    def initTextBlock(self, blk: TextBlock = None, set_format=True):
        self.blk = blk
        if blk is None:
            xyxy = [0, 0, 0, 0]
            blk = TextBlock(xyxy)
            blk.lines = [xyxy]
            bx1, by1, bx2, by2 = xyxy
            xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        self.setRect(blk.bounding_rect())
        if blk.angle != 0:
            self.setRotation(blk.angle)
        
        # blk.vertical = False
        set_char_fmt = False
        if not blk.rich_text:
            if blk.translation:
                self.setPlainText(blk.translation)
                set_char_fmt = True
        else:
            self.setHtml(blk.rich_text)
            
        if set_format:
            font_fmt = FontFormat()
            font_fmt.from_textblock(blk)
            self.set_fontformat(font_fmt, set_char_format=set_char_fmt)

    def setCenterTransform(self):
        center = self.boundingRect().center()
        self.setTransformOriginPoint(center)

    def rect(self) -> QRectF:
        return QRectF(self.pos(), self.boundingRect().size())

    def startReshape(self):
        self.oldRect = self.rect()
        self.reshaping = True

    def endReshape(self):
        self.reshaped.emit(self)
        self.reshaping = False

    def setRect(self, rect: QRectF) -> None:
        if isinstance(rect, List):
            rect = QRectF(*rect)
        
        self.setPos(rect.topLeft())
        doc = self.document()
        layout = doc.documentLayout()
        self._display_rect = rect
        layout.setMaxSize(rect.width(), rect.height())
        doc.setPageSize(QSizeF(rect.width(), rect.height()))

        self.setCenterTransform()
        if self.background_pixmap is not None:
            self.repaint_background()

    def documentSize(self):
        return self.document().documentLayout().documentSize()

    def boundingRect(self) -> QRectF:
        br = super().boundingRect()
        if self._display_rect is not None:
            size = self.documentSize()
            br.setHeight(max(self._display_rect.height(), size.height()))
            br.setWidth(max(self._display_rect.width(), size.width()))
        return br

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        br = self.boundingRect()
        path.addRect(br)
        return path

    def setScale(self, scale: float) -> None:
        self.setTransformOriginPoint(0, 0)
        super().setScale(scale)
        self.setCenterTransform()

    @property
    def is_vertical(self) -> bool:
        return isinstance(self.document().documentLayout(), VerticalTextDocumentLayout)

    @property
    def angle(self) -> int:
        return self.blk.angle

    def toTextBlock(self) -> TextBlock:
        raise NotImplementedError

    def setAngle(self, angle: int):
        self.setCenterTransform()
        if self.blk.angle != angle:
            self.setRotation(angle)
        self.blk.angle = angle

    def setVertical(self, vertical: bool, force=False):
        
        # if not force and self.is_vertical != vertical:
        #     return
        if self.blk is not None:
            self.blk.vertical = vertical
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        doc = self.document()
        doc.disconnect()
        doc.documentLayout().disconnect()
        html = doc.toHtml()
        default_font = doc.defaultFont()
        rect = self.rect()

        doc = QTextDocument()
        if vertical:
            layout = VerticalTextDocumentLayout(doc)
        else:
            layout = HorizontalTextDocumentLayout(doc)
        layout.setMaxSize(rect.width(), rect.height())
        layout.sizeEnlarged.connect(self.on_document_enlarged)
        layout.documentSizeChanged.connect(self.docSizeChanged)
        doc.setDocumentLayout(layout)
        doc.setDefaultFont(default_font)
        doc.setHtml(html)
        doc.setDocumentMargin(0)
        self.setDocument(doc)
        doc.contentsChanged.connect(self.documentContentChanged)
        
        self.setCenterTransform()
        self.setLineSpacing(self.line_spacing)
        if self.background_pixmap is not None:
            self.repaint_background()

        # cursor = QTextCursor(doc)
        # format = cursor.charFormat()
        # cursor.mergeCharFormat(format)
        # cursor.select(QTextCursor.Document)
        # cursor.mergeBlockCharFormat(format)
        # cursor.clearSelection()
        # # https://stackoverflow.com/questions/37160039/set-default-character-format-in-qtextdocument
        # self.setTextCursor(cursor)
        self.doc_size_changed.emit(self.idx)

    def on_document_enlarged(self):
        rect = self.rect()
        old_width = self._display_rect.width()
        otr = self.sceneBoundingRect().topLeft()
        self._display_rect.setWidth(rect.width())
        self._display_rect.setHeight(rect.height())
        new_width = self._display_rect.width()
        self.setCenterTransform()
        self.setPos(self.pos() + otr - self.sceneBoundingRect().topLeft())
        if self.is_vertical and not self.reshaping:
            pos = self.pos()
            delta_x = (old_width - new_width) * self.scale()
            if self.rotation() == 0:
                pos.setX(pos.x() + delta_x)
            else:
                rad = np.deg2rad(self.rotation())
                pos.setX(pos.x() + delta_x * np.cos(rad))
                pos.setY(pos.y() + delta_x * np.sin(rad))
            self.setPos(pos)


    def focusOutEvent(self, event: QFocusEvent) -> None:
        self.end_edit.emit(self.idx)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        super().focusOutEvent(event)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def documentLayout(self) -> VerticalTextDocumentLayout:
        return self.document().documentLayout()

    def setStrokeWidth(self, stroke_width):
        self.stroke_width = stroke_width
        if stroke_width > 0:                
            self.repaint_background()
        else:
            self.background_pixmap = None
        sw = self.stroke_width * self.document().defaultFont().pointSizeF()
        # self.document().setDocumentMargin(sw/2)
        self.documentLayout().updateDocumentMargin(sw/2)
        self.on_document_enlarged()
        self.update()

    def setStrokeColor(self, scolor):
        self.stroke_color = scolor if isinstance(scolor, QColor) else QColor(*scolor)
        if self.background_pixmap is not None:
            self.repaint_background()
        self.update()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        br = self.boundingRect()
        # mouse_over = option.state & QStyle.StateFlag.State_MouseOver
        # selected = option.state & QStyle.StateFlag.State_Selected
        
        painter.save()

        # shadow effect not working ???
        # se = QGraphicsDropShadowEffect()
        # se.setBlurRadius(12)
        # se.setOffset(0, 0)
        # se.setColor(QColor(30, 147, 229))
        # self.setGraphicsEffect(se)

        if self.background_pixmap is not None:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(br.toRect(), self.background_pixmap)

        # https://stackoverflow.com/questions/13966868/qt-outlined-text-without-thinning-font
        # too slow

        # if sw != 0 and self.tcursor:
        #     old_fmt = self.tcursor.charFormat()
        #     self.tcursor.setCharFormat(self.stroke_fmt)

            # cursor = QTextCursor(self.document())
            # old_fmt = cursor.charFormat()
            # format = cursor.charFormat()
            # sw = sw * self.document().defaultFont().pointSizeF()
            # format.setTextOutline(QPen(self.stroke_color, sw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            
            # cursor.select(QTextCursor.Document)
            
            # cursor.setCharFormat(format)
            # layout = self.document().documentLayout()   
            # layout.draw(painter, layout.PaintContext())

        # painter.setCompositionMode(QPainter.RasterOp_NotDestination)
        # pen = painter.pen()
        # pen.setWidth(1)
        # pen.setStyle(Qt.DashLine)
        # pen.setDashPattern([7, 5])
        # # pen.setColor(QColor(0, 127, 127))
        # painter.setPen(pen)
        # painter.setBrush(Qt.NoBrush)
        # painter.drawRect(self.boundingRect())

        draw_rect = self.draw_rect and not self.under_ctrl
        if self.isSelected() and not self.is_editting():
            pen = QPen(TEXTRECT_SELECTED_COLOR, 3.5 / self.scale(), Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(br)
        elif draw_rect:
            pen = QPen(TEXTRECT_SHOW_COLOR, 3 / self.scale(), Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(br)
        
        painter.restore()
        option.state = QStyle.State_None
        super().paint(painter, option, widget)

    def startEdit(self):
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

        self.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.setFocus()
        self.begin_edit.emit(self.idx)
    
    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:    
        self.startEdit()
        super().mouseDoubleClickEvent(event)
        
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if not self.bound_checking or \
            self.textInteractionFlags() == Qt.TextEditorInteraction:
            super().mouseMoveEvent(event)
        else:
            b_rect = self.boundingRect()
            scale = self.scale()
            b_rect = QRectF(b_rect.x()*scale, b_rect.y()*scale, b_rect.width()*scale, b_rect.height()*scale)
            pos = event.pos() - event.lastPos()
            pos = QPointF(pos.x()*scale, pos.y()*scale)
            if self.blk.angle != 0: # angled need text some recalculations
                x, y = pos.x(), pos.y()
                rad = -self.blk.angle * math.pi / 180
                s, c = math.sin(rad), math.cos(rad)
                pos.setX(x * c + y * s)
                pos.setY(-x * s + y * c)
                w, h = b_rect.width(), b_rect.height()
                b_poly = np.array([[0, 0, w, 0, w, h, 0, h]])
                b_poly = rotate_polygons([0, 0], b_poly, -self.blk.angle)[0]
                b_rect.setRect(b_poly[::2].min(), b_poly[1::2].min(), b_poly[::2].max(), b_poly[1::2].max())
            pos += self.pos()
            scene_rect = self.scene().sceneRect()
            pos.setX(np.clip(pos.x(), -b_rect.x(), scene_rect.width()-b_rect.width()))
            pos.setY(np.clip(pos.y(), -b_rect.y(), scene_rect.height()-b_rect.height()))
            self.setPos(pos)
        self.moving.emit(self)

    # QT 5.15.x causing segmentation fault 
    def contextMenuEvent(self, event):
        return super().contextMenuEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.oldPos = self.pos()
            self.leftbutton_pressed.emit(self.idx)
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            if self.oldPos != self.pos():
                self.moved.emit()
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_move.emit(self.idx)
        return super().hoverMoveEvent(event)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_enter.emit(self.idx)
        return super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_leave.emit(self.idx)
        return super().hoverLeaveEvent(event)

    def toPixmap(self) -> QPixmap:
        pixmap = QPixmap(self.boundingRect().size().toSize())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        doc = self.document()
        doc.drawContents(painter)
        painter.end()
        return pixmap

    def toHtml(self) -> str:
        html = super().toHtml()
        tables = table_pattern.findall(html)
        if tables:
            _, td = td_pattern.findall(html)[0]
            html = tables[0] + td + '</body></html>'

        return html.replace('>\n<', '><')

    def setAlignment(self, alignment):
        doc = self.document()
        op = doc.defaultTextOption()
        op.setAlignment(alignment)
        doc.setDefaultTextOption(op)
        if self.background_pixmap is not None:
            self.repaint_background()

    def alignment(self):
        return self.document().defaultTextOption().alignment()

    def get_fontformat(self) -> FontFormat:
        fmt = self.textCursor().charFormat()
        font = fmt.font()
        color = fmt.foreground().color()
        frgb = [color.red(), color.green(), color.blue()]
        srgb = [self.stroke_color.red(), self.stroke_color.green(), self.stroke_color.blue()]
        alignment = self.alignment()
        if alignment == Qt.AlignLeft:
            alignment = 0
        elif alignment == Qt.AlignRight:
            alignment = 2
        else:
            alignment = 1
        weight = font.weight()
        # https://doc.qt.io/qt-5/qfont.html#Weight-enum, 50 is normal
        if weight == 0:
            weight = 50
        font_format = FontFormat(
            font.family(),
            font.pointSizeF(),
            self.stroke_width, 
            frgb,
            srgb,
            font.bold(),
            font.underline(),
            font.italic(),
            alignment, 
            self.is_vertical,
            weight, 
            line_spacing=self.line_spacing
        )
        return font_format

    def setPlainText(self, text: str) -> None:
        # fmt = self.get_fontformat()
        super().setPlainText(text)
        # self.set_fontformat(fmt)

    def setFont(self, font: QFont):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        pass

    def set_fontformat(self, ffmat: FontFormat, set_char_format=False):
        
        if self.is_vertical != ffmat.vertical:
            self.setVertical(ffmat.vertical)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        format = cursor.charFormat()
        font = self.document().defaultFont()
        font.setFamily(ffmat.family)
        font.setPointSize(ffmat.size)
        font.setBold(ffmat.bold)

        self.document().setDefaultFont(font)
        format.setFont(font)
        format.setForeground(QColor(*ffmat.frgb))
        format.setFontWeight(ffmat.weight)
        format.setFontItalic(ffmat.italic)
        format.setFontUnderline(ffmat.underline)
        
        cursor.setCharFormat(format)
        cursor.select(QTextCursor.Document)
        cursor.setBlockCharFormat(format)
        if set_char_format:
            cursor.setCharFormat(format)
        cursor.clearSelection()
        # https://stackoverflow.com/questions/37160039/set-default-character-format-in-qtextdocument
        cursor.movePosition(QTextCursor.Start)
        self.setTextCursor(cursor)

        self.stroke_width = ffmat.stroke_width
        self.setStrokeWidth(ffmat.stroke_width)
        self.setStrokeColor(ffmat.srgb)
        
        alignment = [Qt.AlignLeft, Qt.AlignCenter, Qt.AlignRight][ffmat.alignment]
        doc = self.document()
        op = doc.defaultTextOption()
        op.setAlignment(alignment)
        doc.setDefaultTextOption(op)
        self.setLineSpacing(ffmat.line_spacing)

    def updateBlkFormat(self):
        fmt = self.get_fontformat()
        self.blk.default_stroke_width = fmt.stroke_width
        self.blk.line_spacing = fmt.line_spacing
        self.blk.font_family = fmt.family
        self.blk.font_size = pt2px(fmt.size)
        self.blk.font_weight = fmt.weight
        self.blk._alignment = fmt.alignment
        # self.blk._alignment = self.blk.alignment()
        self.blk.set_font_colors(fmt.frgb, fmt.srgb, accumulate=False)

    def setLineSpacing(self, line_spacing, cursor=None):
        self.line_spacing = line_spacing
        self.document().documentLayout().setLineSpacing(self.line_spacing)
        if self.background_pixmap is not None:
            self.repaint_background()

    def scaleChanged(self) -> None:
        super().scaleChanged()
        if self.background_pixmap is not None:
            self.repaint_background()
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
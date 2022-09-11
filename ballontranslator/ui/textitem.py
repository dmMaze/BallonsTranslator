import math, re
import numpy as np
from typing import List, Union, Tuple

from qtpy.QtWidgets import QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QGraphicsTextItem, QStyleOptionGraphicsItem, QStyle, QGraphicsSceneMouseEvent
from qtpy.QtCore import Qt, QRect, QRectF, QPointF, Signal, QSizeF
from qtpy.QtGui import QKeyEvent, QFont, QTextCursor, QPixmap, QPainterPath, QTextDocument, QFocusEvent, QPainter, QPen, QColor, QTextCursor, QTextCharFormat, QTextDocument

from dl.textdetector.textblock import TextBlock
from utils.imgproc_utils import xywh2xyxypoly, rotate_polygons
from .misc import FontFormat, px2pt, pt2px, td_pattern, table_pattern, html_max_fontsize
from .scene_textlayout import VerticalTextDocumentLayout, HorizontalTextDocumentLayout, SceneTextLayout
from .text_graphical_effect import apply_shadow_effect

TEXTRECT_SHOW_COLOR = QColor(30, 147, 229, 170)
TEXTRECT_SELECTED_COLOR = QColor(248, 64, 147, 170)


class TextBlkItem(QGraphicsTextItem):
    begin_edit = Signal(int)
    end_edit = Signal(int)
    hover_enter = Signal(int)
    hover_move = Signal(int)
    moved = Signal()
    moving = Signal(QGraphicsTextItem)
    rotated = Signal(float)
    reshaped = Signal(QGraphicsTextItem)
    content_changed = Signal(QGraphicsTextItem)
    leftbutton_pressed = Signal(int)
    doc_size_changed = Signal(int)
    pasted = Signal(int)
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
        self.letter_spacing: float = 1.
        self.shadow_radius = 0
        self.shadow_strength = 1
        self.shadow_color = [0, 0, 0]
        self.shadow_offset = [0, 0]
        
        self.background_pixmap: QPixmap = None
        self.stroke_color = QColor(0, 0, 0)
        self.bound_checking = False # not used
        self.oldPos = QPointF()
        self.oldRect = QRectF()
        self.repaint_on_changed = True

        self.layout: Union[VerticalTextDocumentLayout, HorizontalTextDocumentLayout] = None
        self.document().setDocumentMargin(0)
        self.setVertical(False)
        self.initTextBlock(blk, set_format=set_format)
        self.setBoundingRegionGranularity(0)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        
    def is_editting(self):
        return self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction

    def onDocumentContentChanged(self):
        if self.hasFocus():   
            self.content_changed.emit(self)
        if self.repaint_on_changed:
            if not self.repainting:
                self.repaint_background()
        self.update()

    def paint_stroke(self, painter: QPainter):
        doc = self.document().clone()
        doc.setDocumentMargin(self.padding())
        layout = VerticalTextDocumentLayout(doc) if self.is_vertical else HorizontalTextDocumentLayout(doc)
        layout.line_spacing = self.line_spacing
        layout.letter_spacing = self.letter_spacing
        rect = self.rect()
        layout.setMaxSize(rect.width(), rect.height(), False)
        doc.setDocumentLayout(layout)

        layout.relayout_on_changed = False
        cursor = QTextCursor(doc)
        block = doc.firstBlock()
        stroke_pen = QPen(self.stroke_color, 0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                cfmt = fragment.charFormat()
                sw = pt2px(cfmt.fontPointSize()) * self.stroke_width
                stroke_pen.setWidthF(sw)
                pos1 = fragment.position()
                pos2 = pos1 + fragment.length()
                cursor.setPosition(pos1)
                cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                cfmt.setTextOutline(stroke_pen)
                cursor.mergeCharFormat(cfmt)
                it += 1
            block = block.next()
        doc.drawContents(painter)

    def repaint_background(self):
        empty = self.document().isEmpty()
        if self.repainting:
            return

        paint_stroke = self.stroke_width > 0
        paint_shadow = self.shadow_radius > 0 and self.shadow_strength > 0
        if not paint_shadow and not paint_stroke or empty:
            self.background_pixmap = None
            return
        
        self.repainting = True
        font_size = self.layout.max_font_size(to_px=True)
        img_array = None
        target_map = QPixmap(self.boundingRect().size().toSize())
        target_map.fill(Qt.GlobalColor.transparent)
        painter = QPainter(target_map)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        if paint_stroke:
            self.paint_stroke(painter)
        else:
            self.document().drawContents(painter)

        # shadow
        if paint_shadow:
            r = int(round(self.shadow_radius * font_size))
            xoffset, yoffset = int(self.shadow_offset[0] * font_size), int(self.shadow_offset[1] * font_size)
            shadow_map, img_array = apply_shadow_effect(target_map, self.shadow_color, self.shadow_strength, r)
            cm = painter.compositionMode()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
            painter.drawPixmap(xoffset, yoffset, shadow_map)
            painter.setCompositionMode(cm)

        painter.end()
        self.background_pixmap = target_map
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
        self.setVertical(blk.vertical)
        self.setRect(blk.bounding_rect())
        
        if blk.angle != 0:
            self.setRotation(blk.angle)
        
        set_char_fmt = False
        if blk.translation:
            set_char_fmt = True

        font_fmt = FontFormat()
        font_fmt.from_textblock(blk)
        if set_format:
            self.set_fontformat(font_fmt, set_char_format=set_char_fmt, set_stroke_width=False, set_effect=False)

        if not blk.rich_text:
            if blk.translation:
                self.setPlainText(blk.translation)
        else:
            self.setHtml(blk.rich_text)
            self.letter_spacing = 1.
            self.setLetterSpacing(font_fmt.letter_spacing, repaint_background=False)
        self.update_effect(font_fmt, repaint=False)
        self.setStrokeWidth(font_fmt.stroke_width, repaint=False)
        self.repaint_background()

    def setCenterTransform(self):
        center = self.boundingRect().center()
        self.setTransformOriginPoint(center)

    def rect(self) -> QRectF:
        return QRectF(self.pos(), self.boundingRect().size())

    def startReshape(self):
        self.oldRect = self.absBoundingRect()
        self.reshaping = True

    def endReshape(self):
        self.reshaped.emit(self)
        self.reshaping = False

    def padRect(self, rect: QRectF) -> QRectF:
        p = self.padding()
        P = p * 2
        return QRectF(rect.x() - p, rect.y() - p, rect.width() + P, rect.height() + P)
    
    def unpadRect(self, rect: QRectF) -> QRectF:
        p = -self.padding()
        P = p * 2
        return QRectF(rect.x() - p, rect.y() - p, rect.width() + P, rect.height() + P)

    def setRect(self, rect: Union[List, QRectF], padding=True, repaint=True) -> None:
        
        if isinstance(rect, List):
            rect = QRectF(*rect)
        if padding:
            rect = self.padRect(rect)
        self.setPos(rect.topLeft())
        self.prepareGeometryChange()
        self._display_rect = rect
        self.layout.setMaxSize(rect.width(), rect.height())
        self.setCenterTransform()
        if repaint:
            self.repaint_background()

    def documentSize(self):
        return self.layout.documentSize()

    def boundingRect(self) -> QRectF:
        br = super().boundingRect()
        if self._display_rect is not None:
            size = self.documentSize()
            br.setHeight(max(self._display_rect.height(), size.height()))
            br.setWidth(max(self._display_rect.width(), size.width()))
        return br

    def padding(self) -> float:
        return self.document().documentMargin()

    def setPadding(self, p: float):
        _p = self.padding()
        if _p >= p:
            return
        abr = self.absBoundingRect()
        self.layout.relayout_on_changed = False
        self.layout.updateDocumentMargin(p)
        self.layout.relayout_on_changed = True
        self.setRect(abr, repaint=False)

    def absBoundingRect(self, max_h=None, max_w=None, qrect=False) -> Union[List, QRectF]:
        br = self.boundingRect()
        P = 2 * self.padding()
        w, h = br.width() - P, br.height() - P
        pos = self.pos()
        x = pos.x() + self.padding()
        y = pos.y() + self.padding()
        if max_h is not None:
            y = min(max(0, y), max_h)
            y1 = y + h
            h = min(max_h, y1) - y
        if max_w is not None:
            x = min(max(0, x), max_w)
            x1 = x + w
            w = min(max_w, x1) - x
        if qrect:
            return QRectF(x, y, w, h)
        return [x, y, w, h]

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
        return isinstance(self.layout, VerticalTextDocumentLayout)

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

    def setVertical(self, vertical: bool):
        if self.blk is not None:
            self.blk.vertical = vertical

        valid_layout = True
        if self.layout is not None:
            if self.is_vertical == vertical:
                return
        else:
            valid_layout = False

        if valid_layout:
            rect = self.rect() if self.layout is not None else None
        
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        doc = self.document()
        html = doc.toHtml()
        doc_margin = doc.documentMargin()
        doc.disconnect()
        doc.documentLayout().disconnect()
        default_font = doc.defaultFont()

        doc = QTextDocument()
        doc.setDocumentMargin(doc_margin)
        if vertical:
            layout = VerticalTextDocumentLayout(doc)
        else:
            layout = HorizontalTextDocumentLayout(doc)
        
        self.layout = layout
        self.setDocument(doc)
        layout.size_enlarged.connect(self.on_document_enlarged)
        layout.documentSizeChanged.connect(self.docSizeChanged)
        doc.setDocumentLayout(layout)
        doc.setDefaultFont(default_font)
        doc.contentsChanged.connect(self.onDocumentContentChanged)
        
        if valid_layout:
            layout.setMaxSize(rect.width(), rect.height())
            doc.setHtml(html)

            self.setCenterTransform()
            self.setLineSpacing(self.line_spacing)
            self.repaint_background()

            if self.letter_spacing != 1:
                self.setLetterSpacing(self.letter_spacing, force=True)

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

    def setStrokeWidth(self, stroke_width: float, padding=True, repaint=True):
        if self.stroke_width == stroke_width:
            return

        if stroke_width > 0 and padding:
            p = self.layout.max_font_size(to_px=True) * stroke_width / 2
            self.setPadding(p)

        self.stroke_width = stroke_width

        if repaint:
            self.repaint_background()
            self.update()

    def setStrokeColor(self, scolor):
        self.stroke_color = scolor if isinstance(scolor, QColor) else QColor(*scolor)
        self.repaint_background()
        self.update()

    def get_scale(self) -> float:
        tl = self.topLevelItem()
        if tl is not None:
            return tl.scale()
        else:
            return self.scale()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        br = self.boundingRect()
        painter.save()
        if self.background_pixmap is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(br.toRect(), self.background_pixmap)

        draw_rect = self.draw_rect and not self.under_ctrl
        if self.isSelected() and not self.is_editting():
            pen = QPen(TEXTRECT_SELECTED_COLOR, 3.5 / self.get_scale(), Qt.PenStyle.DashLine)
            painter.setPen(pen)
            # painter.drawRect(br)
            painter.drawRect(self.unpadRect(br))
        elif draw_rect:
            pen = QPen(TEXTRECT_SHOW_COLOR, 3 / self.get_scale(), Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.unpadRect(br))
        
        painter.restore()
        option.state = QStyle.State_None
        super().paint(painter, option, widget)

    def startEdit(self) -> None:
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.setFocus()
        self.begin_edit.emit(self.idx)

    def endEdit(self) -> None:
        self.end_edit.emit(self.idx)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setFocus()

    def isEditing(self) -> bool:
        return self.textInteractionFlags() == Qt.TextEditorInteraction
    
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
        if event.button() == Qt.MouseButton.LeftButton:
            self.oldPos = self.pos()
            self.leftbutton_pressed.emit(self.idx)
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.oldPos != self.pos():
                self.moved.emit()
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_move.emit(self.idx)
        return super().hoverMoveEvent(event)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_enter.emit(self.idx)
        return super().hoverEnterEvent(event)

    def toPixmap(self) -> QPixmap:
        pixmap = QPixmap(self.boundingRect().size().toSize())
        pixmap.fill(Qt.GlobalColor.transparent)
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
        if alignment == Qt.AlignmentFlag.AlignLeft:
            alignment = 0
        elif alignment == Qt.AlignmentFlag.AlignRight:
            alignment = 2
        else:
            alignment = 1
        weight = font.weight()
        # https://doc.qt.io/qt-5/qfont.html#Weight-enum, 50 is normal
        if weight == 0:
            weight = 50
        
        return FontFormat(
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
            self.line_spacing,
            self.letter_spacing,
            self.opacity(),
            self.shadow_radius,
            self.shadow_strength, 
            self.shadow_color,
            self.shadow_offset
        )

    def set_fontformat(self, ffmat: FontFormat, set_char_format=False, set_stroke_width=True, set_effect=True):
        if self.is_vertical != ffmat.vertical:
            self.setVertical(ffmat.vertical)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        format = cursor.charFormat()
        font = self.document().defaultFont()
        
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        font.setFamily(ffmat.family)
        font.setPointSize(ffmat.size)
        font.setBold(ffmat.bold)

        self.document().setDefaultFont(font)
        format.setFont(font)
        format.setForeground(QColor(*ffmat.frgb))
        format.setFontWeight(ffmat.weight)
        format.setFontItalic(ffmat.italic)
        format.setFontUnderline(ffmat.underline)
        if not ffmat.vertical:
            format.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
            format.setFontLetterSpacing(ffmat.letter_spacing * 100)
        cursor.setCharFormat(format)
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.setBlockCharFormat(format)
        if set_char_format:
            cursor.setCharFormat(format)
        cursor.clearSelection()
        # https://stackoverflow.com/questions/37160039/set-default-character-format-in-qtextdocument
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        self.stroke_color = QColor(ffmat.srgb[0], ffmat.srgb[1], ffmat.srgb[2])

        if set_effect:
            self.update_effect(ffmat)
        if set_stroke_width:
            self.setStrokeWidth(ffmat.stroke_width)
        
        alignment = [Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignRight][ffmat.alignment]
        doc = self.document()
        op = doc.defaultTextOption()
        op.setAlignment(alignment)
        doc.setDefaultTextOption(op)
        
        if ffmat.vertical:
            self.setLetterSpacing(ffmat.letter_spacing)
        self.letter_spacing = ffmat.letter_spacing
        self.setLineSpacing(ffmat.line_spacing)

    def updateBlkFormat(self):
        fmt = self.get_fontformat()
        self.blk.default_stroke_width = fmt.stroke_width
        self.blk.line_spacing = fmt.line_spacing
        self.blk.letter_spacing = fmt.letter_spacing
        self.blk.font_family = fmt.family
        self.blk.font_size = pt2px(fmt.size)
        self.blk.font_weight = fmt.weight
        self.blk._alignment = fmt.alignment
        self.blk.shadow_color = self.shadow_color
        self.blk.shadow_radius = self.shadow_radius
        self.blk.shadow_strength = self.shadow_strength
        self.blk.shadow_offset = self.shadow_offset
        self.blk.opacity = self.opacity()
        self.blk.set_font_colors(fmt.frgb, fmt.srgb, accumulate=False)

    def setLineSpacing(self, line_spacing: float):
        self.line_spacing = line_spacing
        self.layout.setLineSpacing(self.line_spacing)
        self.repaint_background()

    def setLetterSpacing(self, letter_spacing: float, repaint_background=True, force=False):
        if self.letter_spacing == letter_spacing and not force:
            return
        self.letter_spacing = letter_spacing
        if self.is_vertical:
            self.layout.setLetterSpacing(letter_spacing)
        else:
            char_fmt = QTextCharFormat()
            char_fmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
            char_fmt.setFontLetterSpacing(letter_spacing * 100)
            cursor = QTextCursor(self.document())
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.mergeCharFormat(char_fmt)
            # cursor.mergeBlockCharFormat(char_fmt)
        if repaint_background:
            self.repaint_background()

    def get_char_fmts(self) -> List[QTextCharFormat]:
        cursor = self.textCursor()
        
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        char_fmts = []
        while True:
            
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
            cursor.clearSelection()
            char_fmts.append(cursor.charFormat())
            if cursor.atEnd():
                break
        return char_fmts

    def update_effect(self, fmt: FontFormat, repaint=True):
        self.setOpacity(fmt.opacity)
        self.shadow_radius = fmt.shadow_radius
        self.shadow_strength = fmt.shadow_strength
        self.shadow_color = fmt.shadow_color
        self.shadow_offset = fmt.shadow_offset
        if self.shadow_radius > 0:
            self.setPadding(self.layout.max_font_size(to_px=True))
        if repaint:
            self.repaint_background()


    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_V and e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if self.isEditing() is not None:
                e.accept()
                self.pasted.emit(self.idx)
                return
        return super().keyPressEvent(e)
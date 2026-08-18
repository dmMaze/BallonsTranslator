from functools import cached_property, lru_cache
from typing import List, Optional, Sequence, Tuple

from qtpy.QtCore import QPointF, QRectF, Signal, QSizeF
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QFont,
    QFontMetricsF,
    QPixmap,
    QTextBlock,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFrame,
)

from ballontranslator.utils.fontformat import FontFormat, LineSpacingType, pt2px
from .font_family import qfont_with_family
from .annotations import letter_spacing_value, line_spacing_values


def selection_segments_excluding(
    start: int,
    end: int,
    exclusions: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Subtract logical ranges from one selection range.

    >>> selection_segments_excluding(0, 6, ((1, 3), (4, 5)))
    [(0, 1), (3, 4), (5, 6)]
    """
    segments = [(start, end)] if end > start else []
    for exclusion_start, exclusion_end in exclusions:
        remaining = []
        for segment_start, segment_end in segments:
            if (
                exclusion_end <= segment_start
                or exclusion_start >= segment_end
            ):
                remaining.append((segment_start, segment_end))
                continue
            if segment_start < exclusion_start:
                remaining.append((segment_start, exclusion_start))
            if exclusion_end < segment_end:
                remaining.append((exclusion_end, segment_end))
        segments = remaining
    return segments


def paint_context_without_selection_ranges(
    document: QTextDocument,
    block: QTextBlock,
    context: QAbstractTextDocumentLayout.PaintContext,
    exclusions: Sequence[Tuple[int, int]],
) -> QAbstractTextDocumentLayout.PaintContext:
    """Copy a paint context while removing layout-owned selection cells.

    >>> callable(paint_context_without_selection_ranges)
    True
    """
    if not exclusions:
        return context
    delegated = QAbstractTextDocumentLayout.PaintContext()
    delegated.clip = QRectF(context.clip)
    delegated.cursorPosition = context.cursorPosition
    delegated.palette = context.palette
    delegated.selections = []
    block_length = max(0, block.length() - 1)
    for selection in context.selections:
        if not selection.cursor.hasSelection():
            delegated.selections.append(selection)
            continue
        local_start = max(
            0, selection.cursor.selectionStart() - block.position()
        )
        local_end = min(
            block_length,
            selection.cursor.selectionEnd() - block.position(),
        )
        for start, end in selection_segments_excluding(
            local_start, local_end, exclusions
        ):
            copied = QAbstractTextDocumentLayout.Selection()
            copied.cursor = QTextCursor(document)
            copied.cursor.setPosition(block.position() + start)
            copied.cursor.setPosition(
                block.position() + end,
                QTextCursor.MoveMode.KeepAnchor,
            )
            copied.format = selection.format
            delegated.selections.append(copied)
    return delegated

def _font_metrics(ffamily: str, size: float, weight: int, italic: bool) -> QFontMetricsF:
    # QFont's string constructor splits comma-bearing family names into a
    # fallback list. The shared boundary preserves one database family name.
    font = qfont_with_family(QFont(), ffamily)
    font.setPointSizeF(size)
    font.setWeight(weight)
    font.setItalic(italic)
    return QFontMetricsF(font)

@lru_cache(maxsize=2048)
def get_punc_rect(char: str, ffamily: str, size: float, weight: int, italic: bool) -> List[QRectF]:
    fm = _font_metrics(ffamily, size, weight, italic)
    br = [fm.tightBoundingRect(char), fm.boundingRect(char)]
    return br

@lru_cache(maxsize=2048)
def get_char_width(char: str, ffamily: str, size: float, weight: int, italic: bool) -> int:
    fm = _font_metrics(ffamily, size, weight, italic)
    return fm.horizontalAdvance(char)

def _block_cursor_position(block: QTextBlock, cursor_position: int) -> int:
    layout = block.layout()
    if cursor_position < -1:
        if not layout.preeditAreaText():
            return -1
        # Qt encodes IME preedit cursor positions as -(preeditCursor + 2).
        return layout.preeditAreaPosition() - (cursor_position + 2)

    block_position = block.position()
    if block_position <= cursor_position < block_position + block.length():
        return cursor_position - block_position
    return -1


class CharFontFormat:
    def __init__(
        self,
        fcmt: QTextCharFormat,
        letter_spacing_fallback: float = 1.0,
    ) -> None:
        font = fcmt.font()
        self.font = font
        self.font_metrics = QFontMetricsF(font)
        self.letter_spacing = letter_spacing_value(
            fcmt,
            letter_spacing_fallback,
        )

    @cached_property
    def br(self) -> QRectF:
        # return get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())[1]
        _, br1 = get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())
        _, br2 = get_punc_rect('木', self.family, self.size, self.weight, self.font.italic())
        return QRectF(min(br1.left(), br2.left()), br2.top(), max(br1.right(), br2.right()) - min(br1.left(), br2.left()), br2.height())

    @cached_property
    def tbr(self) -> QRectF:
        # return get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())[0]
        br1, _ = get_punc_rect('啊', self.family, self.size, self.weight, self.font.italic())
        br2, _ = get_punc_rect('木', self.family, self.size, self.weight, self.font.italic())
        return QRectF(min(br1.left(), br2.left()), br2.top(), max(br1.right(), br2.right()) - min(br1.left(), br2.left()), br2.height())

    @cached_property
    def space_width(self) -> int:
        return get_char_width(' ', self.family, self.size, self.weight, self.font.italic())

    def punc_rect(self, punc: str, family: str = None) -> List[QRectF]:
        if family is None:
            family = self.family
        return get_punc_rect(punc, family, self.size, self.weight, self.font.italic())

    @property
    def family(self) -> str:
        return self.font.family()

    @property
    def weight(self) -> float:
        return self.font.weight()

    @property
    def size(self) -> float:
        return self.font.pointSizeF()

class SceneTextLayout(QAbstractTextDocumentLayout):
    size_enlarged = Signal()
    def __init__(self, doc: QTextDocument, fontformat: FontFormat) -> None:
        super().__init__(doc)
        self.max_height = 0
        self.max_width = 0
        self.available_width = 0
        self.available_height = 0
        self.line_spacing = fontformat.line_spacing
        self.letter_spacing = fontformat.letter_spacing
        self.linespacing_type = fontformat.line_spacing_type
        self.fontformat = fontformat
        self.render_delegate = None
        self.layout_generation = 0
        self.render_failure_handler = None
        self.defer_cursor_paint = False
        self.deferred_cursor_position = -1
        self.publishing_size_enlargement = False
        # QWidgetTextControl routes its mouse and drag hit tests through this
        # layout.  Nonlinear visual effects can therefore restore source
        # coordinates here without replacing Qt's editing state machine.
        self.input_point_mapper = None

        self.x_offset_lst = []
        self.y_offset_lst = []

        self.block_charfmt_lst = []
        self.block_qcharfmt_lst = []
        self.block_ideal_width = []
        self.need_ideal_width = False
        self.block_ideal_height = []
        self.need_ideal_height = False
        self._map_charidx2frag = []
        self._max_font_size = -1

        self.foreground_pixmap: QPixmap = None
        self.relayout_on_changed = True

        # Effect padding is derived layout state, not rich-text content.
        # QTextDocument margins create undo entries in supported Qt bindings.
        self._effect_padding = max(0.0, float(doc.documentMargin()))

        # relative bottom/right
        self.shrink_height = 0 
        self.shrink_width = 0

        # The upstream vertical-stroke renderer clones a document and reuses
        # the already-computed draw offsets.  Keep that neutral-state path
        # intact; feature rendering only replaces it while Glyph Slant is
        # active.
        self._is_painting_stroke = False
        self._draw_offset = []
        self.text_padding = 0

    def source_cursor_rect(self, cursor_position: int):
        """Return a layout-owned caret rectangle, or defer to Qt.

        Horizontal layout uses Qt's native cursor geometry. Vertical layout
        overrides this because its caret is horizontal.

        >>> SceneTextLayout.source_cursor_rect(None, 0) is None
        True
        """
        return None

    def annotation_ink_bounds(self) -> QRectF:
        """Return annotation paint overflow without changing layout bounds."""
        return QRectF()

    def base_ink_bounds(self) -> QRectF:
        """Return layout-owned base-text paint overflow, when required."""
        return QRectF()

    def _begin_layout_generation(self):
        self.layout_generation += 1

    def _emit_size_enlarged(self) -> None:
        """Publish a resize without exposing half-settled paint geometry."""
        self.publishing_size_enlargement = True
        try:
            self.size_enlarged.emit()
        finally:
            self.publishing_size_enlargement = False

    def map_input_point(self, point: QPointF) -> QPointF:
        mapper = self.input_point_mapper
        return QPointF(point) if mapper is None else mapper(QPointF(point))

    def _report_render_failure(self, error, effect_pass=False):
        handler = self.render_failure_handler
        if handler is not None:
            handler(error, effect_pass)

    def setMaxSize(self, max_width: int, max_height: int, relayout=True):
        self.max_height = max_height
        self.max_width = max_width
        doc_margin = self._effect_padding * 2
        self.available_width = max(max_width -  doc_margin, 0)
        self.available_height = max(max_height - doc_margin, 0)
        if relayout:
            self.reLayoutForResize()

    def reLayoutForResize(self):
        self.reLayout()

    def block_line_spacing(
        self,
        block: QTextBlock,
    ) -> tuple[float, LineSpacingType]:
        return line_spacing_values(
            block.blockFormat(),
            self.line_spacing,
            self.linespacing_type,
        )

    def calculate_line_spacing(
        self,
        size: float,
        line_spacing: float = 1,
        linespacing_type: Optional[LineSpacingType] = None,
    ) -> float:
        if linespacing_type is None:
            linespacing_type = self.linespacing_type
        if linespacing_type == LineSpacingType.Proportional:
            return line_spacing * size
        elif linespacing_type == LineSpacingType.Distance:
            return line_spacing * 10 + size
        else:
            raise Exception(f'Invalid line spacing type: {linespacing_type}')

    def identity_linespacing(
        self,
        linespacing_type: Optional[LineSpacingType] = None,
    ) -> float:
        if linespacing_type is None:
            linespacing_type = self.linespacing_type
        if linespacing_type == LineSpacingType.Proportional:
            return 1.
        elif linespacing_type == LineSpacingType.Distance:
            return 0.
        else:
            raise Exception(f'Invalid line spacing type: {linespacing_type}')

    def blockBoundingRect(self, block: QTextBlock) -> QRectF:
        if not block.isValid():
            return QRectF()
        br = block.layout().boundingRect()
        rect = QRectF(0, 0, br.width(), br.height())
        return rect

    def setEffectPadding(self, padding):
        self._effect_padding = max(0.0, float(padding))
        doubled_margin = self._effect_padding * 2
        self.max_height = doubled_margin + self.available_height
        self.max_width = doubled_margin + self.available_width

    def effectPadding(self) -> float:
        return self._effect_padding

    def documentSize(self) -> QSizeF:
        return QSizeF(self.max_width, self.max_height)

    def pageCount(self) -> int:
        return 1

    def frameBoundingRect(self, frame: QTextFrame) -> QRectF:
        return QRectF(
            0,
            0,
            max(self.document().pageSize().width(), self.max_width),
            2147483647,
        )

    def documentChanged(self, position: int, charsRemoved: int, charsAdded: int) -> None:
        if not self.relayout_on_changed:
            return
        self.reLayoutEverything()
        
    def reLayoutEverything(self):
        self._max_font_size = -1
        block = self.document().firstBlock()
        self.block_charfmt_lst = []
        self.block_qcharfmt_lst = []
        self.block_ideal_width = []
        self.block_ideal_height = []
        self._map_charidx2frag = []
        while block.isValid():
            charfmt_lst, qcharfmt_lst, ideal_width, char_idx = [], [], -1, 0
            ideal_height = 0
            charidx_map = {}
            it = block.begin()
            frag_idx = 0
            while not it.atEnd():
                fragment = it.fragment()
                fcmt = fragment.charFormat()
                cfmt = CharFontFormat(fcmt, self.letter_spacing)
                charfmt_lst.append(cfmt)
                qcharfmt_lst.append(QTextCharFormat(fcmt))
                if cfmt.size > self._max_font_size:
                    self._max_font_size = cfmt.size

                if self.need_ideal_width:
                    w_ = cfmt.br.width()
                    if ideal_width < w_:
                        ideal_width = w_

                if self.need_ideal_height:
                    h_ = cfmt.punc_rect('木fg')[0].height()
                    if ideal_height < h_:
                        ideal_height = h_

                text_len = fragment.length()
                for _ in range(text_len):
                    charidx_map[char_idx] = frag_idx
                    char_idx += 1
                it += 1
                frag_idx += 1

            self.block_charfmt_lst.append(charfmt_lst)
            self.block_qcharfmt_lst.append(qcharfmt_lst)
            self.block_ideal_width.append(ideal_width)
            self.block_ideal_height.append(ideal_height)
            self._map_charidx2frag.append(charidx_map)
            block = block.next()
        self.reLayout()

    def max_font_size(self, to_px=False) -> float:
        fs = self._max_font_size if self._max_font_size > 0 else self.document().defaultFont().pointSizeF()
        if to_px:
            fs = pt2px(fs)
        return fs

    def minSize(self):
        return (self.shrink_height + self.text_padding, self.shrink_width + self.text_padding)
    
    def get_char_fontfmt(self, block_number: int, char_idx: int) -> CharFontFormat:
        charidx2frag_map = self._map_charidx2frag[block_number]
        if len(charidx2frag_map) == 0:
            return None
        if char_idx not in charidx2frag_map:    # caused by inputmethod
            char_idx = len(charidx2frag_map) - 1
        frag_idx = charidx2frag_map[char_idx]
        return self.block_charfmt_lst[block_number][frag_idx]

    def fragment_format_ranges(
        self,
        block_number: int,
        start: int,
        end: int,
    ) -> Tuple[Tuple[int, int, QTextCharFormat], ...]:
        """Return indexed QTextCharFormat runs intersecting one block range."""
        position_map = self._map_charidx2frag[block_number]
        if not position_map or end <= start:
            return ()
        start = max(0, start)
        end = min(end, len(position_map))
        if end <= start:
            return ()
        ranges = []
        range_start = start
        fragment_index = position_map[start]
        for position in range(start + 1, end):
            candidate = position_map[position]
            if candidate == fragment_index:
                continue
            ranges.append((
                range_start,
                position,
                self.block_qcharfmt_lst[block_number][fragment_index],
            ))
            range_start = position
            fragment_index = candidate
        ranges.append((
            range_start,
            end,
            self.block_qcharfmt_lst[block_number][fragment_index],
        ))
        return tuple(ranges)

from bisect import bisect_right
from functools import cached_property, lru_cache
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from qtpy.QtCore import QPointF, QRectF, Signal, QSizeF
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QFont,
    QFontMetricsF,
    QPixmap,
    QRawFont,
    QTextBlock,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFrame,
)

from ballontranslator.utils.fontformat import FontFormat, LineSpacingType, TextAlignment, VerticalAlignment, pt2px
from ballontranslator.utils.bubble_shape import BubbleShape
from .font_family import qfont_with_family
from .annotations import AnnotationProperty, emphasis_values, letter_spacing_value, line_spacing_values


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
    selections = []
    block_length = max(0, block.length() - 1)
    for selection in context.selections:
        if not selection.cursor.hasSelection():
            selections.append(selection)
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
            selections.append(copied)
    # PaintContext list access may return a copy; publish the completed list.
    delegated.selections = selections
    return delegated

def _metrics_font(ffamily: str, size: float, weight: int, italic: bool) -> QFont:
    # QFont's string constructor splits comma-bearing family names into a
    # fallback list. The shared boundary preserves one database family name.
    font = qfont_with_family(QFont(), ffamily)
    font.setPointSizeF(size)
    font.setWeight(weight)
    font.setItalic(italic)
    return font

@lru_cache(maxsize=2048)
def get_punc_rect(char: str, ffamily: str, size: float, weight: int, italic: bool) -> List[QRectF]:
    """Return representative ink metrics with a full-metric fallback.

    The horizontal baseline probe must use the selected face's line metrics
    when none of its reference glyphs exist. Ordinary CJK/punctuation queries
    retain valid substitute ink because Qt also uses it to draw those glyphs.
    Empty tight boxes fall back to ordinary bounds, then line metrics.

    >>> len(get_punc_rect('A', 'Sans Serif', 12, QFont.Weight.Normal, False))
    2
    """
    font = _metrics_font(ffamily, size, weight, italic)
    fm = QFontMetricsF(font)
    tight = fm.tightBoundingRect(char)
    bounds = fm.boundingRect(char)
    if char == '木fg':
        raw_font = QRawFont.fromFont(font)
        if raw_font.isValid() and not any(
            raw_font.supportsCharacter(ord(character)) for character in char
        ):
            # Substitute probe ink can have an unrelated baseline. This does
            # not apply to vertical cells, which use actual CJK fallback ink.
            tight = QRectF(
                bounds.left(), -fm.ascent(), max(0.0, bounds.width()),
                max(0.0, fm.ascent() + fm.descent()),
            )
    if tight.width() <= 0 or tight.height() <= 0:
        tight = QRectF(bounds)
    if tight.width() <= 0 or tight.height() <= 0:
        tight = QRectF(
            0.0,
            -fm.ascent(),
            max(0.0, fm.horizontalAdvance(char)),
            max(0.0, fm.ascent() + fm.descent()),
        )
    return [tight, bounds]

@lru_cache(maxsize=2048)
def get_char_width(char: str, ffamily: str, size: float, weight: int, italic: bool) -> float:
    fm = QFontMetricsF(_metrics_font(ffamily, size, weight, italic))
    return fm.horizontalAdvance(char)


def clear_font_metrics_cache() -> None:
    """Discard metrics that may have been measured using a fallback font."""
    get_char_width.cache_clear()
    get_punc_rect.cache_clear()


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
        self.flow_profile = None
        self.flow_slots: List[QRectF] = []
        self.flow_line = 0
        self.flow_overflow = False
        self.flow_vertical = False
        self.flow_step = 1.
        self.flow_ink: Optional[QRectF] = None
        self.flow_breaks: Optional[List[int]] = None
        self.flow_actual_ends: List[int] = []
        self.line_spacing = fontformat.line_spacing
        self.letter_spacing = fontformat.letter_spacing
        self.linespacing_type = fontformat.line_spacing_type
        self.fontformat = fontformat
        self.render_delegate = None
        self.layout_generation = 0
        self.render_failure_handler = None
        self.defer_cursor_paint = False
        self.deferred_cursor_position = -1
        # Installed only for one native paint call while effects capture Qt's
        # authoritative selection/IME context; the renderer clears it in
        # finally before returning to the event loop.
        self.paint_context_observer: Optional[
            Callable[[QAbstractTextDocumentLayout.PaintContext], None]
        ] = None
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
        self._block_fragment_ends: List[List[int]] = []
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

    def _observe_paint_context(
        self,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> None:
        observer = self.paint_context_observer
        if observer is not None:
            observer(context)

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

    def set_flow_shape(self, shape: Optional[BubbleShape]) -> None:
        from .bubble_layout import BubbleProfile

        if shape is None:
            self.flow_profile = None
        elif self.flow_profile is None or self.flow_profile.shape != shape:
            self.flow_profile = BubbleProfile(shape)
        self.flow_slots = []
        self.flow_overflow = False

    def flow_layout_passes(self, vertical: bool = False) -> Iterator[None]:
        """Try contour bands until Qt consumes the document inside them.

        >>> callable(SceneTextLayout.flow_layout_passes)
        True
        """
        from .bubble_layout import balanced_flow_breaks, measure_flow_units

        self.flow_overflow = False
        self.flow_breaks = None
        self.flow_ink = None
        if self.flow_profile is None:
            yield
            return
        self.flow_vertical = vertical
        if not vertical:
            ink = QRectF()
            block = self.document().firstBlock()
            while block.isValid():
                fragments = block.begin()
                for cfmt in self.block_charfmt_lst[block.blockNumber()]:
                    ink = ink.united(cfmt.font_metrics.tightBoundingRect(fragments.fragment().text()))
                    fragments += 1
                block = block.next()
            if not ink.isEmpty():
                self.flow_ink = ink
        # Match layoutBlock's tight column ink; ordinary font bounds can be
        # narrower on CoreText and falsely reject every candidate font size.
        metrics = self.block_ideal_height
        if vertical:
            metrics = [fmt.tbr.width() for formats in self.block_charfmt_lst for fmt in formats]
        elif self.flow_ink is not None:
            metrics = [self.flow_ink.height()]
        extent = max(max(metrics, default=self.max_font_size(to_px=True)), 1.) + .5
        formats = [fmt for block_formats in self.block_qcharfmt_lst for fmt in block_formats]
        if any(fmt.hasProperty(AnnotationProperty.RUBY_TEXT) for fmt in formats):
            extent += self.max_font_size(to_px=True) * .75 + 2.
        if any(emphasis_values(fmt)[0] != 'none' for fmt in formats):
            extent += self.max_font_size(to_px=True) * .5
        # Ink, not the interline leading, is constrained by the bubble edge.
        block = self.document().firstBlock()
        step = extent
        while block.isValid():
            spacing, kind = self.block_line_spacing(block)
            step = max(step, self.calculate_line_spacing(extent, spacing, kind))
            block = block.next()
        self.flow_step = step
        width, height = self.available_width, self.available_height
        across, inline = (width, height) if vertical else (height, width)
        # Keep half an ink band of breathing room, with a four-pixel floor.
        inset = max(4., extent / 2)
        maximum = max(1, min(64, int(max(0., across - inset * 2 - extent) / step) + 1))
        units = []
        fallback = None
        for count in range(1, maximum + 1):
            self.flow_slots = []
            self.flow_line = 0
            self.flow_overflow = False
            self.flow_actual_ends = []
            slack = across - ((count - 1) * step + extent)
            origin = slack / 2
            if vertical and self.fontformat.alignment == TextAlignment.Left:
                origin = inset
            elif vertical and self.fontformat.alignment == TextAlignment.Right:
                origin = slack - inset
            starts = [origin + index * step for index in range(count)]
            spans = [self.flow_profile.band(start / max(across, 1), (start + extent) / max(across, 1), vertical)
                     for start in starts]
            common_left = max(low for low, _ in spans)
            common_right = min(high for _, high in spans)
            # Keep one visual axis, but derive it from the usable spans rather
            # than the frame center, which can lie in a tail or near a cut.
            center = sum((low + high) / 2 for low, high in spans) / count
            center = min(max(center, common_left), common_right)
            for start, (low, high) in zip(starts, spans):
                half = max(0., min(center - low, high - center) * inline - inset)
                begin, length = center * inline - half, half * 2
                # Keep a common top/bottom across columns without reserving
                # unused space on the opposite side of an irregular bubble.
                if vertical and self.fontformat.vertical_alignment == VerticalAlignment.Top:
                    begin = common_left * inline + inset
                    length = max(0., high * inline - inset - begin)
                elif vertical and self.fontformat.vertical_alignment == VerticalAlignment.Bottom:
                    begin = low * inline + inset
                    length = max(0., common_right * inline - inset - begin)
                self.flow_slots.append(QRectF(start, begin, extent, length) if vertical
                                       else QRectF(begin, start, length, extent))
            if vertical:
                self.flow_slots.reverse()
            if units:
                self.flow_breaks = balanced_flow_breaks(units, [slot.height() if vertical else slot.width()
                                                               for slot in self.flow_slots])
            yield
            if not self.flow_overflow and self.flow_line == count:
                if not units or (self.flow_breaks is not None and self.flow_actual_ends == self.flow_breaks):
                    return
                # Keep a fully occupied native wrap for editing if no legal
                # plan succeeds; the final trial can leave many empty bands.
                if fallback is None:
                    fallback = self.flow_slots, self.flow_breaks
            if count == 1:
                units = measure_flow_units(self, vertical)
        if fallback is not None and self.flow_slots is not fallback[0]:
            self.flow_slots, self.flow_breaks = fallback
            self.flow_line = 0
            self.flow_actual_ends = []
            self.flow_overflow = False
            yield
        # Native wrapping may split a protected word or punctuation run. It
        # remains editable, but must not qualify as a successful automatic fit.
        self.flow_overflow = True

    def flow_rect(self, index: int) -> QRectF:
        if index >= len(self.flow_slots):
            self.flow_overflow = True
            # Overset remains editable in successive rows/columns, rather
            # than stacking every exhausted line at the frame origin.
            last = self.flow_slots[-1]
            distance = (index - len(self.flow_slots) + 1) * self.flow_step
            if self.flow_vertical:
                return QRectF(last.left() - distance, 0, last.width(), max(1., self.available_height))
            return QRectF(0, last.top() + distance, max(1., self.available_width), last.height())
        rect = self.flow_slots[index]
        if rect.width() < 1 or rect.height() < 1:
            self.flow_overflow = True
        return rect

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
        
    def reLayoutEverything(self) -> None:
        """Rebuild format metrics and their UTF-16 index before placing text.

        >>> callable(SceneTextLayout.reLayoutEverything)
        True
        """
        self._max_font_size = -1
        block = self.document().firstBlock()
        self.block_charfmt_lst = []
        self.block_qcharfmt_lst = []
        self.block_ideal_width = []
        self.block_ideal_height = []
        self._block_fragment_ends = []
        # Qt interns formats within a document. Reuse their read-only metrics
        # only in this rebuild, so font and layout changes cannot leave stale
        # measurements behind across generations.
        format_metrics = {}
        while block.isValid():
            charfmt_lst, qcharfmt_lst, ideal_width, char_idx = [], [], -1, 0
            ideal_height = 0
            fragment_ends = []
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                format_index = fragment.charFormatIndex()
                metrics = format_metrics.get(format_index)
                if metrics is None:
                    fcmt = fragment.charFormat()
                    cfmt = CharFontFormat(fcmt, self.letter_spacing)
                    width = cfmt.br.width() if self.need_ideal_width else -1
                    height = (
                        cfmt.punc_rect('木fg')[0].height()
                        if self.need_ideal_height else 0
                    )
                    metrics = cfmt, QTextCharFormat(fcmt), cfmt.size, width, height
                    format_metrics[format_index] = metrics
                cfmt, qfmt, size, width, height = metrics
                charfmt_lst.append(cfmt)
                qcharfmt_lst.append(qfmt)
                if size > self._max_font_size:
                    self._max_font_size = size

                if self.need_ideal_width:
                    if ideal_width < width:
                        ideal_width = width

                if self.need_ideal_height:
                    if ideal_height < height:
                        ideal_height = height

                # Qt fragment lengths are UTF-16 units, not Python characters.
                # Store one exclusive end per format run, not one dict entry
                # per unit. Rebuild with the metrics for this generation.
                char_idx += fragment.length()
                fragment_ends.append(char_idx)
                it += 1

            self.block_charfmt_lst.append(charfmt_lst)
            self.block_qcharfmt_lst.append(qcharfmt_lst)
            self.block_ideal_width.append(ideal_width)
            self.block_ideal_height.append(ideal_height)
            self._block_fragment_ends.append(fragment_ends)
            block = block.next()
        self.reLayout()

    def max_font_size(self, to_px=False) -> float:
        fs = self._max_font_size if self._max_font_size > 0 else self.document().defaultFont().pointSizeF()
        if to_px:
            fs = pt2px(fs)
        return fs

    def minSize(self):
        return (self.shrink_height + self.text_padding, self.shrink_width + self.text_padding)
    
    def get_char_fontfmt(
        self, block_number: int, char_idx: int
    ) -> Optional[CharFontFormat]:
        ends = self._block_fragment_ends[block_number]
        if not ends or ends[-1] == 0:
            return None
        if char_idx < 0 or char_idx >= ends[-1]:
            # Preserve the existing last-character fallback during IME preedit.
            char_idx = ends[-1] - 1
        frag_idx = bisect_right(ends, char_idx)
        return self.block_charfmt_lst[block_number][frag_idx]

    def largest_font_format(
        self, block_number: int, start: int, end: int
    ) -> Optional[CharFontFormat]:
        """Find the first largest font in a line without visiting each unit.

        Seed the unclipped start so IME fallback wins equal-size ties. The strict
        size comparison also preserves the fallback for fonts without point sizes.

        >>> callable(SceneTextLayout.largest_font_format)
        True
        """
        if end <= start:
            return None
        first = self.get_char_fontfmt(block_number, start)
        if first is None:
            return None
        best = None
        best_size = -1
        first_size = first.size
        if first_size > best_size:
            best, best_size = first, first_size
        ends = self._block_fragment_ends[block_number]
        position = max(0, start)
        limit = min(end, ends[-1])
        index = bisect_right(ends, position)
        formats = self.block_charfmt_lst[block_number]
        while position < limit:
            run_end = ends[index]
            if run_end > position:
                candidate = formats[index]
                size = candidate.size
                if size > best_size:
                    best, best_size = candidate, size
                position = run_end
            index += 1
        return best

    def fragment_format_ranges(
        self,
        block_number: int,
        start: int,
        end: int,
    ) -> Tuple[Tuple[int, int, QTextCharFormat], ...]:
        """Return intersecting formats by run, independent of character count.

        Lookup takes O(log F + K) for F runs and K overlapping runs.

        >>> callable(SceneTextLayout.fragment_format_ranges)
        True
        """
        ends = self._block_fragment_ends[block_number]
        if not ends or end <= start:
            return ()
        start = max(0, start)
        end = min(end, ends[-1])
        if end <= start:
            return ()
        ranges = []
        fragment_index = bisect_right(ends, start)
        formats = self.block_qcharfmt_lst[block_number]
        while ends[fragment_index] < end:
            run_end = ends[fragment_index]
            if run_end > start:
                ranges.append((start, run_end, formats[fragment_index]))
                start = run_end
            fragment_index += 1
        ranges.append((start, end, formats[fragment_index]))
        return tuple(ranges)

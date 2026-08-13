"""Transform-aware state layered on top of the existing text layouts."""

from __future__ import annotations

from itertools import count
from typing import Hashable, Iterator, Optional, Tuple, TYPE_CHECKING

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QPainter,
    QPen,
    QTextBlock,
    QTextCharFormat,
    QTextDocument,
    QTextLine,
    QTransform,
)

from ballontranslator.utils import shared as C
from .glyph import (
    GLYPH_STROKE_FORMAT_PROPERTY,
    GLOBAL_GLYPH_GEOMETRY_CACHE,
    GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE,
    draw_slanted_glyph_mask,
    draw_slanted_line,
    draw_uniform_glyph_geometries,
    slanted_line_geometry,
    GlyphGeometry,
)
from .emphasis import draw_emphasis_marks, emphasis_ink_bounds
from .ruby import draw_ruby_placement, ruby_side_margins
from .indexing import _utf16_char_at, _utf16_length

if TYPE_CHECKING:
    from ..layout import SceneTextLayout


_LAYOUT_CACHE_TOKENS = count(1)


class LayoutGlyphGeometryCache:
    """Route committed geometry globally and preview geometry transiently."""

    def __init__(self, renderer: GlyphSlantLayoutRenderer) -> None:
        self.renderer = renderer
        self.layout_token = next(_LAYOUT_CACHE_TOKENS)
        self.persistent = True

    def _global_key(self, key: Hashable) -> tuple:
        self.renderer.ensure_layout_generation()
        return self.layout_token, self.renderer.generation, key

    def get(self, key: Hashable) -> Optional[GlyphGeometry]:
        if self.persistent:
            return GLOBAL_GLYPH_GEOMETRY_CACHE.get(self._global_key(key))
        return GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.get(
            self._global_key(key)
        )

    def store(self, key: Hashable, geometry: GlyphGeometry) -> None:
        if self.persistent:
            GLOBAL_GLYPH_GEOMETRY_CACHE.store(
                self._global_key(key),
                geometry,
                namespace=self.layout_token,
            )
        else:
            GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.store(
                self._global_key(key),
                geometry,
                namespace=self.layout_token,
            )

    def set_persistent(self, persistent: bool) -> bool:
        persistent = bool(persistent)
        if self.persistent == persistent:
            return False
        self.persistent = persistent
        self.clear_transient()
        return True

    def clear_transient(self) -> None:
        GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.discard_namespace(
            self.layout_token
        )

    def invalidate_generation(self) -> None:
        GLOBAL_GLYPH_GEOMETRY_CACHE.discard_namespace(self.layout_token)
        self.clear_transient()


class GlyphSlantLayoutRenderer:
    """Own glyph-slant rendering and caches without owning neutral layout.

    The constructor owns only the layout boundary. Glyph Slant remains one
    pre-geometry typography value independent of the ordered transform stack.

    >>> GlyphSlantLayoutRenderer(object()).glyph_slant_angle
    0.0
    """

    def __init__(self, layout: SceneTextLayout) -> None:
        self.layout = layout
        self.glyph_slant_angle = 0.0
        self.generation = getattr(layout, 'layout_generation', 0)
        self.bounds_cache = {}
        self.geometry_plan = None
        self.geometry_plan_bounds = QRectF()
        self.geometry_cache = LayoutGlyphGeometryCache(self)

    def ensure_layout_generation(self) -> None:
        generation = getattr(self.layout, 'layout_generation', self.generation)
        if generation == self.generation:
            return
        self.geometry_cache.invalidate_generation()
        self.generation = generation
        self.bounds_cache.clear()
        self.geometry_plan = None
        self.geometry_plan_bounds = QRectF()

    def render_cache_key(self) -> tuple:
        """Return state that changes the delegated glyph source image."""
        self.ensure_layout_generation()
        return self.generation, self.glyph_slant_angle

    @property
    def line_spaces_lst(self) -> list:
        return self.layout.line_spaces_lst

    @property
    def _draw_offset(self) -> list:
        return self.layout._draw_offset

    def document(self) -> QTextDocument:
        return self.layout.document()

    def _report_glyph_raster_failure(
        self, error: Exception, effect_pass: bool = False
    ) -> None:
        self.layout._report_render_failure(error, effect_pass)

    def _vertical_line_placement(
        self, block: QTextBlock, line_number: int
    ) -> Optional[Tuple[QTextLine, QPointF, QTransform]]:
        return self.layout.vertical_line_placement(block, line_number)

    def _iter_glyph_line_placements(
        self,
    ) -> Iterator[Tuple[Tuple[int, int], QTextLine, QPointF, QTransform]]:
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for line_number in range(layout.lineCount()):
                placement = self._vertical_line_placement(block, line_number)
                if placement is not None:
                    yield (block.blockNumber(), line_number), *placement
            block = block.next()

    def draw_glyph_selection_mask(
        self,
        painter: QPainter,
        context: QAbstractTextDocumentLayout.PaintContext,
        *,
        include_annotations: bool = True,
    ) -> None:
        """Draw only glyphs named by temporary document-layout selections.

        This is the Qt 5/6 common path used to build vertical effect masks.
        It reads glyph runs from the attached QTextLayout objects, preserving
        their established positions and rotations without changing document
        formats. QText decorations are intentionally excluded from the mask.
        """
        painter.save()
        try:
            painter.setPen(Qt.GlobalColor.white)
            if context.clip.isValid():
                painter.setClipRect(context.clip)
            for selection in context.selections:
                selection_start = selection.cursor.selectionStart()
                selection_end = selection.cursor.selectionEnd()
                if selection_end <= selection_start:
                    continue
                self._draw_glyph_range(
                    painter, selection_start, selection_end
                )
            if include_annotations:
                self._draw_annotation_selection(painter, context)
        finally:
            painter.restore()

    def draw_native_annotation_selection(
        self,
        painter: QPainter,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> None:
        """Paint only native annotation ink named by effect selections."""
        painter.save()
        try:
            if context.clip.isValid():
                painter.setClipRect(context.clip)
            self._draw_annotation_selection(painter, context, native=True)
        finally:
            painter.restore()

    def _draw_annotation_selection(
        self,
        painter: QPainter,
        context: QAbstractTextDocumentLayout.PaintContext,
        *,
        native: bool = False,
    ) -> None:
        """Draw selected annotation ink with mask or native paint formats."""
        if not context.selections:
            return
        paint_context = context
        if not native:
            paint_context = QAbstractTextDocumentLayout.PaintContext()
            paint_context.cursorPosition = -1
            paint_context.selections = []
            for selection in context.selections:
                mask_selection = QAbstractTextDocumentLayout.Selection()
                mask_selection.cursor = selection.cursor
                mask_format = QTextCharFormat()
                mask_format.setProperty(GLYPH_STROKE_FORMAT_PROPERTY, True)
                mask_format.setForeground(Qt.GlobalColor.white)
                mask_format.setTextOutline(QPen(Qt.PenStyle.NoPen))
                mask_selection.format = mask_format
                paint_context.selections.append(mask_selection)

        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for line_number in range(layout.lineCount()):
                placement = self._vertical_line_placement(block, line_number)
                if placement is None:
                    continue
                line, offset, orientation = placement
                draw_emphasis_marks(
                    painter,
                    block,
                    line,
                    paint_context,
                    vertical=True,
                    offset=offset,
                    orientation=orientation,
                    side_offsets=ruby_side_margins(
                        block,
                        line,
                        self.layout._ruby_metrics[block.blockNumber()],
                        vertical=True,
                    ),
                )
            ruby_placements = getattr(
                self.layout, '_vertical_ruby_placements', None
            )
            if ruby_placements is not None:
                for placement in ruby_placements(
                    block, paint_context
                ):
                    if any(
                        selection.cursor.selectionStart() < placement.unit.end
                        and placement.unit.start
                        < selection.cursor.selectionEnd()
                        for selection in context.selections
                    ):
                        draw_ruby_placement(painter, placement)
            block = block.next()

    def _draw_glyph_range(
        self, painter: QPainter, selection_start: int, selection_end: int
    ) -> None:
        block = self.document().firstBlock()
        while block.isValid():
            block_start = block.position()
            block_end = block_start + block.length() - 1
            if selection_start >= block_end:
                block = block.next()
                continue
            if selection_end <= block_start:
                break

            local_start = max(0, selection_start - block_start)
            local_end = min(block_end - block_start, selection_end - block_start)
            block_number = block.blockNumber()
            block_text = block.text()
            block_text_length = _utf16_length(block_text)
            layout = block.layout()
            line_spaces = self.line_spaces_lst[block_number]
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                line_start = line.textStart()
                line_end = line_start + line.textLength()
                run_start = max(line_start, local_start)
                run_end = min(line_end, local_end)
                if run_end <= run_start:
                    continue

                glyph_runs = line.glyphRuns(run_start, run_end - run_start)
                if not glyph_runs:
                    continue
                if self.glyph_slant_angle != 0.0:
                    placement = self._vertical_line_placement(block, line_number)
                    # Preserve the selection mask's empty-line skip.
                    if placement is None or not block_text:
                        continue
                    placed_line, offset, orientation = placement
                    draw_slanted_glyph_mask(
                        painter,
                        placed_line,
                        run_start,
                        run_end - run_start,
                        offset,
                        orientation,
                        self.glyph_slant_angle,
                        self._report_glyph_raster_failure,
                        self.geometry_cache,
                        (block_number, line_number),
                    )
                    continue

                _, leading_spaces, _, line_position = line_spaces[line_number]
                char_offset = min(
                    line_position + leading_spaces, block_text_length - 1
                )
                if char_offset < 0:
                    continue
                char = _utf16_char_at(block_text, char_offset)
                x_offset, y_offset = self._draw_offset[block_number][line_number]

                painter.save()
                try:
                    if char in self.layout.vertical_rotation_chars:
                        line_x, line_y = line.x(), line.y()
                        painter.setTransform(
                            QTransform(
                                0,
                                1,
                                0,
                                -1,
                                0,
                                0,
                                line_y + line_x,
                                line_y - line_x,
                                1,
                            ),
                            True,
                        )
                    for glyph_run in glyph_runs:
                        # QGlyphRun carries decoration flags independently of
                        # its glyph indexes. They belong to the normal text
                        # pass and must not be expanded into a thick outline.
                        glyph_run.setUnderline(False)
                        glyph_run.setOverline(False)
                        glyph_run.setStrikeOut(False)
                        painter.drawGlyphRun(
                            QPointF(x_offset, y_offset), glyph_run
                        )
                finally:
                    painter.restore()
            block = block.next()

    def draw_vertical_line(
        self,
        painter: QPainter,
        block: QTextBlock,
        line_number: int,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> bool:
        placement = self._vertical_line_placement(block, line_number)
        if placement is None:
            return False
        line, offset, orientation = placement
        draw_slanted_line(
            painter,
            block,
            line,
            offset,
            orientation,
            self.glyph_slant_angle,
            context,
            self._report_glyph_raster_failure,
            self.geometry_cache,
            (block.blockNumber(), line_number),
        )
        return True

    def draw_horizontal_block(
        self,
        painter: QPainter,
        block: QTextBlock,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> None:
        if self.draw_uniform_block(painter, block, context):
            return
        layout = block.layout()
        for line_number in range(layout.lineCount()):
            line = layout.lineAt(line_number)
            if line.isValid() and line.textLength() > 0:
                draw_slanted_line(
                    painter,
                    block,
                    line,
                    QPointF(),
                    QTransform(),
                    self.glyph_slant_angle,
                    context,
                    self._report_glyph_raster_failure,
                    self.geometry_cache,
                    (block.blockNumber(), line_number),
                )

    def draw_horizontal_line(
        self,
        painter: QPainter,
        block: QTextBlock,
        line_number: int,
        context: QAbstractTextDocumentLayout.PaintContext,
        horizontal_shifts=(),
    ) -> None:
        """Draw one horizontal line for layout-owned Ruby translations."""
        line = block.layout().lineAt(line_number)
        if not line.isValid() or line.textLength() <= 0:
            return
        draw_slanted_line(
            painter,
            block,
            line,
            QPointF(),
            QTransform(),
            self.glyph_slant_angle,
            context,
            self._report_glyph_raster_failure,
            self.geometry_cache,
            (block.blockNumber(), line_number),
            horizontal_shifts=horizontal_shifts,
        )

    def clear_caches(self) -> None:
        self.bounds_cache.clear()
        self.geometry_plan = None
        self.geometry_plan_bounds = QRectF()
        self.geometry_cache.clear_transient()

    def release_caches(self) -> None:
        """Drop every cache entry derived from the attached layout."""
        self.bounds_cache.clear()
        self.geometry_plan = None
        self.geometry_plan_bounds = QRectF()
        self.geometry_cache.invalidate_generation()

    def apply(
        self,
        angle: float,
        persistent_cache: bool = True,
    ) -> bool:
        angle_changed = angle != self.glyph_slant_angle
        # Global-stack previews do not alter glyph geometry, so keep using the
        # committed global entry instead of creating a redundant scratch copy.
        if (
            persistent_cache
            or angle_changed
            or not self.geometry_cache.persistent
        ):
            self.geometry_cache.set_persistent(persistent_cache)
        if not angle_changed:
            return False
        self.glyph_slant_angle = angle
        self.clear_caches()
        if C.USE_PYSIDE6:
            self.layout.update.emit()
        else:
            self.layout.update.emit(
                QRectF(0, 0, self.layout.max_width, self.layout.max_height)
            )
        return True

    def _iter_horizontal_line_placements(
        self,
    ) -> Iterator[Tuple[Tuple[int, int], QTextLine, QPointF, QTransform]]:
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for line_number in range(layout.lineCount()):
                line = layout.lineAt(line_number)
                if line.isValid() and line.textLength() > 0:
                    yield (
                        (block.blockNumber(), line_number),
                        line,
                        QPointF(),
                        QTransform(),
                    )
            block = block.next()

    @staticmethod
    def _uniform_block_format(
        block: QTextBlock,
    ) -> Optional[QTextCharFormat]:
        """Return the sole undecorated block format, or ``None``."""
        layout = block.layout()
        if layout.formats():
            return None
        fragments = []
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.length() > 0:
                fragments.append(fragment)
            iterator += 1
        if len(fragments) != 1:
            return None
        char_format = fragments[0].charFormat()
        font = char_format.font()
        if (
            char_format.background().style() != Qt.BrushStyle.NoBrush
            or char_format.textOutline().style() != Qt.PenStyle.NoPen
            or char_format.underlineStyle()
            != QTextCharFormat.UnderlineStyle.NoUnderline
            or font.overline()
            or font.strikeOut()
        ):
            return None
        return char_format

    def draw_uniform_block(
        self,
        painter: QPainter,
        block: QTextBlock,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> bool:
        """Batch the common unselected, same-format glyph paint path."""
        if context.selections:
            return False
        char_format = self._uniform_block_format(block)
        if char_format is None:
            return False
        self._ensure_geometry_plan()
        draw_uniform_glyph_geometries(
            painter,
            self.geometry_plan.get(block.blockNumber(), ()),
            char_format,
            self._report_glyph_raster_failure,
        )
        return True

    def _ensure_geometry_plan(self) -> QRectF:
        """Build the exact generation geometry once for bounds and paint."""
        self.ensure_layout_generation()
        if self.geometry_plan is not None:
            return QRectF(self.geometry_plan_bounds)
        bounds = QRectF()
        geometry_plan = {}
        placements = (
            self._iter_glyph_line_placements()
            if hasattr(self.layout, 'line_spaces_lst')
            else self._iter_horizontal_line_placements()
        )
        for namespace, line, offset, orientation in placements:
            geometry = slanted_line_geometry(
                line,
                offset,
                orientation,
                self.glyph_slant_angle,
                self.geometry_cache,
                namespace,
            )
            geometry_plan.setdefault(namespace[0], []).append(geometry)
            line_bounds = geometry.bounds
            if line_bounds.isEmpty():
                continue
            bounds = (
                QRectF(line_bounds)
                if bounds.isNull()
                else bounds.united(line_bounds)
            )
        self.geometry_plan = {
            block_number: tuple(geometries)
            for block_number, geometries in geometry_plan.items()
        }
        self.geometry_plan_bounds = QRectF(bounds)
        return bounds

    def ink_bounds(self) -> QRectF:
        if getattr(self.layout, 'publishing_size_enlargement', False):
            settled = next(
                iter(self.bounds_cache.values()), self.geometry_plan_bounds
            )
            return QRectF(settled)
        self.ensure_layout_generation()
        document = self.layout.document()
        if document.isEmpty():
            return QRectF()
        key = (
            document.revision(),
            self.generation,
            type(self.layout),
            self.glyph_slant_angle,
        )
        cached = self.bounds_cache.get(key)
        if cached is not None:
            return QRectF(cached)
        bounds = QRectF(self._ensure_geometry_plan())
        vertical_placement = getattr(
            self.layout, 'vertical_line_placement', None
        )
        block = document.firstBlock()
        while block.isValid():
            text_layout = block.layout()
            for line_number in range(text_layout.lineCount()):
                if vertical_placement is None:
                    line = text_layout.lineAt(line_number)
                    offset = QPointF()
                    orientation = QTransform()
                    vertical = False
                else:
                    placement = vertical_placement(block, line_number)
                    if placement is None:
                        continue
                    line, offset, orientation = placement
                    vertical = True
                mark_bounds = emphasis_ink_bounds(
                    block,
                    line,
                    vertical=vertical,
                    offset=offset,
                    orientation=orientation,
                    side_offsets=ruby_side_margins(
                        block,
                        line,
                        self.layout._ruby_metrics[block.blockNumber()],
                        vertical=vertical,
                    ),
                )
                if not mark_bounds.isEmpty():
                    bounds = (
                        QRectF(mark_bounds)
                        if bounds.isNull()
                        else bounds.united(mark_bounds)
                    )
            block = block.next()
        annotation_bounds = self.layout.annotation_ink_bounds()
        if not annotation_bounds.isEmpty():
            bounds = (
                QRectF(annotation_bounds)
                if bounds.isNull()
                else bounds.united(annotation_bounds)
            )
        self.bounds_cache = {key: QRectF(bounds)}
        return bounds

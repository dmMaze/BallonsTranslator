"""Transform-aware state layered on top of the existing text layouts."""

from itertools import count

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QPainter,
    QTextBlock,
    QTransform,
)

from ballontranslator.utils import shared as C
from ballontranslator.utils.fontformat import (
    SlantTextTransform,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    normalize_text_transform_value,
)
from .glyph import (
    GLOBAL_GLYPH_GEOMETRY_CACHE,
    GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE,
    draw_slanted_glyph_mask,
    draw_slanted_line,
    slanted_line_ink_bounds,
)
from .indexing import _utf16_char_at, _utf16_length


_LAYOUT_CACHE_TOKENS = count(1)


class LayoutGlyphGeometryCache:
    """Route committed geometry globally and preview geometry transiently."""

    def __init__(self, renderer) -> None:
        self.renderer = renderer
        self.layout_token = next(_LAYOUT_CACHE_TOKENS)
        self.persistent = True

    def _global_key(self, key):
        self.renderer.ensure_layout_generation()
        return self.layout_token, self.renderer.generation, key

    def get(self, key):
        if self.persistent:
            return GLOBAL_GLYPH_GEOMETRY_CACHE.get(self._global_key(key))
        return GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE.get(
            self._global_key(key)
        )

    def store(self, key, geometry) -> None:
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

    The constructor owns only the layout boundary. Complete typed transform
    values enter through :meth:`apply`, matching other variant renderers.

    >>> GlyphSlantLayoutRenderer(object()).glyph_slant_angle
    0.0
    """

    def __init__(self, layout) -> None:
        self.layout = layout
        self.glyph_slant_angle = 0.0
        self.generation = getattr(layout, 'layout_generation', 0)
        self.bounds_cache = {}
        self.geometry_cache = LayoutGlyphGeometryCache(self)

    def bind_layout(self, layout) -> None:
        """Attach a replacement writing-mode layout without leaking caches."""
        if self.layout is layout:
            return
        self.geometry_cache.invalidate_generation()
        self.layout = layout
        self.generation = getattr(layout, 'layout_generation', 0)
        self.bounds_cache.clear()

    def ensure_layout_generation(self) -> None:
        generation = getattr(self.layout, 'layout_generation', self.generation)
        if generation == self.generation:
            return
        self.geometry_cache.invalidate_generation()
        self.generation = generation
        self.bounds_cache.clear()

    @property
    def line_spaces_lst(self):
        return self.layout.line_spaces_lst

    @property
    def _draw_offset(self):
        return self.layout._draw_offset

    def document(self):
        return self.layout.document()

    def _report_glyph_raster_failure(self, error, effect_pass=False):
        self.layout._report_render_failure(error, effect_pass)

    def _vertical_line_placement(self, block: QTextBlock, line_number: int):
        layout = block.layout()
        line = layout.lineAt(line_number)
        if not line.isValid() or line.textLength() <= 0:
            return None
        block_number = block.blockNumber()
        block_text = block.text()
        block_text_length = _utf16_length(block_text)
        _, leading_spaces, _, line_position = self.line_spaces_lst[block_number][
            line_number
        ]
        char_offset = min(line_position + leading_spaces, block_text_length - 1)
        if char_offset < 0:
            return line, QPointF(), QTransform()
        char = _utf16_char_at(block_text, char_offset)
        x_offset, y_offset = self._draw_offset[block_number][line_number]
        orientation = QTransform()
        if char in self.layout.vertical_rotation_chars:
            line_x, line_y = line.x(), line.y()
            orientation = QTransform(
                0,
                1,
                0,
                -1,
                0,
                0,
                line_y + line_x,
                line_y - line_x,
                1,
            )
        return line, QPointF(x_offset, y_offset), orientation

    def _iter_glyph_line_placements(self):
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
        finally:
            painter.restore()

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

    def draw_vertical_line(self, painter, block, line_number, context) -> bool:
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

    def draw_horizontal_block(self, painter, block, context) -> None:
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

    def clear_caches(self) -> None:
        self.bounds_cache.clear()
        self.geometry_cache.clear_transient()

    def apply(
        self,
        transform: SlantTextTransform,
        persistent_cache: bool = True,
    ) -> bool:
        if not isinstance(transform, SlantTextTransform):
            raise TypeError('glyph slant renderer requires SlantTextTransform')
        return self._set_angle(transform.glyph_slant_angle, persistent_cache)

    def deactivate(self, persistent_cache: bool = True) -> bool:
        return self._set_angle(0.0, persistent_cache)

    def _set_angle(
        self,
        angle: float,
        persistent_cache: bool,
    ) -> bool:
        angle = normalize_text_transform_value(
            angle,
            TEXT_TRANSFORM_GLYPH_SLANT_MIN,
            TEXT_TRANSFORM_GLYPH_SLANT_MAX,
        )
        angle_changed = angle != self.glyph_slant_angle
        # Box-only previews do not alter glyph geometry, so keep using the
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

    def _iter_horizontal_line_placements(self):
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

    def ink_bounds(self) -> QRectF:
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
        bounds = QRectF()
        placements = (
            self._iter_glyph_line_placements()
            if hasattr(self.layout, 'line_spaces_lst')
            else self._iter_horizontal_line_placements()
        )
        for namespace, line, offset, orientation in placements:
            line_bounds = slanted_line_ink_bounds(
                line,
                offset,
                orientation,
                self.glyph_slant_angle,
                self.geometry_cache,
                namespace,
            )
            if line_bounds.isEmpty():
                continue
            bounds = line_bounds if bounds.isNull() else bounds.united(line_bounds)
        self.bounds_cache = {key: QRectF(bounds)}
        return bounds

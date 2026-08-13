"""Geometry and transform lifecycle for one ``TextBlkItem``."""

from collections.abc import Generator
from contextlib import contextmanager
import math
from typing import Iterator, Optional, TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QPointF, QRect, QRectF, QSizeF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QPainter,
    QPainterPath,
    QPolygonF,
    QTransform,
)
from qtpy.QtWidgets import (
    QGraphicsItem,
    QGraphicsTextItem,
    QStyleOptionGraphicsItem,
)

from ballontranslator.utils.fontformat import (
    GridTextTransform,
    ProjectiveTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.textblock import TextAlignment
from .rendering.surface import NonlinearTextSurfaceRenderer
from .rendering.glyph_slant import GlyphSlantLayoutRenderer
from .rendering.raster import (
    EffectRasterAllocationError,
    RASTER_BOUNDARY_FAILURES,
)
from .transforms.mapping import (
    CompiledTextTransform,
    CompositeTextTransformMapper,
    compensated_native_transform_matrix,
    grid_transform_stage,
)
from .transforms.registry import compile_text_transform_stack

if TYPE_CHECKING:
    from .item import TextBlkItem


class TextItemGeometryController:
    """Own logical/display geometry and derived transform state.

    The graphics item retains only thin Qt virtual-method hooks. Effect and
    layout calls are explicit host boundaries so new transform stages do
    not need to grow ``TextBlkItem`` itself.

    >>> from ballontranslator.utils.fontformat import ProjectiveTextTransform
    >>> ProjectiveTextTransform().is_neutral()
    True
    """

    def __init__(self, item: "TextBlkItem") -> None:
        self.item = item
        self.display_rect = QRectF(0, 0, 1, 1)
        self.preview: Optional[TextTransformStack] = None
        self.compiled = CompiledTextTransform(
            TextTransformStack(), QTransform()
        )
        self._compiled_input_key = None
        self._compile_deferred = False
        self._compile_defer_depth = 0
        self.layout_renderer = None
        self.visual_mapper = None
        self.surface_renderer = None
        self._surface_cursor_position = -1
        self._input_mapping_active = False
        self._input_previous_source = None
        self.installing = False
        self._update_depth = 1
        self._update_dirty = False

    def bind_model(self) -> None:
        """Reset transient state after the item adopts a ``TextBlock``."""
        self.preview = None
        self.detach_layout_renderer()
        self.detach_surface_mapper()
        self.compiled = CompiledTextTransform(
            self.canonical(), QTransform()
        )
        self._compiled_input_key = None
        self._compile_deferred = False
        self._compile_defer_depth = 0

    def item_change(self, change, value, base_item_change):
        item = self.item
        if self.installing:
            return base_item_change(change, value)

        if item.blk is None or not self.requires_custom_resize():
            return base_item_change(change, value)

        if change in (
            QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformOriginPointHasChanged,
        ):
            # At HasChanged the Qt property already contains its final value.
            with self.update_transaction():
                self.install(
                    self.compiled,
                    angle=item.rotation(),
                    transform_pivot=item.logical_unpadded_rect().center(),
                    rotation_pivot=item.transformOriginPoint(),
                )
                result = base_item_change(change, value)
                self.request_update()
            return result

        result = base_item_change(change, value)
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemScaleHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged,
        ):
            self.request_update()
        return result

    def finish_initialization(self) -> None:
        self.refresh_compiled_geometry()
        self.request_update()
        self._update_depth = 0
        self._flush_update()

    def canonical(self) -> TextTransformStack:
        """Return the committed model state, excluding any active preview."""
        # Persistence and transform undo use the TextBlock-owned format.
        # item.fontformat may be a temporary render copy during formatting.
        fontformat = self.item.blk.fontformat
        if not isinstance(fontformat.text_transform, TextTransformStack):
            raise ValueError('live font format requires a typed transform stack')
        return fontformat.text_transform

    def effective(self) -> TextTransformStack:
        """Return the preview state when active, otherwise committed state."""
        return self.preview if self.preview is not None else self.canonical()

    def is_neutral(self) -> bool:
        return self.effective().is_neutral()

    def visual_is_neutral(self) -> bool:
        return (
            self.item.transform().isIdentity()
            and self.visual_mapper is None
            and not self.has_layout_distortion()
        )

    def uses_surface_warp(self) -> bool:
        return self.visual_mapper is not None

    @contextmanager
    def update_transaction(self):
        """Batch cache and input-method work across nested Qt changes."""
        self._update_depth += 1
        try:
            yield
        finally:
            self._update_depth -= 1
            if self._update_depth == 0:
                self._flush_update()

    @contextmanager
    def defer_compilation(self) -> "Generator[None, None, None]":
        """Compile once after a transient layout transaction settles."""
        self._compile_defer_depth += 1
        try:
            yield
        finally:
            self._compile_defer_depth -= 1
            if self._compile_defer_depth == 0:
                self.flush_deferred_compilation()

    def request_update(self) -> None:
        self._update_dirty = True
        if self._update_depth == 0:
            self._flush_update()

    def _flush_update(self) -> None:
        if not self._update_dirty:
            return
        self._update_dirty = False
        self.item.refresh_cache_policy()
        if self.item.isEditing():
            self.item.updateMicroFocus()

    def compensated_matrix(
        self,
        compiled: CompiledTextTransform = None,
        *,
        angle: Optional[float] = None,
        transform_pivot: Optional[QPointF] = None,
        rotation_pivot: Optional[QPointF] = None,
    ):
        """Build the derived Qt base transform for the current item state."""
        item = self.item
        if angle is None:
            angle = item.rotation()
        if transform_pivot is None:
            transform_pivot = item.logical_unpadded_rect().center()
        if rotation_pivot is None:
            rotation_pivot = item.transformOriginPoint()
        compiled = self.compiled if compiled is None else compiled
        return compensated_native_transform_matrix(
            compiled.native_matrix,
            transform_pivot,
            angle,
            rotation_pivot,
        )

    def bounding_rect(self, base_rect: QRectF) -> QRectF:
        """Return the Qt paint bounds with the managed display size."""
        rect = QRectF(base_rect)
        rect.setSize(self.display_rect.size())
        source_paint_rect = self.source_paint_rect()
        rect = rect.united(source_paint_rect)
        if self.visual_mapper is not None:
            rect = rect.united(
                self.visual_mapper.visual_bounds(source_paint_rect)
            )
        return rect

    def source_rect(self) -> QRectF:
        """Return the unwarped local paint surface, including effect padding."""
        return QRectF(QPointF(), self.display_rect.size())

    def _source_ink_bounds(self) -> QRectF:
        """Return the one layout-owned source-ink union."""
        bounds = self.layout_ink_bounds()
        layout = getattr(self.item, 'layout', None)
        if layout is None:
            return bounds
        annotation_bounds = layout.annotation_ink_bounds()
        if annotation_bounds.isEmpty():
            return bounds
        return (
            QRectF(annotation_bounds)
            if bounds.isEmpty()
            else bounds.united(annotation_bounds)
        )

    def source_paint_rect(self) -> QRectF:
        """Include derived ink overhang without changing logical geometry."""
        rect = self.source_rect()
        ink_bounds = self._source_ink_bounds()
        if not ink_bounds.isEmpty():
            padding = self.item.padding()
            rect = rect.united(
                ink_bounds.adjusted(
                    -padding, -padding, padding, padding
                )
            )
        return rect

    def logical_rect(self) -> QRectF:
        """Return the untransformed, effect-free local rectangle."""
        return self.unpad_rect(self.source_rect())

    def pad_rect(self, rect: QRectF) -> QRectF:
        padding = self.item.padding()
        return rect.adjusted(-padding, -padding, padding, padding)

    def unpad_rect(self, rect: QRectF) -> QRectF:
        padding = self.item.padding()
        return rect.adjusted(padding, padding, -padding, -padding)

    def shape(self) -> QPainterPath:
        ink_bounds = self._source_ink_bounds()
        if self.visual_mapper is not None:
            path = self.visual_mapper.map_rect_path(self.logical_rect())
            if not ink_bounds.isEmpty():
                path = path.united(
                    self.visual_mapper.map_rect_path(ink_bounds)
                )
            return path
        path = QPainterPath()
        path.addRect(
            self.source_rect() if self.is_neutral() else self.logical_rect()
        )
        if not ink_bounds.isEmpty():
            ink_path = QPainterPath()
            ink_path.addRect(ink_bounds)
            path = path.united(ink_path)
        return path

    def contains(self, point: QPointF) -> bool:
        return self.shape().contains(QPointF(point))

    def visual_outline_in_item(self) -> QPainterPath:
        if self.visual_mapper is not None:
            return self.visual_mapper.map_rect_path(self.logical_rect())
        path = QPainterPath()
        path.addRect(self.logical_rect())
        return path

    def visual_outline_in_scene(self) -> QPainterPath:
        return self.item.sceneTransform().map(self.visual_outline_in_item())

    def map_source_to_visual(self, point: QPointF) -> QPointF:
        mapper = self.visual_mapper
        return QPointF(point) if mapper is None else mapper.forward_point(point)

    def map_visual_to_source(
        self,
        point: QPointF,
        previous_source: QPointF = None,
    ) -> QPointF:
        mapper = self.visual_mapper
        if mapper is None:
            return QPointF(point)
        if previous_source is None and self._input_mapping_active:
            previous_source = self._input_previous_source
        source = mapper.inverse_point(point, previous_source)
        if self._input_mapping_active:
            self._input_previous_source = QPointF(source)
        return source

    def begin_input_mapping(self) -> None:
        self._input_mapping_active = True
        self._input_previous_source = None

    def end_input_mapping(self) -> None:
        self._input_mapping_active = False
        self._input_previous_source = None

    def map_source_to_scene(self, point: QPointF) -> QPointF:
        return self.item.mapToScene(self.map_source_to_visual(point))

    def _grid_stage(self, stack_index: int):
        stages = self.compiled.stages
        if stack_index < 0 or stack_index >= len(stages):
            return None
        record = stages[stack_index]
        if not isinstance(record.transform, GridTextTransform):
            return None
        mapper = record.mapper or grid_transform_stage(
            record.transform, record.context
        )
        prefix = tuple(
            stage.mapper
            for stage in stages[:stack_index]
            if stage.mapper is not None
        )
        return record, mapper, prefix

    def grid_control_geometry(self, stack_index: int):
        """Return batched handles and the compiled Grid-output mapper."""
        stage = self._grid_stage(stack_index)
        if stage is None:
            return None
        record, mapper, prefix = stage
        # Controls are expressed at this Grid's output, so their mapper keeps
        # the selected stage and every later stage while excluding its prefix.
        suffix_stages = (mapper,) + tuple(
            stage.mapper
            for stage in self.compiled.stages[stack_index + 1:]
            if stage.mapper is not None
        )
        output_mapper = (
            self.compiled.surface_mapper
            if record.mapper is not None and not prefix
            else CompositeTextTransformMapper(
                suffix_stages,
                record.context.logical_bounds,
                record.context.logical_bounds,
                record.context.vertical,
            )
        )
        points = mapper.control_source_points()
        coordinates = np.asarray(
            [(point.x(), point.y()) for point in points],
            dtype=np.float64,
        )
        visual_x, visual_y = output_mapper.forward_arrays(
            coordinates[:, 0], coordinates[:, 1]
        )
        visual_points = QPolygonF([
            QPointF(float(x), float(y))
            for x, y in zip(visual_x, visual_y)
        ])
        return (
            visual_points,
            output_mapper,
            QRectF(record.context.logical_bounds),
            record.transform,
        )

    def projective_control_center_in_scene(self, stack_index: int):
        """Return the selected stage's fixed pivot in scene coordinates."""
        stages = self.compiled.stages
        if stack_index < 0 or stack_index >= len(stages):
            return None
        record = stages[stack_index]
        if not isinstance(record.transform, ProjectiveTextTransform):
            return None
        source = record.context.source_bounds.center()
        for prefix in reversed(stages[:stack_index]):
            if prefix.mapper is not None:
                source = prefix.mapper.inverse_point(
                    source, extrapolate=True
                )
        return self.map_source_to_scene(source)

    def capture_scene_to_grid_output_mapper(self, stack_index: int):
        """Freeze scene-to-grid coordinates for one control-point drag."""
        stage = self._grid_stage(stack_index)
        scene_to_visual, invertible = self.item.sceneTransform().inverted()
        if stage is None or not invertible:
            return None
        record, mapper, prefix = stage

        if record.mapper is not None and self.compiled.surface_mapper is not None:
            suffix_stages = tuple(
                stage.mapper
                for stage in self.compiled.stages[stack_index + 1:]
                if stage.mapper is not None
            )
            suffix_mapper = (
                CompositeTextTransformMapper(
                    suffix_stages,
                    record.context.logical_bounds,
                    record.context.logical_bounds,
                    record.context.vertical,
                )
                if suffix_stages
                else None
            )

            def map_point(scene_point: QPointF, previous_source=None):
                grid_output = scene_to_visual.map(QPointF(scene_point))
                if suffix_mapper is not None:
                    grid_output = suffix_mapper.inverse_point(
                        grid_output,
                        previous_source,
                        extrapolate=True,
                    )
                return grid_output

            return map_point, mapper.normalized_output_delta

        scene_to_source = self.capture_scene_to_source_mapper()
        if scene_to_source is None:
            return None

        def map_point(scene_point: QPointF, previous_source=None):
            source = scene_to_source(scene_point, previous_source)
            stage_input = QPointF(source)
            for prefix_mapper in prefix:
                stage_input = prefix_mapper.forward_point(stage_input)
            return mapper.forward_point(stage_input)

        return map_point, mapper.normalized_output_delta

    def capture_scene_to_grid_output_array_mapper(self, stack_index: int):
        """Freeze a batched scene-to-selected-Grid-output mapping.

        >>> callable(TextItemGeometryController.capture_scene_to_grid_output_array_mapper)
        True
        """
        stage = self._grid_stage(stack_index)
        scene_to_visual, invertible = self.item.sceneTransform().inverted()
        if stage is None or not invertible:
            return None
        record, mapper, prefix = stage

        def scene_arrays(scene_points):
            visual_points = scene_to_visual.map(QPolygonF(scene_points))
            coordinates = np.asarray(
                [(point.x(), point.y()) for point in visual_points],
                dtype=np.float64,
            )
            return coordinates[:, 0], coordinates[:, 1]

        if record.mapper is not None and self.compiled.surface_mapper is not None:
            suffix_stages = tuple(
                stage.mapper
                for stage in self.compiled.stages[stack_index + 1:]
                if stage.mapper is not None
            )
            suffix_mapper = (
                CompositeTextTransformMapper(
                    suffix_stages,
                    record.context.logical_bounds,
                    record.context.logical_bounds,
                    record.context.vertical,
                )
                if suffix_stages
                else None
            )

            def map_points(scene_points):
                visual_x, visual_y = scene_arrays(scene_points)
                if suffix_mapper is None:
                    valid = np.isfinite(visual_x) & np.isfinite(visual_y)
                    return visual_x, visual_y, valid
                return suffix_mapper.inverse_arrays(
                    visual_x, visual_y, return_valid=True
                )

            return map_points

        visual_mapper = self.visual_mapper

        def map_points(scene_points):
            source_x, source_y = scene_arrays(scene_points)
            valid = np.isfinite(source_x) & np.isfinite(source_y)
            if visual_mapper is not None:
                source_x, source_y, mapper_valid = visual_mapper.inverse_arrays(
                    source_x, source_y, return_valid=True
                )
                valid &= mapper_valid
            for prefix_mapper in prefix:
                source_x, source_y = prefix_mapper.forward_arrays(
                    source_x, source_y
                )
            output_x, output_y = mapper.forward_arrays(source_x, source_y)
            valid &= np.isfinite(output_x) & np.isfinite(output_y)
            return output_x, output_y, valid

        return map_points

    def capture_scene_to_source_mapper(self):
        """Freeze the item mapping used for one shape-controller drag.

        Resizing changes both the nonlinear mapper and the item's position.
        Mapping later mouse events through that moving geometry creates
        feedback, so a drag must stay in its start coordinate system.
        """
        scene_to_visual, invertible = self.item.sceneTransform().inverted()
        if not invertible:
            return None
        visual_mapper = self.visual_mapper

        def map_point(
            scene_point: QPointF,
            previous_source: QPointF = None,
        ) -> QPointF:
            visual_point = scene_to_visual.map(QPointF(scene_point))
            if visual_mapper is None:
                return visual_point
            return visual_mapper.inverse_point(
                visual_point,
                previous_source,
                # A reshape may continue outside the warped surface. Ordinary
                # text hit testing deliberately keeps the bounded inverse.
                extrapolate=True,
            )

        return map_point

    def source_handle_points(self):
        rect = self.logical_rect()
        return (
            rect.topLeft(),
            QPointF(rect.center().x(), rect.top()),
            rect.topRight(),
            QPointF(rect.right(), rect.center().y()),
            rect.bottomRight(),
            QPointF(rect.center().x(), rect.bottom()),
            rect.bottomLeft(),
            QPointF(rect.left(), rect.center().y()),
        )

    def visual_handle_points_in_scene(self):
        source_points = self.source_handle_points()
        return [
            self.map_source_to_scene(point) for point in source_points
        ]

    def visual_handle_tangents_in_scene(self):
        """Return local text-flow tangents at the eight visual handles."""
        mapper = self.visual_mapper
        vertical = self.item.fontformat.vertical
        source_flow = QPointF(0.0, 1.0) if vertical else QPointF(1.0, 0.0)
        tangents = []
        for source in self.source_handle_points():
            visual = self.map_source_to_visual(source)
            if mapper is None:
                next_visual = self.map_source_to_visual(source + source_flow)
            else:
                next_visual = visual + mapper.local_tangent(source)
            scene = self.item.mapToScene(visual)
            tangents.append(self.item.mapToScene(next_visual) - scene)
        return tangents

    def visual_rotation_center_in_scene(self) -> QPointF:
        # Bend is translated so its visual outline bounds remain centered
        # on the logical rectangle; Qt rotation uses that stable visual center.
        return self.item.mapToScene(self.logical_rect().center())

    def visual_bounds_in_scene(self) -> QRectF:
        return self.visual_outline_in_scene().boundingRect()

    def absolute_rect(self, max_h=None, max_w=None, qrect=False):
        """Return the persistent logical rectangle in parent coordinates."""
        rect = self.logical_rect()
        width, height = rect.width(), rect.height()
        pos = self.item.pos()
        x = pos.x() + rect.x()
        y = pos.y() + rect.y()
        if max_h is not None:
            y = min(max(0, y), max_h)
            height = min(max_h, y + height) - y
        if max_w is not None:
            x = min(max(0, x), max_w)
            width = min(max_w, x + width) - x
        if qrect:
            return QRectF(x, y, width, height)
        return [
            int(round(x)),
            int(round(y)),
            math.ceil(width),
            math.ceil(height),
        ]

    def logical_position(self) -> QPointF:
        return self.absolute_rect(qrect=True).topLeft()

    def set_logical_position(self, point: QPointF) -> bool:
        point = QPointF(point)
        delta = point - self.logical_position()
        if delta.isNull():
            return False
        item = self.item
        item.setPos(item.pos() + delta)
        item.blk._bounding_rect = self.absolute_rect()
        return True

    def set_rect(
        self,
        rect,
        *,
        padding: bool = True,
        repaint: bool = True,
        update_blk_rect: bool = True,
    ) -> None:
        """Set logical geometry while keeping paint padding derived."""
        item = self.item
        old_logical_rect = self.logical_rect()
        if isinstance(rect, list):
            rect = QRectF(*rect)
        else:
            rect = QRectF(rect)
        if padding:
            rect = self.pad_rect(rect)
        item.setPos(rect.topLeft())
        item.prepareGeometryChange()
        self.display_rect = rect
        item.layout.setMaxSize(rect.width(), rect.height())
        if self.effective().has_active_stages:
            self.refresh_compiled_geometry()
        self.sync_origin()
        if (
            item.fontformat.gradient_enabled
            and not item.repainting
            and self.logical_rect() != old_logical_rect
        ):
            item._refresh_gradient_geometry()
        if repaint:
            item.repaint_background()
        if update_blk_rect:
            item.blk._bounding_rect = self.absolute_rect()

    def _size_alignment_anchor(self, rect: QRectF) -> QPointF:
        item = self.item
        alignment = item.fontformat.alignment
        if item.fontformat.vertical:
            if alignment == TextAlignment.Left:
                return rect.topLeft()
            if alignment == TextAlignment.Center:
                return QPointF(rect.center().x(), rect.top())
            return rect.topRight()
        if alignment == TextAlignment.Right:
            return rect.topRight()
        if alignment == TextAlignment.Left:
            return rect.topLeft()
        return rect.center()

    def resize(
        self,
        width: float,
        height: float,
        *,
        set_layout_maxsize: bool = False,
        set_blk_size: bool = True,
    ) -> None:
        """Resize through the current transform strategy's geometry policy."""
        if self.requires_custom_resize() or self.has_layout_distortion():
            self._resize_transformed(
                width,
                height,
                set_layout_maxsize=set_layout_maxsize,
                set_blk_size=set_blk_size,
            )
            return
        self._resize_standard(
            width,
            height,
            set_layout_maxsize=set_layout_maxsize,
            set_blk_size=set_blk_size,
        )

    def _resize_standard(
        self,
        width: float,
        height: float,
        *,
        set_layout_maxsize: bool,
        set_blk_size: bool,
    ) -> None:
        item = self.item
        if set_layout_maxsize:
            item.layout.setMaxSize(width, height)

        old_rect = self.logical_rect()
        old_anchor_parent = item.mapToParent(
            self._size_alignment_anchor(old_rect)
        )
        self.display_rect.setWidth(width)
        self.display_rect.setHeight(height)
        self.sync_origin()
        new_anchor_parent = item.mapToParent(
            self._size_alignment_anchor(self.logical_rect())
        )
        item.setPos(item.pos() + old_anchor_parent - new_anchor_parent)
        if item.blk is not None and set_blk_size:
            item.blk._bounding_rect = self.absolute_rect()

    def _resize_transformed(
        self,
        width: float,
        height: float,
        *,
        set_layout_maxsize: bool,
        set_blk_size: bool,
    ) -> None:
        item = self.item
        if item.transformations():
            raise RuntimeError(
                'TextBlkItem requires an empty QGraphicsTransform list'
            )
        old_rect = self.logical_rect()
        old_anchor_parent = item.mapToParent(
            self._size_alignment_anchor(old_rect)
        )

        item.prepareGeometryChange()
        signals_were_blocked = None
        final_size = None
        if set_layout_maxsize:
            signals_were_blocked = item.layout.blockSignals(True)
        try:
            if set_layout_maxsize:
                item.layout.setMaxSize(width, height)
                final_size = QSizeF(item.layout.documentSize())
                width = final_size.width()
                height = final_size.height()

            with self.update_transaction():
                self.display_rect.setWidth(width)
                self.display_rect.setHeight(height)
                if self.effective().has_active_stages:
                    self.refresh_compiled_geometry()
                self.sync_origin()
                new_anchor_parent = item.mapToParent(
                    self._size_alignment_anchor(self.logical_rect())
                )
                item.setPos(item.pos() + old_anchor_parent - new_anchor_parent)

            if item.blk is not None and set_blk_size:
                item.blk._bounding_rect = self.absolute_rect()
        finally:
            if set_layout_maxsize:
                item.layout.blockSignals(signals_were_blocked)

        if set_layout_maxsize and not signals_were_blocked:
            item.layout.documentSizeChanged.emit(QSizeF(final_size))

    def requires_no_cache(self) -> bool:
        return not self.compiled.is_identity

    def requires_custom_resize(self) -> bool:
        return not self.compiled.is_identity

    def _set_surface_mapper(self, mapper) -> bool:
        """Install the compiler's sole optional nonlinear surface mapper."""
        if mapper is None:
            return self.detach_surface_mapper()
        previous = self.visual_mapper
        geometry_changed = (
            previous is None
            or previous.geometry_key != mapper.geometry_key
        )
        if not geometry_changed:
            if self.item.layout is not None:
                self.item.layout.input_point_mapper = self.map_visual_to_source
            return False
        self.item.prepareGeometryChange()
        self.visual_mapper = mapper
        self._surface_cursor_position = -1
        if self.surface_renderer is None:
            self.surface_renderer = NonlinearTextSurfaceRenderer()
        else:
            self.surface_renderer.release()
        if self.item.layout is not None:
            self.item.layout.input_point_mapper = self.map_visual_to_source
        self.item.update()
        self.request_update()
        return geometry_changed

    def refresh_compiled_geometry(self, *, force: bool = False) -> bool:
        # Rich-text formatting emits several intermediate document sizes.
        # Only the settled geometry is observable after the edit block.
        if (
            self.item.is_formatting or self._compile_defer_depth > 0
        ) and not force:
            self._compile_deferred = True
            return False
        self._compile_deferred = False
        state = self.effective()
        logical_rect = self.logical_rect()
        source_rect = self.source_rect()
        input_key = (
            state.transforms,
            self.item.fontformat.vertical,
            logical_rect.x(),
            logical_rect.y(),
            logical_rect.width(),
            logical_rect.height(),
            source_rect.x(),
            source_rect.y(),
            source_rect.width(),
            source_rect.height(),
        )
        previous = self.compiled
        if input_key != self._compiled_input_key:
            if (
                logical_rect.width() <= 0.0
                or logical_rect.height() <= 0.0
                or source_rect.width() <= 0.0
                or source_rect.height() <= 0.0
            ):
                compiled = CompiledTextTransform(state, QTransform())
            else:
                compiled = compile_text_transform_stack(
                    state,
                    logical_rect,
                    source_rect,
                    self.item.fontformat.vertical,
                )
            self.compiled = compiled
            self._compiled_input_key = input_key
        else:
            compiled = previous

        # Reapply cached output as well as reusing it: page/layout lifecycle
        # code may have released the mapper or changed Qt's installed matrix.
        surface_changed = self._set_surface_mapper(
            compiled.surface_mapper
        )
        box_changed = self._apply_box(compiled)
        return (
            surface_changed
            or box_changed
            or previous.geometry_key != compiled.geometry_key
        )

    def flush_deferred_compilation(self) -> bool:
        if not self._compile_deferred:
            return False
        changed = self.refresh_compiled_geometry(force=True)
        if changed:
            # Formatting can emit its last size signal before the deferred
            # mapper is rebuilt. Notify every visual overlay after the settled
            # geometry is installed, not while it still observes stale bounds.
            self.item.visual_geometry_changed.emit()
        return changed

    def detach_surface_mapper(self) -> bool:
        changed = (
            self.visual_mapper is not None
            or self.surface_renderer is not None
        )
        if not changed:
            if self.item.layout is not None:
                self.item.layout.input_point_mapper = None
            self.end_input_mapping()
            return False
        if self.item.layout is not None:
            self.item.prepareGeometryChange()
            self.item.layout.input_point_mapper = None
        if self.surface_renderer is not None:
            self.surface_renderer.release()
        self.visual_mapper = None
        self.surface_renderer = None
        self._surface_cursor_position = -1
        self.end_input_mapping()
        self.item.update()
        self.request_update()
        return True

    def release_render_resources(self) -> None:
        """Release every item-owned renderer/cache at the page boundary."""
        self.detach_layout_renderer()
        self.detach_surface_mapper()
        self.item.effect_renderer.release_caches()

    def invalidate_surface_cache(self) -> None:
        if self.surface_renderer is not None:
            self.surface_renderer.invalidate_surface()

    def _paint_surface_cursor(
        self,
        painter: QPainter,
        mapper,
        *,
        export_render: bool,
    ) -> None:
        layout = self.item.layout
        if export_render:
            return
        cursor_position = layout.deferred_cursor_position
        cursor_changed = cursor_position != self._surface_cursor_position
        self._surface_cursor_position = cursor_position
        if cursor_position < 0:
            if cursor_changed:
                self.item.update()
            return
        cursor_rect = layout.source_cursor_rect(cursor_position)
        if cursor_rect is None:
            cursor_rect = QGraphicsTextItem.inputMethodQuery(
                self.item, Qt.InputMethodQuery.ImCursorRectangle
            )
        if not isinstance(cursor_rect, (QRectF, QRect)):
            return
        cursor_path = mapper.map_rect_path(QRectF(cursor_rect))
        if cursor_path.isEmpty():
            return
        painter.save()
        try:
            # Invert the completed scene destination, not a transparent source
            # pixmap, so the caret contrasts with both text and page content.
            painter.setCompositionMode(
                QPainter.CompositionMode.RasterOp_NotDestination
            )
            painter.fillPath(cursor_path, Qt.GlobalColor.white)
        finally:
            painter.restore()
        if cursor_changed:
            # QWidgetTextControl may invalidate only the unwarped caret rect.
            # One full follow-up paint clears or redraws its mapped position.
            self.item.update()

    def _probe_surface_cursor(self, painter, option, widget, base_paint):
        """Refresh Qt's cursor visibility without repainting source pixels."""
        layout = self.item.layout
        previous = layout.defer_cursor_paint
        layout.defer_cursor_paint = True
        painter.save()
        try:
            painter.setOpacity(0.0)
            base_paint(
                painter,
                QStyleOptionGraphicsItem(option),
                widget,
            )
        finally:
            painter.restore()
            layout.defer_cursor_paint = previous

    def paint_item(self, painter, option, widget, base_paint) -> None:
        """Paint directly or through the active nonlinear surface warp."""
        mapper = self.visual_mapper
        renderer = self.surface_renderer
        if mapper is None or renderer is None:
            self.item.effect_renderer.paint_item(
                painter, option, widget, base_paint
            )
            return

        effect_renderer = self.item.effect_renderer
        effect_generation, export_render = (
            effect_renderer.surface_cache_state()
        )
        layout_generation = getattr(self.item.layout, 'layout_generation', 0)
        layout_render_key = (
            None
            if self.layout_renderer is None
            else self.layout_renderer.render_cache_key()
        )
        selection_key = None
        if self.item.isEditing():
            cursor = self.item.textCursor()
            if cursor.hasSelection():
                selection_key = (
                    cursor.selectionStart(),
                    cursor.selectionEnd(),
                    self.item.hasFocus(),
                )
        cache_key = (
            mapper.geometry_key,
            layout_generation,
            layout_render_key,
            effect_generation,
            (
                0
                if effect_renderer.background_pixmap is None
                else effect_renderer.background_pixmap.cacheKey()
            ),
            self.item.document().revision(),
            selection_key,
        )

        def paint_source(source_painter, source_option, source_widget):
            layout = self.item.layout
            previous = layout.defer_cursor_paint
            layout.defer_cursor_paint = True
            try:
                effect_renderer.paint_item(
                    source_painter,
                    source_option,
                    source_widget,
                    base_paint,
                )
            finally:
                layout.defer_cursor_paint = previous

        interactive = (
            self.item.isEditing()
            or self.item.reshaping
            or self.preview is not None
        )
        maximum_scale = (
            0.5
            if self.item.reshaping or self.preview is not None
            else (2.0 if interactive else None)
        )
        try:
            cache_hit = renderer.paint(
                painter,
                option,
                mapper,
                self.source_paint_rect(),
                cache_key,
                cache_allowed=(
                    not export_render
                    and not self.item.reshaping
                    and self.preview is None
                ),
                paint_source=paint_source,
                maximum_scale=maximum_scale,
                high_quality=(
                    not self.item.reshaping and self.preview is None
                ),
            )
            if cache_hit and self.item.isEditing():
                self._probe_surface_cursor(
                    painter, option, widget, base_paint
                )
            self._paint_surface_cursor(
                painter, mapper, export_render=export_render
            )
        except RASTER_BOUNDARY_FAILURES as error:
            # Exceptions cannot cross a Qt virtual paint callback. Export
            # records the failure; interactive rendering remains usable.
            failure = EffectRasterAllocationError(str(error))
            effect_renderer._warn_effect_allocation_once(failure)
            if effect_renderer.export_render:
                effect_renderer.export_error = failure
            effect_renderer.paint_item(
                painter, option, widget, base_paint
            )

    def attach_layout_renderer(self):
        renderer = self.layout_renderer
        if renderer is None:
            renderer = GlyphSlantLayoutRenderer(self.item.layout)
            self.layout_renderer = renderer
        elif renderer.layout is not self.item.layout:
            raise RuntimeError(
                'glyph renderer must be detached before layout replacement'
            )
        self.item.layout.render_delegate = renderer
        self.item.layout.render_failure_handler = (
            self.item.effect_renderer._on_glyph_raster_failure
        )
        return renderer

    def detach_layout_renderer(self) -> bool:
        renderer = self.layout_renderer
        if renderer is None:
            if self.item.layout is not None:
                self.item.layout.render_delegate = None
                self.item.layout.render_failure_handler = None
            return False
        renderer.release_caches()
        if self.item.layout is not None:
            self.item.layout.render_delegate = None
            self.item.layout.render_failure_handler = None
        self.layout_renderer = None
        return True

    def layout_ink_bounds(self) -> QRectF:
        renderer = self.layout_renderer
        if renderer is not None:
            return renderer.ink_bounds()
        layout = getattr(self.item, 'layout', None)
        return QRectF() if layout is None else layout.base_ink_bounds()

    def has_layout_distortion(self) -> bool:
        """Return whether glyph painting is delegated to a transform renderer."""
        return self.layout_renderer is not None

    def draw_layout_selection_mask(
        self,
        painter: QPainter,
        context: QAbstractTextDocumentLayout.PaintContext,
        *,
        include_annotations: bool = True,
    ) -> None:
        """Draw an effect mask through the active transform renderer."""
        renderer = self.layout_renderer
        if renderer is None:
            raise RuntimeError('no custom text layout renderer is active')
        renderer.draw_glyph_selection_mask(
            painter,
            context,
            include_annotations=include_annotations,
        )

    def draw_layout_annotations(
        self,
        painter: QPainter,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> None:
        """Draw selected Ruby/emphasis through native annotation renderers."""
        renderer = self.layout_renderer
        if renderer is None:
            raise RuntimeError('no custom text layout renderer is active')
        renderer.draw_native_annotation_selection(painter, context)

    def initialize_layout(self, *, persistent_cache: bool = True) -> bool:
        state = self.effective()
        return self._apply_layout(
            state.glyph_slant_angle,
            persistent_cache=persistent_cache,
        )

    def _apply_layout(
        self,
        glyph_slant_angle: float,
        *,
        persistent_cache: bool = True,
    ) -> bool:
        if self.item.layout is None:
            return False
        if glyph_slant_angle == 0.0:
            return self.detach_layout_renderer()
        return self.attach_layout_renderer().apply(
            glyph_slant_angle, persistent_cache
        )

    def install(
        self,
        compiled: CompiledTextTransform = None,
        *,
        angle: Optional[float] = None,
        transform_pivot: Optional[QPointF] = None,
        rotation_pivot: Optional[QPointF] = None,
    ) -> bool:
        """Install derived Qt geometry without lifecycle side effects."""
        matrix = self.compensated_matrix(
            compiled,
            angle=angle,
            transform_pivot=transform_pivot,
            rotation_pivot=rotation_pivot,
        )
        if self.item.transform() == matrix:
            return False
        self.installing = True
        try:
            self.item.setTransform(matrix, combine=False)
        finally:
            self.installing = False
        return True

    def _apply_box(self, compiled: CompiledTextTransform) -> bool:
        self.item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            not compiled.is_identity,
        )
        with self.update_transaction():
            changed = self.install(compiled)
            if changed:
                self.request_update()
        return changed

    def _refresh_effect_geometry(self, rendering_changed: bool) -> bool:
        item = self.item
        if rendering_changed:
            if any(item.effect_renderer._effect_flags()):
                item.effect_renderer._mark_effect_cache_dirty()
            item.refresh_cache_policy()
            item.update()
        padding_changed = item.effect_renderer._update_effect_padding()
        if padding_changed and self.layout_renderer is not None:
            # Padding relayout advances the layout namespace after the first
            # ink measurement. Rebuild that small committed geometry now so a
            # later preview does not discover and evict stale entries.
            self.layout_renderer.ink_bounds()
        if item.fontformat.gradient_enabled and not padding_changed:
            item.effect_renderer._refresh_gradient_geometry()
        return padding_changed

    def _finalize_neutral(
        self,
        was_visual_neutral: bool,
        was_effect_neutral: bool,
        target: TextTransformStack,
    ) -> bool:
        item = self.item
        effect_neutral = item.effect_renderer._text_transform_is_neutral()
        became_effect_neutral = not was_effect_neutral and effect_neutral
        became_visual_neutral = (
            not was_visual_neutral
            and target.is_neutral()
        )
        if not became_effect_neutral and not became_visual_neutral:
            return False
        item.effect_renderer.finalize_neutral_cache()
        return True

    def _apply_effective_transition(
        self,
        target: TextTransformStack,
        *,
        was_visual_neutral: bool,
        was_effect_neutral: bool,
        persistent_cache: bool = True,
    ) -> bool:
        """Apply all layout, effect, and box consequences of one transition."""
        layout_changed = self._apply_layout(
            target.glyph_slant_angle,
            persistent_cache=persistent_cache,
        )
        padding_changed = (
            self._refresh_effect_geometry(True)
            if layout_changed
            else False
        )
        geometry_changed = (
            self.refresh_compiled_geometry()
            if self.compiled.stack.transforms != target.transforms
            else False
        )
        finalized = self._finalize_neutral(
            was_visual_neutral,
            was_effect_neutral,
            target,
        )
        return (
            layout_changed
            or padding_changed
            or geometry_changed
            or finalized
        )

    def set(
        self,
        state: TextTransformStack,
        *,
        preview: bool = False,
    ) -> bool:
        """Apply complete stack/layout state, optionally as a preview."""
        item = self.item
        canonical = self.canonical()
        current = self.effective()
        if not isinstance(state, TextTransformStack):
            raise TypeError('text transform edits require TextTransformStack')
        target = state

        if preview:
            if target == current:
                return False
            was_visual_neutral = self.visual_is_neutral()
            was_effect_neutral = (
                item.effect_renderer._text_transform_is_neutral()
            )
            self.preview = None if target == canonical else target
            return self._apply_effective_transition(
                target,
                was_visual_neutral=was_visual_neutral,
                was_effect_neutral=was_effect_neutral,
                persistent_cache=False,
            )

        model_format = item.blk.fontformat
        render_format = item.fontformat
        model_changed = canonical != target
        render_format_changed = (
            render_format is not None
            and render_format is not model_format
            and render_format.text_transform != target
        )
        if target == current and not model_changed and not render_format_changed:
            return False
        had_preview = self.preview is not None
        was_visual_neutral = self.visual_is_neutral()
        was_effect_neutral = (
            item.effect_renderer._text_transform_is_neutral()
        )
        if model_changed:
            model_format.text_transform = target
        if render_format_changed:
            render_format.text_transform = target
        self.preview = None
        visual_changed = self._apply_effective_transition(
            target,
            was_visual_neutral=was_visual_neutral,
            was_effect_neutral=was_effect_neutral,
        )
        if had_preview and not visual_changed:
            # Committing identical preview geometry still switches surface
            # quality and glyph caches back to their persistent render path.
            item.update()
        changed = (
            model_changed
            or render_format_changed
            or visual_changed
        )
        return changed

    def clear_preview(self) -> bool:
        if self.preview is None:
            return False
        was_visual_neutral = self.visual_is_neutral()
        was_effect_neutral = (
            self.item.effect_renderer._text_transform_is_neutral()
        )
        self.preview = None
        target = self.canonical()
        return self._apply_effective_transition(
            target,
            was_visual_neutral=was_visual_neutral,
            was_effect_neutral=was_effect_neutral,
        )

    def sync_origin(self) -> bool:
        """Keep the Qt transform origin aligned with logical geometry.

        ``ItemTransformOriginPointHasChanged`` installs the compensated matrix
        synchronously when the origin changes, so doing that again here would
        duplicate the same matrix calculation.
        """
        item = self.item
        center = self.logical_rect().center()
        if item.transformOriginPoint() == center:
            return False
        with self.update_transaction():
            item.setTransformOriginPoint(center)
        return True

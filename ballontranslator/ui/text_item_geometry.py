"""Geometry and transform lifecycle for one ``TextBlkItem``."""

from contextlib import contextmanager
import math
from typing import Optional, TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QPoint, QPointF, QRect, QRectF, QSizeF, Qt
from qtpy.QtGui import QPainter, QPainterPath
from qtpy.QtWidgets import QGraphicsItem, QGraphicsTextItem

from ballontranslator.utils.fontformat import (
    TextTransform,
    coerce_text_transform,
    create_text_transform,
)
from ballontranslator.utils.textblock import TextAlignment
from .text_effects.curvature import NonlinearTextSurfaceRenderer
from .text_effects.raster import (
    EffectRasterAllocationError,
    RASTER_BOUNDARY_FAILURES,
)
from .text_transform_variants import text_transform_strategy

if TYPE_CHECKING:
    from .textitem import TextBlkItem


class TextItemGeometryController:
    """Own logical/display geometry and derived transform state.

    The graphics item retains only thin Qt virtual-method hooks. Effect and
    layout calls are explicit host boundaries so new transform strategies do
    not need to grow ``TextBlkItem`` itself.

    >>> from ballontranslator.utils.fontformat import SlantTextTransform
    >>> SlantTextTransform().is_neutral()
    True
    """

    def __init__(self, item: "TextBlkItem") -> None:
        self.item = item
        self.display_rect = QRectF(0, 0, 1, 1)
        self.preview: Optional[TextTransform] = None
        self.layout_renderer = None
        self.layout_renderer_type = None
        self.visual_mapper = None
        self.surface_renderer = None
        self._surface_cursor_position = -1
        self._surface_mapping_active = False
        self._transform_values_by_type = {}
        self._input_mapping_active = False
        self._input_previous_source = None
        self.installing = False
        self._box_geometry_active = False
        self._update_depth = 1
        self._update_dirty = False

    def bind_model(self) -> None:
        """Reset transient state after the item adopts a ``TextBlock``."""
        self.preview = None
        self.detach_surface_mapper()
        transform = self.canonical()
        self._transform_values_by_type = {
            transform.transform_type: transform
        }
        self._box_geometry_active = text_transform_strategy(
            transform
        ).requires_custom_resize(transform)
        self._surface_mapping_active = (
            text_transform_strategy(transform).uses_surface_mapping
            and not transform.is_neutral()
        )

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
                    self.effective(),
                    angle=item.rotation(),
                    box_pivot=item.logical_unpadded_rect().center(),
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
        transform = self.effective()
        self.item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            self.requires_custom_resize(),
        )
        # Initial layout construction establishes the final logical rectangle
        # and transform origin before ItemSendsGeometryChanges is enabled.
        # Install persisted box geometry explicitly; otherwise it remains
        # dormant until a later reshape happens to change the origin.
        self.install(transform)
        self.request_update()
        self._update_depth = 0
        self._flush_update()

    def canonical(self) -> TextTransform:
        transform = self.item.blk.fontformat.text_transform
        if not isinstance(transform, TextTransform):
            raise ValueError('live font format requires a typed text transform')
        return transform

    def effective(self) -> TextTransform:
        return self.preview if self.preview is not None else self.canonical()

    def transform_for_type(self, transform_type: str) -> TextTransform:
        """Return this item's last committed value for a transform variant."""
        current = self.canonical()
        if current.transform_type == transform_type:
            return current
        remembered = self._transform_values_by_type.get(transform_type)
        return (
            remembered
            if remembered is not None
            else create_text_transform(transform_type)
        )

    def _remember_transform(self, transform: TextTransform) -> None:
        self._transform_values_by_type[transform.transform_type] = transform

    def is_neutral(self) -> bool:
        return self.effective().is_neutral()

    def visual_is_neutral(self) -> bool:
        transform = self.effective()
        return text_transform_strategy(transform).visual_is_neutral(self.item)

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

    def request_update(self) -> None:
        self._update_dirty = True
        if self._update_depth == 0:
            self._flush_update()

    def _flush_update(self) -> None:
        if not self._update_dirty:
            return
        self._update_dirty = False
        self.item.refresh_cache_policy()
        if self.item.is_editting():
            self.item.updateMicroFocus()

    def compensated_matrix(
        self,
        values: TextTransform,
        *,
        angle: Optional[float] = None,
        box_pivot: Optional[QPointF] = None,
        rotation_pivot: Optional[QPointF] = None,
    ):
        """Build the derived Qt base transform for the current item state."""
        item = self.item
        if angle is None:
            angle = item.rotation()
        if box_pivot is None:
            box_pivot = item.logical_unpadded_rect().center()
        if rotation_pivot is None:
            rotation_pivot = item.transformOriginPoint()
        return text_transform_strategy(values).compensated_matrix(
            values,
            self.source_rect(),
            box_pivot,
            angle,
            rotation_pivot,
        )

    def bounding_rect(self, base_rect: QRectF) -> QRectF:
        """Return the Qt paint bounds with the managed display size."""
        rect = QRectF(base_rect)
        rect.setSize(self.display_rect.size())
        if self.visual_mapper is not None:
            rect = rect.united(
                self.visual_mapper.visual_bounds(self.source_rect())
            )
        return rect

    def source_rect(self) -> QRectF:
        """Return the unwarped local paint surface, including effect padding."""
        return QRectF(QPointF(), self.display_rect.size())

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
        if self.visual_mapper is not None:
            return self.visual_mapper.map_rect_path(self.logical_rect())
        path = QPainterPath()
        path.addRect(
            self.source_rect() if self.is_neutral() else self.logical_rect()
        )
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

    def map_scene_to_source(
        self,
        point: QPointF,
        previous_source: QPointF = None,
    ) -> QPointF:
        return self.map_visual_to_source(
            self.item.mapFromScene(point), previous_source
        )

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
        # Curvature is translated so its visual outline bounds remain centered
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
        if self._surface_mapping_active:
            self.refresh_surface_mapper()
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
        if (
            item.fontformat.vertical
            or item.fontformat.alignment == TextAlignment.Right
        ):
            return rect.topRight()
        if item.fontformat.alignment == TextAlignment.Left:
            return rect.topLeft()
        return rect.center()

    def _scene_scale_factor(self):
        scene = self.item.scene()
        return scene.scale_factor if hasattr(scene, 'scale_factor') else 1

    def resize(
        self,
        width: float,
        height: float,
        *,
        set_layout_maxsize: bool = False,
        set_blk_size: bool = True,
    ) -> None:
        """Resize through the current transform strategy's geometry policy."""
        if self.requires_custom_resize():
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

        old_width = self.display_rect.width()
        old_height = self.display_rect.height()
        old_center = item.sceneBoundingRect().center()
        self.display_rect.setWidth(width)
        self.display_rect.setHeight(height)
        self.sync_origin()
        pos_shift = (
            old_center - item.sceneBoundingRect().center()
        ) / self._scene_scale_factor()

        align_center = align_top_right = False
        if item.fontformat.vertical:
            align_top_right = True
        else:
            alignment = item.fontformat.alignment
            if alignment == TextAlignment.Right:
                align_top_right = True
            elif alignment != TextAlignment.Left:
                align_center = True

        if not align_center:
            delta_width = (width - old_width) / 2
            delta_height = (height - old_height) / 2
            if align_top_right:
                delta_width = -delta_width
            radians = -np.deg2rad(item.rotation())
            cosine, sine = np.cos(radians), np.sin(radians)
            pos_shift += QPointF(
                cosine * delta_width + sine * delta_height,
                -sine * delta_width + cosine * delta_height,
            )

        item.setPos(item.pos() + pos_shift)
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
                if self._surface_mapping_active:
                    self.refresh_surface_mapper()
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
        transform = self.effective()
        return text_transform_strategy(transform).requires_no_cache(transform)

    def requires_custom_resize(self) -> bool:
        return self._box_geometry_active

    def attach_surface_mapper(
        self,
        transform: TextTransform,
    ) -> bool:
        """Attach or update the nonlinear source-to-visual mapping."""
        strategy = text_transform_strategy(transform)
        wants_surface = (
            strategy.uses_surface_mapping and not transform.is_neutral()
        )
        if self.item.layout is None:
            changed = self.detach_surface_mapper()
            self._surface_mapping_active = wants_surface
            return changed
        logical_rect = self.logical_rect()
        source_rect = self.source_rect()
        if (
            logical_rect.width() <= 0.0
            or logical_rect.height() <= 0.0
            or source_rect.width() <= 0.0
            or source_rect.height() <= 0.0
        ):
            changed = self.detach_surface_mapper()
            self._surface_mapping_active = wants_surface
            return changed
        mapper = strategy.create_surface_mapper(
            logical_rect,
            source_rect,
            self.item.fontformat.vertical,
            transform,
        )
        if mapper is None:
            return self.detach_surface_mapper()
        previous = self.visual_mapper
        self.item.prepareGeometryChange()
        self.visual_mapper = mapper
        self._surface_mapping_active = True
        self._surface_cursor_position = -1
        if self.surface_renderer is None:
            self.surface_renderer = NonlinearTextSurfaceRenderer()
        else:
            self.surface_renderer.release()
        self.item.layout.input_point_mapper = self.map_visual_to_source
        self.item.update()
        self.request_update()
        return (
            previous is None
            or previous.geometry_key != mapper.geometry_key
        )

    def refresh_surface_mapper(self) -> bool:
        if not self._surface_mapping_active and self.visual_mapper is None:
            return False
        transform = self.effective()
        return self.attach_surface_mapper(transform)

    def detach_surface_mapper(self) -> bool:
        changed = (
            self.visual_mapper is not None
            or self.surface_renderer is not None
        )
        if not changed:
            if self.item.layout is not None:
                self.item.layout.input_point_mapper = None
            return False
        if self.item.layout is not None:
            self.item.prepareGeometryChange()
            self.item.layout.input_point_mapper = None
        if self.surface_renderer is not None:
            self.surface_renderer.release()
        self.visual_mapper = None
        self.surface_renderer = None
        self._surface_cursor_position = -1
        self._surface_mapping_active = False
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
            self.surface_renderer.release()

    def _paint_surface_cursor(self, painter: QPainter, mapper) -> None:
        layout = self.item.layout
        if self.item.effect_renderer.export_render:
            return
        cursor_position = layout.deferred_cursor_position
        cursor_changed = cursor_position != self._surface_cursor_position
        self._surface_cursor_position = cursor_position
        if cursor_position < 0:
            if cursor_changed:
                self.item.update()
            return
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
        layout_generation = getattr(self.item.layout, 'layout_generation', 0)
        cache_key = (
            mapper.geometry_key,
            layout_generation,
            effect_renderer.cache_generation,
            (
                0
                if effect_renderer.background_pixmap is None
                else effect_renderer.background_pixmap.cacheKey()
            ),
            self.item.document().revision(),
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
            self.item.is_editting()
            or self.item.reshaping
            or self.preview is not None
        )
        try:
            renderer.paint(
                painter,
                option,
                mapper,
                self.source_rect(),
                cache_key,
                cache_allowed=(
                    not interactive
                    and not effect_renderer.export_render
                ),
                paint_source=paint_source,
                maximum_scale=2.0 if interactive else None,
            )
            self._paint_surface_cursor(painter, mapper)
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

    def attach_layout_renderer(self, transform_type, factory):
        renderer = self.layout_renderer
        if renderer is None or self.layout_renderer_type != transform_type:
            self.detach_layout_renderer()
            renderer = factory(self.item.layout)
            self.layout_renderer = renderer
            self.layout_renderer_type = transform_type
        else:
            renderer.bind_layout(self.item.layout)
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
        self.layout_renderer_type = None
        return True

    def layout_ink_bounds(self):
        renderer = self.layout_renderer
        return QRectF() if renderer is None else renderer.ink_bounds()

    def has_layout_distortion(self) -> bool:
        """Return whether glyph painting is delegated to a transform renderer."""
        return self.layout_renderer is not None

    def draw_layout_selection_mask(self, painter, context) -> None:
        """Draw an effect mask through the active transform renderer."""
        renderer = self.layout_renderer
        if renderer is None:
            raise RuntimeError('no custom text layout renderer is active')
        renderer.draw_glyph_selection_mask(painter, context)

    def initialize_layout(self, *, persistent_cache: bool = True) -> bool:
        transform = self.effective()
        return text_transform_strategy(transform).initialize_layout(
            self.item,
            transform,
            persistent_cache,
        )

    def _apply_layout(
        self,
        previous: TextTransform,
        target: TextTransform,
        *,
        persistent_cache: bool = True,
    ) -> bool:
        rendering_changed = False
        if previous.transform_type != target.transform_type:
            rendering_changed = text_transform_strategy(previous).deactivate_layout(
                self.item,
                previous,
                persistent_cache,
            )
        target_changed = text_transform_strategy(target).apply_layout(
            self.item,
            target,
            persistent_cache,
        )
        return rendering_changed or target_changed

    def install(
        self,
        values: TextTransform,
        *,
        angle: Optional[float] = None,
        box_pivot: Optional[QPointF] = None,
        rotation_pivot: Optional[QPointF] = None,
    ) -> bool:
        """Install derived Qt geometry without lifecycle side effects."""
        matrix = self.compensated_matrix(
            values,
            angle=angle,
            box_pivot=box_pivot,
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

    def _apply_box(self, values: TextTransform) -> bool:
        self._box_geometry_active = text_transform_strategy(
            values
        ).requires_custom_resize(values)
        self.item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            self._box_geometry_active,
        )
        with self.update_transaction():
            changed = self.install(values)
            if changed:
                self.request_update()
        return changed

    def _refresh_effect_geometry(self, rendering_changed: bool) -> bool:
        item = self.item
        if rendering_changed:
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
        target: TextTransform,
    ) -> bool:
        item = self.item
        if was_visual_neutral or not target.is_neutral():
            return False
        item.effect_renderer.finalize_neutral_cache()
        return True

    def _apply_effective_transition(
        self,
        previous: TextTransform,
        target: TextTransform,
        *,
        was_visual_neutral: bool,
        persistent_cache: bool = True,
    ) -> bool:
        """Apply all layout, effect, and box consequences of one transition."""
        layout_changed = self._apply_layout(
            previous,
            target,
            persistent_cache=persistent_cache,
        )
        # Curvature changes only the final surface warp. Rebuilding the
        # unwarped stroke/shadow cache on every slider preview is both costly
        # and can briefly drop effects. Glyph-slant transitions do alter the
        # source composite and still invalidate that cache.
        source_rendering_changed = (
            layout_changed
            and (
                previous.transform_type == 'slant'
                or target.transform_type == 'slant'
            )
        )
        padding_changed = (
            self._refresh_effect_geometry(True)
            if source_rendering_changed
            else False
        )
        box_changed = self._apply_box(target)
        finalized = self._finalize_neutral(was_visual_neutral, target)
        return layout_changed or padding_changed or box_changed or finalized

    def set(self, transform: TextTransform = None, *, preview: bool = False) -> bool:
        """Apply a complete transform, optionally as transient preview state."""
        item = self.item
        raw_canonical = item.blk.fontformat.text_transform
        canonical = self.canonical()
        current = self.effective()
        target = coerce_text_transform(
            current if transform is None and preview else (
                canonical if transform is None else transform
            )
        )

        if preview:
            if target == current:
                return False
            was_visual_neutral = self.visual_is_neutral()
            self.preview = None if target == canonical else target
            return self._apply_effective_transition(
                current,
                target,
                was_visual_neutral=was_visual_neutral,
                persistent_cache=False,
            )

        model_format = item.blk.fontformat
        render_format = item.fontformat
        model_changed = raw_canonical != target
        render_format_changed = (
            render_format is not None
            and render_format is not model_format
            and render_format.text_transform != target
        )
        if target == current and not model_changed and not render_format_changed:
            self._remember_transform(target)
            return False
        was_visual_neutral = self.visual_is_neutral()
        self._remember_transform(canonical)
        if model_changed:
            model_format.text_transform = target
        if render_format_changed:
            render_format.text_transform = target
        self._remember_transform(target)
        self.preview = None
        visual_changed = self._apply_effective_transition(
            current,
            target,
            was_visual_neutral=was_visual_neutral,
        )
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
        previous = self.preview
        self.preview = None
        target = self.canonical()
        return self._apply_effective_transition(
            previous,
            target,
            was_visual_neutral=was_visual_neutral,
        )

    def sync_origin(self) -> bool:
        """Keep the Qt transform origin aligned with logical geometry.

        ``ItemTransformOriginPointHasChanged`` installs the compensated matrix
        synchronously when the origin changes, so doing that again here would
        duplicate the same matrix calculation.
        """
        item = self.item
        center = (
            self.logical_rect().center()
            if self.requires_custom_resize()
            else item.boundingRect().center()
        )
        if item.transformOriginPoint() == center:
            return False
        with self.update_transaction():
            item.setTransformOriginPoint(center)
        return True

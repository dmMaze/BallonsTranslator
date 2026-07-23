"""Transform lifecycle owned by one ``TextBlkItem`` boundary object."""

from contextlib import contextmanager
import math
from typing import Optional, TYPE_CHECKING

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtWidgets import QGraphicsItem

from ballontranslator.utils.fontformat import TextTransform, coerce_text_transform
from ballontranslator.utils.logger import logger as LOGGER
from .text_transform import text_transform_strategy

if TYPE_CHECKING:
    from .textitem import TextBlkItem


class TextItemTransformController:
    """Own committed/preview transform state and derived Qt geometry.

    The graphics item retains only thin Qt virtual-method hooks. Effect and
    layout calls are explicit host boundaries so new transform strategies do
    not need to grow ``TextBlkItem`` itself.

    >>> from ballontranslator.utils.fontformat import SlantTextTransform
    >>> SlantTextTransform().is_neutral()
    True
    """

    def __init__(self, item: "TextBlkItem") -> None:
        self.item = item
        self.preview: Optional[TextTransform] = None
        self.layout_renderer = None
        self.layout_renderer_type = None
        self.installing = False
        self._update_depth = 1
        self._update_dirty = False

    @staticmethod
    def validate_rotation_angle(angle) -> float:
        if isinstance(angle, bool):
            raise ValueError('rotation angle must be a finite number')
        try:
            angle = float(angle)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                'rotation angle must be a finite number'
            ) from error
        if not math.isfinite(angle):
            raise ValueError('rotation angle must be a finite number')
        return angle

    @staticmethod
    def report_rejected_change(change, error) -> None:
        try:
            LOGGER.warning(
                f'Rejected unsafe TextBlkItem graphics change {change}: '
                f'{error}'
            )
        except Exception:
            # Logging must never turn a rejected Qt virtual callback into an
            # exception crossing the C++/Python boundary.
            pass

    @staticmethod
    def _finite_point(point: QPointF) -> bool:
        return math.isfinite(point.x()) and math.isfinite(point.y())

    def _item_change(self, change, value, base_item_change):
        item = self.item
        if self.installing:
            return base_item_change(change, value)

        if change in (
            QGraphicsItem.GraphicsItemChange.ItemRotationChange,
            QGraphicsItem.GraphicsItemChange.ItemTransformOriginPointChange,
        ):
            candidate = base_item_change(change, value)
            try:
                if change == QGraphicsItem.GraphicsItemChange.ItemRotationChange:
                    angle = float(candidate)
                    if not math.isfinite(angle):
                        raise ValueError('rotation angle must be finite')
                    rotation_pivot = item.transformOriginPoint()
                else:
                    rotation_pivot = QPointF(candidate)
                    if not self._finite_point(rotation_pivot):
                        raise ValueError(
                            'transform origin coordinates must be finite'
                        )
                    angle = item.rotation()

                if item.blk is not None:
                    # Validate while Qt can still reject the property write.
                    self.compensated_matrix(
                        self.effective(),
                        angle=angle,
                        box_pivot=item.logical_unpadded_rect().center(),
                        rotation_pivot=rotation_pivot,
                    )
            except Exception as error:
                self.report_rejected_change(change, error)
                if change == QGraphicsItem.GraphicsItemChange.ItemRotationChange:
                    return item.rotation()
                return QPointF(item.transformOriginPoint())
            return candidate

        if change in (
            QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformOriginPointHasChanged,
        ) and item.blk is not None:
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

    def item_change(self, change, value, base_item_change):
        """Keep exceptions inside Qt's C++ virtual-call boundary."""
        try:
            return self._item_change(change, value, base_item_change)
        except Exception as error:
            self.report_rejected_change(change, error)
            try:
                return base_item_change(change, value)
            except Exception:
                return value

    def finish_initialization(self) -> None:
        self.item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            self.requires_custom_resize(),
        )
        self.request_update()
        self._update_depth = 0
        self._flush_update()

    def canonical(self) -> TextTransform:
        return coerce_text_transform(self.item.blk.fontformat.text_transform)

    def effective(self) -> TextTransform:
        return self.preview if self.preview is not None else self.canonical()

    def is_neutral(self) -> bool:
        return self.effective().is_neutral()

    def visual_is_neutral(self) -> bool:
        transform = self.effective()
        return text_transform_strategy(transform).visual_is_neutral(self.item)

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
            box_pivot,
            angle,
            rotation_pivot,
        )

    def visual_polygon(self, logical_rect):
        return text_transform_strategy(self.effective()).visual_polygon(
            self.item, logical_rect
        )

    def requires_no_cache(self) -> bool:
        transform = self.effective()
        return text_transform_strategy(transform).requires_no_cache(transform)

    def requires_custom_resize(self) -> bool:
        transform = self.effective()
        strategy = text_transform_strategy(transform)
        return strategy.requires_custom_resize(transform)

    def glyph_slant_angle(self) -> float:
        transform = self.effective()
        return text_transform_strategy(transform).glyph_slant_angle(transform)

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
        renderer.geometry_cache.invalidate_generation()
        if self.item.layout is not None:
            self.item.layout.render_delegate = None
            self.item.layout.render_failure_handler = None
        self.layout_renderer = None
        self.layout_renderer_type = None
        return True

    def layout_ink_bounds(self):
        renderer = self.layout_renderer
        return QRectF() if renderer is None else renderer.ink_bounds()

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
    ):
        rendering_changed = False
        padding_changed = False
        if previous.transform_type != target.transform_type:
            rendering_changed, padding_changed = text_transform_strategy(
                previous
            ).deactivate_layout(
                self.item,
                previous,
                persistent_cache,
            )
        target_changed, target_padding_changed = text_transform_strategy(
            target
        ).apply_layout(
            self.item,
            target,
            persistent_cache,
        )
        return (
            rendering_changed or target_changed,
            padding_changed or target_padding_changed,
        )

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
        self.item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges,
            text_transform_strategy(values).requires_custom_resize(values),
        )
        with self.update_transaction():
            changed = self.install(values)
            if changed:
                self.request_update()
        return changed

    def _refresh_effect_geometry(self, padding_changed: bool) -> bool:
        item = self.item
        padding_changed = (
            padding_changed
            or item.effect_renderer._update_effect_padding()
        )
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

    def set(self, transform: TextTransform = None, *, preview: bool = False) -> bool:
        """Apply a complete transform, optionally as transient preview state."""
        item = self.item
        raw_canonical = item.blk.fontformat.text_transform
        canonical = coerce_text_transform(raw_canonical)
        current = self.effective()
        target = coerce_text_transform(
            current if transform is None and preview else (
                canonical if transform is None else transform
            )
        )
        was_visual_neutral = self.visual_is_neutral()

        if preview:
            if target == current:
                return False
            self.preview = None if target == canonical else target
            glyph_changed, glyph_padding_changed = self._apply_layout(
                current,
                target,
                persistent_cache=False,
            )
            active_state_changed = self._refresh_effect_geometry(
                glyph_padding_changed
            )
            box_changed = self._apply_box(target)
            finalized = self._finalize_neutral(was_visual_neutral, target)
            return (
                glyph_changed
                or active_state_changed
                or box_changed
                or finalized
            )

        model_format = item.blk.fontformat
        render_format = item.fontformat
        model_changed = raw_canonical != target
        render_format_changed = (
            render_format is not None
            and render_format is not model_format
            and render_format.text_transform != target
        )
        if model_changed:
            model_format.text_transform = target
        if render_format_changed:
            render_format.text_transform = target
        self.preview = None
        glyph_changed, glyph_padding_changed = self._apply_layout(current, target)
        active_state_changed = self._refresh_effect_geometry(
            glyph_padding_changed
        )
        visual_changed = self._apply_box(target)
        finalized = self._finalize_neutral(was_visual_neutral, target)
        changed = (
            model_changed
            or render_format_changed
            or glyph_changed
            or active_state_changed
            or visual_changed
            or finalized
        )
        if target.transform_type == 'none':
            item._transform_controller = None
        return changed

    def clear_preview(self) -> bool:
        if self.preview is None:
            return False
        item = self.item
        was_visual_neutral = self.visual_is_neutral()
        previous = self.preview
        self.preview = None
        target = self.canonical()
        glyph_changed, glyph_padding_changed = self._apply_layout(previous, target)
        active_state_changed = self._refresh_effect_geometry(
            glyph_padding_changed
        )
        box_changed = self._apply_box(target)
        finalized = self._finalize_neutral(was_visual_neutral, target)
        changed = (
            glyph_changed
            or active_state_changed
            or box_changed
            or finalized
        )
        if target.transform_type == 'none':
            item._transform_controller = None
        return changed

    def recenter(self) -> bool:
        item = self.item
        center = item.logical_unpadded_rect().center()
        with self.update_transaction():
            origin_changed = item.transformOriginPoint() != center
            if origin_changed:
                item.setTransformOriginPoint(center)
            transform_changed = self.install(
                self.effective(),
                box_pivot=center,
                rotation_pivot=item.transformOriginPoint(),
            )
            if transform_changed:
                self.request_update()
        return origin_changed or transform_changed

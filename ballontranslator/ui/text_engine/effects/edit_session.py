"""Selection/global preview and undo boundaries for text effects."""

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from qtpy.QtCore import QCoreApplication

from ballontranslator.utils import config as C
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.text_effects import (
    EffectPaint,
    FilterEffect,
    GeneratedEffectPaint,
    GlowEffect,
    GradientStop,
    HollowEffect,
    ImageEffect,
    ImageGenerationRecipe,
    LinearGradientPaint,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextFillEffect,
    TextEffect,
    TextEffectStack,
    TexturePaint,
    effect_structure_key,
    effect_paint_fallback_color,
    without_project_raster_effects,
)
from .filters import FilterUnavailableError, get_filter_registry
from .image_generation import (
    ImageGenerationController,
    ImageGenerationRequest,
    create_image_generation_backend,
    prepare_image_generation_context,
)

from ... import shared_widget as SW
from ..editing.commands import SetTextEffectStackCommand

if TYPE_CHECKING:
    from .panel import TextEffectPanel
    from ..formatting.panel import FontFormatPanel
    from ..item import TextBlkItem


OVERALL_OPACITY_INDEX = -1


def matched_effect_occurrences(
    states: Sequence[TextEffectStack],
) -> Dict[int, Tuple[int, ...]]:
    """Map primary card indices to same-structure occurrences on every item.

    Occurrences pair in panel-visible order. Relative order among unrelated
    effect types is deliberately irrelevant, and Image remains item-specific.

    >>> first = TextEffectStack(effects=(StrokeEffect(), ShadowEffect()))
    >>> second = TextEffectStack(effects=(ShadowEffect(), StrokeEffect()))
    >>> matched_effect_occurrences((first, second))
    {1: (1, 0), 0: (0, 1)}
    """
    values = tuple(states)
    if len(values) < 2:
        return {}
    candidates: Dict[object, List[int]] = {}
    for index in range(len(values[0].effects) - 1, -1, -1):
        effect = values[0].effects[index]
        if isinstance(effect, (HollowEffect, ImageEffect)):
            continue
        candidates.setdefault(effect_structure_key(effect), []).append(index)
    if not candidates:
        return {}

    matches = {
        index: [index]
        for indices in candidates.values()
        for index in indices
    }
    for state in values[1:]:
        available: Dict[object, List[int]] = {}
        for index in range(len(state.effects) - 1, -1, -1):
            effect = state.effects[index]
            key = effect_structure_key(effect)
            if key in candidates and not isinstance(
                effect, (HollowEffect, ImageEffect)
            ):
                available.setdefault(key, []).append(index)
        next_candidates = {}
        for key, primary_indices in candidates.items():
            target_indices = available.get(key, ())
            paired_primary = primary_indices[:len(target_indices)]
            if not paired_primary:
                continue
            next_candidates[key] = paired_primary
            for primary_index, target_index in zip(
                paired_primary, target_indices
            ):
                matches[primary_index].append(target_index)
        candidates = next_candidates
        if not candidates:
            return {}
    return {
        index: tuple(matches[index])
        for indices in candidates.values()
        for index in indices
    }


def effect_reorder_is_aligned(
    states: Sequence[TextEffectStack], index: int
) -> bool:
    """Return whether the card's relevant visible sequences align exactly."""
    values = tuple(states)
    if not values or not 0 <= index < len(values[0].effects):
        return False
    reference = values[0].effects[index]
    family = (
        (TextFillEffect,)
        if isinstance(reference, TextFillEffect)
        else (StrokeEffect, ShadowEffect, GlowEffect, ImageEffect, FilterEffect)
    )
    if not isinstance(reference, family):
        return False
    sequences = [
        tuple(
            effect_structure_key(effect)
            for effect in reversed(state.effects)
            if isinstance(effect, family)
        )
        for state in values
    ]
    return all(sequence == sequences[0] for sequence in sequences[1:])


class TextEffectEditSession:
    """Own one complete-stack preview and commit transaction.

    >>> session = object.__new__(TextEffectEditSession)
    >>> session.items = []
    >>> session.items
    []
    """

    def __init__(
        self,
        host: "FontFormatPanel",
        controls: Optional["TextEffectPanel"] = None,
    ) -> None:
        self.host = host
        self.controls = controls
        self.items = []
        self.preview_before = None
        self.preview_key = None
        self._matched_occurrences: Dict[int, Tuple[int, ...]] = {}
        self._pending_image_generation = None
        self._image_generation_controller = None
        if controls is not None:
            controls.value_commit_requested.connect(self.commit_value)
            controls.value_preview_requested.connect(self.preview_value)
            controls.parameter_preview_requested.connect(
                self.preview_parameter_delta
            )
            controls.parameter_commit_requested.connect(
                self.commit_parameter_delta
            )
            controls.preview_canceled.connect(self.cancel_preview)
            controls.add_effect_requested.connect(self.add_effect)
            controls.add_filter_requested.connect(self.add_filter)
            controls.hollow_enabled_requested.connect(
                self.set_hollow_enabled
            )
            controls.remove_effect_requested.connect(self.remove_effect)
            controls.move_effect_requested.connect(self.move_effect)
            controls.texture_file_requested.connect(self.import_texture)
            controls.image_file_requested.connect(self.import_image)
            controls.image_generation_requested.connect(
                self.generate_image
            )
            controls.image_generation_stop_requested.connect(
                self.stop_image_generation
            )
            controller = ImageGenerationController(controls)
            controller.generated.connect(self._finish_image_generation)
            controller.failed.connect(self._fail_image_generation)
            controller.state_changed.connect(
                controls.set_image_generation_state
            )
            controller.state_changed.connect(
                self._on_image_generation_state_changed
            )
            self._image_generation_controller = controller

    @staticmethod
    def _state_for_item(item: "TextBlkItem") -> TextEffectStack:
        return item.blk.fontformat.text_effects

    def _current_states(self) -> Tuple[TextEffectStack, ...]:
        if self.items:
            return tuple(self._state_for_item(item) for item in self.items)
        return (self.host.global_format.text_effects,)

    def _validate_states(
        self, states: Sequence[TextEffectStack]
    ) -> Tuple[TextEffectStack, ...]:
        values = tuple(states)
        expected = len(self.items) if self.items else 1
        if len(values) != expected:
            raise ValueError('owners and effect states must have the same length')
        if any(not isinstance(value, TextEffectStack) for value in values):
            raise TypeError('effect edit session requires TextEffectStack values')
        return values

    @staticmethod
    def _convert_effect_paint(
        paint: EffectPaint,
        paint_type: str,
    ) -> GeneratedEffectPaint:
        """Convert an effect Fill while preserving its visible color.

        >>> converted = TextEffectEditSession._convert_effect_paint(
        ...     SolidPaint((1, 2, 3)), 'linear_gradient'
        ... )
        >>> converted.stops[-1].opacity
        0.0
        """
        if paint_type not in {'solid', 'linear_gradient'}:
            raise ValueError('unsupported effect paint type')
        if paint_type == 'solid':
            if isinstance(paint, SolidPaint):
                return paint
            return SolidPaint(effect_paint_fallback_color(paint))
        if isinstance(paint, LinearGradientPaint):
            return paint
        color = effect_paint_fallback_color(paint)
        return LinearGradientPaint(stops=(
            GradientStop(0.0, color, 1.0),
            GradientStop(1.0, color, 0.0),
        ))

    @staticmethod
    def _with_value(
        state: TextEffectStack,
        index: int,
        param_name: str,
        value,
    ) -> TextEffectStack:
        if index == OVERALL_OPACITY_INDEX:
            if param_name != 'overall_opacity':
                raise ValueError('unknown overall text effect field')
            return replace(state, overall_opacity=value)
        if index < 0 or index >= len(state.effects):
            raise IndexError('text effect index is no longer current')
        effect = state.effects[index]
        parameters = {}
        if isinstance(effect, StrokeEffect):
            if param_name not in {
                'enabled', 'width', 'opacity', 'paint', 'paint_type',
                'position', 'blend_mode',
            }:
                raise ValueError('unknown Stroke field')
            if param_name == 'paint':
                if not isinstance(value, (SolidPaint, LinearGradientPaint)):
                    value = SolidPaint(value)
                parameters['paint'] = value
            elif param_name == 'paint_type':
                parameters['paint'] = (
                    TextEffectEditSession._convert_effect_paint(
                        effect.paint, value
                    )
                )
            else:
                parameters[param_name] = value
        elif isinstance(effect, ShadowEffect):
            if param_name not in {
                'enabled', 'opacity', 'shadow_type', 'paint', 'paint_type',
                'angle', 'distance', 'blur', 'spread', 'blend_mode',
            }:
                raise ValueError('unknown Shadow field')
            elif param_name == 'paint':
                if not isinstance(value, (SolidPaint, LinearGradientPaint)):
                    value = SolidPaint(value)
                parameters['paint'] = value
            elif param_name == 'paint_type':
                parameters['paint'] = (
                    TextEffectEditSession._convert_effect_paint(
                        effect.paint, value
                    )
                )
            else:
                parameters[param_name] = value
        elif isinstance(effect, GlowEffect):
            if param_name not in {
                'enabled', 'opacity', 'glow_type', 'paint', 'paint_type',
                'size', 'spread', 'blend_mode',
            }:
                raise ValueError('unknown Glow field')
            if param_name == 'paint':
                if not isinstance(value, (SolidPaint, LinearGradientPaint)):
                    value = SolidPaint(value)
                parameters['paint'] = value
            elif param_name == 'paint_type':
                parameters['paint'] = (
                    TextEffectEditSession._convert_effect_paint(
                        effect.paint, value
                    )
                )
            else:
                parameters[param_name] = value
        elif isinstance(effect, HollowEffect):
            if param_name != 'enabled':
                raise ValueError('unknown Hollow field')
            parameters['enabled'] = value
        elif isinstance(effect, TextFillEffect):
            if param_name not in {
                'enabled', 'paint', 'texture_mapping',
                'texture_scale', 'opacity', 'blend_mode',
            }:
                raise ValueError('unknown Text Fill field')
            if param_name == 'paint':
                if not isinstance(value, (LinearGradientPaint, TexturePaint)):
                    raise TypeError(
                        'Text Fill paint must be Gradient or Texture paint'
                    )
                parameters['paint'] = value
            elif param_name in {'texture_mapping', 'texture_scale'}:
                if not isinstance(effect.paint, TexturePaint):
                    raise TypeError('texture controls require Texture paint')
                parameters['paint'] = replace(
                    effect.paint,
                    **{
                        'mapping' if param_name == 'texture_mapping' else 'scale':
                        value
                    },
                )
            else:
                parameters[param_name] = value
        elif isinstance(effect, ImageEffect):
            if param_name not in {'asset', 'enabled', 'mode'}:
                raise ValueError('unknown Image field')
            parameters[param_name] = value
        elif isinstance(effect, FilterEffect):
            if param_name == 'enabled':
                parameters['enabled'] = value
            elif param_name.startswith('param:'):
                key = param_name.removeprefix('param:')
                if not key:
                    raise ValueError('filter parameter name is empty')
                runtime = get_filter_registry().resolve(effect)
                params = dict(runtime.params)
                params[key] = value
                parameters['params'] = params
                parameters['schema_version'] = runtime.spec.schema_version
            else:
                raise ValueError('unknown Filter field')
        else:
            raise ValueError('selected text effect type is unsupported')
        effects = list(state.effects)
        effects[index] = replace(effect, **parameters)
        return replace(state, effects=tuple(effects))

    @staticmethod
    def _value_at(
        state: TextEffectStack, index: int, param_name: str
    ):
        if index == OVERALL_OPACITY_INDEX:
            if param_name != 'overall_opacity':
                raise ValueError('unknown overall text effect field')
            return state.overall_opacity
        if index < 0 or index >= len(state.effects):
            raise IndexError('text effect index is no longer current')
        effect = state.effects[index]
        if isinstance(effect, TextFillEffect) and param_name in {
            'texture_mapping', 'texture_scale'
        }:
            if not isinstance(effect.paint, TexturePaint):
                raise TypeError('texture controls require Texture paint')
            return (
                effect.paint.mapping
                if param_name == 'texture_mapping'
                else effect.paint.scale
            )
        if isinstance(effect, FilterEffect) and param_name.startswith('param:'):
            key = param_name.removeprefix('param:')
            return get_filter_registry().resolve(effect).params[key]
        return getattr(effect, param_name)

    def _set_global_effects(self, state: TextEffectStack) -> None:
        state = without_project_raster_effects(state)
        self.host.global_format.text_effects = state
        active = C.active_format
        if active is self.host.global_format:
            active.text_effects = state

    def _apply_preview_states(
        self, states: Sequence[TextEffectStack]
    ) -> bool:
        targets = self._validate_states(states)
        if self.items:
            changed = False
            for item, state in zip(self.items, targets):
                changed = item.set_text_effects(state, preview=True) or changed
            return changed
        changed = self.host.global_format.text_effects != targets[0]
        self._set_global_effects(targets[0])
        return changed

    def _sync_effect_ui(self) -> None:
        self._refresh_occurrence_mapping()
        controls = self.controls
        if self.items:
            if controls is not None:
                controls.set_effect_items(self.items)
            if len(self.items) == 1:
                item = self.items[0]
                current_item = getattr(self.host, 'textblk_item', None)
                if current_item is item and C.active_format is not None:
                    C.active_format.text_effects = self._state_for_item(item)
        elif controls is not None:
            controls.set_active_format(self.host.global_format)

    def _commit_complete_states(
        self,
        before: Sequence[TextEffectStack],
        after: Sequence[TextEffectStack],
    ) -> bool:
        before = tuple(before)
        after = tuple(after)
        if not self.items:
            after = tuple(
                without_project_raster_effects(state) for state in after
            )
            changed = before != after
            self._set_global_effects(after[0])
            if changed and hasattr(self.host, 'update_text_style_label'):
                self.host.update_text_style_label()
            self._sync_effect_ui()
            return changed
        command = SetTextEffectStackCommand.create(
            self.items, before, after, self._sync_effect_ui
        )
        if command is None:
            for item in self.items:
                item.clear_text_effect_preview()
            self._sync_effect_ui()
            return False
        SW.canvas.push_undo_command(command)
        return True

    def replace_targets(self, items: Sequence["TextBlkItem"]) -> None:
        replacements = list(items)
        changed = len(replacements) != len(self.items) or any(
            current is not replacement
            for current, replacement in zip(self.items, replacements)
        )
        if changed:
            self.stop_image_generation(detach_card=True)
            self.cancel_preview()
        self.items = replacements
        self._refresh_occurrence_mapping()

    def _refresh_occurrence_mapping(
        self, states: Optional[Sequence[TextEffectStack]] = None
    ) -> None:
        values = self._current_states() if states is None else tuple(states)
        self._matched_occurrences = matched_effect_occurrences(values)

    def preview_states(self, states: Sequence[TextEffectStack]) -> bool:
        """Preview complete selected-item states for the item boundary API."""
        if not self.items:
            return False
        targets = self._validate_states(states)
        if self.preview_before is None:
            self.preview_before = self._current_states()
            self.preview_key = ('complete-stack',)
        return self._apply_preview_states(targets)

    def commit_states(
        self, states: Optional[Sequence[TextEffectStack]] = None
    ) -> bool:
        if not self.items and not hasattr(self.host, 'global_format'):
            self.preview_before = None
            self.preview_key = None
            return False
        before = (
            self._current_states()
            if self.preview_before is None
            else self.preview_before
        )
        if states is None:
            after = (
                tuple(item.effective_text_effects() for item in self.items)
                if self.items else self._current_states()
            )
        else:
            after = self._validate_states(states)
        self.preview_before = None
        self.preview_key = None
        return self._commit_complete_states(before, after)

    def _begin_preview(self, key: tuple) -> Tuple[TextEffectStack, ...]:
        if self.preview_before is not None and self.preview_key != key:
            self.cancel_preview()
        if self.preview_before is None:
            self.preview_before = self._current_states()
            self.preview_key = key
        return self.preview_before

    def _target_indices(
        self, states: Sequence[TextEffectStack], index: int
    ) -> Tuple[Optional[int], ...]:
        if index == OVERALL_OPACITY_INDEX:
            return (index,) * len(states)
        if len(states) <= 1:
            return (index,) * len(states)
        matched = self._matched_occurrences.get(index)
        if matched is not None:
            return matched
        return (index,) + (None,) * (len(states) - 1)

    def preview_value(
        self, index: int, param_name: str, value
    ) -> None:
        key = (int(index), str(param_name))
        before = self._begin_preview(key)
        target_indices = self._target_indices(before, index)
        try:
            after = [
                state if target_index is None else self._with_value(
                    state, target_index, param_name, value
                )
                for state, target_index in zip(before, target_indices)
            ]
        except (
            AttributeError,
            FilterUnavailableError,
            IndexError,
            TypeError,
            ValueError,
        ):
            self.cancel_preview()
            return
        self._apply_preview_states(after)

    def preview_parameter_delta(
        self, index: int, param_name: str, canonical_delta: float
    ) -> None:
        key = (int(index), str(param_name))
        before = self._begin_preview(key)
        target_indices = self._target_indices(before, index)
        try:
            after = [
                state if target_index is None else self._with_value(
                    state,
                    target_index,
                    param_name,
                    self._value_at(state, target_index, param_name)
                    + canonical_delta,
                )
                for state, target_index in zip(before, target_indices)
            ]
        except (
            AttributeError,
            FilterUnavailableError,
            IndexError,
            TypeError,
            ValueError,
        ):
            self.cancel_preview()
            return
        self._apply_preview_states(after)

    def commit_value(self, index: int, param_name: str, value) -> bool:
        key = (int(index), str(param_name))
        if self.preview_before is not None and self.preview_key != key:
            self.cancel_preview()
        before = self.preview_before or self._current_states()
        target_indices = self._target_indices(before, index)
        try:
            after = [
                state if target_index is None else self._with_value(
                    state, target_index, param_name, value
                )
                for state, target_index in zip(before, target_indices)
            ]
        except (
            AttributeError,
            FilterUnavailableError,
            IndexError,
            TypeError,
            ValueError,
        ):
            self.cancel_preview()
            self._sync_effect_ui()
            return False
        self.preview_before = None
        self.preview_key = None
        return self._commit_complete_states(before, after)

    def commit_parameter_delta(
        self, index: int, param_name: str, canonical_delta: float
    ) -> bool:
        key = (int(index), str(param_name))
        if self.preview_before is None or self.preview_key != key:
            return False
        before = self.preview_before
        target_indices = self._target_indices(before, index)
        try:
            after = [
                state if target_index is None else self._with_value(
                    state,
                    target_index,
                    param_name,
                    self._value_at(state, target_index, param_name)
                    + canonical_delta,
                )
                for state, target_index in zip(before, target_indices)
            ]
        except (
            AttributeError,
            FilterUnavailableError,
            IndexError,
            TypeError,
            ValueError,
        ):
            self.cancel_preview()
            return False
        self.preview_before = None
        self.preview_key = None
        return self._commit_complete_states(before, after)

    def _prepare_structure_change(self) -> None:
        self.stop_image_generation(detach_card=True)
        if self.controls is not None:
            self.controls.finish_pending_effect_edits()
            self.controls.cancel_effect_previews()
        self.cancel_preview()
        self._refresh_occurrence_mapping()

    @staticmethod
    def _insertion_index(
        state: TextEffectStack, effect: TextEffect
    ) -> int:
        if isinstance(effect, HollowEffect):
            return len(state.effects)
        # Raw order is topmost-first. New movable effects and structural Fills
        # therefore land at zero so reverse/application order appends them.
        return 0

    @staticmethod
    def _matched_insertion_index(
        state: TextEffectStack,
        effect: TextEffect,
        visible_occurrence: int,
    ) -> int:
        """Insert after the occurrences already common to every target."""
        key = effect_structure_key(effect)
        visible_indices = [
            index
            for index in range(len(state.effects) - 1, -1, -1)
            if effect_structure_key(state.effects[index]) == key
        ]
        if not visible_indices:
            return TextEffectEditSession._insertion_index(state, effect)
        if visible_occurrence >= len(visible_indices):
            return TextEffectEditSession._insertion_index(state, effect)
        return visible_indices[visible_occurrence] + 1

    @staticmethod
    def _common_occurrence_budget(
        states: Sequence[TextEffectStack], effect: TextEffect
    ) -> int:
        key = effect_structure_key(effect)
        return min(
            sum(
                effect_structure_key(candidate) == key
                for candidate in state.effects
            )
            for state in states
        )

    def _insert_effect(
        self,
        before: Sequence[TextEffectStack],
        effect: TextEffect,
    ) -> bool:
        """Insert one effect across the active targets and reveal its card.

        >>> session = object.__new__(TextEffectEditSession)
        >>> session.controls = None
        >>> session._commit_complete_states = lambda before, after: True
        >>> session._insert_effect((TextEffectStack(),), StrokeEffect())
        True
        """
        common_budget = (
            self._common_occurrence_budget(before, effect)
            if len(before) > 1 else None
        )
        after = []
        primary_insert_index: Optional[int] = None
        for state in before:
            effects = list(state.effects)
            insert_index = (
                self._insertion_index(state, effect)
                if common_budget is None
                else self._matched_insertion_index(
                    state, effect, common_budget
                )
            )
            if primary_insert_index is None:
                primary_insert_index = insert_index
            effects.insert(insert_index, effect)
            after.append(replace(state, effects=tuple(effects)))
        changed = self._commit_complete_states(before, after)
        if (
            changed
            and self.controls is not None
            and primary_insert_index is not None
        ):
            self.controls.reveal_effect_card(primary_insert_index)
        return changed

    def add_effect(self, effect_type: str) -> bool:
        self._prepare_structure_change()
        before = self._current_states()
        constructors = {
            'stroke': StrokeEffect,
            'shadow': ShadowEffect,
            'glow': GlowEffect,
            'gradient': lambda: TextFillEffect(
                paint=LinearGradientPaint()
            ),
            'texture': lambda: TextFillEffect(paint=TexturePaint()),
            'image': ImageEffect,
        }
        constructor = constructors.get(effect_type)
        if constructor is None or (
            effect_type in {'texture', 'image'}
            and (
                not self.items
                or getattr(SW.canvas, 'imgtrans_proj', None) is None
            )
        ) or (
            effect_type == 'image' and len(self.items) != 1
        ):
            self._sync_effect_ui()
            return False
        effect = constructor()
        return self._insert_effect(before, effect)

    def add_filter(self, filter_id: str) -> bool:
        """Append one repeatable filter with its metadata defaults."""
        self._prepare_structure_change()
        before = self._current_states()
        spec = get_filter_registry().get_spec(filter_id)
        if spec is None:
            self._sync_effect_ui()
            return False
        effect = FilterEffect(
            spec.filter_id,
            schema_version=spec.schema_version,
            params=spec.default_params(),
        )
        return self._insert_effect(before, effect)

    def import_texture(self, index: int, source_path: str) -> bool:
        """Import a managed texture and commit it as one complete-stack edit."""
        return self._import_project_raster(index, source_path, 'texture')

    def import_image(self, index: int, source_path: str) -> bool:
        """Import one managed asset for the primary Image card."""
        return self._import_project_raster(index, source_path, 'image')

    def generate_image(
        self, index: int, recipe: ImageGenerationRecipe
    ) -> bool:
        """Start one single-item Image generation without mutating state."""
        from ..item import TextBlkItem

        controller = self._image_generation_controller
        if (
            controller is None
            or controller.active
            or not isinstance(recipe, ImageGenerationRecipe)
            or len(self.items) != 1
            or not isinstance(self.items[0], TextBlkItem)
        ):
            return False
        self.finish_pending_edits()
        self.cancel_preview()
        item = self.items[0]
        state = self._state_for_item(item)
        if index < 0 or index >= len(state.effects):
            return False
        effect = state.effects[index]
        if not isinstance(effect, ImageEffect):
            return False
        scene = item.scene()
        project = (
            None if scene is None else getattr(scene, 'imgtrans_proj', None)
        )
        try:
            if project is None:
                raise ValueError(
                    QCoreApplication.translate(
                        'TextEffectEditSession',
                        'Open a project before generating an Image.',
                    )
                )
            context_image = prepare_image_generation_context(
                item, project, recipe.context
            )
            backend = create_image_generation_backend(recipe)
        except (MemoryError, RuntimeError, TypeError, ValueError) as error:
            if self.controls is not None:
                self.controls.show_image_generation_context_error(
                    index, str(error)
                )
            return False
        request = ImageGenerationRequest(recipe, context_image)
        self._pending_image_generation = (
            item,
            int(index),
            effect,
            project,
            project.load_identity,
            project.current_img,
            recipe,
        )
        if not controller.start(index, backend, request):
            self._pending_image_generation = None
            backend.close()
            return False
        return True

    def stop_image_generation(self, *, detach_card: bool = False) -> bool:
        controller = self._image_generation_controller
        stopped = bool(controller is not None and controller.stop())
        if detach_card and stopped and self.controls is not None:
            self.controls.detach_image_generation_card()
        return stopped

    def _generation_target_is_current(self) -> bool:
        pending = self._pending_image_generation
        if pending is None:
            return False
        (
            item,
            index,
            original_effect,
            project,
            load_identity,
            current_img,
            _recipe,
        ) = pending
        try:
            scene = item.scene()
            if (
                len(self.items) != 1
                or self.items[0] is not item
                or project.load_identity is not load_identity
                or project.current_img != current_img
                or scene is None
                or getattr(scene, 'imgtrans_proj', None) is not project
            ):
                return False
            state = self._state_for_item(item)
        except RuntimeError:
            # An asynchronous result may arrive after Qt deleted its target.
            return False
        if not (0 <= index < len(state.effects)):
            return False
        current = state.effects[index]
        return bool(
            isinstance(current, ImageEffect)
            and current.asset == original_effect.asset
            and current.generation == original_effect.generation
        )

    def _finish_image_generation(self, index: int, payload: bytes) -> None:
        pending = self._pending_image_generation
        if pending is None or index != pending[1]:
            return
        if not self._generation_target_is_current():
            self._pending_image_generation = None
            return
        item, index, _effect, project, *_rest, recipe = pending
        before = (self._state_for_item(item),)
        try:
            asset = project.import_raster_asset_bytes(
                payload, 'generated.png'
            )
            current = before[0].effects[index]
            if not isinstance(current, ImageEffect):
                raise TypeError('selected effect is no longer Image')
            effects = list(before[0].effects)
            effects[index] = replace(
                current, asset=asset, generation=recipe
            )
            after = (replace(before[0], effects=tuple(effects)),)
        except (IndexError, OSError, TypeError, ValueError) as error:
            self._pending_image_generation = None
            if self.controls is not None:
                self.controls.show_image_generation_error(index, error)
            return
        self._pending_image_generation = None
        changed = self._commit_complete_states(before, after)
        if not changed and self.controls is not None:
            # Content-addressed import may have restored a missing file even
            # when the immutable asset and recipe are already identical.
            self.controls.project_assets_changed()

    def _fail_image_generation(
        self, index: int, error: Exception
    ) -> None:
        pending = self._pending_image_generation
        self._pending_image_generation = None
        if (
            pending is not None
            and index == pending[1]
            and self.controls is not None
        ):
            self.controls.show_image_generation_error(index, error)

    def _on_image_generation_state_changed(
        self, index: int, state: str
    ) -> None:
        del index
        if state == 'idle':
            self._pending_image_generation = None

    def _import_project_raster(
        self, index: int, source_path: str, kind: str
    ) -> bool:
        """Run one project import and whole-stack commit transaction."""
        if kind not in {'texture', 'image'}:
            raise ValueError('unsupported project raster effect kind')
        self._prepare_structure_change()
        if not self.items:
            self._sync_effect_ui()
            return False
        before = self._current_states()
        target_indices = self._target_indices(before, index)
        project = getattr(SW.canvas, 'imgtrans_proj', None)
        label = 'Text Fill texture' if kind == 'texture' else 'Image'
        try:
            if project is None:
                article = 'an' if kind == 'image' else 'a'
                raise ValueError(
                    f'Open a project before importing {article} {label}.'
                )
            asset = project.import_raster_asset(source_path)
            after = []
            for state, target_index in zip(before, target_indices):
                if target_index is None:
                    after.append(state)
                    continue
                if target_index < 0 or target_index >= len(state.effects):
                    raise IndexError(f'{label} index is no longer current')
                effect = state.effects[target_index]
                if kind == 'texture':
                    if not isinstance(effect, TextFillEffect):
                        raise TypeError('selected effect is not Text Fill')
                    paint = (
                        replace(effect.paint, asset=asset)
                        if isinstance(effect.paint, TexturePaint)
                        else TexturePaint(asset)
                    )
                    replacement = replace(effect, paint=paint)
                else:
                    if not isinstance(effect, ImageEffect):
                        raise TypeError('selected effect is not Image')
                    replacement = replace(effect, asset=asset)
                effects = list(state.effects)
                effects[target_index] = replacement
                after.append(replace(state, effects=tuple(effects)))
        except (IndexError, OSError, TypeError, ValueError) as error:
            LOGGER.warning('Unable to import %s: %s', label, error)
            self._sync_effect_ui()
            if self.controls is not None:
                if kind == 'texture':
                    self.controls.show_texture_import_error(index, str(error))
                else:
                    self.controls.show_image_import_error(index, str(error))
            return False
        changed = self._commit_complete_states(before, after)
        if not changed and self.controls is not None:
            # Same-digest import can restore a missing managed file without
            # changing the immutable effect value.
            self.controls.project_assets_changed()
        return changed

    def set_hollow_enabled(self, enabled: bool) -> bool:
        """Enable the unique Hollow value, inserting it when first used.

        >>> from types import SimpleNamespace
        >>> owner = SimpleNamespace(text_effects=TextEffectStack())
        >>> session = TextEffectEditSession(
        ...     SimpleNamespace(global_format=owner)
        ... )
        >>> session.set_hollow_enabled(True)
        True
        >>> owner.text_effects.effects[0].enabled
        True
        """
        self._prepare_structure_change()
        before = self._current_states()
        after = []
        for state in before:
            effects = list(state.effects)
            index = next(
                (
                    index
                    for index, effect in enumerate(effects)
                    if isinstance(effect, HollowEffect)
                ),
                None,
            )
            if index is None:
                if enabled:
                    effect = HollowEffect()
                    effects.insert(
                        self._insertion_index(state, effect), effect
                    )
            elif effects[index].enabled != enabled:
                effects[index] = replace(effects[index], enabled=enabled)
            after.append(replace(state, effects=tuple(effects)))
        return self._commit_complete_states(before, after)

    def remove_effect(self, index: int) -> bool:
        self._prepare_structure_change()
        before = self._current_states()
        target_indices = self._target_indices(before, index)
        if index < 0:
            self._sync_effect_ui()
            return False
        after = []
        for state, target_index in zip(before, target_indices):
            if target_index is None:
                after.append(state)
                continue
            if target_index >= len(state.effects):
                self._sync_effect_ui()
                return False
            effects = list(state.effects)
            del effects[target_index]
            after.append(replace(state, effects=tuple(effects)))
        return self._commit_complete_states(before, after)

    def move_effect(self, index: int, direction: int) -> bool:
        self._prepare_structure_change()
        before = self._current_states()
        if (
            direction not in (-1, 1)
            or index < 0
            or index >= len(before[0].effects)
        ):
            self._sync_effect_ui()
            return False
        effect = before[0].effects[index]
        movable_types = (
            (TextFillEffect,)
            if isinstance(effect, TextFillEffect)
            else (
                StrokeEffect, ShadowEffect, GlowEffect,
                ImageEffect, FilterEffect,
            )
        )
        if not isinstance(effect, movable_types):
            self._sync_effect_ui()
            return False
        target_indices = self._target_indices(before, index)
        if (
            len(before) > 1
            and all(target_index is not None for target_index in target_indices)
            and not effect_reorder_is_aligned(before, index)
        ):
            self._sync_effect_ui()
            return False
        after = []
        for state, target_index in zip(before, target_indices):
            if target_index is None:
                after.append(state)
                continue
            movable_indices = [
                effect_index
                for effect_index, candidate in enumerate(state.effects)
                if isinstance(candidate, movable_types)
            ]
            try:
                position = movable_indices.index(target_index)
                destination = movable_indices[position + direction]
            except (IndexError, ValueError):
                self._sync_effect_ui()
                return False
            effects = list(state.effects)
            effects[target_index], effects[destination] = (
                effects[destination], effects[target_index]
            )
            after.append(replace(state, effects=tuple(effects)))
        return self._commit_complete_states(before, after)

    def cancel_preview(self, *_key) -> bool:
        before = self.preview_before
        changed = False
        if self.items:
            for item in self.items:
                changed = item.clear_text_effect_preview() or changed
        elif before is not None:
            changed = self.host.global_format.text_effects != before[0]
            self._set_global_effects(before[0])
        self.preview_before = None
        self.preview_key = None
        if before is not None:
            self._sync_effect_ui()
        return changed

    def finish_pending_edits(self) -> None:
        if self.controls is not None:
            self.controls.finish_pending_effect_edits()

    def resolve_for_save(self) -> None:
        self.finish_pending_edits()
        if self.controls is not None:
            self.controls.cancel_effect_previews()
        self.cancel_preview()

    def resolve_for_history_change(self) -> None:
        self.stop_image_generation(detach_card=True)
        if self.controls is not None:
            self.controls.cancel_pending_effect_edits()
            self.controls.cancel_effect_previews()
        self.cancel_preview()

    def resolve_for_page_change(self) -> None:
        self.stop_image_generation(detach_card=True)
        self.resolve_for_save()
        self.items = []
        self._matched_occurrences = {}

    def cancel_for_scene_change(self) -> None:
        self.stop_image_generation(detach_card=True)
        if self.controls is not None:
            self.controls.cancel_pending_effect_edits()
            self.controls.cancel_effect_previews()
        self.cancel_preview()
        self.items = []
        self._matched_occurrences = {}

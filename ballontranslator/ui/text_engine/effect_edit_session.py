"""Selection/global preview and undo boundaries for text effects."""

from dataclasses import replace
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

from ballontranslator.utils import config as C
from ballontranslator.utils.text_effects import (
    EffectPaint,
    GradientOverlayEffect,
    GradientStop,
    HollowEffect,
    LinearGradientPaint,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffect,
    TextEffectStack,
    effect_phase,
    effect_paint_fallback_color,
)

from .. import shared_widget as SW
from .editing.commands import SetTextEffectStackCommand

if TYPE_CHECKING:
    from .formatting.effects import TextEffectPanel
    from .formatting.panel import FontFormatPanel
    from .item import TextBlkItem


OVERALL_OPACITY_INDEX = -1


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
            controls.remove_effect_requested.connect(self.remove_effect)
            controls.move_effect_requested.connect(self.move_effect)

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
    def _effect_sequence(state: TextEffectStack) -> Tuple[str, ...]:
        return tuple(effect.effect_type for effect in state.effects)

    @classmethod
    def _has_common_stack_shape(
        cls, states: Sequence[TextEffectStack]
    ) -> bool:
        sequences = [cls._effect_sequence(state) for state in states]
        return not sequences or all(
            sequence == sequences[0] for sequence in sequences
        )

    @staticmethod
    def _convert_stroke_paint(
        paint: EffectPaint,
        paint_type: str,
        mixed_values: bool,
    ) -> EffectPaint:
        """Convert Stroke Fill without inventing a shared mixed value.

        >>> converted = TextEffectEditSession._convert_stroke_paint(
        ...     SolidPaint((1, 2, 3)), 'linear_gradient', False
        ... )
        >>> converted.stops[-1].opacity
        0.0
        """
        if paint_type not in {'solid', 'linear_gradient'}:
            raise ValueError('unsupported Stroke paint type')
        if mixed_values:
            return (
                SolidPaint()
                if paint_type == 'solid'
                else LinearGradientPaint()
            )
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
                'position',
            }:
                raise ValueError('unknown Stroke field')
            if param_name == 'paint':
                if not isinstance(value, (SolidPaint, LinearGradientPaint)):
                    value = SolidPaint(value)
                parameters['paint'] = value
            elif param_name == 'paint_type':
                paint_type, mixed_values = value
                parameters['paint'] = (
                    TextEffectEditSession._convert_stroke_paint(
                        effect.paint, paint_type, mixed_values
                    )
                )
            else:
                parameters[param_name] = value
        elif isinstance(effect, ShadowEffect):
            if param_name in {'offset_x', 'offset_y'}:
                offset = list(effect.offset)
                offset[0 if param_name == 'offset_x' else 1] = value
                parameters['offset'] = tuple(offset)
            elif param_name in {
                'enabled', 'opacity', 'shadow_type', 'color', 'blur', 'spread'
            }:
                parameters[param_name] = value
            else:
                raise ValueError('unknown Shadow field')
        elif isinstance(effect, HollowEffect):
            if param_name != 'enabled':
                raise ValueError('unknown Hollow field')
            parameters['enabled'] = value
        elif isinstance(effect, GradientOverlayEffect):
            if param_name not in {'enabled', 'opacity', 'paint'}:
                raise ValueError('unknown Gradient Overlay field')
            if param_name == 'paint' and not isinstance(
                value, LinearGradientPaint
            ):
                raise TypeError(
                    'Gradient Overlay paint must be LinearGradientPaint'
                )
            parameters[param_name] = value
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
        if isinstance(effect, ShadowEffect) and param_name in {
            'offset_x', 'offset_y'
        }:
            return effect.offset[0 if param_name == 'offset_x' else 1]
        return getattr(effect, param_name)

    def _set_global_effects(self, state: TextEffectStack) -> None:
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
            self.cancel_preview()
        self.items = replacements

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

    def preview_value(
        self, index: int, param_name: str, value
    ) -> None:
        key = (int(index), str(param_name))
        before = self._begin_preview(key)
        if (
            index != OVERALL_OPACITY_INDEX
            and not self._has_common_stack_shape(before)
        ):
            self.cancel_preview()
            return
        try:
            after = [
                self._with_value(state, index, param_name, value)
                for state in before
            ]
        except (AttributeError, IndexError, TypeError, ValueError):
            self.cancel_preview()
            return
        self._apply_preview_states(after)

    def preview_parameter_delta(
        self, index: int, param_name: str, canonical_delta: float
    ) -> None:
        key = (int(index), str(param_name))
        before = self._begin_preview(key)
        if (
            index != OVERALL_OPACITY_INDEX
            and not self._has_common_stack_shape(before)
        ):
            self.cancel_preview()
            return
        try:
            after = [
                self._with_value(
                    state,
                    index,
                    param_name,
                    self._value_at(state, index, param_name)
                    + canonical_delta,
                )
                for state in before
            ]
        except (AttributeError, IndexError, TypeError, ValueError):
            self.cancel_preview()
            return
        self._apply_preview_states(after)

    def commit_value(self, index: int, param_name: str, value) -> bool:
        key = (int(index), str(param_name))
        if self.preview_before is not None and self.preview_key != key:
            self.cancel_preview()
        before = self.preview_before or self._current_states()
        if (
            index != OVERALL_OPACITY_INDEX
            and not self._has_common_stack_shape(before)
        ):
            self.cancel_preview()
            self._sync_effect_ui()
            return False
        if param_name == 'paint_type':
            try:
                paints = {
                    state.effects[index].paint for state in before
                }
            except (AttributeError, IndexError):
                self.cancel_preview()
                self._sync_effect_ui()
                return False
            value = (value, len(paints) > 1)
        try:
            after = [
                self._with_value(state, index, param_name, value)
                for state in before
            ]
        except (AttributeError, IndexError, TypeError, ValueError):
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
        try:
            after = [
                self._with_value(
                    state,
                    index,
                    param_name,
                    self._value_at(state, index, param_name)
                    + canonical_delta,
                )
                for state in before
            ]
        except (AttributeError, IndexError, TypeError, ValueError):
            self.cancel_preview()
            return False
        self.preview_before = None
        self.preview_key = None
        return self._commit_complete_states(before, after)

    def _prepare_structure_change(self) -> None:
        if self.controls is not None:
            self.controls.finish_pending_effect_edits()
            self.controls.cancel_effect_previews()
        self.cancel_preview()

    @staticmethod
    def _insertion_index(
        state: TextEffectStack, effect: TextEffect
    ) -> int:
        phase = effect_phase(effect)
        phase_order = {
            'exterior': 0,
            'stroke': 1,
            'foreground': 2,
            'interior': 3,
        }
        rank = phase_order[phase]
        insertion = len(state.effects)
        for index, current in enumerate(state.effects):
            current_rank = phase_order[effect_phase(current)]
            if current_rank > rank:
                insertion = index
                break
            if current_rank == rank:
                insertion = index + 1
        return insertion

    def add_effect(self, effect_type: str) -> bool:
        self._prepare_structure_change()
        before = self._current_states()
        if not self._has_common_stack_shape(before):
            self._sync_effect_ui()
            return False
        constructors = {
            'stroke': StrokeEffect,
            'shadow': ShadowEffect,
            'hollow': HollowEffect,
            'gradient_overlay': GradientOverlayEffect,
        }
        constructor = constructors.get(effect_type)
        unique_type = {
            'hollow': HollowEffect,
            'gradient_overlay': GradientOverlayEffect,
        }.get(effect_type)
        if constructor is None or (
            unique_type is not None
            and any(
                any(
                    isinstance(effect, unique_type)
                    for effect in state.effects
                )
                for state in before
            )
        ):
            self._sync_effect_ui()
            return False
        after = []
        for state in before:
            effects = list(state.effects)
            effect = constructor()
            effects.insert(self._insertion_index(state, effect), effect)
            after.append(replace(state, effects=tuple(effects)))
        return self._commit_complete_states(before, after)

    def remove_effect(self, index: int) -> bool:
        self._prepare_structure_change()
        before = self._current_states()
        if (
            not self._has_common_stack_shape(before)
            or index < 0
            or any(
                index >= len(state.effects)
                for state in before
            )
        ):
            self._sync_effect_ui()
            return False
        after = []
        for state in before:
            effects = list(state.effects)
            del effects[index]
            after.append(replace(state, effects=tuple(effects)))
        return self._commit_complete_states(before, after)

    def move_effect(self, index: int, direction: int) -> bool:
        self._prepare_structure_change()
        before = self._current_states()
        if (
            not self._has_common_stack_shape(before)
            or direction not in (-1, 1)
            or index < 0
            or any(index >= len(state.effects) for state in before)
        ):
            self._sync_effect_ui()
            return False
        phase_sequences = [
            tuple(effect_phase(effect) for effect in state.effects)
            for state in before
        ]
        if any(
            sequence != phase_sequences[0]
            for sequence in phase_sequences[1:]
        ):
            self._sync_effect_ui()
            return False
        phase = phase_sequences[0][index]
        if phase == 'foreground':
            self._sync_effect_ui()
            return False
        phase_indices = [
            effect_index
            for effect_index, current_phase in enumerate(phase_sequences[0])
            if current_phase == phase
        ]
        try:
            position = phase_indices.index(index)
            destination = phase_indices[position + direction]
        except (IndexError, ValueError):
            self._sync_effect_ui()
            return False
        after = []
        for state in before:
            effects = list(state.effects)
            effects[index], effects[destination] = (
                effects[destination], effects[index]
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

    def _refresh_owner(self) -> None:
        """Compatibility callback used by package-3 focused tests."""
        self._sync_effect_ui()

    def finish_pending_edits(self) -> None:
        if self.controls is not None:
            self.controls.finish_pending_effect_edits()

    def resolve_for_save(self) -> None:
        self.finish_pending_edits()
        if self.controls is not None:
            self.controls.cancel_effect_previews()
        self.cancel_preview()

    def resolve_for_history_change(self) -> None:
        if self.controls is not None:
            self.controls.cancel_pending_effect_edits()
            self.controls.cancel_effect_previews()
        self.cancel_preview()

    def resolve_for_page_change(self) -> None:
        self.resolve_for_save()
        self.items = []

    def cancel_for_scene_change(self) -> None:
        if self.controls is not None:
            self.controls.cancel_pending_effect_edits()
            self.controls.cancel_effect_previews()
        self.cancel_preview()
        self.items = []

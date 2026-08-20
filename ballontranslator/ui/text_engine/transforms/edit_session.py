"""Selection-scoped edit sessions for composable text transforms."""

from dataclasses import replace
from typing import TYPE_CHECKING

from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import (
    GridTextTransform,
    ProjectiveTextTransform,
    TEXT_TRANSFORM_PRECISION,
    TextTransformStack,
    create_text_transform,
)

from ... import shared_widget as SW
from ..editing.commands import SetTextTransformCommand

if TYPE_CHECKING:
    from ..formatting.panel import FontFormatPanel
    from ..item import TextBlkItem
    from .panel import TextTransformPanel


GLYPH_SLANT_INDEX = -1


def _canonical_grid_points(points) -> tuple:
    return tuple(
        tuple(
            round(float(value), TEXT_TRANSFORM_PRECISION) or 0.0
            for value in point
        )
        for point in points
    )


class TextTransformEditSession:
    """Own transform targets, previews, and undo-command boundaries.

    Structure and parameter edits use one complete immutable state per item,
    so every selected-item operation enters the canvas undo stack atomically.

    >>> session = object.__new__(TextTransformEditSession)
    >>> session.items = []
    >>> session.items
    []
    """

    def __init__(
        self,
        host: "FontFormatPanel",
        controls: "TextTransformPanel",
    ) -> None:
        self.host = host
        self.controls = controls
        self.items = []
        self.drag_before = None
        self.drag_key = None
        self.grid_before = None
        self.grid_index = None
        self.projective_before = None
        self.projective_index = None
        self.selected_index = None

        controls.transform_commit_requested.connect(self.commit_value)
        controls.transform_preview_requested.connect(
            self.preview_parameter_delta
        )
        controls.transform_drag_commit_requested.connect(
            self.commit_parameter_delta
        )
        controls.transform_preview_canceled.connect(self.cancel_preview)
        controls.transform_add_requested.connect(self.add_transform)
        controls.transform_remove_requested.connect(self.remove_transform)
        controls.transform_move_requested.connect(self.move_transform)
        controls.transform_selected.connect(self.select_transform)

    @staticmethod
    def _state_for_item(item) -> TextTransformStack:
        return item.blk.fontformat.text_transform

    @staticmethod
    def _with_value(
        state: TextTransformStack,
        index: int,
        param_name: str,
        value,
    ) -> TextTransformStack:
        if index == GLYPH_SLANT_INDEX:
            if param_name != 'glyph_slant_angle':
                raise ValueError(f'unknown glyph transform field {param_name}')
            return replace(state, glyph_slant_angle=value)
        if index < 0 or index >= len(state):
            raise IndexError('text transform index is no longer current')
        transforms = list(state)
        transforms[index] = transforms[index].with_value(param_name, value)
        if isinstance(transforms[index], GridTextTransform):
            transforms[index] = transforms[index].with_control_points(
                _canonical_grid_points(transforms[index].control_points)
            )
        return replace(state, transforms=tuple(transforms))

    @staticmethod
    def _value_at(
        state: TextTransformStack,
        index: int,
        param_name: str,
    ):
        if index == GLYPH_SLANT_INDEX:
            if param_name != 'glyph_slant_angle':
                raise ValueError(f'unknown glyph transform field {param_name}')
            return state.glyph_slant_angle
        if index < 0 or index >= len(state):
            raise IndexError('text transform index is no longer current')
        return getattr(state[index], param_name)

    def _current_states(self):
        if self.items:
            return [self._state_for_item(item) for item in self.items]
        return [self.host.global_format.text_transform]

    @staticmethod
    def _has_common_stack_shape(states) -> bool:
        sequences = [
            tuple(transform.transform_type for transform in state)
            for state in states
        ]
        return not sequences or all(
            sequence == sequences[0] for sequence in sequences
        )

    def _sync_transform_controller(self) -> None:
        canvas = getattr(SW, 'canvas', None)
        if canvas is None or not hasattr(canvas, 'clear_text_transform_controls'):
            return
        selected = getattr(self, 'selected_index', None)
        # Canvas overlays bind callbacks to one concrete item and stack index,
        # so they are valid only while that ownership remains unambiguous.
        if selected is not None and len(self.items) == 1:
            stack = self._state_for_item(self.items[0])
            if (
                0 <= selected < len(stack)
                and isinstance(stack[selected], GridTextTransform)
            ):
                canvas.bind_text_grid_control(
                    self.items[0],
                    selected,
                    begin_edit=self.begin_grid_edit,
                    preview_points=self.preview_grid_points,
                    commit_points=self.commit_grid_points,
                    cancel_edit=self.cancel_grid_edit,
                )
                return
            if (
                0 <= selected < len(stack)
                and isinstance(stack[selected], ProjectiveTextTransform)
            ):
                canvas.bind_text_projective_control(
                    self.items[0],
                    selected,
                    begin_edit=self.begin_projective_edit,
                    preview_transform=self.preview_projective_transform,
                    commit_transform=self.commit_projective_transform,
                    cancel_edit=self.cancel_projective_edit,
                )
                return
        # Deselection, multi-selection, or a stage without an overlay must
        # release any controller that was bound by an earlier selection.
        canvas.clear_text_transform_controls()

    def select_transform(self, index: int) -> None:
        index = int(index)
        selected = index if index >= 0 else None
        if getattr(self, 'selected_index', None) == selected:
            self._sync_transform_controller()
            return
        self.cancel_grid_edit(getattr(self, 'grid_index', -1))
        self.cancel_projective_edit(getattr(self, 'projective_index', -1))
        self.selected_index = selected
        if selected is None:
            self.controls.clear_transform_selection(emit=False)
        else:
            self.controls.select_transform(selected, emit=False)
        self._sync_transform_controller()

    def _set_global_state(self, state: TextTransformStack) -> None:
        self.host.global_format.text_transform = state
        self.host.update_text_style_label()

    def _commit_states(self, before, after) -> None:
        if not self.items:
            if before[0] != after[0]:
                self._set_global_state(after[0])
            self._sync_transform_ui()
            return
        command = SetTextTransformCommand.create(
            self.items,
            before,
            after,
            self._sync_transform_ui,
        )
        if command is not None:
            SW.canvas.push_undo_command(command)
        else:
            self._sync_transform_ui()

    def _sync_transform_ui(self) -> None:
        if self.items:
            self.controls.set_transform_items(self.items)
            if len(self.items) == 1 and C.active_format is not None:
                state = self._state_for_item(self.items[0])
                C.active_format.text_transform = state
        else:
            active_format = (
                self.host.global_format
                if self.host.global_mode()
                else C.active_format
            )
            if active_format is None:
                return
            self.controls.set_active_format(active_format)
        self._sync_transform_controller()

    def replace_targets(self, items) -> None:
        items = list(items)
        targets_changed = len(items) != len(self.items) or any(
            current is not replacement
            for current, replacement in zip(self.items, items)
        )
        if targets_changed:
            self.cancel_control_previews()
        else:
            # A focus-only refresh keeps the physical press alive but restores
            # model-owned state before controls are refreshed.
            self.cancel_preview()
        self.items = items
        self._sync_transform_controller()

    def commit_value(self, index: int, param_name: str, value) -> None:
        before = self._current_states()
        if (
            index != GLYPH_SLANT_INDEX
            and not self._has_common_stack_shape(before)
        ):
            self._sync_transform_ui()
            return
        try:
            after = [
                self._with_value(state, index, param_name, value)
                for state in before
            ]
        except (AttributeError, IndexError, ValueError):
            self._sync_transform_ui()
            return
        self._commit_states(before, after)

    def _prepare_structure_change(self) -> None:
        # A typed value owns an earlier transaction and must land before the
        # operation list changes its indices.
        self.controls.finish_pending_transform_edits()
        self.cancel_control_previews()

    def add_transform(self, transform_type: str) -> None:
        self._prepare_structure_change()
        try:
            transform = create_text_transform(transform_type)
        except ValueError:
            self._sync_transform_ui()
            return
        before = self._current_states()
        after = [
            replace(state, transforms=(*state.transforms, transform))
            for state in before
        ]
        new_index = (
            len(after[0]) - 1
            if self._has_common_stack_shape(after)
            else None
        )
        self.selected_index = new_index
        self._commit_states(before, after)
        if new_index is None:
            self.controls.clear_transform_selection(emit=False)
        else:
            self.controls.select_transform(new_index, emit=False)

    def activate_last_projective(self, item: "TextBlkItem") -> None:
        if len(self.items) != 1 or self.items[0] is not item:
            return
        state = self._state_for_item(item)
        for index in range(len(state) - 1, -1, -1):
            if isinstance(state[index], ProjectiveTextTransform):
                self.select_transform(index)
                return
        self.add_transform('projective')

    def remove_transform(self, index: int) -> None:
        self._prepare_structure_change()
        before = self._current_states()
        if (
            not self._has_common_stack_shape(before)
            or index < 0
            or any(index >= len(state) for state in before)
        ):
            self._sync_transform_ui()
            return
        selected = getattr(self, 'selected_index', None)
        if selected == index:
            self.selected_index = None
            self.controls.clear_transform_selection(emit=False)
        elif selected is not None and selected > index:
            self.selected_index = selected - 1
        after = []
        for state in before:
            transforms = list(state)
            del transforms[index]
            after.append(replace(state, transforms=tuple(transforms)))
        self._commit_states(before, after)

    def move_transform(self, index: int, direction: int) -> None:
        self._prepare_structure_change()
        before = self._current_states()
        destination = index + direction
        if (
            not self._has_common_stack_shape(before)
            or direction not in (-1, 1)
            or index < 0
            or any(
                index >= len(state)
                or destination < 0
                or destination >= len(state)
                for state in before
            )
        ):
            self._sync_transform_ui()
            return
        selected = getattr(self, 'selected_index', None)
        if selected == index:
            self.selected_index = destination
        elif selected == destination:
            self.selected_index = index
        after = []
        for state in before:
            transforms = list(state)
            transforms[index], transforms[destination] = (
                transforms[destination],
                transforms[index],
            )
            after.append(replace(state, transforms=tuple(transforms)))
        self._commit_states(before, after)

    @staticmethod
    def _with_grid_points(state, index, points):
        if index < 0 or index >= len(state):
            raise IndexError('grid transform index is no longer current')
        transform = state[index]
        if not isinstance(transform, GridTextTransform):
            raise ValueError('selected transform is not a Grid transform')
        transforms = list(state)
        transforms[index] = transform.with_control_points(
            _canonical_grid_points(points)
        )
        return replace(state, transforms=tuple(transforms))

    def begin_grid_edit(self, index: int) -> None:
        self.controls.finish_pending_transform_edits()
        self.cancel_control_previews()
        before = self._current_states()
        if (
            len(before) != 1
            or index < 0
            or index >= len(before[0])
            or not isinstance(before[0][index], GridTextTransform)
        ):
            return
        self.grid_before = before
        self.grid_index = index

    def preview_grid_points(self, index: int, points) -> None:
        if (
            getattr(self, 'grid_before', None) is None
            or getattr(self, 'grid_index', None) != index
            or len(self.items) != 1
        ):
            return
        try:
            state = self._with_grid_points(
                self.grid_before[0], index, points
            )
        except (IndexError, TypeError, ValueError):
            self.cancel_grid_edit(index)
            return
        self.items[0].set_text_transform(state, preview=True)

    def commit_grid_points(self, index: int, points) -> None:
        before = getattr(self, 'grid_before', None)
        if before is None or getattr(self, 'grid_index', None) != index:
            return
        try:
            after = [self._with_grid_points(before[0], index, points)]
        except (IndexError, TypeError, ValueError):
            self.cancel_grid_edit(index)
            return
        self.grid_before = None
        self.grid_index = None
        self._commit_states(before, after)

    def cancel_grid_edit(self, _index=-1) -> None:
        if getattr(self, 'grid_before', None) is None:
            return
        for item in self.items:
            item.clear_text_transform_preview()
        self.grid_before = None
        self.grid_index = None

    @staticmethod
    def _with_projective_transform(state, index, transform):
        if index < 0 or index >= len(state):
            raise IndexError('projective transform index is no longer current')
        if not isinstance(state[index], ProjectiveTextTransform):
            raise ValueError('selected transform is not a Projective transform')
        transforms = list(state)
        transforms[index] = transform
        return replace(state, transforms=tuple(transforms))

    def begin_projective_edit(self, index: int) -> None:
        self.controls.finish_pending_transform_edits()
        self.cancel_control_previews()
        before = self._current_states()
        if (
            len(before) != 1
            or index < 0
            or index >= len(before[0])
            or not isinstance(before[0][index], ProjectiveTextTransform)
        ):
            return
        self.projective_before = before
        self.projective_index = index

    def preview_projective_transform(self, index: int, transform) -> None:
        before = getattr(self, 'projective_before', None)
        if (
            before is None
            or getattr(self, 'projective_index', None) != index
            or len(self.items) != 1
        ):
            return
        try:
            state = self._with_projective_transform(
                before[0], index, transform
            )
        except (IndexError, TypeError, ValueError):
            self.cancel_projective_edit(index)
            return
        self.items[0].set_text_transform(state, preview=True)

    def commit_projective_transform(self, index: int, transform) -> None:
        before = getattr(self, 'projective_before', None)
        if before is None or getattr(self, 'projective_index', None) != index:
            return
        try:
            after = [self._with_projective_transform(before[0], index, transform)]
        except (IndexError, TypeError, ValueError):
            self.cancel_projective_edit(index)
            return
        self.projective_before = None
        self.projective_index = None
        self._commit_states(before, after)

    def cancel_projective_edit(self, _index=-1) -> None:
        if getattr(self, 'projective_before', None) is None:
            return
        for item in self.items:
            item.clear_text_transform_preview()
        self.projective_before = None
        self.projective_index = None

    def preview_parameter_delta(
        self,
        index: int,
        param_name: str,
        canonical_delta: float,
    ) -> None:
        key = (index, param_name)
        current = self._current_states()
        if (
            index != GLYPH_SLANT_INDEX
            and not self._has_common_stack_shape(current)
        ):
            self.cancel_preview()
            self._sync_transform_ui()
            return
        changed_items = {}
        if self.drag_key != key or self.drag_before is None:
            if self.drag_before is not None:
                for item in self.items:
                    if item.clear_text_transform_preview():
                        changed_items[id(item)] = item
            self.drag_key = key
            self.drag_before = current
        if not self.items:
            return
        try:
            preview_after = [
                self._with_value(
                    state,
                    index,
                    param_name,
                    self._value_at(state, index, param_name)
                    + canonical_delta,
                )
                for state in self.drag_before
            ]
        except (AttributeError, IndexError, ValueError):
            for item in changed_items.values():
                item.update()
            self.cancel_preview()
            return
        for item, state in zip(self.items, preview_after):
            if item._effective_text_transform() == state:
                continue
            if item.set_text_transform(state, preview=True):
                changed_items[id(item)] = item
        for item in changed_items.values():
            item.update()

    def commit_parameter_delta(
        self,
        index: int,
        param_name: str,
        canonical_delta: float,
    ) -> None:
        key = (index, param_name)
        if self.drag_key != key or self.drag_before is None:
            return
        before = self.drag_before
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
        except (AttributeError, IndexError, ValueError):
            self.cancel_preview()
            return
        self.drag_before = None
        self.drag_key = None
        self._commit_states(before, after)

    def cancel_preview(self, *_key) -> None:
        if self.drag_before is not None:
            for item in self.items:
                if item.clear_text_transform_preview():
                    item.update()
        self.drag_before = None
        self.drag_key = None
        if not self.items:
            self._sync_transform_ui()

    def cancel_control_previews(self) -> None:
        self.controls.cancel_transform_previews()
        if self.drag_before is not None:
            self.cancel_preview()
        self.cancel_grid_edit(getattr(self, 'grid_index', -1))
        self.cancel_projective_edit(getattr(self, 'projective_index', -1))

    def resolve_for_save(self) -> None:
        """Commit typed values and cancel any still-held drag preview."""
        self.controls.finish_pending_transform_edits()
        self.cancel_control_previews()

    def resolve_for_history_change(self) -> None:
        """Cancel a live preview before application undo or redo."""
        self.cancel_control_previews()

    def resolve_for_page_change(self) -> None:
        """End transform ownership before the old page is saved and removed."""
        self.resolve_for_save()
        self.detach_scene_owner()

    def cancel_for_scene_change(self) -> None:
        """Discard transient control state after an external model replacement."""
        self.controls.cancel_pending_transform_edits()
        self.cancel_control_previews()
        self.detach_scene_owner()

    def finish_pending_edits(self) -> None:
        self.controls.finish_pending_transform_edits()

    def detach_scene_owner(self) -> None:
        host = self.host
        if host.textblk_item is not None:
            # Keep the TextBlock and live layout attached to their canonical
            # format object when the panel returns to its global owner.
            letter_spacing = host.textblk_item.fontformat.letter_spacing
            host.textblk_item.fontformat.merge(C.active_format)
            host.textblk_item.fontformat.letter_spacing = letter_spacing
        host.textblk_item = None
        self.items = []
        self.selected_index = None
        self._sync_transform_controller()
        host.set_active_format(host.global_format)
        host.set_globalfmt_title()

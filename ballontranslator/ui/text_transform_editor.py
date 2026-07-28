"""Selection-scoped editing transactions for text-transform controls."""

import copy

from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import create_text_transform

from . import shared_widget as SW
from .textedit_commands import SetTextTransformCommand


class TextTransformEditSession:
    """Own transform targets, previews, and undo-command boundaries.

    The format panel remains responsible for choosing the active formatting
    owner; this session owns only transform-specific interaction state.

    >>> session = object.__new__(TextTransformEditSession)
    >>> session.items = []
    >>> session.has_items()
    False
    """

    def __init__(self, host, controls) -> None:
        self.host = host
        self.controls = controls
        self.items = []
        self.drag_before = None
        self.drag_param = None
        self.global_values_by_type = {}

        controls.transform_commit_requested.connect(self.commit_value)
        controls.transform_preview_requested.connect(self.preview_delta)
        controls.transform_drag_commit_requested.connect(self.commit_drag)
        controls.transform_preview_canceled.connect(self.cancel_preview)
        controls.transform_type_change_requested.connect(self.change_type)

    def has_items(self) -> bool:
        return bool(self.items)

    @staticmethod
    def _with_value(transform, param_name, value):
        return transform.with_value(param_name, value)

    def _refresh_geometry(self) -> None:
        for item in self.items:
            item.update()

    def _remember_global_transform(self, transform) -> None:
        self.global_values_by_type[transform.transform_type] = transform

    def _global_transform_for_type(self, transform_type):
        current = self.host.global_format.text_transform
        self._remember_global_transform(current)
        if current.transform_type == transform_type:
            return current
        remembered = self.global_values_by_type.get(transform_type)
        return (
            remembered
            if remembered is not None
            else create_text_transform(transform_type)
        )

    def refresh_controls(self, refresh_shape=True) -> None:
        if self.items:
            self.controls.set_transform_items(self.items)
            if len(self.items) == 1 and C.active_format is not None:
                C.active_format.text_transform = (
                    self.items[0].blk.fontformat.text_transform
                )
        else:
            active_format = (
                self.host.global_format
                if self.host.global_mode()
                else C.active_format
            )
            if active_format is None:
                return
            self.controls.set_transform(active_format.text_transform)
        if refresh_shape:
            self._refresh_geometry()

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
            # the model-owned transform before the controls are refreshed.
            self.cancel_preview(self.drag_param)
        self.items = items

    def commit_value(self, param_name: str, value: float) -> None:
        if not self.items:
            before = self.host.global_format.text_transform
            after = self._with_value(before, param_name, value)
            if before != after:
                self.host.global_format.text_transform = after
                self._remember_global_transform(after)
                self.host.update_text_style_label()
            self.refresh_controls(refresh_shape=False)
            return

        before = [item.blk.fontformat.text_transform for item in self.items]
        after = [
            self._with_value(transform, param_name, value)
            for transform in before
        ]
        command = SetTextTransformCommand.create(
            self.items,
            before,
            after,
            self.refresh_controls,
        )
        if command is not None:
            SW.canvas.push_undo_command(command)
        else:
            self.refresh_controls(refresh_shape=False)

    def change_type(self, transform_type: str) -> None:
        if not self.items:
            before = self.host.global_format.text_transform
            after = self._global_transform_for_type(transform_type)
            if before != after:
                self.host.global_format.text_transform = after
                self._remember_global_transform(after)
                self.host.update_text_style_label()
            self.refresh_controls(refresh_shape=False)
            return

        before = [item.blk.fontformat.text_transform for item in self.items]
        after = [
            item.geometry_controller.transform_for_type(transform_type)
            for item in self.items
        ]
        command = SetTextTransformCommand.create(
            self.items,
            before,
            after,
            self.refresh_controls,
        )
        if command is not None:
            SW.canvas.push_undo_command(command)
        else:
            self.refresh_controls(refresh_shape=False)

    def preview_delta(self, param_name: str, canonical_delta: float) -> None:
        if not self.items:
            if self.drag_param != param_name or self.drag_before is None:
                self.drag_param = param_name
                self.drag_before = [self.host.global_format.text_transform]
            return
        if self.drag_param != param_name or self.drag_before is None:
            # The emitting control owns its cumulative delta until release.
            if self.drag_before is not None:
                for item in self.items:
                    item.clear_text_transform_preview()
            self.drag_param = param_name
            self.drag_before = [
                item.blk.fontformat.text_transform for item in self.items
            ]
        preview_after = [
            self._with_value(
                transform,
                param_name,
                getattr(transform, param_name) + canonical_delta,
            )
            for transform in self.drag_before
        ]
        geometry_changed = False
        for item, transform in zip(self.items, preview_after):
            # Clamped drag deltas can repeatedly produce the current endpoint.
            if item._effective_text_transform() == transform:
                continue
            geometry_changed = (
                item.set_text_transform(transform, preview=True)
                or geometry_changed
            )
        if geometry_changed:
            self._refresh_geometry()

    def commit_drag(self, param_name: str, canonical_delta: float) -> None:
        if self.drag_param != param_name or self.drag_before is None:
            return
        before = self.drag_before
        after = [
            self._with_value(
                transform,
                param_name,
                getattr(transform, param_name) + canonical_delta,
            )
            for transform in before
        ]
        items = list(self.items)
        self.drag_before = None
        self.drag_param = None
        if not items:
            global_before = before[0]
            global_after = after[0]
            if global_before != global_after:
                self.host.global_format.text_transform = global_after
                self._remember_global_transform(global_after)
                self.host.update_text_style_label()
            self.refresh_controls(refresh_shape=False)
            return
        command = SetTextTransformCommand.create(
            items,
            before,
            after,
            self.refresh_controls,
        )
        if command is None:
            geometry_changed = False
            for item in items:
                geometry_changed = (
                    item.clear_text_transform_preview() or geometry_changed
                )
            if geometry_changed:
                self._refresh_geometry()
        else:
            SW.canvas.push_undo_command(command)

    def cancel_preview(self, _param_name=None) -> None:
        geometry_changed = False
        if self.drag_before is not None:
            for item in self.items:
                geometry_changed = (
                    item.clear_text_transform_preview() or geometry_changed
                )
        self.drag_before = None
        self.drag_param = None
        if not self.items:
            self.refresh_controls(refresh_shape=False)
        elif geometry_changed:
            self._refresh_geometry()

    def cancel_control_previews(self) -> None:
        for control in self.controls.transform_controls.values():
            control.cancel_preview()
        if self.drag_before is not None:
            self.cancel_preview(self.drag_param)

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
        for control in self.controls.transform_controls.values():
            control.cancel_pending()
            control.cancel_preview()
        self.cancel_preview(self.drag_param)
        self.detach_scene_owner()

    def finish_pending_edits(self) -> None:
        self.controls.finish_pending_transform_edits()

    def detach_scene_owner(self) -> None:
        host = self.host
        if host.textblk_item is not None:
            host.textblk_item.fontformat = copy.deepcopy(C.active_format)
        host.textblk_item = None
        self.items = []
        host.set_active_format(host.global_format)
        host.set_globalfmt_title()

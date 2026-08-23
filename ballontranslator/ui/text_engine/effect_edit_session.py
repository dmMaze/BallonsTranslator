"""Selection-scoped preview and undo boundaries for text effects."""

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

from ballontranslator.utils import config as C
from ballontranslator.utils.text_effects import TextEffectStack

from .. import shared_widget as SW
from .editing.commands import SetTextEffectStackCommand

if TYPE_CHECKING:
    from .formatting.panel import FontFormatPanel
    from .item import TextBlkItem


class TextEffectEditSession:
    """Own one selection's complete-stack preview and commit transaction.

    >>> session = object.__new__(TextEffectEditSession)
    >>> session.items = []
    >>> session.items
    []
    """

    def __init__(self, host: "FontFormatPanel") -> None:
        self.host = host
        self.items = []
        self.preview_before = None

    @staticmethod
    def _state_for_item(item: "TextBlkItem") -> TextEffectStack:
        return item.blk.fontformat.text_effects

    def _current_states(self) -> Tuple[TextEffectStack, ...]:
        return tuple(self._state_for_item(item) for item in self.items)

    def _validate_states(
        self, states: Sequence[TextEffectStack]
    ) -> Tuple[TextEffectStack, ...]:
        values = tuple(states)
        if len(values) != len(self.items):
            raise ValueError('items and effect states must have the same length')
        if any(not isinstance(value, TextEffectStack) for value in values):
            raise TypeError('effect edit session requires TextEffectStack values')
        return values

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
        targets = self._validate_states(states)
        if not self.items:
            return False
        if self.preview_before is None:
            self.preview_before = self._current_states()
        changed = False
        for item, state in zip(self.items, targets):
            changed = item.set_text_effects(state, preview=True) or changed
        return changed

    def commit_states(
        self, states: Optional[Sequence[TextEffectStack]] = None
    ) -> bool:
        if not self.items:
            self.preview_before = None
            return False
        before = (
            self._current_states()
            if self.preview_before is None
            else self.preview_before
        )
        after = (
            tuple(item.effective_text_effects() for item in self.items)
            if states is None
            else self._validate_states(states)
        )
        self.preview_before = None
        command = SetTextEffectStackCommand.create(
            self.items, before, after, self._refresh_owner
        )
        if command is None:
            self.cancel_preview()
            return False
        SW.canvas.push_undo_command(command)
        return True

    def _refresh_owner(self) -> None:
        if len(self.items) == 1:
            item = self.items[0]
            current_item = getattr(self.host, 'textblk_item', None)
            if current_item is item and C.active_format is not None:
                C.active_format.text_effects = self._state_for_item(item)
        if hasattr(self.host, 'update_text_style_label'):
            self.host.update_text_style_label()

    def cancel_preview(self) -> bool:
        changed = False
        for item in self.items:
            changed = item.clear_text_effect_preview() or changed
        self.preview_before = None
        return changed

    def resolve_for_save(self) -> None:
        self.cancel_preview()

    def resolve_for_history_change(self) -> None:
        self.cancel_preview()

    def resolve_for_page_change(self) -> None:
        self.cancel_preview()
        self.items = []

    def cancel_for_scene_change(self) -> None:
        self.cancel_preview()
        self.items = []

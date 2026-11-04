"""Live preview for the letter and line spacing drag labels.

Dragging a spacing label used to change nothing until the mouse was released,
because a size box only emits its parameter while it holds focus and the drag
puts focus on the label. These sessions apply every step to the items instead,
then collapse the whole drag into one undo entry on release.
"""

from typing import List

from qtpy.QtCore import QSignalBlocker

from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import LineSpacingType

from ... import shared_widget as SW


#: Value change per pixel dragged, by parameter and spacing type.
PROPORTIONAL_STEP = 0.01
DISTANCE_STEP = 0.1


class SpacingDragSession:
    """Preview one spacing parameter while its label is dragged.

    The first movement opens the session, reading the value that is really
    in effect so the drag continues from the current spacing instead of
    from whatever the box happens to display. Every later movement applies
    the running value to the items with their change signals muted, which
    keeps the previews out of the undo stack. On release the previews are
    rewound and the final value is applied once, leaving a single entry.
    """

    def __init__(self, panel, param_name: str, box) -> None:
        self.panel = panel
        self.param_name = param_name
        self.box = box
        self.items: List = []
        self.start_steps: List[int] = []
        self.value = 0.0
        self.active = False

    def _multiplier(self) -> float:
        if (
            self.param_name == 'line_spacing'
            and C.active_format.line_spacing_type == LineSpacingType.Distance
        ):
            return DISTANCE_STEP
        return PROPORTIONAL_STEP

    def _targets(self) -> List:
        item = getattr(self.panel, 'textblk_item', None)
        if item is not None:
            return [item]
        if SW.canvas is None:
            return []
        return list(SW.canvas.selected_text_items())

    def begin(self) -> None:
        self.items = self._targets()
        if self.items:
            # The effective value, which for a selection is its own spacing
            # rather than the item default.
            source = self.items[0].get_fontformat()
        else:
            source = self.panel.global_format
        self.value = float(getattr(source, self.param_name))
        self.start_steps = [
            item.document().availableUndoSteps() for item in self.items
        ]
        self.active = True

    def preview(self, delta: int) -> None:
        if not self.active:
            self.begin()
        if not delta:
            return
        self.value = min(
            self.box.max_val,
            max(self.box.min_val, self.value + delta * self._multiplier()),
        )
        with QSignalBlocker(self.box):
            self.box.setValue(self.value)
        self._apply(muted=True)

    def commit(self) -> None:
        if not self.active:
            return
        self.active = False
        self._rewind()
        self._apply(muted=False)
        self.items = []
        self.start_steps = []

    def _apply(self, muted: bool) -> None:
        """Push the running value through the panel's normal parameter path."""
        if not muted:
            self.panel.on_param_changed(self.param_name, self.value)
            return
        blocked = [item.block_change_signal for item in self.items]
        for item in self.items:
            item.block_change_signal = True
        try:
            self.panel.on_param_changed(self.param_name, self.value)
        finally:
            for item, was_blocked in zip(self.items, blocked):
                item.block_change_signal = was_blocked

    def _rewind(self) -> None:
        """Undo the muted previews so the commit is one undo step."""
        for item, steps in zip(self.items, self.start_steps):
            document = item.document()
            was_replaying = item.in_redo_undo
            was_blocked = item.block_change_signal
            item.in_redo_undo = True
            item.block_change_signal = True
            try:
                while document.availableUndoSteps() > steps:
                    document.undo()
            finally:
                item.in_redo_undo = was_replaying
                item.block_change_signal = was_blocked
            item.old_undo_steps = document.availableUndoSteps()


def bind_spacing_drag(panel, label, param_name: str, box) -> SpacingDragSession:
    """Connect one drag label to a preview session."""
    session = SpacingDragSession(panel, param_name, box)
    label.size_ctrl_changed.connect(session.preview)
    label.btn_released.connect(session.commit)
    return session

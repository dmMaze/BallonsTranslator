from typing import List, Callable, Dict
import copy

from qtpy.QtCore import QPointF
from qtpy.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
)
try:
    from qtpy.QtWidgets import QUndoCommand
except ImportError:
    from qtpy.QtGui import QUndoCommand

from ... import shared_widget as SW
from ballontranslator.utils.fontformat import FontFormat, FontWeight, px2pt
from ballontranslator.utils.io_utils import empty_func
from ..item import TextBlkItem

global_default_set_kwargs = dict(set_selected=False, restore_cursor=False)
local_default_set_kwargs = dict(set_selected=True, restore_cursor=True)


def restore_canvas_view_focus() -> None:
    focus_widget = QApplication.focusWidget()
    if isinstance(
        focus_widget,
        (
            QAbstractSlider,
            QAbstractSpinBox,
            QComboBox,
            QLineEdit,
            QPlainTextEdit,
            QTextEdit,
        ),
    ):
        # Keep keyboard editors active while their live value updates repaint
        # the selected text. Click-only controls still return focus below.
        return
    # Scene focus keeps the text caret alive; the view owns keyboard input.
    if not SW.canvas.gv.hasFocus():
        SW.canvas.gv.setFocus()


class TextStyleUndoCommand(QUndoCommand):

    def __init__(self, style_func: Callable, params: Dict, redo_values: List, undo_values: List):
        super().__init__()
        self.style_func = style_func
        self.params = params
        self.redo_values = redo_values
        self.undo_values = undo_values

    def redo(self) -> None:
        self.style_func(values=self.redo_values, **self.params)

    def undo(self) -> None:
        self.style_func(values=self.undo_values, **self.params)


def _is_live_item(item) -> bool:
    """Return False for a missing or already deleted scene item.

    The format panel can hold a binding to an item that a page reload has
    already destroyed; passing it on raises from inside the Qt wrapper.

    >>> _is_live_item(None)
    False
    """
    if item is None:
        return False
    try:
        item.scene()
    except RuntimeError:
        return False
    return True


def wrap_fntformat_input(values: str, blkitems: List[TextBlkItem], is_global: bool):
    if is_global:
        blkitems = SW.canvas.selected_text_items()
    else:
        if not isinstance(blkitems, List):
            blkitems = [blkitems]
    blkitems = [item for item in blkitems if _is_live_item(item)]
    values = [values] * len(blkitems)
    return blkitems, values

def font_formating(push_undostack: bool = False, is_property = True):

    def func_wrapper(formatting_func):

        def wrapper(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem] = None, set_focus: bool = False, *args, **kwargs):
            if is_global and is_property:
                if hasattr(act_ffmt, param_name):
                    act_ffmt[param_name] = values
                else:
                    print(f'undefined param name: {param_name}')

            blkitems, values = wrap_fntformat_input(values, blkitems, is_global)
            if len(blkitems) > 0:
                if is_property:
                    act_ffmt[param_name] = values[0]
                if push_undostack:
                    params = copy.deepcopy(kwargs)
                    params.update({'param_name': param_name, 'act_ffmt': act_ffmt, 'is_global': is_global, 'blkitems': blkitems})
                    undo_values = [getattr(blkitem.fontformat, param_name) for blkitem in blkitems]
                    cmd = TextStyleUndoCommand(formatting_func, params, values, undo_values)
                    SW.canvas.push_undo_command(cmd)
                else:
                    formatting_func(param_name, values, act_ffmt, is_global, blkitems, *args, **kwargs)
            if set_focus:
                restore_canvas_view_focus()
        return wrapper
    
    return func_wrapper

@font_formating()
def ffmt_change_font_family(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontFamily(value, **set_kwargs)

@font_formating()
def ffmt_change_italic(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontItalic(value, **set_kwargs)

@font_formating()
def ffmt_change_underline(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontUnderline(value, **set_kwargs)

@font_formating()
def ffmt_change_font_weight(
    param_name: str,
    values: FontWeight,
    act_ffmt: FontFormat,
    is_global: bool,
    blkitems: List[TextBlkItem],
    **kwargs,
) -> None:
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontWeight(value, **set_kwargs)


def ffmt_change_font_family_and_weight(
    font_family: str,
    font_weight: FontWeight,
    act_ffmt: FontFormat,
    is_global: bool,
    blkitems: List[TextBlkItem] = None,
    set_focus: bool = False,
) -> None:
    """Apply the face family and its weight as one formatting action.

    >>> callable(ffmt_change_font_family_and_weight)
    True
    """
    act_ffmt.font_family = font_family
    act_ffmt.font_weight = font_weight
    if is_global:
        targets = SW.canvas.selected_text_items()
        set_kwargs = global_default_set_kwargs
    else:
        targets = blkitems if isinstance(blkitems, list) else [blkitems]
        set_kwargs = local_default_set_kwargs
    for blkitem in targets:
        if blkitem is not None:
            blkitem.setFontFamilyAndWeight(
                font_family,
                font_weight,
                **set_kwargs,
            )
    if set_focus:
        restore_canvas_view_focus()


@font_formating()
def ffmt_change_letter_spacing(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.setLetterSpacing(value)

@font_formating()
def ffmt_change_line_spacing(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.setLineSpacing(value)

@font_formating(push_undostack=True)
def ffmt_change_vertical(param_name: str, values: bool, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    # set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setVertical(value)

@font_formating(push_undostack=True)
def ffmt_change_standard_vertical_roman_alignment(
    param_name: str,
    values: bool,
    act_ffmt: FontFormat,
    is_global: bool,
    blkitems: List[TextBlkItem],
    **kwargs,
) -> None:
    for blkitem, value in zip(blkitems, values):
        blkitem.setStandardVerticalRomanAlignment(value)

@font_formating()
def ffmt_change_frgb(param_name: str, values: tuple, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontColor(value, **set_kwargs)

@font_formating()
def ffmt_change_font_size(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], clip_size=False, **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        if value < 0:
            continue
        value = px2pt(value)
        blkitem.setFontSize(value, clip_size=clip_size, **set_kwargs)

@font_formating(is_property=False)
def ffmt_change_rel_font_size(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], clip_size=False, **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setRelFontSize(value, clip_size=clip_size, **set_kwargs)

def _apply_scale_snapshot(param_name: str, values, act_ffmt: FontFormat,
                          is_global: bool, blkitems: List[TextBlkItem],
                          **kwargs):
    """Put a captured (size, frame, centre) snapshot back on each block."""
    for blkitem, snapshot in zip(blkitems, values):
        try:
            blkitem.fontformat.font_size = snapshot['size']
            blkitem.setFontSize(snapshot['pt'])
            if snapshot.get('w'):
                blkitem.set_size(
                    snapshot['w'], snapshot['h'], set_layout_maxsize=True
                )
                blkitem.layout.reLayoutEverything()
                centre = QPointF(snapshot['cx'], snapshot['cy'])
                current = blkitem.mapToScene(blkitem.boundingRect().center())
                blkitem.setPos(blkitem.pos() + (centre - current))
        except (RuntimeError, AttributeError):
            continue


def push_scale_snapshot(canvas, blkitems: List[TextBlkItem],
                        before: List[dict], after: List[dict]) -> None:
    """Record one undoable step for a finished scale gesture.

    The drag previews every intermediate size without touching the undo stack;
    on release this pushes a single command, which is what marks the project
    unsaved (and therefore saved when the page changes).

    >>> callable(push_scale_snapshot)
    True
    """
    items = [item for item in blkitems if _is_live_item(item)]
    if not items or len(before) != len(items) or len(after) != len(items):
        return
    params = {
        'param_name': 'font_size',
        'act_ffmt': items[0].fontformat,
        'is_global': False,
        'blkitems': items,
    }
    push = getattr(canvas, 'push_text_command', None)
    if push is None:
        _apply_scale_snapshot(values=after, **params)
        return
    push(TextStyleUndoCommand(
        _apply_scale_snapshot, params, after, before
    ))


def apply_no_forced_wrap(param_name: str, values, act_ffmt: FontFormat,
                         is_global: bool, blkitems: List[TextBlkItem],
                         **kwargs):
    """Set the flag and relayout; used as the command's payload."""
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.no_forced_wrap = bool(value)
        try:
            blkitem.layout.reLayoutEverything()
        except (RuntimeError, AttributeError):
            continue


def push_no_forced_wrap(canvas, blkitems: List[TextBlkItem],
                        enabled: bool) -> None:
    """Apply the toggle and register it as an undoable text change.

    The project's unsaved state comes from the text undo stack, so a direct
    mutation looked saved and a page change skipped writing it. Pushing onto the
    stack also makes the toggle undoable.

    >>> callable(push_no_forced_wrap)
    True
    """
    items = [item for item in blkitems if _is_live_item(item)]
    if not items:
        return
    values = [bool(enabled)] * len(items)
    undo_values = [bool(item.fontformat.no_forced_wrap) for item in items]
    params = {
        'param_name': 'no_forced_wrap',
        'act_ffmt': items[0].fontformat,
        'is_global': False,
        'blkitems': items,
    }
    push = getattr(canvas, 'push_text_command', None)
    if push is None:
        # No text stack here (an embedded view or a test scene): still apply
        # the change rather than silently doing nothing.
        apply_no_forced_wrap(values=values, **params)
        return
    command = TextStyleUndoCommand(
        apply_no_forced_wrap, params, values, undo_values
    )
    push(command)


@font_formating(push_undostack=True)
def ffmt_change_alignment(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    restore_cursor = not is_global
    for blkitem, value in zip(blkitems, values):
        blkitem.setAlignment(value, restore_cursor=restore_cursor)

@font_formating(push_undostack=True)
def ffmt_change_vertical_alignment(
    param_name: str,
    values: float,
    act_ffmt: FontFormat,
    is_global: bool,
    blkitems: List[TextBlkItem],
    **kwargs,
) -> None:
    for blkitem, value in zip(blkitems, values):
        blkitem.setVerticalAlignment(value)

@font_formating(push_undostack=True)
def ffmt_change_roman_word_rotation(
    param_name: str,
    values: bool,
    act_ffmt: FontFormat,
    is_global: bool,
    blkitems: List[TextBlkItem],
    **kwargs,
) -> None:
    for blkitem, value in zip(blkitems, values):
        blkitem.setRomanWordRotation(value)

@font_formating()
def ffmt_change_line_spacing_type(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.setLineSpacingType(value)


handle_ffmt_change = {
    name: globals().get(f'ffmt_change_{name}', empty_func)
    for name in (*FontFormat.params(), 'rel_font_size')
    if name not in {
        'opacity', 'srgb', 'stroke_width',
        'shadow_radius', 'shadow_strength', 'shadow_color', 'shadow_offset',
    }
}

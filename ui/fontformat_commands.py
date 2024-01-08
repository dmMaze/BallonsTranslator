from typing import List, Callable, Dict
import copy

from qtpy.QtGui import QFont
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from . import shared_widget as SW
from utils.fontformat import FontFormat
from .textitem import TextBlkItem

global_default_set_kwargs = dict(set_selected=False, restore_cursor=False)
local_default_set_kwargs = dict(set_selected=True, restore_cursor=True)



class TextStyleUndoCommand(QUndoCommand):

    def __init__(self, style_func: Callable, redo_params: Dict, undo_params: Dict):
        super().__init__()
        self.style_func = style_func
        self.redo_params = redo_params
        self.undo_params = undo_params

    def redo(self) -> None:
        self.style_func(**self.redo_params)

    def undo(self) -> None:
        self.style_func(**self.undo_params)


def font_formating(push_undostack: bool = False):

    """
    let's hope it will make it easier to implement redo/undo behavior for these formatting op
    """

    def func_wrapper(formatting_func):

        def wrapper(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem] = None, set_focus: bool = False, *args, **kwargs):
            if not isinstance(values, List):
                values = [values]
            act_ffmt[param_name] = values[0]
            if is_global:
                blkitems = SW.canvas.selected_text_items()
            else:
                blkitems = blkitems if isinstance(blkitems, List) else [blkitems]
            if len(blkitems) > 0:
                if push_undostack:
                    redo_params = copy.deepcopy(kwargs)
                    redo_params.update({'param_name': param_name, 'values': values, 'act_ffmt': act_ffmt, 'is_global': is_global, 'blkitems': blkitems})
                    undo_params = copy.deepcopy(kwargs)
                    undo_values = [blkitem.getFontFormatAttr(param_name) for blkitem in blkitems]
                    undo_params.update({'param_name': param_name, 'values': undo_values, 'act_ffmt': act_ffmt, 'is_global': is_global, 'blkitems': blkitems})
                    cmd = TextStyleUndoCommand(formatting_func, redo_params, undo_params)
                    SW.canvas.push_undo_command(cmd)
                else:
                    formatting_func(param_name, values, act_ffmt, is_global, blkitems, *args, **kwargs)
            if set_focus:
                if not SW.canvas.hasFocus():
                    SW.canvas.setFocus()
        return wrapper
    
    return func_wrapper

@font_formating()
def ffmt_change_family(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
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
def ffmt_change_weight(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontWeight(value, **set_kwargs)

def ffmt_change_bold(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem] = None, **kwargs):
    values = [QFont.Bold if value else QFont.Normal for value in values]
    ffmt_change_weight('weight', values, act_ffmt, is_global, blkitems, **kwargs)

@font_formating(push_undostack=True)
def ffmt_change_letter_spacing(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setLetterSpacing(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_line_spacing(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setLineSpacing(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_vertical(param_name: str, values: bool, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    # set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setVertical(value)

@font_formating()
def ffmt_change_frgb(param_name: str, values: tuple, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontColor(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_srgb(param_name: str, values: tuple, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setStrokeColor(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_stroke_width(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setStrokeWidth(value, **set_kwargs)

@font_formating()
def ffmt_change_size(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        if value < 0:
            continue
        blkitem.setFontSize(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_alignment(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    restore_cursor = not is_global
    for blkitem, value in zip(blkitems, values):
        blkitem.setAlignment(value, restore_cursor=restore_cursor)
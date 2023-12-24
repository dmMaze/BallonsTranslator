from typing import List
from qtpy.QtGui import QFont

from . import shared_widget as SW
from utils.fontformat import FontFormat
from .textitem import TextBlkItem

global_default_set_kwargs = dict(set_selected=False, restore_cursor=False)
local_default_set_kwargs = dict(set_selected=True, restore_cursor=True)

def font_formating(push_undostack: bool = False):

    """
    let's hope it will make it easier to implement redo/undo behavior for these formatting op
    """

    def func_wrapper(formatting_func):

        def wrapper(param_name: str, value: str, act_ffmt: FontFormat, is_global: bool, blkitems: TextBlkItem = None, set_focus: bool = False, *args, **kwargs):
            act_ffmt[param_name] = value
            if is_global:
                blkitems = SW.canvas.selected_text_items()
            else:
                blkitems = blkitems if isinstance(blkitems, List) else [blkitems]
            if len(blkitems) > 0:
                formatting_func(param_name, value, act_ffmt, is_global, blkitems, *args, **kwargs)
            if set_focus:
                if not SW.canvas.hasFocus():
                    SW.canvas.setFocus()
        return wrapper
    
    return func_wrapper

@font_formating()
def ffmt_change_family(param_name: str, value: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setFontFamily(value, **set_kwargs)

@font_formating()
def ffmt_change_italic(param_name: str, value: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setFontItalic(value, **set_kwargs)

@font_formating()
def ffmt_change_underline(param_name: str, value: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setFontUnderline(value, **set_kwargs)

@font_formating()
def ffmt_change_weight(param_name: str, value: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setFontWeight(value, **set_kwargs)

def ffmt_change_bold(param_name: str, value: bool, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem] = None, **kwargs):
    weight = QFont.Bold if value else QFont.Normal
    ffmt_change_weight('weight', weight, act_ffmt, is_global, blkitems, **kwargs)

@font_formating()
def ffmt_change_letter_spacing(param_name: str, value: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setLetterSpacing(value, **set_kwargs)

@font_formating()
def ffmt_change_line_spacing(param_name: str, value: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setLineSpacing(value, **set_kwargs)

@font_formating()
def ffmt_change_vertical(param_name: str, value: bool, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    # set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setVertical(value)

@font_formating()
def ffmt_change_frgb(param_name: str, value: tuple, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setFontColor(value, **set_kwargs)

@font_formating()
def ffmt_change_srgb(param_name: str, value: tuple, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setStrokeColor(value, **set_kwargs)

@font_formating()
def ffmt_change_stroke_width(param_name: str, value: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setStrokeWidth(value, **set_kwargs)

@font_formating()
def ffmt_change_size(param_name: str, value: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    if value <= 0:
        return
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem in blkitems:
        blkitem.setFontSize(value, **set_kwargs)

@font_formating()
def ffmt_change_alignment(param_name: str, value: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    restore_cursor = not is_global
    for blkitem in blkitems:
        blkitem.setAlignment(value, restore_cursor=restore_cursor)
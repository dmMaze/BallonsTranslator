from typing import Union
import enum
import re

from . import shared
from .structures import Tuple, Union, List, Dict, Config, field, nested_dataclass
from .textblock import TextBlock, fix_fontweight_qt


def pt2px(pt) -> float:
    return int(round(pt * shared.LDPI / 72.))

def px2pt(px) -> float:
    return px / shared.LDPI * 72.


class LineSpacingType(enum.IntEnum):
    Proportional = 0
    Distance = 1


@nested_dataclass
class FontFormat(Config):

    family: str = field(default_factory=lambda: shared.DEFAULT_FONT_FAMILY) # to always apply shared.DEFAULT_FONT_FAMILY
    size: float = 24
    stroke_width: float = 0
    frgb: List = field(default_factory=lambda: [0, 0, 0])
    srgb: List = field(default_factory=lambda: [0, 0, 0])
    bold: bool = False
    underline: bool = False
    italic: bool = False
    alignment: int = 0
    vertical: bool = False
    weight: int = None
    line_spacing: float = 1.2
    letter_spacing: float = 1.
    opacity: float = 1.
    shadow_radius: float = 0.
    shadow_strength: float = 1.
    shadow_color: List = field(default_factory=lambda: [0, 0, 0])
    shadow_offset: List = field(default_factory=lambda: [0., 0.])
    _style_name: str = ''
    line_spacing_type: int = LineSpacingType.Proportional

    def update_from_textblock(self, text_block: TextBlock):
        self.family = text_block.font_family
        self.size = px2pt(text_block.font_size)
        self.stroke_width = text_block.stroke_width
        self.frgb, self.srgb = text_block.get_font_colors()
        self.bold = text_block.bold
        self.weight = text_block.font_weight
        self.underline = text_block.underline
        self.italic = text_block.italic
        self.alignment = text_block.alignment()
        self.vertical = text_block.vertical
        self.line_spacing = text_block.line_spacing
        self.letter_spacing = text_block.letter_spacing
        self.opacity = text_block.opacity
        self.shadow_radius = text_block.shadow_radius
        self.shadow_strength = text_block.shadow_strength
        self.shadow_color = text_block.shadow_color
        self.shadow_offset = text_block.shadow_offset

    def __post_init__(self):
        self.weight = fix_fontweight_qt(self.weight)

    def update_textblock_format(self, blk: TextBlock):
        blk.default_stroke_width = self.stroke_width
        blk.line_spacing = self.line_spacing
        blk.letter_spacing = self.letter_spacing
        blk.font_family = self.family
        blk.font_size = pt2px(self.size)
        blk.font_weight = self.weight
        blk._alignment = self.alignment
        blk.shadow_color = self.shadow_color
        blk.shadow_radius = self.shadow_radius
        blk.shadow_strength = self.shadow_strength
        blk.shadow_offset = self.shadow_offset
        blk.opacity = self.opacity
        blk.vertical = self.vertical
        blk.set_font_colors(self.frgb, self.srgb)

    @staticmethod
    def from_textblock(text_block: TextBlock):
        ffmt = FontFormat()
        ffmt.update_from_textblock(text_block)
        return ffmt

    def merge(self, target: Config):
        tgt_keys = target.annotations_set()
        updated_keys = set()
        for key in tgt_keys:
            if key != '_style_name' and self[key] != target[key]:
                self.update(key, target[key])
                updated_keys.add(key)
        return updated_keys
    


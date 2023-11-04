from . import shared
from .structures import Tuple, Union, List, Dict, Config, field, nested_dataclass
from .textblock import TextBlock


def pt2px(pt) -> float:
    return int(round(pt * shared.LDPI / 72.))

def px2pt(px) -> float:
    return px / shared.LDPI * 72.


@nested_dataclass
class FontFormat(Config):

    family: str = field(default_factory=lambda: shared.DEFAULT_FONT_FAMILY) # to always apply shared.DEFAULT_FONT_FAMILY
    size: float = 24
    stroke_width: float = 0
    frgb: Tuple = (0, 0, 0)
    srgb: Tuple = (0, 0, 0)
    bold: bool = False
    underline: bool = False
    italic: bool = False
    alignment: int = 0
    vertical: bool = False
    weight: int = 50
    line_spacing: float = 1.2
    letter_spacing: float = 1.
    opacity: float = 1.
    shadow_radius: float = 0.
    shadow_strength: float = 1.
    shadow_color: Tuple = (0, 0, 0)
    shadow_offset: List = field(default_factory=lambda: [0., 0.])

    def from_textblock(self, text_block: TextBlock):
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
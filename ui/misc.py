import cv2, re, json, os
from pathlib import Path
import numpy as np
import os.path as osp
from qtpy.QtGui import QPixmap,  QColor, QImage, QTextDocument, QTextCursor

from . import constants as C
from utils.io_utils import NumpyEncoder
from utils.structures import Tuple, Union, List, Dict, Config, field, nested_dataclass
from modules.textdetector.textblock import TextBlock

# return bgr tuple
def qrgb2bgr(color: Union[QColor, Tuple, List] = None) -> Tuple[int, int, int]:
    if color is not None:
        if isinstance(color, QColor):
            color = (color.blue(), color.green(), color.red())
        else:
            assert isinstance(color, (tuple, list))
            color = (color[2], color[1], color[0])
    return color

# https://stackoverflow.com/questions/45020672/convert-pyqt5-qpixmap-to-numpy-ndarray
def pixmap2ndarray(pixmap: Union[QPixmap, QImage], keep_alpha=True):
    size = pixmap.size()
    h = size.width()
    w = size.height()
    if isinstance(pixmap, QPixmap):
        qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    else:
        qimg = pixmap

    byte_str = qimg.bits().asstring(h * w * 4)
    img = np.fromstring(byte_str, dtype=np.uint8).reshape((w,h,4))
    
    if keep_alpha:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img
    else:
        return np.copy(img[:,:,:3])

def ndarray2pixmap(img, return_qimg=False):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    height, width, channel = img.shape
    bytesPerLine = channel * width
    if channel == 4:
        img_format = QImage.Format.Format_RGBA8888
    else:
        img_format = QImage.Format.Format_RGB888
    img = np.ascontiguousarray(img)
    qImg = QImage(img.data, width, height, bytesPerLine, img_format).rgbSwapped()
    if return_qimg:
        return qImg
    return QPixmap(qImg)

class TextBlkEncoder(NumpyEncoder):
    def default(self, obj):
        if isinstance(obj, TextBlock):
            return obj.to_dict()
        return NumpyEncoder.default(self, obj)

class ProjectDirNotExistException(Exception):
    pass

class ProjectLoadFailureException(Exception):
    pass

class ProjectNotSupportedException(Exception):
    pass

class ImgnameNotInProjectException(Exception):
    pass

class NotImplementedProjException(Exception):
    pass

class InvalidModuleConfigException(Exception):
    pass

class InvalidProgramConfigException(Exception):
    pass

@nested_dataclass
class FontFormat(Config):

    family: str = C.DEFAULT_FONT_FAMILY
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


class ProjHardSubExtract:
    def __init__(self):
        self.type = 'hardsubextract'
        raise NotImplementedProjException('hardsubextract')


class LruIgnoreArg:

    def __init__(self, **kwargs) -> None:
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __hash__(self) -> int:
        return hash(type(self))

    def __eq__(self, other):
        return isinstance(other, type(self))


span_pattern = re.compile(r'<span style=\"(.*?)\">', re.DOTALL)
p_pattern = re.compile(r'<p style=\"(.*?)\">', re.DOTALL)
fragment_pattern = re.compile(r'<!--(.*?)Fragment-->', re.DOTALL)
color_pattern = re.compile(r'color:(.*?);', re.DOTALL)
td_pattern = re.compile(r'<td(.*?)>(.*?)</td>', re.DOTALL)
table_pattern = re.compile(r'(.*?)<table', re.DOTALL)
fontsize_pattern = re.compile(r'font-size:(.*?)pt;', re.DOTALL)


def span_repl_func(matched, color):
    style = "<p style=\"" + matched.group(1) + " color:" + color + ";\">"
    return style

def p_repl_func(matched, color):
    style = "<p style=\"" + matched.group(1) + " color:" + color + ";\">"
    return style

def set_html_color(html, rgb):
    hex_color = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
    html = fragment_pattern.sub('', html)
    html = p_pattern.sub(lambda matched: p_repl_func(matched, hex_color), html)
    if color_pattern.findall(html):
        return color_pattern.sub(f'color:{hex_color};', html)
    else:
        return span_pattern.sub(lambda matched: span_repl_func(matched, hex_color), html)

def pt2px(pt):
    return int(round(pt * C.LDPI / 72.))

def px2pt(px):
    return px / C.LDPI * 72.

def html_max_fontsize(html:  str) -> float:
    size_list = fontsize_pattern.findall(html)
    size_list = [float(size) for size in size_list]
    if len(size_list) > 0:
        return max(size_list)
    else:
        return None

def doc_replace(doc: QTextDocument, span_list: List, target: str) -> List:
    len_replace = len(target)
    cursor = QTextCursor(doc)
    cursor.setPosition(0)
    cursor.beginEditBlock()
    pos_delta = 0
    sel_list = []
    for span in span_list:
        sel_start = span[0] + pos_delta
        sel_end = span[1] + pos_delta
        cursor.setPosition(sel_start)
        cursor.setPosition(sel_end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(target)
        sel_list.append([sel_start, sel_end])
        pos_delta += len_replace - (sel_end - sel_start)
    cursor.endEditBlock()
    return sel_list

def doc_replace_no_shift(doc: QTextDocument, span_list: List, target: str):
    cursor = QTextCursor(doc)
    cursor.setPosition(0)
    cursor.beginEditBlock()
    for span in span_list:
        cursor.setPosition(span[0])
        cursor.setPosition(span[1], QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(target)
    cursor.endEditBlock()

def hex2rgb(h: str):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def parse_stylesheet(theme: str = '', reverse_icon: bool = False) -> str:
    if reverse_icon:
        dark2light = True if theme == 'eva-light' else False
        reverse_icon_color(dark2light)
    with open(C.STYLESHEET_PATH, "r", encoding='utf-8') as f:
        stylesheet = f.read()
    with open(C.THEME_PATH, 'r', encoding='utf8') as f:
        theme_dict: Dict = json.loads(f.read())
    if not theme or theme not in theme_dict:
        tgt_theme: Dict = theme_dict[list(theme_dict.keys())[0]]
    else:
        tgt_theme: Dict = theme_dict[theme]

    C.FOREGROUND_FONTCOLOR = hex2rgb(tgt_theme['@qwidgetForegroundColor'])
    C.SLIDERHANDLE_COLOR = hex2rgb(tgt_theme['@sliderHandleColor'])
    for key, val in tgt_theme.items():
        stylesheet = stylesheet.replace(key, val)
    return stylesheet


ICON_DIR = 'icons'

LIGHTFILL_ACTIVE = "fill=\"#697187\""
LIGHTFILL = "fill=\"#b3b6bf\""
DARKFILL_ACTIVE = "fill=\"#96a4cd\""
DARKFILL = "fill=\"#697186\""

ICONREVERSE_DICT_LIGHT2DARK = {LIGHTFILL_ACTIVE: DARKFILL_ACTIVE, LIGHTFILL: DARKFILL}
ICONREVERSE_DICT_DARK2LIGHT = {DARKFILL_ACTIVE: LIGHTFILL_ACTIVE, DARKFILL: LIGHTFILL}
ICON_LIST = []

def reverse_icon_color(dark2light: bool = False):
    global ICON_LIST
    if not ICON_LIST:
        for filename in os.listdir(ICON_DIR):
            file_suffix = Path(filename).suffix
            if file_suffix.lower() != '.svg':
                continue
            else:
                ICON_LIST.append(osp.join(ICON_DIR, filename))

    if dark2light:
        pattern = re.compile(re.escape(DARKFILL) + '|' + re.escape(DARKFILL_ACTIVE))
        rep_dict = ICONREVERSE_DICT_DARK2LIGHT
    else:
        pattern = re.compile(re.escape(LIGHTFILL) + '|' + re.escape(LIGHTFILL_ACTIVE))
        rep_dict = ICONREVERSE_DICT_LIGHT2DARK
    for svgpath in ICON_LIST:
        with open(svgpath, "r", encoding="utf-8") as f:
            svg_content = f.read()
            svg_content = pattern.sub(lambda m:rep_dict[m.group()], svg_content)
        with open(svgpath, "w", encoding="utf-8") as f:
            f.write(svg_content)
import cv2, re
import numpy as np
import os.path as osp
from typing import Tuple, Union, List, Dict
from qtpy.QtGui import QPixmap,  QColor, QImage

from . import constants
from .constants import DEFAULT_FONT_FAMILY
from utils.io_utils import find_all_imgs, NumpyEncoder, imread, imwrite
from dl.textdetector.textblock import TextBlock


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

class InvalidDLModuleConfigException(Exception):
    pass

class InvalidProgramConfigException(Exception):
    pass


class FontFormat:
    def __init__(self, 
                 family: str = None,
                 size: float = 24,
                 stroke_width: float = 0,
                 frgb=(0, 0, 0),
                 srgb=(0, 0, 0),
                 bold: bool = False,
                 underline: bool = False,
                 italic: bool = False, 
                 alignment: int = 0,
                 vertical: bool = False, 
                 weight: int = 50, 
                 line_spacing: float = 1.2,
                 letter_spacing: float = 1.,
                 opacity: float = 1.,
                 shadow_radius: float = 0.,
                 shadow_strength: float = 1.,
                 shadow_color: Tuple = (0, 0, 0),
                 shadow_offset: List = [0, 0],
                 **kwargs) -> None:
        self.family = family if family is not None else DEFAULT_FONT_FAMILY
        self.size = size
        self.stroke_width = stroke_width
        self.frgb = frgb                  # font color
        self.srgb = srgb                    # stroke color
        self.bold = bold
        self.underline = underline
        self.italic = italic
        self.weight: int = weight
        self.alignment: int = alignment
        self.vertical: bool = vertical
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        self.opacity = opacity
        self.shadow_radius = shadow_radius
        self.shadow_strength = shadow_strength
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset

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


class DLModuleConfig:
    def __init__(self, 
                 textdetector: str = 'ctd',
                 ocr = "mit32px",
                 inpainter: str = 'lama_mpe',
                 translator = "google",
                 enable_ocr = True,
                 enable_translate = True,
                 enable_inpaint = True,
                 textdetector_setup_params = None,
                 ocr_setup_params = None,
                 translator_setup_params = None,
                 inpainter_setup_params = None,
                 translate_source = '日本語',
                 translate_target = '简体中文',
                 check_need_inpaint = True
                 ) -> None:
        self.textdetector = textdetector
        self.ocr = ocr
        self.inpainter = inpainter
        self.translator = translator
        self.enable_ocr = enable_ocr
        self.enable_translate = enable_translate
        self.enable_inpaint = enable_inpaint
        if textdetector_setup_params is None:
            self.textdetector_setup_params = dict()
        else:
            self.textdetector_setup_params = textdetector_setup_params
        if ocr_setup_params is None:
            self.ocr_setup_params = dict()
        else:
            self.ocr_setup_params = ocr_setup_params
        if translator_setup_params is None:
            self.translator_setup_params = dict()
        else:
            self.translator_setup_params = translator_setup_params
        if inpainter_setup_params is None:
            self.inpainter_setup_params = dict()
        else:
            self.inpainter_setup_params = inpainter_setup_params
        self.translate_source = translate_source
        self.translate_target = translate_target
        self.check_need_inpaint = check_need_inpaint

    def __getitem__(self, item: str):
        if item == 'textdetector':
            return self.textdetector
        elif item == 'ocr':
            return self.ocr
        elif item == 'translator':
            return self.translator
        elif item == 'inpainter':
            return self.inpainter
        else:
            raise KeyError(item)

    def get_setup_params(self, module_key: str) -> dict:
        if module_key == 'textdetector':
            return self.textdetector_setup_params
        elif module_key == 'ocr':
            return self.ocr_setup_params
        elif module_key == 'translator':
            return self.translator_setup_params
        elif module_key == 'inpainter':
            return self.inpainter_setup_params


class DrawPanelConfig:
    def __init__(self, 
                 pentool_color: List = None,
                 pentool_width: float = 30.,
                 inpainter_width: float = 30.,
                 current_tool: int = 0,
                 rectool_auto: bool = False, 
                 rectool_method: int = 0,
                 recttool_dilate_ksize: int = 0,
                 **kwargs) -> None:
        self.pentool_color = pentool_color if pentool_color is not None else [0, 0, 0]
        self.pentool_width = pentool_width
        self.inpainter_width = inpainter_width
        self.current_tool = current_tool
        self.rectool_auto = rectool_auto
        self.rectool_method = rectool_method
        self.recttool_dilate_ksize = recttool_dilate_ksize


class ProgramConfig:
    def __init__(
        self, dl: Union[Dict, DLModuleConfig] = None,
        drawpanel: Union[Dict, DrawPanelConfig] = None,
        global_fontformat: Union[Dict, FontFormat] = None,
        recent_proj_list: List[str] = list(),
        imgtrans_paintmode: bool = False,
        imgtrans_textedit: bool = True,
        imgtrans_textblock: bool = True,
        mask_transparency: float = 0.,
        original_transparency: float = 0.,
        open_recent_on_startup: bool = True, 
        let_fntsize_flag: int = 0,
        let_fntstroke_flag: int = 0,
        let_fntcolor_flag: int = 0,
        let_fnteffect_flag: int = 1,
        let_alignment_flag: int = 0,
        let_autolayout_flag: bool = True,
        let_uppercase_flag: bool = True,
        font_presets: dict = None,
        fsearch_case: bool = False,
        fsearch_whole_word: bool = False,
        fsearch_regex: bool = False,
        fsearch_range: int = 0,
        gsearch_case: bool = False,
        gsearch_whole_word: bool = False,
        gsearch_regex: bool = False,
        gsearch_range: int = 0,
        **kwargs) -> None:

        if isinstance(dl, dict):
            self.dl = DLModuleConfig(**dl)
        elif dl is None:
            self.dl = DLModuleConfig()
        else:
            self.dl = dl
        if isinstance(drawpanel, dict):
            self.drawpanel = DrawPanelConfig(**drawpanel)
        elif drawpanel is None:
            self.drawpanel = DrawPanelConfig()
        else:
            self.drawpanel = drawpanel
        if isinstance(global_fontformat, dict):
            self.global_fontformat = FontFormat(**global_fontformat)
        elif global_fontformat is None:
            self.global_fontformat = FontFormat()
        else:
            self.global_fontformat = global_fontformat
        self.recent_proj_list = recent_proj_list
        self.imgtrans_paintmode = imgtrans_paintmode
        self.imgtrans_textedit = imgtrans_textedit
        self.imgtrans_textblock = imgtrans_textblock
        self.mask_transparency = mask_transparency
        self.original_transparency = original_transparency
        self.open_recent_on_startup = open_recent_on_startup
        self.let_fntsize_flag = let_fntsize_flag
        self.let_fntstroke_flag = let_fntstroke_flag
        self.let_fntcolor_flag = let_fntcolor_flag
        self.let_fnteffect_flag = let_fnteffect_flag
        self.let_alignment_flag = let_alignment_flag
        self.let_autolayout_flag = let_autolayout_flag
        self.let_uppercase_flag = let_uppercase_flag
        self.font_presets = {} if font_presets is None else font_presets
        self.fsearch_case = fsearch_case
        self.fsearch_whole_word = fsearch_whole_word
        self.fsearch_regex = fsearch_regex
        self.fsearch_range = fsearch_range
        self.gsearch_case = gsearch_case
        self.gsearch_whole_word = gsearch_whole_word
        self.gsearch_regex = gsearch_regex
        self.gsearch_range = gsearch_range

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
    return int(round(pt * constants.LDPI / 72.))

def px2pt(px):
    return px / constants.LDPI * 72.

def html_max_fontsize(html:  str) -> float:
    size_list = fontsize_pattern.findall(html)
    size_list = [float(size) for size in size_list]
    if len(size_list) > 0:
        return max(size_list)
    else:
        return None
import cv2, re, json, os
from pathlib import Path
import numpy as np
import os.path as osp
from typing import Tuple, Union, List, Dict
from qtpy.QtGui import QPixmap,  QColor, QImage, QTextDocument, QTextCursor

from . import constants as C
from .constants import DEFAULT_FONT_FAMILY, STYLESHEET_PATH, THEME_PATH
from utils.io_utils import find_all_imgs, NumpyEncoder, imread, imwrite
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


class ModuleConfig:
    def __init__(self, 
                 textdetector: str = 'ctd',
                 ocr = "mit48px_ctc",
                 inpainter: str = 'lama_mpe',
                 translator = "google",
                 enable_ocr = True,
                 enable_translate = True,
                 enable_inpaint = True,
                 textdetector_params = None,
                 ocr_params = None,
                 translator_params = None,
                 inpainter_params = None,
                 translate_source = '日本語',
                 translate_target = '简体中文',
                 check_need_inpaint = True,
                 ) -> None:

        self.textdetector = textdetector
        self.ocr = ocr
        self.inpainter = inpainter
        self.translator = translator
        self.enable_ocr = enable_ocr
        self.enable_translate = enable_translate
        self.enable_inpaint = enable_inpaint
        if textdetector_params is None:
            self.textdetector_params = dict()
        else:
            self.textdetector_params = textdetector_params
        if ocr_params is None:
            self.ocr_params = dict()
        else:
            self.ocr_params = ocr_params
        if translator_params is None:
            self.translator_params = dict()
        else:
            self.translator_params = translator_params
            if 'google' in translator_params:
                if 'url' in translator_params['google'] and \
                    translator_params['google']['url']['select'] == 'https://translate.google.cn/m':
                    translator_params['google']['url']['select'] = 'https://translate.google.com/m'
        if inpainter_params is None:
            self.inpainter_params = dict()
        else:
            self.inpainter_params = inpainter_params
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

    def get_params(self, module_key: str) -> dict:
        if module_key == 'textdetector':
            return self.textdetector_params
        elif module_key == 'ocr':
            return self.ocr_params
        elif module_key == 'translator':
            return self.translator_params
        elif module_key == 'inpainter':
            return self.inpainter_params


class DrawPanelConfig:
    def __init__(self, 
                 pentool_color: List = None,
                 pentool_width: float = 30.,
                 pentool_shape: int = 0,
                 inpainter_width: float = 30.,
                 inpainter_shape: int = 0,
                 current_tool: int = 0,
                 rectool_auto: bool = False, 
                 rectool_method: int = 0,
                 recttool_dilate_ksize: int = 0,
                 **kwargs) -> None:
        self.pentool_color = pentool_color if pentool_color is not None else [0, 0, 0]
        self.pentool_width = pentool_width
        self.pentool_shape = pentool_shape
        self.inpainter_width = inpainter_width
        self.inpainter_shape = inpainter_shape
        self.current_tool = current_tool
        self.rectool_auto = rectool_auto
        self.rectool_method = rectool_method
        self.recttool_dilate_ksize = recttool_dilate_ksize


class ProgramConfig:
    def __init__(
        self, module: Union[Dict, ModuleConfig] = None,
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
        let_fnt_scolor_flag: int = 0,
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
        darkmode: bool = False,
        src_link_flag: str = '',
        textselect_mini_menu: bool = True,
        saladict_shortcut: str = "Alt+S",
        search_url: str = "https://www.google.com/search?q=",
        ocr_sublist: dict = None,
        mt_sublist: dict = None,
        **kwargs) -> None:



        if isinstance(module, dict):
            self.module = ModuleConfig(**module)
        elif module is None:
            self.module = ModuleConfig()
        else:
            self.module = module
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
        self.let_fnt_scolor_flag = let_fnt_scolor_flag
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
        self.darkmode = darkmode
        self.src_link_flag = src_link_flag
        self.textselect_mini_menu = textselect_mini_menu
        self.saladict_shortcut = saladict_shortcut
        self.search_url = search_url
        self.ocr_sublist = [] if ocr_sublist is None else ocr_sublist
        self.mt_sublist = [] if mt_sublist is None else mt_sublist

    @staticmethod
    def load(cfg_path: str):
        
        with open(cfg_path, 'r', encoding='utf8') as f:
            config_dict = json.loads(f.read())

        # for backward compatibility
        if 'dl' in config_dict:
            dl = config_dict.pop('dl')
            if not 'module' in config_dict:
                if 'textdetector_setup_params' in dl:
                    textdetector_params = dl.pop('textdetector_setup_params')
                    dl['textdetector_params'] = textdetector_params
                if 'inpainter_setup_params' in dl:
                    inpainter_params = dl.pop('inpainter_setup_params')
                    dl['inpainter_params'] = inpainter_params
                if 'ocr_setup_params' in dl:
                    ocr_params = dl.pop('ocr_setup_params')
                    dl['ocr_params'] = ocr_params
                if 'translator_setup_params' in dl:
                    translator_params = dl.pop('translator_setup_params')
                    dl['translator_params'] = translator_params
                config_dict['module'] = dl

        if 'module' in config_dict:
            module_cfg = config_dict['module']
            trans_params = module_cfg['translator_params']
            repl_pairs = {'baidu': 'Baidu', 'caiyun': 'Caiyun', 'chatgpt': 'ChatGPT', 'Deepl': 'DeepL', 'papago': 'Papago'}
            for k, i in repl_pairs.items():
                if k in trans_params:
                    trans_params[i] = trans_params.pop(k)
            if module_cfg['translator'] in repl_pairs:
                module_cfg['translator'] = repl_pairs[module_cfg['translator']]
            

        return ProgramConfig(**config_dict)

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
    with open(STYLESHEET_PATH, "r", encoding='utf-8') as f:
        stylesheet = f.read()
    with open(THEME_PATH, 'r', encoding='utf8') as f:
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
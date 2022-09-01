import cv2
import numpy as np
import os.path as osp
import json
import re
import os
from collections import OrderedDict
from typing import Tuple, Union, List, Dict
from qtpy.QtGui import QPixmap,  QColor, QImage

from . import constants
from .constants import DEFAULT_FONT_FAMILY
from utils.logger import logger as LOGGER
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
                 alpha: int = 255,
                 line_spacing: float = 1.2,
                 letter_spacing: float = 1.) -> None:
        self.family = family if family is not None else DEFAULT_FONT_FAMILY
        self.size = size
        self.stroke_width = stroke_width
        self.frgb = frgb                  # font color
        self.srgb = srgb                    # stroke color
        self.bold = bold
        self.underline = underline
        self.italic = italic
        self.alpha = alpha
        self.weight: int = weight
        self.alignment: int = alignment
        self.vertical: bool = vertical
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing

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


PROJTYPE_IMGTRANS = 'imgtrans'
PROJTYPE_HARDSUBEXTRACT = 'hardsubextract'

class ProjImgTrans:

    def __init__(self, directory: str = None):
        self.type = PROJTYPE_IMGTRANS
        self.directory: str = None
        self.pages: OrderedDict[List[TextBlock]] = OrderedDict()
        self.not_found_pages: Dict[List[TextBlock]] = {}
        self.new_pages: List[str] = []
        self.proj_path: str = None
        self.current_img: str = None
        self.img_array: np.ndarray = None
        self.mask_array: np.ndarray = None
        self.inpainted_array: np.ndarray = None
        if directory is not None:
            self.load(directory)

    def load(self, directory: str) -> bool:
        self.directory = directory
        self.proj_path = osp.join(self.directory, 
                            self.type+'_'+osp.basename(self.directory) + '.json')
        new_proj = False
        if not osp.exists(self.proj_path):
            new_proj = True
            self.new_project()
        else:
            try:
                with open(self.proj_path, 'r', encoding='utf8') as f:
                    proj_dict = json.loads(f.read())
            except Exception as e:
                raise ProjectLoadFailureException(e)
            self.load_from_dict(proj_dict)
        if not osp.exists(self.inpainted_dir()):
            os.makedirs(self.inpainted_dir())
        if not osp.exists(self.mask_dir()):
            os.makedirs(self.mask_dir())
        return new_proj

    def mask_dir(self):
        return osp.join(self.directory, 'mask')

    def inpainted_dir(self):
        return osp.join(self.directory, 'inpainted')

    def result_dir(self):
        return osp.join(self.directory, 'result')

    def load_from_dict(self, proj_dict: dict):
        self.set_current_img(None)
        try:
            self.pages = OrderedDict()
            page_dict = proj_dict['pages']
            not_found_pages = list(page_dict.keys())
            found_pages = find_all_imgs(img_dir=self.directory, abs_path=False)
            for imname in found_pages:
                if imname in page_dict:
                    self.pages[imname] = [TextBlock(**blk_dict) for blk_dict in page_dict[imname]]
                    not_found_pages.remove(imname)
                else:
                    self.pages[imname] = []
                    self.new_pages.append(imname)
            for imname in not_found_pages:
                self.not_found_pages[imname] = [TextBlock(**blk_dict) for blk_dict in page_dict[imname]]
        except Exception as e:
            raise ProjectNotSupportedException(e)
        set_img_failed = False
        if 'current_img' in proj_dict:
            current_img = proj_dict['current_img']
            try:
                self.set_current_img(current_img)
            except ImgnameNotInProjectException:
                set_img_failed = True
        else:
            set_img_failed = True
            LOGGER.warning(f'{current_img} not found.')
        if set_img_failed:
            if len(self.pages) > 0:
                self.set_current_img_byidx(0)

    def set_current_img(self, imgname: str):
        if imgname is not None:
            if imgname not in self.pages:
                raise ImgnameNotInProjectException
            self.current_img = imgname
            img_path = self.current_img_path()
            mask_path = self.mask_path()
            inpainted_path = self.inpainted_path()
            self.img_array = imread(img_path)
            im_h, im_w = self.img_array.shape[:2]
            if osp.exists(mask_path):
                self.mask_array = imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                self.mask_array = np.zeros((im_h, im_w), dtype=np.uint8)
            self.inpainted_array = imread(inpainted_path) if osp.exists(inpainted_path) else np.copy(self.img_array)
        else:
            self.current_img = None
            self.img_array = None
            self.mask_array = None
            self.inpainted_array = None

    def set_current_img_byidx(self, idx: int):
        num_pages = self.num_pages
        if idx < 0:
            idx = idx + self.num_pages
        if idx < 0 or idx > num_pages - 1:
            self.set_current_img(None)
        else:
            self.set_current_img(list(self.pages)[idx])

    def get_blklist_byidx(self, idx: int) -> List[TextBlock]:
        return self.pages[list(self.pages)[idx]]

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def new_project(self):
        if not osp.exists(self.directory):
            raise ProjectDirNotExistException
        self.set_current_img(None)
        imglist = find_all_imgs(self.directory, abs_path=False)
        self.pages = OrderedDict()
        for imgname in imglist:
            self.pages[imgname] = []
        self.set_current_img_byidx(0)
        self.save()
        
    def save(self):
        if not osp.exists(self.directory):
            raise ProjectDirNotExistException
        with open(self.proj_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False, cls=TextBlkEncoder))
        # if save_mask and self.mask_valid:
        #     self.save_mask(self.current_img, self.mask_array)
        # if save_inpainted and self.inpainted_valid:
        #     self.save_inpainted(self.current_img, self.inpainted_array)

    def to_dict(self) -> Dict:
        pages = self.pages.copy()
        pages.update(self.not_found_pages)
        return {
            'directory': self.directory,
            'pages': pages,
            'current_img': self.current_img
        }

    def read_img(self, imgname: str) -> np.ndarray:
        if imgname not in self.pages:
            raise ImgnameNotInProjectException
        return imread(osp.join(self.directory, imgname))

    def save_mask(self, img_name, mask: np.ndarray):
        imwrite(self.get_mask_path(img_name), mask)

    def save_inpainted(self, img_name, inpainted: np.ndarray):
        imwrite(self.get_inpainted_path(img_name), inpainted)

    def current_img_path(self) -> str:
        if self.current_img is None:
            return None
        return osp.join(self.directory, self.current_img)

    def mask_path(self) -> str:
        if self.current_img is None:
            return None
        return self.get_mask_path(self.current_img)

    def inpainted_path(self) -> str:
        if self.current_img is None:
            return None
        return self.get_inpainted_path(self.current_img)

    def get_mask_path(self, imgname: str = None) -> str:
        if imgname is None:
            imgname = self.current_img
        return osp.join(self.mask_dir(), osp.splitext(imgname)[0]+'.png')

    def get_inpainted_path(self, imgname: str = None) -> str:
        if imgname is None:
            imgname = self.current_img
        return osp.join(self.inpainted_dir(), osp.splitext(imgname)[0]+'.png')

    def get_result_path(self, imgname: str) -> str:
        return osp.join(self.result_dir(), osp.splitext(imgname)[0]+'.png')
        
    def backup(self):
        raise NotImplementedError

    @property
    def is_empty(self):
        return len(self.pages) == 0

    @property
    def img_valid(self):
        return self.img_array is not None
    
    @property
    def mask_valid(self):
        return self.mask_array is not None

    @property
    def inpainted_valid(self):
        return self.inpainted_array is not None

    def set_next_img(self):
        if self.current_img is not None:
            keylist = list(self.pages.keys())
            current_index = keylist.index(self.current_img)
            next_index = (current_index + 1) % len(keylist)
            self.set_current_img(keylist[next_index])

    def set_prev_img(self):
        if self.current_img is not None:
            keylist = list(self.pages.keys())
            current_index = keylist.index(self.current_img)
            next_index = (current_index - 1 + len(keylist)) % len(keylist)
            self.set_current_img(keylist[next_index])

    def current_block_list(self) -> List[TextBlock]:
        if self.current_img is not None:
            assert self.current_img in self.pages
            return self.pages[self.current_img]
        else:
            return None


class ProjHardSubExtract:
    def __init__(self):
        self.type = PROJTYPE_HARDSUBEXTRACT
        raise NotImplementedProjException(PROJTYPE_HARDSUBEXTRACT)


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
                 rectool_method: int = 0) -> None:
        self.pentool_color = pentool_color if pentool_color is not None else [0, 0, 0]
        self.pentool_width = pentool_width
        self.inpainter_width = inpainter_width
        self.current_tool = current_tool
        self.rectool_auto = rectool_auto
        self.rectool_method = rectool_method


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
        let_alignment_flag: int = 0,
        let_autolayout_flag: bool = True,
        let_uppercase_flag: bool = True) -> None:

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
        self.let_alignment_flag = let_alignment_flag
        self.let_autolayout_flag = let_autolayout_flag
        self.let_uppercase_flag = let_uppercase_flag


class LruIgnoreArgs:

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
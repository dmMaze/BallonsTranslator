import cv2, re, json, os
from pathlib import Path
import numpy as np
import os.path as osp
from qtpy.QtGui import QPixmap,  QColor, QImage, QTextDocument, QTextCursor
from qtpy.QtCore import Qt, QPointF

from ballontranslator.utils.config import pcfg
from ballontranslator.utils import shared
from ballontranslator.utils.structures import Tuple, Union, List, Dict, Config, field, nested_dataclass


QKEY = Qt.Key
QNUMERIC_KEYS = {QKEY.Key_0:0,QKEY.Key_1:1,QKEY.Key_2:2,QKEY.Key_3:3,QKEY.Key_4:4,QKEY.Key_5:5,QKEY.Key_6:6,QKEY.Key_7:7,QKEY.Key_8:8,QKEY.Key_9:9}

ARROWKEY2DIRECTION = {
    QKEY.Key_Left: QPointF(-1., 0.),
    QKEY.Key_Right: QPointF(1., 0.),
    QKEY.Key_Up: QPointF(0., -1.),
    QKEY.Key_Down: QPointF(0., 1.),
}

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
        qimg = pixmap.convertToFormat(QImage.Format.Format_RGBA8888)

    byte_str = qimg.bits()
    if byte_str is None:
        return None

    if hasattr(byte_str, 'asstring'):
        byte_str = qimg.bits().asstring(h * w * 4)
    else:
        byte_str = byte_str.tobytes()

    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4)).copy()
    
    if keep_alpha:
        return img
    else:
        return np.ascontiguousarray(img[:,:,:3])

def ndarray2pixmap(img, return_qimg=False):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    height, width, channel = img.shape
    bytesPerLine = channel * width
    if channel == 4:
        img_format = QImage.Format.Format_RGBA8888
    else:
        img_format = QImage.Format.Format_RGB888
    img = np.ascontiguousarray(img)
    qImg = QImage(img.data, width, height, bytesPerLine, img_format)
    if return_qimg:
        return qImg
    return QPixmap(qImg)


span_pattern = re.compile(r'<span style=\"(.*?)\">', re.DOTALL)
p_pattern = re.compile(r'<p style=\"(.*?)\">', re.DOTALL)
fragment_pattern = re.compile(r'<!--(.*?)Fragment-->', re.DOTALL)
color_pattern = re.compile(r'color:(.*?);', re.DOTALL)
td_pattern = re.compile(r'<td(.*?)>(.*?)</td>', re.DOTALL)
table_pattern = re.compile(r'(.*?)<table', re.DOTALL)
fontsize_pattern = re.compile(r'font-size:(.*?)pt;', re.DOTALL)
ffamily_pattern = re.compile(r'font-family:\'(.*?)\'', re.DOTALL)


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
    
def set_html_family(html, family):
    return ffamily_pattern.sub(f'font-family:\'{family}\'', html)

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

THEMED_ICON_CACHE_DIR = osp.join(shared.cache_dir, 'icons')

LIGHTFILL_ACTIVE = "fill=\"#697187\""
LIGHTFILL = "fill=\"#b3b6bf\""
DARKFILL_ACTIVE = "fill=\"#96a4cd\""
DARKFILL = "fill=\"#697186\""

ICONREVERSE_DICT_LIGHT2DARK = {LIGHTFILL_ACTIVE: DARKFILL_ACTIVE, LIGHTFILL: DARKFILL}
ICONREVERSE_DICT_DARK2LIGHT = {DARKFILL_ACTIVE: LIGHTFILL_ACTIVE, DARKFILL: LIGHTFILL}
THEMED_ICON_LIST = []
THEMED_ICON_READY = set()
STYLESHEET_ICON_PATTERN = re.compile(r'url\((["\']?)(?:resources/)?icons/([^)"\']+)\1\)')

def _default_theme() -> str:
    return 'eva-dark' if pcfg.darkmode else 'eva-light'

def _resolve_theme(theme: str = None) -> str:
    return theme or _default_theme()

def _themed_icon_cache_dir(theme: str) -> str:
    return osp.join(THEMED_ICON_CACHE_DIR, theme)

def _list_svg_icons() -> List[str]:
    global THEMED_ICON_LIST
    if not THEMED_ICON_LIST:
        icon_dir = shared.ICON_DIR
        for filename in os.listdir(icon_dir):
            file_suffix = Path(filename).suffix
            if file_suffix.lower() == '.svg':
                THEMED_ICON_LIST.append(filename)
    return THEMED_ICON_LIST

def ensure_themed_icons(theme: str):
    theme = _resolve_theme(theme)
    if theme in THEMED_ICON_READY:
        return

    cache_dir = _themed_icon_cache_dir(theme)
    os.makedirs(cache_dir, exist_ok=True)

    if theme == 'eva-dark':
        pattern = re.compile(re.escape(LIGHTFILL) + '|' + re.escape(LIGHTFILL_ACTIVE))
        rep_dict = ICONREVERSE_DICT_LIGHT2DARK
    else:
        pattern = re.compile(re.escape(DARKFILL) + '|' + re.escape(DARKFILL_ACTIVE))
        rep_dict = ICONREVERSE_DICT_DARK2LIGHT

    icon_dir = shared.ICON_DIR
    for filename in _list_svg_icons():
        src_path = osp.join(icon_dir, filename)
        cache_path = osp.join(cache_dir, filename)
        with open(src_path, "r", encoding="utf-8") as f:
            svg_content = pattern.sub(lambda m: rep_dict[m.group()], f.read())
        if osp.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                if f.read() == svg_content:
                    continue
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
    THEMED_ICON_READY.add(theme)

def themed_icon_path(filename: str, theme: str = None) -> str:
    theme = _resolve_theme(theme)
    ensure_themed_icons(theme)
    return osp.join(_themed_icon_cache_dir(theme), filename)

def themed_icon_url(filename: str, theme: str = None) -> str:
    return '"' + Path(themed_icon_path(filename, theme)).as_posix().replace('"', '\\"') + '"'

def icon_url(filename: str) -> str:
    return '"' + Path(osp.join(shared.ICON_DIR, filename)).as_posix().replace('"', '\\"') + '"'

def _replace_stylesheet_icon_url(matched, theme: str) -> str:
    filename = matched.group(2)
    if Path(filename).suffix.lower() == '.svg':
        return f'url({themed_icon_url(filename, theme)})'
    return f'url({icon_url(filename)})'

def parse_stylesheet(theme: str = '') -> str:
    with open(shared.STYLESHEET_PATH, "r", encoding='utf-8') as f:
        stylesheet = f.read()
    with open(shared.THEME_PATH, 'r', encoding='utf8') as f:
        theme_dict: Dict = json.loads(f.read())
    if not theme or theme not in theme_dict:
        theme = list(theme_dict.keys())[0]
    tgt_theme: Dict = theme_dict[theme]

    ensure_themed_icons(theme)
    stylesheet = STYLESHEET_ICON_PATTERN.sub(lambda m: _replace_stylesheet_icon_url(m, theme), stylesheet)

    shared.FOREGROUND_FONTCOLOR = hex2rgb(tgt_theme['@qwidgetForegroundColor'])
    shared.SLIDERHANDLE_COLOR = hex2rgb(tgt_theme['@sliderHandleColor'])
    shared.BORDER_COLOR = hex2rgb(tgt_theme['@borderColor'])
    shared.WIDGET_BACKGROUND_COLOR = hex2rgb(tgt_theme['@widgetBackgroundColor'])
    for key, val in tgt_theme.items():
        stylesheet = stylesheet.replace(key, val)
    return stylesheet

def mutate_dict_key(adict: dict, old_key: Union[str, int], new_key: str):
    # https://stackoverflow.com/questions/12150872/change-key-in-ordereddict-without-losing-order
    key_list = list(adict.keys())
    if isinstance(old_key, int):
        old_key = key_list[old_key]
    
    for key in key_list:
        value = adict.pop(key)
        adict[new_key if old_key == key else key] = value

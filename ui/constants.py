import os.path as osp
import os
import sys

ICON_PATH = 'icons/[ICONNAME]'

UI_PATH = osp.dirname(osp.abspath(__file__))
PROGRAM_PATH = osp.dirname(UI_PATH)
LOGGING_PATH = osp.join(PROGRAM_PATH, 'logs')

LIBS_PATH = osp.join(PROGRAM_PATH, 'data/libs')

STYLESHEET_PATH = osp.join(PROGRAM_PATH, 'config/stylesheet.css')
THEME_PATH = osp.join(PROGRAM_PATH, 'config/themes.json')
CONFIG_PATH = osp.join(PROGRAM_PATH, 'config/config.json')

DOWNLOAD_PATH = osp.join(PROGRAM_PATH, 'gallery-dl')

CONFIG_FONTSIZE_HEADER = 18
CONFIG_FONTSIZE_TABLE = 16
CONFIG_FONTSIZE_CONTENT = 16

CONFIG_COMBOBOX_HEIGHT = 30 
CONFIG_COMBOBOX_SHORT = 200
CONFIG_COMBOBOX_MIDEAN = 332
CONFIG_COMBOBOX_LONG = 468

HORSLIDER_FIXHEIGHT = 36

WIDGET_SPACING_CLOSE = 8
TEXTEDIT_FIXWIDTH = 350

TEXTEFFECT_FIXWIDTH = 400
TEXTEFFECT_MAXHEIGHT = 500

LEFTBAR_WIDTH = 60
LEFTBTN_WIDTH = 38

LDPI = 96.
DPI = 188.75

SCREEN_H = 2160
SCREEN_W = 3840

DEFAULT_FONT_FAMILY = 'Microsoft YaHei UI'
APP_DEFAULT_FONT = 'Microsoft YaHei UI'

WINDOW_BORDER_WIDTH = 4
BOTTOMBAR_HEIGHT = 32
TITLEBAR_HEIGHT = 30

PAGELIST_THUMBNAIL_MAXNUM = 100
PAGELIST_THUMBNAIL_SIZE = 48

FLAG_QT6 = True

SLIDERHANDLE_COLOR = (85,85,96)
FOREGROUND_FONTCOLOR = (93,93,95)

MAX_NUM_LOG = 7

TRANSLATE_DIR = osp.join(PROGRAM_PATH, 'translate')
DISPLAY_LANGUAGE_MAP = {
    'English': 'English',
    '简体中文': 'zh_CN',
    'Русский': 'ru_RU'
}
VALID_LANG_SET = set(list(DISPLAY_LANGUAGE_MAP.values()))

for p in os.listdir(TRANSLATE_DIR):
    if p.endswith('.qm'):
        lang = p.replace('.qm', '')
        if lang not in VALID_LANG_SET:
            DISPLAY_LANGUAGE_MAP[lang] = lang

DEFAULT_DISPLAY_LANG = 'English'

USE_PYSIDE6 = False
ON_MACOS = sys.platform == 'darwin'

DEBUG = False
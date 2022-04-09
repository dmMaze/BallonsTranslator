import os.path as osp

UI_PATH = osp.dirname(osp.abspath(__file__))
PROGRAM_PATH = osp.dirname(UI_PATH)
LIBS_PATH = osp.join(PROGRAM_PATH, 'data/libs')

STYLESHEET_PATH = osp.join(PROGRAM_PATH, 'config/stylesheet.css')
CONFIG_PATH = osp.join(PROGRAM_PATH, 'config/config.json')

CONFIG_FONTSIZE_HEADER = 24
CONFIG_FONTSIZE_TABLE = 14
CONFIG_FONTSIZE_CONTENT = 14

CONFIG_COMBOBOX_SHORT = 300
CONFIG_COMBOBOX_MIDEAN = 500
CONFIG_COMBOBOX_LONG = 700

LDPI = 96.
DPI = 188.75

LANG_SUPPORT_VERTICAL = [
    '简体中文',
    '繁體中文',
    '日本語',
    '한국어'
]

DEFAULT_FONT_FAMILY = 'Arial'
from typing import Dict, Iterable, List, Optional
import os
import os.path as osp
import json
import sys
from pathlib import Path

def _resolve_program_path() -> str:
    env_root = os.environ.get('BALLOONTRANS_APP_ROOT')
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd())
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        has_app_dirs = all((candidate / name).exists() for name in ('config', 'data'))
        has_resources = (candidate / 'resources' / 'icons').exists() and (candidate / 'resources' / 'translate').exists()
        if has_app_dirs and has_resources:
            return str(candidate.resolve())

    # Package layout fallback: ballontranslator/utils/shared.py -> repo root.
    return str(Path(__file__).resolve().parents[2])


PROGRAM_PATH = _resolve_program_path()
RESOURCE_DIR = osp.join(PROGRAM_PATH, 'resources')
ICON_DIR = osp.join(RESOURCE_DIR, 'icons')
ICON_PATH = osp.join(ICON_DIR, 'icon.icns')
LOGGING_PATH = osp.join(PROGRAM_PATH, 'logs')

LIBS_PATH = osp.join(PROGRAM_PATH, 'data/libs')

STYLESHEET_PATH = osp.join(RESOURCE_DIR, 'stylesheet.css')
THEME_PATH = osp.join(RESOURCE_DIR, 'themes.json')
CONFIG_PATH = osp.join(PROGRAM_PATH, 'config/config.json')

DEFAULT_TEXTSTYLE_DIR = osp.join(PROGRAM_PATH, 'config/textstyles')
if not osp.exists(DEFAULT_TEXTSTYLE_DIR):
    os.makedirs(DEFAULT_TEXTSTYLE_DIR)


CONFIG_FONTSIZE_HEADER = 15
CONFIG_FONTSIZE_TABLE = 13
CONFIG_FONTSIZE_CONTENT = 13
CONFIG_CONTENT_MARGIN = 27
CONFIG_CONTENT_MARGINS = (CONFIG_CONTENT_MARGIN,) * 4
CONFIG_CONTENT_ROW_SPACING = 6

CONFIG_COMBOBOX_HEIGHT = 26
CONFIG_COMBOBOX_SHORT = 180
CONFIG_COMBOBOX_MIDEAN = 300
CONFIG_COMBOBOX_LONG = 420

_size2width = {
    'short': CONFIG_COMBOBOX_SHORT,
    'median': CONFIG_COMBOBOX_MIDEAN,
    'long':CONFIG_COMBOBOX_LONG
}

def size2width(size: str):
    global _size2width
    return _size2width[size]

HORSLIDER_FIXHEIGHT = 36

WIDGET_SPACING_CLOSE = 8
TEXTEDIT_FIXWIDTH = 350

TEXTEFFECT_FIXWIDTH = 400
TEXTEFFECT_MAXHEIGHT = 500

LEFTBAR_WIDTH = 48
LEFTBTN_WIDTH = 28

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
BORDER_COLOR = (179,182,191)
WIDGET_BACKGROUND_COLOR = (235,238,245)

MAX_NUM_LOG = 7

TRANSLATE_DIR = osp.join(RESOURCE_DIR, 'translate')
DISPLAY_LANGUAGE_MAP = {
    "English": "English",
    "简体中文": "zh_CN",
    "繁體中文": "zh_TW",
    "Русский": "ru_RU",
    "Português (Brasil)": "pt_BR",
    "한국어": "ko_KR",
    "Español": "es_MX",
    "Hungarian": "hu_HU",
    "Français": "fr_FR"
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
ON_WINDOWS = sys.platform == 'win32'

def _detect_apple_silicon() -> bool:
    if not ON_MACOS:
        return False
    import platform
    if platform.machine().lower() in {'arm64', 'aarch64'}:
        return True
    try:
        import subprocess
        out = subprocess.run(
            ['sysctl', '-n', 'hw.optional.arm64'],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() == '1'
    except Exception:
        return False

ON_APPLE_SILICON = _detect_apple_silicon()
HEADLESS = False
DEBUG = False
args = None
TORCH_INSTALL_PREFERRED_DEVICE = None
TORCH_INSTALL_PREFERRED_PROFILE = None

FUZZY_MATCH_IMAGE_NAME = False

cache_data: Dict = None
cache_dir: str = osp.join(PROGRAM_PATH, '.btrans_cache')
cache_path: str = osp.join(PROGRAM_PATH, '.btrans_cache/cache.json')
CACHE_UPDATED = False
check_local_file_hash = True

FONT_FAMILIES: set = None
CUSTOM_FONTS = []
# Windows 自带的老旧字体：渲染时可能触发 DirectWrite CreateFontFaceFromHDC 告警
LEGACY_FONTS = frozenset({
    "MS Sans Serif", "MS Serif", "Small Fonts",
    "System", "Fixedsys", "Terminal",
    "Courier", "Modern", "Roman", "Script",
})


def get_filtered_font_list(
    font_list: Iterable[str],
    excluded: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return a sorted font list without the excluded names.

    >>> get_filtered_font_list(['Times', 'Arial', 'Courier'], ['Times'])
    ['Arial', 'Courier']
    >>> get_filtered_font_list(['Arial', 'Times'])
    ['Arial', 'Times']
    """
    excluded_set = set(excluded or ())
    return sorted(
        (font for font in font_list if font not in excluded_set),
        key=str.casefold,
    )


pbar = {}
runtime_widget_set = set()

def add_to_runtime_widget_set(widget):
    runtime_widget_set.add(widget)

def remove_from_runtime_widget_set(widget):
    if widget in runtime_widget_set:
        runtime_widget_set.remove(widget)

showed_exception = set()

# it will be set to ui.mainwindow.create_errdialog.emit after UI initialized
create_errdialog_in_mainthread = lambda *args, **kwargs: None

create_infodialog_in_mainthread = lambda *args, **kwargs: None

show_llm_key_dialog_in_mainthread = lambda *args, **kwargs: None

show_llm_model_dialog_in_mainthread = lambda *args, **kwargs: None

show_llm_base_url_dialog_in_mainthread = lambda *args, **kwargs: None

def load_cache():
    global cache_data
    if cache_data is None:
        if osp.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf8") as file:
                    cache_data = json.load(file)
            except:
                print(f'cached file {cache_path} is invalid')
                cache_data = {}
        else:
            cache_data = {}

def dump_cache():
    global cache_data
    if cache_data is None:
        return
    
    cache_dir = osp.dirname(cache_path)
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)

    with open(cache_path, "w", encoding="utf8") as file:
        json.dump(cache_data, file, indent=4)

    global CACHE_UPDATED
    CACHE_UPDATED = False

config_name_to_view_widget = {}
action_to_view_config_name = {}
register_view_widget: lambda *args, **kwargs: None

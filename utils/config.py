import json, os, traceback
import os.path as osp

from . import shared
from .fontformat import FontFormat
from .structures import Tuple, Union, List, Dict, Config, field, nested_dataclass
from .logger import logger as LOGGER
from .io_utils import json_dump_nested_obj


@nested_dataclass
class ModuleConfig(Config):
    textdetector: str = 'ctd'
    ocr: str = "mit48px"
    inpainter: str = 'lama_large_512px'
    translator: str = "google"
    enable_detect: bool = True
    enable_ocr: bool = True
    enable_translate: bool = True
    enable_inpaint: bool = True
    textdetector_params: Dict = field(default_factory=lambda: dict())
    ocr_params: Dict = field(default_factory=lambda: dict())
    translator_params: Dict = field(default_factory=lambda: dict())
    inpainter_params: Dict = field(default_factory=lambda: dict())
    translate_source: str = '日本語'
    translate_target: str = '简体中文'
    check_need_inpaint: bool = True
    load_model_on_demand: bool = False
    empty_runcache: bool = False

    def get_params(self, module_key: str) -> dict:
        return self[module_key + '_params']
    
    def stage_enabled(self, idx: int):
        if idx == 0:
            return self.enable_detect
        elif idx == 1:
            return self.enable_ocr
        elif idx == 2:
            return self.enable_translate
        elif idx == 3:
            return self.enable_inpaint
        else:
            raise Exception(f'not supported stage idx: {idx}')
        
    def all_stages_disabled(self):
        return (self.enable_detect or self.enable_ocr or self.enable_translate or self.enable_inpaint) is False
        

@nested_dataclass
class DrawPanelConfig(Config):
    pentool_color: List = field(default_factory=lambda: [0, 0, 0])
    pentool_width: float = 30.
    pentool_shape: int = 0
    inpainter_width: float = 30.
    inpainter_shape: int = 0
    current_tool: int = 0
    rectool_auto: bool = False
    rectool_method: int = 0
    recttool_dilate_ksize: int = 0

@nested_dataclass
class ProgramConfig(Config):

    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    drawpanel: DrawPanelConfig = field(default_factory=lambda: DrawPanelConfig())
    global_fontformat: FontFormat = field(default_factory=lambda: FontFormat())
    recent_proj_list: List = field(default_factory=lambda: list())
    imgtrans_paintmode: bool = False
    imgtrans_textedit: bool = True
    imgtrans_textblock: bool = True
    mask_transparency: float = 0.
    original_transparency: float = 0.
    open_recent_on_startup: bool = True 
    let_fntsize_flag: int = 0
    let_fntstroke_flag: int = 0
    let_fntcolor_flag: int = 0
    let_fnt_scolor_flag: int = 0
    let_fnteffect_flag: int = 1
    let_alignment_flag: int = 0
    let_writing_mode_flag: int = 0
    let_family_flag: int = 0
    let_autolayout_flag: bool = True
    let_autolayout_adaptive_fntsz: bool = True
    let_uppercase_flag: bool = True
    let_textstyle_indep_flag: bool = False
    text_styles_path: str = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')
    expand_tstyle_panel: bool = True
    fsearch_case: bool = False
    fsearch_whole_word: bool = False
    fsearch_regex: bool = False
    fsearch_range: int = 0
    gsearch_case: bool = False
    gsearch_whole_word: bool = False
    gsearch_regex: bool = False
    gsearch_range: int = 0
    darkmode: bool = False
    textselect_mini_menu: bool = True
    fold_textarea: bool = False
    show_source_text: bool = True
    show_trans_text: bool = True
    saladict_shortcut: str = "Alt+S"
    search_url: str = "https://www.google.com/search?q="
    ocr_sublist: List = field(default_factory=lambda: list())
    restore_ocr_empty: bool = False
    mt_sublist: List = field(default_factory=lambda: list())
    display_lang: str = field(default_factory=lambda: shared.DEFAULT_DISPLAY_LANG) # to always apply shared.DEFAULT_DISPLAY_LANG
    imgsave_quality: int = 100
    imgsave_ext: str = '.png'
    show_text_style_preset: bool = True

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
    

pcfg: ProgramConfig = None
text_styles: List[FontFormat] = []
active_format: FontFormat = None

def load_textstyle_from(p: str, raise_exception = False):

    if not osp.exists(p):
        LOGGER.warning(f'Text style {p} does not exist.')
        return

    try:
        with open(p, 'r', encoding='utf8') as f:
            style_list = json.loads(f.read())
            styles_loaded = []
            for style in style_list:
                try:
                    styles_loaded.append(FontFormat(**style))
                except Exception as e:
                    LOGGER.warning(f'Skip invalid text style: {style}')
    except Exception as e:
        LOGGER.error(f'Failed to load text style from {p}: {e}')
        if raise_exception:
            raise e
        return

    global text_styles, pcfg
    if len(text_styles) > 0:
        text_styles.clear()
    text_styles.extend(styles_loaded)
    pcfg.text_styles_path = p

def load_config():

    if osp.exists(shared.CONFIG_PATH):
        try:
            config = ProgramConfig.load(shared.CONFIG_PATH)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            config = ProgramConfig()
    else:
        LOGGER.info(f'{shared.CONFIG_PATH} does not exist, new config file will be created.')
        config = ProgramConfig()
    
    global pcfg
    pcfg = config

    p = pcfg.text_styles_path
    if not osp.exists(pcfg.text_styles_path):
        dp = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')
        if p != dp and osp.exists(dp):
            p = dp
            LOGGER.warning(f'Text style {p} does not exist, use the default from {dp}.')
        else:
            with open(dp, 'w', encoding='utf8') as f:
                f.write(json.dumps([],  ensure_ascii=False))
            LOGGER.info(f'New text style file created at {dp}.')
    load_textstyle_from(p)

def save_config():
    global pcfg
    try:
        with open(shared.CONFIG_PATH, 'w', encoding='utf8') as f:
            f.write(json_dump_nested_obj(pcfg))
        LOGGER.info('Config saved')
        return True
    except Exception as e:
        LOGGER.error(f'Failed save config to {shared.CONFIG_PATH}: {e}')
        LOGGER.error(traceback.format_exc())
        return False

def save_text_styles(raise_exception = False):
    global pcfg, text_styles
    try:
        style_dir = osp.dirname(pcfg.text_styles_path)
        if not osp.exists(style_dir):
            os.makedirs(style_dir)
        with open(pcfg.text_styles_path, 'w', encoding='utf8') as f:
            f.write(json_dump_nested_obj(text_styles))
        LOGGER.info('Text style saved')
        return True
    except Exception as e:
        LOGGER.error(f'Failed save text style to {pcfg.text_styles_path}: {e}')
        LOGGER.error(traceback.format_exc())
        if raise_exception:
            raise e
        return False
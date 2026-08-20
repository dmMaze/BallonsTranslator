import json, os, string, traceback
import os.path as osp
import copy
from dataclasses import fields
from typing import Callable, Optional

from . import shared
from .fontformat import FontFormat
from .structures import List, Dict, Config, field, nested_dataclass
from .logger import logger as LOGGER
from .io_utils import json_dump_nested_obj, np, serialize_np
from .llm_profiles import default_profiles, load_profiles, migrate_module_llm_profiles, profile_by_id, profile_to_dict, LLMProfile
from .secret_store import SecretStore

class RunStatus:
    FIN_DET = 1
    FIN_OCR = 2
    FIN_INPAINT = 4
    FIN_TRANSLATE = 8
    FIN_ALL = 15


class TranslateContext:
    """Canonical translation grouping values stored in module config.

    >>> TranslateContext.Page
    'page'
    """

    TextBlock = 'textblock'
    Page = 'page'
    Valid = (TextBlock, Page)


class LLMTranslateContext:
    """Canonical LLM translation-context modes stored in module config.

    >>> LLMTranslateContext.HISTORY
    'history'
    """

    PAGE = 'page'
    HISTORY = 'history'
    Valid = (PAGE, HISTORY)


class LLMGlossaryMode:
    """Canonical glossary selection modes stored in module config.

    >>> LLMGlossaryMode.Matching
    'matching'
    """

    Matching = 'matching'
    All = 'all'
    Valid = (Matching, All)


class OCRTextPostprocess:
    """Canonical OCR text postprocessing modes stored in module config.

    >>> OCRTextPostprocess.CAPITALIZE
    'capitalize'
    """

    NONE = 'none'
    CAPITALIZE = 'capitalize'
    UPPERCASE = 'uppercase'
    Valid = (NONE, CAPITALIZE, UPPERCASE)


@nested_dataclass
class ModuleConfig(Config):
    textdetector: str = 'ctd'
    ocr: str = "mit48px"
    inpainter: str = 'lama_large_512px'
    translator: str = "google"
    enable_detect: bool = True
    keep_exist_textlines: bool = False
    filter_mask_by_bboxes: bool = False
    enable_ocr: bool = True
    enable_translate: bool = True
    enable_inpaint: bool = True
    # 是否在 OCR 后进行字体检测（默认不启用）
    ocr_font_detect: bool = False
    ocr_text_postprocess: str = OCRTextPostprocess.NONE
    textdetector_params: Dict = field(default_factory=lambda: dict())
    ocr_params: Dict = field(default_factory=lambda: dict())
    translator_params: Dict = field(default_factory=lambda: dict())
    llm_profiles: List[LLMProfile] = field(default_factory=lambda: list())
    translator_llm_id: str = ''
    ocr_llm_id: str = ''
    inpaint_llm_id: str = ''
    inpainter_params: Dict = field(default_factory=lambda: dict())
    translate_source: str = '日本語'
    translate_target: str = '简体中文'
    translate_context: str = TranslateContext.Page
    llm_translate_context: str = LLMTranslateContext.PAGE
    llm_prior_context_token_budget: int = 4096
    llm_glossary_path: str = ''
    llm_glossary_mode: str = LLMGlossaryMode.Matching

    check_need_inpaint: bool = True
    empty_runcache: bool = False
    finish_code: int = 15

    def get_params(self, module_key: str, for_saving=False) -> dict:
        d = self[module_key + '_params']
        if not for_saving:
            return d
        sd = {}
        for module_key, module_params in d.items():
            if module_params is None:
                continue
            saving_module_params = {}
            sd[module_key] = saving_module_params
            for pk, pv in module_params.items():
                if pk in {'description'}:
                    continue
                if pk.startswith('__'):
                    continue
                if isinstance(pv, dict) and 'value' in pv:
                    # UI param metadata stores the saved value under "value";
                    # plain dict params are already the user's persisted value.
                    pv = pv['value']
                saving_module_params[pk] = pv
        return sd

    def get_saving_params(self, to_dict=True):
        params = copy.copy(self)
        params.ocr_params = self.get_params('ocr', for_saving=True)
        params.inpainter_params = self.get_params('inpainter', for_saving=True)
        params.textdetector_params = self.get_params('textdetector', for_saving=True)
        params.translator_params = self.get_params('translator', for_saving=True)
        params.llm_profiles = self.get_saving_llm_profiles()
        if to_dict:
            return params.__dict__
        return params

    def get_saving_llm_profiles(self):
        profiles = []
        secret_store = SecretStore()
        for profile in self.llm_profiles:
            saving_profile = profile_to_dict(profile)
            if 'api_key' in saving_profile:
                saving_profile['api_key'] = secret_store.prepare_for_save(
                    saving_profile.get('id', ''),
                    saving_profile.get('api_key', ''),
                )
            profiles.append(saving_profile)
        return profiles
    
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

    def set_stage_enabled(self, idx: int, enabled: bool):
        stage_attrs = (
            'enable_detect',
            'enable_ocr',
            'enable_translate',
            'enable_inpaint',
        )
        if idx < 0 or idx >= len(stage_attrs):
            raise Exception(f'not supported stage idx: {idx}')
        stage_attr = stage_attrs[idx]
        setattr(self, stage_attr, bool(enabled))
        self.update_finish_code()
        
    def all_stages_disabled(self):
        return (self.enable_detect or self.enable_ocr or self.enable_translate or self.enable_inpaint) is False

    def __post_init__(self):
        if self.ocr_text_postprocess not in OCRTextPostprocess.Valid:
            self.ocr_text_postprocess = OCRTextPostprocess.NONE
        if self.translate_context not in TranslateContext.Valid:
            self.translate_context = TranslateContext.Page
        if self.llm_translate_context not in LLMTranslateContext.Valid:
            self.llm_translate_context = LLMTranslateContext.PAGE
        if not isinstance(self.llm_glossary_path, str):
            self.llm_glossary_path = ''
        if self.llm_glossary_mode not in LLMGlossaryMode.Valid:
            self.llm_glossary_mode = LLMGlossaryMode.Matching
        if (
            not isinstance(self.llm_prior_context_token_budget, int)
            or isinstance(self.llm_prior_context_token_budget, bool)
            or self.llm_prior_context_token_budget <= 0
        ):
            self.llm_prior_context_token_budget = 4096
        if not self.llm_profiles:
            self.llm_profiles = default_profiles()
        else:
            self.llm_profiles = load_profiles(self.llm_profiles)
        if (not self.translator_llm_id or not profile_by_id(self.llm_profiles, self.translator_llm_id)) and self.llm_profiles:
            self.translator_llm_id = self.llm_profiles[0].id
        if (not self.ocr_llm_id or not profile_by_id(self.llm_profiles, self.ocr_llm_id)) and self.llm_profiles:
            self.ocr_llm_id = self.llm_profiles[0].id
        if (not self.inpaint_llm_id or not profile_by_id(self.llm_profiles, self.inpaint_llm_id)) and self.llm_profiles:
            self.inpaint_llm_id = self.llm_profiles[0].id
        self.update_finish_code()

    def update_finish_code(self):
        self.finish_code = self.enable_detect * RunStatus.FIN_DET + \
            self.enable_ocr * RunStatus.FIN_OCR + \
                self.enable_translate * RunStatus.FIN_TRANSLATE + \
                    self.enable_inpaint * RunStatus.FIN_INPAINT
        

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
    recttool_dilate_ksize: int = 2

@nested_dataclass
class PackageManagerConfig(Config):
    auto_install_missing_packages: bool = True
    installer_backend: str = 'auto'
    extra_install_args: str = ''

@nested_dataclass
class NetworkMirrorsConfig(Config):
    huggingface: Optional[str] = None
    pypi: Optional[str] = None


@nested_dataclass
class AutoTateChuYokoConfig(Config):
    """Settings reserved for automatic tate-chu-yoko detection.

    >>> AutoTateChuYokoConfig().enabled
    False
    """

    enabled: bool = False
    max_length: int = 4
    include_numbers: bool = True
    include_letters: bool = False
    additional_chars: str = ''

    def allowed_characters(self) -> frozenset[str]:
        """Return the configured character categories as one lookup set.

        >>> AutoTateChuYokoConfig(include_letters=True).allowed_characters() >= {'A', 'z'}
        True
        """
        characters = set(self.additional_chars)
        if self.include_numbers:
            characters.update(string.digits)
        if self.include_letters:
            characters.update(string.ascii_letters)
        return frozenset(characters)

    def __post_init__(self) -> None:
        for setting in fields(self):
            value = getattr(self, setting.name)
            valid = type(value) is setting.type
            if setting.name == 'max_length':
                valid = valid and 1 <= value <= 99
            if not valid:
                LOGGER.warning(
                    f'Discard invalid auto_tate_chu_yoko.{setting.name} config.'
                )
                setattr(self, setting.name, setting.default)


@nested_dataclass
class ProgramConfig(Config):

    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    package_manager: PackageManagerConfig = field(default_factory=lambda: PackageManagerConfig())
    mirrors: NetworkMirrorsConfig = field(default_factory=lambda: NetworkMirrorsConfig())
    drawpanel: DrawPanelConfig = field(default_factory=lambda: DrawPanelConfig())
    auto_tate_chu_yoko: AutoTateChuYokoConfig = field(default_factory=AutoTateChuYokoConfig)
    compact_vertical_punctuation_spacing: bool = True
    quick_insert_characters: str = '『』「」♥♡★☆※♩♬'
    global_fontformat: FontFormat = field(default_factory=lambda: FontFormat())
    recent_proj_list: List = field(default_factory=lambda: list())
    show_page_list: bool = False
    imgtrans_paintmode: bool = False
    imgtrans_textedit: bool = True
    imgtrans_textblock: bool = True
    mask_transparency: float = 0.
    original_transparency: float = 0.
    open_recent_on_startup: bool = True 
    check_update_on_startup: bool = True
    spellcheck_enabled: bool = False
    spellcheck_external_dict_path: str = ""
    spellcheck_repo_dicts: str = ""
    spellcheck_distance: int = 1
    spellcheck_on_source_enabled: bool = False
    show_textdetector_tool: bool = True
    show_ocr_tool: bool = True
    show_translator_tool: bool = True
    show_inpainter_tool: bool = True
    run_pipeline_mode: str = 'pipeline'
    render_without_text_style_update: bool = False

    let_fntsize_flag: int = 0
    let_fntstroke_flag: int = 0
    let_fntcolor_flag: int = 0
    let_fnt_scolor_flag: int = 0
    let_fnteffect_flag: int = 1
    let_alignment_flag: int = 0
    let_writing_mode_flag: int = 0
    let_family_flag: int = 0
    let_autolayout_flag: bool = True
    let_letter_case: str = OCRTextPostprocess.NONE
    let_show_only_custom_fonts_flag: bool = False
    let_textstyle_indep_flag: bool = False
    text_styles_path: str = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')

    fsearch_case: bool = False
    fsearch_whole_word: bool = False
    fsearch_regex: bool = False
    fsearch_range: int = 0
    gsearch_case: bool = False
    gsearch_whole_word: bool = False
    gsearch_regex: bool = False
    gsearch_range: int = 0

    darkmode: bool = False
    fold_textarea: bool = False
    show_source_text: bool = True
    show_trans_text: bool = True
    ocr_sublist: List = field(default_factory=lambda: list())
    restore_ocr_empty: bool = False
    pre_mt_sublist: List = field(default_factory=lambda: list())
    mt_sublist: List = field(default_factory=lambda: list())
    display_lang: str = field(default_factory=lambda: shared.DEFAULT_DISPLAY_LANG) # to always apply shared.DEFAULT_DISPLAY_LANG
    imgsave_quality: int = 100
    imgsave_ext: str = '.png'
    intermediate_imgsave_ext: str = '.png'
    show_text_style_preset: bool = True
    expand_tstyle_panel: bool = True
    show_text_effect_panel: bool = True
    expand_teffect_panel: bool = True
    text_advanced_format_panel: bool = True
    expand_tadvanced_panel: bool = True
    text_transform_panel: bool = True
    expand_ttransform_panel: bool = True
    excluded_fonts: List[str] = field(default_factory=list)

    @staticmethod
    def load(cfg_path: str):
        
        with open(cfg_path, 'r', encoding='utf8') as f:
            config_dict = json.loads(f.read())

        if not isinstance(config_dict.get('quick_insert_characters', ''), str):
            LOGGER.warning(
                'Discard invalid quick_insert_characters config: expected a string.'
            )
            config_dict.pop('quick_insert_characters')

        if 'excluded_fonts' in config_dict:
            excluded_fonts = config_dict['excluded_fonts']
            if not isinstance(excluded_fonts, list):
                LOGGER.warning(
                    'Discard invalid excluded_fonts config: expected a list of font names.'
                )
                config_dict.pop('excluded_fonts')
            else:
                normalized_fonts = sorted(
                    {
                        font
                        for font in excluded_fonts
                        if isinstance(font, str) and font.strip()
                    },
                    key=str.casefold,
                )
                if len(normalized_fonts) != len(excluded_fonts):
                    LOGGER.warning(
                        'Discard invalid or duplicate entries in excluded_fonts config.'
                    )
                config_dict['excluded_fonts'] = normalized_fonts

        if 'module' in config_dict:
            module_cfg = config_dict['module']
            if 'translate_context' not in module_cfg and 'translate_by_textblock' in module_cfg:
                module_cfg['translate_context'] = (
                    TranslateContext.TextBlock
                    if module_cfg['translate_by_textblock']
                    else TranslateContext.Page
                )
            if module_cfg.get('textdetector') == 'rtdetr_v2':
                module_cfg['textdetector'] = 'ctbd'
            if 'textdetector_params' in module_cfg:
                params = module_cfg['textdetector_params']
                if 'rtdetr_v2' in params:
                    params['ctbd'] = params.pop('rtdetr_v2')
            # LLM translator keys must be consumed before module-param patching drops unknown keys.
            migrate_module_llm_profiles(module_cfg)

        return ProgramConfig(**config_dict)
    

pcfg = ProgramConfig()
text_styles: List[FontFormat] = []
active_format: FontFormat = None
config_created_on_load = False

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

def load_config(config_path: str = None):
    global config_created_on_load
    config_created_on_load = False
    if config_path is None:
        config_path = shared.CONFIG_PATH
    if config_path != shared.CONFIG_PATH:
        shared.CONFIG_PATH = config_path
        LOGGER.info(f'Using specified config file at {shared.CONFIG_PATH}')

    if osp.exists(shared.CONFIG_PATH):
        try:
            config = ProgramConfig.load(shared.CONFIG_PATH)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            config = ProgramConfig()
            config_created_on_load = True
    else:
        LOGGER.info(f'{shared.CONFIG_PATH} does not exist, new config file will be created.')
        config = ProgramConfig()
        config_created_on_load = True
    
    global pcfg
    pcfg.merge(config)

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


def json_dump_program_config(obj, **kwargs):
    def _default(obj):
        if isinstance(obj, (np.ndarray, np.ScalarType)):
            return serialize_np(obj)
        elif isinstance(obj, ModuleConfig):
            return obj.get_saving_params()
        serializer = getattr(obj, 'to_serializable_dict', None)
        if serializer is not None:
            return serializer()
        return obj.__dict__
    return json.dumps(obj, default=lambda o: _default(o), ensure_ascii=False, **kwargs)


def save_config():
    global pcfg
    try:
        config_dir = osp.dirname(shared.CONFIG_PATH)
        if config_dir and not osp.exists(config_dir):
            os.makedirs(config_dir)
        tmp_save_tgt = shared.CONFIG_PATH + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            f.write(json_dump_program_config(pcfg))
    except Exception as e:
        LOGGER.error(f'Failed save config to {tmp_save_tgt}: {e}')
        LOGGER.error(traceback.format_exc())
        return False
    
    os.replace(tmp_save_tgt, shared.CONFIG_PATH)
    LOGGER.debug('Config saved')
    return True

def save_text_styles(raise_exception = False):
    global pcfg, text_styles
    try:
        style_dir = osp.dirname(pcfg.text_styles_path)
        if not osp.exists(style_dir):
            os.makedirs(style_dir)
        tmp_save_tgt = pcfg.text_styles_path + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            f.write(json_dump_nested_obj(text_styles))

    except Exception as e:
        LOGGER.error(f'Failed save text style to {tmp_save_tgt}: {e}')
        LOGGER.error(traceback.format_exc())
        if raise_exception:
            raise e
        return False

    os.replace(tmp_save_tgt, pcfg.text_styles_path)
    LOGGER.info('Text style saved')
    return True

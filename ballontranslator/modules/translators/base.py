import urllib.request
from ordered_set import OrderedSet
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Set, Union
import time, requests, re, uuid, base64, hmac, functools, json, copy

from .exceptions import InvalidSourceOrTargetLanguage, TranslatorSetupFailure, MissingTranslatorParams, TranslatorNotValid
from ballontranslator.utils.textblock import TextBlock
from ..base import BaseModule, DEVICE_SELECTOR
from ballontranslator.utils.registry import Registry
from ballontranslator.utils.io_utils import text_is_empty
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.config import OCRTextPostprocess, TranslateContext, pcfg
from ballontranslator.utils.text_processing import (
    finalize_translation_text,
    substitute_keywords,
)

if TYPE_CHECKING:
    from ballontranslator.utils.proj_imgtrans import ProjImgTrans


TRANSLATORS = Registry('translators')
register_translator = TRANSLATORS.register_module

PROXY = urllib.request.getproxies()

LANGMAP_GLOBAL = {
    'Auto': '',
    '简体中文': '',
    '繁體中文': '',
    '日本語': '',
    'English': '',
    '한국어': '',
    'Tiếng Việt': '',
    'čeština': '',
    'Nederlands': '',
    'Français': '',
    'Deutsch': '',
    'magyar nyelv': '',
    'Italiano': '',
    'Polski': '',
    'Português': '',
    'Brazilian Portuguese': '',
    'limba română': '',
    'русский язык': '',
    'Español': '',
    'Türk dili': '',
    'украї́нська мо́ва': '',  
    'Thai': '',
    'Arabic': '',
    'Hindi': '',
    'Malayalam': '',
    'Tamil': '',
}

SYSTEM_LANG = ''
SYSTEM_LANGMAP = {
    'zh-CN': '简体中文'        
}

_CHS2CHT_CONVERTER = None


def preprocess_translation_text(
    text: str,
    substitutions: Sequence[Mapping],
) -> str:
    """Apply source substitutions before sending text to a translator.

    >>> rules = [{'keyword': 'Hero', 'sub': 'Champion', 'use_reg': False,
    ...           'case_sens': True}]
    >>> preprocess_translation_text('Hero returns', rules)
    'Champion returns'
    """

    return substitute_keywords(text, substitutions)


def postprocess_translation_text(
    text: str,
    source_language: str,
    target_language: str,
    substitutions: Sequence[Mapping],
    *,
    letter_case: str = OCRTextPostprocess.NONE,
    convert_to_traditional: bool = False,
    full_page: bool = False,
) -> str:
    """Apply the fixed translation finalization order for one result.

    Full-page processing applies normalization, substitution, then letter case.
    Selected-block translation retains its historical substitution-only behavior.

    >>> rules = [{'keyword': 'A', 'sub': 'X', 'use_reg': False,
    ...           'case_sens': True}]
    >>> postprocess_translation_text(
    ...     'Ａ', 'English', 'English', rules, full_page=True)
    'X'
    >>> postprocess_translation_text(
    ...     'Ａ', 'English', 'English', rules, full_page=False)
    'Ａ'
    """

    if convert_to_traditional and target_language == '繁體中文':
        global _CHS2CHT_CONVERTER
        if _CHS2CHT_CONVERTER is None:
            import opencc

            _CHS2CHT_CONVERTER = opencc.OpenCC('s2t')
        text = _CHS2CHT_CONVERTER.convert(text)

    if full_page:
        return finalize_translation_text(
            text,
            source_language,
            target_language,
            substitute=lambda value: substitute_keywords(value, substitutions),
            letter_case=letter_case,
        )
    return substitute_keywords(text, substitutions)


def check_language_support(check_type: str = 'source'):
    
    def decorator(set_lang_method):
        @functools.wraps(set_lang_method)
        def wrapper(self, lang: str = ''):
            if check_type == 'source':
                supported_lang_list = self.supported_src_list
            else:
                supported_lang_list = self.supported_tgt_list
            if not lang in supported_lang_list:
                msg = '\n'.join(supported_lang_list)
                raise InvalidSourceOrTargetLanguage(f'Invalid {check_type}: {lang}\n', message=msg)
            return set_lang_method(self, lang)
        return wrapper

    return decorator


class BaseTranslator(BaseModule):

    concate_text = True
    cht_require_convert = False

    def __init__(self,
                 lang_source: str, 
                 lang_target: str,
                 raise_unsupported_lang: bool = True,
                 **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in TRANSLATORS.module_dict:
            if TRANSLATORS.module_dict[key] == self.__class__:
                self.name = key
                break
        self.textblk_break = '\n##\n'
        self.lang_source: str = lang_source
        self.lang_target: str = lang_target
        self.lang_map: Dict = LANGMAP_GLOBAL.copy()
        
        try:
            self.setup_translator()
        except Exception as e:
            if isinstance(e, MissingTranslatorParams):
                raise e
            else:
                raise TranslatorSetupFailure(e)
            
        # enable traditional chinese by converting from simplified chinese
        if self.cht_require_convert and not self.lang_map['繁體中文']:
            self.lang_map['繁體中文'] = self.lang_map['简体中文']

        self.valid_lang_list = [lang for lang in self.lang_map if self.lang_map[lang] != '']

        try:
            self.set_source(lang_source)
            self.set_target(lang_target)
        except InvalidSourceOrTargetLanguage as e:
            if raise_unsupported_lang:
                raise e
            else:
                lang_source = self.supported_src_list[0]
                lang_target = self.supported_tgt_list[0]
                self.set_source(lang_source)
                self.set_target(lang_target)

    def _setup_translator(self):
        raise NotImplementedError

    def setup_translator(self):
        self._setup_translator()

    @check_language_support(check_type='source')
    def set_source(self, lang: str):
        self.lang_source = lang

    @check_language_support(check_type='target')
    def set_target(self, lang: str):
        self.lang_target = lang

    def _translate(self, src_list: List[str]) -> List[str]:
        raise NotImplementedError

    def translate(
        self,
        text: Union[str, List],
        *,
        project: Optional['ProjImgTrans'] = None,
        page_key: Optional[str] = None,
        commit_history_window: bool = False,
    ) -> Union[str, List]:
        """Translate text while accepting optional page context from the UI boundary.

        Base translators intentionally ignore the context keywords. Third-party
        translators that override this public method should accept the same keywords.

        >>> TransSource('日本語', 'English').translate('text', page_key='001.png')
        'text'
        """
        if text_is_empty(text):
            return text
        if not self.all_model_loaded():
            self.load_model()

        is_list = isinstance(text, List)
        concate_text = (
            is_list
            and self.concate_text
            and pcfg.module.translate_context == TranslateContext.Page
        )
        text_source = self.textlist2text(text) if concate_text else text
        
        src_is_list = isinstance(text_source, List)
        if src_is_list: 
            text_trans = self._translate(text_source)
        else:
            text_trans = self._translate([text_source])[0]
        
        if text_trans is None:
            if is_list:
                text_trans = [''] * len(text)
            else:
                text_trans = ''
        elif concate_text:
            text_trans = self.text2textlist(text_trans)
            
        if is_list:
            try:
                assert len(text_trans) == len(text)
            except:
                LOGGER.error('This translator seems to messed up the translation which resulted in inconsistent translated line count.\n \
                             Set concate_text to False or change textblk_break in the source code may solve the problem.')
                raise

        return text_trans

    def textlist2text(self, text_list: List[str]) -> str:
        # some translators automatically strip '\n'
        # so we insert '\n###\n' between concated text instead of '\n' to avoid mismatch
        return self.textblk_break.join(text_list)

    def text2textlist(self, text: str) -> List[str]:
        breaker = self.textblk_break.replace('\n', '') or '\n'
        text_list = text.split(breaker)
        return [text.lstrip().rstrip() for text in text_list]

    def translate_textblk_lst(
        self,
        textblk_lst: List[TextBlock],
        *,
        project: Optional['ProjImgTrans'] = None,
        page_key: Optional[str] = None,
        full_page: bool = False,
    ):
        """Translate non-empty blocks and apply the fixed finalization rules."""
        non_empty_ids, text_list, translations = (
            BaseTranslator._prepare_textblock_sources(self, textblk_lst)
        )

        if len(text_list) > 0:
            commit_history_window = full_page
            if (
                not commit_history_window
                and project is not None
                and page_key is not None
            ):
                pages = getattr(project, 'pages', None)
                page = (
                    pages.get(page_key)
                    if isinstance(pages, Mapping)
                    else None
                )
                if page is not None:
                    # A selected request represents a page only when it includes
                    # every block that has source text.
                    selected_blocks = {id(block) for block in textblk_lst}
                    commit_history_window = all(
                        not block.get_text().strip()
                        or id(block) in selected_blocks
                        for block in page
                    )
            _translations = self.translate(
                text_list,
                project=project,
                page_key=page_key,
                commit_history_window=commit_history_window,
            )
            for ii, idx in enumerate(non_empty_ids):
                translations[idx] = _translations[ii]

        translations = [
            postprocess_translation_text(
                translation,
                self.lang_source,
                self.lang_target,
                pcfg.mt_sublist,
                letter_case=pcfg.let_letter_case,
                convert_to_traditional=self.cht_require_convert,
                full_page=full_page,
            )
            for translation in translations
        ]

        for tr, blk in zip(translations, textblk_lst):
            blk.translation = tr

    def _prepare_textblock_sources(
        self,
        textblk_lst: List[TextBlock],
    ):
        """Collect non-empty sources after applying configured substitutions.

        >>> translator = TransSource('日本語', 'English')
        >>> translator._prepare_textblock_sources([TextBlock(text=['text'])])[:2]
        ([0], ['text'])
        """
        non_empty_ids = []
        text_list = []
        translations = []
        for ii, blk in enumerate(textblk_lst):
            text = blk.get_text()
            if text.strip() != '':
                non_empty_ids.append(ii)
                text_list.append(
                    preprocess_translation_text(text, pcfg.pre_mt_sublist)
                )
            translations.append(text)

        return non_empty_ids, text_list, translations

    def supported_languages(self) -> List[str]:
        return self.valid_lang_list

    @property
    def supported_tgt_list(self) -> List[str]:
        return self.valid_lang_list

    @property
    def supported_src_list(self) -> List[str]:
        return self.valid_lang_list
        
    def delay(self) -> float:
        if 'delay' in self.params:
            delay = self.params['delay']
            if delay:
                try:
                    return float(delay)
                except:
                    pass
        return 0.

@register_translator('Copy Source')
class TransSource(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'description': 'Return source text as translation'
    }

    def _setup_translator(self):
        for k in self.lang_map.keys():
            self.lang_map[k] = 'dummy language'
        
    def _translate(self, src_list: List[str]) -> List[str]:
        return copy.copy(src_list)

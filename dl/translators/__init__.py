import urllib.request
from typing import Dict, List, Union
import time, requests, re, uuid, base64, hmac
import functools
import json
from .exceptions import InvalidSourceOrTargetLanguage, TranslatorSetupFailure, MissingTranslatorParams, TranslatorNotValid
from ..textdetector.textblock import TextBlock
from ..moduleparamparser import ModuleParamParser, DEFAULT_DEVICE
from utils.registry import Registry
from utils.io_utils import text_is_empty

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
    'français': '',
    'Deutsch': '',
    'magyar nyelv': '',
    'italiano': '',
    'polski': '',
    'português': '',
    'limba română': '',
    'русский язык': '',
    'español': '',
    'Türk dili': ''        
}

SYSTEM_LANG = ''
SYSTEM_LANGMAP = {
    'zh-CN': '简体中文'        
}


def check_language_support(set_lang_method):
    @functools.wraps(set_lang_method)
    def wrapper(self, lang: str):
        if not lang in self.lang_map or not self.lang_map[lang]:
            msg = '\n'.join(self.supported_languages())
            raise InvalidSourceOrTargetLanguage(lang, message=msg)
        return set_lang_method(self, lang)
    return wrapper


class TranslatorBase(ModuleParamParser):
    concate_text = True
    def __init__(self,
                 lang_source: str = None, 
                 lang_target: str = None,
                 **setup_params) -> None:
        super().__init__(**setup_params)
        self.sys_lang = SYSTEM_LANGMAP[SYSTEM_LANG] if SYSTEM_LANG in SYSTEM_LANGMAP else ''
        self.name = ''
        for key in TRANSLATORS.module_dict:
            if TRANSLATORS.module_dict[key] == self.__class__:
                self.name = key
                break
        self.textblk_break = '\n###\n'
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

        if lang_source is None or lang_source == '':
            lang_source = 'Auto' if self.support_auto_souce() else self.default_lang()
        if lang_target is None or lang_target == '':
            lang_target = self.default_lang()

        self.set_source(lang_source)
        self.set_target(lang_target)

    def support_auto_souce(self):
        if self.lang_map['Auto']:
            return True
        return False

    def default_lang(self):
        if self.sys_lang in self.lang_map:
            return self.sys_lang
        return self.supported_languages()[0]

    def _setup_translator(self):
        raise NotImplementedError

    def setup_translator(self):
        self._setup_translator()

    @check_language_support
    def set_source(self, lang: str):
        self.lang_source = lang

    @check_language_support
    def set_target(self, lang: str):
        self.lang_target = lang

    def _translate(self, text: Union[str, List]) -> Union[str, List]:
        raise NotImplementedError

    def translate(self, text: Union[str, List]) -> Union[str, List]:
        if text_is_empty(text):
            return text

        concate_text = isinstance(text, List) and self.concate_text
        text_source = self.textlist2text(text) if concate_text else text
        
        text_trans = self._translate(text_source)
        
        if text_trans is None:
            if isinstance(text, List):
                text_trans = [''] * len(text)
            else:
                text_trans = ''
        elif concate_text:
            text_trans = self.text2textlist(text_trans)
            
        if isinstance(text, List):
            assert len(text_trans) == len(text)

        return text_trans

    def textlist2text(self, text_list: List[str]) -> str:
        # some translators automatically strip '\n'
        # so we insert '\n###\n' between concated text instead of '\n' to avoid mismatch
        return self.textblk_break.join(text_list)

    def text2textlist(self, text: str) -> List[str]:
        breaker = self.textblk_break.replace('\n', '') or '\n'
        text_list = text.split(breaker)
        return [text.lstrip().rstrip() for text in text_list]

    def translate_textblk_lst(self, textblk_lst: List[TextBlock]):
        text_list = [blk.get_text() for blk in textblk_lst]
        translations = self.translate(text_list)
        for tr, blk in zip(translations, textblk_lst):
            blk.translation = tr

    def supported_languages(self) -> List[str]:
        return [lang for lang in self.lang_map if self.lang_map[lang]]

    def support_language(self, lang: str) -> bool:
        if lang in self.lang_map:
            if self.lang_map[lang]:
                return True
        return False

@register_translator('google')
class GoogleTranslator(TranslatorBase):

    concate_text = True
    setup_params: Dict = {
        'url': {
            'type': 'selector',
            'options': [
                'https://translate.google.cn/m',
                'https://translate.google.com/m'
            ],
            'select': 'https://translate.google.cn/m'
        }
    }
    
    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh-CN'
        self.lang_map['繁體中文'] = 'zh-TW'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        self.lang_map['한국어'] = 'ko'
        self.lang_map['Tiếng Việt'] = 'vi'
        self.lang_map['čeština'] = 'cs'
        self.lang_map['Nederlands'] = 'nl'
        self.lang_map['français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['magyar nyelv'] = 'hu'
        self.lang_map['italiano'] = 'it'
        self.lang_map['polski'] = 'pl'
        self.lang_map['português'] = 'pt'
        self.lang_map['limba română'] = 'ro'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['español'] = 'es'
        self.lang_map['Türk dili'] = 'tr'

        from .google_trans import GoogleTranslator
        self.googletrans = GoogleTranslator()
        
    def _translate(self, text: Union[str, List]) -> Union[str, List]:
        self.googletrans._source = self.lang_map[self.lang_source]
        self.googletrans._url_params['sl'] = self.lang_map[self.lang_source]
        self.googletrans._target = self.lang_map[self.lang_target]
        self.googletrans._url_params['tl'] = self.lang_map[self.lang_target]
        self.googletrans.__base_url = self.setup_params['url']['select']
        translations = self.googletrans.translate(text)
        return translations


@register_translator('papago')
class PapagoTranslator(TranslatorBase):

    concate_text = True
    setup_params: Dict = None
    papagoVer: str = None

    # https://github.com/zyddnys/manga-image-translator/blob/main/translators/papago.py
    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh-CN'
        self.lang_map['繁體中文'] = 'zh-TW'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        self.lang_map['한국어'] = 'ko'
        self.lang_map['Tiếng Việt'] = 'vi'
        self.lang_map['français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['italiano'] = 'it'
        self.lang_map['português'] = 'pt'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['español'] = 'es'        
        
        if self.papagoVer is None:
            script = requests.get('https://papago.naver.com', proxies=PROXY)
            mainJs = re.search(r'\/(main.*\.js)', script.text).group(1)
            papagoVerData = requests.get('https://papago.naver.com/' + mainJs, proxies=PROXY)
            papagoVer = re.search(r'"PPG .*,"(v[^"]*)', papagoVerData.text).group(1)
            self.papagoVer = PapagoTranslator.papagoVer = papagoVer

    def _translate(self, text: Union[str, List]) -> Union[str, List]:
        data = {}
        data['source'] = self.lang_map[self.lang_source]
        data['target'] = self.lang_map[self.lang_target]
        data['text'] = text
        data['honorific'] = "false"

        PAPAGO_URL = 'https://papago.naver.com/apis/n2mt/translate'
        guid = uuid.uuid4()
        timestamp = int(time.time() * 1000)
        key = self.papagoVer.encode("utf-8")
        code = f"{guid}\n{PAPAGO_URL}\n{timestamp}".encode("utf-8")
        token = base64.b64encode(hmac.new(key, code, "MD5").digest()).decode("utf-8")
        
        headers = {
            "Authorization": f"PPG {guid}:{token}",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Timestamp": str(timestamp),
        }
        resp = requests.post(PAPAGO_URL, data, headers=headers)
        translations = resp.json()['translatedText']
        
        return translations
        


@register_translator('caiyun')
class CaiyunTranslator(TranslatorBase):

    concate_text = False
    setup_params: Dict = {
        'token': '',
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'  
        
    def _translate(self, text: Union[str, List]) -> Union[str, List]:

        url = "http://api.interpreter.caiyunai.com/v1/translator"
        token = self.setup_params['token']
        if token == '' or token is None:
            raise MissingTranslatorParams('token')

        direction = self.lang_map[self.lang_source] + '2' + self.lang_map[self.lang_target]
        payload = {
            "source": text,
            "trans_type": direction,
            "request_id": "demo",
            "detect": True,
        }

        headers = {
            "content-type": "application/json",
            "x-authorization": "token " + token,
        }

        response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
        translations = json.loads(response.text)["target"]

        return translations

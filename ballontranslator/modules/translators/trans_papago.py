from .base import *

@register_translator('Papago')
class PapagoTranslator(BaseTranslator):

    concate_text = True
    params: Dict = {'delay': 0.0}
    papagoVer: str = None

    # https://github.com/zyddnys/manga-image-translator/blob/main/translators/papago.py
    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh-CN'
        self.lang_map['繁體中文'] = 'zh-TW'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        self.lang_map['한국어'] = 'ko'
        self.lang_map['Tiếng Việt'] = 'vi'
        self.lang_map['Français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['Italiano'] = 'it'
        self.lang_map['Português'] = 'pt'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['Español'] = 'es'
        self.lang_map['Arabic'] = 'ar'
        self.lang_map['Malayalam'] = 'ml'
        self.lang_map['Tamil'] = 'ta'
        self.lang_map['Hindi'] = 'hi'        

    def _translate(self, src_list: List[str]) -> List[str]:
        data = {}
        data['source'] = self.lang_map[self.lang_source]
        data['target'] = self.lang_map[self.lang_target]
        data['text'] = src_list[0]
        data['dict'] = "false"
        data['useGlossary'] = "false"
        data['honorific'] = "false"

        PAPAGO_URL = 'https://papago.naver.com/api/text/translation'
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://papago.naver.com",
            "Referer": "https://papago.naver.com/",
        }
        resp = requests.post(PAPAGO_URL, data, headers=headers, proxies=PROXY)
        resp.raise_for_status()
        translations = resp.json()['translatedText']
    
        return [translations]

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
        
        if self.papagoVer is None:
            script = requests.get('https://papago.naver.com', proxies=PROXY)
            mainJs = re.search(r'\/(main.*\.js)', script.text).group(1)
            papagoVerData = requests.get('https://papago.naver.com/' + mainJs, proxies=PROXY)
            papagoVer = re.search(r'"PPG .*,"(v[^"]*)', papagoVerData.text).group(1)
            self.papagoVer = PapagoTranslator.papagoVer = papagoVer

    def _translate(self, src_list: List[str]) -> List[str]:
        data = {}
        data['source'] = self.lang_map[self.lang_source]
        data['target'] = self.lang_map[self.lang_target]
        data['text'] = src_list[0]
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
    
        return [translations]
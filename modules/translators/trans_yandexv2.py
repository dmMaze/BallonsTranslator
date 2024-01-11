from .base import *

@register_translator('Yandexv2')
class YandexTranslatorv2(BaseTranslator):

    concate_text = False
    params: Dict = {
        'api_key': '',
        'delay': 0.0,
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        self.lang_map['한국어'] = 'ko'
        self.lang_map['Tiếng Việt'] = 'vi'
        self.lang_map['čeština'] = 'cs'
        self.lang_map['Nederlands'] = 'nl'
        self.lang_map['Français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['magyar nyelv'] = 'hu'
        self.lang_map['Italiano'] = 'it'
        self.lang_map['Polski'] = 'pl'
        self.lang_map['Português'] = 'pt'
        self.lang_map['limba română'] = 'ro'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['Español'] = 'es'
        self.lang_map['Türk dili'] = 'tr'

        self.api_url = "https://translate.yandex.net/api/v1.5/tr.json/translate"


    def _translate(self, src_list: List[str]) -> List[str]:
        tr_list = []
        for text in src_list:
            params = {
                'key': self.params['api_key'],
                'text': text,
                'lang': self.lang_map[self.lang_target],
                'format': 'plain',
                # 'options': 1, # If additional options are needed
                # 'callback': 'callback_function_name', # If callback is used
            }
            response = requests.get('https://translate.yandex.net/api/v1.5/tr.json/translate', params=params)
            if response.status_code == 200:
                translated_text = response.json().get('text', [''])[0]
                tr_list.append(translated_text)
            else:
                tr_list.append('')  # Or error handling

        return tr_list

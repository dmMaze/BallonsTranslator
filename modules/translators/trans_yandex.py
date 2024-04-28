from .base import *

@register_translator('Yandex')
class YandexTranslator(BaseTranslator):

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
        self.lang_map['Arabic'] = 'ar'
        self.lang_map['Malayalam'] = 'ml'
        self.lang_map['Tamil'] = 'ta'
        self.lang_map['Hindi'] = 'hi'

        self.api_url_v2 = "https://translate.yandex.net/api/v1.5/tr.json/translate"
        self.api_url = 'https://translate.api.cloud.yandex.net/translate/v2/translate'

    def _translate_with_v2(self, src_list: List[str]) -> List[str]:
        tr_list = []
        for text in src_list:
            params = {
                'key': self.params['api_key'],
                'text': text,
                'lang': self.lang_map[self.lang_target],
                'format': 'plain',
            }
            response = requests.get(self.api_url_v2, params=params)
            if response.status_code == 200:
                translated_text = response.json().get('text', [''])[0]
                tr_list.append(translated_text)
            else:
                tr_list.append('')
        return tr_list

    def _translate_with_standard(self, src_list: List[str]) -> List[str]:
        body = {
            "targetLanguageCode": self.lang_map[self.lang_target],
            "texts": src_list,
            "folderId": '',
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Api-Key {0}".format(self.params['api_key'])
        }

        response = requests.post(self.api_url, json=body, headers=headers)
        if response.status_code == 200:
            translations = response.json().get('translations', [])
            tr_list = [tr.get('text', '') for tr in translations]
        else:
            tr_list = [''] * len(src_list)
        return tr_list

    def _translate(self, src_list: List[str]) -> List[str]:
        if self.params['api_key'].startswith("trnsl."):
            return self._translate_with_v2(src_list)
        else:
            return self._translate_with_standard(src_list)

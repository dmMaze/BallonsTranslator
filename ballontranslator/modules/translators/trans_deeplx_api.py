from .base import *
import httpx

@register_translator('DeepLX API')
class DeepLTranslatorv2(BaseTranslator):

    concate_text = False
    params: Dict = {
        'api_url': '',  # EndPoint will be provided by the user
        'delay': 0.0,
    }
# Setup your endpoint api with https://github.com/OwO-Network/DeepLX

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
        # Add other languages here

    def _translate(self, src_list: List[str]) -> List[str]:
        tr_list = []
        for text in src_list:
            data = {
                'text': text,
                'source_lang': 'auto',  # or your source language
                'target_lang': self.lang_map[self.lang_target]
            }

            response = requests.post(self.params['api_url'], json=data)

            if response.status_code == 200:
                # Extract the translated text from the 'data' key
                translated_text = response.json().get('data', '')
                tr_list.append(translated_text)
            else:
                tr_list.append('')  # Or error handling

        return tr_list

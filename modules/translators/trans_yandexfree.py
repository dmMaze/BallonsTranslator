from .base import *

@register_translator('YandexFree')
class YandexTranslatorFree(BaseTranslator):

    concate_text = False
    params: Dict = {
        'endpoint': 'https://translate.toil.cc/translate',  # Service endpoint 
        'delay': 0.0,                                       # Go to https://github.com/FOSWLY/translate-backend to setup your API backend
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
    
    def _translate(self, src_list: List[str]) -> List[str]:
        tr_list = []
        dest_lang_code = self.lang_map[self.lang_target]  # Get the language code from lang_map

        for text in src_list:
            body = {
                "lang": dest_lang_code, 
                "text": text
            }
            response = requests.post(self.params['endpoint'], json=body)
            if response.status_code == 200:
                translated_text = response.json().get('text', [''])[0]
                tr_list.append(translated_text)
            else:
                tr_list.append('')  # Or error handling

        return tr_list
        
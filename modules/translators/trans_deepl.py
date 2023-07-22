from .base import *
import deepl

@register_translator('DeepL')
class DeeplTranslator(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api_key': '',
        'delay': '0.0',
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'EN-US'
        self.lang_map['Français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['Italiano'] = 'it'
        self.lang_map['Português'] = 'pt'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['Español'] = 'es'
        self.lang_map['български език'] = 'bg'
        self.lang_map['Český Jazyk'] = 'cs'
        self.lang_map['Dansk'] = 'da'
        self.lang_map['Ελληνικά'] = 'el'
        self.lang_map['Eesti'] = 'et'
        self.lang_map['Suomi'] = 'fi'
        self.lang_map['Magyar'] = 'hu'
        self.lang_map['Lietuvių'] = 'lt'
        self.lang_map['latviešu'] = 'lv'
        self.lang_map['Nederlands'] = 'nl'
        self.lang_map['Polski'] = 'pl'
        self.lang_map['Română'] = 'ro'
        self.lang_map['Slovenčina'] = 'sk'
        self.lang_map['Slovenščina'] = 'sl'
        self.lang_map['Svenska'] = 'sv'
        self.lang_map['Indonesia'] = 'id'
        
    def _translate(self, src_list: List[str]) -> List[str]:
        api_key = self.params['api_key']
        translator = deepl.Translator(api_key)
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
        if source == 'EN-US':
            source = "EN"
        result = translator.translate_text(src_list, source_lang=source, target_lang=target)
        return [i.text for i in result]

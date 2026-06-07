from .base import *
import deepl


@register_translator('DeepL')
class DeeplTranslator(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api_key': '',
        'delay': 0.0,
        'formality': {
            'type': 'selector',
            'options': [
                'less',
                'more',
                'default',
                'prefer_more',
                'prefer_less'
            ],
            'value': 'default'
        },
        'context': {
            'type': 'editor',
            'value': ''  
        },
        'preserve_formatting': {
            'type': 'selector',
            'options': ['enabled', 'disabled'],
            'value': 'disabled' 
        }
    }


    @property
    def preserve_formatting(self) -> bool:
        return self.params['preserve_formatting']['value'] == 'enabled'

    @property
    def context(self) -> str:
        return self.params['context']['value']

    @property
    def formality(self) -> str:
        return self.params['formality']['value']

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'EN-US'
        self.lang_map['Français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['Italiano'] = 'it'
        self.lang_map['Português'] = 'pt'
        self.lang_map['Brazilian Portuguese'] = 'pt-br'
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
        self.lang_map['украї́нська мо́ва'] = 'uk'
        self.lang_map['한국어'] = 'ko'
        self.lang_map['Arabic'] = 'ar'

    def _translate(self, src_list: List[str]) -> List[str]:
        api_key = self.params['api_key']
        translator = deepl.Translator(api_key)
        formality_selected = self.formality
        context_text = self.context
        preserve_formatting = self.preserve_formatting
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
        if source == 'EN-US':
            source = "EN"

        # Languages that support formality setting in DeepL
        languages_supporting_formality = {'de', 'fr', 'it', 'es', 'nl', 'pl', 'pt', 'pt-br', 'ru', 'ja'}

        # Check if the target language supports formality
        if target in languages_supporting_formality:
            result = translator.translate_text(src_list, source_lang=source, target_lang=target, formality=formality_selected, context=context_text, preserve_formatting=preserve_formatting)
        else:
            result = translator.translate_text(src_list, source_lang=source, target_lang=target, context=context_text, preserve_formatting=preserve_formatting)

        return [i.text for i in result]
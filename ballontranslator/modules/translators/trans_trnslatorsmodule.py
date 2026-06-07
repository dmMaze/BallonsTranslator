from .base import *
import os

os.environ['translators_default_region'] = os.environ.get('translators_default_region', 'EN')

import translators as ts

@register_translator('TranslatorsPack')
class GeneralTranslator(BaseTranslator):
    def __init__(self, lang_source, lang_target, *args, **kwargs):
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.lang_map = {}  
        super().__init__(lang_source, lang_target, *args, **kwargs)  
        self.raise_unsupported_lang = kwargs.get('raise_unsupported_lang', False)
        self._setup_translator()

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
        self.lang_map['Malayalam'] = 'ml'
        self.lang_map['Tamil'] = 'ta'
        self.lang_map['Hindi'] = 'hi'

    translator_options = ts.translators_pool

    params: Dict = {
        'translator provider': {
            'type': 'selector',
            'options': ts.translators_pool,
            'value': 'bing'  
        },
        'sleep_seconds': 0  
    }

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        for text in src_list:
            if not text:
                translations.append("Translation error or empty text")
                continue

            try:
                translator = self.params['translator']['value']
                source_language = self.lang_map.get(self.lang_source, 'auto')
                target_language = self.lang_map.get(self.lang_target, 'en')

                translated_text = ts.translate_text(
                    query_text=text,
                    translator=translator,
                    from_language=source_language,
                    to_language=target_language,
                    sleep_seconds=self.params['sleep_seconds']
                )
                translations.append(translated_text)
            except Exception as e:
                error_message = str(e)
                if "has been not certified yet" in error_message:
                    print("The translation service is temporarily unavailable. Send logs @bropines")
                    print(f"{e}")
                    translations.append("")
                else:
                    print(f"Error when translating text(send logs from console @bropines in issue on github page https://github.com/dmMaze/BallonsTranslator): {e}")
                    translations.append("Translation error")

        return translations

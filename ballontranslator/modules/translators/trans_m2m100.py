from .base import *
import ctranslate2, sentencepiece as spm
import transformers

CT_MODEL_PATH = 'data/models/m2m100-1.2B-ctranslate2'

@register_translator('m2m100')
class M2M100Translator(BaseTranslator):

    dependencies = ['ctranslate2', 'sentencepiece', 'transformers==4.57.6']

    concate_text = False
    _load_model_keys = {'translator', 'tokenizer'}
    params: Dict = {
        'device': DEVICE_SELECTOR()
    }

    download_file_list = [{
        'url': 'https://huggingface.co/entai2965/m2m100-1.2B-ctranslate2/resolve/main/',
        'files': [
            'config.json',
            'model.bin',
            'sentencepiece.bpe.model',
            'special_tokens_map.json',
            'shared_vocabulary.json',
            'vocab.json',
        ],
        'sha256_pre_calculated': [
            None,
            '1d5ddc1a1049de15acba4f985e23b64d3795b3c435e9b2f80031df93cb893aa5',
            'd8f7c76ed2a5e0822be39f0a4f95a55eb19c78f4593ce609e2edbc2aea4d380a',
            None, None, None
        ],
        'save_dir': CT_MODEL_PATH,
        'concatenate_url_filename': 1,
    }]

    def _setup_translator(self):
        self.device = self.params['device']['value']
        self.lang_map['Afrikaans'] = 'af'
        self.lang_map['Albanian'] = 'sq'
        self.lang_map['Amharic'] = 'am'
        self.lang_map['Arabic'] = 'ar'
        self.lang_map['Armenian'] = 'hy'
        self.lang_map['Asturian'] = 'ast'
        self.lang_map['Azerbaijani'] = 'az'
        self.lang_map['Bashkir'] = 'ba'
        self.lang_map['Belarusian'] = 'be'
        self.lang_map['Bengali'] = 'bn'
        self.lang_map['Bosnian'] = 'bs'
        self.lang_map['Breton'] = 'br'
        self.lang_map['Bulgarian'] = 'bg'
        self.lang_map['Burmese'] = 'my'
        self.lang_map['Catalan'] = 'ca'
        self.lang_map['Cebuano'] = 'ceb'
        self.lang_map['Central Khmer'] = 'km'
        self.lang_map['Chinese'] = 'zh'
        self.lang_map['Croatian'] = 'hr'
        self.lang_map['Czech'] = 'cs'
        self.lang_map['Danish'] = 'da'
        self.lang_map['Dutch'] = 'nl'
        self.lang_map['English'] = 'en'
        self.lang_map['Estonian'] = 'et'
        self.lang_map['Finnish'] = 'fi'
        self.lang_map['French'] = 'fr'
        self.lang_map['Fulah'] = 'ff'
        self.lang_map['Gaelic'] = 'gd'
        self.lang_map['Galician'] = 'gl'
        self.lang_map['Ganda'] = 'lg'
        self.lang_map['Georgian'] = 'ka'
        self.lang_map['German'] = 'de'
        self.lang_map['Greeek'] = 'el'
        self.lang_map['Gujarati'] = 'gu'
        self.lang_map['Haitian'] = 'ht'
        self.lang_map['Hausa'] = 'ha'
        self.lang_map['Hebrew'] = 'he'
        self.lang_map['Hindi'] = 'hi'
        self.lang_map['Hungarian'] = 'hu'
        self.lang_map['Icelandic'] = 'is'
        self.lang_map['Igbo'] = 'ig'
        self.lang_map['Iloko'] = 'ilo'
        self.lang_map['Indonesian'] = 'id'
        self.lang_map['Irish'] = 'ga'
        self.lang_map['Italian'] = 'it'
        self.lang_map['Japanese'] = 'ja'
        self.lang_map['Javanese'] = 'jv'
        self.lang_map['Kannada'] = 'kn'
        self.lang_map['Kazakh'] = 'kk'
        self.lang_map['Korean'] = 'ko'
        self.lang_map['Lao'] = 'lo'
        self.lang_map['Latvian'] = 'lv'
        self.lang_map['Lingala'] = 'ln'
        self.lang_map['Lithuanian'] = 'lt'
        self.lang_map['Luxembourgish'] = 'lb'
        self.lang_map['Macedonian'] = 'mk'
        self.lang_map['Malagasy'] = 'mg'
        self.lang_map['Malay'] = 'ms'
        self.lang_map['Malayalam'] = 'ml'
        self.lang_map['Marathi'] = 'mr'
        self.lang_map['Mongolian'] = 'mn'
        self.lang_map['Nepali'] = 'ne'
        self.lang_map['Northern Sotho'] = 'ns'
        self.lang_map['Norwegian'] = 'no'
        self.lang_map['Occitan (post 1500)'] = 'oc'
        self.lang_map['Oriya'] = 'or'
        self.lang_map['Panjabi'] = 'pa'
        self.lang_map['Persian'] = 'fa'
        self.lang_map['Polish'] = 'pl'
        self.lang_map['Portuguese'] = 'pt'
        self.lang_map['Pushto'] = 'ps'
        self.lang_map['Romanian'] = 'ro'
        self.lang_map['Russian'] = 'ru'
        self.lang_map['Serbian'] = 'sr'
        self.lang_map['Sindhi'] = 'sd'
        self.lang_map['Sinhala'] = 'si'
        self.lang_map['Slovak'] = 'sk'
        self.lang_map['Slovenian'] = 'sl'
        self.lang_map['Somali'] = 'so'
        self.lang_map['Spanish'] = 'es'
        self.lang_map['Sundanese'] = 'su'
        self.lang_map['Swahili'] = 'sw'
        self.lang_map['Swati'] = 'ss'
        self.lang_map['Swedish'] = 'sv'
        self.lang_map['Tagalog'] = 'tl'
        self.lang_map['Tamil'] = 'ta'
        self.lang_map['Thai'] = 'th'
        self.lang_map['Tswana'] = 'tn'
        self.lang_map['Turkish'] = 'tr'
        self.lang_map['Ukrainian'] = 'uk'
        self.lang_map['Urdu'] = 'ur'
        self.lang_map['Uzbek'] = 'uz'
        self.lang_map['Vietnamese'] = 'vi'
        self.lang_map['Welsh'] = 'cy'
        self.lang_map['Western Frisian'] = 'fy'
        self.lang_map['Wolof'] = 'wo'
        self.lang_map['Xhosa'] = 'xh'
        self.lang_map['Yiddish'] = 'yi'
        self.lang_map['Yoruba'] = 'yo'
        self.lang_map['Zulu'] = 'zu'
        self.translator = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.translator is None:
            self.translator = ctranslate2.Translator(CT_MODEL_PATH, device=self.device)
        if self.tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(CT_MODEL_PATH, clean_up_tokenization_spaces=True)

    def moveToDevice(self, device: str, precision: str = None):
        if self.device == device and self.translator is not None:
            return

        old_translator = self.translator
        # CTranslate2 binds a Translator to its initial execution device, so a
        # device change needs a new runtime object. Keep the tokenizer in memory.
        new_translator = ctranslate2.Translator(CT_MODEL_PATH, device=device)
        self.translator = new_translator
        self.device = device
        if old_translator is not None:
            if hasattr(old_translator, 'unload_model'):
                old_translator.unload_model()
            del old_translator

    def _translate(self, src_list: List[str]) -> List[str]:
        self.tokenizer.src_lang = self.lang_map[self.lang_source]

        text = [self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(i)) for i in src_list]
        target_prefix = [self.tokenizer.lang_code_to_token[self.lang_map[self.lang_target]]]

        results = self.translator.translate_batch(text, target_prefix=[target_prefix]*len(src_list))
        text_translated = [self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(i.hypotheses[0][1:])) for i in results]
        
        return text_translated

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            self.moveToDevice(self.params['device']['value'])
    @property
    def supported_tgt_list(self) -> List[str]:
        return ['Afrikaans', 'Amharic', 'Arabic', 'Asturian', 'Azerbaijani', 'Bashkir', 'Belarusian', 'Bulgarian', 'Bengali', 'Breton', 'Bosnian', 'Catalan', 'Cebuano', 'Czech', 'Welsh', 'Danish', 'German', 'Greeek', 'English', 'Spanish', 'Estonian', 'Persian', 'Fulah', 'Finnish', 'French', 'Western Frisian', 'Irish', 'Gaelic', 'Galician', 'Gujarati', 'Hausa', 'Hebrew', 'Hindi', 'Croatian', 'Haitian', 'Hungarian', 'Armenian', 'Indonesian', 'Igbo', 'Iloko', 'Icelandic', 'Italian', 'Japanese', 'Javanese', 'Georgian', 'Kazakh', 'Central Khmer', 'Kannada', 'Korean', 'Luxembourgish', 'Ganda', 'Lingala', 'Lao', 'Lithuanian', 'Latvian', 'Malagasy', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Malay', 'Burmese', 'Nepali', 'Dutch', 'Norwegian', 'Northern Sotho', 'Occitan (post 1500)', 'Oriya', 'Panjabi', 'Polish', 'Pushto', 'Portuguese', 'Romanian', 'Russian', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Albanian', 'Serbian', 'Swati', 'Sundanese', 'Swedish', 'Swahili', 'Tamil', 'Thai', 'Tagalog', 'Tswana', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba', 'Chinese', 'Zulu']

    @property
    def supported_src_list(self) -> List[str]:
        return ['Afrikaans', 'Amharic', 'Arabic', 'Asturian', 'Azerbaijani', 'Bashkir', 'Belarusian', 'Bulgarian', 'Bengali', 'Breton', 'Bosnian', 'Catalan', 'Cebuano', 'Czech', 'Welsh', 'Danish', 'German', 'Greeek', 'English', 'Spanish', 'Estonian', 'Persian', 'Fulah', 'Finnish', 'French', 'Western Frisian', 'Irish', 'Gaelic', 'Galician', 'Gujarati', 'Hausa', 'Hebrew', 'Hindi', 'Croatian', 'Haitian', 'Hungarian', 'Armenian', 'Indonesian', 'Igbo', 'Iloko', 'Icelandic', 'Italian', 'Japanese', 'Javanese', 'Georgian', 'Kazakh', 'Central Khmer', 'Kannada', 'Korean', 'Luxembourgish', 'Ganda', 'Lingala', 'Lao', 'Lithuanian', 'Latvian', 'Malagasy', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Malay', 'Burmese', 'Nepali', 'Dutch', 'Norwegian', 'Northern Sotho', 'Occitan (post 1500)', 'Oriya', 'Panjabi', 'Polish', 'Pushto', 'Portuguese', 'Romanian', 'Russian', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Albanian', 'Serbian', 'Swati', 'Sundanese', 'Swedish', 'Swahili', 'Tamil', 'Thai', 'Tagalog', 'Tswana', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba', 'Chinese', 'Zulu']

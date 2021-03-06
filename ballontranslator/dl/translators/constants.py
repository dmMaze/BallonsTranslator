BASE_URLS = {
    "GOOGLE_TRANSLATE": "https://translate.google.com/m",
    "PONS": "https://en.pons.com/translate/",
    "YANDEX": "https://translate.yandex.net/api/{version}/tr.json/{endpoint}",
    "LINGUEE": "https://www.linguee.com/",
    "MYMEMORY": "http://api.mymemory.translated.net/get",
    "QCRI": "https://mt.qcri.org/api/v1/{endpoint}?",
    "DEEPL": "https://api.deepl.com/{version}/",
    "DEEPL_FREE": "https://api-free.deepl.com/v2/",
    "MICROSOFT_TRANSLATE": "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0",
    "PAPAGO": "https://papago.naver.com/",
    "PAPAGO_API": "https://openapi.naver.com/v1/papago/n2mt"
}

GOOGLE_CODES_TO_LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-CN': 'chinese (simplified)',
    'zh-TW': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'rw': 'kinyarwanda',
    'ko': 'korean',
    'ku': 'kurdish',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar',
    'ne': 'nepali',
    'no': 'norwegian',
    'or': 'odia',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'tt': 'tatar',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'tk': 'turkmen',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'ug': 'uyghur',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu',
}

GOOGLE_LANGUAGES_TO_CODES = {v: k for k, v in GOOGLE_CODES_TO_LANGUAGES.items()}

# This dictionary maps the primary name of language to its secondary names in list manner (if any)
GOOGLE_LANGUAGES_SECONDARY_NAMES = {
    'myanmar': ['burmese'],
    'odia': ['oriya'],
    'kurdish':  ['kurmanji']
}


PONS_CODES_TO_LANGUAGES = {
    'ar': 'arabic',
    'bg': 'bulgarian',
    'zh-cn': 'chinese',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'hu': 'hungarian',
    'it': 'italian',
    'la': 'latin',
    'no': 'norwegian',
    'pl': 'polish',
    'pt': 'portuguese',
    'ru': 'russian',
    'sl': 'slovenian',
    'es': 'spanish',
    'sv': 'swedish',
    'tr': 'turkish',
    'elv': 'elvish'
}

PONS_LANGUAGES_TO_CODES = {v: k for k, v in PONS_CODES_TO_LANGUAGES.items()}

LINGUEE_LANGUAGES_TO_CODES = {
    "maltese": "mt",
    "english": "en",
    "german": "de",
    "bulgarian": "bg",
    "polish": "pl",
    "portuguese": "pt",
    "hungarian": "hu",
    "romanian": "ro",
    "russian": "ru",
    # "serbian": "sr",
    "dutch": "nl",
    "slovakian": "sk",
    "greek": "el",
    "slovenian": "sl",
    "danish": "da",
    "italian": "it",
    "spanish": "es",
    "finnish": "fi",
    "chinese": "zh",
    "french": "fr",
    # "croatian": "hr",
    "czech": "cs",
    "laotian": "lo",
    "swedish": "sv",
    "latvian": "lv",
    "estonian": "et",
    "japanese": "ja"
}

LINGUEE_CODE_TO_LANGUAGE = {v: k for k, v in LINGUEE_LANGUAGES_TO_CODES.items()}

# "72e9e2cc7c992db4dcbdd6fb9f91a0d1"

# obtaining the current list of supported Microsoft languages for translation

# microsoft_languages_api_url = "https://api.cognitive.microsofttranslator.com/languages?api-version=3.0&scope=translation"
# microsoft_languages_response = requests.get(microsoft_languages_api_url)
translation_dict = {"af": {"name": "Afrikaans", "nativeName": "Afrikaans", "dir": "ltr"}, "am": {"name": "Amharic", "nativeName": "????????????", "dir": "ltr"}, "ar": {"name": "Arabic", "nativeName": "??????????????", "dir": "rtl"}, "as": {"name": "Assamese", "nativeName": "?????????????????????", "dir": "ltr"}, "az": {"name": "Azerbaijani", "nativeName": "Az??rbaycan", "dir": "ltr"}, "ba": {"name": "Bashkir", "nativeName": "Bashkir", "dir": "ltr"}, "bg": {"name": "Bulgarian", "nativeName": "??????????????????", "dir": "ltr"}, "bn": {"name": "Bangla", "nativeName": "???????????????", "dir": "ltr"}, "bo": {"name": "Tibetan", "nativeName": "????????????????????????", "dir": "ltr"}, "bs": {"name": "Bosnian", "nativeName": "Bosnian", "dir": "ltr"}, "ca": {"name": "Catalan", "nativeName": "Catal??", "dir": "ltr"}, "cs": {"name": "Czech", "nativeName": "??e??tina", "dir": "ltr"}, "cy": {"name": "Welsh", "nativeName": "Cymraeg", "dir": "ltr"}, "da": {"name": "Danish", "nativeName": "Dansk", "dir": "ltr"}, "de": {"name": "German", "nativeName": "Deutsch", "dir": "ltr"}, "dv": {"name": "Divehi", "nativeName": "????????????????????", "dir": "rtl"}, "el": {"name": "Greek", "nativeName": "????????????????", "dir": "ltr"}, "en": {"name": "English", "nativeName": "English", "dir": "ltr"}, "es": {"name": "Spanish", "nativeName": "Espa??ol", "dir": "ltr"}, "et": {"name": "Estonian", "nativeName": "Eesti", "dir": "ltr"}, "fa": {"name": "Persian", "nativeName": "??????????", "dir": "rtl"}, "fi": {"name": "Finnish", "nativeName": "Suomi", "dir": "ltr"}, "fil": {"name": "Filipino", "nativeName": "Filipino", "dir": "ltr"}, "fj": {"name": "Fijian", "nativeName": "Na Vosa Vakaviti", "dir": "ltr"}, "fr": {"name": "French", "nativeName": "Fran??ais", "dir": "ltr"}, "fr-CA": {"name": "French (Canada)", "nativeName": "Fran??ais (Canada)", "dir": "ltr"}, "ga": {"name": "Irish", "nativeName": "Gaeilge", "dir": "ltr"}, "gu": {"name": "Gujarati", "nativeName": "?????????????????????", "dir": "ltr"}, "he": {"name": "Hebrew", "nativeName": "??????????", "dir": "rtl"}, "hi": {"name": "Hindi", "nativeName": "??????????????????", "dir": "ltr"}, "hr": {"name": "Croatian", "nativeName": "Hrvatski", "dir": "ltr"}, "hsb": {"name": "Upper Sorbian", "nativeName": "Hornjoserb????ina", "dir": "ltr"}, "ht": {"name": "Haitian Creole", "nativeName": "Haitian Creole", "dir": "ltr"}, "hu": {"name": "Hungarian", "nativeName": "Magyar", "dir": "ltr"}, "hy": {"name": "Armenian", "nativeName": "??????????????", "dir": "ltr"}, "id": {"name": "Indonesian", "nativeName": "Indonesia", "dir": "ltr"}, "ikt": {"name": "Inuinnaqtun", "nativeName": "Inuinnaqtun", "dir": "ltr"}, "is": {"name": "Icelandic", "nativeName": "??slenska", "dir": "ltr"}, "it": {"name": "Italian", "nativeName": "Italiano", "dir": "ltr"}, "iu": {"name": "Inuktitut", "nativeName": "??????????????????", "dir": "ltr"}, "iu-Latn": {"name": "Inuktitut (Latin)", "nativeName": "Inuktitut (Latin)", "dir": "ltr"}, "ja": {"name": "Japanese", "nativeName": "?????????", "dir": "ltr"}, "ka": {"name": "Georgian", "nativeName": "?????????????????????", "dir": "ltr"}, "kk": {"name": "Kazakh", "nativeName": "?????????? ????????", "dir": "ltr"}, "km": {"name": "Khmer", "nativeName": "???????????????", "dir": "ltr"}, "kmr": {"name": "Kurdish (Northern)", "nativeName": "Kurd?? (Bakur)", "dir": "ltr"}, "kn": {"name": "Kannada", "nativeName": "???????????????", "dir": "ltr"}, "ko": {"name": "Korean", "nativeName": "?????????", "dir": "ltr"}, "ku": {"name": "Kurdish (Central)", "nativeName": "Kurd?? (Nav??n)", "dir": "rtl"}, "ky": {"name": "Kyrgyz", "nativeName": "Kyrgyz", "dir": "ltr"}, "lo": {"name": "Lao", "nativeName": "?????????", "dir": "ltr"}, "lt": {"name": "Lithuanian", "nativeName": "Lietuvi??", "dir": "ltr"}, "lv": {"name": "Latvian", "nativeName": "Latvie??u", "dir": "ltr"}, "lzh": {"name": "Chinese (Literary)", "nativeName": "?????? (?????????)", "dir": "ltr"}, "mg": {"name": "Malagasy", "nativeName": "Malagasy", "dir": "ltr"}, "mi": {"name": "M??ori", "nativeName": "Te Reo M??ori", "dir": "ltr"}, "mk": {"name": "Macedonian", "nativeName": "????????????????????", "dir": "ltr"}, "ml": {"name": "Malayalam", "nativeName": "??????????????????", "dir": "ltr"}, "mn-Cyrl": {"name": "Mongolian (Cyrillic)", "nativeName": "Mongolian (Cyrillic)", "dir": "ltr"}, "mn-Mong": {"name": "Mongolian (Traditional)", "nativeName": "?????????????????? ????????????", "dir": "ltr"}, "mr": {"name": "Marathi", "nativeName": "???????????????", "dir": "ltr"}, "ms": {"name": "Malay", "nativeName": "Melayu", "dir": "ltr"}, "mt": {"name": "Maltese", "nativeName": "Malti", "dir": "ltr"}, "mww": {"name": "Hmong Daw", "nativeName": "Hmong Daw", "dir": "ltr"}, "my": {"name": "Myanmar (Burmese)", "nativeName": "??????????????????", "dir": "ltr"}, "nb": {"name": "Norwegian", "nativeName": "Norsk Bokm??l", "dir": "ltr"}, "ne": {"name": "Nepali", "nativeName": "??????????????????", "dir": "ltr"}, "nl": {"name": "Dutch", "nativeName": "Nederlands", "dir": "ltr"}, "or": {"name": "Odia", "nativeName": "???????????????", "dir": "ltr"}, "otq": {"name": "Quer??taro Otomi", "nativeName": "H????h??u", "dir": "ltr"}, "pa": {"name": "Punjabi", "nativeName": "??????????????????", "dir": "ltr"}, "pl": {"name": "Polish", "nativeName": "Polski", "dir": "ltr"}, "prs": {"name": "Dari", "nativeName": "??????", "dir": "rtl"}, "ps": {"name": "Pashto", "nativeName": "????????", "dir": "rtl"}, "pt": {"name": "Portuguese (Brazil)", "nativeName": "Portugu??s (Brasil)", "dir": "ltr"}, "pt-PT": {"name": "Portuguese (Portugal)", "nativeName": "Portugu??s (Portugal)", "dir": "ltr"}, "ro": {"name": "Romanian", "nativeName": "Rom??n??", "dir": "ltr"}, "ru": {"name": "Russian", "nativeName": "??????????????", "dir": "ltr"}, "sk": {"name": "Slovak", "nativeName": "Sloven??ina", "dir": "ltr"}, "sl": {"name": "Slovenian", "nativeName": "Sloven????ina", "dir": "ltr"}, "sm": {"name": "Samoan", "nativeName": "Gagana S??moa", "dir": "ltr"}, "so": {"name": "Somali", "nativeName": "Af Soomaali", "dir": "ltr"}, "sq": {"name": "Albanian", "nativeName": "Shqip", "dir": "ltr"}, "sr-Cyrl": {"name": "Serbian (Cyrillic)", "nativeName": "???????????? (????????????????)", "dir": "ltr"}, "sr-Latn": {"name": "Serbian (Latin)", "nativeName": "Srpski (latinica)", "dir": "ltr"}, "sv": {"name": "Swedish", "nativeName": "Svenska", "dir": "ltr"}, "sw": {"name": "Swahili", "nativeName": "Kiswahili", "dir": "ltr"}, "ta": {"name": "Tamil", "nativeName": "???????????????", "dir": "ltr"}, "te": {"name": "Telugu", "nativeName": "??????????????????", "dir": "ltr"}, "th": {"name": "Thai", "nativeName": "?????????", "dir": "ltr"}, "ti": {"name": "Tigrinya", "nativeName": "?????????", "dir": "ltr"}, "tk": {"name": "Turkmen", "nativeName": "T??rkmen Dili", "dir": "ltr"}, "tlh-Latn": {"name": "Klingon (Latin)", "nativeName": "Klingon (Latin)", "dir": "ltr"}, "tlh-Piqd": {"name": "Klingon (pIqaD)", "nativeName": "Klingon (pIqaD)", "dir": "ltr"}, "to": {"name": "Tongan", "nativeName": "Lea Fakatonga", "dir": "ltr"}, "tr": {"name": "Turkish", "nativeName": "T??rk??e", "dir": "ltr"}, "tt": {"name": "Tatar", "nativeName": "??????????", "dir": "ltr"}, "ty": {"name": "Tahitian", "nativeName": "Reo Tahiti", "dir": "ltr"}, "ug": {"name": "Uyghur", "nativeName": "????????????????", "dir": "rtl"}, "uk": {"name": "Ukrainian", "nativeName": "????????????????????", "dir": "ltr"}, "ur": {"name": "Urdu", "nativeName": "????????", "dir": "rtl"}, "uz": {"name": "Uzbek (Latin)", "nativeName": "Uzbek (Latin)", "dir": "ltr"}, "vi": {"name": "Vietnamese", "nativeName": "Ti???ng Vi???t", "dir": "ltr"}, "yua": {"name": "Yucatec Maya", "nativeName": "Yucatec Maya", "dir": "ltr"}, "yue": {"name": "Cantonese (Traditional)", "nativeName": "?????? (??????)", "dir": "ltr"}, "zh-Hans": {"name": "Chinese Simplified", "nativeName": "?????? (??????)", "dir": "ltr"}, "zh-Hant": {"name": "Chinese Traditional", "nativeName": "???????????? (??????)", "dir": "ltr"}, "zu": {"name": "Zulu", "nativeName": "Isi-Zulu", "dir": "ltr"}}

MICROSOFT_CODES_TO_LANGUAGES = {translation_dict[k]['name'].lower(): k for k in translation_dict.keys()}

DEEPL_LANGUAGE_TO_CODE = {
    "bulgarian": "bg",
    "czech": "cs",
    "danish": "da",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "hungarian": "hu",
    "italian": "it",
    "japanese": "ja",
    "lithuanian": "lt",
    "latvian": "lv",
    "dutch": "nl",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "slovak": "sk",
    "slovenian": "sl",
    "swedish": "sv",
    "chinese": "zh"
}

DEEPL_CODE_TO_LANGUAGE = {v: k for k, v in DEEPL_LANGUAGE_TO_CODE.items()}

PAPAGO_CODE_TO_LANGUAGE = {
    'ko': 'Korean',
    'en': 'English',
    'ja': 'Japanese',
    'zh-CN': 'Chinese',
    'zh-TW': 'Chinese traditional',
    'es': 'Spanish',
    'fr': 'French',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesia'
}

PAPAGO_LANGUAGE_TO_CODE = {v: k for v, k in PAPAGO_CODE_TO_LANGUAGE.items()}

QCRI_CODE_TO_LANGUAGE = {
    'ar': 'Arabic',
    'en': 'English',
    'es': 'Spanish'
}

QCRI_LANGUAGE_TO_CODE = {
    v: k for k, v in QCRI_CODE_TO_LANGUAGE.items()
}

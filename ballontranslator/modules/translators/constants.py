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
translation_dict = {"af": {"name": "Afrikaans", "nativeName": "Afrikaans", "dir": "ltr"}, "am": {"name": "Amharic", "nativeName": "አማርኛ", "dir": "ltr"}, "ar": {"name": "Arabic", "nativeName": "العربية", "dir": "rtl"}, "as": {"name": "Assamese", "nativeName": "অসমীয়া", "dir": "ltr"}, "az": {"name": "Azerbaijani", "nativeName": "Azərbaycan", "dir": "ltr"}, "ba": {"name": "Bashkir", "nativeName": "Bashkir", "dir": "ltr"}, "bg": {"name": "Bulgarian", "nativeName": "Български", "dir": "ltr"}, "bn": {"name": "Bangla", "nativeName": "বাংলা", "dir": "ltr"}, "bo": {"name": "Tibetan", "nativeName": "བོད་སྐད་", "dir": "ltr"}, "bs": {"name": "Bosnian", "nativeName": "Bosnian", "dir": "ltr"}, "ca": {"name": "Catalan", "nativeName": "Català", "dir": "ltr"}, "cs": {"name": "Czech", "nativeName": "Čeština", "dir": "ltr"}, "cy": {"name": "Welsh", "nativeName": "Cymraeg", "dir": "ltr"}, "da": {"name": "Danish", "nativeName": "Dansk", "dir": "ltr"}, "de": {"name": "German", "nativeName": "Deutsch", "dir": "ltr"}, "dv": {"name": "Divehi", "nativeName": "ދިވެހިބަސް", "dir": "rtl"}, "el": {"name": "Greek", "nativeName": "Ελληνικά", "dir": "ltr"}, "en": {"name": "English", "nativeName": "English", "dir": "ltr"}, "es": {"name": "Spanish", "nativeName": "Español", "dir": "ltr"}, "et": {"name": "Estonian", "nativeName": "Eesti", "dir": "ltr"}, "fa": {"name": "Persian", "nativeName": "فارسی", "dir": "rtl"}, "fi": {"name": "Finnish", "nativeName": "Suomi", "dir": "ltr"}, "fil": {"name": "Filipino", "nativeName": "Filipino", "dir": "ltr"}, "fj": {"name": "Fijian", "nativeName": "Na Vosa Vakaviti", "dir": "ltr"}, "fr": {"name": "French", "nativeName": "Français", "dir": "ltr"}, "fr-CA": {"name": "French (Canada)", "nativeName": "Français (Canada)", "dir": "ltr"}, "ga": {"name": "Irish", "nativeName": "Gaeilge", "dir": "ltr"}, "gu": {"name": "Gujarati", "nativeName": "ગુજરાતી", "dir": "ltr"}, "he": {"name": "Hebrew", "nativeName": "עברית", "dir": "rtl"}, "hi": {"name": "Hindi", "nativeName": "हिन्दी", "dir": "ltr"}, "hr": {"name": "Croatian", "nativeName": "Hrvatski", "dir": "ltr"}, "hsb": {"name": "Upper Sorbian", "nativeName": "Hornjoserbšćina", "dir": "ltr"}, "ht": {"name": "Haitian Creole", "nativeName": "Haitian Creole", "dir": "ltr"}, "hu": {"name": "Hungarian", "nativeName": "Magyar", "dir": "ltr"}, "hy": {"name": "Armenian", "nativeName": "Հայերեն", "dir": "ltr"}, "id": {"name": "Indonesian", "nativeName": "Indonesia", "dir": "ltr"}, "ikt": {"name": "Inuinnaqtun", "nativeName": "Inuinnaqtun", "dir": "ltr"}, "is": {"name": "Icelandic", "nativeName": "Íslenska", "dir": "ltr"}, "it": {"name": "Italian", "nativeName": "Italiano", "dir": "ltr"}, "iu": {"name": "Inuktitut", "nativeName": "ᐃᓄᒃᑎᑐᑦ", "dir": "ltr"}, "iu-Latn": {"name": "Inuktitut (Latin)", "nativeName": "Inuktitut (Latin)", "dir": "ltr"}, "ja": {"name": "Japanese", "nativeName": "日本語", "dir": "ltr"}, "ka": {"name": "Georgian", "nativeName": "ქართული", "dir": "ltr"}, "kk": {"name": "Kazakh", "nativeName": "Қазақ Тілі", "dir": "ltr"}, "km": {"name": "Khmer", "nativeName": "ខ្មែរ", "dir": "ltr"}, "kmr": {"name": "Kurdish (Northern)", "nativeName": "Kurdî (Bakur)", "dir": "ltr"}, "kn": {"name": "Kannada", "nativeName": "ಕನ್ನಡ", "dir": "ltr"}, "ko": {"name": "Korean", "nativeName": "한국어", "dir": "ltr"}, "ku": {"name": "Kurdish (Central)", "nativeName": "Kurdî (Navîn)", "dir": "rtl"}, "ky": {"name": "Kyrgyz", "nativeName": "Kyrgyz", "dir": "ltr"}, "lo": {"name": "Lao", "nativeName": "ລາວ", "dir": "ltr"}, "lt": {"name": "Lithuanian", "nativeName": "Lietuvių", "dir": "ltr"}, "lv": {"name": "Latvian", "nativeName": "Latviešu", "dir": "ltr"}, "lzh": {"name": "Chinese (Literary)", "nativeName": "中文 (文言文)", "dir": "ltr"}, "mg": {"name": "Malagasy", "nativeName": "Malagasy", "dir": "ltr"}, "mi": {"name": "Māori", "nativeName": "Te Reo Māori", "dir": "ltr"}, "mk": {"name": "Macedonian", "nativeName": "Македонски", "dir": "ltr"}, "ml": {"name": "Malayalam", "nativeName": "മലയാളം", "dir": "ltr"}, "mn-Cyrl": {"name": "Mongolian (Cyrillic)", "nativeName": "Mongolian (Cyrillic)", "dir": "ltr"}, "mn-Mong": {"name": "Mongolian (Traditional)", "nativeName": "ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ", "dir": "ltr"}, "mr": {"name": "Marathi", "nativeName": "मराठी", "dir": "ltr"}, "ms": {"name": "Malay", "nativeName": "Melayu", "dir": "ltr"}, "mt": {"name": "Maltese", "nativeName": "Malti", "dir": "ltr"}, "mww": {"name": "Hmong Daw", "nativeName": "Hmong Daw", "dir": "ltr"}, "my": {"name": "Myanmar (Burmese)", "nativeName": "မြန်မာ", "dir": "ltr"}, "nb": {"name": "Norwegian", "nativeName": "Norsk Bokmål", "dir": "ltr"}, "ne": {"name": "Nepali", "nativeName": "नेपाली", "dir": "ltr"}, "nl": {"name": "Dutch", "nativeName": "Nederlands", "dir": "ltr"}, "or": {"name": "Odia", "nativeName": "ଓଡ଼ିଆ", "dir": "ltr"}, "otq": {"name": "Querétaro Otomi", "nativeName": "Hñähñu", "dir": "ltr"}, "pa": {"name": "Punjabi", "nativeName": "ਪੰਜਾਬੀ", "dir": "ltr"}, "pl": {"name": "Polish", "nativeName": "Polski", "dir": "ltr"}, "prs": {"name": "Dari", "nativeName": "دری", "dir": "rtl"}, "ps": {"name": "Pashto", "nativeName": "پښتو", "dir": "rtl"}, "pt": {"name": "Portuguese (Brazil)", "nativeName": "Português (Brasil)", "dir": "ltr"}, "pt-PT": {"name": "Portuguese (Portugal)", "nativeName": "Português (Portugal)", "dir": "ltr"}, "ro": {"name": "Romanian", "nativeName": "Română", "dir": "ltr"}, "ru": {"name": "Russian", "nativeName": "Русский", "dir": "ltr"}, "sk": {"name": "Slovak", "nativeName": "Slovenčina", "dir": "ltr"}, "sl": {"name": "Slovenian", "nativeName": "Slovenščina", "dir": "ltr"}, "sm": {"name": "Samoan", "nativeName": "Gagana Sāmoa", "dir": "ltr"}, "so": {"name": "Somali", "nativeName": "Af Soomaali", "dir": "ltr"}, "sq": {"name": "Albanian", "nativeName": "Shqip", "dir": "ltr"}, "sr-Cyrl": {"name": "Serbian (Cyrillic)", "nativeName": "Српски (ћирилица)", "dir": "ltr"}, "sr-Latn": {"name": "Serbian (Latin)", "nativeName": "Srpski (latinica)", "dir": "ltr"}, "sv": {"name": "Swedish", "nativeName": "Svenska", "dir": "ltr"}, "sw": {"name": "Swahili", "nativeName": "Kiswahili", "dir": "ltr"}, "ta": {"name": "Tamil", "nativeName": "தமிழ்", "dir": "ltr"}, "te": {"name": "Telugu", "nativeName": "తెలుగు", "dir": "ltr"}, "th": {"name": "Thai", "nativeName": "ไทย", "dir": "ltr"}, "ti": {"name": "Tigrinya", "nativeName": "ትግር", "dir": "ltr"}, "tk": {"name": "Turkmen", "nativeName": "Türkmen Dili", "dir": "ltr"}, "tlh-Latn": {"name": "Klingon (Latin)", "nativeName": "Klingon (Latin)", "dir": "ltr"}, "tlh-Piqd": {"name": "Klingon (pIqaD)", "nativeName": "Klingon (pIqaD)", "dir": "ltr"}, "to": {"name": "Tongan", "nativeName": "Lea Fakatonga", "dir": "ltr"}, "tr": {"name": "Turkish", "nativeName": "Türkçe", "dir": "ltr"}, "tt": {"name": "Tatar", "nativeName": "Татар", "dir": "ltr"}, "ty": {"name": "Tahitian", "nativeName": "Reo Tahiti", "dir": "ltr"}, "ug": {"name": "Uyghur", "nativeName": "ئۇيغۇرچە", "dir": "rtl"}, "uk": {"name": "Ukrainian", "nativeName": "Українська", "dir": "ltr"}, "ur": {"name": "Urdu", "nativeName": "اردو", "dir": "rtl"}, "uz": {"name": "Uzbek (Latin)", "nativeName": "Uzbek (Latin)", "dir": "ltr"}, "vi": {"name": "Vietnamese", "nativeName": "Tiếng Việt", "dir": "ltr"}, "yua": {"name": "Yucatec Maya", "nativeName": "Yucatec Maya", "dir": "ltr"}, "yue": {"name": "Cantonese (Traditional)", "nativeName": "粵語 (繁體)", "dir": "ltr"}, "zh-Hans": {"name": "Chinese Simplified", "nativeName": "中文 (简体)", "dir": "ltr"}, "zh-Hant": {"name": "Chinese Traditional", "nativeName": "繁體中文 (繁體)", "dir": "ltr"}, "zu": {"name": "Zulu", "nativeName": "Isi-Zulu", "dir": "ltr"}}

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
    "chinese": "zh",
    "indonesia": "id"
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

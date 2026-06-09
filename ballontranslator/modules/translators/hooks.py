from typing import List

import opencc

CHS2CHT_CONVERTER = None

from .base import BaseTranslator, TextBlock

def chs2cht(translations: List[str] = None, textblocks: List[TextBlock] = None, translator: BaseTranslator = None, **kwargs) -> str:
    
    if not translator.cht_require_convert or translator.lang_target != '繁體中文':
        return
    
    global CHS2CHT_CONVERTER
    if CHS2CHT_CONVERTER is None:
        CHS2CHT_CONVERTER = opencc.OpenCC('s2t')
        
    for ii, tr in enumerate(translations):
        translations[ii] = CHS2CHT_CONVERTER.convert(tr)
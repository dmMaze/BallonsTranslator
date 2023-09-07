import opencc
CHS2CHT_CONVERTER = None


def chs2cht(text: str) -> str:
    global CHS2CHT_CONVERTER
    if CHS2CHT_CONVERTER is None:
        CHS2CHT_CONVERTER = opencc.OpenCC('s2t')
        
    return CHS2CHT_CONVERTER.convert(text)
from typing import List

CHSEG = None

def seg_to_chars(text: str) -> List[str]:
    text = text.replace('\n', '')
    return [c for c in text]

def seg_ch(text: str) -> List[str]:
    text = text.replace('\n', '')
    global CHSEG
    if CHSEG is None:
        import pkuseg
        CHSEG = pkuseg.pkuseg()

    return CHSEG.cut(text)

def seg_eng(text: str) -> List[str]:
    text = text.upper().replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = ''

    # dumb way to insure spaces between words
    text_len = len(text)
    for ii, c in enumerate(text):
            if c in ['.', '?', '!'] and ii < text_len - 1:
                next_c = text[ii + 1]
                if next_c.isalpha() or next_c.isnumeric():
                    processed_text += c + ' '
                else:
                    processed_text += c
            else:
                processed_text += c
    word_list = processed_text.split(' ')
    words = []
    skip_next = False
    word_num = len(word_list)
    for ii, word in enumerate(word_list):
        if skip_next:
            skip_next = False
            continue
        if ii < word_num - 1:
            if len(word) == 1 or len(word_list[ii + 1]) == 1:
                skip_next = True
                word = word + ' ' + word_list[ii + 1]
        words.append(word)
    return words



        


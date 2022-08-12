from typing import List, Tuple

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
    text = text.replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
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
        if len(word) < 3:
            append_left, append_right = False, False
            len_word, len_next, len_prev = len(word), -1, -1
            if ii < word_num - 1:
                len_next = len(word_list[ii + 1])
            if ii > 0:
                len_prev = len(word_list[ii - 1])
            cond_next = (len_word == 2 and len_next <= 4) or len_word == 1
            cond_prev = (len_word == 2 and len_prev <= 4) or len_word == 1
            if len_next > 0 and len_prev > 0:
                if len_next < len_prev:
                    append_right = cond_next
                else:
                    append_left = cond_prev
            elif len_next > 0:
                append_right = cond_next
            elif len_prev:
                append_left = cond_prev

            if append_left:
                words[-1] = words[-1] + ' ' + word
            elif append_right:
                words.append(word + ' ' + word_list[ii + 1])
                skip_next = True
            else:
                words.append(word)
            continue
        words.append(word)
    return words

def seg_text(text: str, lang: str) -> Tuple[List, str]:
    delimiter = ''
    if lang in ['简体中文', '繁体中文']:
        words = seg_ch(text)    
    elif lang in ['日本語', '한국어']:
        words = seg_to_chars(text)
    else:
        words = seg_eng(text)
        delimiter = ' '
    return words, delimiter

LOGORAMS = ['简体中文', '繁体中文', '日本語', '한국어']

def is_logogram(lang: str) -> bool:
    return lang in LOGORAMS

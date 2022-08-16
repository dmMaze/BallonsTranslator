from typing import List, Tuple
from tqdm import tqdm
import json

HALF2FULL = {i: i + 0xFEE0 for i in range(0x21, 0x7F)}
HALF2FULL[0x20] = 0x3000

FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x21, 0x7F))
FULL2HALF[0x3000] = 0x20
FULL2HALF[0x3002] = 0x2E

LANGSET_CJK = {'简体中文', '繁体中文', '日本語', '한국어'}
LANGSET_CH = {'简体中文', '繁体中文'}

PUNSET_RIGHT_ENG = {'.', '?', '!', ':', ';', ')', '}', "\""}
PUNCTUATION_L = {'「', '『', '【', '《', '〈', '〔', '［', '｛', '（', '(', '[', '{', '“', '‘'}

PKUSEG_PUNCSET = {' ', '.', '　'}
PKUSEGPATH = r'data/pkusegscores.json'
PKUSEGSCORES = None
CHSEG = None

def full_len(s: str):
    """
    Convert all ASCII characters to their full-width counterpart.
    https://stackoverflow.com/questions/2422177/python-how-can-i-replace-full-width-characters-with-half-width-characters 
    """
    return s.translate(HALF2FULL)

def half_len(s):
    '''
    Convert full-width characters to ASCII counterpart
    '''
    return s.translate(FULL2HALF)

def seg_to_chars(text: str) -> List[str]:
    text = text.replace('\n', '')
    return [c for c in text]

def seg_eng(text: str) -> List[str]:
    text = text.replace('  ', ' ').replace(' .', '.').replace('\n', ' ')
    processed_text = ''

    # dumb way to insure spaces between words
    text_len = len(text)
    for ii, c in enumerate(text):
        if c in PUNSET_RIGHT_ENG and ii < text_len - 1:
            next_c = text[ii + 1]
            if next_c.isalpha() or next_c.isnumeric():
                processed_text += c + ' '
            else:
                processed_text += c
        else:
            processed_text += c

    word_list = processed_text.split(' ')
    word_num = len(word_list)
    if word_num <= 1:
        return word_list

    words = []
    skip_next = False
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
                len_prev = len(words[-1])
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

def _seg_ch_pkg(text: str) -> List[str]:

    if text == ' ':
        return [' ']
    elif text == '':
        return []

    segments = CHSEG.cut(text)
    num_segments = len(segments)
    if num_segments == 0:
        return []
    if num_segments == 1:
        return [segments[0][0]]

    words = []
    tags = []
    max_concat_len = 4
    skip_next = False
    try:
        for ii, (word, tag) in enumerate(segments):
            if skip_next:
                skip_next = False
                continue
            
            len_word, len_next, len_prev = len(word), -1, -1
            next_valid, prev_valid = False, False
            word_next, tag_next = '', ''
            word_prev, tag_prev = '', ''
            score_next, score_prev = 0, 0
            if ii < num_segments - 1:
                word_next, tag_next = segments[ii + 1]
                len_next = len(word_next)
                next_valid = True
                if tag_next != 'w' and not word_next in PKUSEG_PUNCSET:
                    score_next = PKUSEGSCORES[tag][tag_next]
            
            if ii > 0:
                word_prev, tag_prev = words[-1], segments[ii - 1][1]
                len_prev = len(word_prev)
                prev_valid = True
                if tag_prev != 'w' and not word_prev[-1] in PKUSEG_PUNCSET:
                    score_prev = PKUSEGSCORES[tag_prev][tag]

            append_prev, append_next = False, False

            if tag == 'w' or word in PKUSEG_PUNCSET:  # puntuation
                if word in PUNCTUATION_L:
                    append_next = next_valid
                elif len_word  <= 1:
                    append_prev = prev_valid
            else:
                next_valid = score_next > 0 and len_next < max_concat_len
                prev_valid = score_prev > 0 and len_prev < max_concat_len
                need_concat = len_word < max_concat_len
                append_prev = score_prev == 1
                append_next = score_next == 1
                if score_prev != 1 and score_next != 1 and need_concat:
                    append_prev = prev_valid
                    append_next = next_valid
                    if append_next and append_prev:
                        if len_prev == len_next:
                            if score_prev >= score_next:
                                append_next = False
                            else:
                                append_prev = False
                        elif len_prev < len_next:
                            append_next = False
                        else:
                            append_prev = False

            if append_next and append_prev:
                words[-1] = word_prev + word + word_next
                tags[-1] = tags[-1] + [tag, tag_next]
                skip_next = True
            elif append_prev:
                words[-1] = words[-1] + word
                tags[-1].append(tag)
            elif append_next:
                words.append(word + word_next)
                tags.append([tag, tag_next])
                skip_next = True
            else:
                words.append(word)
                tags.append([tag])
    except Exception as e:
        print('exp at line: ', text)
        raise e
    return words

def seg_ch_pkg(text: str):

    global CHSEG
    if CHSEG is None:
        import pkuseg
        CHSEG = pkuseg.pkuseg(postag=True)

    # pkuseg won't work with half-width punctuations
    fullen_text = full_len(text).replace('　', ' ')
    cvt_back = False
    if fullen_text != text:
        cvt_back = True
        text = fullen_text

    global PKUSEGSCORES
    if PKUSEGSCORES is None:
        with open(PKUSEGPATH, 'r', encoding='utf8') as f:
            PKUSEGSCORES = json.loads(f.read())
    
    text_list = text.replace('\n', '').replace('　', ' ').split(' ')
    result_list = []
    for ii, text in enumerate(text_list):
        words = None
        if text:
            words = _seg_ch_pkg(text)
        if words is not None:
            if ii > 0:
                words[0] = ' ' + words[0]
            result_list.extend(words)

    if cvt_back:
        # pkuseg w
        result_list = [half_len(word) for word in result_list]
    return result_list

def seg_text(text: str, lang: str) -> Tuple[List, str]:
    delimiter = ''
    if lang in LANGSET_CH:
        words = seg_ch_pkg(text)    
    elif lang in LANGSET_CJK:
        words = seg_to_chars(text)
    else:
        words = seg_eng(text)
        delimiter = ' '
    return words, delimiter

def is_cjk(lang: str) -> bool:
    return lang in LANGSET_CJK
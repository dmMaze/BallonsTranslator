from typing import List, Callable, Tuple
import numpy as np
import cv2

from .text_processing import seg_ch, seg_eng, seg_to_chars
from .imgproc_utils import extract_ballon_region, rotate_image

class Line:

    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 0
        if text:
            self.num_words += 1

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = self.text + delimiter + word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = word + delimiter + self.text
        if word:
            self.num_words += 1
        self.length += w_len

def layout_lines_with_mask(
    mask: np.ndarray, 
    words: List[str], 
    wl_list: List[int], 
    delimiter_len: int, 
    line_height: int,
    alignment: int = 0,
    vertical: bool = False,
    delimiter: str = ' ',
    word_break: bool = False)->List[Line]:

    region_x, region_y, region_w, region_h = cv2.boundingRect(cv2.findNonZero(mask))
    centroid_x = region_x + region_w // 2
    centroid_y = region_y + region_h // 2

    # m = cv2.moments(mask)
    mask = 255 - mask
    # centroid_y = int(m['m01'] / m['m00'])
    # centroid_x = int(m['m10'] / m['m00'])
    
    # layout the central line, the center word is approximately aligned with the centroid of the mask
    num_words = len(words)
    len_left, len_right = [], []
    wlst_left, wlst_right = [], []
    sum_left, sum_right = 0, 0
    if num_words > 1:
        wl_cumsums = np.cumsum(np.array(wl_list, dtype=np.float64))
        wl_cumsums -= wl_cumsums[-1] / 2
        central_index = np.argmin(np.abs(wl_cumsums))
        if wl_list[central_index] < 0:
            central_index += 1
        if central_index > 0:
            wlst_left = words[:central_index]
            len_left = wl_list[:central_index]
            sum_left = np.sum(len_left)
        if central_index < num_words - 1:
            wlst_right = words[central_index + 1:]
            len_right = wl_list[central_index + 1:]
            sum_right = np.sum(len_right)
    else:
        central_index = 0

    pos_y = centroid_y - line_height // 2
    pos_x = centroid_x - wl_list[central_index] // 2

    bh, bw = mask.shape[:2]
    central_line = Line(words[central_index], pos_x, pos_y, wl_list[central_index])
    line_bottom = pos_y + line_height
    while sum_left > 0 or sum_right > 0:
        left_valid, right_valid = False, False

        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if (new_x_l > 0 and new_r_l < bw):
                if mask[pos_y: line_bottom, new_x_l].sum()==0 and mask[pos_y: line_bottom, new_r_l].sum() == 0:
                    left_valid = True
        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2
            new_r_r = new_x_r + new_len_r
            if (new_x_r > 0 and new_r_r < bw):
                if mask[pos_y: line_bottom, new_x_r].sum()==0 and mask[pos_y: line_bottom, new_r_r].sum() == 0:
                    right_valid = True

        insert_left = False
        if left_valid and right_valid:
            if sum_left > sum_right:
                insert_left = True
        elif left_valid:
            insert_left = True
        elif not right_valid:
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r

    lines = [central_line]

    # layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y + line_height // 2
        line_bottom = pos_y + line_height
        line = Line(w, pos_x, pos_y, wl)
        lines.append(line)
        sum_right -= wl
        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
            else:
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line = Line(w, pos_x, pos_y, wl)
                lines.append(line)

    # layout top half
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height // 2 - line_height
        line_bottom = pos_y + line_height
        line = Line(w, pos_x, pos_y, wl)
        lines.insert(0, line)
        sum_left -= wl
        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom, new_x].sum() > 0 or\
                mask[pos_y: line_bottom, right_x].sum() > 0:
                line_valid = False
            else:
                line_valid = True
            if line_valid:
                line.append_left(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
            else:
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line = Line(w, pos_x, pos_y, wl)
                lines.insert(0, line)

    return lines

def layout_text(
    text: str, 
    lang: str,
    angle: float,
    line_spacing: float,
    alignment: int,
    vertical: bool,
    text_size_func: Callable, 
    padding: float = 0,
    mask: np.ndarray = None, 
    img: np.ndarray = None, bounding_rect: List = None) -> Tuple[str, List]:

    # preprocessing 
    delimiter = ''
    if lang in ['简体中文', '繁体中文']:
        words = seg_ch(text)    
    elif lang in ['日本語', '한국어']:
        words = seg_to_chars(text)
    else:
        words = seg_eng(text)
        delimiter = ' '

    delimiter_len = text_size_func(delimiter)[0]

    num_words = len(words)
    if num_words == 0:
        return []
    wl_list = []
    for word in words:
        w, h = text_size_func(word)
        wl_list.append(w)

    if mask is None:
        assert img is not None
        assert bounding_rect is not None
        br = bounding_rect
        enlarge_ratio = min(max(br[2] / br[3], br[3] / br[2]), 3.0)
        mask, ballon_area, xyxy = extract_ballon_region(img, bounding_rect, enlarge_ratio=enlarge_ratio)
    else:
        xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]]

    region_x, region_y, region_w, region_h = cv2.boundingRect(cv2.findNonZero(mask))
    if abs(angle) > 0:
        mask = rotate_image(mask, angle)

    line_height = int(h * line_spacing)

    
    lines = layout_lines_with_mask(mask, words, wl_list, delimiter_len, line_height, alignment, vertical, delimiter)
    center_x = xyxy[0] + region_x + region_w // 2
    center_y = xyxy[1] + region_y + region_h // 2
    
    concated_text = []
    pos_x_lst, pos_right_lst = [], []
    for line in lines:
        pos_x_lst.append(line.pos_x)
        pos_right_lst.append(max(line.pos_x, 0) + line.length)
        concated_text.append(line.text)
    concated_text = '\n'.join(concated_text)

    pos_x_lst = np.array(pos_x_lst)
    pos_right_lst = np.array(pos_right_lst)
    canvas_l, canvas_r = pos_x_lst.min() - padding, pos_right_lst.max() + padding
    canvas_t, canvas_b = lines[0].pos_y - padding, lines[-1].pos_y + line_height + padding

    canvas_h = int(canvas_b - canvas_t)
    canvas_w = int(canvas_r - canvas_l)
    abs_x = int(round(center_x - canvas_w / 2))
    abs_y = int(round(center_y - canvas_h / 2))

    return concated_text, [abs_x, abs_y, canvas_w, canvas_h]
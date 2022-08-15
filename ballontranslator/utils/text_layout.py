from typing import List, Callable, Tuple
import numpy as np
import cv2

from .text_processing import seg_text
from .imgproc_utils import extract_ballon_region, rotate_image

class Line:

    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 0
        if text:
            self.num_words += 1
        self.spacing = 0
        self.add_spacing(spacing)

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

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0

def layout_lines_aligncenter(
    mask: np.ndarray, 
    words: List[str], 
    centroid: List[int],
    wl_list: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf,
    word_break: bool = False)->List[Line]:

    centroid_x, centroid_y = centroid

    # rbgmsk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(rbgmsk, (centroid_x, centroid_y), 10, (255, 0, 0))
    # cv2.imshow('mask', rbgmsk)
    # cv2.waitKey(0)

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
    central_line = Line(words[central_index], pos_x, pos_y, wl_list[central_index], spacing)
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
        if central_line.length > max_central_width:
            break

    central_line.strip_spacing()
    lines = [central_line]

    # layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y + line_height // 2
        line_bottom = pos_y + line_height
        line = Line(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl
        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw or new_len > max_central_width:
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
                line.strip_spacing()
                line = Line(w, pos_x, pos_y, wl, spacing)
                lines.append(line)

    # layout top half
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height // 2 - line_height
        line_bottom = pos_y + line_height
        line = Line(w, pos_x, pos_y, wl, spacing)
        lines.insert(0, line)
        sum_left -= wl
        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2
            right_x = new_x + new_len
            if new_x <= 0 or right_x >= bw or new_len > max_central_width:
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
                line.strip_spacing()
                line = Line(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)

    return lines

def layout_lines_alignleft(
    mask: np.ndarray, 
    words: List[str], 
    origin: List[int],
    wl_list: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    word_break: bool = False)->List[Line]:

    ox, oy = origin
    bh, bw = mask.shape[:2]
    mask = 255 - mask
    num_words = len(words)

    lines = []
    if num_words > 0:
        sum_right = np.array(wl_list).sum()
        w, wl = words.pop(0), wl_list.pop(0)
        line = Line(w, ox, oy, wl)
        lines.append(line)
        sum_right -= wl
        line_bottom = oy + line_height
        pos_y = oy
        while sum_right > 0:
            w, wl = words.pop(0), wl_list.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_r = ox + new_len
            line_valid = False
            if new_r < bw:
                if mask[pos_y: line_bottom, new_r].sum()==0:
                    line_valid = True
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
            else:
                pos_y = line_bottom
                line_bottom += line_height
                line = Line(w, ox, pos_y, wl)
                lines.append(line)
    return lines

def layout_text(
    mask: np.ndarray, 
    mask_xyxy: List,
    centroid: List,
    words: List[str],
    wl_list: List[int],
    delimiter: str,
    delimiter_len: int,
    angle: float,
    line_height: int,
    alignment: int,
    vertical: bool,
    spacing: int = 0,
    padding: float = 0, 
    max_central_width=np.inf) -> Tuple[str, List]:

    num_words = len(words)
    if num_words == 0:
        return []

    centroid_x, centroid_y = centroid
    center_x = mask_xyxy[0] + centroid_x
    center_y = mask_xyxy[1] + centroid_y
    shifted_x, shifted_y = 0, 0
    if abs(angle) > 0:

        old_h, old_w = mask.shape[:2]
        old_origin = (old_w // 2, old_h // 2)
        rel_cx, rel_cy = centroid[0] - old_origin[0], centroid[1] - old_origin[1]
        
        mask = rotate_image(mask, angle)
        rad = np.deg2rad(angle)
        r_sin, r_cos = np.sin(rad), np.cos(rad)
        new_rel_cy =  -rel_cx * r_sin + rel_cy * r_cos
        new_rel_cx =  rel_cy * r_sin + rel_cx * r_cos

        shifted_x, shifted_y = new_rel_cx - rel_cx, new_rel_cy - rel_cy
        
        new_h, new_w = mask.shape[:2]
        new_origin = (new_w // 2, new_h // 2)
        new_cx, new_cy = new_origin[0] + new_rel_cx, new_origin[1] + new_rel_cy
        centroid = [int(new_cx), int(new_cy)]

    if alignment == 1:
        lines = layout_lines_aligncenter(mask, words, centroid, wl_list, delimiter_len, line_height, spacing, delimiter, max_central_width)    
    else:
        lines = layout_lines_alignleft(mask, words, centroid, wl_list, delimiter_len, line_height, spacing, delimiter)
    
    
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

    if alignment == 1:
        abs_x = int(round(center_x - canvas_w / 2))
        abs_y = int(round(center_y - canvas_h / 2))
    else:
        abs_x = shifted_x
        abs_y = shifted_y

    return concated_text, [abs_x, abs_y, canvas_w, canvas_h]
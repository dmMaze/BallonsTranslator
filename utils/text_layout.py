from typing import List, Tuple
import numpy as np

from .imgproc_utils import rotate_image
from .textblock import TextBlock, TextAlignment

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

def line_is_valid(line: Line, new_len: int, delimiter_len, max_width, words_length, srcline_wlist, line_no: int, line_height, ref_src_lines: bool = False):
    if ref_src_lines:
        # if line_no >= 0 and line_no < len(srcline_wlist):
        #     _max_width = min(srcline_wlist[line_no], max_width)
        # else:
        #     _max_width = max_width
        if line_no >= 0 and line_no < len(srcline_wlist):
            _max_width = srcline_wlist[line_no] * words_length
        else:
            _max_width = np.inf
            _max_width = max(srcline_wlist) * words_length
        _max_width = _max_width + delimiter_len * line.num_words
        max_width = min(max_width, _max_width)

    if new_len < max_width:
        return True
    else:
        if line.length / max_width < max_width / new_len:
            return True
        else:
            return False

def layout_lines_aligncenter(
    blk: TextBlock,
    mask: np.ndarray, 
    words: List[str], 
    centroid: List[int],
    wl_list: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf,
    word_break: bool = False,
    ref_src_lines = False,
    srcline_wlist=None,
    start_from_top=False
)->List[Line]:
    
    lh_pad = 0
    if blk.line_spacing > 1:
        lh_pad = int(np.ceil(line_height - line_height / blk.line_spacing))

    centroid_x, centroid_y = centroid
    adjust_x = adjust_y = 0

    border_thr = 220
    
    # layout the central line, the center word is approximately aligned with the centroid of the mask
    num_words = len(words)
    len_left, len_right = [], []
    wlst_left, wlst_right = [], []
    sum_left, sum_right = 0, 0
    words_length = sum(wl_list)
    if num_words > 1:
        wl_array = np.array(wl_list, dtype=np.float64)
        wl_cumsums = np.cumsum(wl_array)
        wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
        central_index = np.argmin(np.abs(wl_cumsums))

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
    while (sum_left > 0 or sum_right > 0) and not start_from_top:
        left_valid, right_valid = False, False

        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if (new_x_l > 0 and new_r_l < bw):
                if mask[pos_y: line_bottom - lh_pad, new_x_l].mean() > border_thr and \
                    mask[pos_y: line_bottom - lh_pad, new_r_l].mean() > border_thr:
                    left_valid = True
        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2 - line_height // 2
            new_r_r = centroid_x + new_len_r // 2 + line_height // 2
            if (new_x_r > 0 and new_r_r < bw):
                if mask[pos_y: line_bottom - lh_pad, new_x_r].mean() > border_thr and \
                    mask[pos_y: line_bottom - lh_pad, new_r_r].mean() > border_thr:
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
            new_len = central_line.length + len_left[-1] + delimiter_len
        else:
            new_len = central_line.length + len_right[0] + delimiter_len

        line_valid = line_is_valid(central_line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, -1, line_height, ref_src_lines)
        if ref_src_lines and not line_valid and len(srcline_wlist) == 1:
            if new_len < max_central_width:
                line_valid = True
        if not line_valid:
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r

    line_right_no = line_left_no = 0
    if ref_src_lines:
        nl = len(srcline_wlist)
        if nl % 2 == 0:
            line_right_no = nl // 2
            line_left_no = nl // 2 - 1
        else:
            line_right_no = nl // 2 + 1
            line_left_no = nl // 2 - 1

    if not start_from_top:
        central_line.strip_spacing()
        lines = [central_line]
    else:
        lines = []
        sum_right = sum(wl_list)
        sum_left = 0
        wlst_right = words
        len_right = wl_list
        line_right_no = 0

    # layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        if start_from_top:
            pos_y = centroid_y - int(blk.bounding_rect()[3] / 2)
        else:
            pos_y = centroid_y + line_height // 2
        top_mean = mask[pos_y, :].mean()
        x_mean = mask.mean(axis=1)
        base_mean = x_mean.max() / 2
        if top_mean < base_mean:
            available_y = np.where(
                x_mean[pos_y:] > base_mean
            )[0]
            if len(available_y) > 0:
                adjust_y = min(available_y[0], line_height)
                pos_y = pos_y + adjust_y
        line_bottom = pos_y + line_height
        line = Line(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl
        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2 - line_height // 2
            right_x = new_x + new_len + line_height // 2
            if new_x < 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom - lh_pad, new_x].mean() < border_thr or\
                mask[pos_y: line_bottom - lh_pad, right_x].mean() < border_thr:
                line_valid = False
                if ref_src_lines and (len(wl_list) == 1 or line_right_no + 1 >= len(srcline_wlist)) and \
                    line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_right_no, line_height, ref_src_lines):
                    line_valid = True
            else:
                line_valid = True
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                line_valid = line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_right_no, line_height, ref_src_lines)
                if not line_valid:
                    if sum_right > 0:
                        w, wl = wlst_right.pop(0), len_right.pop(0)
                        sum_right -= wl
                    else:
                        line.strip_spacing()
                        break

            if not line_valid:
                # import cv2
                # m = mask.copy()
                # m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                # cv2.rectangle(m, (new_x, pos_y), (right_x, line_bottom), (255, 0, 0), 1)
                # cv2.imwrite('mask.jpg', m)
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line.strip_spacing()
                line = Line(w, pos_x, pos_y, wl, spacing)
                lines.append(line)
                line_right_no += 1

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
            new_x = centroid_x - new_len // 2 - line_height // 2
            right_x = new_x + new_len + line_height // 2
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom - lh_pad, new_x].mean() < border_thr or\
                mask[pos_y: line_bottom - lh_pad, right_x].mean() < border_thr:
                line_valid = False
                if ref_src_lines and line_left_no - 1 < 0 and \
                    line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_left_no, line_height, ref_src_lines):
                    line_valid = True
            else:
                line_valid = True
            if line_valid:
                line.append_left(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                line_valid = line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_left_no, line_height, ref_src_lines)
                if not line_valid:
                    if sum_left > 0:
                        w, wl = wlst_left.pop(-1), len_left.pop(-1)
                        sum_left -= wl
                    else:
                        line.strip_spacing()
                        break

            if not line_valid :
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line.strip_spacing()
                line = Line(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)
                line_left_no -= 1

    # rbgmsk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(rbgmsk, (centroid_x, centroid_y), 10, (255, 0, 0))
    # for line in lines:
    #     cv2.rectangle(rbgmsk, (line.pos_x, line.pos_y), (line.pos_x + line.length, line.pos_y + line_height), (0, 255, 0))
    # cv2.imshow('mask', rbgmsk)
    # cv2.waitKey(0)
    
    return lines, (adjust_x, adjust_y)

def layout_lines_alignside(
    blk: TextBlock,
    mask: np.ndarray, 
    words: List[str], 
    origin: List[int],
    wl_list: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    word_break: bool = False,
    max_width: int = np.inf,
    ref_src_lines = False,
    srcline_wlist=None,
)->List[Line]:

    align_right = blk.fontformat.alignment == TextAlignment.Right

    ox, oy = origin
    bh, bw = mask.shape[:2]
    num_words = len(words)
    blk_rect = blk.bounding_rect()
    blk_width = blk_rect[2]
    lines = []
    words_length = sum(wl_list)

    lh_pad = 0
    if blk.line_spacing > 1:
        lh_pad = int(np.ceil(line_height - line_height / blk.line_spacing))

    if num_words > 0:
        sum_right = np.array(wl_list).sum()
        w, wl = words.pop(0), wl_list.pop(0)
        line = Line(w, ox, oy, wl)
        lines.append(line)
        sum_right -= wl
        line_bottom = oy + line_height
        pos_y = oy
        line_id = 0
        while sum_right > 0:
            w, wl = words.pop(0), wl_list.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            if align_right:
                new_x = ox + blk_width - new_len - line_height // 2
            else:
                new_x = ox + new_len + line_height // 2
            line_valid = False
            if new_x < bw and new_x > 0:
                if mask[np.clip(pos_y, 0, bh - 1): np.clip(line_bottom - lh_pad, 0, bh), new_x].mean() > 240:
                    line_valid = True
                else:
                    if ref_src_lines and line_id + 1 >= len(srcline_wlist) and line_is_valid(line, new_len, delimiter_len, max_width, words_length, srcline_wlist, line_id, line_height, ref_src_lines):
                        line_valid = True
            if line_valid:
                line_valid = line_is_valid(line, new_len, delimiter_len, max_width, words_length, srcline_wlist, line_id, line_height, ref_src_lines)
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
            else:
                pos_y = line_bottom
                line_bottom += line_height
                line = Line(w, ox, pos_y, wl)
                line_id += 1
                lines.append(line)
    return lines, (0, 0)



def layout_text(
    blk: TextBlock,
    mask: np.ndarray, 
    mask_xyxy: List,
    centroid: List,
    words: List[str],
    wl_list: List[int],
    delimiter: str,
    delimiter_len: int,
    line_height: int,
    spacing: int = 0,
    max_central_width=np.inf,
    src_is_cjk=False,
    tgt_is_cjk=False,
    ref_src_lines = False
) -> Tuple[str, List]:

    angle = blk.angle
    alignment = blk.alignment

    start_from_top = False
    srcline_wlist = None

    if ref_src_lines:
        srcline_wlist, srcline_width = blk.normalizd_width_list(normalize=False)
        # tgtline_width = sum(wl_list) + delimiter_len * max(len(wl_list) - 1, 0)
        # if tgtline_width < srcline_width:
        #     min_bbox = blk.min_rect(rotate_back=True)[0]
        #     x1, y1 = min_bbox[0]
        #     x2, y2 = min_bbox[2]
        #     w = x2 - x1
        #     max_central_width = min(max_central_width, w)
        #     pass

        if alignment == TextAlignment.Center and \
        len(srcline_wlist) > 1:
            if len(srcline_wlist) == 2:
                start_from_top = True
            else:
                nw = len(srcline_wlist)
                # nl = min(nw // 2, 2)
                nl = 1
                sum_top = sum(srcline_wlist[:nl])
                sum_btn = sum(srcline_wlist[-nl:])
                start_from_top = sum_top / sum_btn > 1.2 and srcline_wlist[0] / max(srcline_wlist) > 0.9

        srcline_wlist = np.array(srcline_wlist) / srcline_width
        srcline_wlist = srcline_wlist.tolist()
        # line_height = min((blk.detected_font_size), line_height)

    # if ref_src_lines:
    #     mask = np.ones_like(mask) * 255

    if max_central_width == np.inf:
        max_central_width = mask.shape[1]

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

    if alignment == TextAlignment.Center:
        lines, adjust_xy = layout_lines_aligncenter(blk, mask, words, centroid, wl_list, delimiter_len, line_height, spacing, delimiter, 
                                         max_central_width, ref_src_lines=ref_src_lines, srcline_wlist=srcline_wlist,
                                         start_from_top=start_from_top)    
    else:
        lines, adjust_xy = layout_lines_alignside(blk, mask, words, centroid, wl_list, delimiter_len, line_height, spacing, delimiter, False, max_central_width, 
                                       ref_src_lines=ref_src_lines, srcline_wlist=srcline_wlist)
    
    concated_text = []
    pos_x_lst, pos_right_lst = [], []
    for line in lines:
        pos_x_lst.append(line.pos_x)
        pos_right_lst.append(max(line.pos_x, 0) + line.length)
        concated_text.append(line.text)
    concated_text = '\n'.join(concated_text)

    pos_x_lst = np.array(pos_x_lst)
    pos_right_lst = np.array(pos_right_lst)
    canvas_l, canvas_r = pos_x_lst.min(), pos_right_lst.max()
    canvas_t, canvas_b = lines[0].pos_y, lines[-1].pos_y + line_height

    canvas_h = int(canvas_b - canvas_t)
    canvas_w = int(canvas_r - canvas_l)

    if alignment == 1:
        abs_x = int(round(center_x - canvas_w / 2))
        abs_y = int(round(center_y - canvas_h / 2))
    else:
        abs_x = shifted_x
        abs_y = shifted_y

    return concated_text, [abs_x, abs_y, canvas_w, canvas_h], start_from_top, adjust_xy
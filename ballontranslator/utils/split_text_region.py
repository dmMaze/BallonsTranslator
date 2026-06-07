import cv2, os, re, random
import numpy as np
# import tesserocr
# from tesserocr import PyTessBaseAPI, PSM, OEM



class TextSpan(object):
    def __init__(self, top_bnd=None, bottom_bnd=None, left_bnd=None, right_bnd=None):
        self.top = top_bnd
        self.bottom = bottom_bnd
        self.height = self.bottom - self.top if bottom_bnd is not None else None

        self.left = left_bnd
        self.right = right_bnd
        self.width = self.right - self.left if right_bnd is not None else None

    def set_top(self, top_bnd):
        self.top = top_bnd
        return True

    def set_bottom(self, bottom_bnd):
        if self.top is None or bottom_bnd <= self.top:
            return False
        self.bottom = bottom_bnd
        self.height = self.bottom - self.top
        return True

    def set_left(self, left_bnd):
        self.left = left_bnd
        return True
    
    def set_right(self, right_bnd):
        if self.left is None or right_bnd <= self.left:
            return False
        self.right = right_bnd
        self.width = right_bnd - self.left
        return True

    def __getitem__(self, index):
        if isinstance(index, int) and index >=0 and index < 4:
            return [self.left, self.top, self.right, self.bottom][index]
        else:
            raise AttributeError(f'Invalid key: {index}')

def split_step0(span, thresh, sumby_yaxis, thresh2=None) -> list[TextSpan]:
    candidate_pnts = (np.where(sumby_yaxis[span.top: span.bottom] > thresh)[0] + span.top).tolist()
    span_list = []
    if len(candidate_pnts) == 0:
        return None
    stride_tol = 1
    span0, span1 = TextSpan(candidate_pnts[0]), TextSpan()
    for pnt_ind in range(len(candidate_pnts)-1):
        if candidate_pnts[pnt_ind+1] - candidate_pnts[pnt_ind] > stride_tol:
            if not span0.set_bottom(candidate_pnts[pnt_ind]):
                continue
            span_list = split_step1(span0, span_list, thresh=thresh2, sumby_yaxis=sumby_yaxis)
            span1.set_top(candidate_pnts[pnt_ind+1])
            span0 = span1
            span1 = TextSpan()

    if len(candidate_pnts)-1 == 0:
        if candidate_pnts[0] == candidate_pnts[-1]:
            span_list = None
        else:
            span0 = TextSpan(candidate_pnts[0], candidate_pnts[-1])
            span_list = split_step1(span0, span_list, thresh=thresh2, sumby_yaxis=sumby_yaxis)
    elif span0.top != candidate_pnts[-1]:
        span0.set_bottom(candidate_pnts[-1])
        span_list = split_step1(span0, span_list, thresh=thresh2, sumby_yaxis=sumby_yaxis)

    return span_list



def split_step1(span, span_list, thresh=None, sumby_yaxis=None):
    if thresh is None:
        span_list.append(span)
        return span_list
    else:
        subspan_list = split_step0(span, thresh, sumby_yaxis)
        # print(np.var(sumby_yaxis[span.top:span.bottom]))
        if subspan_list is not None:

            _, maxspan = find_span(subspan_list, max)
            _, minspan = find_span(subspan_list, min)
            
            sum_height = sum(c.height for c in subspan_list)
            
            if maxspan.height / minspan.height > 2.5 or sum_height / span.height < 0.3 or len(subspan_list) == 1:
                subspan_list = None
        if subspan_list is not None and len(subspan_list) > 1:
            span_list += subspan_list
        else:
            span_list.append(span)
        return span_list



def shrink_span_list(src_img, span_list, shrink_vert_space=True, shrink_hor_space=True):
    height, width = src_img.shape[0], src_img.shape[1]

    sum_spacing = 0
    if shrink_vert_space:
        for ii in range(len(span_list)-1):
            line_spacing = span_list[ii+1].top - span_list[ii].bottom
            sum_spacing += line_spacing
            line_spacing = int(round(line_spacing / 2))
            span_list[ii+1].top -= line_spacing
            span_list[ii].set_bottom(span_list[ii].bottom + line_spacing)
        
        if len(span_list) >= 2:
            mean_spacing = int(0.5 * round(sum_spacing / (len(span_list)-1)))
            span_list[0].top = max(0, span_list[0].top-mean_spacing)
            span_list[0].set_bottom(span_list[0].bottom)
            span_list[-1].set_bottom(min(src_img.shape[0], span_list[-1].bottom))

    left_var, middle_var = -1, -1
    if shrink_hor_space:
        left_pnts, middle_pnts = [], []
        for ii in range(len(span_list)):
            s = span_list[ii]
            im = src_img[s.top: s.bottom, 0: width]
            sumby_yaxis = np.mean(im, axis=0)
            content_array = np.where(sumby_yaxis > 10)[0].tolist()
            left, right = 0, width
            if len(content_array) != 0:
                left, right = content_array[0], content_array[-1]
            span_list[ii].set_left(left)
            span_list[ii].set_right(right)
            s = span_list[ii]
            left_pnts.append(left)
            middle_pnts.append((left+right)/2)
        left_var, middle_var = np.var(np.array(left_pnts)), np.var(np.array(middle_pnts))
            
    return span_list, (left_var, middle_var)
        
        
        
def find_span(span_list, max_or_min=max, key="height"):
    if key=="height":
        return max_or_min(enumerate(span_list), key=(lambda x: span_list[x[0]].height), default = -1)
    else:
        return max_or_min(enumerate(span_list), key=(lambda x: span_list[x[0]].width), default = -1)



def discard_spans(span_list, thresh_ratio=0.3):
    index, max_span = find_span(span_list, max)
    max_height = max_span.height
    height_thresh = max_height * thresh_ratio
    new_spanlist = []
    for sp in span_list:
        if sp.height < height_thresh:
            continue
        new_spanlist.append(sp)

    return new_spanlist



def plot_mapresult(sumbyvector, xlength, span_list=None, thresh=None):
    '''for experiment'''
    try:
        import matplotlib.pyplot as plt
        plt.plot(sumbyvector)
        plt.ylabel('div pnt value')
        plt.xlabel('div pnt coord')
        s = [0, 255]
        x_cords = []
        if span_list is not None:
            for sp in span_list:
                x_cords.append(sp.top)
                x_cords.append(sp.bottom)
        if thresh is not None:
            for tr in thresh:
                plt.vlines(x = x_cords, ymin = 0, ymax = max(s), 
                        colors = 'purple', 
                        label = 'vline_multiple - full height')
                plt.hlines(y = tr * sumbyvector.mean(), xmin = 0, xmax = xlength, linestyles='--')
        plt.show()
    except:
        pass



def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def crop_img(img, crop_ratio=0.2, clip_width=True, dilate=False):
    h, w = img.shape[:2]
    moments = cv2.moments(img)
    area = moments['m00']
    if area != 0:
        mean_x = int(round(moments['m10'] / area))
        mean_y = int(round(moments['m01'] / area))
        crop_r = int(round(crop_ratio * w))
        if clip_width:
            crop_x0 = np.clip(mean_x - crop_r, 0, w)
            crop_x1 = np.clip(mean_x + crop_r, 0, w)
            if crop_x1 > crop_x0:
                img = img[:, crop_x0: crop_x1]
        else:
            crop_r = np.clip(crop_r * 2, 0, w - 1)
            img = img[:, crop_r:]
    img = np.copy(img)
    if clip_width and dilate:
        w = int(round(w/7))
        if w > 1:
            img = cv2.dilate(img, box(w, 1), 1)
    return img, img.shape[0], img.shape[1]



def split_textblock(src_img, crop_ratio=0.2, blur=False, show_process=False, discard=True, shrink=True, recheck=False, clip_width=True, dilate=True):
    
    if blur:
        src_img = cv2.GaussianBlur(src_img,(3,3),cv2.BORDER_DEFAULT)
    if crop_ratio > 0:
        img, height, width = crop_img(src_img, crop_ratio=crop_ratio, clip_width=clip_width, dilate=dilate)
    else:
        img, height, width = src_img, src_img.shape[0], src_img.shape[1]
    
    sumby_yaxis = img.mean(axis=1)
    bound0 = np.where(sumby_yaxis > sumby_yaxis.mean() * 0.1)[0].tolist()
    vars = (-1, -1)
    
    if len(bound0) < 2:
        return [TextSpan(0, height-1, 0, width - 1)], vars

    base_span = TextSpan(bound0[0], bound0[-1])
    meanby_yaxis = sumby_yaxis.mean()

    thresh_ratio = [0.4, 0.8]
    thresh0 = meanby_yaxis * thresh_ratio[0]
    thresh2 = meanby_yaxis * thresh_ratio[1]

    span_list = split_step0(base_span, thresh0, sumby_yaxis, thresh2=thresh2)
    if span_list is None:
        return None, None
    if discard:
        span_list = discard_spans(span_list)
    if shrink:
        span_list, vars = shrink_span_list(src_img, span_list)

    '''for experiment'''
    if show_process:
        plot_mapresult(sumby_yaxis, height, span_list=span_list, thresh=thresh_ratio)

    if recheck and len(span_list) == 1 and crop_ratio > 0:
        return split_textblock(src_img, crop_ratio==-1, show_process=show_process, discard=discard, shrink=shrink, recheck=False)
    
    valid_span_list = []
    for span in span_list:
        if span.top is None:
            span.set_top(0)
        if span.left is None:
            span.set_left(0)
        if span.right is None:
            span.set_right(width)
        if span.bottom is None:
            span.set_bottom(height)
        valid_span_list.append(span)

    return valid_span_list, vars



# def tessocr_img2text(img, lang):
#     img = Image.fromarray(img)
#     if re.findall("vert", lang):
#         psm = PSM.SINGLE_BLOCK_VERT_TEXT
#     else:
#         psm = PSM.SINGLE_LINE
#     return tesserocr.image_to_text(img, psm=psm, lang=lang, path=TESSDATA_PATH)

# def tessocr_img2text(img, lang):
#     psm = "5" if re.findall("vert", lang) else "7"
#     config = r'--tessdata-dir "models\tessdata" --psm ' + psm
#     return pytesseract.image_to_string(img, lang=lang, config=config)


def textspan2list(span_list):
    converted_list = []
    for ii, s in enumerate(span_list):
        converted_list.append([])
        converted_list[ii].append(s.top)
        converted_list[ii].append(s.left)
        converted_list[ii].append(s.bottom)
        converted_list[ii].append(s.right)
    return converted_list



def manga_split(img, bbox=None, show_process=False, clip_width=False) -> list[TextSpan]:

    im = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    imh, imw = im.shape[:2]

    if bbox is None:
        bbox = [0, 0, im.shape[1], im.shape[0]]
    bboxes = [bbox]

    span_list, _ = split_textblock(im, show_process=show_process, shrink=False, recheck=True, discard=False, crop_ratio=0)
    if span_list is None:
        return [TextSpan(0, 0, im.shape[1], im.shape[0])]
    # span_list, _ = shrink_span_list(im, span_list, shrink_vert_space=False)
        
    for ii, span in enumerate(span_list):
        left = span.left
        right = span.right
        if ii == 0:
            span.left = 0
        else:
            span.left = span.top
        if ii == len(span_list) - 1:
            span.right = im.shape[0]
        else:
            span.right = span.bottom
        span.top =  imw - right
        span.bottom = imw - left
        span.height = span.bottom - span.top
        span.width = span.right - span.left

    return span_list


def tessocr_img2text_linemode(img, span_list=None, combine_lines=True, show_process=False, gen_data=False, lang="comic6k", jpn_vert=False):
    if jpn_vert:
        lang = "jpn_vert"
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    hig = img.shape[0]
    wid = img.shape[1]
    if hig * wid < 5:
        return '', -1, -1

    bw = 3
    text = ''
    alignment, vars = 0, (-1, -1)
    if span_list is None:
        span_list, vars = split_textblock(img, show_process=show_process)
        _, maxspan = find_span(span_list, max)
        maxh = bw*2 + maxspan.height
    else:
        maxh = max([s[2]-s[0] for s in span_list])
        maxh = bw*2 + maxh
    
    long_line = []
    word_space = int(round(maxh / 8))
    img = 255 - img
    for ind, s in enumerate(span_list):
        if isinstance(s, list):
            im = img[s[0]: s[2], s[1]: s[3]]
        else:
            im = img[s.top: s.bottom, s.left: s.right]
        
        hw1 = int(round((maxh - im.shape[0])/2))
        hw2 = maxh - hw1 - im.shape[0]
        dst = cv2.copyMakeBorder(im, hw1, hw2, word_space, word_space, cv2.BORDER_CONSTANT, None, value=[255, 255, 255])

        if not combine_lines:
            text += tessocr_img2text(dst, lang=lang) +'\n'
        else:
            long_line.append(dst)
        if show_process:
            cv2.imshow(str(ind), dst)

    if combine_lines:
        long_line = cv2.hconcat(long_line)
        if jpn_vert:
            long_line = cv2.rotate(long_line, cv2.ROTATE_90_CLOCKWISE) 
        if show_process:
            cv2.namedWindow("long line:", cv2.WINDOW_NORMAL)
            cv2.imshow("long line:", long_line)
        if gen_data:
            return long_line
        res = tessocr_img2text(long_line, lang=lang)
    mean_height = -1
    if len(span_list) != 0:
        if isinstance(span_list[0], list):
            mean_height = np.mean(np.array([s[2]-s[0] for s in span_list]))
        else:
            mean_height = np.mean(np.array([s.height for s in span_list]))
        alignment = 1 if vars[1] < vars[0] else 0
    return res, mean_height, alignment
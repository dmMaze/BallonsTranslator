import cv2
import numpy as np
from typing import Tuple
from .imgproc_utils import draw_connected_labels
from .stroke_width_calculator import strokewidth_check

opencv_inpaint = lambda img, mask: cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

def show_img_by_dict(imgdicts):
    for keyname in imgdicts.keys():
        cv2.imshow(keyname, imgdicts[keyname])
    cv2.waitKey(0)

# 计算文本bgr均值
def letter_calculator(img, mask, bground_bgr, show_process=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bgr to grey
    aver_bground_bgr = 0.114 * bground_bgr[0] + 0.587 * bground_bgr[1] + 0.299 * bground_bgr[2]
    thresh_low = 127
    retval, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)

    if aver_bground_bgr < thresh_low:
        threshed = 255 - threshed
    threshed = 255 - threshed

    
    threshed = cv2.bitwise_and(threshed, mask)
    le_region = np.where(threshed==255)
    mat_region = img[le_region]

    if mat_region.shape[0] == 0:
        # retval, threshed = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        # cv2.imshow("xxx", threshed)
        # cv2.imshow("2xxx", img)
        # cv2.waitKey(0)
        return [-1, -1, -1], threshed
    
    letter_bgr = np.mean(mat_region, axis=0).astype(int).tolist()
    
    if show_process:
        cv2.imshow("thresh", threshed)
        # ocr_protest(threshed)
        imgcp = np.copy(img)
        imgcp *= 0
        imgcp += 127
        imgcp[le_region] = letter_bgr
        cv2.imshow("letter_img", imgcp)
        # cv2.waitKey(0)
        
    return letter_bgr, threshed

# 预处理让文本颜色提取准确点
def usm(src):
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    h, w = src.shape[:2]
    result = np.zeros([h, w*2, 3], dtype=src.dtype)
    result[0:h,0:w,:] = src
    result[0:h,w:2*w,:] = usm
    return usm

# 计算文本bgr均值方法2，可能用中位数代替均值会好点
def textbgr_calculator(img, text_mask, show_process=False):
    text_mask = cv2.erode(text_mask, (3, 3), iterations=1)
    usm_img = usm(img)
    overall_meanbgr = np.mean(usm_img[np.where(text_mask==255)], axis=0)
    if show_process:
        colored_text_board = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) + 127
        colored_text_board[np.where(text_mask==255)] = overall_meanbgr
        cv2.imshow("usm", usm_img)
        cv2.imshow("textcolor", colored_text_board)
    return overall_meanbgr.astype(np.uint8)

# 计算背景bgr均值和标准差
def bground_calculator(buble_img, back_ground_mask, dilate=True):
    kernel = np.ones((3,3),np.uint8)
    if dilate:
        back_ground_mask = cv2.dilate(back_ground_mask, kernel, iterations = 1)
    bground_region = np.where(back_ground_mask==0)
    sd = -1
    if len(bground_region[0]) != 0:
        pix_array = buble_img[bground_region]
        bground_aver = np.mean(pix_array, axis=0).astype(int)
        pix_array - bground_aver
        gray = cv2.cvtColor(buble_img, cv2.COLOR_BGR2GRAY)
        gray_pixarray = gray[bground_region]
        gray_aver = np.mean(gray_pixarray)
        gray_pixarray = gray_pixarray - gray_aver
        gray_pixarray = np.power(gray_pixarray, 2)
        # gray_pixarray = np.sqrt(gray_pixarray)
        sd = np.mean(gray_pixarray)
    else: bground_aver = np.array([-1, -1, -1])

    return bground_aver, bground_region, sd

# 输入：文本块roi，分割出文本mask，根据mask计算文本bgr均值和标准差，决定纯色覆盖/inpaint修复
def canny_flood(img, show_process=False, inpaint_sdthresh=10, **kwargs):
    # cv2.setNumThreads(4)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    kernel = np.ones((3,3),np.uint8)
    orih, oriw = img.shape[0], img.shape[1]
    scaleR = 1
    if orih > 300 and oriw > 300:
        scaleR = 0.6
    elif orih < 120 or oriw < 120:
        scaleR = 1.4

    if scaleR != 1:
        h, w = img.shape[0], img.shape[1]
        orimg = np.copy(img)
        img = cv2.resize(img, (int(w*scaleR), int(h*scaleR)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[0], img.shape[1]
    img_area = h * w

    cpimg = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    detected_edges = cv2.Canny(cpimg, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)

    cons, hiers = cv2.findContours(detected_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), BLACK, 1, cv2.LINE_8)

    ballon_mask, outer_index = np.zeros((h, w), np.uint8), -1

    min_retval = np.inf
    mask = np.zeros((h, w), np.uint8)
    difres = 10
    seedpnt = (int(w/2), int(h/2))
    for ii in range(len(cons)):
        rect = cv2.boundingRect(cons[ii])
        if rect[2]*rect[3] < img_area*0.4:
            continue
        
        mask = cv2.drawContours(mask, cons, ii, (255), 2)
        cpmask = np.copy(mask)
        cv2.rectangle(mask, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)
        retval, _, _, rect = cv2.floodFill(cpmask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))

        if retval <= img_area * 0.3:
            mask = cv2.drawContours(mask, cons, ii, (0), 2)
        if retval < min_retval and retval > img_area * 0.3:
            min_retval = retval
            ballon_mask = cpmask

    ballon_mask = 127 - ballon_mask
    ballon_mask = cv2.dilate(ballon_mask, kernel,iterations = 1)
    outer_area, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(30), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
    ballon_mask = 30 - ballon_mask    
    retval, ballon_mask = cv2.threshold(ballon_mask, 1, 255, cv2.THRESH_BINARY)
    ballon_mask = cv2.bitwise_not(ballon_mask, ballon_mask)

    detected_edges = cv2.dilate(detected_edges, kernel, iterations = 1)
    for ii in range(2):
        detected_edges = cv2.bitwise_and(detected_edges, ballon_mask)
        mask = np.copy(detected_edges)
        bgarea1, _, _, rect = cv2.floodFill(mask, mask=None, seedPoint=(0, 0),  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
        bgarea2, _, _, rect = cv2.floodFill(mask, mask=None, seedPoint=(detected_edges.shape[1]-1, detected_edges.shape[0]-1),  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
        txt_area = min(img_area - bgarea1, img_area - bgarea2)
        ratio_ob = txt_area / outer_area
        ballon_mask = cv2.erode(ballon_mask, kernel,iterations = 1)
        if ratio_ob < 0.85:
            break

    mask = 127 - mask
    retval, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    if scaleR != 1:
        img = orimg
        ballon_mask = cv2.resize(ballon_mask, (oriw, orih))
        mask = cv2.resize(mask, (oriw, orih))

    bg_mask = cv2.bitwise_or(mask, 255-ballon_mask)
    mask = cv2.bitwise_and(mask, ballon_mask)

    bground_aver, bground_region, sd = bground_calculator(img, bg_mask)
    inner_rect = None
    threshed = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    if bground_aver[0] != -1:
        letter_aver, threshed = letter_calculator(img, mask, bground_aver, show_process=show_process)
        if letter_aver[0] != -1:
            mask = cv2.dilate(threshed, kernel, iterations=1)
            inner_rect = cv2.boundingRect(cv2.findNonZero(mask))
    else: letter_aver = [0, 0, 0]

    if sd != -1 and sd < inpaint_sdthresh:
        need_inpaint = False
    else:
        need_inpaint = True
    if show_process:
        print(f"\nneed_inpaint: {need_inpaint}, sd: {sd}, {type(inner_rect)}")
        show_img_by_dict({"outermask": ballon_mask, "detect": detected_edges, "mask": mask})


    if isinstance(inner_rect, tuple):
        inner_rect = [ii for ii in inner_rect]
    if inner_rect is None:
        inner_rect = [-1, -1, -1, -1]
    else:
        inner_rect.append(-1)
    
    bground_aver = bground_aver.astype(np.uint8)
    bub_dict = {"bgr": letter_aver,
                "bground_bgr": bground_aver,
                "inner_rect": inner_rect,
                "need_inpaint": need_inpaint}
    return mask, ballon_mask, bub_dict

# 输入：文本块roi，分割出文本mask，根据mask计算文本bgr均值和标准差，决定纯色覆盖/inpaint修复
def connected_canny_flood(img, show_process=False, inpaint_sdthresh=10, apply_strokewidth_check=0, **kwargs):

    # 寻找最可能是气泡的外轮廓mask
    def find_outermask(img):
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_16U)
        drawtext = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        
        max_ind = np.argmax(stats[:, 4])
        maxbbox_area, sec_ind = -1, -1
        for ind, stat in enumerate(stats):
            if ind != max_ind:
                bbarea = stat[2] * stat[3]
                if bbarea > maxbbox_area:
                    maxbbox_area = bbarea
                    sec_ind = ind
        drawtext[np.where(labels==max_ind)] = 255
        
        cv2.rectangle(drawtext, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 0, 0), 1, cv2.LINE_8)
        cons, hiers = cv2.findContours(drawtext, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        img_area = img.shape[0] * img.shape[1]

        rects = np.array([cv2.boundingRect(cnt) for cnt in cons])
        rect_area = np.array([rect[2] * rect[3] for rect in rects])
        quali_ind = np.where(rect_area > img_area * 0.3)[0]
        ballon_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for ind in quali_ind:
            ballon_mask = cv2.drawContours(ballon_mask, cons, ind, (255), 2)
        
        seedpnt = (int(ballon_mask.shape[1]/2), int(ballon_mask.shape[0]/2))
        difres = 10
        retval, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
        ballon_mask = 255 - cv2.threshold(ballon_mask - 127, 1, 255, cv2.THRESH_BINARY)[1]
        return num_labels, labels, stats, centroids, ballon_mask

    # BGR直接转灰度图可能导致文本区域和背景难以区分，比如测试样例中的黑底红字
    # 但是总有一个通道文本和背景容易区分
    # 返回最容易区分的那个通道
    def ccctest(img, crop_r=0.1):
        # img = usm(img)
        maxh = 100
        if img.shape[0] > maxh:
            scaleR = maxh / img.shape[0]
            im = cv2.resize(img, (int(img.shape[1]*scaleR), int(img.shape[0]*scaleR)), interpolation=cv2.INTER_AREA)
        else:
            im = img

        textlabel_counter = 0
        reverse = False
        c_ind = 0

        num_labels, labels, stats, centroids, pseduo_outermask = find_outermask(cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1])
        grayim = np.expand_dims(np.array(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)), axis=2)
        im = np.append(im, grayim, axis=2)
        outer_cords = np.where(pseduo_outermask==255)
        for bgr_ind in range(4):
            channel = im[:, :, bgr_ind]
            ret, thresh = cv2.threshold(channel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

            tmp_reverse = False
            
            if np.mean(thresh[outer_cords]) > 160:
                thresh = 255 - thresh
                tmp_reverse = True

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_16U)
            # draw_connected_labels(num_labels, labels, stats, centroids)
            # cv2.waitKey(0)
            max_ind = np.argmax(stats[:, 4])
            maxr, minr = 0.5, 0.001
            maxw, maxh = stats[max_ind][2] * maxr, stats[max_ind][3] * maxr
            minarea = im.shape[0] * im.shape[1] * minr

            tmp_counter = 0
            for stat in stats:
                bboxarea = stat[2] * stat[3]
                if stat[2] < maxw and stat[3] < maxh and bboxarea > minarea:
                    tmp_counter += 1
            if tmp_counter > textlabel_counter:
                textlabel_counter = tmp_counter
                c_ind = bgr_ind
                reverse = tmp_reverse
        return c_ind, reverse
    
    channel_index, reverse = ccctest(img)
    chanel = img[:, :, channel_index] if channel_index < 3 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(chanel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    # reverse to get white text on black bg
    if reverse:
        thresh = 255 - thresh
    num_labels, labels, stats, centroids, ballon_mask = find_outermask(thresh)
    img_area = img.shape[0] * img.shape[1]
    text_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    max_ind = np.argmax(stats[:, 4])
    for lab in (range(num_labels)):
        stat = stats[lab]
        if lab != max_ind and stat[4] < img_area * 0.4:
            labcord = np.where(labels==lab)
            text_mask[labcord] = 255

    text_mask = cv2.bitwise_and(text_mask, ballon_mask)
    if apply_strokewidth_check > 0:
        text_mask = strokewidth_check(text_mask, labels, num_labels, stats, debug_type=show_process-1)
        
    text_color = textbgr_calculator(img, text_mask, show_process=show_process)
    inner_rect = cv2.boundingRect(cv2.findNonZero(cv2.dilate(text_mask, (3, 3), iterations=1)))
    inner_rect = [ii for ii in inner_rect]
    inner_rect.append(-1)

    bg_mask = cv2.bitwise_or(text_mask, 255-ballon_mask)

    bground_aver, bground_region, sd = bground_calculator(img, bg_mask)

    mask = cv2.GaussianBlur(text_mask,(3,3),cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    if sd != -1 and sd < inpaint_sdthresh:
        need_inpaint = False
    else:
        need_inpaint = True

    if show_process:
        print(f"\nuse inpaint: {need_inpaint}, sd: {sd}, {type(inner_rect)}")
        draw_connected_labels(num_labels, labels, stats, centroids)
        show_img_by_dict({"thresh": thresh, "ori": img, "outer": ballon_mask, "text": text_mask, "bgmask": bg_mask})

    bground_aver = bground_aver.astype(np.uint8)
    bub_dict = {"bgr": text_color,
                "bground_bgr": bground_aver,
                "inner_rect": inner_rect,
                "need_inpaint": need_inpaint}
    return mask, ballon_mask, bub_dict


def existing_mask(img, mask: np.ndarray):
    bub_dict = {"bgr": [0, 0, 0],"bground_bgr": [255, 255, 255],"need_inpaint": True}
    return mask, mask, bub_dict


def extract_ballon_mask(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Given original img and text mask (cropped)
    return ballon mask & non text mask
    '''
    img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    h, w = img.shape[:2]
    text_sum = np.sum(mask)
    cannyed = cv2.Canny(img, 70, 140, L2gradient=True, apertureSize=3)
    e_size = 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
    cannyed = cv2.dilate(cannyed, element, iterations=1)
    br = cv2.boundingRect(cv2.findNonZero(mask))
    br_xyxy = [br[0], br[1], br[0] + br[2], br[1] + br[3]]

    # draw the bounding rect in case there is no closed ballon
    cv2.rectangle(cannyed, (0, 0), (w-1, h-1), (255, 255, 255), 1, cv2.LINE_8)
    cannyed = cv2.bitwise_and(cannyed, 255 - mask)

    cons, _ = cv2.findContours(cannyed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    min_ballon_area = w * h
    ballon_mask = None
    non_text_mask = None
    # minimum contour which covers all text mask must be the ballon
    for ii, con in enumerate(cons):
        br_c = cv2.boundingRect(con)
        br_c = [br_c[0], br_c[1], br_c[0] + br_c[2], br_c[1] + br_c[3]]
        if br_c[0] > br_xyxy[0] or br_c[1] > br_xyxy[1] or br_c[2] < br_xyxy[2] or br_c[3] < br_xyxy[3]:
            continue
        tmp = np.zeros_like(cannyed)
        cv2.drawContours(tmp, cons, ii, (255, 255, 255), -1, cv2.LINE_8)
        if cv2.bitwise_and(tmp, mask).sum() >= text_sum:
            con_area = cv2.contourArea(con)
            if con_area < min_ballon_area:
                min_ballon_area = con_area
                ballon_mask = tmp
    if ballon_mask is not None:
        non_text_mask = cv2.bitwise_and(ballon_mask, 255 - mask)
    #     cv2.imshow('ballon', ballon_mask)
    #     cv2.imshow('non_text', non_text_mask)
    # cv2.imshow('im', img)
    # cv2.imshow('msk', mask)
    # cv2.imshow('canny', cannyed)
    # cv2.waitKey(0)

    return ballon_mask, non_text_mask
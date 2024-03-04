import numpy as np
import cv2
import random
from typing import List, Tuple, Union

def hex2bgr(hex):
    gmask = 254 << 8
    rmask = 254
    b = hex >> 16
    g = (hex & gmask) >> 8
    r = hex & rmask
    return np.stack([b, g, r]).transpose()

def union_area(bboxa, bboxb):
    x1 = max(bboxa[0], bboxb[0])
    y1 = max(bboxa[1], bboxb[1])
    x2 = min(bboxa[2], bboxb[2])
    y2 = min(bboxa[3], bboxb[3])
    if y2 < y1 or x2 < x1:
        return -1
    return (y2 - y1) * (x2 - x1)

def get_yololabel_strings(clslist, labellist):
    content = ''
    for cls, xywh in zip(clslist, labellist):
        content += str(int(cls)) + ' ' + ' '.join([str(e) for e in xywh]) + '\n'
    if len(content) != 0:
        content = content[:-1]
    return content

# 4 points bbox to 8 points polygon
def xywh2xyxypoly(xywh, to_int=True):
    xyxypoly = np.tile(xywh[:, [0, 1]], 4)
    xyxypoly[:, [2, 4]] += xywh[:, [2]]
    xyxypoly[:, [5, 7]] += xywh[:, [3]]
    if to_int:
        xyxypoly = xyxypoly.astype(np.int64)
    return xyxypoly

def xyxy2yolo(xyxy, w: int, h: int):
    if xyxy == [] or xyxy == np.array([]) or len(xyxy) == 0:
        return None
    if isinstance(xyxy, list):
        xyxy = np.array(xyxy)
    if len(xyxy.shape) == 1:
        xyxy = np.array([xyxy])
    yolo = np.copy(xyxy).astype(np.float64)
    yolo[:, [0, 2]] =  yolo[:, [0, 2]] / w
    yolo[:, [1, 3]] = yolo[:, [1, 3]] / h
    yolo[:, [2, 3]] -= yolo[:, [0, 1]]
    yolo[:, [0, 1]] += yolo[:, [2, 3]] / 2
    return yolo

def yolo_xywh2xyxy(xywh: np.array, w: int, h:  int, to_int=True):
    if xywh is None:
        return None
    if len(xywh) == 0:
        return None
    if len(xywh.shape) == 1:
        xywh = np.array([xywh])
    xywh[:, [0, 2]] *= w
    xywh[:, [1, 3]] *= h
    xywh[:, [0, 1]] -= xywh[:, [2, 3]] / 2
    xywh[:, [2, 3]] += xywh[:, [0, 1]]
    if to_int:
        xywh = xywh.astype(np.int64)
    return xywh

def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)
    
    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=128):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # dw /= 2  # divide padding into 2 sides
    # dh /= 2
    dh, dw = int(dh), int(dw)

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def resize_keepasp(im, new_shape=640, scaleup=True, interpolation=cv2.INTER_LINEAR, stride=None):
    shape = im.shape[:2]  # current shape [height, width]

    if new_shape is not None:
        if not isinstance(new_shape, tuple):
            new_shape = (new_shape, new_shape)
    else:
        new_shape = shape

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    if stride is not None:
        h, w = new_unpad
        if h % stride != 0 :
            new_h = (stride - (h % stride)) + h
        else :
            new_h = h
        if w % stride != 0 :
            new_w = (stride - (w % stride)) + w
        else :
            new_w = w
        new_unpad = (new_h, new_w)
        
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=interpolation)
    return im

def expand_textwindow(img_size, xyxy, expand_r=8, shrink=False):
    im_h, im_w = img_size[:2]
    x1, y1 , x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    paddings = int(round((max(h, w) * 0.25 + min(h, w) * 0.75) / expand_r))
    if shrink:
        paddings *= -1
    x1, y1 = max(0, x1 - paddings), max(0, y1 - paddings)
    x2, y2 = min(im_w-1, x2+paddings), min(im_h-1, y2+paddings)
    return [x1, y1, x2, y2]

def enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0) -> List:
    assert ratio > 1.0
    
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]

    # https://numpy.org/doc/stable/reference/generated/numpy.roots.html
    coeff = [aspect_ratio, w+h*aspect_ratio, (1-ratio)*w*h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(roots[-1] / 2 ))
    delta_w = int(delta * aspect_ratio)
    delta_w = min(x1, im_w - x2, delta_w)
    delta = min(y1, im_h - y2, delta)
    rect = np.array([x1-delta_w, y1-delta, x2+delta_w, y2+delta], dtype=np.int64)
    rect[::2] = np.clip(rect[::2], 0, im_w)
    rect[1::2] = np.clip(rect[1::2], 0, im_h)
    return rect.tolist()

def draw_connected_labels(num_labels, labels, stats, centroids, names="draw_connected_labels", skip_background=True):
    labdraw = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    max_ind = 0
    if isinstance(num_labels, int):
        num_labels = range(num_labels)
    
    # for ind, lab in enumerate((range(num_labels))):
    for lab in num_labels:
        if skip_background and lab == 0:
            continue
        randcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        labdraw[np.where(labels==lab)] = randcolor
        maxr, minr = 0.5, 0.001
        maxw, maxh = stats[max_ind][2] * maxr, stats[max_ind][3] * maxr
        minarea = labdraw.shape[0] * labdraw.shape[1] * minr

        stat = stats[lab]
        bboxarea = stat[2] * stat[3]
        if stat[2] < maxw and stat[3] < maxh and bboxarea > minarea:
            pix = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)
            pix[np.where(labels==lab)] = 255

            rect = cv2.minAreaRect(cv2.findNonZero(pix))
            box = np.int0(cv2.boxPoints(rect))
            labdraw = cv2.drawContours(labdraw, [box], 0, randcolor, 2)
            labdraw = cv2.circle(labdraw, (int(centroids[lab][0]),int(centroids[lab][1])), radius=5, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), thickness=-1)                

    cv2.imshow(names, labdraw)
    return labdraw

def rotate_image(mat: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    # https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def color_difference(rgb1: List, rgb2: List) -> float:
    # https://en.wikipedia.org/wiki/Color_difference#CIE76
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float64) - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float64)
    diff[..., 0] *= 0.392
    diff = np.linalg.norm(diff, axis=2) 
    return diff.item()

def extract_ballon_region(img: np.ndarray, ballon_rect: List, show_process=False, enlarge_ratio=2.0, cal_region_rect=False) -> Tuple[np.ndarray, int, List]:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    x1, y1, x2, y2 = ballon_rect[0], ballon_rect[1], \
        ballon_rect[2] + ballon_rect[0], ballon_rect[3] + ballon_rect[1]
    if enlarge_ratio > 1:
        x1, y1, x2, y2 = enlarge_window([x1, y1, x2, y2], img.shape[1], img.shape[0], enlarge_ratio, aspect_ratio=ballon_rect[3] / ballon_rect[2])

    img = img[y1:y2, x1:x2].copy()

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
    ballon_area, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(30), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
    ballon_mask = 30 - ballon_mask    
    retval, ballon_mask = cv2.threshold(ballon_mask, 1, 255, cv2.THRESH_BINARY)
    ballon_mask = cv2.bitwise_not(ballon_mask, ballon_mask)

    box_kernel = int(np.sqrt(ballon_area) / 30)
    if box_kernel > 1:
        box_kernel = np.ones((box_kernel,box_kernel),np.uint8)
        ballon_mask = cv2.dilate(ballon_mask, box_kernel, iterations = 1)
        ballon_mask = cv2.erode(ballon_mask, box_kernel, iterations = 1)

    if scaleR != 1:
        img = orimg
        ballon_mask = cv2.resize(ballon_mask, (oriw, orih))

    if show_process:
        cv2.imshow('ballon_mask', ballon_mask)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    if cal_region_rect:
        return ballon_mask, (ballon_mask > 0).sum(), [x1, y1, x2, y2], cv2.boundingRect(ballon_mask)
    return ballon_mask, (ballon_mask > 0).sum(), [x1, y1, x2, y2]

def square_pad_resize(img: np.ndarray, tgt_size: int):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    
    # make square image
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h

    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size

    if pad_h > 0 or pad_w > 0:    
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)

    down_scale_ratio = tgt_size / img.shape[0]
    assert down_scale_ratio <= 1
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_AREA)

    return img, down_scale_ratio, pad_h, pad_w



def get_block_mask(xywh: List, mask_array: np.ndarray, angle: int):
    x, y, w, h = xywh
    im_h, im_w = mask_array.shape[:2]

    if angle != 0:
        cx, cy = x + int(round(w / 2)), y + int(round(h / 2))
        poly = xywh2xyxypoly(np.array([[x, y, w, h]]))
        poly = rotate_polygons([cx, cy], poly, -angle)
        
        x1, x2 = np.min(poly[..., ::2]), np.max(poly[..., ::2])
        y1, y2 = np.min(poly[..., 1::2]), np.max(poly[..., 1::2])
        
        if x2 < 0 or x2 - x1 < 2 or x1 >= im_w - 1 \
            or y2 < 0 or y2 - y1 < 2 or y1 >= im_h - 1:
            return None, None
        else:
            poly[..., ::2] -= cx - int((x2 - x1) / 2)
            poly[..., 1::2] -= cy - int((y2 - y1) / 2)
            itmsk = np.zeros((y2 - y1, x2 - x1), np.uint8)
            
            cv2.fillPoly(itmsk, poly.reshape(-1, 4, 2), color=(255))
            px1, px2, py1, py2 = 0, itmsk.shape[1], 0, itmsk.shape[0]
            if x1 < 0:
                px1 = -x1
                x1 = 0
            if x2 > im_w:
                px2 = im_w - x2
                x2 = im_w
            if y1 < 0:
                py1 = -y1
                y1 = 0
            if y2 > im_h:
                py2 = im_h - y2
                y2 = im_h
            itmsk = itmsk[py1: py2, px1: px2]
            msk = cv2.bitwise_and(mask_array[y1: y2, x1: x2], itmsk)
    else:
        x1, y1, x2, y2 = x, y, x+w, y+h
        if x2 < 0 or x2 - x1 < 2 or x1 >= im_w - 1 \
            or y2 < 0 or y2 - y1 < 2 or y1 >= im_h - 1:
            return None, None
        else:
            if x1 < 0:
                x1 = 0
            if x2 > im_w:
                x2 = im_w
            if y1 < 0:
                y1 = 0
            if y2 > im_h:
                y2 = im_h
            msk = mask_array[y1: y2, x1: x2]

    return msk, [x1, y1, x2, y2]
        
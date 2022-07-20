import json
from .basemodel import TextDetBase, TextDetBaseDNN
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import torch
from pathlib import Path
import torch

from utils.io_utils import find_all_imgs, NumpyEncoder
from utils.imgproc_utils import letterbox, xyxy2yolo, get_yololabel_strings

from ..yolov5.yolov5_utils import non_max_suppression
from ..db_utils import SegDetectorRepresenter
from ..textblock import TextBlock, group_output
from .textmask import refine_mask, refine_undetected_mask, REFINEMASK_INPAINT, REFINEMASK_ANNOTATION
from pathlib import Path
from typing import Union, List, Tuple

CTD_MODEL_PATH = r'data/models/comictextdetector.pt'

def model2annotations(model_path, img_dir_list, save_dir, save_json=False):
    if isinstance(img_dir_list, str):
        img_dir_list = [img_dir_list]
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model = TextDetector(model_path=model_path, detect_size=1024, device=device, act='leaky')  
    imglist = []
    for img_dir in img_dir_list:
        imglist += find_all_imgs(img_dir, abs_path=True)
    for img_path in tqdm(imglist):
        imgname = osp.basename(img_path)
        img = cv2.imread(img_path)
        im_h, im_w = img.shape[:2]
        imname = imgname.replace(Path(imgname).suffix, '')
        maskname = 'mask-'+imname+'.png'
        poly_save_path = osp.join(save_dir, 'line-' + imname + '.txt')
        mask, mask_refined, blk_list = model(img, refine_mode=REFINEMASK_ANNOTATION, keep_undetected_mask=True)
        polys = []
        blk_xyxy = []
        blk_dict_list = []
        for blk in blk_list:
            polys += blk.lines
            blk_xyxy.append(blk.xyxy)
            blk_dict_list.append(blk.to_dict(extra_info=True))
        blk_xyxy = xyxy2yolo(blk_xyxy, im_w, im_h)
        if blk_xyxy is not None:
            cls_list = [1] * len(blk_xyxy)
            yolo_label = get_yololabel_strings(cls_list, blk_xyxy)
        else:
            yolo_label = ''
        with open(osp.join(save_dir, imname+'.txt'), 'w', encoding='utf8') as f:
            f.write(yolo_label)

        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        # _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        # draw_connected_labels(num_labels, labels, stats, centroids)
        # visualize_textblocks(img, blk_list)
        # cv2.imshow('rst', img)
        # cv2.imshow('mask', mask)
        # cv2.imshow('mask_refined', mask_refined)
        # cv2.waitKey(0)

        if len(polys) != 0:
            if isinstance(polys, list):
                polys = np.array(polys)
            polys = polys.reshape(-1, 8)
            np.savetxt(poly_save_path, polys, fmt='%d')
        if save_json:
            with open(osp.join(save_dir, imname+'.json'), 'w', encoding='utf8') as f:
                f.write(json.dumps(blk_dict_list, ensure_ascii=False, cls=NumpyEncoder))
        cv2.imwrite(osp.join(save_dir, imgname), img)
        cv2.imwrite(osp.join(save_dir, maskname), mask_refined)

def preprocess_img(img, detect_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=detect_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)

def postprocess_mask(img: Union[torch.Tensor, np.ndarray], thresh=None):
    # img = img.permute(1, 2, 0)
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach_().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    # if isinstance(img, torch.Tensor):

    return img.astype(np.uint8)

def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    # bbox = det[..., 0:4]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs

class TextDetector:
    lang_list = ['eng', 'ja', 'unknown']
    langcls2idx = {'eng': 0, 'ja': 1, 'unknown': 2}

    def __init__(self, model_path, detect_size=1024, device='cpu', half=False, nms_thresh=0.35, conf_thresh=0.4):
        super(TextDetector, self).__init__()

        self.net: Union[TextDetBase, TextDetBaseDNN] = None
        self.backend: str = None
        
        if isinstance(detect_size, int):
            detect_size = (detect_size, detect_size)
        self.detect_size = detect_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

        self.backend = 'torch'
        self.load_model(model_path)

    def load_model(self, model_path: str):
        if Path(model_path).suffix == '.onnx':
            self.net = TextDetBaseDNN(1024, model_path)
            self.backend = 'opencv'
        else:
            self.net = TextDetBase(model_path, device=self.device, act='leaky', half=self.half)
            self.backend = 'torch'

    def set_device(self, device: str):
        if self.device == device:
            return
        model_path = CTD_MODEL_PATH+'.onnx' if device == 'cpu' else CTD_MODEL_PATH
        if not osp.exists(model_path):
            raise FileNotFoundError(f'CTD model not found: {model_path}')
        self.load_model(model_path)

    @torch.no_grad()
    def __call__(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False) -> Tuple[np.ndarray, np.ndarray, List[TextBlock]]:
        detect_size = self.detect_size if self.backend == 'torch' else (1024, 1024)
        img_in, ratio, dw, dh = preprocess_img(img, detect_size=detect_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')

        im_h, im_w = img.shape[:2]

        blks, mask, lines_map = self.net(img_in)
        
        resize_ratio = (im_w / (detect_size[0] - dw), im_h / (detect_size[1] - dh))
        blks = postprocess_yolo(blks, self.conf_thresh, self.nms_thresh, resize_ratio)
        mask = postprocess_mask(mask)
        lines, scores = self.seg_rep(detect_size, lines_map)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]
        
        # map output to input img
        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0 :
            lines = []
        else :
            lines = lines.astype(np.float64)
            lines[..., 0] *= resize_ratio[0]
            lines[..., 1] *= resize_ratio[1]
            lines = lines.astype(np.int32)
        blk_list = group_output(blks, lines, im_w, im_h, mask)
        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
        if keep_undetected_mask:
            mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)
    
        return mask, mask_refined, blk_list
import json
from .basemodel import TextDetBase, TextDetBaseDNN
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import torch
from pathlib import Path
import einops

from utils.io_utils import find_all_imgs, NumpyEncoder
from utils.imgproc_utils import letterbox, xyxy2yolo, get_yololabel_strings, square_pad_resize

from ..yolov5.yolov5_utils import non_max_suppression
from ..db_utils import SegDetectorRepresenter
from utils.textblock import TextBlock, group_output, mit_merge_textlines
from .textmask import refine_mask, refine_undetected_mask, REFINEMASK_INPAINT, REFINEMASK_ANNOTATION
from pathlib import Path
from typing import Union, List, Tuple, Callable

CTD_MODEL_PATH = r'data/models/comictextdetector.pt'

def det_rearrange_forward(
    img: np.ndarray, 
    dbnet_batch_forward: Callable[[np.ndarray, str], Tuple[np.ndarray, np.ndarray]], 
    tgt_size: int = 1280, 
    max_batch_size: int = 4, 
    device='cuda', 
    crop_as_square=False, verbose=False):
    '''
    Rearrange image to square batches before feeding into network if following conditions are satisfied: \n
    1. Extreme aspect ratio
    2. Is too tall or wide for detect size (tgt_size)

    Returns:
        DBNet output, mask or None, None if rearrangement is not required
    '''

    def _unrearrange(patch_lst: List[np.ndarray], transpose: bool, channel=1, pad_num=0):
        _psize = _h = patch_lst[0].shape[-1]
        _step = int(ph_step * _psize / patch_size)
        _pw = int(_psize / pw_num)
        _h = int(_pw / w * h)
        tgtmap = np.zeros((channel, _h, _pw), dtype=np.float32)
        num_patches = len(patch_lst) * pw_num - pad_num
        for ii, p in enumerate(patch_lst):
            if transpose:
                p = einops.rearrange(p, 'c h w -> c w h')
            for jj in range(pw_num):
                pidx = ii * pw_num + jj
                rel_t = rel_step_list[pidx]
                t = int(round(rel_t * _h))
                b = min(t + _psize, _h)
                l = jj * _pw
                r = l + _pw
                tgtmap[..., t: b, :] += p[..., : b - t, l: r]
                if pidx > 0:
                    interleave = _psize - _step
                    tgtmap[..., t: t+interleave, :] /= 2.

                if pidx >= num_patches - 1:
                    break

        if transpose:
            tgtmap = einops.rearrange(tgtmap, 'c h w -> c w h')
        return tgtmap[None, ...]

    def _patch2batches(patch_lst: List[np.ndarray], p_num: int, transpose: bool):
        if transpose:
            patch_lst = einops.rearrange(patch_lst, '(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c', p_num=p_num)
        else:
            patch_lst = einops.rearrange(patch_lst, '(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c', p_num=p_num)
        
        batches = [[]]
        for ii, patch in enumerate(patch_lst):

            if len(batches[-1]) >= max_batch_size:
                batches.append([])
            p, down_scale_ratio, pad_h, pad_w = square_pad_resize(patch, tgt_size=tgt_size)

            assert pad_h == pad_w
            pad_size = pad_h
            batches[-1].append(p)
            if verbose:
                cv2.imwrite(f'result/rearrange_{ii}.png', p[..., ::-1])
        return batches, down_scale_ratio, pad_size

    h, w = img.shape[:2]
    transpose = False
    if h < w:
        transpose = True
        h, w = img.shape[1], img.shape[0]

    asp_ratio = h / w
    down_scale_ratio = h / tgt_size

    # rearrange condition
    require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
    if not require_rearrange:
        return None, None

    if verbose:
        print(f'Input image will be rearranged to square batches before fed into network.\
            \n Rearranged batches will be saved to result/rearrange_%d.png')

    if transpose:
        img = einops.rearrange(img, 'h w c -> w h c')
    
    if crop_as_square:
        pw_num = 1
    else:
        pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
    patch_size = ph = pw_num * w

    ph_num = int(np.ceil(h / ph))
    ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
    rel_step_list = []
    patch_list = []
    for ii in range(ph_num):
        t = ii * ph_step
        b = t + ph
        rel_step_list.append(t / h)
        patch_list.append(img[t: b])

    p_num = int(np.ceil(ph_num / pw_num))
    pad_num = p_num * pw_num - ph_num
    for ii in range(pad_num):
        patch_list.append(np.zeros_like(patch_list[0]))

    batches, down_scale_ratio, pad_size = _patch2batches(patch_list, p_num, transpose)

    db_lst, mask_lst = [], []
    for batch in batches:
        batch = np.array(batch)
        db, mask = dbnet_batch_forward(batch, device=device)

        for ii, (d, m) in enumerate(zip(db, mask)):
            if pad_size > 0:
                paddb = int(db.shape[-1] / tgt_size * pad_size)
                padmsk = int(mask.shape[-1] / tgt_size * pad_size)
                d = d[..., :-paddb, :-paddb]
                m = m[..., :-padmsk, :-padmsk]
            db_lst.append(d)
            mask_lst.append(m)
            if verbose:
                cv2.imwrite(f'result/rearrange_db_{ii}.png', (d[0] * 255).astype(np.uint8))
                cv2.imwrite(f'result/rearrange_thr_{ii}.png', (d[1] * 255).astype(np.uint8))

    db = _unrearrange(db_lst, transpose, channel=2, pad_num=pad_num)
    mask = _unrearrange(mask_lst, transpose, channel=1, pad_num=pad_num)
    return db, mask

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
            blk_dict_list.append(blk.to_dict())
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
    if isinstance(detect_size, int):
        detect_size = (detect_size, detect_size)
    
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
            img = img.detach().cpu()
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

    def __init__(self, model_path, detect_size=1024, device='cpu', half=False, nms_thresh=0.35, conf_thresh=0.4, det_rearrange_max_batches=4):
        super(TextDetector, self).__init__()

        self.net: Union[TextDetBase, TextDetBaseDNN] = None
        self.backend: str = None
        
        self.detect_size = detect_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

        self.backend = 'torch'
        self.load_model(model_path)

        self.det_rearrange_max_batches = det_rearrange_max_batches

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

    def det_batch_forward_ctd(self, batch: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
        
        if isinstance(self.net, TextDetBase):
            batch = einops.rearrange(batch.astype(np.float32) / 255., 'n h w c -> n c h w')
            batch = torch.from_numpy(batch).to(device)
            _, mask, lines = self.net(batch)
            mask = mask.cpu().numpy()
            lines = lines.cpu().numpy()
        elif isinstance(self.net, TextDetBaseDNN):
            mask_lst, line_lst = [], []
            for b in batch:
                _, mask, lines = self.net(b)
                if mask.shape[1] == 2:     # some version of opencv spit out reversed result
                    tmp = mask
                    mask = lines
                    lines = tmp
                mask_lst.append(mask)
                line_lst.append(lines)
            lines, mask = np.concatenate(line_lst, 0), np.concatenate(mask_lst, 0)
        else:
            raise NotImplementedError
        return lines, mask

    @torch.no_grad()
    def __call__(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False) -> Tuple[np.ndarray, np.ndarray, List[TextBlock]]:
        
        detect_size = self.detect_size if not self.backend == 'opencv' else 1024
        im_h, im_w = img.shape[:2]
        lines_map, mask = det_rearrange_forward(img, self.det_batch_forward_ctd, detect_size, self.det_rearrange_max_batches, self.device)
        blks = []
        resize_ratio = [1, 1]
        if lines_map is None:
            img_in, ratio, dw, dh = preprocess_img(img, bgr2rgb=False, detect_size=detect_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')
            blks, mask, lines_map = self.net(img_in)
            if self.backend == 'opencv':
                if mask.shape[1] == 2:     # some version of opencv spit out reversed result
                    tmp = mask
                    mask = lines_map
                    lines_map = tmp
            mask = mask.squeeze()
            resize_ratio = (im_w / (detect_size - dw), im_h / (detect_size - dh))
            blks = postprocess_yolo(blks, self.conf_thresh, self.nms_thresh, resize_ratio)
            mask = mask[..., :mask.shape[0]-dh, :mask.shape[1]-dw]
            lines_map = lines_map[..., :lines_map.shape[2]-dh, :lines_map.shape[3]-dw]

        mask = postprocess_mask(mask)
        lines, scores = self.seg_rep(None, lines_map, height=im_h, width=im_w)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]

        # map output to input img
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0:
            lines = []
        else:
            lines = lines.astype(np.int64)
        blk_list = group_output([], lines, im_w, im_h, mask, canvas=img)
        # print(lines)
        # blk_list = mit_merge_textlines(lines, im_w, im_w)
        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
        if keep_undetected_mask:
            mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)

        return mask, mask_refined, blk_list
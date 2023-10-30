import numpy as np
import cv2
from typing import Dict, List
from collections import OrderedDict

from utils.registry import Registry
from utils.textblock_mask import extract_ballon_mask
from utils.imgproc_utils import enlarge_window

from ..base import BaseModule, DEFAULT_DEVICE, gc_collect, DEVICE_SELECTOR, GPUINTENSIVE_SET
from ..textdetector import TextBlock

INPAINTERS = Registry('inpainters')
register_inpainter = INPAINTERS.register_module


class InpainterBase(BaseModule):

    inpaint_by_block = True
    check_need_inpaint = True

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in INPAINTERS.module_dict:
            if INPAINTERS.module_dict[key] == self.__class__:
                self.name = key
                break
        self.setup_inpainter()

    def setup_inpainter(self):
        raise NotImplementedError

    def inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None, check_need_inpaint: bool = False) -> np.ndarray:
        if not self.inpaint_by_block or textblock_list is None:
            if check_need_inpaint:
                ballon_msk, non_text_msk = extract_ballon_mask(img, mask)
                if ballon_msk is not None:
                    non_text_region = np.where(non_text_msk > 0)
                    non_text_px = img[non_text_region]
                    average_bg_color = np.mean(non_text_px, axis=0)
                    std_bgr = np.std(non_text_px - average_bg_color, axis=0)
                    std_max = np.max(std_bgr)
                    inpaint_thresh = 7 if np.std(std_bgr) > 1 else 10
                    if std_max < inpaint_thresh:
                        img = img.copy()
                        img[np.where(ballon_msk > 0)] = average_bg_color
                        return img
            try:
                return self._inpaint(img, mask)
            except Exception as e:
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    gc_collect()
                    return self._inpaint(img, mask)
                else:
                    raise e
        else:
            im_h, im_w = img.shape[:2]
            inpainted = np.copy(img)
            for blk in textblock_list:
                xyxy = blk.xyxy
                xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=1.7)
                im = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                msk = mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                need_inpaint = True
                if self.check_need_inpaint or check_need_inpaint:
                    ballon_msk, non_text_msk = extract_ballon_mask(im, msk)
                    if ballon_msk is not None:
                        non_text_region = np.where(non_text_msk > 0)
                        non_text_px = im[non_text_region]
                        average_bg_color = np.mean(non_text_px, axis=0)
                        std_bgr = np.std(non_text_px - average_bg_color, axis=0)
                        std_max = np.max(std_bgr)
                        inpaint_thresh = 7 if np.std(std_bgr) > 1 else 10
                        if std_max < inpaint_thresh:
                            need_inpaint = False
                            im[np.where(ballon_msk > 0)] = average_bg_color
                        # cv2.imshow('im', im)
                        # cv2.imshow('ballon', ballon_msk)
                        # cv2.imshow('non_text', non_text_msk)
                        # cv2.waitKey(0)
                
                if need_inpaint:
                    inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = self._inpaint(im, msk)

                mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0
            return inpainted

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        raise NotImplementedError


@register_inpainter('opencv-tela')
class OpenCVInpainter(InpainterBase):

    def setup_inpainter(self):
        self.inpaint_method = lambda img, mask, *args, **kwargs: cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


@register_inpainter('patchmatch')
class PatchmatchInpainter(InpainterBase):

    def setup_inpainter(self):
        from . import patch_match
        self.inpaint_method = lambda img, mask, *args, **kwargs: patch_match.inpaint(img, mask, patch_size=3)
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


import torch
from utils.imgproc_utils import resize_keepasp
from .aot import AOTGenerator, load_aot_model
AOTMODEL: AOTGenerator = None
AOTMODEL_PATH = 'data/models/aot_inpainter.ckpt'


@register_inpainter('aot')
class AOTInpainter(InpainterBase):

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'select': 2048
        }, 
        'device': DEVICE_SELECTOR(),
        'description': 'manga-image-translator inpainter'
    }

    device = DEFAULT_DEVICE
    inpaint_size = 2048
    model: AOTGenerator = None

    def setup_inpainter(self):
        global AOTMODEL
        self.device = self.params['device']['select']
        if AOTMODEL is None:
            self.model = AOTMODEL = load_aot_model(AOTMODEL_PATH, self.device)
        else:
            self.model = AOTMODEL
            self.model.to(self.device)
        self.inpaint_size = int(self.params['inpaint_size']['select'])

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None

        img = resize_keepasp(img, new_shape, stride=None)
        mask = resize_keepasp(mask, new_shape, stride=None)

        im_h, im_w = img.shape[:2]
        pad_bottom = 128 - im_h if im_h < 128 else 0
        pad_right = 128 - im_w if im_w < 128 else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch)
        img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['select']
            self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['select'])


from .lama import LamaFourier, load_lama_mpe

LAMA_MPE: LamaFourier = None
@register_inpainter('lama_mpe')
class LamaInpainterMPE(InpainterBase):

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'select': 2048
        }, 
        'device': DEVICE_SELECTOR()
    }

    device = DEFAULT_DEVICE
    inpaint_size = 2048

    def setup_inpainter(self):
        global LAMA_MPE

        self.device = self.params['device']['select']
        if LAMA_MPE is None:
            self.model = LAMA_MPE = load_lama_mpe(r'data/models/lama_mpe.ckpt', self.device)
        else:
            self.model = LAMA_MPE
            self.model.to(self.device)
        self.inpaint_size = int(self.params['inpaint_size']['select'])

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None
        # high resolution input could produce cloudy artifacts
        img = resize_keepasp(img, new_shape, stride=64)
        mask = resize_keepasp(mask, new_shape, stride=64)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
        rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
        direct = torch.LongTensor(direct).unsqueeze_(0)

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            rel_pos = rel_pos.to(self.device)
            direct = direct.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        
        img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['select']
            self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['select'])


# LAMA_ORI: LamaFourier = None
# @register_inpainter('lama_ori')
# class LamaInpainterORI(InpainterBase):

#     params = {
#         'inpaint_size': {
#             'type': 'selector',
#             'options': [
#                 1024, 
#                 2048
#             ], 
#             'select': 2048
#         }, 
#         'device': {
#             'type': 'selector',
#             'options': [
#                 'cpu',
#                 'cuda'
#             ],
#             'select': DEFAULT_DEVICE
#         }
#     }

#     device = DEFAULT_DEVICE
#     inpaint_size = 2048

#     def setup_inpainter(self):
#         global LAMA_ORI

#         self.device = self.params['device']['select']
#         if LAMA_ORI is None:
#             self.model = LAMA_ORI = load_lama_mpe(r'data/models/lama_org.ckpt', self.device, False)
#         else:
#             self.model = LAMA_ORI
#             self.model.to(self.device)
#         self.inpaint_by_block = True if self.device == 'cuda' else False
#         self.inpaint_size = int(self.params['inpaint_size']['select'])

#     def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

#         img_original = np.copy(img)
#         mask_original = np.copy(mask)
#         mask_original[mask_original < 127] = 0
#         mask_original[mask_original >= 127] = 1
#         mask_original = mask_original[:, :, None]

#         new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None
#         # high resolution input could produce cloudy artifacts
#         img = resize_keepasp(img, new_shape, stride=64)
#         mask = resize_keepasp(mask, new_shape, stride=64)

#         im_h, im_w = img.shape[:2]
#         longer = max(im_h, im_w)
#         pad_bottom = longer - im_h if im_h < longer else 0
#         pad_right = longer - im_w if im_w < longer else 0
#         mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
#         img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

#         img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
#         mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
#         mask_torch[mask_torch < 0.5] = 0
#         mask_torch[mask_torch >= 0.5] = 1
#         rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
#         rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
#         direct = torch.LongTensor(direct).unsqueeze_(0)

#         if self.device == 'cuda':
#             img_torch = img_torch.cuda()
#             mask_torch = mask_torch.cuda()
#             rel_pos = rel_pos.cuda()
#             direct = direct.cuda()
#         img_torch *= (1 - mask_torch)
#         return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

#     @torch.no_grad()
#     def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

#         im_h, im_w = img.shape[:2]
#         img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
#         img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        
#         img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#         if pad_bottom > 0:
#             img_inpainted = img_inpainted[:-pad_bottom]
#         if pad_right > 0:
#             img_inpainted = img_inpainted[:, :-pad_right]
#         new_shape = img_inpainted.shape[:2]
#         if new_shape[0] != im_h or new_shape[1] != im_w :
#             img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
#         img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
#         return img_inpainted

#     def updateParam(self, param_key: str, param_content):
#         super().updateParam(param_key, param_content)

#         if param_key == 'device':
#             param_device = self.params['device']['select']
#             self.model.to(param_device)
#             self.device = param_device
#             if param_device == 'cuda':
#                 self.inpaint_by_block = False
#             else:
#                 self.inpaint_by_block = True

#         elif param_key == 'inpaint_size':
#             self.inpaint_size = int(self.params['inpaint_size']['select'])
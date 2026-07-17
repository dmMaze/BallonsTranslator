import sys
from typing import List

import cv2
import numpy as np

from ballontranslator.utils.imgproc_utils import resize_keepasp, smart_resize

from ..base import DEFAULT_DEVICE, DEVICE_SELECTOR, TORCH_DTYPE_MAP, BF16_SUPPORTED
from ..textdetector import TextBlock
from .base import InpainterBase, register_inpainter

TORCH_DEPENDENCIES = ['torch']

try:
    # OpenCV/PatchMatch inpainters can be imported without torch installed.
    import torch
except ModuleNotFoundError:
    torch = None


def torch_no_grad(func):
    if torch is None:
        return func
    return torch.no_grad()(func)


@register_inpainter('opencv-tela')
class OpenCVInpainter(InpainterBase):

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.inpaint_method = lambda img, mask, *args, **kwargs: cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


@register_inpainter('patchmatch')
class PatchmatchInpainter(InpainterBase):

    if sys.platform == 'darwin':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/macos_arm64_patchmatch_libs.7z',
                'sha256_pre_calculated': ['843704ab096d3afd8709abe2a2c525ce3a836bb0a629ed1ee9b8f5cee9938310', '849ca84759385d410c9587d69690e668822a3fc376ce2219e583e7e0be5b5e9a'],
                'files': ['macos_libopencv_world.4.8.0.dylib', 'macos_libpatchmatch_inpaint.dylib'],
                'save_dir': 'data/libs',
                'archived_files': 'macos_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': '9f332c888be0f160dbe9f6d6887eb698a302e62f4c102a0f24359c540d5858ea'
        }]
    elif sys.platform == 'win32':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/windows_patchmatch_libs.7z',
                'sha256_pre_calculated': ['3b7619caa29dc3352b939de4e9981217a9585a13a756e1101a50c90c100acd8d', '0ba60cfe664c97629daa7e4d05c0888ebfe3edcb3feaf1ed5a14544079c6d7af'],
                'files': ['opencv_world455.dll', 'patchmatch_inpaint.dll'],
                'save_dir': 'data/libs',
                'archived_files': 'windows_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': 'c991ff61f7cb3efaf8e75d957e62d56ba646083bc25535f913ac65775c16ca65'
        }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        from . import patch_match
        self.inpaint_method = lambda img, mask, *args, **kwargs: patch_match.inpaint(img, mask, patch_size=3)
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


@register_inpainter('aot')
class AOTInpainter(InpainterBase):

    dependencies = TORCH_DEPENDENCIES

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'value': 2048,
            'display_name': 'Inpaint Size'
        }, 
        'device': DEVICE_SELECTOR(),
        'description': 'manga-image-translator inpainter'
    }

    device = DEFAULT_DEVICE
    inpaint_size = 2048
    model = None
    _load_model_keys = {'model'}

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/aot_inpainter.ckpt',
            'sha256_pre_calculated': '878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f',
            'files': 'data/models/aot_inpainter.ckpt',
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.model = None
        
    def _load_model(self):
        from .aot import load_aot_model
        AOTMODEL_PATH = 'data/models/aot_inpainter.ckpt'
        self.model = load_aot_model(AOTMODEL_PATH, self.device)

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device

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

    @torch_no_grad
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch)
        img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5)
        img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
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
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])


@register_inpainter('lama_mpe')
class LamaInpainterMPE(InpainterBase):

    dependencies = TORCH_DEPENDENCIES

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'value': 2048,
            'display_name': 'Inpaint Size'
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone'])
    }

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/lama_mpe.ckpt',
            'sha256_pre_calculated': 'd625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc',
            'files': 'data/models/lama_mpe.ckpt',
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.precision = 'fp32'
        self.model = None

    def _load_model(self):
        from .lama import load_lama_mpe
        self.model = load_lama_mpe(r'data/models/lama_mpe.ckpt', self.device)

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

    @torch_no_grad
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        
        precision = TORCH_DTYPE_MAP[self.precision]
        if self.device in {'cuda'}:
            try:
                with torch.autocast(device_type=self.device, dtype=precision):
                    img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
            except Exception as e:
                self.logger.error(e)
                self.logger.error(f'{precision} inference is not supported for this device, use fp32 instead.')
                img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        else:
            img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

        img_inpainted = (img_inpainted_torch.to(device='cpu', dtype=torch.float32).squeeze_(0).permute(1, 2, 0).numpy() * 255)
        img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
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
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])

        elif param_key == 'precision':
            precision = self.params['precision']['value']
            self.precision = precision

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device
        if precision is not None:
            self.precision = precision

@register_inpainter('lama_large_512px')
class LamaLarge(LamaInpainterMPE):

    dependencies = TORCH_DEPENDENCIES

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                512,
                768,
                1024,
                1536, 
                2048
            ], 
            'value': 1536,
            'display_name': 'Inpaint Size'
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone']),
        'precision': {
            'type': 'selector',
            'options': [
                'fp32',
                'bf16'
            ], 
            'value': 'bf16' if BF16_SUPPORTED == 'cuda' else 'fp32'
        }, 
    }

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt',
            'sha256_pre_calculated': '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935',
            'files': 'data/models/lama_large_512px.ckpt',
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.precision = self.params['precision']['value']

    def _load_model(self):
        from .lama import load_lama_mpe
        device = self.params['device']['value']
        precision = self.params['precision']['value']

        self.model = load_lama_mpe(r'data/models/lama_large_512px.ckpt', device='cpu', use_mpe=False, large_arch=True)
        self.moveToDevice(device, precision=precision)


FLUX_MODEL_MAPPER = {
    '4b-Q4_K_M': 'black-forest-labs/FLUX.2-klein-4B'
}

@register_inpainter('flux2-klein')
class Flux2Klein(InpainterBase):

    dependencies = [
        'torch',
        'diffusers>=0.37.1',
        'safetensors',
        'transformers==4.57.6',
        'gguf>=0.10.0',
        'accelerate>=0.26.0',
        'hf_transfer'
    ]

    params = {
        'model': {
            'type': 'selector',
            'options': [
                '4b-Q4_K_M', 
            ], 
            'value': '4b-Q4_K_M'
        },
        'max_resolution': {
            'type': 'selector',
            'options': [
                512,
                768,
                1024,
                1280,
                1536,
                2048
            ], 
            'value': 1024,
            'display_name': 'Max Resolution'
        }, 
        'device': DEVICE_SELECTOR(),
        'step': 8
    }
    inpaint_by_block = False

    download_file_list = [
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/model_index.json',
                'files': 'data/models/flux-2-klein-4b/model_index.json',
            },
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/scheduler/scheduler_config.json',
                'files': 'data/models/flux-2-klein-4b/scheduler/scheduler_config.json'
            },
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/transformer/config.json',
                'files': 'data/models/flux-2-klein-4b/transformer/config.json',
            },
            {
                'url': 'https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_K_M.gguf',
                'files': 'data/models/flux-2-klein-4b-Q4_K_M.gguf',
                'sha256_pre_calculated': '0b25d143c8469b342bc5af3bce92b783bf6b0636d285f7b2f75e38af63af9a15'
            },
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/config.json',
                'files': 'data/models/flux-2-vae/config.json',
            },
            {
                'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors',
                'files': 'data/models/flux-2-vae/diffusion_pytorch_model.safetensors',
                'sha256_pre_calculated': 'ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04'
            },
            {
                'url': 'https://huggingface.co/dreMaz/flux2-klein-inpaint/resolve/main/flux2_inpaint_prompt.safetensors',
                'files': 'data/models/flux2_inpaint_prompt.safetensors',
                'sha256_pre_calculated': '7d7b19ec266581cb1faa51ad92f49a302932b0c589feae633f97da2d925cb6a4'
            }
        ]

    _load_model_keys = {'pipeline'}

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _load_model(self):
        
        from .flux_inpaint_pipeline import Flux2KleinInpaintPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2
        from safetensors.torch import load_file
        from diffusers import GGUFQuantizationConfig

        model_type = self.get_param_value('model')
        source = FLUX_MODEL_MAPPER[model_type]

        transformer = Flux2Transformer2DModel.from_single_file(
            "data/models/flux-2-klein-4b-Q4_K_M.gguf",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config='data/models/flux-2-klein-4b/transformer/config.json',
            
        )
        self.prompt_embeds = load_file('data/models/flux2_inpaint_prompt.safetensors')['prompt_embeds'].to(dtype=torch.bfloat16, device=self.get_param_value('device'))

        vae = AutoencoderKLFlux2.from_pretrained(f'data/models/flux-2-vae').to(device=self.get_param_value('device'), dtype=torch.bfloat16)
        pipeline = Flux2KleinInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path='data/models/flux-2-klein-4b',
            text_encoder=None,
            tokenizer=None,
            vae=vae,
            transformer=transformer,
            local_files_only=True
        )
        self.pipeline = pipeline.to(device=self.get_param_value('device'), )

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        max_resolution = self.get_param_value('max_resolution')
        div = 16
        mask_original = (mask > 127)[..., None].astype(np.uint8)
        img_original = img.copy()

        input_sz = img.shape[:2]

        h, w = input_sz
        th, tw = h, w
        resize_ratio = max_resolution / max(th, tw)
        if resize_ratio < 1:
            th, tw = int(round(resize_ratio * th)), int(round(resize_ratio * tw))
        th = int(round(th / div)) * div
        tw = int(round(tw / div)) * div
        img = smart_resize(img, (th, tw))
        mask = smart_resize(mask, (th, tw))

        rst = self.pipeline(
            image=img,
            mask=mask,
            prompt_embeds=self.prompt_embeds,
            height=img.shape[0],
            width=img.shape[1],
            num_inference_steps=self.get_param_value('step'),
            guidance_scale=1, return_dict=False, output_type='numpy'
        )
        img_inpainted = (np.round(rst[0] * 255)).astype(np.uint8)
        img_inpainted = smart_resize(img_inpainted, img_original.shape[:2])
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted
    
    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if hasattr(self, 'pipeline'):
            if param_key == 'device':
                param_device = self.get_param_value('device')
                self.pipeline.to(device=param_device)

import gc
import os
import time
from typing import Dict, List, Callable, Union
from copy import deepcopy
from collections import OrderedDict
import re
import importlib

from utils.logger import logger as LOGGER
from utils import shared


GPUINTENSIVE_SET = {'cuda', 'mps', 'xpu', 'privateuseone'}

def register_hooks(hooks_registered: OrderedDict, callbacks: Union[List, Callable, Dict]):
    if callbacks is None:
        return
    if isinstance(callbacks, (Dict, OrderedDict)):
        for k, v in callbacks.items():
            hooks_registered[k] = v
    else:
        nhooks = len(hooks_registered)

        if isinstance(callbacks, Callable):
            callbacks = [callbacks]
        for callback in callbacks:
            hk = 'hook_' + str(nhooks).zfill(2)
            while True:
                if hk not in hooks_registered:
                    break
                hk = hk + '_' + str(time.time_ns())
            hooks_registered[hk] = callback
            nhooks += 1

class BaseModule:

    params: Dict = None
    logger = LOGGER

    _preprocess_hooks: OrderedDict = None
    _postprocess_hooks: OrderedDict = None

    download_file_list: List = None
    download_file_on_load = False

    _load_model_keys: set = None

    def __init__(self, **params) -> None:
        if params:
            if self.params is None:
                self.params = params
            else:
                self.params.update(params)

    @classmethod
    def register_postprocess_hooks(cls, callbacks: Union[List, Callable]):
        """
        these hooks would be shared among all objects inherited from the same super class
        """
        assert cls._postprocess_hooks is not None
        register_hooks(cls._postprocess_hooks, callbacks)

    @classmethod
    def register_preprocess_hooks(cls, callbacks: Union[List, Callable, Dict]):
        """
        these hooks would be shared among all objects inherited from the same super class
        """
        assert cls._preprocess_hooks is not None
        register_hooks(cls._preprocess_hooks, callbacks)

    def get_param_value(self, param_key: str):
        assert self.params is not None and param_key in self.params
        p = self.params[param_key]
        if isinstance(p, dict):
            return p['value']
        return p
    
    def set_param_value(self, param_key: str, param_value, convert_dtype=True):
        assert self.params is not None and param_key in self.params
        p = self.params[param_key]
        if isinstance(p, dict):
            if convert_dtype:
                try:
                    param_value = type(p['value'])(param_value)
                except ValueError:
                    dtype = type(p['value'])
                    self.logger.warning(f'Invalid param value {param_value} for defined dtype: {dtype}')
            p['value'] = param_value
        else:
            if convert_dtype:
                try:
                    param_value = type(p)(param_value)
                except ValueError:
                    self.logger.warning(f'Invalid param value {param_value} for defined dtype: {type(p)}, revert to original value {p}')
                    param_value = p
            self.params[param_key] = param_value

    def updateParam(self, param_key: str, param_content):
        self.set_param_value(param_key, param_content)

    @property
    def low_vram_mode(self):
        if 'low vram mode' in self.params:
            return self.get_param_value('low vram mode')
        return False

    def is_cpu_intensive(self)->bool:
        if self.params is not None and 'device' in self.params:
            return self.params['device']['value'] == 'cpu'
        return False

    def is_gpu_intensive(self) -> bool:
        if self.params is not None and 'device' in self.params:
            return self.params['device']['value'] in GPUINTENSIVE_SET
        return False

    def is_computational_intensive(self) -> bool:
        if self.params is not None and 'device' in self.params:
            return True
        return False
    
    def unload_model(self, empty_cache=False):
        model_deleted = False
        if self._load_model_keys is not None:
            for k in self._load_model_keys:
                if hasattr(self, k):
                    model = getattr(self, k)
                    if model is not None:
                        if hasattr(model, 'unload_model'):
                            model.unload_model(empty_cache=False)
                        del model
                        setattr(self, k, None)
                        model_deleted = True
    
        if empty_cache and model_deleted:
            soft_empty_cache()

        return model_deleted

    def load_model(self):
        # TODO: check and download files
        self._load_model()
        return

    def _load_model(self):
        return

    def all_model_loaded(self):
        if self._load_model_keys is None:
            return True
        for k in self._load_model_keys:
            if not hasattr(self, k) or getattr(self, k) is None:
                return False
        return True
    
    def __del__(self):
        self.unload_model()

    @property
    def debug_mode(self):
        return shared.DEBUG
    
    def flush(self, param_key: str):
        return None

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch

DEFAULT_DEVICE = 'cpu'
AVAILABLE_DEVICES = ['cpu']
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda'
    AVAILABLE_DEVICES.append(DEFAULT_DEVICE)
if hasattr(torch, 'xpu')  and torch.xpu.is_available():
    DEFAULT_DEVICE = 'xpu' if torch.xpu.is_available() else 'cpu'
    AVAILABLE_DEVICES.append(DEFAULT_DEVICE)
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEFAULT_DEVICE = 'mps'
    AVAILABLE_DEVICES.append(DEFAULT_DEVICE)

try: 
    import torch_directml
    if hasattr(torch, 'privateuseone') and torch_directml.device_count() > 0:
        torch.dml = torch_directml
        DEFAULT_DEVICE = f'privateuseone:{torch.dml.default_device()}'
        AVAILABLE_DEVICES += [f"privateuseone:{d}" for d in range(torch.dml.device_count())]
except:
    # directml is not supported
    pass
BF16_SUPPORTED = DEFAULT_DEVICE == 'cuda' and torch.cuda.is_bf16_supported() or DEFAULT_DEVICE == 'xpu' and torch.xpu.is_bf16_supported()

def is_nvidia():
    if DEFAULT_DEVICE == 'cuda':
        if torch.version.cuda:
            return True
    return False

def is_intel():
    if DEFAULT_DEVICE == 'xpu':
        if torch.version.xpu:
            return True
    return False

def soft_empty_cache():
    gc.collect()
    if DEFAULT_DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif DEFAULT_DEVICE == 'xpu':
       torch.xpu.empty_cache()
       # torch.xpu.ipc_collect()
    elif DEFAULT_DEVICE == 'mps':
        torch.mps.empty_cache()


def DEVICE_SELECTOR(not_supported=[]): return deepcopy(
    {
        'type': 'selector',
        'options': [opt for opt in AVAILABLE_DEVICES if all(device not in opt for device in not_supported)],
        'value': DEFAULT_DEVICE if not any(DEFAULT_DEVICE in device for device in not_supported) else 'cpu'
    }
)

TORCH_DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}
    
def load_modules():
    def _load_module(module_dir: str, module_pattern: str):
        modules = os.listdir(module_dir)
        pattern = re.compile(module_pattern)
        module_path = module_dir.replace('/', '.')
        if not module_path.endswith('.'):
            module_path += '.'
        for module_name in modules:
            if pattern.match(module_name) is not None:
                try:
                    module = module_path + module_name.replace('.py', '')
                    importlib.import_module(module)
                except Exception as e:
                    LOGGER.warning(f'Failed to import {module}: {e}')

    for kwargs in [
        {'module_dir': 'modules/translators', 'module_pattern': r'trans_(.*?).py'},
        {'module_dir': 'modules/textdetector', 'module_pattern': r'detector_(.*?).py'},
        {'module_dir': 'modules/inpaint', 'module_pattern': r'inpaint_(.*?).py'},
        {'module_dir': 'modules/ocr', 'module_pattern': r'ocr_(.*?).py'},
    ]:
        _load_module(**kwargs)

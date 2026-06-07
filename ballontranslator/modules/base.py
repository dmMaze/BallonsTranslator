import gc
import os
import time
from typing import Dict, List, Callable, Union
from copy import deepcopy
from collections import OrderedDict
import re
import importlib
import importlib.util
import traceback
from pathlib import Path

from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils import shared
from ballontranslator.utils.lock import aquire_model_loading_lock, release_model_loading_lock


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


def patch_module_params(cfg_param, module_params, module_name: str = ''):
    # cfg_param = config_params[module_key]
    cfg_key_set = set(cfg_param.keys())
    module_key_set = set(module_params.keys())
    for ck in cfg_key_set:
        if ck not in module_key_set:
            LOGGER.warning(f'Found invalid {module_name} config: {ck}')
            cfg_param.pop(ck)

    for mk in module_key_set:
        if mk not in cfg_key_set:
            if not mk.startswith('__') and mk != 'description':
                LOGGER.info(f'Found new {module_name} config: {mk}')
            cfg_param[mk] = module_params[mk]
        else:
            mparam = module_params[mk]
            cparam = cfg_param[mk]
            if isinstance(mparam, dict):
                tgt_type = mparam.get('data_type', type(mparam['value']))
                if isinstance(cparam, dict):
                    if 'value' in cparam:
                        v = cparam['value']
                    elif isinstance(mparam['value'], dict):
                        for k in mparam['value']:
                            if k in cparam:
                                mparam['value'][k] = cparam[k]
                        v = mparam['value']
                    else:
                        v = mparam['value']
                else:
                    v = cparam
                valid = True
                if tgt_type != type(v):
                    try:
                        v = tgt_type(v)
                    except:
                        valid = False
                        LOGGER.warning(f'Invalid param value {v} for defined dtype: {tgt_type}, it will be set to default value: {mparam}')
                if valid:
                    mparam['value'] = v
                cfg_param[mk] = mparam
            else:
                if type(cparam) != type(mparam):
                    if not isinstance(mparam, dict) and isinstance(cparam, dict):
                        cparam = cparam['value']
                    try:
                        cfg_param[mk] = type(mparam)(cparam)
                    except ValueError:
                        LOGGER.warning(f'Invalid param value {cparam} for defined dtype: {type(mparam)}, it will be set to default value: {mparam}')
                        cfg_param[mk] = mparam
    
    cfg_key_list = list(cfg_param.keys())
    module_key_list = list(module_params.keys())
    if cfg_key_list != module_key_list:
        new_params = {key: cfg_param[key] for key in module_key_list}
        cfg_param.clear()
        cfg_param.update(new_params)
        module_key_set = set(module_params.keys())
    cfg_param['__param_patched'] = True
    return cfg_param


def refresh_runtime_device_defaults(module_params: Dict):
    if module_params is None:
        return
    device_param = module_params.get('device')
    if not isinstance(device_param, dict) or device_param.get('type') != 'selector':
        return
    not_supported = device_param.get('__device_not_supported', [])
    # Fresh config generation is allowed to probe torch so CUDA can be the initial default.
    selector = DEVICE_SELECTOR(not_supported=not_supported)
    device_param['options'] = selector['options']
    device_param['value'] = selector['value']


def merge_config_module_params(
    config_params: Dict,
    module_keys: List,
    get_module: Callable,
    resolve_runtime_device_defaults: bool = False,
) -> Dict:
    for module_key in module_keys:
        module = get_module(module_key)
        # module may be a ModuleSpec, so config panels can load without imports.
        if hasattr(module, 'params_copy'):
            module_params = module.params_copy()
        else:
            module_params = deepcopy(getattr(module, 'params', None))
        if resolve_runtime_device_defaults:
            refresh_runtime_device_defaults(module_params)
        if module_key not in config_params or config_params[module_key] is None:
            config_params[module_key] = module_params
        elif module_params is None:
            continue
        else:
            patch_module_params(config_params[module_key], module_params, module_key)
    return config_params


def _try_unload_nested_model(model):
    if not hasattr(model, 'unload_model'):
        return
    try:
        model.unload_model(empty_cache=False)
    except TypeError:
        # Third-party runtimes such as CTranslate2 expose unload_model(), but
        # do not accept this project's empty_cache keyword.
        model.unload_model()


def standardize_module_params(params):
    if params is None:
        return
    for k, v in params.items():
        if not isinstance(v, dict) and k not in {'description'}:  # remember to exclude special keys here
            v = {'value': v}
        if isinstance(v, dict) and 'data_type' not in v:
            v['data_type'] = type(v['value'])
        params[k] = v


class BaseModule:

    params: Dict = None
    logger = LOGGER

    _preprocess_hooks: OrderedDict = None
    _postprocess_hooks: OrderedDict = None

    download_file_list: List = None
    download_file_on_load = False

    _load_model_keys: set = None

    def __init__(self, **params) -> None:
        standardize_module_params(self.params)
        if self.params is not None and '__param_patched' not in params:
            params = patch_module_params(params, self.params, self)
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
                    val_type = p.get('data_type', type(p['value']))
                    param_value = val_type(param_value)
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
                        _try_unload_nested_model(model)
                        setattr(self, k, None)
                        del model
                        model_deleted = True
    
        if empty_cache and model_deleted:
            soft_empty_cache()

        return model_deleted

    def load_model(self):
        # TODO: check and download files & inform UIs
        aquire_model_loading_lock()
        try:
            self._load_model()
        finally:
            release_model_loading_lock()
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

DEFAULT_DEVICE = 'cpu'
AVAILABLE_DEVICES = ['cpu']
BF16_SUPPORTED = False
# Torch stays optional for UI startup; selected modules call require_torch().
_TORCH = None
_TORCH_CHECKED = False
_TORCH_DEVICE_INFO_CHECKED = False


def torch_available() -> bool:
    try:
        return importlib.util.find_spec('torch') is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def zluda_available(device_name):
    return "[ZLUDA]" in device_name


def enable_zluda_config():

    import torch

    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print('Device name: ', device_name)
        print('Cuda is available: ', torch.cuda.is_available())
        print('Cuda version: ', torch.version.cuda)
        print('ZLUDA is available: ', zluda_available(device_name))

        if zluda_available(device_name):
            torch.backends.cudnn.enabled = False
            cuda_attr = torch.backends.cuda
            if hasattr(cuda_attr, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
                print('Cuda enable flash sdp: ', False)
            if hasattr(cuda_attr, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
                print('Cuda enable math sdp: ', True)
            if hasattr(cuda_attr, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                print('Cuda enable mem efficient sdp: ', False)
            if hasattr(cuda_attr, 'enable_cudnn_sdp'):
                torch.backends.cuda.enable_cudnn_sdp(False)
                print('Cuda enable cudnn sdp: ', False)


def require_torch():
    global _TORCH, _TORCH_CHECKED
    if _TORCH is not None:
        return _TORCH
    try:
        import torch
        enable_zluda_config()
    except ModuleNotFoundError as e:
        _TORCH_CHECKED = True
        raise ModuleNotFoundError(
            'PyTorch is required by the selected module but is not installed. '
            'Install torch/torchvision or select a module that does not require PyTorch.'
        ) from e
    _TORCH = torch
    _TORCH_CHECKED = True
    return _TORCH


def refresh_torch_device_info(raise_missing: bool = False):
    global DEFAULT_DEVICE, AVAILABLE_DEVICES, BF16_SUPPORTED, _TORCH_DEVICE_INFO_CHECKED
    if _TORCH_DEVICE_INFO_CHECKED:
        return
    # Device probing must not import torch during startup unless requested.
    if not raise_missing and not torch_available():
        _TORCH_DEVICE_INFO_CHECKED = True
        return

    try:
        torch = require_torch()
    except ModuleNotFoundError:
        if raise_missing:
            raise
        _TORCH_DEVICE_INFO_CHECKED = True
        return

    DEFAULT_DEVICE = 'cpu'
    AVAILABLE_DEVICES = ['cpu']
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        DEFAULT_DEVICE = 'cuda'
        AVAILABLE_DEVICES.append(DEFAULT_DEVICE)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEFAULT_DEVICE = 'xpu'
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
    except Exception:
        pass

    BF16_SUPPORTED = False
    if DEFAULT_DEVICE == 'cuda' and torch.cuda.is_bf16_supported() or DEFAULT_DEVICE == 'xpu' and torch.xpu.is_bf16_supported():
        BF16_SUPPORTED = True
    if DEFAULT_DEVICE == 'mps':
        BF16_SUPPORTED = True
    _TORCH_DEVICE_INFO_CHECKED = True


def soft_empty_cache():
    gc.collect()

    if _TORCH is not None:
        torch = _TORCH
        if DEFAULT_DEVICE == 'cuda':
            try:
                torch.cuda.synchronize()
            except Exception:
                LOGGER.debug('Failed to clear CUDA library workspaces.')
                LOGGER.debug(traceback.format_exc())
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif DEFAULT_DEVICE == 'xpu':
            torch.xpu.empty_cache()
        elif DEFAULT_DEVICE == 'mps':
            torch.mps.empty_cache()
    try:
        if os.name == 'posix':
            import ctypes
            ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass



def DEVICE_SELECTOR(not_supported:list[str]=[]):
    refresh_torch_device_info()
    return deepcopy(
        {
            'type': 'selector',
            'options': [opt for opt in AVAILABLE_DEVICES if all(device not in opt for device in not_supported)],
            'value': DEFAULT_DEVICE if not any(DEFAULT_DEVICE in device for device in not_supported) else 'cpu'
        }
    )


class _TorchDTypeMap:
    _names = {
        'fp32': 'float32',
        'fp16': 'float16',
        'bf16': 'bfloat16',
    }

    def __getitem__(self, key):
        torch = require_torch()
        return getattr(torch, self._names[key])


TORCH_DTYPE_MAP = _TorchDTypeMap()

MODULE_ROOT = Path(__file__).resolve().parent
MODULE_SCRIPTS = {
    'translator': {
        'module_dir': str(MODULE_ROOT / 'translators'),
        'module_package': 'ballontranslator.modules.translators',
        'module_pattern': r'trans_(.*?).py',
    },
    'textdetector': {
        'module_dir': str(MODULE_ROOT / 'textdetector'),
        'module_package': 'ballontranslator.modules.textdetector',
        'module_pattern': r'detector_(.*?).py',
    },
    'inpainter': {
        'module_dir': str(MODULE_ROOT / 'inpaint'),
        'module_package': 'ballontranslator.modules.inpaint',
        'module_pattern': r'inpaint_(.*?).py',
    },
    'ocr': {
        'module_dir': str(MODULE_ROOT / 'ocr'),
        'module_package': 'ballontranslator.modules.ocr',
        'module_pattern': r'ocr_(.*?).py',
    },
}
    
def import_module_registries(target_modules=None):
    # Eager import path kept for explicit compatibility/debug use only.
    def _load_module(module_dir: str, module_package: str, module_pattern: str):
        modules = os.listdir(module_dir)
        pattern = re.compile(module_pattern)
        for module_name in modules:
            if pattern.match(module_name) is not None:
                try:
                    module = module_package + '.' + module_name.replace('.py', '')
                    importlib.import_module(module)
                except Exception as e:
                    LOGGER.warning(f'Failed to import {module}: {e}')

    if target_modules is None:
        target_modules = MODULE_SCRIPTS
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    for k in target_modules:
        _load_module(**MODULE_SCRIPTS[k])


def init_module_registries(target_modules=None):
    # Startup registers lightweight specs; real module imports happen on selection.
    from .lazy_registry import init_lazy_module_registries
    init_lazy_module_registries(target_modules)

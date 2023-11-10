import gc
import os
import time
from typing import Dict, List, Callable, Union
from copy import deepcopy
from collections import OrderedDict

from utils.logger import logger as LOGGER

GPUINTENSIVE_SET = {'cuda', 'mps'}

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

    def updateParam(self, param_key: str, param_content):
        self_param_content = self.params[param_key]
        if isinstance(self_param_content, (str, float, int)):
            self.params[param_key] = param_content
        else:
            param_dict = self.params[param_key]
            if param_dict['type'] == 'selector':
                param_dict['select'] = param_content
            elif param_dict['type'] == 'editor':
                param_dict['content'] = param_content

    def is_cpu_intensive(self)->bool:
        if self.params is not None and 'device' in self.params:
            return self.params['device']['select'] == 'cpu'
        return False

    def is_gpu_intensive(self) -> bool:
        if self.params is not None and 'device' in self.params:
            return self.params['device']['select'] in GPUINTENSIVE_SET
        return False

    def is_computational_intensive(self) -> bool:
        if self.params is not None and 'device' in self.params:
            return True
        return False

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch

DEFAULT_DEVICE = 'cpu'
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEFAULT_DEVICE = 'mps'

def gc_collect():
    gc.collect()
    if DEFAULT_DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif DEFAULT_DEVICE == 'mps':
        torch.mps.empty_cache()

DEVICE_SELECTOR = lambda : deepcopy(
    {
        'type': 'selector',
        'options': [
            'cpu',
            'cuda',
            'mps'
        ],
        'select': DEFAULT_DEVICE
    }
)

TORCH_DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

import gc
import os
from typing import Dict
from copy import deepcopy

from utils.logger import logger as LOGGER

GPUINTENSIVE_SET = {'cuda', 'mps'}

class BaseModule:

    params: Dict = None
    logger = LOGGER

    def __init__(self, **params) -> None:
        if params:
            if self.params is None:
                self.params = params
            else:
                self.params.update(params)

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


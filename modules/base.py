from typing import Dict
from utils.logger import logger as LOGGER

GPUINTENSIVE_SET = {'cuda', 'hip'}

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


import torch

if hasattr(torch, 'cuda'):
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    DEFAULT_DEVICE = 'cpu'


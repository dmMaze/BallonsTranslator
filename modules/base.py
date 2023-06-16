from typing import Dict

GPUINTENSIVE_SET = {'cuda', 'hip'}

class BaseModule:

    params: Dict = None

    def __init__(self, **params) -> None:
        if params:
            self.params = params

    def updateParam(self, param_key: str, param_content):
        if isinstance(self.params[param_key], str):
            self.params[param_key] = param_content
        else:
            param_dict = self.params[param_key]
            if param_dict['type'] == 'selector':
                param_dict['select'] = param_content

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


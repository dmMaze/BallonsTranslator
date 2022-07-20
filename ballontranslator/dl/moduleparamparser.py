from typing import Dict

class ModuleParamParser:

    setup_params: Dict = None

    def __init__(self, **setup_params) -> None:
        if setup_params:
            self.setup_params = setup_params

    def updateParam(self, param_key: str, param_content):
        if isinstance(self.setup_params[param_key], str):
            self.setup_params[param_key] = param_content
        else:
            param_dict = self.setup_params[param_key]
            if param_dict['type'] == 'selector':
                param_dict['select'] = param_content


import torch

if hasattr(torch, 'cuda'):
    DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    DEFAULT_DEVICE = 'cpu'


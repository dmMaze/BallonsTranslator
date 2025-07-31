from typing import Tuple, List, ClassVar, Union, Any, Dict, Set
from dataclasses import dataclass, field, is_dataclass
import copy
import os

import numpy as np


# decorator to wrap original __init__
# https://www.geeksforgeeks.org/creating-nested-dataclass-objects-in-python/
def nested_dataclass(*args, **dataclass_kwargs):
    '''
    nested dataclass support \n
    also ignore extra arguments 
    '''
    def wrapper(check_class):
          
        # passing class to investigate
        check_class = dataclass(check_class, **dataclass_kwargs)
        o_init = check_class.__init__
          
        def __init__(self, *args, **kwargs):
              
            store_deprecated = 'deprecated_attributes' in self.__annotations__
            deprecated = {}
            for name in list(kwargs.keys()):
                if name not in self.__annotations__:
                    # print(f'warning: type object \'{self.__class__.__name__}\' has no attribute {name}, might be loading from an older config')
                    val = kwargs.pop(name)
                    if store_deprecated:
                        deprecated[name] = val
                    continue
                value = kwargs[name]
                # getting field type
                ft = check_class.__annotations__.get(name, None)
                  
                if is_dataclass(ft) and isinstance(value, dict):
                    obj = ft(**value)
                    kwargs[name]= obj

            if len(deprecated) > 0:
                kwargs['deprecated_attributes'] = deprecated        
            
            o_init(self, *args, **kwargs)
        check_class.__init__=__init__
          
        return check_class
      
    return wrapper(args[0]) if args else wrapper


@dataclass
class Config:
    
    def update(self, key: str, value):
        assert key in self.__annotations__, f'type object \'{self.__class__.__name__}\' has no attribute {key}'
        self.__setattr__(key, value)

    @classmethod
    def annotations_set(cls):
        return set(list(cls.__annotations__))
    
    def __getitem__(self, key: str):
        assert key in self.__annotations__, f'type object \'{self.__class__.__name__}\' has no attribute {key}'
        return self.__getattribute__(key)
    
    def __setitem__(self, key: str, value):
        self.__setattr__(key, value)

    @classmethod
    def params(cls):
        return cls.__annotations__
    
    def merge(self, target):
        tgt_keys = target.annotations_set()
        for key in tgt_keys:
            if isinstance(self[key], Config):
                self[key].merge(target[key])
            else:
                self.update(key, target[key])

    def copy(self):
        return copy.deepcopy(self)
    

MODULE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BASE_PATH = os.path.dirname(MODULE_PATH)
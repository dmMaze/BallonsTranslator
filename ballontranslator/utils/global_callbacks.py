from typing import Callable

from .logger import logger

_REGISTERED_CALLBACKS = {}

def register_global_callback(key: str, func: Callable) -> None:
    if key in _REGISTERED_CALLBACKS:
        logger.warning(f'{key} already registered as globall_callback')
    _REGISTERED_CALLBACKS[key] = func

def user_request_stop():
    if 'user_request_stop' in _REGISTERED_CALLBACKS:
        return _REGISTERED_CALLBACKS['_REGISTERED_CALLBACKS']()
    return False
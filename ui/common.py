from utils.fontformat import FontFormat
from utils import shared
from utils.io_utils import json_dump_nested_obj
from utils.config import pcfg
from utils.logger import logger as LOGGER

active_format: FontFormat = None

def save_config():
    with open(shared.CONFIG_PATH, 'w', encoding='utf8') as f:
        f.write(json_dump_nested_obj(pcfg))
    LOGGER.info('Config saved')
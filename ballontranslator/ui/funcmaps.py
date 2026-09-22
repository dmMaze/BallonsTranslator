from typing import Callable

import numpy as np

from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock_mask import canny_flood, connected_canny_flood, existing_mask


def get_maskseg_method() -> Callable[..., tuple[np.ndarray, np.ndarray, dict]]:
    return [canny_flood, connected_canny_flood, existing_mask][pcfg.drawpanel.rectool_method]

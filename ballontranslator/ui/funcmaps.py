from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock_mask import canny_flood, connected_canny_flood, existing_mask, region_mask


def get_maskseg_method():
    if pcfg.module.inpainter == 'LLMInpaint':
        return region_mask
    return [canny_flood, connected_canny_flood, existing_mask][pcfg.drawpanel.rectool_method]

from utils.io_utils import build_funcmap
from utils.fontformat import FontFormat
from utils.config import pcfg
from utils.textblock_mask import canny_flood, connected_canny_flood, existing_mask

handle_ffmt_change = build_funcmap('ui.fontformat_commands', 
                                     list(FontFormat.params().keys()), 
                                     'ffmt_change_', verbose=False)


def get_maskseg_method():
    return [canny_flood, connected_canny_flood, existing_mask][pcfg.drawpanel.rectool_method]
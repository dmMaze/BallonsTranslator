from ballontranslator.utils.io_utils import build_funcmap
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock_mask import canny_flood, connected_canny_flood, existing_mask

# Build base function map
handle_ffmt_change = build_funcmap('ballontranslator.ui.fontformat_commands', 
                                     list(FontFormat.params().keys()) + ['rel_font_size', 'angle'], 
                                     'ffmt_change_', verbose=False)


def get_maskseg_method():
    return [canny_flood, connected_canny_flood, existing_mask][pcfg.drawpanel.rectool_method]

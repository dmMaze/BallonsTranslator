from utils.io_utils import build_funcmap
from utils.fontformat import FontFormat


handle_ffmt_change = build_funcmap('ui.fontformat_commands', 
                                     list(FontFormat.params().keys()), 
                                     'ffmt_change_', verbose=False)
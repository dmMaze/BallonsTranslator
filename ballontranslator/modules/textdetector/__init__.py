from .base import *

# ysg_ctd wraps ysgyolo: it keeps its own config section but is documented to
# reuse the base detector's text detection settings. Newly created sections are
# seeded from the saved ysgyolo config instead of leaking template defaults.
TEXTDETECTOR_PARAM_INHERITANCE = {'ysg_ctd': 'ysgyolo'}

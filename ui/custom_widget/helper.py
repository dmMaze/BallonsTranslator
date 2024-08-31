from qtpy.QtGui import QColor

from utils.config import pcfg


def isDarkTheme():
    return pcfg.darkmode

def themeColor():
    return QColor(30, 147, 229, 127)
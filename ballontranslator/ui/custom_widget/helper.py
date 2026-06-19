from qtpy.QtGui import QColor

from ballontranslator.utils.config import pcfg
from ballontranslator.utils import shared


def isDarkTheme():
    return pcfg.darkmode

def themeColor():
    return QColor(73, 136, 190)

def borderColor():
    return QColor(*shared.BORDER_COLOR)

def widgetBackgroundColor():
    return QColor(*shared.WIDGET_BACKGROUND_COLOR)

from qtpy.QtGui import QColor


# Shared modality palette. Active source SVG icons use the same fills.
LLM_MODALITY_TEXT = 'text'
LLM_MODALITY_VISION = 'vision'
LLM_MODALITY_IMAGE = 'image'

LLM_MODALITY_TEXT_COLOR = '#FE80A0'
LLM_MODALITY_VISION_COLOR = '#1E93E5'
LLM_MODALITY_IMAGE_COLOR = '#9858A2'
LLM_MODALITY_BADGE_ALPHA = 46


def modality_badge_qcolor(color: str) -> QColor:
    """Return the translucent background color used by active modality badges.

    Example:
        >>> modality_badge_qcolor('#1E93E5').alpha()
        46
    """
    qcolor = QColor(color)
    qcolor.setAlpha(LLM_MODALITY_BADGE_ALPHA)
    return qcolor

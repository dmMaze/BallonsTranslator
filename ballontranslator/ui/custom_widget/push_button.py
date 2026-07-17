from qtpy.QtCore import Qt
from qtpy.QtWidgets import QToolButton, QPushButton, QSizePolicy


class NoBorderPushBtn(QPushButton):
    pass


class ExpandingToolButton(QToolButton):
    """A left-aligned tool button that fills the available horizontal space.

    >>> ExpandingToolButton.__name__
    'ExpandingToolButton'
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

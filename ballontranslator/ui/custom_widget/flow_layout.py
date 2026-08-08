from qtpy.QtWidgets import QLayout, QWidgetItem, QLayoutItem, QWidget
from qtpy.QtCore import QParallelAnimationGroup, Qt, QPropertyAnimation, QEasingCurve, QSize, QRect, QPoint
from typing import List

class WidgetItem(QWidgetItem):

    def sizeHint(self) -> QSize:
        return self.widget().sizeHint()


class FlowLayout(QLayout):
    """ Flow layout """

    def __init__(self, parent=None, needAni=False, isTight=False):
        """
        Parameters
        ----------
        parent:
            parent window or layout

        needAni: bool
            whether to add moving animation

        isTight: bool
            whether to use the tight layout when widgets are hidden
        """
        super().__init__(parent)
        self._items = []    # type: List[QLayoutItem]
        self._anis = []
        self._aniGroup = QParallelAnimationGroup(self)
        self._verticalSpacing = 10
        self._horizontalSpacing = 10
        self.duration = 300
        self.ease = QEasingCurve.Linear
        self.needAni = needAni
        self.isTight = isTight

        self.height = 0

    def insertWidget(self, idx: int, w: QWidget):
        self.addChildWidget(w)
        self.insertItem(idx, WidgetItem(w))

    def insertItem(self, idx:int, item):
        self._items.insert(idx, item)

    def moveItem(self, source_index: int, target_index: int) -> bool:
        """Move an existing layout item without replacing its Qt wrapper."""
        if not 0 <= source_index < len(self._items):
            return False

        target_index = max(0, min(target_index, len(self._items) - 1))
        if source_index == target_index:
            return False

        item = self._items.pop(source_index)
        self._items.insert(target_index, item)
        if self.needAni:
            animation = self._anis.pop(source_index)
            self._anis.insert(target_index, animation)
        self.invalidate()
        return True

    def addItem(self, item):
        self._items.append(item)

    def addWidget(self, w):
        super().addWidget(w)
        if not self.needAni:
            return

        ani = QPropertyAnimation(w, b'geometry')
        ani.setEndValue(QRect(QPoint(0, 0), w.size()))
        ani.setDuration(self.duration)
        ani.setEasingCurve(self.ease)
        w.setProperty('flowAni', ani)
        self._anis.append(ani)
        self._aniGroup.addAnimation(ani)

    def setAnimation(self, duration, ease=QEasingCurve.Linear):
        """ set the moving animation

        Parameters
        ----------
        duration: int
            the duration of animation in milliseconds

        ease: QEasingCurve
            the easing curve of animation
        """
        if not self.needAni:
            return

        self.duration = duration
        self.ease = ease

        for ani in self._anis:
            ani.setDuration(duration)
            ani.setEasingCurve(ease)

    def count(self):
        return len(self._items)

    def itemAt(self, index: int):
        if 0 <= index < len(self._items):
            return self._items[index]

        return None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            item = self._items[index]   # type: QWidgetItem
            ani = item.widget().property('flowAni')
            if ani:
                self._anis.remove(ani)
                self._aniGroup.removeAnimation(ani)
                ani.deleteLater()

            return self._items.pop(index).widget()

        return None

    def removeWidget(self, widget):
        for i, item in enumerate(self._items):
            if item.widget() is widget:
                return self.takeAt(i)

    def removeAllWidgets(self):
        """ remove all widgets from layout """
        while self._items:
            self.takeAt(0)

    def takeAllWidgets(self):
        """ remove all widgets from layout and delete them """
        while self._items:
            w = self.takeAt(0)
            if w:
                w.deleteLater()

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width: int):
        """ get the minimal height according to width """
        return self._doLayout(QRect(0, 0, width, 0), False)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._doLayout(rect, True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()

        for item in self._items:
            size = size.expandedTo(item.minimumSize())

        m = self.contentsMargins()
        size += QSize(m.left()+m.right(), m.top()+m.bottom())

        return size

    def setVerticalSpacing(self, spacing: int):
        """ set vertical spacing between widgets """
        self._verticalSpacing = spacing

    def verticalSpacing(self):
        """ get vertical spacing between widgets """
        return self._verticalSpacing

    def setHorizontalSpacing(self, spacing: int):
        """ set horizontal spacing between widgets """
        self._horizontalSpacing = spacing

    def horizontalSpacing(self):
        """ get horizontal spacing between widgets """
        return self._horizontalSpacing

    def _doLayout(self, rect: QRect, move: bool):
        """ adjust widgets position according to the window size """
        aniRestart = False
        margin = self.contentsMargins()
        x = rect.x() + margin.left()
        y = rect.y() + margin.top()
        rowHeight = 0
        spaceX = self.horizontalSpacing()
        spaceY = self.verticalSpacing()

        for i, item in enumerate(self._items):
            if item.widget() and not item.widget().isVisible() and self.isTight:
                continue

            nextX = x + item.sizeHint().width() + spaceX

            if nextX - spaceX > rect.right() and rowHeight > 0:
                x = rect.x() + margin.left()
                y = y + rowHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                rowHeight = 0

            if move:
                target = QRect(QPoint(x, y), item.sizeHint())
                if not self.needAni:
                    item.setGeometry(target)
                elif target != self._anis[i].endValue():
                    self._anis[i].stop()
                    self._anis[i].setEndValue(target)
                    aniRestart = True

            x = nextX
            rowHeight = max(rowHeight, item.sizeHint().height())

        if self.needAni and aniRestart:
            self._aniGroup.stop()
            self._aniGroup.start()

        self.height = y + rowHeight + margin.bottom() - rect.y()
        return self.height


class JustifiedFlowLayout(FlowLayout):
    """Flow layout that distributes extra horizontal space within each row.

    Example:
        >>> JustifiedFlowLayout.__name__
        'JustifiedFlowLayout'
    """

    def _itemHiddenForTightLayout(self, item) -> bool:
        widget = item.widget()
        return bool(widget and widget.isHidden() and self.isTight)

    def _visibleItems(self):
        return [item for item in self._items if not self._itemHiddenForTightLayout(item)]

    def sizeHint(self):
        items = self._visibleItems()
        if not items:
            return self.minimumSize()

        rows = []
        for row_start in range(0, len(items), 2):
            rows.append(items[row_start:row_start + 2])

        spaceX = self.horizontalSpacing()
        spaceY = self.verticalSpacing()
        width = 0
        height = 0
        for row in rows:
            rowWidth = sum(item.sizeHint().width() for item in row)
            if len(row) > 1:
                rowWidth += spaceX * (len(row) - 1)
            rowHeight = max(item.sizeHint().height() for item in row)
            width = max(width, rowWidth)
            height += rowHeight

        if len(rows) > 1:
            height += spaceY * (len(rows) - 1)

        m = self.contentsMargins()
        return QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    def minimumSize(self):
        size = QSize()

        for item in self._visibleItems():
            size = size.expandedTo(item.minimumSize())

        m = self.contentsMargins()
        size += QSize(m.left()+m.right(), m.top()+m.bottom())

        return size

    def _doLayout(self, rect: QRect, move: bool):
        """Adjust widget positions after grouping visible items into rows."""
        margin = self.contentsMargins()
        left = rect.x() + margin.left()
        content_width = max(0, rect.width() - margin.left() - margin.right())
        y = rect.y() + margin.top()
        spaceX = self.horizontalSpacing()
        spaceY = self.verticalSpacing()
        rows = []
        row = []
        rowWidth = 0
        rowHeight = 0

        for item in self._items:
            if self._itemHiddenForTightLayout(item):
                continue

            itemSize = item.sizeHint()
            itemWidth = itemSize.width()
            nextWidth = itemWidth if not row else rowWidth + spaceX + itemWidth

            if row and nextWidth > content_width:
                rows.append((row, rowWidth, rowHeight))
                row = []
                rowWidth = 0
                rowHeight = 0

            wasEmpty = not row
            row.append((item, itemSize))
            rowWidth = itemWidth if wasEmpty else rowWidth + spaceX + itemWidth
            rowHeight = max(rowHeight, itemSize.height())

        if row:
            rows.append((row, rowWidth, rowHeight))

        aniRestart = False
        for rowIndex, (row, rowWidth, rowHeight) in enumerate(rows):
            x = left
            gap = spaceX
            if len(row) > 1 and content_width > rowWidth:
                gap += (content_width - rowWidth) / (len(row) - 1)

            for item, itemSize in row:
                if move:
                    target = QRect(QPoint(round(x), y), itemSize)
                    ani = item.widget().property('flowAni') if item.widget() is not None else None
                    if not self.needAni or ani is None:
                        item.setGeometry(target)
                    elif target != ani.endValue():
                        ani.stop()
                        ani.setEndValue(target)
                        aniRestart = True
                x += itemSize.width() + gap

            if rowIndex < len(rows) - 1:
                y = y + rowHeight + spaceY

        if self.needAni and aniRestart:
            self._aniGroup.stop()
            self._aniGroup.start()

        self.height = y + rowHeight + margin.bottom() - rect.y()
        return self.height

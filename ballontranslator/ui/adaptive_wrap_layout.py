"""Height-for-width wrapping layout for indivisible UI control units."""

from __future__ import annotations

from typing import Optional, Sequence

from qtpy.QtCore import QRect, QSize
from qtpy.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QStyle, QWidget


def _pack_preferred_widths(
    preferred_widths: Sequence[int],
    available_width: int,
    spacing: int,
) -> list[tuple[int, ...]]:
    """Return greedy rows of indexes without splitting an atomic item.

    An over-wide item occupies a row by itself; geometry assignment later gives
    it the available width instead of allowing horizontal overflow.

    >>> _pack_preferred_widths([40, 30, 50], 75, 5)
    [(0, 1), (2,)]
    >>> _pack_preferred_widths([100, 20], 60, 5)
    [(0,), (1,)]
    """
    available_width = max(0, int(available_width))
    spacing = max(0, int(spacing))
    rows = []
    row = []
    used_width = 0
    for index, preferred_width in enumerate(preferred_widths):
        preferred_width = max(0, int(preferred_width))
        next_width = (
            preferred_width
            if not row
            else used_width + spacing + preferred_width
        )
        if row and next_width > available_width:
            rows.append(tuple(row))
            row = [index]
            used_width = preferred_width
        else:
            row.append(index)
            used_width = next_width
    if row:
        rows.append(tuple(row))
    return rows


class AdaptiveWrapLayout(QLayout):
    """Height-for-width layout that never splits an atomic control unit.

    The layout owns only ``QLayoutItem`` objects and changes only their
    geometries. It never reparents or recreates widgets.

    >>> _pack_preferred_widths([30, 30, 30], 65, 5)
    [(0, 1), (2,)]
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        horizontal_spacing: int = -1,
        vertical_spacing: int = -1,
    ) -> None:
        super().__init__(parent)
        self._items = []
        self._horizontal_spacing = horizontal_spacing
        self._vertical_spacing = vertical_spacing

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self) -> bool:
        return True

    def _style_spacing(self, horizontal: bool) -> int:
        explicit = (
            self._horizontal_spacing if horizontal else self._vertical_spacing
        )
        if explicit >= 0:
            return explicit
        inherited = self.spacing()
        if inherited >= 0:
            return inherited
        parent = self.parentWidget()
        if parent is not None:
            metric_name = (
                'PM_LayoutHorizontalSpacing'
                if horizontal
                else 'PM_LayoutVerticalSpacing'
            )
            pixel_metrics = getattr(QStyle, 'PixelMetric', QStyle)
            metric = getattr(pixel_metrics, metric_name)
            value = parent.style().pixelMetric(metric)
            if value >= 0:
                return value
        return 6

    def horizontalSpacing(self) -> int:
        return self._style_spacing(True)

    def verticalSpacing(self) -> int:
        return self._style_spacing(False)

    @staticmethod
    def _item_height(item: QLayoutItem, width: int) -> int:
        if item.hasHeightForWidth():
            height = item.heightForWidth(width)
        else:
            height = item.sizeHint().height()
        return max(item.minimumSize().height(), height)

    def _visible_items(self) -> list[QLayoutItem]:
        return [item for item in self._items if not item.isEmpty()]

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        content_x = rect.x() + left
        content_y = rect.y() + top
        available_width = max(0, rect.width() - left - right)
        items = self._visible_items()
        if not items:
            return top + bottom

        horizontal_spacing = self.horizontalSpacing()
        vertical_spacing = self.verticalSpacing()
        preferred_widths = [
            max(item.minimumSize().width(), item.sizeHint().width())
            for item in items
        ]
        rows = _pack_preferred_widths(
            preferred_widths, available_width, horizontal_spacing
        )

        y = content_y
        for row_index, row in enumerate(rows):
            widths = [
                min(preferred_widths[index], available_width)
                for index in row
            ]
            expanding = [
                position
                for position, index in enumerate(row)
                if items[index].widget() is not None
                and items[index].widget().sizePolicy().horizontalPolicy()
                in (
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.MinimumExpanding,
                )
            ]
            spare_width = max(
                0,
                available_width
                - sum(widths)
                - horizontal_spacing * (len(row) - 1),
            )
            if expanding and spare_width:
                extra, remainder = divmod(spare_width, len(expanding))
                for offset, position in enumerate(expanding):
                    widths[position] += extra + (offset < remainder)
            heights = [
                self._item_height(items[index], width)
                for index, width in zip(row, widths)
            ]
            row_height = max(heights, default=0)
            x = content_x
            for index, width in zip(row, widths):
                if not test_only:
                    items[index].setGeometry(
                        QRect(x, y, width, row_height)
                    )
                x += width + horizontal_spacing
            y += row_height
            if row_index + 1 < len(rows):
                y += vertical_spacing
        return (y - rect.y()) + bottom

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, max(0, width), 0), True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def minimumSize(self) -> QSize:
        items = self._visible_items()
        left, top, right, bottom = self.getContentsMargins()
        width = max(
            (item.minimumSize().width() for item in items),
            default=0,
        )
        width += left + right
        return QSize(width, self.heightForWidth(width))

    def sizeHint(self) -> QSize:
        items = self._visible_items()
        left, _top, right, _bottom = self.getContentsMargins()
        spacing = self.horizontalSpacing()
        width = sum(
            max(item.minimumSize().width(), item.sizeHint().width())
            for item in items
        )
        if items:
            width += spacing * (len(items) - 1)
        width += left + right
        return QSize(width, self.heightForWidth(width))

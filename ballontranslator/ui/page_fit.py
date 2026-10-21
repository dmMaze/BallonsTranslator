"""A floating magnifier that toggles whole-page fit and restores the view."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

from qtpy.QtCore import QPoint, QRectF, QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QToolButton, QWidget

DEFAULT_SIZE = 156


_icon_cache = {}


def _asset_icon(size: int, outline: int) -> QIcon | None:
    """Rasterise the supplied SVG at ``size`` with an optional white rim."""
    key = ('asset', size, int(outline))
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached
    try:
        from ballontranslator.utils import shared
        from ballontranslator.ui.layer_peek import _with_white_outline
    except Exception:
        return None
    base = getattr(shared, 'ICON_DIR', '')
    if not base:
        return None
    for suffix in ('.svg', '.png'):
        candidate = Path(base) / 'page_fit' / f'magnifier{suffix}'
        if not candidate.is_file():
            continue
        supersample = size * 2
        if suffix == '.svg':
            from qtpy.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(str(candidate))
            if not renderer.isValid():
                continue
            pixmap = QPixmap(supersample, supersample)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            renderer.render(painter)
            painter.end()
        else:
            pixmap = QPixmap(str(candidate))
            if pixmap.isNull():
                continue
        if outline > 0:
            icon_pixmap = _with_white_outline(
                pixmap, max(2, size // 22), size
            )
        else:
            icon_pixmap = pixmap.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        icon = QIcon(icon_pixmap)
        _icon_cache[key] = icon
        return icon
    return None


def make_magnifier_icon(size: int = 44, outline: int = 0) -> QIcon:
    """Draw a manga-style magnifier: ink ring plus handle.

    >>> from qtpy.QtWidgets import QApplication
    >>> _ = QApplication.instance() or QApplication([])
    >>> isinstance(make_magnifier_icon(64), QIcon)
    True
    """
    asset = _asset_icon(size, outline)
    if asset is not None:
        return asset
    canvas = QPixmap(size, size)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin = max(2.0, size * 0.10)
    lens = size - margin * 2 - size * 0.18
    lens_rect = QRectF(margin, margin, lens, lens)
    handle_start = (margin + lens * 0.78, margin + lens * 0.78)
    handle_end = (size - margin * 0.6, size - margin * 0.6)

    if outline > 0:
        white = QColor(255, 255, 255)
        painter.setPen(QPen(white, max(2.0, size * 0.06) + outline * 2))
        painter.drawEllipse(lens_rect)
        painter.drawLine(
            int(handle_start[0]), int(handle_start[1]),
            int(handle_end[0]), int(handle_end[1]),
        )

    painter.setPen(QPen(QColor(24, 26, 30), max(2.0, size * 0.06)))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(lens_rect)
    painter.drawLine(
        int(handle_start[0]), int(handle_start[1]),
        int(handle_end[0]), int(handle_end[1]),
    )
    painter.end()
    return QIcon(canvas)


class PageFitController:
    """Fit the whole page, then put the previous zoom and offset back."""

    def __init__(self, canvas) -> None:
        self.canvas = canvas
        self.fitted = False
        self._saved: Optional[Tuple[float, int, int]] = None

    def _scrollbars(self):
        view = self.canvas.gv
        return view.horizontalScrollBar(), view.verticalScrollBar()

    def _page_rect(self):
        return self.canvas.sceneRect()

    def fit(self) -> bool:
        """Scale so the page fits the viewport; returns False if it cannot."""
        canvas = self.canvas
        view = getattr(canvas, 'gv', None)
        if view is None:
            return False
        rect = self._page_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        viewport = view.viewport()
        width = viewport.width()
        height = viewport.height()
        if width <= 0 or height <= 0:
            return False
        current = float(getattr(canvas, 'scale_factor', 1.0) or 1.0)
        factor = min(width / rect.width(), height / rect.height())
        if factor <= 0:
            return False
        hbar, vbar = self._scrollbars()
        self._saved = (current, hbar.value(), vbar.value())
        canvas.scaleImage(factor)
        view.centerOn(rect.center())
        self.fitted = True
        return True

    def restore(self) -> bool:
        """Return to the zoom and offset captured by :meth:`fit`."""
        if self._saved is None:
            self.fitted = False
            return False
        canvas = self.canvas
        saved_factor, h_value, v_value = self._saved
        current = float(getattr(canvas, 'scale_factor', 1.0) or 1.0)
        if current > 0:
            canvas.scaleImage(saved_factor / current)
        hbar, vbar = self._scrollbars()
        hbar.setValue(h_value)
        vbar.setValue(v_value)
        self._saved = None
        self.fitted = False
        return True

    def toggle(self) -> bool:
        """Fit when showing the page, restore when already fitted."""
        if self.fitted:
            return self.restore()
        return self.fit()


class PageFitButton(QToolButton):
    """Draggable magnifier that toggles the whole-page view."""

    def __init__(self, controller: PageFitController, parent: QWidget,
                 size: int = DEFAULT_SIZE) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName('PageFitButton')
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setAutoRaise(False)
        self._drag_from: Optional[QPoint] = None
        self._dragged = False
        self.user_moved = False
        self._size = size
        self._prepare_look(size)
        self.set_button_size(size)
        self.clicked.connect(self._on_clicked)

    def _prepare_look(self, size: int) -> None:
        self.setStyleSheet(
            'QToolButton#PageFitButton { border: none;'
            ' background: transparent; padding: 0px;'
            ' border-radius: %dpx; }' % (size // 2)
        )

    def set_button_size(self, size: int) -> None:
        size = max(40, int(size))
        self._size = size
        self.setFixedSize(size, size)
        self.setIconSize(QSize(size - 8, size - 8))
        self._prepare_look(size)
        self.refresh()

    def refresh(self) -> None:
        self.setIcon(make_magnifier_icon(self._size - 8, outline=3))
        self.setToolTip(
            self.tr('Fit the whole page')
            if not self.controller.fitted
            else self.tr('Back to the previous zoom and position')
        )

    def _on_clicked(self) -> None:
        self.controller.toggle()
        self.refresh()

    # -- dragging ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.position().toPoint()
            self._dragged = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is None:
            return super().mouseMoveEvent(event)
        delta = event.position().toPoint() - self._drag_from
        if not self._dragged and delta.manhattanLength() < 4:
            return
        self._dragged = True
        self.user_moved = True
        target = self.pos() + delta
        parent = self.parentWidget()
        if parent is not None:
            bounds = parent.rect().adjusted(
                0, 0, -self.width(), -self.height()
            )
            target.setX(max(bounds.left(), min(target.x(), bounds.right())))
            target.setY(max(bounds.top(), min(target.y(), bounds.bottom())))
        self.move(target)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
        super().mouseReleaseEvent(event)

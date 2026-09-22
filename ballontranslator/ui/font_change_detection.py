"""Observe platform font notifications without querying or changing fonts."""
from __future__ import annotations

import sys

from qtpy.QtCore import QAbstractNativeEventFilter, QObject, Signal
from qtpy.QtGui import QGuiApplication


class _WindowsFontMessages(QAbstractNativeEventFilter):
    def __init__(self, owner: FontChangeDetector) -> None:
        super().__init__()
        self.owner = owner

    def nativeEventFilter(self, event_type: object, message: object) -> tuple[bool, int]:
        if bytes(event_type) not in (b'windows_generic_MSG', b'windows_dispatcher_MSG'):
            return False, 0
        from ctypes.wintypes import MSG
        if MSG.from_address(int(message)).message == 0x001D:
            self.owner.system_fonts_changed.emit()
        return False, 0


class FontChangeDetector(QObject):
    """Report native and Qt changes; the consumer decides whether to rescan.

    >>> issubclass(FontChangeDetector, QObject)
    True
    """
    system_fonts_changed = Signal()
    qt_database_changed = Signal()

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._app = QGuiApplication.instance()
        self._stopped = False
        self._native_filter = None
        self._app.fontDatabaseChanged.connect(self._on_qt_database_changed)
        if sys.platform == 'win32':
            self._native_filter = _WindowsFontMessages(self)
            self._app.installNativeEventFilter(self._native_filter)

    def _on_qt_database_changed(self) -> None:
        if not self._stopped:
            self.qt_database_changed.emit()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._app.fontDatabaseChanged.disconnect(self._on_qt_database_changed)
        if self._native_filter is not None:
            self._app.removeNativeEventFilter(self._native_filter)
            self._native_filter = None

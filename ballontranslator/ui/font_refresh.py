"""Main-window-owned runtime font refresh; Qt mutations stay on the GUI thread."""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Optional

from qtpy.QtCore import QObject, QThread, QTimer, Signal
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import QApplication

from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.font_refresh import (
    FontconfigRefresh, invalidate_qt_fonts, log_font_refresh_debug, refresh_font_registry,
    reinitialize_current_fontconfig, runtime_font_refresh_supported, scan_custom_fonts,
)
from ballontranslator.utils.font_registry import load_custom_group_table, load_system_alias_table
from ballontranslator.utils.message import create_info_dialog
from .text_engine.font_family import register_qt_font_family_aliases
from .text_engine.layout import clear_font_metrics_cache

from ballontranslator.utils.logger import logger as LOGGER


class _FontPreparation(QThread):
    """Read fonts and metadata without running Qt font APIs on a worker.

    >>> issubclass(_FontPreparation, QThread)
    True
    """
    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.manual = False
        self.force = False
        self.fingerprints = {}
        self.locale = ''
        self.seed = b''
        self.files = None
        self.custom_groups = {}
        self.system_aliases = {}
        self.error: Optional[Exception] = None
        self.elapsed = 0.0

    def run(self) -> None:
        started = perf_counter()
        self.error = None
        self.files = None
        log_font_refresh_debug('[font-refresh][prepare] start; scan custom fonts=%s', self.manual)
        try:
            if self.force and not self.seed:
                self.seed = (Path(__file__).resolve().parents[1] /
                             'assets/font_refresh/Abel-Regular.ttf').read_bytes()
            root = Path(shared.PROGRAM_PATH)
            overrides = root / 'config/font_registry_overrides.json'
            if not overrides.exists():
                overrides = root / 'resources/font_registry_overrides.json'
            self.custom_groups = load_custom_group_table(str(overrides), self.locale)
            self.system_aliases = load_system_alias_table(str(overrides), self.locale)
            if self.manual:
                self.files = scan_custom_fonts(root / 'fonts', self.fingerprints)
        except Exception as error:
            self.error = error
        finally:
            self.elapsed = perf_counter() - started
            log_font_refresh_debug('[font-refresh][prepare] finished in %.1f ms; error=%s', self.elapsed * 1000, self.error)


class FontRefreshController(QObject):
    """Execute refresh requests and publish one settled font generation.

    Qt add/remove emits signals synchronously. Only the mutation phase ignores
    those signals; external requests received during file IO remain pending.

    >>> issubclass(FontRefreshController, QObject)
    True
    """
    refreshed = Signal()
    busy_changed = Signal(bool)
    status_changed = Signal(str, str)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.enabled = runtime_font_refresh_supported()
        self._stopped = False
        self._applying = False
        self._pending = False
        self._manual = False
        self._force = False
        self._generation = 0
        self._app = QApplication.instance()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._start)
        self._worker = _FontPreparation(self)
        self._worker.finished.connect(self._finish)

    def request_manual_refresh(self) -> None:
        log_font_refresh_debug('[font-refresh][signal] manual reload requested')
        self._queue(manual=True, force=True)

    def request_system_refresh(self) -> None:
        log_font_refresh_debug('[font-refresh][signal] Windows WM_FONTCHANGE received')
        self._queue(manual=False, force=True)

    def request_database_sync(self) -> None:
        LOGGER.info('[font-refresh][signal] Qt fontDatabaseChanged; self-generated=%s', self._applying)
        self._queue(manual=False, force=False)

    def _queue(self, manual: bool, force: bool) -> None:
        if not self.enabled or self._stopped or self._applying:
            return
        log_font_refresh_debug('[font-refresh][queue] merged=%s manual=%s force=%s; debounce=300 ms',
                    self._pending or self._worker.isRunning(), manual, force)
        self.busy_changed.emit(True)
        self.status_changed.emit(self.tr('Refreshing…'), self.tr('Refreshing fonts. Please wait.'))
        self._pending = True
        self._manual |= manual
        self._force |= force
        self._timer.start()

    def _start(self) -> None:
        if self._stopped or not self._pending or self._worker.isRunning():
            return
        if shared.FONT_REGISTRY is None:
            self._pending = self._manual = self._force = False
            self.busy_changed.emit(False)
            self.status_changed.emit(self.tr('Failed'), self.tr('The font registry is not initialized.'))
            LOGGER.error('[font-refresh][prepare] font registry is not initialized')
            return
        self._generation += 1
        log_font_refresh_debug('[font-refresh][%d][start] manual=%s force=%s', self._generation, self._manual, self._force)
        self._worker.manual = self._manual
        self._worker.force = self._force
        self._worker.locale = pcfg.display_lang
        self._worker.fingerprints = {
            path: registration.fingerprint
            for path, registration in shared.FONT_REGISTRY.registrations.items()
        }
        self._pending = self._manual = self._force = False
        self.busy_changed.emit(True)
        self._worker.start()

    def _finish(self) -> None:
        if self._stopped:
            return
        started = perf_counter()
        worker = self._worker
        self._applying = True
        fontconfig = FontconfigRefresh('skipped')
        metrics_cleared = False
        previous_families = set(shared.FONT_FAMILIES)
        success = False
        try:
            if worker.error is not None:
                raise worker.error
            if worker.force:
                if not sys.platform.startswith('linux') or self._app.platformName() in ('xcb', 'wayland', 'wayland-egl'):
                    fontconfig = reinitialize_current_fontconfig()
                log_font_refresh_debug('[font-refresh][%d][fontconfig] status=%s interval=%s',
                            self._generation, fontconfig.status, fontconfig.interval)
                invalidate_qt_fonts(QFontDatabase, worker.seed)
            log_font_refresh_debug('[font-refresh][%d][registry] rebuilding from Qt families', self._generation)
            registry = refresh_font_registry(
                QFontDatabase, shared.FONT_REGISTRY, worker.locale,
                worker.custom_groups, worker.system_aliases, worker.files,
            )
            families = set(QFontDatabase.families())
            register_qt_font_family_aliases(families, QFontDatabase.styles)
            clear_font_metrics_cache()
            metrics_cleared = True
            shared.FONT_REGISTRY = registry
            shared.FONT_FAMILIES = families
            self.refreshed.emit()
            success = True
            added, removed = families - previous_families, previous_families - families
            log_font_refresh_debug('[font-refresh][%d][publish] families=%d added=%d removed=%d; metrics cleared; selectors synchronized',
                        self._generation, len(families), len(added), len(removed))
            LOGGER.info('[font-refresh][%d][families] added=%s removed=%s',
                         self._generation, sorted(added), sorted(removed))
            self.status_changed.emit(
                self.tr('Refreshed') if fontconfig.status in ('skipped', 'refreshed') else self.tr('Check fonts'),
                self.tr('Font list refreshed: {count} families (+{added}, -{removed}) in {ms} ms.').format(
                    count=len(families), added=len(added), removed=len(removed),
                    ms=round((worker.elapsed + perf_counter() - started) * 1000)))
            if fontconfig.status not in ('skipped', 'refreshed'):
                LOGGER.warning('Fontconfig refresh %s: %s', fontconfig.status, fontconfig.detail)
                if worker.manual:
                    self._show_fontconfig_notice(fontconfig)
        except Exception as error:
            LOGGER.exception('[font-refresh][%d][failed] Font refresh failed', self._generation)
            self.status_changed.emit(self.tr('Failed'), self.tr('Could not refresh fonts: {error}').format(error=error))
            if worker.manual:
                create_info_dialog(self.tr('Could not refresh fonts. Please try again or restart the application.')
                                   + '\n' + str(error), modal=False)
        finally:
            # Even a failed metadata update may follow a successful Qt invalidate.
            if not metrics_cleared:
                clear_font_metrics_cache()
            self._applying = False
            self.busy_changed.emit(self._pending)
            LOGGER.info('[font-refresh][%d][done] success=%s preparation=%.1f ms Qt/UI=%.1f ms',
                        self._generation, success, worker.elapsed * 1000, (perf_counter() - started) * 1000)
            if self._pending:
                self.status_changed.emit(self.tr('Refreshing…'), self.tr('Another font refresh is pending.'))
                self._timer.start()

    def _show_fontconfig_notice(self, result: FontconfigRefresh) -> None:
        if result.status == 'failed':
            message = self.tr('Fontconfig could not reload its configuration. Check the font configuration and try again, or restart the application.')
        elif result.interval == 0:
            message = self.tr('Immediate system font refresh is unavailable and automatic rescanning is disabled. Restart the application to reload system fonts.')
        elif result.interval is not None:
            message = self.tr('Immediate system font refresh is unavailable. Try refreshing again after about {seconds} seconds; if fonts are still missing, restart the application.').format(seconds=result.interval)
        else:
            message = self.tr('Immediate system font refresh is unavailable. With the usual fontconfig settings, try again after about 30 seconds; if fonts are still missing, restart the application.')
        create_info_dialog(message, modal=False)

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._timer.stop()
        # Never destroy a running QThread; wait for any in-flight file read before releasing its owner.
        self._worker.requestInterruption()
        self._worker.wait()

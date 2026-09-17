from __future__ import annotations

import ctypes
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import weakref

import pytest
from qtpy import QT6
from qtpy.QtCore import QCoreApplication, QEvent, QObject
from qtpy.QtGui import QFontDatabase
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from ballontranslator.utils import shared
from ballontranslator.utils.font_registry import build_font_registry, load_custom_group_table
from ballontranslator.utils.font_refresh import (
    FontconfigRefresh, invalidate_qt_fonts, refresh_font_registry,
    reinitialize_current_fontconfig, runtime_font_refresh_supported, scan_custom_fonts,
)

SEED = Path(__file__).resolve().parents[1] / 'ballontranslator/assets/font_refresh/Abel-Regular.ttf'


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def database(app: QApplication) -> QFontDatabase | type[QFontDatabase]:
    return QFontDatabase if QT6 else QFontDatabase()


@pytest.fixture
def runtime_app(app: QApplication) -> QApplication:
    if not runtime_font_refresh_supported():
        pytest.skip('Qt 6.4+ GUI required')
    return app


@pytest.mark.parametrize('version,headless,expected', [
    ('5.15.11', False, False), ('6.3.2', False, False), ('6.4.0', False, True),
    ('6.11.2', False, True), ('6.11.2', True, False),
])
def test_support_gate(monkeypatch, version, headless, expected):
    monkeypatch.setattr('qtpy.QT_VERSION', version)
    monkeypatch.setattr(shared, 'HEADLESS', headless)
    assert runtime_font_refresh_supported() is expected


def test_non_linux_does_not_load_fontconfig():
    with patch('ballontranslator.utils.font_refresh.sys.platform', 'darwin'), patch(
        'ballontranslator.utils.font_refresh.ctypes.CDLL', side_effect=AssertionError,
    ):
        assert reinitialize_current_fontconfig().status == 'skipped'


def test_fontconfig_unavailable():
    with patch('ballontranslator.utils.font_refresh.sys.platform', 'linux'), patch(
        'ballontranslator.utils.font_refresh.ctypes.CDLL', side_effect=OSError('missing'),
    ):
        result = reinitialize_current_fontconfig()
    assert result.status == 'unavailable'
    assert result.interval is None


@pytest.mark.parametrize('success,status', [(1, 'refreshed'), (0, 'failed')])
def test_fontconfig_c_signature_and_failure(success, status):
    library = SimpleNamespace(FcConfigGetCurrent=Mock(return_value=123),
        FcConfigGetRescanInterval=Mock(return_value=0), FcInitReinitialize=Mock(return_value=success))
    with patch('ballontranslator.utils.font_refresh.sys.platform', 'linux'), patch(
        'ballontranslator.utils.font_refresh.ctypes.CDLL', return_value=library,
    ):
        result = reinitialize_current_fontconfig()
    assert (result.status, result.interval) == (status, 0)
    assert library.FcConfigGetCurrent.restype == ctypes.c_void_p
    assert library.FcInitReinitialize.argtypes == []
    assert library.FcInitReinitialize.restype == ctypes.c_int
    library.FcConfigGetRescanInterval.assert_called_once_with(123)


def test_invalidate_preserves_other_application_fonts(database) -> None:
    font_id = QFontDatabase.addApplicationFont(str(SEED))
    assert font_id >= 0
    try:
        expected = QFontDatabase.applicationFontFamilies(font_id)
        invalidate_qt_fonts(QFontDatabase, SEED.read_bytes())
        assert all(family in database.families() for family in expected)
        assert QFontDatabase.applicationFontFamilies(font_id) == expected
    finally:
        QFontDatabase.removeApplicationFont(font_id)


def test_scan_unchanged_and_unreadable_files(tmp_path):
    path = tmp_path / 'font.ttf'
    path.write_bytes(SEED.read_bytes())
    initial = scan_custom_fonts(tmp_path, {})
    key = str(path.resolve())
    assert initial[key].names
    assert scan_custom_fonts(tmp_path, {key: initial[key].fingerprint}) == {key: None}
    with patch.object(Path, 'read_bytes', side_effect=PermissionError('locked')):
        assert scan_custom_fonts(tmp_path, {}) == {key: None}


def test_custom_add_replace_remove_and_automatic_reuse(database, tmp_path: Path) -> None:
    path = tmp_path / 'font.ttf'
    path.write_bytes(SEED.read_bytes())
    registry = build_font_registry(database, [str(path)], database.families())
    try:
        key = str(path.resolve())
        old_id = registry.registrations[key].font_id
        # A system-only refresh reuses custom entries without reading files.
        updated = refresh_font_registry(database, registry, 'en-US', {}, {})
        assert updated.custom_entries is registry.custom_entries
        assert updated.registrations[key].font_id == old_id
        files = scan_custom_fonts(tmp_path, {key: registry.registrations[key].fingerprint})
        registry = refresh_font_registry(database, updated, 'en-US', {}, {}, files)
        assert registry.registrations[key].font_id == old_id
        path.write_bytes(SEED.read_bytes() + b'\0')
        files = scan_custom_fonts(tmp_path, {key: registry.registrations[key].fingerprint})
        registry = refresh_font_registry(database, registry, 'en-US', {}, {}, files)
        assert registry.registrations[key].font_id != old_id
        assert QFontDatabase.applicationFontFamilies(old_id) == []
        path.unlink()
        registry = refresh_font_registry(database, registry, 'en-US', {}, {}, scan_custom_fonts(tmp_path, {}))
        assert not registry.custom_entries
        assert not registry.registrations
    finally:
        for record in registry.registrations.values():
            QFontDatabase.removeApplicationFont(record.font_id)


def test_bad_replacement_keeps_registered_font(database, tmp_path: Path) -> None:
    path = tmp_path / 'font.ttf'
    path.write_bytes(SEED.read_bytes())
    registry = build_font_registry(database, [str(path)], database.families())
    try:
        key = str(path.resolve())
        old_id = registry.registrations[key].font_id
        path.write_bytes(b'broken font')
        updated = refresh_font_registry(database, registry, 'en-US', {}, {}, scan_custom_fonts(tmp_path, {}))
        assert updated.registrations[key].font_id == old_id
        assert updated.custom_entries
    finally:
        for record in registry.registrations.values():
            QFontDatabase.removeApplicationFont(record.font_id)


def test_refresh_preserves_group_exclusions_and_selected_hidden_font(
    database, monkeypatch, tmp_path: Path,
) -> None:
    from ballontranslator.ui.configpanel import FontExcludeDialog
    from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox
    from ballontranslator.utils.config import pcfg

    overrides = tmp_path / 'groups.json'
    overrides.write_text(json.dumps({'custom_groups': [{
        'canonical': 'Review Group', 'display': '审阅字体',
        'members': [{'canonical': 'Abel'}],
    }]}), encoding='utf-8')
    groups = load_custom_group_table(str(overrides))
    font = tmp_path / 'font.ttf'
    font.write_bytes(SEED.read_bytes())
    registry = build_font_registry(database, [str(font)], database.families(),
                                   font_registry_path=str(overrides))
    monkeypatch.setattr(shared, 'FONT_REGISTRY', registry)
    monkeypatch.setattr(shared, 'FONT_FAMILIES', set(database.families()))
    monkeypatch.setattr(pcfg, 'excluded_fonts', ['Abel', 'Missing Font'])
    picker = FontFamilyComboBox()
    picker.update_font_entries(registry.entries(excluded=pcfg.excluded_fonts))
    picker.set_current_family('Abel')
    changes = []
    picker.param_changed.connect(lambda *args: changes.append(args))
    dialog = FontExcludeDialog()
    try:
        # Automatic sync, manual sync, deletion, and return all retain exclusions.
        for files in (None, scan_custom_fonts(tmp_path, {}), {}, scan_custom_fonts(tmp_path, {})):
            registry = refresh_font_registry(database, registry, 'en-US', groups, {}, files)
            shared.FONT_REGISTRY = registry
            picker.update_font_entries(registry.entries(excluded=pcfg.excluded_fonts))
            for excluded in ('Abel', 'Review Group', '审阅字体'):
                for only_custom in (False, True):
                    assert not any(e.canonical_family in ('Abel', 'Review Group')
                                   for e in registry.entries(only_custom, [excluded]))
            assert picker.currentText() == 'Abel'
            assert dialog.get_excluded_fonts() == ['Abel', 'Missing Font']
            assert pcfg.excluded_fonts == ['Abel', 'Missing Font']
        assert not changes
    finally:
        picker.deleteLater()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        for record in registry.registrations.values():
            database.removeApplicationFont(record.font_id)


@pytest.mark.parametrize('vertical', [False, True])
@pytest.mark.parametrize('stroke', [False, True])
def test_refreshed_text_matches_fresh_layout_and_export(
    runtime_app: QApplication, database, monkeypatch, vertical: bool, stroke: bool,
) -> None:
    from qtpy.QtCore import QRectF
    from qtpy.QtGui import QColor, QImage, QPainter, QTextCursor
    from qtpy.QtWidgets import QGraphicsScene
    from ballontranslator.ui.text_engine.item import TextBlkItem
    from ballontranslator.ui.text_engine.layout import clear_font_metrics_cache
    from ballontranslator.utils.textblock import TextBlock
    from ballontranslator.utils.text_effects import StrokeEffect, TextEffectStack

    if 'Abel' in database.families():
        pytest.skip('The probe font is already installed')
    registry = build_font_registry(database, [], database.families())
    monkeypatch.setattr(shared, 'FONT_REGISTRY', registry)

    def make_item() -> tuple[TextBlkItem, QGraphicsScene]:
        block = TextBlock([0, 0, 500, 300])
        block._bounding_rect = [0, 0, 500, 300]
        block.translation = 'WWWW abcdef 0123'
        block.vertical = vertical
        block.fontformat.font_family = 'Abel'
        block.fontformat.font_size = 26
        if stroke:
            block.fontformat.text_effects = TextEffectStack(effects=(StrokeEffect(width=0.12),))
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)
        return item, scene

    def render(scene: QGraphicsScene) -> QImage:
        image = QImage(600, 400, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor('white'))
        painter = QPainter(image)
        scene.render(painter, QRectF(0, 0, 600, 400), QRectF(-30, -30, 600, 400))
        painter.end()
        return image

    item, scene = make_item()
    before = render(scene)
    cursor = item.textCursor()
    cursor.setPosition(5)
    cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
    item.setTextCursor(cursor)
    document = item.document()
    html, undo = document.toHtml(), document.availableUndoSteps()
    font_id = database.addApplicationFont(str(SEED))
    assert font_id >= 0
    try:
        shared.FONT_REGISTRY = refresh_font_registry(database, registry, 'en-US', {}, {})
        clear_font_metrics_cache()
        item.refresh_font_metrics()
        assert (item.textCursor().position(), item.textCursor().anchor()) == (1, 5)
        cursor = item.textCursor()
        cursor.clearSelection()
        item.setTextCursor(cursor)
        preview = render(scene)
        assert preview != before
        fresh, fresh_scene = make_item()
        try:
            assert preview == render(fresh_scene)
            item.set_export_effect_render(True)
            assert preview == render(scene)
        finally:
            item.set_export_effect_render(False)
            fresh_scene.clear()
        assert document.toHtml() == html
        assert document.availableUndoSteps() == undo
    finally:
        scene.clear()
        database.removeApplicationFont(font_id)


def wait_refresh(controller, expected, results):
    for _ in range(100):
        QTest.qWait(25)
        if len(results) >= expected:
            break
    assert len(results) == expected


def test_controller_coalesces_and_clears_metrics(runtime_app: QApplication, monkeypatch, tmp_path: Path) -> None:
    from ballontranslator.ui.font_refresh import FontRefreshController
    from ballontranslator.ui.text_engine.layout import get_char_width, get_punc_rect
    from qtpy.QtGui import QFont
    monkeypatch.setattr(shared, 'PROGRAM_PATH', str(tmp_path))
    monkeypatch.setattr(shared, 'HEADLESS', False)
    monkeypatch.setattr(shared, 'FONT_REGISTRY', build_font_registry(QFontDatabase, [], QFontDatabase.families()))
    monkeypatch.setattr(shared, 'FONT_FAMILIES', set(QFontDatabase.families()))
    owner = QObject()
    controller = FontRefreshController(owner)
    from ballontranslator.ui.font_change_detection import FontChangeDetector
    detector = FontChangeDetector(owner)
    detector.system_fonts_changed.connect(controller.request_system_refresh)
    detector.qt_database_changed.connect(controller.request_database_sync)
    results = []
    statuses = []
    busy = []
    controller.status_changed.connect(lambda label, detail: statuses.append((label, detail)))
    controller.busy_changed.connect(lambda value: busy.append(value))
    # Use a pure Python callback that captures no QObject.
    controller.refreshed.connect(lambda: results.append(True))
    try:
        get_char_width('W', 'Missing Runtime Font', 18, QFont.Weight.Normal, False)
        get_punc_rect('.', 'Missing Runtime Font', 18, QFont.Weight.Normal, False)
        for _ in range(5):
            controller.request_system_refresh()
        assert busy[-1] is True
        assert statuses[-1][0] == 'Refreshing…'
        wait_refresh(controller, 1, results)
        assert statuses[-1][0] == 'Refreshed'
        assert 'families' in statuses[-1][1]
        assert busy[-1] is False
        assert get_char_width.cache_info().currsize == 0
        assert get_punc_rect.cache_info().currsize == 0
        QTest.qWait(450)
        assert len(results) == 1
        controller.request_manual_refresh()
        wait_refresh(controller, 2, results)
    finally:
        detector.stop()
        controller.shutdown()
        controller.deleteLater()
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_headless_controller_never_starts_worker(app, monkeypatch):
    from ballontranslator.ui.font_refresh import FontRefreshController
    monkeypatch.setattr(shared, 'HEADLESS', True)
    owner = QObject()
    controller = FontRefreshController(owner)
    try:
        controller.request_manual_refresh()
        assert not controller._pending
        assert not controller._worker.seed
    finally:
        controller.shutdown()


def test_removed_picker_selection_survives_multiple_refreshes(app, monkeypatch):
    from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox
    from ballontranslator.utils.font_registry import FontEntry, FontRegistry
    entry = FontEntry('Removed', 'Removed', 'Removed', 'system')
    monkeypatch.setattr(shared, 'FONT_REGISTRY', FontRegistry(system_entries=[entry]))
    picker = FontFamilyComboBox()
    changes = []
    picker.param_changed.connect(lambda *args: changes.append(args))
    picker.update_font_entries([entry])
    picker.set_current_family('Removed')
    changes.clear()
    monkeypatch.setattr(shared, 'FONT_REGISTRY', FontRegistry())
    picker.update_font_entries([])
    picker.update_font_entries([])
    assert picker.currentText() == 'Removed'
    assert not changes
    picker.deleteLater()


def test_controller_shutdown_releases_owner(app, monkeypatch):
    from ballontranslator.ui.font_refresh import FontRefreshController
    monkeypatch.setattr(shared, 'HEADLESS', False)
    owner = QObject()
    controller = FontRefreshController(owner)
    owner_ref, controller_ref = weakref.ref(owner), weakref.ref(controller)
    controller.shutdown()
    owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del owner, controller
    gc.collect()
    assert owner_ref() is None
    assert controller_ref() is None


def test_invalidate_failure_removes_only_owned_id():
    database = Mock()
    database.addApplicationFontFromData.return_value = -1
    with pytest.raises(RuntimeError):
        invalidate_qt_fonts(database, b'seed')
    database.removeApplicationFont.assert_not_called()
    database.addApplicationFontFromData.return_value = 17
    database.removeApplicationFont.side_effect = [False, True]
    with pytest.raises(RuntimeError):
        invalidate_qt_fonts(database, b'seed')
    assert [call.args for call in database.removeApplicationFont.call_args_list] == [(17,), (17,)]
    database.removeAllApplicationFonts.assert_not_called()


def test_directory_enumeration_error_aborts_snapshot(tmp_path):
    def fail_walk(directory, onerror):
        onerror(PermissionError('unreadable directory'))
        return iter(())
    with patch('os.walk', side_effect=fail_walk), pytest.raises(PermissionError):
        scan_custom_fonts(tmp_path, {})


def test_windows_filter_ignores_other_messages(app):
    import sys
    if sys.platform != 'win32':
        pytest.skip('Windows MSG layout required')
    from ctypes.wintypes import MSG
    from ballontranslator.ui.font_change_detection import _WindowsFontMessages
    owner = SimpleNamespace(system_fonts_changed=SimpleNamespace(emit=Mock()))
    native_filter = _WindowsFontMessages(owner)
    assert native_filter.nativeEventFilter(b'irrelevant', None) == (False, 0)
    message = MSG()
    message.message = 0x001D
    for _ in range(5):
        assert native_filter.nativeEventFilter(b'windows_generic_MSG', ctypes.addressof(message)) == (False, 0)
    assert owner.system_fonts_changed.emit.call_count == 5
    message.message = 0x000F
    native_filter.nativeEventFilter(b'windows_generic_MSG', ctypes.addressof(message))
    assert owner.system_fonts_changed.emit.call_count == 5


def test_refresh_failure_reports_status(runtime_app: QApplication, monkeypatch, tmp_path: Path) -> None:
    from ballontranslator.ui.font_refresh import FontRefreshController
    monkeypatch.setattr(shared, 'HEADLESS', False)
    monkeypatch.setattr(shared, 'PROGRAM_PATH', str(tmp_path))
    monkeypatch.setattr(shared, 'FONT_REGISTRY', build_font_registry(QFontDatabase, [], QFontDatabase.families()))
    monkeypatch.setattr(shared, 'FONT_FAMILIES', set(QFontDatabase.families()))
    owner = QObject()
    controller = FontRefreshController(owner)
    statuses = []
    controller.status_changed.connect(lambda label, detail: statuses.append((label, detail)))
    try:
        with patch('ballontranslator.ui.font_refresh.sys.platform', 'win32'), patch(
            'ballontranslator.ui.font_refresh.invalidate_qt_fonts', side_effect=RuntimeError('seed rejected'),
        ), patch(
            'ballontranslator.ui.font_refresh.create_info_dialog',
        ):
            controller.request_manual_refresh()
            for _ in range(100):
                QTest.qWait(25)
                if statuses and statuses[-1][0] == 'Failed':
                    break
        assert statuses[-1] == ('Failed', 'Could not refresh fonts: seed rejected')
    finally:
        controller.shutdown()
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_reload_button_animates_only_while_visible_and_busy(app):
    from qtpy.QtWidgets import QWidget
    from ballontranslator.ui.text_engine.formatting.panel import FontReloadButton
    owner = QWidget()
    button = FontReloadButton(owner)
    try:
        owner.show()
        button.set_busy(True)
        QTest.qWait(100)
        assert button._angle > 0
        assert not button.isEnabled()
        assert button.text() == ''
        button.hide()
        assert not button._rotation_timer.isActive()
        button.show()
        assert button._rotation_timer.isActive()
        button.set_busy(False)
        assert not button._rotation_timer.isActive()
        assert button._angle == 0
        assert button.isEnabled()
    finally:
        owner.close()
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize('setting,enabled', [(None, False), ('0', False), ('1', True)])
def test_font_diagnostics_are_opt_in(monkeypatch, tmp_path, setting, enabled):
    import logging
    from ballontranslator.utils.font_refresh import LOGGER
    if setting is None:
        monkeypatch.delenv('BT_FONT_REFRESH_DEBUG', raising=False)
    else:
        monkeypatch.setenv('BT_FONT_REFRESH_DEBUG', setting)
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    handler = Capture()
    previous_level = LOGGER.level
    previous_handlers = list(LOGGER.handlers)
    LOGGER.addHandler(handler)
    try:
        scan_custom_fonts(tmp_path, {})
        diagnostics = [r for r in records if r.levelno == logging.DEBUG]
        assert bool(diagnostics) is enabled
        assert all(r.funcName == 'scan_custom_fonts' for r in diagnostics)
        assert LOGGER.level == previous_level
        assert LOGGER.handlers == previous_handlers + [handler]
    finally:
        LOGGER.removeHandler(handler)


def test_detector_reports_without_refreshing_and_disconnects(app):
    from ballontranslator.ui.font_change_detection import FontChangeDetector
    owner = QObject()
    detector = FontChangeDetector(owner)
    notifications = []
    detector.qt_database_changed.connect(lambda: notifications.append(True))
    reference = weakref.ref(detector)
    try:
        with patch.object(QFontDatabase, 'families', side_effect=AssertionError('detection must not query fonts')), patch.object(
            QFontDatabase, 'addApplicationFontFromData', side_effect=AssertionError('detection must not register fonts'),
        ):
            app.fontDatabaseChanged.emit()
            assert notifications == [True]
            detector.stop()
            detector.stop()
            app.fontDatabaseChanged.emit()
            assert notifications == [True]
    finally:
        detector.stop()
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del detector, owner
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize('platform,backend,force,expected', [
    ('win32', 'windows', True, ['invalidate']),
    ('win32', 'windows', False, []),
    ('darwin', 'cocoa', True, []),
    ('linux', 'xcb', True, ['fontconfig', 'invalidate']),
    ('linux', 'wayland', True, ['fontconfig', 'invalidate']),
    ('linux', 'offscreen', True, ['invalidate']),
    ('linux', 'xcb', False, []),
])
def test_platform_reload_paths(runtime_app: QApplication, monkeypatch, platform: str, backend: str, force: bool, expected: list[str]) -> None:
    from ballontranslator.ui.font_refresh import FontRefreshController
    monkeypatch.setattr(shared, 'HEADLESS', False)
    registry = build_font_registry(QFontDatabase, [], QFontDatabase.families())
    monkeypatch.setattr(shared, 'FONT_REGISTRY', registry)
    monkeypatch.setattr(shared, 'FONT_FAMILIES', set(QFontDatabase.families()))
    owner = QObject()
    controller = FontRefreshController(owner)
    controller._app = SimpleNamespace(platformName=lambda: backend)
    controller._worker.force = force
    calls = []
    def fontconfig():
        calls.append('fontconfig')
        return FontconfigRefresh('refreshed')
    try:
        with patch('ballontranslator.ui.font_refresh.sys.platform', platform), patch(
            'ballontranslator.ui.font_refresh.reinitialize_current_fontconfig', side_effect=fontconfig,
        ), patch('ballontranslator.ui.font_refresh.invalidate_qt_fonts', side_effect=lambda *args: calls.append('invalidate')), patch(
            'ballontranslator.ui.font_refresh.refresh_font_registry', return_value=registry,
        ) as rebuild:
            controller._finish()
            assert calls == expected
            rebuild.assert_called_once()
            controller.request_manual_refresh()
            assert controller._manual
            assert controller._force is (platform != 'darwin')
    finally:
        controller.shutdown()
        owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

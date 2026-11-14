import gc
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

import ballontranslator.ui.mainwindow as mainwindow_module
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.run_pipeline_dialog import RunPipelineDialog
from ballontranslator.utils import shared
from ballontranslator.utils.config import MULTI_EXTRACT_MAX_SLOTS, pcfg
from ballontranslator.utils.multi_extract import conflict_count, has_conflict
from ballontranslator.utils.textblock import TextBlock

# The dialog persists its settings when it is destroyed; point the config at a
# scratch file for the whole module so a late save cannot touch the developer's
# config/config.json.
_config_dir = None
_original_config_path = None
_APP = None


def get_app() -> QApplication:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(sys.argv[:1])
    return _APP


def setUpModule():
    global _config_dir, _original_config_path
    _original_config_path = shared.CONFIG_PATH
    _config_dir = tempfile.TemporaryDirectory()
    shared.CONFIG_PATH = os.path.join(_config_dir.name, 'config.json')


def tearDownModule():
    get_app().processEvents()
    gc.collect()
    shared.CONFIG_PATH = _original_config_path
    _config_dir.cleanup()


def make_block(left, top, text):
    block = TextBlock([left, top, 40, 20])
    block._bounding_rect = [left, top, 40, 20]
    block.text = [text]
    return block


class _ProjectStub:
    def __init__(self, pages, current='page-1'):
        self.pages = dict(pages)
        self.current_img = current
        self.progress = {}

    def update_page_progress(self, page, code):
        self.progress[page] = self.progress.get(page, 0) | code

    def save(self):
        self.saved = True


class _ManagerStub:
    def __init__(self, stopped=False, running=False):
        self.thread_running = [running]
        self.imgtrans_thread = SimpleNamespace(
            isStopRequested=lambda: stopped,
            isRunning=lambda: self.thread_running[0],
            clearStopRequest=lambda: None,
        )
        self.selections = []

    def selectTextDetector(self, name):
        self.selections.append(('textdetector', name))

    def selectOCR(self, name):
        self.selections.append(('ocr', name))


class MultiExtractDialogTest(unittest.TestCase):
    """The fourth workflow configures slots without touching the live modules."""

    def setUp(self):
        self._mode = pcfg.run_pipeline_mode
        self._slots = list(pcfg.module.multi_extract_slots)
        self._detector = pcfg.module.textdetector
        self._ocr = pcfg.module.ocr

    def tearDown(self):
        pcfg.run_pipeline_mode = self._mode
        pcfg.module.multi_extract_slots = self._slots
        pcfg.module.textdetector = self._detector
        pcfg.module.ocr = self._ocr

    def _dialog(self):
        app = get_app()
        dialog = RunPipelineDialog()
        dialog.show()
        app.processEvents()
        self.addCleanup(self._destroy, dialog)
        return dialog

    @staticmethod
    def _destroy(dialog):
        dialog.setParent(None)
        dialog.deleteLater()
        get_app().processEvents()

    def test_workflow_list_ends_with_multi_pass_extraction(self):
        dialog = self._dialog()
        selector = dialog.workflow_selector

        self.assertEqual(
            [selector.itemText(i) for i in range(selector.count())],
            ['Pipeline', 'Rendering', 'Text Extraction', 'Multi-Pass Extraction'],
        )
        self.assertEqual(RunPipelineDialog.WORKFLOW_MODES[3], 'multi_extraction')

    def test_selecting_the_workflow_switches_page_and_buttons(self):
        dialog = self._dialog()

        dialog.workflow_selector.setCurrentIndex(3)
        get_app().processEvents()

        self.assertTrue(dialog.is_multi_extract_mode())
        self.assertFalse(dialog.is_extraction_mode())
        self.assertEqual(dialog.content_stack.currentIndex(), 2)
        self.assertTrue(dialog.run_button.isVisible())
        # Continue would resume the single-pass pipeline; multi-pass has no such
        # state to resume from.
        self.assertFalse(dialog.continue_button.isVisible())

    def test_slot_edits_persist_without_changing_the_active_modules(self):
        dialog = self._dialog()
        row = dialog.multi_extract_slot_rows[1]

        row['enabled'].setChecked(True)
        row['ocr'].setCurrentText('manga_ocr')
        get_app().processEvents()

        slots = pcfg.module.multi_extract_slots
        self.assertEqual(len(slots), MULTI_EXTRACT_MAX_SLOTS)
        self.assertTrue(slots[1]['enabled'])
        self.assertEqual(slots[1]['ocr'], 'manga_ocr')
        # The slots describe a run; they must not retarget the project itself.
        self.assertEqual(pcfg.module.ocr, self._ocr)

    def test_enabled_slots_keep_table_order(self):
        dialog = self._dialog()
        dialog.multi_extract_slot_rows[0]['enabled'].setChecked(True)
        dialog.multi_extract_slot_rows[2]['enabled'].setChecked(True)
        dialog.multi_extract_slot_rows[2]['textdetector'].setCurrentText('ctd')

        enabled = dialog.enabled_multi_extract_slots()

        self.assertEqual(len(enabled), 2)
        self.assertEqual(enabled[0], dialog.multi_extract_slot_configs()[0])
        self.assertEqual(enabled[1], dialog.multi_extract_slot_configs()[2])


class MultiExtractRunTest(unittest.TestCase):
    """Pass chaining, stop handling and the merge/write-back."""

    def setUp(self):
        self._flags = (pcfg.module.enable_detect, pcfg.module.enable_ocr)
        self._slots = list(pcfg.module.multi_extract_slots)

    def tearDown(self):
        pcfg.module.enable_detect, pcfg.module.enable_ocr = self._flags
        pcfg.module.multi_extract_slots = self._slots

    def _host(self, pages, stopped=False, state=None):
        host = SimpleNamespace(
            tr=lambda text, *args, **kwargs: text,
            MULTI_EXTRACT_START_RETRY_MS=MainWindow.MULTI_EXTRACT_START_RETRY_MS,
            MULTI_EXTRACT_START_RETRIES=MainWindow.MULTI_EXTRACT_START_RETRIES,
            cleanup=[],
            imgtrans_proj=_ProjectStub(pages),
            module_manager=_ManagerStub(stopped=stopped),
            canvas=SimpleNamespace(
                cleared=[], saves=[],
                clear_undostack=lambda update_saved_step=False: host.canvas.cleared.append(True),
                setProjSaveState=lambda value: host.canvas.saves.append(value),
            ),
            st_manager=SimpleNamespace(refreshed=0),
        )
        host.st_manager.updateSceneTextitems = lambda: None
        host._multi_extract_state = state
        host._queued_passes = []
        host._run_multi_extract_pass = lambda *args: host._queued_passes.append('pass')
        host._launch_multi_extract_pass = (
            lambda state, attempt=0: MainWindow._launch_multi_extract_pass(host, state, attempt)
        )
        host._end_extract_only_run = lambda: None
        host.on_imgtrans_pipeline_finished = lambda: host.cleanup.append(True)
        host._refresh_multi_extract_review_action = lambda: None
        host._merge_multi_extract_pages = (
            lambda state: MainWindow._merge_multi_extract_pages(host, state)
        )
        host._finish_multi_extract_run = (
            lambda merge: MainWindow._finish_multi_extract_run(host, merge)
        )
        host._advance_multi_extract_run = (
            lambda: MainWindow._advance_multi_extract_run(host)
        )
        return host

    @staticmethod
    def _state(slots, pages):
        return {
            'slots': slots,
            'pages': list(pages),
            'index': 0,
            'pass_pending': True,
            'snapshots': {page: [] for page in pages},
            'stage_flags': (True, True),
            'modules': ('ctd', 'mit48px'),
        }

    def test_start_refuses_a_single_pass(self):
        host = self._host({'page-1': []})
        calls = []

        with mock.patch.object(mainwindow_module, 'create_info_dialog', lambda msg, *a, **k: calls.append(msg)):
            MainWindow._start_multi_extract_run(host, ['page-1'], [{'enabled': True}])

        self.assertEqual(len(calls), 1)
        self.assertIsNone(host._multi_extract_state)

    def test_advance_snapshots_the_page_and_queues_the_next_pass(self):
        pages = {'page-1': [make_block(0, 0, 'あ')]}
        host = self._host(pages)
        host._multi_extract_state = self._state(
            [{'textdetector': 'ctd'}, {'textdetector': 'ysgyolo'}], ['page-1'],
        )

        queued = MainWindow._advance_multi_extract_run(host)

        self.assertTrue(queued)
        self.assertEqual(host._multi_extract_state['index'], 1)
        self.assertEqual(len(host._multi_extract_state['snapshots']['page-1']), 1)
        self.assertEqual(host._queued_passes, ['pass'])

    def test_advance_merges_after_the_last_pass(self):
        pages = {'page-1': []}
        host = self._host(pages)
        state = self._state([{'textdetector': 'ctd'}, {'textdetector': 'ysgyolo'}], ['page-1'])
        state['index'] = 1
        state['snapshots']['page-1'] = [[make_block(0, 0, 'あ')], [make_block(0, 0, 'い')]]
        host._multi_extract_state = state
        calls = []

        with mock.patch.object(mainwindow_module, 'create_info_dialog', lambda msg, *a, **k: calls.append(msg)):
            queued = MainWindow._advance_multi_extract_run(host)

        self.assertFalse(queued)
        self.assertEqual(host._queued_passes, [])
        self.assertEqual(len(host.imgtrans_proj.pages['page-1']), 1)
        self.assertEqual(conflict_count(host.imgtrans_proj.pages), 1)
        self.assertEqual(len(calls), 1)

    def test_stopping_keeps_the_last_pass_and_skips_the_merge(self):
        page = [make_block(0, 0, 'あ')]
        host = self._host({'page-1': page}, stopped=True)
        state = self._state([{'textdetector': 'ctd'}, {'textdetector': 'ysgyolo'}], ['page-1'])
        state['index'] = 1
        host._multi_extract_state = state

        queued = MainWindow._advance_multi_extract_run(host)

        self.assertFalse(queued)
        self.assertIsNone(host._multi_extract_state)
        self.assertIs(host.imgtrans_proj.pages['page-1'], page)
        self.assertEqual(conflict_count(host.imgtrans_proj.pages), 0)

    def test_next_pass_waits_for_the_previous_worker_thread(self):
        host = self._host({'page-1': []})
        host.module_manager.thread_running[0] = True
        host._run_multi_extract_pass = (
            lambda attempt=0: MainWindow._run_multi_extract_pass(host, attempt)
        )
        host._multi_extract_state = self._state(
            [
                {'textdetector': 'ctd', 'ocr': 'mit48px'},
                {'textdetector': 'ctd', 'ocr': 'manga_ocr'},
            ],
            ['page-1'],
        )
        scheduled = []

        with mock.patch.object(
            mainwindow_module.QTimer,
            'singleShot',
            lambda ms, cb=None, *args: scheduled.append((ms, cb)),
        ):
            MainWindow._run_multi_extract_pass(host, 0)

            # A QThread that is still running ignores start(); wait it out.
            self.assertEqual(host.module_manager.selections, [])
            # Still owed a snapshot: the pass is queued, not finished.
            self.assertTrue(host._multi_extract_state['pass_pending'])
            self.assertEqual(
                scheduled[0][0], MainWindow.MULTI_EXTRACT_START_RETRY_MS,
            )

            host.module_manager.thread_running[0] = False
            scheduled[0][1]()

        self.assertEqual(
            host.module_manager.selections,
            [('textdetector', 'ctd'), ('ocr', 'mit48px')],
        )
        self.assertTrue(host._multi_extract_state['pass_pending'])

    def test_giving_up_when_the_thread_never_stops(self):
        host = self._host({'page-1': []})
        host.module_manager.thread_running[0] = True
        host._run_multi_extract_pass = (
            lambda attempt=0: MainWindow._run_multi_extract_pass(host, attempt)
        )
        host._multi_extract_state = self._state([{}, {}], ['page-1'])

        with mock.patch.object(mainwindow_module.QTimer, 'singleShot', lambda *a, **k: None):
            MainWindow._run_multi_extract_pass(
                host, MainWindow.MULTI_EXTRACT_START_RETRIES,
            )

        self.assertIsNone(host._multi_extract_state)
        # Only the restore of the user's own modules is left in the list.
        self.assertEqual(
            host.module_manager.selections,
            [('textdetector', 'ctd'), ('ocr', 'mit48px')],
        )

    def test_repeated_finish_event_does_not_advance_twice(self):
        host = self._host({'page-1': [make_block(0, 0, 'あ')]})
        state = self._state([{}, {}, {}], ['page-1'])
        state['pass_pending'] = False
        host._multi_extract_state = state

        queued = MainWindow._advance_multi_extract_run(host)

        self.assertTrue(queued)
        self.assertEqual(host._multi_extract_state['index'], 0)
        self.assertEqual(host._multi_extract_state['snapshots']['page-1'], [])

    def test_thread_finished_signal_drives_the_chain(self):
        host = self._host({'page-1': [make_block(0, 0, 'あ')]})
        host._multi_extract_state = self._state([{}, {}], ['page-1'])

        MainWindow._on_imgtrans_thread_finished(host)

        self.assertEqual(host._multi_extract_state['index'], 1)
        self.assertEqual(host._queued_passes, ['pass'])

    def test_progress_finished_handler_yields_to_the_chain(self):
        host = self._host({'page-1': []})
        host._multi_extract_state = self._state([{}, {}], ['page-1'])

        MainWindow.on_imgtrans_pipeline_finished(host)

        # No bookkeeping while passes remain; the chain restores it at the end.
        self.assertEqual(host.cleanup, [])

    def test_finishing_the_last_pass_runs_the_normal_cleanup(self):
        host = self._host({'page-1': []})
        state = self._state([{}, {}], ['page-1'])
        state['index'] = 1
        state['snapshots']['page-1'] = [[make_block(0, 0, 'あ')], [make_block(0, 0, 'い')]]
        host._multi_extract_state = state

        with mock.patch.object(mainwindow_module, 'create_info_dialog', lambda *a, **k: None):
            MainWindow._advance_multi_extract_run(host)

        self.assertEqual(host.cleanup, [True])

    def test_merge_marks_conflicts_and_progress(self):
        host = self._host({'page-1': []})
        state = self._state([{}, {}], ['page-1'])
        state['snapshots']['page-1'] = [[make_block(0, 0, 'あ')], [make_block(0, 0, 'い')]]

        marked = MainWindow._merge_multi_extract_pages(host, state)

        blocks = host.imgtrans_proj.pages['page-1']
        self.assertEqual(marked, 1)
        self.assertTrue(has_conflict(blocks[0]))
        self.assertEqual(blocks[0].get_text(), 'あ')
        self.assertEqual(host.imgtrans_proj.progress['page-1'], 3)  # FIN_DET | FIN_OCR
        self.assertEqual(host.canvas.saves, [True, False])
        self.assertFalse(host._imported_text_dirty)

    def test_merge_skips_pages_a_pass_never_produced(self):
        host = self._host({'page-1': [], 'page-2': []})
        state = self._state([{}, {}], ['page-1', 'page-2'])
        state['snapshots']['page-1'] = [[make_block(0, 0, 'あ')], []]

        MainWindow._merge_multi_extract_pages(host, state)

        self.assertEqual(len(host.imgtrans_proj.pages['page-1']), 1)
        self.assertEqual(host.imgtrans_proj.pages['page-2'], [])
        self.assertNotIn('page-2', host.imgtrans_proj.progress)


if __name__ == '__main__':
    unittest.main()

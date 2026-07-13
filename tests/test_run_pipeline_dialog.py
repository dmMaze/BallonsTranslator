import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QObject, QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QToolButton,
)

from ballontranslator.ui.run_pipeline_dialog import (
    DialogCloseButton,
    PipelineModuleButton,
    RUN_PIPELINE_DIALOG_WIDTH,
    RunPipelineDialog,
)
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.mainwindowbars import TitleBar
from ballontranslator.ui.module_manager import ModuleManager
from ballontranslator.ui.llm_modality import (
    LLM_MODALITY_IMAGE,
    LLM_MODALITY_TEXT,
    LLM_MODALITY_VISION,
)
from ballontranslator.utils.config import ProgramConfig, json_dump_program_config, pcfg
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


def get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class RunPipelineDialogTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def setUp(self):
        self._stage_states = [pcfg.module.stage_enabled(idx) for idx in range(4)]
        self._pipeline_mode = pcfg.run_pipeline_mode
        self._render_without_text_style_update = (
            pcfg.render_without_text_style_update
        )
        self._visibility_states = (
            pcfg.show_textdetector_tool,
            pcfg.show_ocr_tool,
            pcfg.show_translator_tool,
            pcfg.show_inpainter_tool,
        )
        for idx in range(4):
            pcfg.module.set_stage_enabled(idx, True)
        pcfg.show_textdetector_tool = True
        pcfg.show_ocr_tool = True
        pcfg.show_translator_tool = True
        pcfg.show_inpainter_tool = True
        pcfg.run_pipeline_mode = 'automation'
        pcfg.render_without_text_style_update = False

    def tearDown(self):
        for idx, enabled in enumerate(self._stage_states):
            pcfg.module.set_stage_enabled(idx, enabled)
        (
            pcfg.show_textdetector_tool,
            pcfg.show_ocr_tool,
            pcfg.show_translator_tool,
            pcfg.show_inpainter_tool,
        ) = self._visibility_states
        pcfg.run_pipeline_mode = self._pipeline_mode
        pcfg.render_without_text_style_update = (
            self._render_without_text_style_update
        )

    def test_dialog_uses_frameless_rounded_surface(self):
        dialog = RunPipelineDialog()
        window_type = getattr(Qt, 'WindowType', Qt)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)

        self.assertEqual(dialog.width(), RUN_PIPELINE_DIALOG_WIDTH)
        self.assertTrue(dialog.windowFlags() & window_type.Dialog)
        self.assertTrue(dialog.windowFlags() & window_type.FramelessWindowHint)
        self.assertTrue(dialog.testAttribute(widget_attribute.WA_TranslucentBackground))
        self.assertIsNotNone(dialog.findChild(QFrame, 'RunPipelineSurface'))
        self.assertIsNotNone(dialog.findChild(DialogCloseButton, 'RunPipelineCloseButton'))
        selector = dialog.findChild(QComboBox, 'RunPipelineWorkflowSelector')
        self.assertEqual(selector.currentIndex(), 0)
        self.assertEqual(
            [selector.itemText(i) for i in range(selector.count())],
            ['Automation', 'Rendering'],
        )
        module_buttons = dialog.findChildren(PipelineModuleButton, 'RunPipelineModuleButton')
        self.assertEqual(len(module_buttons), 4)
        self.assertTrue(all(button.isChecked() for button in module_buttons))
        self.assertTrue(all(not button.icon_label.pixmap().isNull() for button in module_buttons))
        self.assertEqual(module_buttons[0].active_icon_name, 'textdetect_activate.svg')
        self.assertEqual(module_buttons[0].inactive_icon_name, 'textdetect.svg')
        self.assertEqual(
            [button.modality for button in module_buttons],
            [
                LLM_MODALITY_VISION,
                LLM_MODALITY_VISION,
                LLM_MODALITY_TEXT,
                LLM_MODALITY_IMAGE,
            ],
        )
        self.assertIn('rgba(30, 147, 229, 46)', module_buttons[0].icon_label.styleSheet())
        module_buttons[0].click()
        self.assertFalse(module_buttons[0].isChecked())
        self.assertFalse(pcfg.module.enable_detect)
        self.assertFalse(module_buttons[0].text_label.property('moduleActive'))
        self.assertIn('background-color: transparent', module_buttons[0].icon_label.styleSheet())
        self.assertEqual(
            {
                label.text()
                for label in dialog.findChildren(QLabel, 'RunPipelineSectionTitle')
            },
            {'Activate Modules', 'Settings'},
        )
        self.assertIsNotNone(dialog.findChild(QDockWidget, 'RunPipelineContentDock'))
        stack = dialog.findChild(QStackedWidget, 'RunPipelineContentStack')
        self.assertEqual(stack.currentIndex(), 0)
        self.assertIsNotNone(dialog.findChild(QToolButton, 'RunPipelineSettingsHeader'))
        self.assertTrue(dialog.continue_button.isDefault())
        self.assertFalse(dialog.run_button.isHidden())
        self.assertFalse(dialog.continue_button.isHidden())
        self.assertTrue(dialog.render_button.isHidden())

        selector.setCurrentIndex(1)
        self.assertEqual(stack.currentIndex(), 1)
        self.assertTrue(dialog.run_button.isHidden())
        self.assertTrue(dialog.continue_button.isHidden())
        self.assertFalse(dialog.render_button.isHidden())
        self.assertTrue(dialog.render_button.isDefault())
        render_option = dialog.findChild(
            QCheckBox,
            'RunPipelineRenderWithoutTextStyleUpdate',
        )
        self.assertIsNotNone(render_option)
        self.assertFalse(render_option.isChecked())
        self.assertFalse(hasattr(dialog, 'cancel_button'))
        dialog.close()

    def test_action_buttons_return_the_pipeline_choice(self):
        dialog = RunPipelineDialog()
        dialog.run_button.click()
        self.assertEqual(dialog.result(), RunPipelineDialog.RUN)

        dialog = RunPipelineDialog()
        dialog.continue_button.click()
        self.assertEqual(dialog.result(), RunPipelineDialog.CONTINUE)

        dialog = RunPipelineDialog()
        dialog.workflow_selector.setCurrentIndex(1)
        dialog.render_button.click()
        self.assertEqual(dialog.result(), RunPipelineDialog.RENDER)

        dialog = RunPipelineDialog()
        dialog.close_button.click()
        self.assertEqual(dialog.result(), dialog.Rejected)

    def test_dialog_uses_platform_move_resize_backend(self):
        dialog = RunPipelineDialog()
        dialog.show()
        self.app.processEvents()

        original_size = dialog.size()
        dialog.resize(original_size.width() + 80, original_size.height() + 60)
        self.assertEqual(dialog.width(), original_size.width() + 80)
        self.assertEqual(dialog.height(), original_size.height() + 60)

        with patch(
            'ballontranslator.ui.run_pipeline_dialog.'
            'FramelessMoveResize.startSystemMove'
        ) as start_move:
            QTest.mousePress(
                dialog.title_label,
                Qt.MouseButton.LeftButton,
                pos=dialog.title_label.rect().center(),
            )
            self.app.processEvents()
            QTest.mouseRelease(
                dialog.title_label,
                Qt.MouseButton.LeftButton,
                pos=dialog.title_label.rect().center(),
            )

            selector_center = dialog.workflow_selector.mapTo(
                dialog,
                dialog.workflow_selector.rect().center(),
            ).x()
            QTest.mouseClick(
                dialog,
                Qt.MouseButton.LeftButton,
                pos=QPoint(selector_center, 8),
            )
            QTest.mouseClick(
                dialog,
                Qt.MouseButton.LeftButton,
                pos=QPoint(dialog.width() - 10, 8),
            )
            self.app.processEvents()
            self.assertEqual(start_move.call_count, 3)

        with patch(
            'ballontranslator.ui.run_pipeline_dialog.'
            'FramelessMoveResize.starSystemResize'
        ) as start_resize:
            QTest.mousePress(
                dialog,
                Qt.MouseButton.LeftButton,
                pos=QPoint(1, dialog.height() // 2),
            )
            self.app.processEvents()
            start_resize.assert_called_once()

        dialog.close()
        self.assertFalse(dialog._app_event_filter_installed)

    def test_global_event_filter_checks_relevance_before_event_type(self):
        dialog = RunPipelineDialog()
        dialog.show()
        self.app.processEvents()
        irrelevant_event = SimpleNamespace(
            type=lambda: self.fail('event type should not be requested')
        )
        with patch('qtpy.QtWidgets.QDialog.eventFilter', return_value=False):
            self.assertFalse(dialog.eventFilter(QObject(), irrelevant_event))
        dialog.close()

    def test_mainwindow_dispatches_the_selected_pipeline_action(self):
        calls = []
        owner = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(is_all_pages_no_text=True),
            on_run_imgtrans=lambda **kwargs: calls.append(kwargs),
        )

        class FakeDialog:
            RUN = RunPipelineDialog.RUN
            CONTINUE = RunPipelineDialog.CONTINUE
            RENDER = RunPipelineDialog.RENDER
            result = 0
            preserve_style = False

            def __init__(self, parent):
                self.parent = parent
                self.render_without_text_style_update = SimpleNamespace(
                    isChecked=lambda: self.preserve_style
                )

            def exec_(self):
                return self.result

        with patch('ballontranslator.ui.mainwindow.RunPipelineDialog', FakeDialog):
            FakeDialog.result = FakeDialog.CONTINUE
            MainWindow.run_imgtrans(owner)
            self.assertEqual(calls, [{'continue_mode': True}])

            calls.clear()
            FakeDialog.result = FakeDialog.RUN
            MainWindow.run_imgtrans(owner)
            self.assertEqual(calls, [{}])

            calls.clear()
            FakeDialog.result = FakeDialog.RENDER
            FakeDialog.preserve_style = True
            MainWindow.run_imgtrans(owner)
            self.assertEqual(calls, [{'render_only': True}])
            self.assertTrue(owner._run_imgtrans_wo_textstyle_update)

            calls.clear()
            FakeDialog.result = 0
            MainWindow.run_imgtrans(owner)
            self.assertEqual(calls, [])

    def test_view_actions_only_control_bottom_bar_visibility(self):
        window = QMainWindow()
        title_bar = TitleBar(window)
        visibility_texts = [action.text() for action in title_bar.moduleVisibilityActions]
        self.assertEqual(
            visibility_texts,
            [
                'Show Text Detection',
                'Show OCR',
                'Show Translation',
                'Show Inpainting',
            ],
        )
        self.assertTrue(all(action.isChecked() for action in title_bar.moduleVisibilityActions))
        view_actions = title_bar.viewMenu.actions()
        self.assertEqual(
            [action.text() for action in view_actions[:2]],
            ['Dark Mode', 'Display Language'],
        )
        self.assertFalse(any(action.isSeparator() for action in view_actions[:2]))
        self.assertFalse(hasattr(title_bar, 'runToolBtn'))

        emitted = []
        title_bar.show_module.connect(lambda idx, checked: emitted.append((idx, checked)))
        title_bar.moduleVisibilityActions[0].trigger()
        self.assertEqual(emitted, [(0, False)])
        self.assertTrue(pcfg.module.enable_detect)

        widgets = [SimpleNamespace(visible=True) for _ in range(4)]
        for widget in widgets:
            widget.setVisible = lambda visible, target=widget: setattr(target, 'visible', visible)
        owner = SimpleNamespace(
            bottomBar=SimpleNamespace(
                textdet_selector=widgets[0],
                ocr_selector=widgets[1],
                trans_selector=widgets[2],
                inpaint_selector=widgets[3],
            )
        )
        owner._set_module_tool_visibility = lambda idx, visible: (
            MainWindow._set_module_tool_visibility(owner, idx, visible)
        )
        with patch('ballontranslator.ui.mainwindow.save_config') as save:
            MainWindow.on_show_module(owner, 1, False)
        self.assertFalse(pcfg.show_ocr_tool)
        self.assertFalse(widgets[1].visible)
        save.assert_called_once_with()

        MainWindow._set_module_tool_visibility(owner, 0, False)
        self.assertFalse(widgets[0].visible)
        self.assertTrue(widgets[2].visible)
        self.assertTrue(widgets[3].visible)
        title_bar.deleteLater()
        window.deleteLater()

    def test_tool_visibility_round_trips_through_program_config(self):
        config = ProgramConfig(show_ocr_tool=False)
        restored = ProgramConfig(**json.loads(json_dump_program_config(config)))
        self.assertFalse(restored.show_ocr_tool)

    def test_pipeline_selection_round_trips_through_program_config(self):
        pcfg.run_pipeline_mode = 'rendering'
        pcfg.render_without_text_style_update = True
        dialog = RunPipelineDialog()
        self.assertEqual(dialog.workflow_selector.currentIndex(), 1)
        self.assertFalse(dialog.render_button.isHidden())
        self.assertTrue(dialog.render_without_text_style_update.isChecked())

        dialog.workflow_selector.setCurrentIndex(0)
        self.assertEqual(pcfg.run_pipeline_mode, 'automation')
        dialog.render_without_text_style_update.setChecked(False)
        self.assertFalse(pcfg.render_without_text_style_update)
        dialog.close()

        config = ProgramConfig(
            run_pipeline_mode='rendering',
            render_without_text_style_update=True,
        )
        restored = ProgramConfig(**json.loads(json_dump_program_config(config)))
        self.assertEqual(restored.run_pipeline_mode, 'rendering')
        self.assertTrue(restored.render_without_text_style_update)

    def test_render_only_snapshots_complete_global_format(self):
        global_format = FontFormat(
            font_family='Render Font',
            font_size=47,
            stroke_width=0.18,
            frgb=[1, 2, 3],
            srgb=[4, 5, 6],
            alignment=2,
            vertical=True,
            opacity=0.75,
        )
        calls = []
        owner = SimpleNamespace(
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=False,
            _render_global_format=None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=global_format)
            ),
            bottomBar=SimpleNamespace(
                textblockChecker=SimpleNamespace(isChecked=lambda: False)
            ),
            postprocess_mt_toggle=True,
            imgtrans_proj=SimpleNamespace(pages={}),
            st_manager=SimpleNamespace(updateTextBlkList=lambda: None),
            module_manager=SimpleNamespace(
                runImgtransPipeline=lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                )
            ),
        )

        MainWindow.on_run_imgtrans(owner, render_only=True)

        snapshot = owner._render_global_format
        self.assertIsNot(snapshot, global_format)
        self.assertEqual(snapshot.font_family, 'Render Font')
        self.assertEqual(snapshot.font_size, 47)
        self.assertEqual(snapshot.stroke_width, 0.18)
        self.assertEqual(snapshot.frgb, [1, 2, 3])
        self.assertEqual(snapshot.srgb, [4, 5, 6])
        self.assertEqual(snapshot.alignment, 2)
        self.assertTrue(snapshot.vertical)
        self.assertEqual(snapshot.opacity, 0.75)
        global_format.font_size = 12
        self.assertEqual(snapshot.font_size, 47)
        self.assertEqual(calls, [((None,), {'render_only': True})])

    def test_render_only_applies_typesetting_flags_to_every_block(self):
        global_format = FontFormat(
            font_family='Render Font',
            font_size=47,
            stroke_width=0.18,
            frgb=[1, 2, 3],
            srgb=[4, 5, 6],
            alignment=2,
            vertical=True,
            opacity=0.75,
            shadow_radius=3,
            shadow_strength=0.4,
            shadow_color=[7, 8, 9],
            shadow_offset=[2, 1],
        )
        blocks = [TextBlock(), TextBlock()]
        project = SimpleNamespace(
            num_pages=1,
            get_blklist_byidx=lambda _: blocks,
            set_current_img_byidx=lambda _: None,
            save=lambda: None,
        )
        owner = SimpleNamespace(
            imgtrans_proj=project,
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=True,
            _render_global_format=global_format,
            postprocess_translations=lambda _: None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=FontFormat())
            ),
            st_manager=SimpleNamespace(
                auto_textlayout_flag=False,
                updateSceneTextitems=lambda: None,
                textblk_item_list=[],
            ),
            pageList=SimpleNamespace(
                currentIndex=lambda: SimpleNamespace(row=lambda: 0)
            ),
            canvas=SimpleNamespace(updateCanvas=lambda: None),
            saveCurrentPage=lambda *args: None,
        )
        flag_names = (
            'let_fntsize_flag',
            'let_fntstroke_flag',
            'let_fntcolor_flag',
            'let_fnt_scolor_flag',
            'let_alignment_flag',
            'let_fnteffect_flag',
            'let_writing_mode_flag',
            'let_family_flag',
        )
        old_flags = {name: getattr(pcfg, name) for name in flag_names}
        try:
            for name in flag_names:
                setattr(pcfg, name, 1)
            MainWindow.on_pagtrans_finished(owner, 0)
        finally:
            for name, value in old_flags.items():
                setattr(pcfg, name, value)

        for block in blocks:
            self.assertEqual(block.font_size, 47)
            self.assertEqual(block.stroke_width, 0.18)
            self.assertEqual(block.fontformat.frgb, [1, 2, 3])
            self.assertEqual(block.fontformat.srgb, [4, 5, 6])
            self.assertEqual(block.alignment, 2)
            self.assertTrue(block.vertical)
            self.assertEqual(block.font_family, 'Render Font')
            self.assertEqual(block.fontformat.opacity, 0.75)
            self.assertEqual(block.fontformat.shadow_radius, 3)
            self.assertEqual(block.fontformat.shadow_strength, 0.4)
            self.assertEqual(block.fontformat.shadow_color, [7, 8, 9])
            self.assertEqual(block.fontformat.shadow_offset, [2, 1])

    def test_render_only_skips_module_preparation_and_finishes_every_page(self):
        pages = []
        finished = []
        owner = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(is_empty=False, num_pages=3),
            progress_msgbox=SimpleNamespace(hide=lambda: None),
            terminateRunningThread=lambda: None,
            page_trans_finished=SimpleNamespace(emit=pages.append),
            imgtrans_pipeline_finished=SimpleNamespace(
                emit=lambda: finished.append(True)
            ),
            _prepare_modules_then=lambda *args, **kwargs: self.fail(
                'Rendering must not prepare automation modules'
            ),
        )

        ModuleManager.runImgtransPipeline(owner, render_only=True)

        self.assertEqual(pages, [0, 1, 2])
        self.assertEqual(finished, [True])


if __name__ == '__main__':
    unittest.main()

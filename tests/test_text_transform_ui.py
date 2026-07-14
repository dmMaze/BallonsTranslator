import os
import unittest
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QFocusEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QLineEdit, QWidget

from ballontranslator.ui.text_advanced_format import (
    CommittedTransformControl,
    TextAdvancedFormatPanel,
)
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils import shared as app_shared


_APP = QApplication.instance() or QApplication([])


def item_with_transform(horizontal=1.0, vertical=1.0, slant=0.0):
    return SimpleNamespace(
        blk=SimpleNamespace(
            fontformat=FontFormat(
                horizontal_scale=horizontal,
                vertical_scale=vertical,
                slant_angle=slant,
            )
        )
    )


class CommittedTransformControlTest(unittest.TestCase):
    def make_control(self, percentage=True):
        control = CommittedTransformControl(
            'Scale' if percentage else 'Angle',
            'horizontal_scale' if percentage else 'slant_angle',
            percentage,
        )
        control.set_model_value(1.0 if percentage else 0.0)
        return control

    def test_percentage_display_and_all_accepted_inputs(self):
        control = self.make_control()
        self.assertEqual(control.editor.text(), '100.0%')
        commits = []
        control.commit_requested.connect(
            lambda name, value: commits.append((name, value))
        )

        for user_text in ('120', '120.0', '120%', '120.0%'):
            with self.subTest(user_text=user_text):
                count_before_typing = len(commits)
                control.set_model_value(1.0)
                control.editor.setText(user_text)
                control._on_text_edited()
                self.assertEqual(len(commits), count_before_typing)
                self.assertTrue(control.commit_pending())
                self.assertEqual(commits[-1], ('horizontal_scale', 1.2))
                self.assertEqual(control.editor.text(), '120.0%')

        control.set_model_value(0.5)
        self.assertEqual(control.editor.text(), '50.0%')

    def test_typing_is_pending_escape_reverts_and_invalid_focus_out_does_not_commit(self):
        host = QWidget()
        control = self.make_control()
        other = QLineEdit(host)
        control.setParent(host)
        host.show()
        commits = []
        control.commit_requested.connect(lambda *args: commits.append(args))

        control.editor.setFocus()
        control.editor.selectAll()
        QTest.keyClicks(control.editor, '175')
        self.assertEqual(control.state, control.PENDING_TEXT)
        self.assertEqual(commits, [])
        QTest.keyClick(control.editor, Qt.Key.Key_Escape)
        self.assertEqual(control.state, control.IDLE)
        self.assertEqual(control.editor.text(), '100.0%')
        self.assertEqual(commits, [])

        control.editor.setFocus()
        control.editor.selectAll()
        QTest.keyClicks(control.editor, '-')
        _APP.sendEvent(
            control.editor,
            QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason),
        )
        self.assertEqual(control.state, control.IDLE)
        self.assertEqual(control.editor.text(), '100.0%')
        self.assertEqual(commits, [])
        host.close()

    def test_enter_and_focus_out_each_commit_once(self):
        host = QWidget()
        control = self.make_control()
        other = QLineEdit(host)
        control.setParent(host)
        host.show()
        commits = []
        control.commit_requested.connect(lambda name, value: commits.append(value))

        control.editor.setFocus()
        control.editor.selectAll()
        QTest.keyClicks(control.editor, '125%')
        QTest.keyClick(control.editor, Qt.Key.Key_Return)
        other.setFocus()
        _APP.processEvents()
        self.assertEqual(commits, [1.25])

        control.editor.setFocus()
        control.editor.selectAll()
        QTest.keyClicks(control.editor, '80')
        other.setFocus()
        _APP.processEvents()
        self.assertEqual(commits, [1.25, 0.8])
        self.assertEqual(control.editor.text(), '80.0%')
        host.close()

    def test_drag_many_previews_has_one_release_and_zero_or_escape_cancels(self):
        control = self.make_control()
        previews = []
        releases = []
        cancels = []
        control.preview_requested.connect(
            lambda name, delta: previews.append((name, delta))
        )
        control.drag_commit_requested.connect(
            lambda name, delta: releases.append((name, delta))
        )
        control.preview_canceled.connect(lambda name: cancels.append(name))

        control._start_drag()
        control._move_drag(4)
        control._move_drag(3)
        control._move_drag(-2)
        self.assertEqual([delta for _name, delta in previews], [0.04, 0.07, 0.05])
        self.assertEqual(control.editor.text(), '105.0%')
        control._finish_drag()
        self.assertEqual(releases, [('horizontal_scale', 0.05)])

        control._start_drag()
        control._finish_drag()
        self.assertEqual(len(releases), 1)
        self.assertEqual(cancels, ['horizontal_scale'])

        control._start_drag()
        control._move_drag(10)
        control.cancel_preview()
        self.assertEqual(len(releases), 1)
        self.assertEqual(cancels, ['horizontal_scale', 'horizontal_scale'])
        self.assertEqual(control.editor.text(), '100.0%')

    def test_angle_uses_degree_format_and_finite_range(self):
        control = self.make_control(percentage=False)
        self.assertEqual(control.editor.text(), '0.0\N{DEGREE SIGN}')
        commits = []
        control.commit_requested.connect(lambda _name, value: commits.append(value))
        control.editor.setText('-17.25\N{DEGREE SIGN}')
        control._on_text_edited()
        self.assertTrue(control.commit_pending())
        self.assertEqual(commits, [-17.25])
        self.assertEqual(control.editor.text(), '-17.2\N{DEGREE SIGN}')

        for invalid in ('nan', 'inf', '-46', '46'):
            control.editor.setText(invalid)
            control._on_text_edited()
            self.assertFalse(control.commit_pending())
        self.assertEqual(commits, [-17.25])


class TextAdvancedFormatPanelTransformTest(unittest.TestCase):
    def make_panel(self):
        app_shared.register_view_widget = lambda *_args, **_kwargs: None
        return TextAdvancedFormatPanel(
            'Advanced Text Format',
            config_name='text_transform_test_panel',
            config_expand_name='text_transform_test_expand',
            on_format_changed=lambda *_args: None,
        )

    def test_mixed_selection_and_precise_refresh_are_model_views_only(self):
        panel = self.make_panel()
        first = item_with_transform(1.234567, 0.5, -7.0)
        second = item_with_transform(1.5, 0.5, 3.0)
        commits = []
        panel.transform_commit_requested.connect(lambda *args: commits.append(args))

        panel.set_transform_items([first])
        self.assertEqual(panel.horizontal_scale_control.editor.text(), '123.5%')
        self.assertEqual(first.blk.fontformat.horizontal_scale, 1.234567)
        self.assertEqual(commits, [])

        panel.set_transform_items([first, second])
        self.assertEqual(panel.horizontal_scale_control.editor.text(), '\N{EM DASH}')
        self.assertEqual(panel.vertical_scale_control.editor.text(), '50.0%')
        self.assertEqual(panel.slant_angle_control.editor.text(), '\N{EM DASH}')
        self.assertEqual(commits, [])

        panel.horizontal_scale_control.editor.setText('120.00%')
        panel.horizontal_scale_control._on_text_edited()
        panel.finish_pending_transform_edits()
        self.assertEqual(commits, [('horizontal_scale', 1.2)])

    def test_selection_boundary_commits_pending_value_before_refresh(self):
        panel = self.make_panel()
        old_item = item_with_transform(1.0, 1.0, 0.0)
        new_item = item_with_transform(0.5, 2.0, 10.0)
        observations = []
        panel.transform_commit_requested.connect(
            lambda name, value: observations.append((name, value))
        )

        panel.set_transform_items([old_item])
        control = panel.horizontal_scale_control
        control.editor.setText('135%')
        control._on_text_edited()
        panel.finish_pending_transform_edits()
        panel.set_transform_items([new_item])

        self.assertEqual(observations, [('horizontal_scale', 1.35)])
        self.assertEqual(control.editor.text(), '50.0%')


if __name__ == '__main__':
    unittest.main()

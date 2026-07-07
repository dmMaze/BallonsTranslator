import os
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QObject, QEvent
from qtpy.QtWidgets import QApplication


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()

from ballontranslator.ui.llm_profile_widgets import ProfileCardWidget
from ballontranslator.utils.llm_profiles import default_profile


class TypeSensitiveEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type.User)
        self.type_requested = False

    def type(self):
        self.type_requested = True
        raise RecursionError('event type should not be requested')


class ProfileCardEventFilterTest(unittest.TestCase):
    def setUp(self):
        qapp()
        self.card = ProfileCardWidget(default_profile('OpenAI'))

    def tearDown(self):
        self.card._remove_app_event_filter()
        self.card.deleteLater()

    def test_app_filter_is_only_installed_while_expanded(self):
        self.assertFalse(self.card._app_filter_installed)

        self.card.setExpanded(True)
        self.assertTrue(self.card._app_filter_installed)

        self.card.setExpanded(False)
        self.assertFalse(self.card._app_filter_installed)

    def test_non_widget_app_events_do_not_request_event_type(self):
        self.card.setExpanded(True)
        event = TypeSensitiveEvent()

        self.card.eventFilter(QObject(), event)

        self.assertFalse(event.type_requested)


if __name__ == '__main__':
    unittest.main()

from typing import List, Optional, Tuple

from qtpy.QtCore import QCoreApplication, Qt
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import QApplication, QMenu, QWidget

from ballontranslator.utils.config import pcfg


def create_text_edit_context_menu(
    parent: Optional[QWidget],
    *,
    has_selection: bool,
    can_undo: bool,
    can_redo: bool,
) -> Tuple[QMenu, List[QAction]]:
    menu = QMenu(parent)
    menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    undo_action = menu.addAction(
        QCoreApplication.translate('TextEditingContextMenu', 'Undo')
    )
    undo_action.setData('undo')
    undo_action.setShortcut(QKeySequence.StandardKey.Undo)
    undo_action.setEnabled(can_undo)

    redo_action = menu.addAction(
        QCoreApplication.translate('TextEditingContextMenu', 'Redo')
    )
    redo_action.setData('redo')
    redo_action.setShortcut(QKeySequence.StandardKey.Redo)
    redo_action.setEnabled(can_redo)

    menu.addSeparator()

    cut_action = menu.addAction(
        QCoreApplication.translate('TextEditingContextMenu', 'Cut')
    )
    cut_action.setData('cut')
    cut_action.setShortcut(QKeySequence.StandardKey.Cut)
    cut_action.setEnabled(has_selection)

    copy_action = menu.addAction(
        QCoreApplication.translate('TextEditingContextMenu', 'Copy')
    )
    copy_action.setData('copy')
    copy_action.setShortcut(QKeySequence.StandardKey.Copy)
    copy_action.setEnabled(has_selection)

    paste_action = menu.addAction(
        QCoreApplication.translate('TextEditingContextMenu', 'Paste')
    )
    paste_action.setData('paste')
    paste_action.setShortcut(QKeySequence.StandardKey.Paste)
    clipboard_mime = QApplication.clipboard().mimeData()
    paste_action.setEnabled(
        clipboard_mime is not None and clipboard_mime.hasText()
    )

    delete_action = menu.addAction(
        QCoreApplication.translate('TextEditingContextMenu', 'Delete')
    )
    delete_action.setData('delete')
    delete_action.setShortcut(QKeySequence.StandardKey.Delete)
    delete_action.setEnabled(has_selection)

    menu.addSeparator()
    quick_insert_menu = menu.addMenu(
        QCoreApplication.translate('TextEditingContextMenu', 'Quick Insert')
    )
    quick_insert_actions = []
    for character in pcfg.quick_insert_characters:
        action = quick_insert_menu.addAction(character.replace('&', '&&'))
        action.setData(character)
        quick_insert_actions.append(action)
    quick_insert_menu.setEnabled(bool(quick_insert_actions))
    return menu, quick_insert_actions

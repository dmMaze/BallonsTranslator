"""User-configurable keyboard shortcut editor (settings panel section).

The defaults mirror the previously hard-coded shortcuts; each action keeps
its effective binding unless the user overrides it in ``pcfg.shortcuts``.
"""

from typing import Dict, List

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.config import pcfg
from ballontranslator.utils.shortcut_conflicts import find_conflict_keys

from .custom_widget import PanelGroupBox

DEFAULT_SHORTCUTS: Dict[str, List[str]] = {
    "prev_page": ["A"],
    "prev_page_alt": ["PgUp"],
    "next_page": ["D"],
    "next_page_alt": ["PgDown"],
    "textedit_mode": ["T"],
    "textblock_mode": ["W"],
    "drawboard_mode": ["P"],
    "zoom_in": ["Ctrl++"],
    "zoom_out": ["Ctrl+-"],
    "delete_blks": ["Del"],
    "delete_blks_alt": ["Ctrl+D"],
    "select_all": ["Ctrl+A"],
    "bold": ["Ctrl+B"],
    "italic": ["Ctrl+I"],
    "underline": ["Ctrl+U"],
    "undo": ["Ctrl+Z"],
    "redo": ["Ctrl+Y"],
    "page_search": ["Ctrl+F"],
    "global_search": ["Ctrl+G"],
    "escape": ["Escape"],
    "space_inpaint": ["Space"],
    "hand_tool": ["H"],
    "rect_tool": ["R"],
    "inpaint_tool": ["J"],
    "pen_tool": ["B"],
    "merge_tool": ["Ctrl+Shift+M"],
    "path_reorder": [],
}

_ACTION_NAMES: Dict[str, str] = {
    "prev_page": "Page Up",
    "next_page": "Page Down",
    "prev_page_alt": "Page Up (alt)",
    "next_page_alt": "Page Down (alt)",
    "textedit_mode": "Text Editor",
    "textblock_mode": "Text Block",
    "drawboard_mode": "Draw Board",
    "zoom_in": "Zoom In",
    "zoom_out": "Zoom Out",
    "delete_blks": "Delete",
    "delete_blks_alt": "Delete (alt)",
    "select_all": "Select All",
    "bold": "Bold",
    "italic": "Italic",
    "underline": "Underline",
    "undo": "Undo",
    "redo": "Redo",
    "page_search": "Page Search",
    "global_search": "Global Search",
    "escape": "Escape",
    "space_inpaint": "Inpaint",
    "hand_tool": "Hand Tool",
    "rect_tool": "Rect Tool",
    "inpaint_tool": "Inpaint Tool",
    "pen_tool": "Pen Tool",
    "merge_tool": "Merge Tool",
    "path_reorder": "Path Reorder",
}

_SHORTCUT_GROUPS: List[tuple] = [
    ("Navigation", ["prev_page", "next_page", "prev_page_alt", "next_page_alt"]),
    ("View", ["zoom_in", "zoom_out"]),
    (
        "Edit",
        [
            "textedit_mode",
            "textblock_mode",
            "drawboard_mode",
            "delete_blks",
            "delete_blks_alt",
            "select_all",
            "bold",
            "italic",
            "underline",
            "undo",
            "redo",
        ],
    ),
    (
        "Tools",
        [
            "hand_tool",
            "rect_tool",
            "inpaint_tool",
            "pen_tool",
            "merge_tool",
            "path_reorder",
            "space_inpaint",
        ],
    ),
    ("Search", ["page_search", "global_search"]),
    ("General", ["escape"]),
]


def resolve_shortcut_keys(action_id: str) -> List[str]:
    """Effective key sequences for *action_id* — user override, else default."""
    if action_id in pcfg.shortcuts:
        keys = pcfg.shortcuts[action_id]
        if not isinstance(keys, list):
            keys = [keys] if keys else []
        return list(keys)
    return list(DEFAULT_SHORTCUTS.get(action_id, []))


class _ShortcutRow(QWidget):
    """A row for editing the shortcuts of a single action."""

    shortcut_changed = Signal()

    def __init__(self, action_id: str, parent=None):
        super().__init__(parent)
        self.action_id = action_id
        self._conflict_keys: set = set()

        h = QHBoxLayout(self)
        h.setContentsMargins(2, 6, 2, 6)
        h.setSpacing(6)

        name = QLabel(self.tr(_ACTION_NAMES.get(action_id, action_id)))
        name.setObjectName("ShortcutActionName")
        name.setFixedWidth(150)
        h.addWidget(name)

        self.shortcuts_widget = QWidget()
        self.shortcuts_layout = QHBoxLayout(self.shortcuts_widget)
        self.shortcuts_layout.setContentsMargins(0, 0, 0, 0)
        self.shortcuts_layout.setSpacing(4)
        self.shortcuts_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        h.addWidget(self.shortcuts_widget, 1)

        add_btn = QPushButton("+")
        add_btn.setObjectName("ShortcutAddBtn")
        add_btn.setFixedSize(24, 24)
        add_btn.setToolTip(self.tr("Add shortcut"))
        add_btn.clicked.connect(self._add_shortcut)
        h.addWidget(add_btn)

        clear_btn = QPushButton(self.tr("Del"))
        clear_btn.setObjectName("ShortcutDelBtn")
        clear_btn.setFixedSize(32, 24)
        clear_btn.setToolTip(self.tr("Remove all shortcuts"))
        clear_btn.clicked.connect(self._clear)
        h.addWidget(clear_btn)

        reset_btn = QPushButton(self.tr("Rst"))
        reset_btn.setObjectName("ShortcutRstBtn")
        reset_btn.setFixedSize(32, 24)
        reset_btn.setToolTip(self.tr("Reset to default"))
        reset_btn.clicked.connect(self._reset)
        h.addWidget(reset_btn)

        self._rebuild_pills()

    def effective_keys(self) -> List[str]:
        return resolve_shortcut_keys(self.action_id)

    def _rebuild_pills(self):
        while self.shortcuts_layout.count():
            item = self.shortcuts_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        keys = self.effective_keys()
        if keys:
            for k in keys:
                pill = QFrame()
                pill.setObjectName("ShortcutPill")
                is_conflict = k in self._conflict_keys
                pill.setProperty("conflict", is_conflict)
                pill_layout = QHBoxLayout(pill)
                pill_layout.setContentsMargins(8, 1, 4, 1)
                pill_layout.setSpacing(2)
                lbl = QLabel(k)
                lbl.setObjectName("ShortcutPillLabel")
                lbl.setProperty("conflict", is_conflict)
                pill_layout.addWidget(lbl)
                close_btn = QPushButton("x")
                close_btn.setObjectName("ShortcutPillCloseBtn")
                close_btn.setFixedSize(20, 20)
                close_btn.clicked.connect(
                    lambda checked, ks=k: self._remove_shortcut(ks)
                )
                pill_layout.addWidget(close_btn)
                self.shortcuts_layout.addWidget(pill)
        else:
            placeholder = QLabel(self.tr("— None —"))
            placeholder.setObjectName("ShortcutNoneLabel")
            self.shortcuts_layout.addWidget(placeholder)

    def _add_shortcut(self):
        edit = QKeySequenceEdit()
        edit.setFixedWidth(120)
        edit.setFixedHeight(24)
        self.shortcuts_layout.addWidget(edit)
        edit.setFocus()

        def on_finished():
            seq = edit.keySequence().toString()
            edit.deleteLater()
            if seq:
                keys = self.effective_keys()
                if seq not in keys:
                    keys.append(seq)
                    pcfg.shortcuts[self.action_id] = keys
                self._rebuild_pills()
                self.shortcut_changed.emit()
            else:
                self._rebuild_pills()

        edit.editingFinished.connect(on_finished)

    def _remove_shortcut(self, key_seq: str):
        keys = self.effective_keys()
        if key_seq in keys:
            keys.remove(key_seq)
            pcfg.shortcuts[self.action_id] = keys
        self._rebuild_pills()
        self.shortcut_changed.emit()

    def _clear(self):
        pcfg.shortcuts[self.action_id] = []
        self._rebuild_pills()
        self.shortcut_changed.emit()

    def _reset(self):
        defaults = list(DEFAULT_SHORTCUTS.get(self.action_id, []))
        if defaults:
            pcfg.shortcuts[self.action_id] = defaults
        elif self.action_id in pcfg.shortcuts:
            del pcfg.shortcuts[self.action_id]
        self._rebuild_pills()
        self.shortcut_changed.emit()

    def refresh(self, conflict_keys=None):
        self._conflict_keys = conflict_keys or set()
        self._rebuild_pills()


class ShortcutEditor(QWidget):
    """Grouped shortcut rows; each row records new keys via QKeySequenceEdit."""

    shortcut_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: Dict[str, _ShortcutRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for group_name, action_ids in _SHORTCUT_GROUPS:
            group_box = PanelGroupBox(self.tr(group_name))
            group_layout = QVBoxLayout(group_box)
            group_layout.setSpacing(0)
            for action_id in action_ids:
                row = _ShortcutRow(action_id)
                row.shortcut_changed.connect(self._on_row_changed)
                self._rows[action_id] = row
                group_layout.addWidget(row)
            layout.addWidget(group_box)

        layout.addStretch()
        self.refresh()

    def _on_row_changed(self):
        self.refresh()
        self.shortcut_changed.emit()

    def _compute_conflicts(self) -> set:
        return find_conflict_keys(
            {aid: row.effective_keys() for aid, row in self._rows.items()}
        )

    def refresh(self):
        conflicts = self._compute_conflicts()
        for row in self._rows.values():
            row.refresh(conflicts)

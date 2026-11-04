"""Propagate a drag-and-drop insertion out of a text editor.

An editor only records and forwards its edits while it holds focus, which a
paste always does and a drop never does: the drag starts somewhere else, so
the text lands without the editor ever being clicked. The paired text block
is then left showing the old text until the next focused edit.

A drop is also not one clean insertion. Moving text inside the same editor
removes it at the source and inserts it at the target, so the running change
counters cannot describe it. The whole text is compared instead, before and
after, and the single changed range is forwarded.
"""

from typing import Callable, Tuple

from qtpy.QtGui import QTextCursor


def utf16_length(text: str) -> int:
    """Return a string's length in the UTF-16 units Qt counts positions in.

    >>> utf16_length('abc')
    3
    >>> utf16_length('\U0001f600')
    2
    """
    return len(text.encode('utf-16-le')) // 2


def changed_range(before: str, after: str) -> Tuple[int, int, int]:
    """Return the single changed range as ``(position, removed, added)``.

    Lengths are in UTF-16 units, matching Qt document positions. A
    supplementary character is one Python character, so trimming the common
    prefix and suffix never splits a surrogate pair.

    >>> changed_range('abcd', 'abXYcd')
    (2, 0, 2)
    >>> changed_range('abcd', 'ad')
    (1, 2, 0)
    """
    prefix = 0
    limit = min(len(before), len(after))
    while prefix < limit and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    limit -= prefix
    while (
        suffix < limit
        and before[len(before) - 1 - suffix] == after[len(after) - 1 - suffix]
    ):
        suffix += 1
    return (
        utf16_length(before[:prefix]),
        utf16_length(before[prefix:len(before) - suffix]),
        utf16_length(after[prefix:len(after) - suffix]),
    )


def propagate_drop(editor, insert: Callable[[], None]) -> None:
    """Run a drop insertion and forward what it changed.

    ``in_acts`` mutes the editor's own change handler for the duration, the
    way the context menu does for its actions, so the drop is reported once
    here rather than twice by two disagreeing paths.
    """
    before = editor.toPlainText()
    was_in_acts = editor.in_acts
    editor.in_acts = True
    try:
        insert()
    finally:
        editor.in_acts = was_in_acts

    after = editor.toPlainText()
    if after == before:
        return

    position, removed, added = changed_range(before, after)
    added_text = ''
    if added:
        # Read the inserted text back from the document so paragraph breaks
        # arrive as the separators the paired document expects.
        cursor = QTextCursor(editor.document())
        cursor.setPosition(position)
        cursor.setPosition(
            position + added, QTextCursor.MoveMode.KeepAnchor
        )
        added_text = cursor.selectedText()

    editor.propagate_user_edited.emit(position, removed, added_text, False)

    undo_steps = editor.document().availableUndoSteps()
    new_steps = undo_steps - editor.old_undo_steps
    if new_steps > 0:
        editor.old_undo_steps = undo_steps
        editor.push_undo_stack.emit(new_steps)

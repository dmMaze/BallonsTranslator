"""Final text-document formatting at page and project boundaries."""

import re
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from qtpy import QT6
from qtpy.QtCore import QThread, Signal
from qtpy.QtGui import (
    QColor,
    QFont,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)

from ballontranslator.utils.config import AutoTateChuYokoConfig
from ballontranslator.utils.fontformat import FontFormat, font_weight_to_qt
from ballontranslator.utils.message import create_error_dialog
from ballontranslator.utils.textblock import TextBlock
from .font_family import qfont_with_family
from .annotations import (
    apply_auto_text_combine_upright,
    apply_letter_spacing,
    apply_line_spacing,
    load_rich_text_html,
    to_rich_text_html,
)


def _load_text_block_document(block: TextBlock) -> QTextDocument:
    """Build a formatting document without constructing a scene item."""
    document = QTextDocument()
    font_format = block.fontformat
    font = qfont_with_family(
        document.defaultFont(),
        font_format.font_family,
    )
    font.setPointSizeF(font_format.size_pt)
    font.setWeight(QFont.Weight(font_weight_to_qt(
        font_format.font_weight,
        qt6=QT6,
    )))
    font.setItalic(font_format.italic)
    font.setUnderline(font_format.underline)
    document.setDefaultFont(font)

    if block.rich_text:
        load_rich_text_html(
            document,
            block.rich_text,
            letter_spacing_fallback=font_format.letter_spacing,
            vertical=font_format.vertical,
        )
        return document

    document.setPlainText(block.translation)
    cursor = QTextCursor(document)
    cursor.select(QTextCursor.SelectionType.Document)
    char_format = QTextCharFormat()
    char_format.setFont(font)
    char_format.setForeground(QColor(*font_format.foreground_color()))
    cursor.mergeCharFormat(char_format)
    cursor.mergeBlockCharFormat(char_format)
    apply_letter_spacing(
        cursor,
        font_format.letter_spacing,
        vertical=font_format.vertical,
    )
    apply_line_spacing(
        cursor,
        font_format.line_spacing,
        font_format.line_spacing_type,
    )
    return document


_ELLIPSIS_RUN = re.compile('\u2026{2,}')


def collapse_ellipsis_runs(text: str) -> str:
    """Collapse a run of U+2026 into a single ellipsis mark.

    Japanese sources frequently use one ``…`` while a translator emits ``……``;
    the source's single mark is what the reader expects in the Chinese text.

    >>> collapse_ellipsis_runs('等等……')
    '等等…'
    >>> collapse_ellipsis_runs('...')   # ASCII dots are left alone
    '...'
    """
    return _ELLIPSIS_RUN.sub('\u2026', text or '')


def normalize_translation_punctuation(blocks: Sequence[TextBlock]) -> int:
    """Apply the ellipsis rule to finished translations and their rich text.

    Return the number of blocks whose stored text changed.

    >>> block = TextBlock(translation='等等……')
    >>> normalize_translation_punctuation([block])
    1
    >>> block.translation
    '等等…'
    """
    changed = 0
    for block in blocks:
        translation = collapse_ellipsis_runs(block.translation)
        rich_text = block.rich_text
        if rich_text:
            rich_text = collapse_ellipsis_runs(rich_text)
        if (
            translation == (block.translation or '')
            and rich_text == block.rich_text
        ):
            continue
        block.translation = translation
        if block.rich_text:
            block.rich_text = rich_text
        changed += 1
    return changed


def apply_layout_overrides(
    blocks: Sequence[TextBlock],
    global_format: FontFormat,
    *,
    roman_word_rotation: bool = False,
    vertical_alignment: bool = False,
) -> int:
    """Copy the settled-layout options onto blocks before they are laid out.

    The pipeline normally leaves per-property state alone unless the matching
    ``let_*`` flag asks for the global setting; these two options follow the
    same rule so a batch RUN produces the placement a manual change would.
    Return the number of blocks whose stored format changed.

    >>> from ballontranslator.utils.fontformat import FontFormat
    >>> block = TextBlock(translation="master")
    >>> global_format = FontFormat()
    >>> global_format.roman_word_rotation = True
    >>> apply_layout_overrides([block], global_format, roman_word_rotation=True)
    1
    >>> block.fontformat.roman_word_rotation
    True
    """
    changed = 0
    for block in blocks:
        block_format = block.fontformat
        updated = False
        if (
            roman_word_rotation
            and block_format.roman_word_rotation
            != global_format.roman_word_rotation
        ):
            block_format.roman_word_rotation = global_format.roman_word_rotation
            updated = True
        if (
            vertical_alignment
            and block_format.vertical_alignment
            != global_format.vertical_alignment
        ):
            block_format.vertical_alignment = global_format.vertical_alignment
            updated = True
        changed += int(updated)
    return changed


def apply_auto_tate_chu_yoko(
    blocks: Sequence[TextBlock],
    settings: AutoTateChuYokoConfig,
) -> int:
    """Replace automatic Tate-chu-yoko on finalized text blocks.

    Horizontal blocks have existing Tate-chu-yoko removed but receive no new
    runs. Return the number of blocks whose persisted rich text changed.

    >>> settings = AutoTateChuYokoConfig(enabled=True, max_length=2)
    >>> block = TextBlock(translation='12')
    >>> block.vertical = True
    >>> apply_auto_tate_chu_yoko([block], settings)
    1
    """
    if not settings.enabled:
        return 0
    allowed_characters = settings.allowed_characters()
    no_characters: frozenset[str] = frozenset()
    changed = 0
    for block in blocks:
        if not block.rich_text and not block.translation:
            continue
        if (
            not block.rich_text
            and (
                not block.vertical
                or not allowed_characters
                or not any(
                    character in allowed_characters
                    for character in block.translation
                )
            )
        ):
            continue
        document = _load_text_block_document(block)
        block_allowed_characters = (
            allowed_characters if block.vertical else no_characters
        )
        if not apply_auto_text_combine_upright(
            document,
            block_allowed_characters,
            settings.max_length,
        ):
            continue
        block.rich_text = to_rich_text_html(
            document,
            line_spacing_fallback=block.fontformat.line_spacing,
            line_spacing_type_fallback=block.fontformat.line_spacing_type,
        )
        changed += 1
    return changed


class AutoTateChuYokoThread(QThread):
    """Apply automatic Tate-chu-yoko to project documents off the UI thread.

    >>> issubclass(AutoTateChuYokoThread, QThread)
    True
    """

    progress_changed = Signal(int, str)
    processing_finished = Signal(int, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pages: List[Tuple[str, List[TextBlock]]] = []
        self._settings = AutoTateChuYokoConfig()
        self._stop_event = threading.Event()

    def start_processing(
        self,
        pages: Dict[str, List[TextBlock]],
        settings: AutoTateChuYokoConfig,
    ) -> bool:
        if self.isRunning():
            return False
        self._pages = list(pages.items())
        self._settings = settings.copy()
        self._stop_event.clear()
        self.start()
        return True

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        changed_blocks: List[TextBlock] = []
        error: Optional[Exception] = None
        try:
            total_pages = len(self._pages)
            for page_index, (page_name, blocks) in enumerate(
                self._pages,
                start=1,
            ):
                for block in blocks:
                    if self._stop_event.is_set():
                        break
                    if apply_auto_tate_chu_yoko(
                        (block,),
                        self._settings,
                    ):
                        changed_blocks.append(block)
                if self._stop_event.is_set():
                    break
                self.progress_changed.emit(
                    round(page_index / total_pages * 100),
                    page_name,
                )
        except Exception as exception:
            error = exception

        self.processing_finished.emit(
            len(changed_blocks),
            tuple(changed_blocks),
        )
        if error is not None:
            create_error_dialog(
                error,
                self.tr('Failed to apply automatic Tate-chu-yoko.'),
                'AutoTateChuYokoApplyFailed',
            )

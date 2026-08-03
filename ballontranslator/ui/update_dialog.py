import html
import math
import re
from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

HTML_IMAGE_PATTERN = re.compile(r'<img\b[^>]*/?>', re.IGNORECASE | re.DOTALL)
MARKDOWN_IMAGE_PATTERN = re.compile(
    r'!\[[^\]]*\]\([^)]*\)',
    re.IGNORECASE,
)
MARKDOWN_UNORDERED_LIST_PATTERN = re.compile(r'^(?P<marker>\s*[-+*])\s+', re.MULTILINE)
MARKDOWN_HEADING_PATTERN = re.compile(
    r'^\s{0,3}(?P<marks>#{1,6})\s+(?P<title>.*?)\s*#*\s*$',
    re.MULTILINE,
)
RELEASE_WINDOW_WIDTH = 480
RELEASE_LIST_INDENT_WIDTH = 20
RELEASE_SURFACE_HORIZONTAL_MARGIN = 24
RELEASE_NOTES_HORIZONTAL_MARGIN = 14


def strip_release_images(markdown: str) -> str:
    r"""Remove image markup while preserving the remaining release Markdown.

    >>> strip_release_images('<img src="https://example.com/a.gif" />\n\n- Fixed UI')
    '- Fixed UI'
    """

    markdown = HTML_IMAGE_PATTERN.sub('', markdown)
    return MARKDOWN_IMAGE_PATTERN.sub('', markdown).strip()


def select_release_note_section(markdown: str, display_language: str) -> str:
    r"""Select localized release notes and remove all Markdown headings.

    >>> notes = '## Changelog\n- Fixed UI\n## 更新说明\n- 修复界面'
    >>> select_release_note_section(notes, 'zh_CN')
    '- 修复界面'
    >>> select_release_note_section('# Notes\n- Fixed UI', 'en_US')
    '- Fixed UI'
    """

    lines = (markdown or '').splitlines()
    headings = []
    localized_headings = {}
    for index, line in enumerate(lines):
        match = MARKDOWN_HEADING_PATTERN.match(line)
        if match is None:
            continue
        title = match.group('title').strip()
        heading = (index, len(match.group('marks')), title)
        headings.append(heading)
        if title.casefold() == 'changelog':
            localized_headings.setdefault('en', heading)
        elif title == '更新说明':
            localized_headings.setdefault('zh', heading)

    normalized_language = (display_language or '').replace('_', '-').casefold()
    requested_language = 'zh' if normalized_language.startswith('zh-cn') else 'en'
    selected_heading = localized_headings.get(requested_language)

    if selected_heading is None:
        if localized_headings:
            return ''
        selected_lines = lines
    else:
        start_index, start_level, _title = selected_heading
        end_index = len(lines)
        for heading_index, heading_level, heading_title in headings:
            is_localized_boundary = (
                heading_title.casefold() == 'changelog'
                or heading_title == '更新说明'
            )
            if heading_index > start_index and (
                is_localized_boundary or heading_level <= start_level
            ):
                end_index = heading_index
                break
        selected_lines = lines[start_index + 1:end_index]

    selected_markdown = '\n'.join(selected_lines)
    return MARKDOWN_HEADING_PATTERN.sub('', selected_markdown).strip()


def format_release_markdown(markdown: str, display_language: str = '') -> str:
    """Prepare image-free Markdown with a wider bullet-to-text gap.

    >>> format_release_markdown('* Fixed UI')
    '* \xa0Fixed UI'
    """

    markdown = select_release_note_section(markdown, display_language)
    markdown = strip_release_images(markdown)
    return MARKDOWN_UNORDERED_LIST_PATTERN.sub(
        lambda match: match.group('marker') + ' \u00a0',
        markdown,
    )


def simplified_release_date(published_at: str) -> str:
    """Return the date portion of a GitHub release timestamp.

    >>> simplified_release_date('2026-07-03T06:45:13Z')
    '2026-07-03'
    """

    match = re.match(r'^(\d{4}-\d{2}-\d{2})', published_at or '')
    return match.group(1) if match else (published_at or '')


class UpdateReleaseDialog(QDialog):
    """Present update metadata and Markdown release notes in the app theme.

    >>> UpdateReleaseDialog.__name__
    'UpdateReleaseDialog'
    """

    def __init__(
        self,
        result,
        parent: Optional[QWidget] = None,
        display_language: str = '',
        allow_update: bool = True,
    ):
        super().__init__(parent)
        self.setObjectName('UpdateReleaseDialog')
        self.setWindowTitle(self.tr('Update Available'))
        self.setFixedWidth(RELEASE_WINDOW_WIDTH)
        window_type = getattr(Qt, 'WindowType', Qt)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        self.setWindowFlags(window_type.Dialog | window_type.FramelessWindowHint)
        self.setAttribute(widget_attribute.WA_TranslucentBackground)

        release_info = result.release_info
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        surface = QFrame(self)
        surface.setObjectName('UpdateReleaseSurface')
        outer_layout.addWidget(surface)

        layout = QVBoxLayout(surface)
        layout.setContentsMargins(
            RELEASE_SURFACE_HORIZONTAL_MARGIN,
            20,
            RELEASE_SURFACE_HORIZONTAL_MARGIN,
            18,
        )
        layout.setSpacing(12)

        release_markdown = format_release_markdown(
            release_info.body or '',
            display_language,
        )

        title = QLabel(self.tr('A new version is available'), surface)
        title.setObjectName('UpdateReleaseTitle')
        layout.addWidget(title)

        version_url = html.escape(release_info.html_url or '', quote=True)
        current_version = html.escape(result.current_version)
        latest_version = html.escape(result.latest_version)
        version = QLabel(surface)
        version.setObjectName('UpdateReleaseVersion')
        if version_url:
            latest_text = (
                f'<a style="color: rgb(30, 147, 229); text-decoration: none;" '
                f'href="{version_url}">{latest_version}</a>'
            )
        else:
            latest_text = latest_version
        release_date = html.escape(simplified_release_date(release_info.published_at))
        if release_date:
            latest_text = '{version} - {date}'.format(
                version=latest_text,
                date=release_date,
            )
        version.setText(
            current_version + '&nbsp;&nbsp;→&nbsp;&nbsp;' + latest_text
        )
        version.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        version.setOpenExternalLinks(True)
        version.setWordWrap(True)
        layout.addWidget(version)

        if release_info.name and release_info.name not in {
            release_info.tag_name,
            result.latest_version,
        }:
            metadata_label = QLabel(release_info.name, surface)
            metadata_label.setObjectName('UpdateReleaseMetadata')
            metadata_label.setTextFormat(Qt.TextFormat.PlainText)
            metadata_label.setWordWrap(True)
            layout.addWidget(metadata_label)

        normalized_language = (display_language or '').replace('_', '-').casefold()
        sponsor_url = (
            'https://afdian.com/a/dmMaze'
            if normalized_language == 'zh-cn'
            else 'https://patreon.com/dreMaze'
        )
        sponsor_link_text = self.tr('sponsoring this project')
        sponsor_link = (
            f'<a style="color: rgb(30, 147, 229); text-decoration: none;" '
            f'href="{sponsor_url}">{sponsor_link_text}</a>'
        )
        sponsor_message = QLabel(surface)
        sponsor_message.setObjectName('UpdateReleaseSponsorMessage')
        sponsor_message.setText(
            self.tr('Consider %1 if it has been helpful to you.').replace(
                '%1', sponsor_link
            )
        )
        sponsor_message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        sponsor_message.setOpenExternalLinks(True)
        sponsor_message.setWordWrap(True)
        layout.addWidget(sponsor_message)

        notes_section = QVBoxLayout()
        notes_section.setContentsMargins(0, 0, 0, 0)
        notes_section.setSpacing(4)

        notes_heading = QLabel(self.tr("What's new"), surface)
        notes_heading.setObjectName('UpdateReleaseNotesHeading')
        notes_section.addWidget(notes_heading)

        notes_frame = QFrame(surface)
        notes_frame.setObjectName('UpdateReleaseNotesCard')
        notes_layout = QVBoxLayout(notes_frame)
        notes_layout.setContentsMargins(
            RELEASE_NOTES_HORIZONTAL_MARGIN,
            12,
            RELEASE_NOTES_HORIZONTAL_MARGIN,
            12,
        )
        notes_layout.setSpacing(0)

        self.release_notes = QTextBrowser(notes_frame)
        self.release_notes.setObjectName('UpdateReleaseNotes')
        self.release_notes.setOpenExternalLinks(True)
        scroll_bar_off = getattr(getattr(Qt, 'ScrollBarPolicy', Qt), 'ScrollBarAlwaysOff')
        self.release_notes.setHorizontalScrollBarPolicy(scroll_bar_off)
        self.release_notes.setVerticalScrollBarPolicy(scroll_bar_off)
        self.release_notes.document().setDocumentMargin(0)
        self.release_notes.document().setIndentWidth(RELEASE_LIST_INDENT_WIDTH)
        self.release_notes.document().setDefaultStyleSheet(
            'a { color: rgb(30, 147, 229); text-decoration: none; }'
        )
        self.release_notes.document().documentLayout().documentSizeChanged.connect(
            self._resize_release_notes
        )
        self.release_notes.setMarkdown(release_markdown or self.tr('No release notes.'))
        notes_text_width = (
            RELEASE_WINDOW_WIDTH
            - RELEASE_SURFACE_HORIZONTAL_MARGIN * 2
            - RELEASE_NOTES_HORIZONTAL_MARGIN * 2
        )
        self.release_notes.document().setTextWidth(notes_text_width)
        self._resize_release_notes()
        notes_layout.addWidget(self.release_notes)
        notes_section.addWidget(notes_frame)
        layout.addLayout(notes_section)

        if allow_update:
            restart_notice = QLabel(
                self.tr('Updating will restart the app.'),
                surface,
            )
            restart_notice.setObjectName('UpdateReleaseRestartNotice')
            restart_notice.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            restart_notice.setWordWrap(True)
            layout.addWidget(restart_notice)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        button_layout.addStretch()
        if allow_update:
            cancel_btn = QPushButton(self.tr('Cancel'), surface)
            cancel_btn.setObjectName('UpdateDialogCancelButton')
            cancel_btn.clicked.connect(self.reject)
            button_layout.addWidget(cancel_btn)
        update_btn = QPushButton(
            self.tr('Update') if allow_update else self.tr('Close'),
            surface,
        )
        update_btn.setObjectName('UpdateDialogPrimaryButton')
        update_btn.setDefault(True)
        update_btn.clicked.connect(self.accept)
        button_layout.addWidget(update_btn)
        layout.addLayout(button_layout)

    def _resize_release_notes(self, _document_size=None) -> None:
        document_height = math.ceil(self.release_notes.document().size().height())
        frame_height = self.release_notes.frameWidth() * 2
        self.release_notes.setFixedHeight(max(document_height + frame_height, 24))

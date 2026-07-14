import copy
import math
import os, json, shutil, re, docx, docx2txt, piexif, cv2
from docx.shared import Inches
from docx import Document
import piexif.helper
import numpy as np
import os.path as osp
from typing import Tuple, Union, List, Dict
from PIL import Image
from qtpy.QtGui import QFont, QTextCharFormat, QTextCursor, QTextDocument

from .logger import logger as LOGGER
from .io_utils import find_all_imgs, imread, imwrite, NumpyEncoder
from .textblock import TextBlock, FontFormat
from .fontformat import normalize_text_transform, px2pt
from .config import pcfg, RunStatus
from . import shared

class ProjectDirNotExistException(Exception):
    pass

class ProjectLoadFailureException(Exception):
    pass

class ProjectNotSupportedException(Exception):
    pass

class ImgnameNotInProjectException(Exception):
    pass


TEXT_TRANSFORM_SCHEMA_VERSION = 1
_MISSING = object()
_LEGACY_STRETCH_PATTERN = re.compile(
    r'<!--\s*ballontranslator-logical-stretch-v1:(.*?)\s*-->', re.DOTALL
)


class TextTransformPayloadError(ValueError):
    """Base class for project text-transform payload failures."""


class UnsupportedTextTransformVersionError(TextTransformPayloadError):
    pass


class AmbiguousLegacyTextTransformError(TextTransformPayloadError):
    pass


class InvalidTextTransformPayloadError(TextTransformPayloadError):
    pass


def _payload_version(value, location: str, *, missing=0) -> int:
    if value is _MISSING:
        return missing
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTextTransformPayloadError(
            f"{location} must be an integer schema version"
        )
    value = float(value)
    if not math.isfinite(value) or not value.is_integer() or value < 0:
        raise InvalidTextTransformPayloadError(
            f"{location} must be an integer schema version"
        )
    return int(value)


def _payload_number(value, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTextTransformPayloadError(
            f"{location} must be a finite number"
        )
    value = float(value)
    if not math.isfinite(value):
        raise InvalidTextTransformPayloadError(
            f"{location} must be a finite number"
        )
    return value


def _block_transform_values(block: dict, location: str, migration_warnings: List[str]):
    fontformat = block.get('fontformat', {})
    if fontformat is None:
        fontformat = {}
    if not isinstance(fontformat, dict):
        raise InvalidTextTransformPayloadError(f"{location}.fontformat must be an object")

    sources = {
        'horizontal_scale': (
            ('fontformat.horizontal_scale', fontformat.get('horizontal_scale', _MISSING)),
            ('horizontal_scale', block.get('horizontal_scale', _MISSING)),
        ),
        'vertical_scale': (
            ('fontformat.vertical_scale', fontformat.get('vertical_scale', _MISSING)),
            ('vertical_scale', block.get('vertical_scale', _MISSING)),
        ),
        'slant_angle': (
            ('fontformat.slant_angle', fontformat.get('slant_angle', _MISSING)),
            ('fontformat.italic_angle', fontformat.get('italic_angle', _MISSING)),
            ('italic_angle', block.get('italic_angle', _MISSING)),
        ),
    }
    defaults = {
        'horizontal_scale': 1.0,
        'vertical_scale': 1.0,
        'slant_angle': 0.0,
    }
    raw = {}
    for target, candidates in sources.items():
        present = []
        for source, value in candidates:
            if value is not _MISSING:
                present.append((source, _payload_number(value, f"{location}.{source}")))
        if present:
            first = present[0][1]
            if any(value != first for _, value in present[1:]):
                names = ', '.join(source for source, _ in present)
                raise InvalidTextTransformPayloadError(
                    f"{location} contains conflicting aliases: {names}"
                )
            raw[target] = first
        else:
            raw[target] = defaults[target]

    normalized = normalize_text_transform(
        raw['horizontal_scale'], raw['vertical_scale'], raw['slant_angle']
    )
    for name, value, canonical in zip(
        ('horizontal_scale', 'vertical_scale', 'slant_angle'), raw.values(), normalized
    ):
        if value < (0.1 if name != 'slant_angle' else -45.0) or value > (
            4.0 if name != 'slant_angle' else 45.0
        ):
            migration_warnings.append(
                f"{location}.{name} was clamped from {value} to {canonical}"
            )
    return normalized, fontformat


def _resolved_font(char_format: QTextCharFormat, default_font: QFont) -> QFont:
    return char_format.font().resolve(default_font)


def _font_signature(font: QFont):
    return (
        round(font.pointSizeF(), 6),
        font.stretch(),
        font.family(),
        font.weight(),
        font.italic(),
        font.underline(),
        font.overline(),
        font.strikeOut(),
    )


def _char_signature(char_format: QTextCharFormat, default_font: QFont):
    foreground = char_format.foreground().color()
    return (
        _font_signature(_resolved_font(char_format, default_font)),
        foreground.rgba(),
        char_format.fontLetterSpacingType(),
        round(char_format.fontLetterSpacing(), 6),
    )


def _block_format_signature(block):
    block_format = block.blockFormat()
    return (
        int(block_format.alignment()),
        block_format.indent(),
        round(block_format.leftMargin(), 6),
        round(block_format.rightMargin(), 6),
        round(block_format.topMargin(), 6),
        round(block_format.bottomMargin(), 6),
        round(block_format.textIndent(), 6),
        round(block_format.lineHeight(), 6),
        block_format.lineHeightType(),
        block_format.nonBreakableLines(),
    )


def _document_signature(document: QTextDocument):
    default_font = document.defaultFont()
    blocks = []
    block = document.firstBlock()
    while block.isValid():
        fragments = []
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                fragments.append(
                    (
                        fragment.position(),
                        fragment.length(),
                        _char_signature(fragment.charFormat(), default_font),
                    )
                )
            iterator += 1
        blocks.append(
            (
                block.position(),
                block.length(),
                _block_format_signature(block),
                _char_signature(block.charFormat(), default_font),
                tuple(fragments),
            )
        )
        block = block.next()
    return document.toPlainText(), _font_signature(default_font), tuple(blocks)


def _set_document_font_geometry(
    document: QTextDocument, point_size_factor: float, stretch: int
) -> None:
    old_default = document.defaultFont()
    new_default = QFont(old_default)
    new_default.setPointSizeF(old_default.pointSizeF() * point_size_factor)
    new_default.setStretch(stretch)
    document.setDefaultFont(new_default)

    block = document.firstBlock()
    while block.isValid():
        ranges = []
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                resolved = _resolved_font(fragment.charFormat(), old_default)
                ranges.append((fragment.position(), fragment.length(), resolved.pointSizeF()))
            iterator += 1

        cursor = QTextCursor(document)
        for position, length, point_size in ranges:
            cursor.setPosition(position)
            cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
            delta = QTextCharFormat()
            delta.setFontPointSize(point_size * point_size_factor)
            delta.setFontStretch(stretch)
            cursor.mergeCharFormat(delta)

        block_cursor = QTextCursor(block)
        block_font = _resolved_font(block.charFormat(), old_default)
        block_delta = QTextCharFormat()
        block_delta.setFontPointSize(block_font.pointSizeF() * point_size_factor)
        block_delta.setFontStretch(stretch)
        block_cursor.mergeBlockCharFormat(block_delta)
        block = block.next()


def _legacy_default_font(fontformat: dict, vertical_scale: float, stretch: int) -> QFont:
    try:
        point_size = px2pt(float(fontformat.get('font_size', 24)))
    except (TypeError, ValueError) as error:
        raise InvalidTextTransformPayloadError(
            "legacy fontformat.font_size must be numeric"
        ) from error
    if not math.isfinite(point_size) or point_size <= 0:
        raise InvalidTextTransformPayloadError(
            "legacy fontformat.font_size must be a positive finite number"
        )
    font = QFont(str(fontformat.get('font_family') or ''))
    font.setPointSizeF(point_size * vertical_scale)
    font.setStretch(stretch)
    font.setBold(bool(fontformat.get('bold', False)))
    font.setItalic(bool(fontformat.get('italic', False)))
    font.setUnderline(bool(fontformat.get('underline', False)))
    return font


def _migrate_effective_legacy_html(
    html: str,
    fontformat: dict,
    horizontal_scale: float,
    vertical_scale: float,
    location: str,
) -> str:
    if horizontal_scale == 1.0 and vertical_scale == 1.0:
        return html
    if not html:
        raise AmbiguousLegacyTextTransformError(
            f"{location}.rich_text is empty and cannot prove whether the "
            "non-neutral failed transform was baked"
        )

    legacy_stretch = max(1, int(round(horizontal_scale / vertical_scale * 100)))
    metadata_matches = list(_LEGACY_STRETCH_PATTERN.finditer(html))
    if len(metadata_matches) != 1:
        raise AmbiguousLegacyTextTransformError(
            f"{location}.rich_text does not have exactly one failed stretch metadata record"
        )
    try:
        stretch_runs = json.loads(metadata_matches[-1].group(1))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AmbiguousLegacyTextTransformError(
            f"{location}.rich_text has invalid failed stretch metadata"
        ) from error
    if not isinstance(stretch_runs, list) or not stretch_runs:
        raise AmbiguousLegacyTextTransformError(
            f"{location}.rich_text has incomplete failed stretch metadata"
        )
    validated_runs = []
    for run in stretch_runs:
        ratio = run.get('ratio') if isinstance(run, dict) else None
        empty_block = isinstance(run, dict) and run.get('empty_block') is True
        if (
            not isinstance(run, dict)
            or isinstance(run.get('start'), bool)
            or not isinstance(run.get('start'), int)
            or isinstance(run.get('length'), bool)
            or not isinstance(run.get('length'), int)
            or isinstance(run.get('stretch'), bool)
            or not isinstance(run.get('stretch'), int)
            or run['start'] < 0
            or run['length'] < 0
            or (run['length'] == 0) != empty_block
            or run.get('stretch') != legacy_stretch
            or not (
                ratio is None
                or (
                    isinstance(ratio, list)
                    and len(ratio) == 2
                    and all(type(value) is int for value in ratio)
                    and ratio == [1, 1]
                )
            )
        ):
            raise AmbiguousLegacyTextTransformError(
                f"{location}.rich_text has an ambiguous failed stretch run"
            )
        validated_runs.append(
            (run['start'], run['length'], empty_block)
        )
    clean_html = _LEGACY_STRETCH_PATTERN.sub('', html)
    effective_document = QTextDocument()
    effective_document.setDefaultFont(
        _legacy_default_font(fontformat, vertical_scale, legacy_stretch)
    )
    effective_document.setHtml(clean_html)

    valid_character_positions = set()
    valid_empty_positions = set()
    block = effective_document.firstBlock()
    while block.isValid():
        if block.text() == '':
            valid_empty_positions.add(block.position())
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                valid_character_positions.update(
                    range(fragment.position(), fragment.position() + fragment.length())
                )
            iterator += 1
        block = block.next()

    covered_positions = set()
    covered_empty_positions = set()
    for start, length, empty_block in validated_runs:
        if empty_block:
            if start not in valid_empty_positions or start in covered_empty_positions:
                raise AmbiguousLegacyTextTransformError(
                    f"{location}.rich_text has a stale failed empty-block run"
                )
            covered_empty_positions.add(start)
            continue
        positions = set(range(start, start + length))
        if (
            not positions
            or not positions.issubset(valid_character_positions)
            or positions.intersection(covered_positions)
        ):
            raise AmbiguousLegacyTextTransformError(
                f"{location}.rich_text has a stale or overlapping failed stretch run"
            )
        covered_positions.update(positions)
    if (
        covered_positions != valid_character_positions
        or covered_empty_positions != valid_empty_positions
    ):
        raise AmbiguousLegacyTextTransformError(
            f"{location}.rich_text has incomplete failed stretch metadata"
        )

    signature = _document_signature(effective_document)
    for block_signature in signature[2]:
        block_char = block_signature[3]
        fragment_signatures = block_signature[4]
        fonts = [block_char[0], *(fragment[2][0] for fragment in fragment_signatures)]
        if any(font[1] != legacy_stretch for font in fonts):
            raise AmbiguousLegacyTextTransformError(
                f"{location}.rich_text does not have the exact failed stretch signature"
            )

    logical_document = QTextDocument()
    logical_document.setDefaultFont(effective_document.defaultFont())
    logical_document.setHtml(clean_html)
    _set_document_font_geometry(logical_document, 1.0 / vertical_scale, 100)
    logical_html = logical_document.toHtml()

    replay = QTextDocument()
    replay_default = QFont(logical_document.defaultFont())
    replay.setDefaultFont(replay_default)
    replay.setHtml(logical_html)
    _set_document_font_geometry(replay, vertical_scale, legacy_stretch)
    if _document_signature(replay) != signature:
        raise AmbiguousLegacyTextTransformError(
            f"{location}.rich_text cannot be reversed without losing formatting"
        )
    return logical_html


def migrate_text_transform_payload(proj_dict: dict):
    """Return a canonical schema-v1 copy without mutating the input payload."""
    if not isinstance(proj_dict, dict):
        raise InvalidTextTransformPayloadError("project payload must be an object")
    migrated = copy.deepcopy(proj_dict)
    root_version = _payload_version(
        migrated.get('text_transform_schema_version', _MISSING),
        'text_transform_schema_version',
    )
    if root_version > TEXT_TRANSFORM_SCHEMA_VERSION:
        raise UnsupportedTextTransformVersionError(
            f"unsupported text transform schema version {root_version}"
        )
    pages = migrated.get('pages')
    if not isinstance(pages, dict):
        raise InvalidTextTransformPayloadError("pages must be an object")

    # Reject every future marker before canonicalizing any block in our copy.
    for page_name, blocks in pages.items():
        if not isinstance(blocks, list):
            raise InvalidTextTransformPayloadError(f"pages.{page_name} must be a list")
        for index, block in enumerate(blocks):
            location = f"pages.{page_name}[{index}]"
            if not isinstance(block, dict):
                raise InvalidTextTransformPayloadError(f"{location} must be an object")
            marker_value = block.get('rich_text_transform_version', _MISSING)
            marker = _payload_version(
                marker_value,
                f"{location}.rich_text_transform_version",
            )
            if marker > 1:
                raise UnsupportedTextTransformVersionError(
                    f"unsupported rich-text transform version {marker} at {location}"
                )

    migration_warnings = []
    for page_name, blocks in pages.items():
        for index, block in enumerate(blocks):
            location = f"pages.{page_name}[{index}]"
            marker_value = block.get('rich_text_transform_version', _MISSING)
            marker = _payload_version(
                marker_value,
                f"{location}.rich_text_transform_version",
            )
            transform, fontformat = _block_transform_values(
                block, location, migration_warnings
            )
            horizontal_scale, vertical_scale, slant_angle = transform

            # Marker 1 is the failed branch's logical representation. An
            # explicit marker 0 always denotes its effective size/stretch HTML,
            # even if a partially written file already has a v1 root marker.
            # Only a missing marker under a v1 root denotes canonical HTML.
            explicit_effective_marker = marker_value is not _MISSING and marker == 0
            if marker == 0 and (root_version == 0 or explicit_effective_marker):
                old_html = block.get('rich_text', '')
                logical_html = _migrate_effective_legacy_html(
                    old_html,
                    fontformat,
                    horizontal_scale,
                    vertical_scale,
                    location,
                )
                if logical_html != old_html:
                    block['rich_text'] = logical_html
                    migration_warnings.append(
                        f"{location}.rich_text was restored from the failed effective format"
                    )

            canonical_fontformat = dict(fontformat)
            canonical_fontformat.pop('italic_angle', None)
            canonical_fontformat.update(
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                slant_angle=slant_angle,
            )
            block['fontformat'] = canonical_fontformat
            block.pop('horizontal_scale', None)
            block.pop('vertical_scale', None)
            block.pop('italic_angle', None)
            block.pop('rich_text_transform_version', None)

    migrated['text_transform_schema_version'] = TEXT_TRANSFORM_SCHEMA_VERSION
    return migrated, migration_warnings


def get_last_modified_file(file_prefix, exts, ext_fallback=None):
    '''
    get last modified file from files sharing same prefix
    '''
    latest_time = -1
    latest_f = None
    for ext in exts:
        tmp_p = file_prefix + ext
        if osp.exists(tmp_p) and osp.getmtime(tmp_p) > latest_time:
            latest_time = osp.getmtime(tmp_p)
            latest_f = tmp_p
    if latest_f is None:
        if ext_fallback is not None:
            latest_f = file_prefix + ext_fallback
        else:
            latest_f = file_prefix + exts[0]
    return latest_f


def write_jpg_metadata(imgpath: str, metadata="a metadata"):
    exif_dict = {"Exif":{piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(metadata, encoding='unicode')}}
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, imgpath)

def read_jpg_metadata(imgpath: str):
    exif_dict = piexif.load(imgpath)
    user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    bubdict = json.loads(user_comment)
    return bubdict

page_start_pattern = re.compile(r'^###\s+', re.MULTILINE)
text_blkid_start_pattern = re.compile(r'^\d+\.', re.MULTILINE)

def parse_txt_translation(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = f.read()
    page_start = None
    page_list = []
    for matched in page_start_pattern.finditer(content):
        start, end = matched.span()
        if page_start is not None:
            page_list.append({'page_content': content[page_start: start]})
        page_start = start
    if page_start is not None:
        page_list.append({'page_content': content[page_start:]})

    for page_dict in page_list:
        page_content = page_dict['page_content']
        page_dict['page_name'] = page_start_pattern.sub('', page_content.split('\n')[0]).strip()
        blkid_start = blkid_end = None
        blk_list = []
        for matched in text_blkid_start_pattern.finditer(page_content):
            start, end = matched.span()
            if blkid_start is not None:
                blk_list.append(page_content[blkid_end: start].strip())
            blkid_start = start
            blkid_end = end
        if blkid_start is not None:
            blk_list.append(page_content[blkid_end:].strip())
        page_dict['blk_list'] = blk_list

    return page_list


class TextBlkEncoder(NumpyEncoder):
    def default(self, obj):
        if isinstance(obj, TextBlock):
            return obj.to_dict()
        elif isinstance(obj, FontFormat):
            return vars(obj)
        return NumpyEncoder.default(self, obj)


class ProjImgTrans:

    def __init__(self, directory: str = None):
        self.type = 'imgtrans'
        self.directory: str = None
        self.pages: Dict[str, List[TextBlock]] = {}
        self._pagename2idx = {}
        self._idx2pagename = {}
        self._image_info = {}

        self._fuzzy_inpainted_list = None

        self.not_found_pages: Dict[str, List[TextBlock]] = {}
        self.new_pages: List[str] = []
        self.proj_path: str = None

        self.current_img: str = None
        self.img_array: np.ndarray = None
        self.mask_array: np.ndarray = None
        self.inpainted_array: np.ndarray = None
        self.text_transform_migration_warnings: List[str] = []
        if directory is not None:
            self.load(directory)

    def idx2pagename(self, idx: int) -> str:
        return self._idx2pagename[idx]

    def pagename2idx(self, pagename: str) -> int:
        if pagename in self.pages:
            return self._pagename2idx[pagename]
        return -1

    def proj_name(self) -> str:
        return self.type+'_'+osp.basename(self.directory)

    def load(self, directory: str, json_path: str = None) -> bool:
        target_directory = directory
        if json_path is None:
            target_proj_path = osp.join(
                target_directory,
                self.type + '_' + osp.basename(target_directory) + '.json',
            )
        else:
            target_proj_path = json_path
        new_proj = False
        candidate = ProjImgTrans()
        if not osp.exists(target_proj_path):
            new_proj = True
            candidate.directory = target_directory
            candidate.proj_path = target_proj_path
            candidate.new_project()
        else:
            try:
                with open(target_proj_path, 'r', encoding='utf8') as f:
                    proj_dict = json.loads(f.read())
            except Exception as e:
                raise ProjectLoadFailureException(str(e)) from e
            candidate.load_from_dict(
                proj_dict,
                directory=target_directory,
                proj_path=target_proj_path,
            )
        target_inpainted_dir = osp.join(target_directory, 'inpainted')
        target_mask_dir = osp.join(target_directory, 'mask')
        if not osp.exists(target_inpainted_dir):
            os.makedirs(target_inpainted_dir)
        if not osp.exists(target_mask_dir):
            os.makedirs(target_mask_dir)

        self._adopt_project_state(candidate)

        return new_proj

    def _adopt_project_state(self, candidate: 'ProjImgTrans') -> None:
        """Commit one fully constructed project state to this instance."""
        self.directory = candidate.directory
        self.proj_path = candidate.proj_path
        self.pages = candidate.pages
        self._pagename2idx = candidate._pagename2idx
        self._idx2pagename = candidate._idx2pagename
        self.not_found_pages = candidate.not_found_pages
        self.new_pages = candidate.new_pages
        self._image_info = candidate._image_info
        self.current_img = candidate.current_img
        self.img_array = candidate.img_array
        self.mask_array = candidate.mask_array
        self.inpainted_array = candidate.inpainted_array
        self._fuzzy_inpainted_list = candidate._fuzzy_inpainted_list
        self.text_transform_migration_warnings = (
            candidate.text_transform_migration_warnings
        )

    def mask_dir(self):
        return osp.join(self.directory, 'mask')

    def inpainted_dir(self):
        return osp.join(self.directory, 'inpainted')

    def result_dir(self):
        return osp.join(self.directory, 'result')

    def load_from_dict(
        self,
        proj_dict: dict,
        *,
        directory: str = None,
        proj_path: str = None,
    ):
        migrated, migration_warnings = migrate_text_transform_payload(proj_dict)
        page_dict = migrated['pages']
        load_directory = self.directory if directory is None else directory
        load_proj_path = self.proj_path if proj_path is None else proj_path

        try:
            found_pages = find_all_imgs(
                img_dir=load_directory, abs_path=False, sort=True
            )
            pages = {}
            pagename_to_idx = {}
            idx_to_pagename = {}
            not_found_pages = {}
            new_pages = []
            missing_names = list(page_dict.keys())
            for index, image_name in enumerate(found_pages):
                if image_name in page_dict:
                    pages[image_name] = [
                        TextBlock(**block) for block in page_dict[image_name]
                    ]
                    missing_names.remove(image_name)
                else:
                    pages[image_name] = []
                    new_pages.append(image_name)
                pagename_to_idx[image_name] = index
                idx_to_pagename[index] = image_name
            for image_name in missing_names:
                not_found_pages[image_name] = [
                    TextBlock(**block) for block in page_dict[image_name]
                ]
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectNotSupportedException(str(error)) from error

        image_info = copy.deepcopy(migrated.get('image_info', {}))
        if not isinstance(image_info, dict):
            raise ProjectNotSupportedException("image_info must be an object")
        for page_name, page_blocks in pages.items():
            if page_name not in image_info:
                image_info[page_name] = {}
            if not isinstance(image_info[page_name], dict):
                raise ProjectNotSupportedException(
                    f"image_info.{page_name} must be an object"
                )
            if 'finish_code' not in image_info[page_name]:
                has_empty_block = len(page_blocks) == 0 or any(
                    not block.text or len(block.text) == 0 for block in page_blocks
                )
                image_info[page_name]['finish_code'] = (
                    0 if has_empty_block else RunStatus.FIN_ALL
                )

        # Image IO is also performed on an isolated candidate. A failed current
        # image never leaves the existing project half replaced.
        candidate = ProjImgTrans()
        candidate.directory = load_directory
        candidate.proj_path = load_proj_path
        candidate.pages = pages
        candidate._pagename2idx = pagename_to_idx
        candidate._idx2pagename = idx_to_pagename
        candidate.not_found_pages = not_found_pages
        candidate.new_pages = new_pages
        candidate._image_info = image_info
        current_image = migrated.get('current_img')
        if current_image not in pages:
            current_image = idx_to_pagename.get(0)
        candidate.set_current_img(current_image)

        candidate.text_transform_migration_warnings = migration_warnings
        self._adopt_project_state(candidate)

    def get_page_progress(self, pagename: str):
        fin_code = self._image_info[pagename]['finish_code']
        return (fin_code & pcfg.module.finish_code) == pcfg.module.finish_code

    def set_page_progress(self, pagename, code):
        self._image_info[pagename]['finish_code'] = code 

    def update_page_progress(self, pagename, code):
        self._image_info[pagename]['finish_code'] |= code 

    def load_translation_from_txt(self, file_path: str):
        page_list = parse_txt_translation(file_path)
        missing_pages = []
        unmatched_pages = []
        unexpected_pages = []
        matched_pages = []
        for page_dict in page_list:
            page_name = page_dict['page_name']
            if page_name in self.pages:
                matched_pages.append(page_name)
            else:
                unexpected_pages.append(page_name)
                continue
            blklist = self.pages[page_name]
            n_blk = len(blklist)
            src_blk_list = page_dict['blk_list']
            n_src_blk = len(src_blk_list)
            if n_src_blk != n_blk:
                LOGGER.warning(f'Unmatched text blocks in {page_name}, number of text blocks in this page vs source file: {n_blk}-{n_src_blk}')
                unmatched_pages.append(page_name)
            for blkid in range(min(n_blk, n_src_blk)):
                blk = blklist[blkid]
                blk.rich_text = ''
                blk.translation = src_blk_list[blkid]

        matched_pages = set(matched_pages)
        if len(matched_pages) != self.num_pages:
            for page_name in self.pages:
                if page_name not in matched_pages:
                    missing_pages.append(page_name)
        
        all_matched = len(missing_pages) == 0 and len(unmatched_pages) == 0 and len(unexpected_pages) == 0
        return all_matched, {'missing_pages': missing_pages, 'unmatched_pages': unmatched_pages, 'unexpected_pages': unexpected_pages, 'matched_pages': matched_pages}

    def load_from_json(self, json_path: str):
        directory = osp.dirname(json_path)
        try:
            self.load(directory, json_path=json_path)
        except Exception as e:
            raise ProjectLoadFailureException(str(e)) from e

    def set_current_img(self, imgname: str):
        if imgname is not None:
            if imgname not in self.pages:
                raise ImgnameNotInProjectException
            self.current_img = imgname
            img_path = self.current_img_path()
            mask_path = self.get_mask_path(get_last_modified=True)
            self.img_array = imread(img_path)
            im_h, im_w = self.img_array.shape[:2]
            if osp.exists(mask_path):
                self.mask_array = imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                self.mask_array = np.zeros((im_h, im_w), dtype=np.uint8)
            self.inpainted_array = self.load_inpainted_by_imgname(imgname)
            if self.inpainted_array is None:
                self.inpainted_array = np.copy(self.img_array)
        else:
            self.current_img = None
            self.img_array = None
            self.mask_array = None
            self.inpainted_array = None

    def current_has_alpha(self):
        if self.current_img is None:
            return False
        return len(self.img_array.shape) and self.img_array.shape[-1] == 4

    def set_current_img_byidx(self, idx: int):
        num_pages = self.num_pages
        if idx < 0:
            idx = idx + self.num_pages
        if idx < 0 or idx > num_pages - 1:
            self.set_current_img(None)
        else:
            self.set_current_img(self.idx2pagename(idx))

    def get_blklist_byidx(self, idx: int) -> List[TextBlock]:
        return self.pages[self.idx2pagename(idx)]

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def current_idx(self) -> int:
        return self.pagename2idx(self.current_img)

    def new_project(self):
        if not osp.exists(self.directory):
            raise ProjectDirNotExistException
        self.set_current_img(None)
        imglist = find_all_imgs(self.directory, abs_path=False, sort=True)
        self.pages = {}
        self._pagename2idx = {}
        self._idx2pagename = {}
        self._image_info = {}
        for ii, imgname in enumerate(imglist):
            self.pages[imgname] = []
            self._pagename2idx[imgname] = ii
            self._idx2pagename[ii] = imgname
            self._image_info[imgname] = {'finish_code': 0}
        self.set_current_img_byidx(0)
        self.save()
        
    def save(self, keep_exist_as_backup=False):
        if not osp.exists(self.directory):
            raise ProjectDirNotExistException
        tmp_save_tgt = self.proj_path + '.tmp'
        try:
            with open(tmp_save_tgt, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.to_dict(), ensure_ascii=False, cls=TextBlkEncoder))
        except:
            raise Exception(f'Failed to write {self.to_dict()}')
        if osp.exists(self.proj_path) and keep_exist_as_backup:
            os.replace(self.proj_path, self.proj_path + '.backup')
            os.replace(tmp_save_tgt, self.proj_path)
        else:
            os.replace(tmp_save_tgt, self.proj_path)
        LOGGER.debug(f'project saved to {self.proj_path}')

    def to_dict(self) -> Dict:
        pages = self.pages.copy()
        pages.update(self.not_found_pages)        
        image_info = self._image_info.copy()
        return {
            'text_transform_schema_version': TEXT_TRANSFORM_SCHEMA_VERSION,
            'directory': self.directory,
            'pages': pages,
            'current_img': self.current_img,
            'image_info': image_info,
        }

    def read_img(self, imgname: str) -> np.ndarray:
        if imgname not in self.pages:
            raise ImgnameNotInProjectException
        img_path = osp.join(self.directory, imgname)
        img = imread(img_path)
        h, w = img.shape[:2]
        self._image_info[imgname].update({'width': w, 'height': h})
        return img

    def save_mask(self, img_name, mask: np.ndarray):
        imwrite(self.get_mask_path(img_name), mask, ext=pcfg.intermediate_imgsave_ext)

    def save_inpainted(self, img_name, inpainted: np.ndarray):
        imwrite(self.get_inpainted_path(img_name), inpainted, ext=pcfg.intermediate_imgsave_ext)

    def current_img_path(self) -> str:
        if self.current_img is None:
            return None
        return osp.join(self.directory, self.current_img)

    def get_mask_path(self, imgname: str = None, get_last_modified=False) -> str:
        if imgname is None:
            imgname = self.current_img

        fileprefix = osp.join(self.mask_dir(), osp.splitext(imgname)[0])
        if get_last_modified:
            p = get_last_modified_file(fileprefix, ['.jxl', '.png'], ext_fallback=pcfg.intermediate_imgsave_ext)
        else:
            p = fileprefix+pcfg.intermediate_imgsave_ext

        return p
    
    def load_mask_by_imgname(self, imgname: str) -> np.ndarray:
        mask = None
        mp = self.get_mask_path(imgname, get_last_modified=True)
        if osp.exists(mp):
            mask = imread(mp, cv2.IMREAD_GRAYSCALE)
        return mask

    def get_inpainted_path(self, imgname: str = None, get_last_modified=False) -> str:
        if imgname is None:
            imgname = self.current_img

        fileprefix = osp.join(self.inpainted_dir(), osp.splitext(imgname)[0])
        if get_last_modified:
            p = get_last_modified_file(fileprefix, ['.jxl', '.png'], ext_fallback=pcfg.intermediate_imgsave_ext)
        else:
            p = fileprefix+pcfg.intermediate_imgsave_ext

        if not osp.exists(p) and shared.FUZZY_MATCH_IMAGE_NAME:
            if self._fuzzy_inpainted_list is None:
                if osp.exists(self.inpainted_dir()):
                    self._fuzzy_inpainted_list = find_all_imgs(self.inpainted_dir(), sort=True)
                else:
                    self._fuzzy_inpainted_list = []
            pidx = self.pagename2idx(imgname)
            if pidx < len(self._fuzzy_inpainted_list):
                return osp.join(self.inpainted_dir(), self._fuzzy_inpainted_list[pidx])
        return p
    
    def load_inpainted_by_imgname(self, imgname: str, scale_to_src: bool = True) -> np.ndarray:
        inpainted = None
        mp = self.get_inpainted_path(imgname, get_last_modified=True)
        if mp is not None and osp.exists(mp):
            inpainted = imread(mp)
            if imgname == self.current_img and self.img_array is not None:
                h, w = self.img_array.shape[:2]
            else:
                i = Image.open(osp.join(self.directory, imgname))
                h, w = i.height, i.width
            ih, iw = inpainted.shape[:2]
            if ih != h or iw != w:
                inpainted = Image.fromarray(inpainted).resize((w, h), resample=Image.Resampling.LANCZOS)
                inpainted = np.array(inpainted)
        return inpainted

    def get_result_path(self, imgname: str) -> str:
        ext = '.png'
        if pcfg is not None:
            if pcfg.imgsave_ext not in {'.jpg', '.png', '.webp', '.jxl'}:
                LOGGER.warning('invalid image saving ext in config.json')
            else:
                ext = pcfg.imgsave_ext
        return osp.join(self.result_dir(), osp.splitext(imgname)[0]+ext)
        
    def backup(self):
        raise NotImplementedError

    @property
    def is_empty(self):
        return len(self.pages) == 0

    @property
    def is_all_pages_no_text(self):
        return all([len(blklist) == 0 for blklist in self.pages.values()])

    @property
    def img_valid(self):
        return self.img_array is not None
    
    @property
    def mask_valid(self):
        return self.mask_array is not None

    @property
    def inpainted_valid(self):
        return self.inpainted_array is not None

    def set_next_img(self):
        if self.current_img is not None:
            next_idx = (self.current_idx + 1) % self.num_pages
            self.set_current_img(self.idx2pagename(next_idx))

    def set_prev_img(self):
        if self.current_img is not None:
            next_idx = (self.current_idx - 1 + self.num_pages) % self.num_pages
            self.set_current_img(self.idx2pagename(next_idx))

    def current_block_list(self) -> List[TextBlock]:
        if self.current_img is not None:
            assert self.current_img in self.pages
            return self.pages[self.current_img]
        else:
            return None

    def doc_path(self) -> str:
        return os.path.join(self.directory, self.proj_name() + ".docx")

    def doc_exist(self) -> bool:
        return osp.exists(self.doc_path())

    def dump_doc(self, delete_tmp_folder=True, fin_page_signal=None):
        
        cuts_dir = os.path.join(self.directory, "bubcuts")
        if os.path.exists(cuts_dir):
            shutil.rmtree(cuts_dir)
        os.mkdir(cuts_dir)
        
        document = Document()
        style = document.styles['Normal']
        font = style.font
        target_font = 'Arial'
        font.name = target_font
        for pagename, blklist in self.pages.items():
            imgpath = os.path.join(self.directory, pagename)
            
            cuts_path_list, cut_width_list = gen_ballon_cuts(cuts_dir, imgpath, blklist)
            paragraph = document.add_paragraph(pagename)
            paragraph.style = document.styles['Normal']
            table = document.add_table(rows=len(cuts_path_list), cols=2, style='Table Grid')

            for index, (cut_path, width) in enumerate(zip(cuts_path_list, cut_width_list)):
                run = table.cell(index, 0).paragraphs[0].add_run()
                run.style.font.name = target_font
                blk: TextBlock = blklist[index]
                bubdict = vars(blk).copy()
                bubdict["imgkey"] = pagename
                bubdict["rich_text"] = ''
                bubdict["text"] = blk.get_text()
                write_jpg_metadata(cut_path, metadata=json.dumps(bubdict, ensure_ascii=False, cls=TextBlkEncoder))
                run.add_picture(cut_path, width=Inches(width/96 * 0.85))
                table.cell(index, 1).text = bubdict["translation"]

            document.add_page_break()
            
            if fin_page_signal is not None:
                fin_page_signal.emit()
                # time.sleep(1)

        doc_path = self.doc_path()
        document.save(doc_path)
        if delete_tmp_folder:
            shutil.rmtree(cuts_dir)

    def dump_txt_path(self, dump_target, suffix):
        save_path = osp.join(self.directory, self.proj_name() + f'_{dump_target}{suffix}')
        return save_path

    def dump_txt(self, dump_target: str, suffix='.txt'):
        save_path = self.dump_txt_path(dump_target, suffix=suffix)
        text_all = []
        assert dump_target in {'source', 'translation'}
        assert suffix in {'.txt', '.md'}
        for page_name, blk_list in self.pages.items():
            text_in_page = ['### ' + page_name]
            for ii, blk in enumerate(blk_list):
                if dump_target == 'translation':
                    text = blk.translation.strip()
                elif dump_target == 'source':
                    text = blk.get_text().strip()
                text_in_page.append(f'{ii + 1}. {text}')
            text_all.append('\n\n'.join(text_in_page))
        with open(save_path, 'w', encoding='utf8') as f:
            f.write('\n\n\n'.join(text_all))

    def load_doc(self, doc_path, delete_tmp_folder=True, fin_page_signal=None):
        tmp_bubble_folder = osp.join(self.directory, 'img_folder')
        os.makedirs(tmp_bubble_folder, exist_ok=True)
        docx2txt.process(doc_path, tmp_bubble_folder)

        doc = docx.Document(doc_path)
        body_xml_str = doc._body._element.xml

        pages = {}
        bub_index = 0
        for tbl in re.findall(r'<w:tbl>(.*?)</w:tbl>', body_xml_str, re.DOTALL):
            for tr in re.findall(r'<w:tr(.*?)>(.*?)</w:tr>', tbl, re.DOTALL):
                if re.findall(r'<pic:cNvPr id=\"(.*?)\" name=\"(.*?)\"(.*?)>', tr[1]):
                    bub_index += 1
                    translation = ""
                    for paragraph in re.findall(r'<w:p(.*?)>(.*?)</w:p>', tr[1], re.DOTALL):
                        for wt in re.findall(r'<w:t>(.*?)</w:t>', paragraph[1], re.DOTALL):
                            translation += wt
                        translation += "\n"
                    translation = translation[:-1]
                    if len(translation) != 0 and translation[0] == "\n":
                        translation = translation[1:]


                    bubpath = os.path.join(tmp_bubble_folder, "image"+str(bub_index))
                    if osp.exists(bubpath+'.jpg'):
                        bubpath = bubpath + '.jpg'
                    else:
                        bubpath = bubpath + '.jpeg'

                    meta_dict = read_jpg_metadata(bubpath)
                    meta_dict["translation"] = translation
                    imgkey = meta_dict.pop("imgkey")
                    if not imgkey in pages:
                        pages[imgkey] = []
                    pages[imgkey].append(TextBlock(**meta_dict))
                    
                    if fin_page_signal is not None:
                        fin_page_signal.emit()

        self.merge_from_proj_dict(pages)
        if delete_tmp_folder:
            shutil.rmtree(tmp_bubble_folder)

    def merge_from_proj_dict(self, tgt_dict: Dict) -> Dict:
        if self.pages is None:
            self.pages = {}
        src_dict = self.pages if self.pages is not None else {}
        key_lst = list(dict.fromkeys(list(src_dict.keys()) + list(tgt_dict.keys())))
        key_lst.sort()
        rst_dict = {}
        pagename2idx = {}
        idx2pagename = {}
        page_counter = 0
        for key in key_lst:
            if key in src_dict and not key in tgt_dict:
                rst_dict[key] = src_dict[key]
            else:
                rst_dict[key] = tgt_dict[key]
            pagename2idx[key] = page_counter
            idx2pagename[page_counter] = key
            page_counter += 1
        self.pages.clear()
        self.pages.update(rst_dict)
        self._pagename2idx = pagename2idx
        self._idx2pagename = idx2pagename        


def gen_ballon_cuts(cuts_dir: str, imgpath: str, blk_list: List[TextBlock], resize=True) -> Tuple[List[str], List[int]]:
    img = imread(imgpath)
    imgname = os.path.basename(imgpath)
    cuts_path_list = []
    cut_width_list = []
    for ii, blk in enumerate(blk_list):
        
        x, y, w, h = blk.bounding_rect()
        x, y = max(x, 0), max(y, 0)
        w = max(w, 1)
        h = max(h, 1)
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)

        cut_path = os.path.join(cuts_dir, f'{imgname}-{ii}.jpg')
        bub = img[y1:y2, x1:x2]
        max_width = 448

        if bub.shape[0] < 1 or bub.shape[1] < 1:
            emptyw = 60
            resized = np.full((emptyw, emptyw, 3), fill_value=0, dtype=np.uint8)
            width = emptyw
        else:
            # scale_percent = 60 # percent of original size
            scale_percent = min(1920 / img.shape[0], max_width / w)
            
            if scale_percent < 1:
                width = max(1, int(bub.shape[1] * scale_percent))
                height = max(1, int(bub.shape[0] * scale_percent))
                dim = (width, height)
                resized = cv2.resize(bub, dim, interpolation = cv2.INTER_AREA) if resize else bub
            else:
                width = w
                resized = bub

        imwrite(cut_path, resized, '.jpg')
        cuts_path_list.append(cut_path)
        cut_width_list.append(width)

    return cuts_path_list, cut_width_list




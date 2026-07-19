"""Fill missing Traditional Chinese Qt translations from the Simplified catalog.

The target catalog is edited as text so that Qt Linguist formatting, locations,
and unrelated entries remain unchanged.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from xml.etree import ElementTree

import opencc


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / 'resources' / 'translate' / 'zh_CN.ts'
DEFAULT_TARGET = REPOSITORY_ROOT / 'resources' / 'translate' / 'zh_TW.ts'
MESSAGE_RE = re.compile(r'<message(?:\s[^>]*)?>.*?</message>', re.DOTALL)
CONTEXT_RE = re.compile(r'<context>.*?</context>', re.DOTALL)
TRANSLATION_RE = re.compile(
    r'<translation(?P<attributes>[^>]*)>(?P<body>.*?)</translation>',
    re.DOTALL,
)
TRANSLATION_EMPTY_RE = re.compile(r'<translation(?P<attributes>[^>]*)\s*/>')
RecordKey = Tuple[str, str]


@dataclass(frozen=True)
class TranslationRecord:
    key: RecordKey
    text: str
    attributes: Dict[str, str]


def _attributes(element: ElementTree.Element) -> Dict[str, str]:
    return dict(element.attrib)


def _records(catalog: str) -> Iterator[Tuple[str, TranslationRecord]]:
    """Yield message XML and its translation record.

    Example:
        >>> list(_records('<TS><context><name>C</name><message><source>S</source><translation>T</translation></message></context></TS>'))[0][1].text
        'T'
    """

    root = ElementTree.fromstring(catalog)
    for context in root.findall('context'):
        context_name = context.findtext('name') or ''
        for message in context.findall('message'):
            source = message.find('source')
            translation = message.find('translation')
            if source is None or translation is None:
                continue
            message_xml = ElementTree.tostring(message, encoding='unicode')
            yield message_xml, _message_record(message_xml, context_name)


def _message_record(message_xml: str, context_name: str) -> TranslationRecord:
    message = ElementTree.fromstring(message_xml)
    source = message.find('source')
    translation = message.find('translation')
    if source is None or translation is None:
        raise ValueError('message has no source or translation element')
    return TranslationRecord(
        key=(context_name, source.text or ''),
        text=translation.text or '',
        attributes=_attributes(translation),
    )


def _source_translations(catalog: str) -> Dict[RecordKey, str]:
    """Return usable source translations indexed by context and source text."""

    translations: Dict[RecordKey, str] = {}
    for _, record in _records(catalog):
        if record.text and not record.attributes:
            translations.setdefault(record.key, record.text)
    return translations


def _replace_translation(message_xml: str, translated_text: str) -> Optional[str]:
    match = TRANSLATION_RE.search(message_xml)
    if match:
        if match.group('body').strip():
            return None
        replacement = f'<translation>{escape(translated_text, quote=False)}</translation>'
        return message_xml[:match.start()] + replacement + message_xml[match.end():]

    match = TRANSLATION_EMPTY_RE.search(message_xml)
    if match:
        replacement = f'<translation>{escape(translated_text, quote=False)}</translation>'
        return message_xml[:match.start()] + replacement + message_xml[match.end():]
    return None


def merge_catalogs(source_path: Path, target_path: Path, *, dry_run: bool = False) -> int:
    """Merge empty target translations and return the number changed.

    An empty ``type="unfinished"`` element is treated as missing; filling it
    removes the marker. Non-empty unfinished translations are left untouched.

    Example:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     source = Path(directory) / 'source.ts'
        ...     target = Path(directory) / 'target.ts'
        ...     _ = source.write_text('<TS/>', encoding='utf-8')
        ...     _ = target.write_text('<TS/>', encoding='utf-8')
        ...     merge_catalogs(source, target, dry_run=True)
        0
    """

    source_catalog = source_path.read_text(encoding='utf-8')
    target_catalog = target_path.read_text(encoding='utf-8')
    source_translations = _source_translations(source_catalog)
    converter = opencc.OpenCC('s2t')
    changed = 0

    def replace_context(match: re.Match[str]) -> str:
        context_xml = match.group(0)
        context_name = ElementTree.fromstring(context_xml).findtext('name') or ''

        def replace_context_message(message_match: re.Match[str]) -> str:
            message_xml = message_match.group(0)
            record = _message_record(message_xml, context_name)
            source_text = source_translations.get(record.key)
            if source_text is None or record.text.strip():
                return message_xml
            replacement = _replace_translation(message_xml, converter.convert(source_text))
            if replacement is None:
                return message_xml
            nonlocal changed
            changed += 1
            return replacement

        return MESSAGE_RE.sub(replace_context_message, context_xml)

    merged_catalog = CONTEXT_RE.sub(replace_context, target_catalog)
    if changed and not dry_run:
        target_path.write_text(merged_catalog, encoding='utf-8', newline='')
    return changed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--dry-run', action='store_true', help='Report changes without writing the target.')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    changed = merge_catalogs(args.source, args.target, dry_run=args.dry_run)
    action = 'would update' if args.dry_run else 'updated'
    print(f'{action} {changed} translation(s) in {args.target}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

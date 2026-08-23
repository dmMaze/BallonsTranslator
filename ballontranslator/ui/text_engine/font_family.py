"""Qt font-family compatibility at the text-engine boundary."""

from hashlib import sha1
from html import escape, unescape
import re
from typing import Callable, Iterable, Sequence

from qtpy.QtGui import QFont, QTextCursor, QTextDocument

from ballontranslator.utils import shared
from ballontranslator.utils.font_registry import normalize_key
from ballontranslator.utils.fontformat import font_weight_from_qt


_QT_FAMILY_BY_PROJECT_NAME: dict[str, str] = {}
_PROJECT_FAMILY_BY_QT_NAME: dict[str, str] = {}
_FONT_FAMILY_DECLARATION = re.compile(
    r'font-family\s*:'
    r'(?:&(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);|[^;>])*',
    re.IGNORECASE,
)


def _font_families_from_declaration(declaration: str) -> tuple[str, ...]:
    """Return CSS family names without splitting quoted commas.

    >>> _font_families_from_declaration("font-family:'A, Display', serif")
    ('A, Display', 'serif')
    """
    value = declaration.split(':', 1)[-1]
    families = []
    chunk = []
    quote = None
    escaped = False
    for character in value:
        if escaped:
            chunk.append(character)
            escaped = False
        elif character == '\\':
            escaped = True
        elif quote is not None:
            if character == quote:
                quote = None
            else:
                chunk.append(character)
        elif character in {'\'', '"'}:
            quote = character
        elif character == ',':
            family = unescape(''.join(chunk).strip())
            if family:
                families.append(family)
            chunk = []
        else:
            chunk.append(character)
    if escaped:
        chunk.append('\\')
    family = unescape(''.join(chunk).strip())
    if family:
        families.append(family)
    return tuple(families)


def _safe_font_family_alias(family: str, used_names: set[str]) -> str:
    """Return a stable family name that Qt cannot parse as foundry syntax.

    >>> alias = _safe_font_family_alias('[toolbox]BuDing-JF', set())
    >>> alias.startswith('BalloonsTranslator Font ') and '[' not in alias
    True
    """
    digest = sha1(family.encode('utf-8')).hexdigest()[:12]
    base = f'BalloonsTranslator Font {digest}'
    alias = base
    suffix = 2
    while alias.casefold() in used_names:
        alias = f'{base} {suffix}'
        suffix += 1
    return alias


def register_qt_font_family_aliases(
    font_families: Iterable[str],
    style_names: Callable[[str], Sequence[str]],
) -> dict[str, str]:
    """Register safe aliases for bracketed families Qt cannot select.

    Qt treats square brackets in a family name as foundry syntax. Some real
    fonts begin with a bracketed vendor tag, so Qt advertises their family but
    returns no styles and resolves a different face when that name is used.
    """
    families = tuple(font_families)
    used_names = {family.casefold() for family in families}
    used_names.update(_PROJECT_FAMILY_BY_QT_NAME)
    registered: dict[str, str] = {}
    for family in sorted(families, key=str.casefold):
        key = family.casefold()
        if key in _QT_FAMILY_BY_PROJECT_NAME:
            registered[family] = _QT_FAMILY_BY_PROJECT_NAME[key]
            continue
        if '[' not in family and ']' not in family:
            continue
        if style_names(family):
            # A newer Qt/font backend may learn to handle this name directly.
            continue
        alias = _safe_font_family_alias(family, used_names)
        QFont.insertSubstitution(alias, family)
        _QT_FAMILY_BY_PROJECT_NAME[key] = alias
        _PROJECT_FAMILY_BY_QT_NAME[alias.casefold()] = family
        used_names.add(alias.casefold())
        registered[family] = alias
    return registered


def _registry_resolution(family: str, weight: int | None = None):
    registry = getattr(shared, 'FONT_REGISTRY', None)
    if registry is None:
        return None
    return registry.resolve_family(family, weight)


def font_family_for_qt(family: str, weight: int | None = None) -> str:
    """Return the internal family name Qt can resolve correctly."""
    resolution = _registry_resolution(family, weight)
    render_family = (
        resolution.qt_family
        if resolution is not None and resolution.qt_family
        else family
    )
    return _QT_FAMILY_BY_PROJECT_NAME.get(
        render_family.casefold(),
        render_family,
    )


def font_family_for_project(
    family: str,
    weight: int | None = None,
) -> str:
    """Return the stable user/project-facing name for an internal family."""
    project_family = _PROJECT_FAMILY_BY_QT_NAME.get(
        family.casefold(),
        family,
    )
    resolution = _registry_resolution(project_family, weight)
    if resolution is not None and resolution.canonical_family:
        return resolution.canonical_family
    return project_family


def qfont_with_family(font: QFont, family: str) -> QFont:
    """Copy ``font`` and safely set any project-facing family."""
    result = QFont(font)
    weight = int(font_weight_from_qt(result.weight()))
    resolved = font_family_for_qt(family, weight)
    # Qt 5 can retain an HTML font's old family list after setFamily(), while
    # family() can retain its old value after setFamilies(). Set both so the
    # renderer and the persisted/UI-facing accessor agree on one family.
    result.setFamilies([resolved])
    result.setFamily(resolved)
    return result


def html_uses_project_font_family(html: str) -> bool:
    """Return whether HTML can require internal family normalization."""
    registry = getattr(shared, 'FONT_REGISTRY', None)
    registry_keys = registry.entries_by_key if registry is not None else {}
    project_keys = {
        normalize_key(family)
        for family in _QT_FAMILY_BY_PROJECT_NAME
    }
    for match in _FONT_FAMILY_DECLARATION.finditer(html):
        for family in _font_families_from_declaration(match.group(0)):
            key = normalize_key(family)
            if key in project_keys or key in registry_keys:
                return True
    return False


def normalize_document_font_families(document: QTextDocument) -> int:
    """Replace project-facing aliases in a live document with Qt-safe names."""
    replacements = 0
    default_font = document.defaultFont()
    resolved_default = font_family_for_qt(
        default_font.family(),
        int(font_weight_from_qt(default_font.weight())),
    )
    if resolved_default != default_font.family():
        document.setDefaultFont(
            qfont_with_family(default_font, default_font.family())
        )
        replacements += 1

    cursor = QTextCursor(document)
    block = document.firstBlock()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            char_format = fragment.charFormat()
            family = char_format.font().family()
            weight = int(font_weight_from_qt(char_format.font().weight()))
            if font_family_for_qt(family, weight) != family:
                char_format.setFont(
                    qfont_with_family(char_format.font(), family)
                )
                cursor.setPosition(fragment.position())
                cursor.setPosition(
                    fragment.position() + fragment.length(),
                    QTextCursor.MoveMode.KeepAnchor,
                )
                cursor.setCharFormat(char_format)
                replacements += 1
            iterator += 1
        block = block.next()
    return replacements


def restore_project_font_families_in_html(html: str) -> str:
    """Hide internal aliases in serialized ``font-family`` declarations."""
    registry = getattr(shared, 'FONT_REGISTRY', None)

    def restore_declaration(match: re.Match) -> str:
        declaration = match.group(0)
        for family in _font_families_from_declaration(declaration):
            project_family = _PROJECT_FAMILY_BY_QT_NAME.get(
                family.casefold(), family
            )
            storage_family = (
                registry.family_for_export(project_family)
                if registry is not None
                else project_family
            )
            if storage_family == family:
                continue
            escaped_storage = escape(storage_family, quote=True)
            source_names = {
                family,
                escape(family, quote=False),
                escape(family, quote=True),
            }
            for source_name in sorted(source_names, key=len, reverse=True):
                declaration = re.sub(
                    rf'(?<![\w]){re.escape(source_name)}(?![\w])',
                    lambda _match, value=escaped_storage: value,
                    declaration,
                    flags=re.IGNORECASE,
                )
        return declaration

    return _FONT_FAMILY_DECLARATION.sub(restore_declaration, html)

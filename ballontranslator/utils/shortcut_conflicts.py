"""Shared shortcut-conflict detection.

Used by the shortcut editor (``ShortcutEditor``) — every user-assignable
key sequence must not be bound to more than one action.
"""

from typing import Dict, Iterable, List, Set


def find_conflict_keys(mapping: Dict[str, Iterable[str]]) -> Set[str]:
    """Return the set of key sequences bound to more than one action.

    *mapping* maps an action id to its list of effective key sequences;
    empty / ``None`` entries are skipped.  The returned set contains the
    duplicated key strings.
    """
    seen: Dict[str, List[str]] = {}
    for owner, keys in mapping.items():
        for k in keys or []:
            if k:
                seen.setdefault(k, []).append(owner)
    return {k for k, owners in seen.items() if len(owners) > 1}

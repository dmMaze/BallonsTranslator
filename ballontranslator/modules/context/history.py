"""Reusable immutable history-window state and page-budget operations."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from ballontranslator.utils.proj_imgtrans import ProjImgTrans


HISTORY_LOW_WATER_RATIO = 0.60


class ContextAction(Enum):
    """How translation history was handled for one request.

    >>> ContextAction.REBUILD.value
    'rebuild'
    """

    DISABLED = 'disabled'
    EMPTY = 'empty'
    REBUILD = 'rebuild'
    REUSE = 'reuse'
    GROW = 'grow'
    EVICT = 'evict'
    CONTEXT_RECOVERY = 'context-recovery'


class ContextReason(Enum):
    """Why translation history was rebuilt or could not be used normally.

    >>> ContextReason.WINDOW_EMPTY.value
    'window-empty'
    """

    HISTORY_DISABLED = 'history-disabled'
    MISSING_PROJECT_PAGE = 'missing-project-page'
    WINDOW_EMPTY = 'window-empty'
    MISSING_LOAD_IDENTITY = 'missing-load-identity'
    PROJECT_CHANGED = 'project-changed'
    SETTINGS_CHANGED = 'settings-changed'
    MISSING_PAGES = 'missing-pages'
    NON_ADJACENT = 'non-adjacent'
    SNAPSHOT_CHANGED = 'snapshot-changed'
    PREVIOUS_INCOMPLETE = 'previous-incomplete'
    OVERSIZED_PAGE = 'oversized-page'


@dataclass(frozen=True)
class HistoryPage:
    """One indivisible page of input/output context.

    >>> HistoryPage('001.png', ('hello',), ('bonjour',)).page_key
    '001.png'
    """

    page_key: str
    sources: Tuple[str, ...]
    translations: Tuple[str, ...]
    summary: str = ''


@dataclass(frozen=True)
class RenderedHistoryPage:
    """A page snapshot plus its immutable provider messages and token cost.

    >>> page = HistoryPage('001.png', ('hello',), ('bonjour',))
    >>> RenderedHistoryPage(page, (), 3).page_key
    '001.png'
    """

    snapshot: HistoryPage
    messages: Tuple[Tuple[str, str], ...]
    token_count: int

    @property
    def page_key(self) -> str:
        return self.snapshot.page_key


@dataclass(frozen=True)
class HistoryWindowKey:
    """Project identity plus hashable modality-specific context settings.

    >>> key = HistoryWindowKey(object(), (('model', 'demo'),))
    >>> key.settings[0]
    ('model', 'demo')
    """

    load_identity: object
    settings: Tuple[Tuple[str, object], ...]


@dataclass(frozen=True)
class HistoryWindow:
    """Committed history state from the most recent successful request.

    >>> key = HistoryWindowKey(None, ())
    >>> HistoryWindow(key, '001.png', (), 0).request_page_key
    '001.png'
    """

    key: HistoryWindowKey
    request_page_key: str
    history: Tuple[RenderedHistoryPage, ...]
    token_count: int


@dataclass(frozen=True)
class ContextDiagnostic:
    """Safe aggregate context-window details for one request.

    >>> str(ContextDiagnostic('001.png', ContextAction.EMPTY, 0, 0, 10))
    'LLM Context: page=001.png, action=empty, pages=0, tokens=0/10'
    """

    page_key: str
    action: ContextAction
    page_count: int
    token_count: int
    token_budget: int
    appended: int = 0
    evicted: int = 0
    summaries_evicted: int = 0
    rebuild_reason: Optional[ContextReason] = None

    def __str__(self) -> str:
        """Return one compact, line-safe context-window diagnostic."""

        page_key = self.page_key.replace('\r', ' ').replace('\n', ' ')
        details = [
            'LLM Context: page={}'.format(page_key or '-'),
            'action={}'.format(self.action.value),
            'pages={}'.format(self.page_count),
            'tokens={}/{}'.format(self.token_count, self.token_budget),
        ]
        if self.appended:
            details.append('appended={}'.format(self.appended))
        if self.evicted:
            details.append('evicted={}'.format(self.evicted))
        if self.summaries_evicted:
            details.append(
                'summaries_evicted={}'.format(self.summaries_evicted)
            )
        if self.rebuild_reason is not None:
            details.append('reason={}'.format(self.rebuild_reason.value))
        return ', '.join(details)


def eligible_history_for_request(
    *,
    window: Optional[HistoryWindow],
    project: Optional['ProjImgTrans'],
    page_key: str,
    previous_page: Optional[HistoryPage],
    token_budget: int,
    rebuild_reason: Optional[ContextReason],
    snapshot_page: Callable[[str], Optional[HistoryPage]],
    render_page: Callable[[HistoryPage], RenderedHistoryPage],
    reserved_tokens: int = 0,
) -> Tuple[Tuple[RenderedHistoryPage, ...], ContextDiagnostic]:
    """Select eligible pages and fit them into the runtime history window.

    Rebuilds choose a recent chronological suffix below the low-water target,
    leaving room for adjacent requests to extend the provider-cache prefix.

    >>> int(10 * HISTORY_LOW_WATER_RATIO)
    6
    """

    if rebuild_reason is not None:
        pages = getattr(project, 'pages', None)
        # Rebuilding to the full budget would force the next adjacent request to
        # evict immediately. Walk backward from the current page and snapshot
        # only the recent pages needed to reach low water.
        rebuild_limit = max(
            0,
            int(token_budget * HISTORY_LOW_WATER_RATIO) - reserved_tokens,
        )
        history_limit = max(0, token_budget - reserved_tokens)
        remaining = max(0, rebuild_limit)
        selected = []
        if isinstance(pages, dict) and page_key in pages:
            reached_current = False
            for candidate_key in reversed(pages):
                if not reached_current:
                    reached_current = candidate_key == page_key
                    continue
                page = snapshot_page(candidate_key)
                if page is None:
                    continue
                rendered_page = render_page(page)
                if rendered_page.token_count > history_limit:
                    if selected:
                        break
                    continue
                if not selected and rendered_page.token_count > rebuild_limit:
                    # Retain one recent page that fits the configured budget even
                    # when that indivisible page alone exceeds the soft target.
                    selected.append(rendered_page)
                    break
                if rendered_page.token_count > remaining:
                    break
                selected.append(rendered_page)
                remaining -= rendered_page.token_count
        selected.reverse()
        history = tuple(selected)
        token_count = sum(page.token_count for page in history)
        return history, ContextDiagnostic(
            page_key=page_key,
            action=ContextAction.REBUILD if history else ContextAction.EMPTY,
            page_count=len(history),
            token_count=token_count + reserved_tokens,
            token_budget=token_budget,
            rebuild_reason=rebuild_reason,
        )

    if window is None:
        raise RuntimeError('A reusable context window is required.')
    if previous_page is None:
        raise RuntimeError('An eligible previous page is required.')
    rendered_previous_page = render_page(previous_page)
    history = list(window.history)
    token_count = window.token_count
    history_limit = max(0, token_budget - reserved_tokens)
    if rendered_previous_page.token_count > history_limit:
        # Keep the existing prefix stable rather than splitting an oversized page.
        evicted = 0
        if token_count > history_limit:
            low_water = max(
                0,
                int(token_budget * HISTORY_LOW_WATER_RATIO) - reserved_tokens,
            )
            while history and token_count > low_water:
                token_count -= history.pop(0).token_count
                evicted += 1
        return tuple(history), ContextDiagnostic(
            page_key=page_key,
            action=(
                ContextAction.EVICT if evicted else ContextAction.REUSE
            ),
            page_count=len(history),
            token_count=token_count + reserved_tokens,
            token_budget=token_budget,
            evicted=evicted,
            rebuild_reason=ContextReason.OVERSIZED_PAGE,
        )

    if token_count + rendered_previous_page.token_count <= history_limit:
        history.append(rendered_previous_page)
        token_count += rendered_previous_page.token_count
        return tuple(history), ContextDiagnostic(
            page_key=page_key,
            action=ContextAction.GROW,
            page_count=len(history),
            token_count=token_count + reserved_tokens,
            token_budget=token_budget,
            appended=1,
        )

    low_water = max(
        0,
        int(token_budget * HISTORY_LOW_WATER_RATIO) - reserved_tokens,
    )
    evicted = 0
    # Bulk eviction creates headroom for several later appends instead of
    # invalidating the provider cache on every following page.
    while history and (
        token_count > low_water
        or token_count + rendered_previous_page.token_count > history_limit
    ):
        token_count -= history.pop(0).token_count
        evicted += 1
    history.append(rendered_previous_page)
    token_count += rendered_previous_page.token_count
    return tuple(history), ContextDiagnostic(
        page_key=page_key,
        action=ContextAction.EVICT,
        page_count=len(history),
        token_count=token_count + reserved_tokens,
        token_budget=token_budget,
        appended=1,
        evicted=evicted,
    )


def window_rebuild_reason(
    window: Optional[HistoryWindow],
    project: Optional['ProjImgTrans'],
    page_key: str,
    window_key: HistoryWindowKey,
) -> Optional[ContextReason]:
    """Return why a runtime window is unsafe to reuse, otherwise ``None``.

    >>> window_rebuild_reason(None, None, '001.png', HistoryWindowKey(None, ()))
    <ContextReason.WINDOW_EMPTY: 'window-empty'>
    """

    if window is None:
        return ContextReason.WINDOW_EMPTY
    if window_key.load_identity is None:
        return ContextReason.MISSING_LOAD_IDENTITY
    # Identity comparison is intentional: reopening the same path is a new project load.
    if window.key.load_identity is not window_key.load_identity:
        return ContextReason.PROJECT_CHANGED
    if window.key != window_key:
        return ContextReason.SETTINGS_CHANGED

    pages = getattr(project, 'pages', None)
    if not isinstance(pages, dict):
        return ContextReason.MISSING_PAGES
    page_keys = list(pages)
    # Production callers obtain page_key from this project; a stale key is a bug.
    page_index = page_keys.index(page_key)
    if (
        page_index == 0
        or page_keys[page_index - 1] != window.request_page_key
    ):
        return ContextReason.NON_ADJACENT
    return None

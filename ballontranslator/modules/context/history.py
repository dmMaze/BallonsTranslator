"""Reusable immutable history-window state and page-budget operations."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .glossary import GlossaryEntry


HISTORY_LOW_WATER_RATIO = 0.60


@dataclass(frozen=True)
class HistoryPage:
    """One indivisible page of input/output context.

    >>> HistoryPage('001.png', ('hello',), ('bonjour',)).page_key
    '001.png'
    """

    page_key: str
    sources: Tuple[str, ...]
    translations: Tuple[str, ...]


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

    >>> ContextDiagnostic('001.png', 'empty', 0, 0, 10).action
    'empty'
    """

    page_key: str
    action: str
    page_count: int
    token_count: int
    token_budget: int
    appended: int = 0
    evicted: int = 0
    rebuild_reason: str = ''


@dataclass(frozen=True)
class RequestContext:
    """Immutable history and optional glossary used for provider retries.

    >>> RequestContext(()).history
    ()
    """

    history: Tuple[RenderedHistoryPage, ...]
    glossary: Tuple[GlossaryEntry, ...] = ()
    glossary_mode: str = ''
    history_budget: int = 0
    window_key: Optional[HistoryWindowKey] = None
    request_page_key: Optional[str] = None
    diagnostic: Optional[ContextDiagnostic] = None


def get_context_diagnostic(diagnostic: ContextDiagnostic) -> str:
    """Return one compact, line-safe context-window diagnostic.

    >>> get_context_diagnostic(ContextDiagnostic('001.png', 'empty', 0, 0, 10))
    'LLM Context: page=001.png, action=empty, pages=0, tokens=0/10'
    """

    page_key = diagnostic.page_key.replace('\r', ' ').replace('\n', ' ')
    details = [
        'LLM Context: page={}'.format(page_key or '-'),
        'action={}'.format(diagnostic.action),
        'pages={}'.format(diagnostic.page_count),
        'tokens={}/{}'.format(
            diagnostic.token_count,
            diagnostic.token_budget,
        ),
    ]
    if diagnostic.appended:
        details.append('appended={}'.format(diagnostic.appended))
    if diagnostic.evicted:
        details.append('evicted={}'.format(diagnostic.evicted))
    if diagnostic.rebuild_reason:
        details.append('reason={}'.format(diagnostic.rebuild_reason))
    return ', '.join(details)


def snapshot_eligible_history(
    project,
    page_key,
    snapshot_page: Callable[[object], Optional[HistoryPage]],
) -> Tuple[HistoryPage, ...]:
    """Copy eligible pages preceding ``page_key`` through a modality callback.

    >>> snapshot_eligible_history(None, '001.png', lambda _key: None)
    ()
    """

    pages = getattr(project, 'pages', None)
    if not isinstance(pages, dict) or page_key not in pages:
        return ()

    history = []
    for candidate_key in pages:
        if candidate_key == page_key:
            break
        page = snapshot_page(candidate_key)
        if page is not None:
            history.append(page)
    return tuple(history)


def history_for_request(
    *,
    window: Optional[HistoryWindow],
    page_key: str,
    eligible_history: Tuple[HistoryPage, ...],
    token_budget: int,
    rebuild_reason: str,
    render_page: Callable[[HistoryPage], RenderedHistoryPage],
) -> Tuple[Tuple[RenderedHistoryPage, ...], ContextDiagnostic]:
    """Select a safe rebuild or extend the cache-oriented runtime window.

    Rebuilds choose a recent chronological suffix. Adjacent requests append to
    the committed prefix until bulk eviction is required.

    >>> int(10 * HISTORY_LOW_WATER_RATIO)
    6
    """

    if rebuild_reason:
        # Walk newest-first, then restore page order. Stop at the first ordinary
        # overflow so older small pages cannot create a non-contiguous suffix.
        remaining = max(0, int(token_budget))
        selected = []
        for page in reversed(eligible_history):
            rendered_page = render_page(page)
            if rendered_page.token_count > token_budget:
                if selected:
                    break
                continue
            if rendered_page.token_count > remaining:
                break
            selected.append(rendered_page)
            remaining -= rendered_page.token_count
        selected.reverse()
        history = tuple(selected)
        token_count = sum(page.token_count for page in history)
        return history, ContextDiagnostic(
            page_key=page_key,
            action='rebuild' if history else 'empty',
            page_count=len(history),
            token_count=token_count,
            token_budget=token_budget,
            rebuild_reason=rebuild_reason,
        )

    if window is None:
        raise RuntimeError('A reusable context window is required.')
    previous_page = next(
        page
        for page in eligible_history
        if page.page_key == window.request_page_key
    )
    rendered_page = render_page(previous_page)
    history = list(window.history)
    token_count = window.token_count
    if rendered_page.token_count > token_budget:
        # Keep the existing prefix stable rather than splitting an oversized page.
        return tuple(history), ContextDiagnostic(
            page_key=page_key,
            action='reuse',
            page_count=len(history),
            token_count=token_count,
            token_budget=token_budget,
            rebuild_reason='oversized-page',
        )

    if token_count + rendered_page.token_count <= token_budget:
        history.append(rendered_page)
        token_count += rendered_page.token_count
        return tuple(history), ContextDiagnostic(
            page_key=page_key,
            action='grow',
            page_count=len(history),
            token_count=token_count,
            token_budget=token_budget,
            appended=1,
        )

    low_water = int(token_budget * HISTORY_LOW_WATER_RATIO)
    evicted = 0
    # Bulk eviction creates headroom for several later appends instead of
    # invalidating the provider cache on every following page.
    while history and (
        token_count > low_water
        or token_count + rendered_page.token_count > token_budget
    ):
        token_count -= history.pop(0).token_count
        evicted += 1
    history.append(rendered_page)
    token_count += rendered_page.token_count
    return tuple(history), ContextDiagnostic(
        page_key=page_key,
        action='evict',
        page_count=len(history),
        token_count=token_count,
        token_budget=token_budget,
        appended=1,
        evicted=evicted,
    )


def window_rebuild_reason(
    window: Optional[HistoryWindow],
    project,
    page_key: str,
    window_key: HistoryWindowKey,
) -> str:
    """Return an enum-like reason when a runtime window is unsafe to reuse.

    >>> window_rebuild_reason(None, None, '001.png', HistoryWindowKey(None, ()))
    'window-empty'
    """

    if window is None:
        return 'window-empty'
    if window_key.load_identity is None:
        return 'missing-load-identity'
    # Identity comparison is intentional: reopening the same path is a new project load.
    if window.key.load_identity is not window_key.load_identity:
        return 'project-changed'
    if window.key != window_key:
        return 'settings-changed'

    pages = getattr(project, 'pages', None)
    if not isinstance(pages, dict):
        return 'missing-pages'
    page_keys = list(pages)
    try:
        page_index = page_keys.index(page_key)
    except ValueError:
        return 'missing-page'
    if (
        page_index == 0
        or page_keys[page_index - 1] != window.request_page_key
    ):
        return 'non-adjacent'
    return ''


def recover_context_length(
    request_context: Optional[RequestContext],
) -> Optional[RequestContext]:
    """Remove whole oldest pages toward the shared low-water target.

    >>> int(4096 * HISTORY_LOW_WATER_RATIO)
    2457
    """

    if request_context is None or not request_context.history:
        return None

    history = list(request_context.history)
    token_count = sum(page.token_count for page in history)
    low_water = int(request_context.history_budget * HISTORY_LOW_WATER_RATIO)
    evicted = 0
    # Remove at least one whole page because provider tokenization may exceed
    # the estimator even when the estimated window is below its configured limit.
    while history and (token_count > low_water or evicted == 0):
        token_count -= history.pop(0).token_count
        evicted += 1

    diagnostic = ContextDiagnostic(
        page_key=str(request_context.request_page_key or ''),
        action='context-recovery',
        page_count=len(history),
        token_count=token_count,
        token_budget=request_context.history_budget,
        evicted=evicted,
    )
    return RequestContext(
        history=tuple(history),
        glossary=request_context.glossary,
        glossary_mode=request_context.glossary_mode,
        history_budget=request_context.history_budget,
        window_key=request_context.window_key,
        request_page_key=request_context.request_page_key,
        diagnostic=diagnostic,
    )


def history_window_from_context(
    request_context: Optional[RequestContext],
) -> Optional[HistoryWindow]:
    """Build committed window state from a successfully used request context.

    The caller owns the success boundary; this helper only freezes that state.

    >>> history_window_from_context(None) is None
    True
    """

    if (
        request_context is None
        or request_context.window_key is None
        or request_context.request_page_key is None
    ):
        return None
    return HistoryWindow(
        key=request_context.window_key,
        request_page_key=request_context.request_page_key,
        history=request_context.history,
        token_count=sum(page.token_count for page in request_context.history),
    )

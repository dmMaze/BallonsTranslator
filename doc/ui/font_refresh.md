# Runtime font refresh

Runtime refresh is supported in GUI mode with Qt 6.4 or later. Qt 5, older Qt 6,
and headless startup retain normal font registration without refresh hooks.
This support boundary does not imply a public Qt database-refresh API.

## Ownership and ordering

| Concern | Owner |
| --- | --- |
| Native Windows and Qt change notifications | [`ui/font_change_detection.py`](../../ballontranslator/ui/font_change_detection.py) |
| Debounce, worker preparation, Qt mutation, and publication | [`ui/font_refresh.py`](../../ballontranslator/ui/font_refresh.py) |
| Fontconfig refresh, file snapshots, and registration updates | [`utils/font_refresh.py`](../../ballontranslator/utils/font_refresh.py) |

Detection only emits notifications. The main window connects it to the refresh
controller for supported sessions and stops detection before worker shutdown.
Manual reload calls the controller directly. Disk reads and metadata parsing run
on a worker; Qt mutation/publication run on the GUI thread. Shutdown waits for
pending reads. Requests coalesce, and self-generated Qt notifications are ignored.

| Trigger | Refresh boundary |
| --- | --- |
| Windows `WM_FONTCHANGE` | Debounce, force Qt invalidation, publish |
| Qt `fontDatabaseChanged` | Synchronize without another forced invalidation |
| Manual reload | Scan changed application fonts; Windows invalidates Qt, macOS synchronizes its current database, Linux xcb/Wayland refreshes fontconfig then invalidates Qt |

Automatic refresh retains application registrations without rescanning `fonts/`.
Manual refresh uses only IDs owned by `FontRegistry.registrations`. Failed
replacements retain the old registration; incomplete directory scans abort
instead of treating unseen files as deleted. Forced invalidation briefly adds
and removes a bundled seed font: an observed Qt side effect, not a public
refresh guarantee. Never clear all application fonts.

Publication preserves custom groups, exclusions, and missing selected families,
updates `shared.FONT_FAMILIES` and font aliases, and clears metric caches. The
main window rebinds and reshapes every text block on the current page, refreshes
style preset labels, then publishes family/weight choices. Transient layout
formats prevent same-family replacements from retaining an old native font
engine without changing rich text, project data, or undo history.
Picker refresh discards unaccepted search text and restores the committed family
without emitting a font-change action, even if that family is no longer listed.

## Font identity

Persist family names unchanged; `qfont_with_family()` supplies Qt-safe runtime
aliases. Picker labels prefer localized typographic-family metadata, with explicit
display overrides taking precedence. Resolve and cache system labels on demand;
autocomplete must not open every font. Display aliases must not replace saved-name
mappings or exported family names. Font weights use CSS/Qt 6 values (`100`–`900`),
with legacy Qt 5 normalization only at the Qt/HTML boundary.

## Failure boundaries and verification

Linux refresh calls `FcInitReinitialize` on the process-default configuration and
assumes Qt uses that fontconfig library. This replaces programmatic default-config
changes; custom Qt builds/configurations and non-fontconfig backends need separate
validation. An unchanged family list is not itself a failure. There is no Linux
filesystem watcher; WSL testing needs Linux Python and its font environment.

Manual helper failures are reported while Qt refresh is still attempted;
automatic failures are logged without dialogs. Logs use `[font-refresh]`; enable
`BT_FONT_REFRESH_DEBUG=1` for detailed diagnostics.

Follow [repository verification](../../AGENTS.md#verification). Run
`tests/test_font_refresh.py` with font registry, family-resolution, and weight
tests. Native validation must cover font install/removal on Windows/macOS and
manual refresh on Linux xcb/Wayland, repeated refresh, corrupt custom-file
replacement, missing selections, exclusions, and preservation of unowned font
IDs. Record Qt/backend versions; offscreen tests cannot certify native discovery.

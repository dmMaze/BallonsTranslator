# Runtime font refresh

The font panel reload button is available in GUI mode with Qt 6.4 or later.
This is a product support boundary, not a claim that Qt 6.4 introduced a
public font-database refresh API. Qt 5, older Qt 6, and headless startup keep
the existing registration path without automatic refresh hooks.

## Ownership and ordering

`ui/font_change_detection.py` owns native Windows and Qt signal subscriptions.
It emits separate system-change and Qt-database-change notifications without
querying fonts, registering fonts, or triggering a reload itself.

`ui/font_refresh.py` accepts explicit refresh/sync requests and owns debounce,
background file preparation, Qt mutation, and publication to the formatting
panel. It installs no native filters or application font-change subscriptions.
The main window connects detection to refresh only for supported GUI sessions
and stops detection before shutting down the refresh worker. Manual reload
calls the refresh controller directly and does not depend on detection.
`utils/font_refresh.py` owns fontconfig refresh, changed-file snapshots, and
Qt registration updates. `FontRegistry.registrations` retains the IDs and
file fingerprints created by startup and subsequent manual refreshes.

| Trigger | Behavior |
| --- | --- |
| Windows `WM_FONTCHANGE` | Debounce 300 ms, force Qt invalidation, synchronize app state |
| Qt `fontDatabaseChanged`, including native macOS notifications | Debounce and synchronize app state without another forced invalidation |
| Manual reload, all supported platforms | Read changed application `fonts/` files; on Linux xcb/Wayland refresh fontconfig; invalidate Qt and synchronize app state |

There is no Linux filesystem watcher. WSL tests must use Linux Python and
its own font environment; Windows-installed fonts are not an equivalent test.
Automatic refresh retains application font registrations without rescanning
`fonts/`. Manual refresh adds, replaces, or removes files using owned IDs;
a failed replacement retains the previous registration. Incomplete directory
enumeration aborts the update instead of treating unseen files as deleted.

Qt mutation and UI publication run on the GUI thread. Disk reads and font
metadata parsing run on a worker. Shutdown waits for that worker, including
any pending filesystem read. Requests received during preparation coalesce
into a subsequent refresh; signals emitted by our Qt mutations are ignored.

Forced invalidation briefly registers the bundled OFL Abel font and removes
only that ID. It relies on an observed Qt implementation side effect, not a
public refresh guarantee. It never clears all application fonts. Resources
are packaged with the application and do not require matplotlib or downloads.

Publication replaces the system registry and `shared.FONT_FAMILIES`, registers
safe Qt aliases, clears character-width and punctuation caches, and updates
the family/weight choices. Missing selected families remain stored; refresh
must not emit formatting edits or add undo commands. Project JSON is unchanged.
Existing canvas documents are not forcibly reshaped or rerendered.

The reload arrow rotates while refreshing and stops when finished. Animation
pauses while hidden; the tooltip retains the last completion or failure state
and reports family counts, additions,
removals, and elapsed preparation/apply time (excluding debounce). Logs use
the application's logger with `[font-refresh]` stages: signal, queue, prepare,
scan, fontconfig, qt, custom, registry, publish, and done. Batch numbers identify
controller refreshes; self-generated Qt signals are explicitly marked. INFO retains Qt notifications, added/removed family names, and the
completion timing summary. Other process details (including Windows notifications and manual requests)
use DEBUG; warnings and errors retain their severity. Detailed DEBUG records
are emitted only when `BT_FONT_REFRESH_DEBUG=1` is set in the application
environment. By default they reach neither the terminal nor the log file.
The shared logger and handler configuration is unchanged.

## Linux failure boundaries

The ctypes helper calls `FcInitReinitialize` on the process-default fontconfig
configuration. This assumes the loaded library is the one used by Qt; custom
Qt builds, separately loaded libraries, and non-fontconfig backends need
separate validation. Reinitialization replaces programmatic default-config
changes. The application does not attempt to refresh arbitrary custom FcConfig
objects or equate an unchanged family list with failure.

The helper reports skipped, unavailable, failed, or refreshed. Manual refresh
shows a nonmodal message when the helper is unavailable or fails, while still
attempting Qt refresh. An unavailable helper may suggest the reported rescan
interval (or the usual 30 seconds if unknown); interval zero recommends restart.
An explicit fontconfig failure recommends checking configuration, not waiting.
Automatic refresh logs errors without opening dialogs.

## Verification

Run `tests/test_font_refresh.py` together with the font registry, family
resolution, and weight tests. They cover registration ownership, corrupt
replacement retention, request coalescing, cache invalidation, feature gates,
and fontconfig failure classification. Native external-font behavior requires
interactive testing on each platform.

Use `scripts/watch_font_database.py --family "ABeeZee"` to observe native
messages, forced-reload signals, discovery, matching, and wall times.
Add `--observe-only` to measure native Qt behavior without invalidation.
Use `--probe` for a startup-only query. The diagnostic uses the packaged seed
by default; `--reload-font` overrides it without installing a system font.

Before release, activate/deactivate and install/remove a test font on Windows
and macOS; on Linux test manual refresh through xcb and Wayland. Check repeated
refresh, selected missing families, custom-file replacement, other application
font IDs, excluded/custom-only filters, and themed UI responsiveness. Record
Qt version and backend; a passing offscreen check does not certify native
font discovery or immediate canvas reflow.

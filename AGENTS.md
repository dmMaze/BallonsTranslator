# AGENTS.md

## Project Shape

BallonsTranslator is a PyQt/qtpy desktop app for comic image translation.

- `ballontranslator/launch.py`: startup, dependencies, Qt setup, and headless mode.
- `ballontranslator/ui/`: canvas, panels, module manager, and workers.
- `ballontranslator/modules/`: registered detector/OCR/translator/inpainter implementations.
- `ballontranslator/utils/proj_imgtrans.py`: project persistence and image/textblock state.
- `ballontranslator/utils/textblock.py`: central TextBlock domain object.
- `ballontranslator/utils/config.py`: persistent config and module settings.
- Read [Text engine](doc/ui/text_engine.md) before changing text layout, effects, interaction, geometry, performance, or rendering; follow its topic links.

Use `rg` for repo search.

## Changes and Data Safety

- Preserve behavior unless explicitly asked to change it; prefer small, reviewable changes within the existing architecture.
- Treat pre-existing modified, untracked, and ignored files as user-owned. Never delete, overwrite, move, clean, or restore them without explicit authorization for the exact paths, especially config, backups, credentials, projects, and models. Follow existing save/backup behavior for source images, translations, masks, and project JSON.
- Passive config/project loading must warn about unknown, removed, renamed, malformed, or out-of-range optional data, discard only invalid portions, and continue. Never let it replace an existing file with empty/template state. Keep live runtime values and explicit write/export boundaries strict; give new saved fields defaults for older projects.
- Use the existing module registries and stable keys; renames require compatibility aliases. Preserve lazy/eager model-loading behavior. Config UI metadata must be lazy or `SafeEval`-compatible and pure, without executing initialization, `_setup_*`, updates, `flush`, model loading, downloads, or network calls.
- New dependencies require approval. Optional integrations must fail gracefully with a clear setup message.
- Make surprising behavior opt-in or clearly discoverable through existing config/UI/module parameters. Preserve Qt localization and ensure pipeline features work or safely no-op in headless mode.
- Keep long-running inference, IO, downloads, and model loading off the Qt main thread; trace signal and worker ordering, especially in `ui/module_manager.py`.

## Performance and Maintainability

- Trace production callers and data flow before changing code. Search for existing methods, helpers, and patterns; reuse or extend the owning implementation before adding another. Prefer standard-library and Qt facilities over custom equivalents.
- Use the simplest structure that meets current requirements. Add a helper or abstraction only when it removes demonstrated duplication or establishes a meaningful responsibility boundary. Avoid pass-through wrappers, speculative extension points, generic frameworks for one concrete operation, and layers that merely forward the same data.
- Put integration in the feature's owning module and shape APIs around production callers. Avoid test-only constructors, injection points, and shared hooks; tests can patch small instance attributes or narrower internals. Use names that make ownership clear.
- Check call frequency across the full operation and repeated lifecycle events. Reuse available results instead of repeating lookups, conversions, copies, validation, IO, or initialization. Fix duplicate calls and signal-driven work at their owner before adding caches or parallelism; justify performance complexity with tracing or measurement.
- Update only changed data and widgets. Keep setters and refreshers idempotent and coalesce equivalent work. Sanitize config at load/migration boundaries; reserve whole-panel rebuilds for reset/restore rather than ordinary row/card edits.
- Use indexed lookups for repeated access and derive values on demand, avoiding per-item full scans and eagerly materialized maps. Reuse existing caches; new caches need a demonstrated cost and explicit invalidation/lifetime. Keep render-only state out of shared config.
- Give state transitions one clear owner. Avoid redundant synchronization paths, equivalent signal emissions, and broad `blockSignals()` used to hide update loops.
- Simplification must remove the obsolete path too: unused helpers, shims, hooks, duplicate state, and architecture-preserving tests. Review the complete caller chain for leftover indirection and repeated work.
- Add useful type hints and return types to new or modified functions. Prefer domain/Qt types and precise callables; use `Any` only at dynamic boundaries and avoid unrelated annotation churn.

## UI Styling

- Lightweight settings/tools should reuse a frameless `Qt.Dialog`, transparent outer widget, rounded theme-token `QFrame`, and scoped stylesheet. Keep native dialogs for platform workflows; confirmations, progress, and costly-to-dismiss flows must not close on outside clicks.
- Keep selectors and lists in the application font; use a dedicated preview for content fonts. Match widget structure before adjusting alignment, such as bare checkboxes plus `ParamNameLabel`.
- Scope config styles by object/section names, never broad widget selectors. Use `resources/themes.json` and `resources/stylesheet.css` tokens, except established accents such as `rgb(30, 147, 229)`. Custom config rows needing a painted background require an object name and `WA_StyledBackground`.
- Scope ordinary checkbox indicators without affecting icon-based controls. `QListWidget` indicators are item-view subcontrols; style their selected, hover, and disabled states separately and check readability in both themes.
- Render manually painted SVG pixmaps through `ui/icon_rendering.py`.
- Never change style machinery inside `Polish`, `StyleChange`, or `Paint`: no `setStyle`, `setStyleSheet`, explicit polish/unpolish, `ensurePolished`, or delegate replacement. Re-entering native style code can crash Qt. Prefer scoped subcontrols, construction-time setup, or idempotent setup in safe show/input events.

## Qt Event Filters

- Reuse `OutsideClickFramelessMixin` from `ui/framelesswindow.py` before the Qt base for lightweight centered, draggable windows with Escape/outside-click dismissal, `title_bar`, and `close_button`. Keep `hide()` for cached panels; override `_dismiss_transient_window()` with `reject()` for staged dialogs and `_preserve_on_outside_click()` for owned popups/dialogs.
- Application event filters are global hooks: install only while active where possible, and remove on hide, collapse, close, or destruction.
- Check relevance before event details. Global filters first check visibility, receiver, and widget/event types; local filters first guard the watched object. Only then access `event.type()`, global positions, or widget-specific methods.
- Outside-click handling should use widget-target mouse presses and explicit popup/dialog whitelists, not broad geometry or `QWindow` interpretation.

## Qt Signals and QObject Lifetimes

- Never connect child/transient signals to lambdas, nested functions, or partials capturing parents or other QObjects. Bindings can retain callbacks and invalid wrappers after native deletion. Prefer bound QObject slots, action data/properties read through `sender()`, or typed row signals.
- Pure-Python or fixed-lifetime callbacks without a demonstrated lifetime problem need no blanket rewrite.
- `close()`/`hide()` do not delete objects. For one-shot modal dialogs, read state after `exec_()` and call `deleteLater()` in `finally`; preserve intentionally cached panels.

## Comments and Documentation

- Comment non-obvious intent, invariants, compatibility, and ordering/failure constraints, especially around Qt, model loading, persistence, and IO. Explain subtle preserved behavior during refactors; omit boilerplate and narration. Include a standard Python `>>>` doctest example for core classes and complex functions.
- Maintainer guides cover current ownership, stable contracts, failure modes, extension points, and verification. Omit UI walkthroughs, pixel measurements, temporary decisions, change history, and behavior already clear from code/tests. Keep each fact in one owning guide, link from overviews, and replace stale prose and nearby duplication when behavior changes.

## Verification

- Test observable behavior and failure modes through real construction paths, not obsolete APIs, incidental widget hierarchy, object names, sizes, margins, spacing, or stretch factors.
- Run targeted import checks and relevant tests for Python changes. UI-heavy changes require `python -m py_compile` on touched files, `git diff --check`, and an offscreen Qt smoke check when practical. Use a themed-app pass for styling and state any unverified UI/threading behavior.
- Risky global-filter changes need a regression proving irrelevant receivers/non-mouse events are ignored before `event.type()` is requested.
- QObject lifecycle fixes need deferred-delete processing and assertions for behavior plus wrapper/registry release. Run both PyQt5 and PyQt6 when binding behavior matters.
- Completion requires existing workflows and older projects to work, new behavior to be configurable or discoverable, and verification results or limitations to be stated.

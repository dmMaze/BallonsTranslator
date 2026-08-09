# AGENTS.md

## Project Shape

BallonsTranslator is a PyQt/qtpy desktop app for comic image translation.

Important areas:
- `ballontranslator/launch.py`: startup, dependency checks, Qt setup, headless mode.
- `ballontranslator/ui/`: Qt UI, canvas, panels, module manager, worker threads.
- `ballontranslator/modules/`: pluggable detector/OCR/translator/inpainter implementations.
- `ballontranslator/utils/proj_imgtrans.py`: project persistence and image/textblock state.
- `ballontranslator/utils/textblock.py`: central TextBlock domain object.
- `ballontranslator/utils/config.py`: persistent config and module settings.
- `doc/ui/text_engine.md`: required entry guide for text layout, effects, interaction, geometry, performance, and rendering; read it before changing those paths and follow its detailed-topic links.

## Refactoring Rules

- Preserve behavior unless explicitly asked to change it.
- Prefer small, reviewable refactors over broad rewrites.
- Treat all pre-existing modified, untracked, and ignored files as user-owned. Never delete, overwrite, move, clean, or restore them unless the user explicitly authorizes that operation for the exact paths involved, this applies especially to `config/`, backup files, credentials, projects, models, and other user-generated state.
- Do not change public project JSON shape without migration/backward compatibility.
- Keep passive config and project loading permissive for optional feature data. Unknown, removed, renamed, malformed, or out-of-range fields and entries must log a warning, discard only the invalid portion, and continue loading the rest; they must not abort loading or cause an existing config/project to be replaced with an empty or template file. Keep strict validation for live runtime values and explicit write/export boundaries.
- Be careful with Qt signal/thread behavior in `ballontranslator/ui/module_manager.py`.
- Do not rename registered module keys unless compatibility aliases are added.
- Keep model-loading lazy/eager behavior intact.
- Keep module selection lazy/config-only. Data needed by the config UI before module initialization must come from lazy metadata or `SafeEval`-compatible pure helpers, not from `__init__`, `_setup_*`, `update_*`, `flush`, model loading, downloads, or network calls.
- Prefer the app's real construction path over test-only knobs. Do not keep constructor arguments, wrappers, helper functions, or public APIs only because tests use them; tests can patch small instance attributes or call narrower internals when needed.
- Prefer one generic lookup path plus small named boundary helpers over parallel field-specific methods.
- Let the owning module be the integration point. Prefer registering a translator, cache, or helper in the module that owns the feature over threading it through unrelated shared utilities.
- When a simplification removes an indirection, remove the surrounding leftovers in the same pass: stale shared hooks, unused helpers, compatibility shims, redundant wrappers, and tests that only preserve the old shape.
- Avoid adding dependencies unless approved.

## New Feature Rules

- Start with the existing architecture. Prefer extending `ballontranslator/ui/`, `ballontranslator/modules/`, `ballontranslator/utils/config.py`, and `ballontranslator/utils/proj_imgtrans.py` patterns before introducing new frameworks or global services.
- Keep features behind explicit config, UI controls, or module parameters when behavior may surprise existing users.
- Preserve project JSON compatibility. If a feature adds saved fields, provide defaults for old projects and avoid breaking older project files.
- For new automation modules, use the existing registry pattern and stable module keys. Do not rename existing keys without compatibility aliases.
- Keep UI work responsive. Long-running OCR, translation, inpainting, IO, downloads, and model loading must not block the Qt main thread.
- Respect headless mode. If a feature affects the translation pipeline, make sure it works or safely no-ops under `--headless`.
- Avoid mandatory new dependencies. Optional integrations should fail gracefully with a clear error or setup message.
- Keep model/download behavior explicit. Do not download large files or contact online services without an existing module/config path or user action.
- Add focused tests or import checks for non-UI logic. For UI-heavy changes, document the manual verification performed.
- Keep user data safe. Do not overwrite source images, existing translations, masks, or project JSON without following existing save/backup behavior.
- Preserve localization. New visible UI strings should use Qt translation patterns already used in the surrounding code.
- Prefer incremental delivery. Large features should be split into domain/config, pipeline, UI, and persistence changes where practical.

## Maintainability Rules

- Add type hints, including return types, to new or modified Python functions and methods when their types are stable and useful. Prefer concrete domain and Qt types and precise callable signatures; reserve `Any` for genuinely dynamic boundaries, and do not churn unrelated legacy code solely for annotation coverage.
- Shape APIs around the app's current caller chain. A small public helper at the UI boundary is easier to review than a stack of generic helpers, constructor injection, and wrapper layers that no production caller needs.
- Prefer clear ownership names over broad global names. A helper named for the boundary that consumes it is easier to review than a shared API with hidden registration state.
- In code review, check both sides of every simplification: the new direct path should be obvious, and the old path should be gone enough that future readers do not have to understand both.
- Tests should protect behavior and failure modes, not obsolete architecture. After simplifying a feature, adjust tests to cover fallback behavior and real public helpers instead of preserving removed injection or wrapper APIs.
- Do not test incidental widget hierarchy, object names, exact sizes, margins, spacing, stretch factors, or other cosmetic implementation details. Protect observable behavior and failure modes; verify styling with themed visual checks instead.

## Performance Rules

- For startup or UI latency regressions, trace the real caller chain and repeated lifecycle events before optimizing. Check whether a signal path, selection mirror, or config-panel refresh is rebuilding the same widget more than once.
- Keep widget updates incremental. Do not use whole-list rebuilds, config re-deduplication, or blanket row recreation for ordinary edits when a single row/card/summary can be synced in place.
- Keep config sanitation in config/migration code. UI widgets such as LLM profile editors should assume already-valid data, and reserve full rebuilds for explicit reset/restore paths.
- Make selection setters and metadata refreshers idempotent. If the selected module/profile and visible widget are already current, return without rebuilding or re-emitting equivalent work.
- For lookup tables, metadata caches, or derived maps, pre-index by the lookup key and resolve entries lazily. Do not scan the whole dataset for each selected module, profile, row, or widget.
- Keep lookup work O(1) and on demand. Avoid eagerly materializing derived maps, injecting render-only data into shared config, or rebuilding widgets just to refresh unchanged metadata.
- Avoid using broad `blockSignals()` as a substitute for correct update ownership. Prefer precise state-transition updates and signal names that describe their real consumer path.

## UI Styling Rules

- For lightweight tool or settings dialogs that should match the config panel, prefer a frameless `Qt.Dialog` with a transparent outer widget, a rounded theme-token-backed `QFrame` surface, and object-name-scoped stylesheet rules. Keep native titled dialogs for platform-native workflows and do not make confirmations, progress dialogs, or other costly-to-dismiss flows close on an outside click.
- Keep control and list typography consistent with the application UI. Do not assign content fonts to selector rows merely to preview them; use a dedicated preview when the content font itself must be shown.
- Keep config-panel styling scoped. Prefer object names and section-specific selectors such as `ConfigContentScrollContent`, profile-card object names, or spell-check object names over broad `QWidget`, `QLabel`, `QCheckBox`, or `QListWidget` rules that can leak into unrelated panels.
- Use existing theme tokens from `resources/themes.json` and `resources/stylesheet.css` instead of hard-coded colors, except for established project accent values such as `rgb(30, 147, 229)`.
- When swapping or aligning panel colors, treat background ownership explicitly: the left section list, config content panel, cards, labels, titles, inline rows, and item views may each paint their own background. Make labels and title widgets match their local container, and avoid changing push-button colors unless that is specifically requested.
- For config rows that contain buttons or custom widgets, set an object name and `WA_StyledBackground` on the row container when its empty space must match the surrounding panel.
- For checkbox styling, do not add broad `QCheckBox::indicator` rules. Scope normal config checkboxes with object names, and leave icon-based checkboxes such as toolbar, titlebar, alignment, font, and leftbar checkers under their existing rules.
- Remember that `QListWidget` check indicators are item-view indicators, not child `QCheckBox` widgets. Style `QListWidget::indicator`, selected, hover, and disabled item states separately, and verify selected items stay readable in both light and dark themes.
- Match widget structure before fighting fonts or spacing. If two checkbox rows need to align, use the same construction pattern, for example a bare checkbox plus `ParamNameLabel`, rather than mixing `QCheckBox(text=...)` with a separate label.
- Render runtime chevrons and other manually painted SVG pixmaps through `ballontranslator/ui/icon_rendering.py`.
- Do not change a widget's style machinery while handling `QEvent.Polish`, `QEvent.StyleChange`, or `QEvent.Paint`: avoid `setStyle()`, `setStyleSheet()`, explicit `polish()`/`unpolish()`, `ensurePolished()`, and item-delegate replacement in those callbacks. These operations can re-enter Qt's native style code and crash on platform-specific bindings. Prefer scoped stylesheet subcontrols such as `QMenu::right-arrow` and `QComboBox::down-arrow`, construction-time setup, or idempotent setup from safe show/input events before painting.
- For UI-heavy changes, run at least `python -m py_compile` on touched Python files, `git diff --check`, and an offscreen Qt smoke check when practical. State when visual polish still needs a real themed-app pass.

## Qt Event Filter Rules

- Reuse `OutsideClickFramelessMixin` from `ballontranslator.ui.framelesswindow` for lightweight frameless widgets or dialogs that center on their parent, drag from a `title_bar`, close on Escape/outside click, and expose a `close_button`. Put the mixin before the Qt widget base; keep its default `hide()` behavior for panels, override `_dismiss_transient_window()` with `reject()` for dialogs with staged edits, and override `_preserve_on_outside_click()` when owned popups or dialogs must keep the parent open.
- Treat `QApplication` and `QCoreApplication` event filters as global hooks. Install them only while the behavior is active when possible, and remove them on hide, collapse, close, or destroy.
- In app-wide `eventFilter` methods, check cheap relevance first, such as visibility, expected receiver, `isinstance(watched, QWidget)`, or `isinstance(event, QMouseEvent)`, before calling `event.type()`, `globalPosition()`, `globalPos()`, or widget-specific event methods.
- In widget-local filters, guard with the watched object first, for example `if obj is not target: return super().eventFilter(obj, event)`, before reading event details.
- For outside-click handling, prefer widget-target mouse press rules and explicit popup/dialog whitelists over broad geometry or `QWindow` event interpretation.
- For risky app-wide filter changes, add an offscreen Qt regression that sends an irrelevant watched object or non-mouse event and proves the filter ignores it before requesting `event.type()`.

## Qt Signal and QObject Lifetime Rules

- Do not connect a child or transient QObject signal to a lambda, nested function, or `functools.partial` that captures its parent, another widget, or another transient QObject. Qt parent ownership destroys the native children, but the binding may retain the Python callback and invalid wrappers after native deletion.
- Prefer a direct bound QObject slot. For repeated menu actions or rows, store small values with `QAction.setData()` or `QObject.setProperty()` and read them through `sender()`, or expose a typed row signal and connect it to the owner's bound method.
- A lambda that captures no QObject is not automatically a leak; do not replace fixed-lifetime or pure-Python callbacks without a demonstrated lifecycle problem.
- Treat `close()` and `hide()` as visibility changes, not deletion. For one-shot modal dialogs, read required state after `exec_()` and call `deleteLater()` in `finally`; preserve intentional cached panels.
- For lifecycle fixes, add a focused offscreen regression that processes deferred deletes and verifies behavior plus wrapper/registry release, and run it under both PyQt5 and PyQt6 when binding behavior is relevant.

## Code Comment Rules
- Include a standard Python >>> doctest snippet in the docstring of core classes and complex functions.
- Add the minimum comments needed to make code review efficient.
- Comment non-obvious intent, invariants, compatibility constraints, and failure modes.
- For Qt threading, signals, model loading, project JSON compatibility, and file IO, add short comments when the ordering or side effect is important.
- Prefer comments that explain why code is structured a certain way, not what each line does.
- Do not add boilerplate comments, redundant docstrings, or comments that merely repeat function or variable names.
- When refactoring complex logic, add a brief comment before the extracted block if it preserves a subtle behavior from the old implementation.

## Done Criteria For Features

- Existing workflows still run.
- New behavior is configurable or clearly discoverable.
- Old projects load without errors.
- Relevant checks were run, or limitations are stated.

## Verification

For narrow Python changes:
- Run targeted import checks where possible.
- Run relevant tests if available.
- For UI/threading changes, explain what was not practically verified.

Use `rg` for repo search.

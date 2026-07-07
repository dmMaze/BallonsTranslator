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

## Refactoring Rules

- Preserve behavior unless explicitly asked to change it.
- Prefer small, reviewable refactors over broad rewrites.
- Do not change public project JSON shape without migration/backward compatibility.
- Be careful with Qt signal/thread behavior in `ballontranslator/ui/module_manager.py`.
- Do not rename registered module keys unless compatibility aliases are added.
- Keep model-loading lazy/eager behavior intact.
- Keep module selection lazy/config-only. Data needed by the config UI before module initialization must come from lazy metadata or `SafeEval`-compatible pure helpers, not from `__init__`, `_setup_*`, `update_*`, `flush`, model loading, downloads, or network calls.
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

## UI Styling Rules

- Keep config-panel styling scoped. Prefer object names and section-specific selectors such as `ConfigContentScrollContent`, profile-card object names, or spell-check object names over broad `QWidget`, `QLabel`, `QCheckBox`, or `QListWidget` rules that can leak into unrelated panels.
- Use existing theme tokens from `resources/themes.json` and `resources/stylesheet.css` instead of hard-coded colors, except for established project accent values such as `rgb(30, 147, 229)`.
- When swapping or aligning panel colors, treat background ownership explicitly: the left section list, config content panel, cards, labels, titles, inline rows, and item views may each paint their own background. Make labels and title widgets match their local container, and avoid changing push-button colors unless that is specifically requested.
- For config rows that contain buttons or custom widgets, set an object name and `WA_StyledBackground` on the row container when its empty space must match the surrounding panel.
- For checkbox styling, do not add broad `QCheckBox::indicator` rules. Scope normal config checkboxes with object names, and leave icon-based checkboxes such as toolbar, titlebar, alignment, font, and leftbar checkers under their existing rules.
- Remember that `QListWidget` check indicators are item-view indicators, not child `QCheckBox` widgets. Style `QListWidget::indicator`, selected, hover, and disabled item states separately, and verify selected items stay readable in both light and dark themes.
- Match widget structure before fighting fonts or spacing. If two checkbox rows need to align, use the same construction pattern, for example a bare checkbox plus `ParamNameLabel`, rather than mixing `QCheckBox(text=...)` with a separate label.
- For UI-heavy changes, run at least `python -m py_compile` on touched Python files, `git diff --check`, and an offscreen Qt smoke check when practical. State when visual polish still needs a real themed-app pass.

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

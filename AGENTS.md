# AGENTS.md

## Project Shape

BallonsTranslator is a PyQt/qtpy desktop app for comic image translation.

Important areas:
- `launch.py`: startup, dependency checks, Qt setup, headless mode.
- `ui/`: Qt UI, canvas, panels, module manager, worker threads.
- `modules/`: pluggable detector/OCR/translator/inpainter implementations.
- `utils/proj_imgtrans.py`: project persistence and image/textblock state.
- `utils/textblock.py`: central TextBlock domain object.
- `utils/config.py`: persistent config and module settings.

## Refactoring Rules

- Preserve behavior unless explicitly asked to change it.
- Prefer small, reviewable refactors over broad rewrites.
- Do not change public project JSON shape without migration/backward compatibility.
- Be careful with Qt signal/thread behavior in `ui/module_manager.py`.
- Do not rename registered module keys unless compatibility aliases are added.
- Keep model-loading lazy/eager behavior intact.
- Avoid adding dependencies unless approved.

## New Feature Rules

- Start with the existing architecture. Prefer extending `ui/`, `modules/`, `utils/config.py`, and `utils/proj_imgtrans.py` patterns before introducing new frameworks or global services.
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

## New Feature Rules

- Start with the existing architecture. Prefer extending `ui/`, `modules/`, `utils/config.py`, and `utils/proj_imgtrans.py` patterns before introducing new frameworks or global services.
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

## Verification

For narrow Python changes:
- Run targeted import checks where possible.
- Run relevant tests if available.
- For UI/threading changes, explain what was not practically verified.

Use `rg` for repo search.
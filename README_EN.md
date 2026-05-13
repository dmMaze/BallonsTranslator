> [!IMPORTANT]
> If you share translated pages publicly and no experienced human translator fully translated or proofread them, label the result clearly as machine translation.

> [!NOTE]
> This fork contains machine-translated content in two places: translated comic text produced by the app, and documentation that was machine-translated or normalized for this fork. Portions of this README were consolidated from machine-translated project materials and then edited for consistency.

# BallonsTranslator Vibe Fork
English | [README mirror](/README.md) | [pt-BR](doc/README_PT-BR.md) | [Russian](doc/README_RU.md) | [Japanese](doc/README_JA.md) | [Indonesian](doc/README_ID.md) | [Vietnamese](doc/README_VI.md) | [Korean](doc/README_KO.md) | [Spanish](doc/README_ES.md) | [French](doc/README_FR.md)

Fork release: `1.4.0-vibe.1`  
Upstream base: `BallonsTranslator 1.4.0`  
Update source: `https://github.com/CoSciBlog/BallonsTranslator-vibe.git` (`dev`)

BallonsTranslator is a desktop tool for comic and manga translation with OCR, text detection, inpainting, translation, and interactive text editing.

This repository is a Codex-expanded fork. It keeps the original desktop workflow, adds Pinokio launcher integration in the project root, documents reproducible local startup paths, and makes the fork identity explicit in the runtime version string.

## What changed in this fork

- Added Pinokio launcher scripts in the project root: `install.js`, `start.js`, `update.js`, `reset.js`, `pinokio.js`, and `pinokio.json`.
- Switched the launcher and Windows helper flow to a shared project virtual environment at `./env` instead of the old bundled `ballontrans_pylibs_win` runtime.
- Pointed the built-in update flow at the `CoSciBlog/BallonsTranslator-vibe` fork on the `dev` branch.
- Introduced a fork-aware application version scheme so this build is distinguishable from upstream releases: `1.4.0-vibe.1`.
- Replaced the mixed-language root `README.md` with the English documentation and refreshed the English README for this fork.
- Documented that translated output and some documentation assets are machine-translated and should be disclosed as such when redistributed.

## Features

- Fully automated translation workflow:
  - automatic text detection
  - OCR
  - inpainting / text removal
  - machine translation
  - automatic typesetting based on the original balloon layout
- Interactive editing workflow:
  - rich text editing
  - search and replace
  - text style presets
  - import and export for Word documents
- Image editing workflow:
  - mask editing
  - inpainting brush style cleanup
  - support for long-strip and webtoon-style pages
- Headless automation for batch processing from the command line
- Multiple OCR, translator, and inpainting backends already wired into the desktop app

## Pinokio launcher

This fork includes Pinokio launcher scripts in the project root.

```bash
# Install dependencies into the project venv at ./env
install.js

# Start BallonsTranslator through the same venv
start.js

# Update from the BallonsTranslator-vibe fork, then refresh dependencies
update.js

# Remove the project venv so it can be recreated
reset.js
```

The launcher update flow tracks `https://github.com/CoSciBlog/BallonsTranslator-vibe.git` on the `dev` branch. The Windows batch launchers also create and reuse the same `env` virtual environment instead of the old bundled `ballontrans_pylibs_win` runtime.

## Programmatic use

BallonsTranslator does not expose a built-in HTTP API, so `curl` is not applicable unless you place a wrapper service in front of it. The supported automation path in this fork is the headless CLI entry point.

### CLI

```bash
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]"
```

### Python

```python
import subprocess

subprocess.run([
    "python",
    "launch.py",
    "--headless",
    "--exec_dirs",
    "[DIR_1],[DIR_2]",
], check=True)
```

### JavaScript

```javascript
const { spawnSync } = require("node:child_process");

const result = spawnSync("python", [
  "launch.py",
  "--headless",
  "--exec_dirs",
  "[DIR_1],[DIR_2]",
], { stdio: "inherit" });

process.exit(result.status ?? 1);
```

### Curl

```bash
# No built-in HTTP API is exposed by this desktop app.
# Use the headless CLI directly, or add your own wrapper server first.
```

## Installation

### Windows packaged path

If you do not want to install Python and Git manually and you have normal Internet access:

- Download `BallonsTranslator_dev_src_with_gitpython.7z` from the upstream distribution links.
- Extract it.
- Run `launch_win.bat`.

The provided packages do not run on Windows 7. Windows 7 users need to install [Python 3.8](https://www.python.org/downloads/release/python-3810/) and run the source code directly.

### Run from source

Install [Python](https://www.python.org/downloads/release/python-31011) `<= 3.12` and [Git](https://git-scm.com/downloads).

```bash
# Clone this fork
git clone https://github.com/CoSciBlog/BallonsTranslator-vibe.git
cd BallonsTranslator-vibe
git checkout dev

# Launch the app
python launch.py

# Update from the fork
python launch.py --update
```

The first launch installs Python dependencies and downloads required models automatically. If model downloads fail, download the missing `data` assets manually and place them in the expected paths inside the project.

### Pinokio path

Use the launcher files from the project root when you want a one-click install and run flow inside Pinokio:

1. Run `install.js`
2. Run `start.js`
3. Use `update.js` for fork updates
4. Use `reset.js` to recreate the environment from scratch

## Usage

Recommended flow:

1. Start the application from a terminal so crashes still print useful information.
2. Open settings and choose the translator, source language, and target language.
3. Open a folder that contains comic or manga images.
4. Click `Run` and wait for detection, OCR, translation, inpainting, and typesetting to finish.
5. Review the translated text manually before publishing or sharing.

The app estimates font size, color, outline, angle, direction, and alignment from the source page. You can override those defaults with global or per-block formatting controls.

## Headless mode

```bash
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]"
```

Notes:

- Runtime configuration is loaded from `config/config.json`.
- If the rendered font size is off, specify logical DPI manually with `--ldpi`.
- Headless mode is the supported automation path for batch translation in this fork.

## Machine translation policy

- Translated comic text can be machine translation.
- The README and some fork-maintained documentation were machine-translated or machine-assisted before manual cleanup.
- Do not present machine-translated output as human translation unless a qualified translator reviewed it fully.

## Credits

- Upstream project: [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)
- AI-modified downstream variant referenced by the project: [thomaswantstobeaskeleton/BallonsTranslator-Pro](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro)
- Fork maintenance and launcher/documentation extension: this `BallonsTranslator-vibe` fork

## License

See [LICENSE](LICENSE).

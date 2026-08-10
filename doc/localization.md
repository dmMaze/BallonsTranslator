# Add Your Language to BallonsTranslator (Windows)

You do not need to be a programmer. The app uses
[Qt Linguist](https://doc.qt.io/qtforpython-6/tools/pyside-linguist.html), a
visual tool where you read an English phrase and type its translation.

## 1. Prepare the translation tools

This guide assumes you are using the extracted
[`Ballonstranslator_win_minium.zip`](https://github.com/dmMaze/BallonsTranslator/releases/download/v1.5.10/Ballonstranslator_win_minium.zip)
package and its bundled Python environment is ready.

Open the package's main folder, right-click an empty area, and choose **Open in
Terminal**. In PowerShell, install the Qt translation tools once:

```powershell
.\ballontrans_pylibs_win\python.exe -m pip install PySide6-Essentials
$env:Path = "$PWD\ballontrans_pylibs_win\Scripts;$env:Path"
$env:QT_API = "pyqt6"
```

The same workflow can be used with a source checkout on Windows, macOS, or
Linux when Python is already installed. Use that environment's `python` and
PySide6 tool paths in place of the bundled Windows paths shown below.

## 2. Create and translate the language file

Choose a locale code in `language_REGION` form, such as `de_DE`, `ja_JP`, or
`pl_PL`. Replace `de_DE` in these examples with yours:

```powershell
.\ballontrans_pylibs_win\python.exe .\scripts\update_translation.py de_DE
.\ballontrans_pylibs_win\Scripts\pyside6-linguist.exe .\resources\translate\de_DE.ts
```

In Qt Linguist, select each English source phrase, enter its translation, mark
it complete, and save. You can close the program and continue later.

- Keep placeholders such as `{count}`, `%1`, and `%n` unchanged.
- Preserve HTML tags, line breaks, and keyboard shortcuts.
- Translate for meaning and for the available UI space, not word for word.

Compile the finished catalog so the app can use it:

```powershell
.\ballontrans_pylibs_win\Scripts\pyside6-lrelease.exe .\resources\translate\de_DE.ts -qm .\resources\translate\de_DE.qm
```

For a friendly name in the language menu, add one line inside
`DISPLAY_LANGUAGE_MAP` in `ballontranslator\utils\shared.py`, using the
language's native name:

```python
"Deutsch": "de_DE",
```

Without this line, the language still works but appears as its code, such as
`de_DE`.

## 3. Check and submit it

Start the app with `launch_win.bat`. Choose your language under **View > Display
Language**, close the app, and open it again. Check the main window, Settings,
Run dialog, and module settings for clipped text, incorrect wording, or English
phrases that you missed.

Submit these changes in a pull request to the `dev` branch (or send them to a
maintainer):

- `resources\translate\de_DE.ts`
- `resources\translate\de_DE.qm`
- `ballontranslator\utils\shared.py`, if you added the friendly name

## Using an agent

A coding agent can install the tools, generate the catalog, add the language
name, suggest translations, compile the `.qm` file, and prepare a pull request.
Tell it the locale code and native language name. **Always proofread every
machine-generated translation yourself**: agents can mistranslate UI context,
alter placeholders, or choose wording that does not fit the interface.

When the English UI changes later, rerun `update_translation.py`; your existing
translations are kept and new or changed phrases are added for review.

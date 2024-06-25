# Changelogs

### 2023-04-15
Src download implementation based on gallery-dl (#131) thanks to [ROKOLYT](https://github.com/ROKOLYT)

### 2023-02-27
[v1.3.34](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.34) released
1. fix incorrect orientation assignment for CHT  (#96)
2. convert CHS to CHT if it is required for Caiyun & DeepL (#100)
3. support for webp (#85)

### 2023-02-23
[v1.3.30](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.30) released
1. Migrate to PyQt6 for better text rendering preview and [compatibility](https://github.com/Nuitka/Nuitka/issues/251) with nuitka
2. Support set transparency of text layer (#88)
3. Dump logs to data/logs

### 2023-01-27
[v1.3.26](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.26) released
1. Add support for [saladict](https://saladict.crimx.com) (*All-in-one professional pop-up dictionary and page translator*) in the mini menu on text selection. [Installation guide](doc/saladict.md) 
<img src = "./src/saladict_doc.jpg">

2. Support keyword substitution for OCR & machine translation results [#78](https://github.com/dmMaze/BallonsTranslator/issues/78): Edit -> ```Keyword substitution for machine translation```
3. Support import folder with drag&drop [#77](https://github.com/dmMaze/BallonsTranslator/issues/77)
4. Hide control blocks on start text editing. [#81](https://github.com/dmMaze/BallonsTranslator/issues/81)
5. Bugfix

### 2023-01-08
[v1.3.22](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.22) released
1. Support delete and restore removed text
2. Support reset angle
3. Bugfixes

### 2022-12-31
[v1.3.20](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.20) released
1. Adapted to images with extreme aspect ratio such as webtoons
2. Support paste text to multiple selected Text blocks.
3. Bugfixes
4. OCR/Translate/Inpaint selected text blocks
   lettering style will inherit from corresponding selected block.
   ctc_48px is more recommended for single line text, mangocr for multi-line Japanese, need to retrain detection model make ctc48_px generalize to multi-lines  
   Note that if you use **ctc_48px** make sure that the box is in vertical mode and fits as close to the single line of text as possible
<img src="./src/ocrselected.gif" div align=center>

### 2022-11-29
[v1.3.15](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.15) released
1. Bugfixes
2. Optimize saving logic
3. The shape of Pen/Inpaint tool can be set to rectangle (experimental)

### 2022-10-25
[v1.3.14](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.14) released
1. Bugfixes

### 2022-09-30
Support Dark Mode since v1.3.13: View->Dark Mode

### 2022-09-24
[v1.3.12](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.12) released

1. Support global Search(Ctrl+G) and search current page(Ctrl+F). 
2. Local redo stack of each texteditor are merged into a main text-edit stack, text-edit stack is split from drawing board's now. 
3. Word doc import/export bugfixes
4. Frameless window rework based on https://github.com/zhiyiYo/PyQt-Frameless-Window

### 2022-09-13
[v1.3.8](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.8) released

1. Pen tool bug fixes & optimization
2. Fix scaling
3. Support making font style presets, text graphical effects(shadow & opacity), see https://github.com/dmMaze/BallonsTranslator/pull/38
4. Support word document(*.docx) import/export: https://github.com/dmMaze/BallonsTranslator/pull/40

### 2022-08-31
[v1.3.4](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.4) released

1. Add Sugoi Translator(Japanese-English only, created & authorized by [mingshiba](https://www.patreon.com/mingshiba)): download the [model](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm) converted by [@Snowad14](https://github.com/Snowad14) and put "sugoi_translator" in the "data" folder.
2. Add support for russian, thanks to [bropines](https://github.com/bropines)
3. Support letter spacing adjustment.
4. Vertical type rework & text rendering bug fixes: https://github.com/dmMaze/BallonsTranslator/pull/30

### 2022-08-17
[v1.3.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.3.0) released


1. Fix deepl translator, thanks to [@Snowad14](https://github.com/Snowad14)
2. Fix font size & stroke bug which makes text unreadable
3. Support **global font format** (determine the font format settings used by auto-translation mode): in config panel->Typesetting, change the corresponding option from "decide by the program" to "use global setting" to enable. Note global settings are those formats shown by the right font format panel when you are not editing any textblock in the scene.
4. Add **new inpainting model**: lama-mpe and set it as default.
5. Support multiple textblocks selection & formatting. 
6. Improved manga->English, English->Chinese typesetting (**Auto-layout** in Config panel->Typesetting, enabled by default), it can also be applied to selected text blocks use the option in the right-click menu.

<img src="./src/multisel_autolayout.gif" div align=center>
<p align=center>
batch text formatting & auto layout
</p>

### 2022-05-19
[v1.2.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.2.0) released

1. Support DeepL, thanks to [@Snowad14](https://github.com/Snowad14)
2. Add new ocr model from manga-image-translator, support korean recognition
3. Bugfixes

### 2022-04-17

[v1.1.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.1.0) released
1. use qthread to write edited images to avoid freezing when turning pages.
2. optimized inpainting policy
3. add rect tool
4. More shortcuts
5. Bugfixes 

### 2022-04-09

1. v1.0.0  released
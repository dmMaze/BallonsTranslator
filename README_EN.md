<p align="center">
  <img
    width="256"
    alt="Spinning fox animation"
    src="https://github.com/user-attachments/assets/fe44e9a6-c7da-4bc5-8421-87fd6c38a0ba"
  />
</p>

<h1 align="center">BallonsTranslator</h1>

<p align="center">Yet another computer-aided comic/manga translation tool powered by deep learning.</p>

<p align="center">
  <a href="/README.md">简体中文</a> | English | <a href="/doc/README_RU.md">Русский</a> | <a href="/doc/README_JA.md">日本語</a> | <a href="/doc/README_ES.md">Español</a> | <a href="/doc/README_FR.md">Français</a> | <a href="/doc/README_PT-BR.md">pt-BR</a> | <a href="/doc/README_KO.md">한국어</a> | <a href="/doc/README_ID.md">Indonesia</a> | <a href="/doc/README_VI.md">Tiếng Việt</a>
</p>

# Features
> [!IMPORTANT]
> **If you're sharing the translated result publicly and no experienced human translator participated in a throughout translating or proofreading, please mark it as machine translation somewhere clear to see.**

* Fully automated translation  
  - Support automatic text-detection, recognition, removal, and translation. Overall performance is dependent upon these modules.
  - Typesetting is based on the formatting estimation of the original text.
  - Works decently with manga and comics.
  - Improved manga->English, English->Chinese typesetting (based on the extraction of balloon regions.).
  
* Image editing  
  - Support mask editing & inpainting (something like spot healing brush tool in PS) 
  - Adapted to images with extreme aspect ratio such as webtoons
  
* Text editing  
  - Support rich text formatting and [text style presets](https://github.com/dmMaze/BallonsTranslator/pull/311), translated texts can be edited interactively.
  - [Text transforms](https://github.com/dmMaze/BallonsTranslator/pull/1238), Search & replace
  - Support export/import to/from word documents

* <details>
  <summary><i>Context-aware LLM translation & Glossary</i></summary>

  **Translation history**

  - Set **LLM Context** to **+history** to show `LLMTranslator` examples from earlier completed pages. This can keep names, terminology, and tone more consistent. Continue and selected-range runs can also use eligible earlier pages.
  - **Token budget** controls how much earlier translated text is included. Newer pages are kept first. The current page, instructions, glossary, and generated reply need additional space. The default is `4096`.
  - A larger budget gives the model more story context and drops old pages less often, but sends more input and may take longer. Local models may also need substantially more RAM/VRAM. The `4096` default is deliberately conservative; mainstream providers with large context windows, such as DeepSeek, can often use a higher limit. About 70% of the model's context limit is a reasonable upper bound (`90000` for a 128K model).
  - The history budget also affects prompt caching. While history grows within the budget, consecutive requests keep the same beginning; OpenAI and DeepSeek can reuse these input tokens at a discount and may respond faster. Dropping old pages changes the beginning and resets the cache. A larger budget means fewer resets but sends more history, so it is not guaranteed to cost less.

  The table below is a rough manga-page example using DeepSeek, where cached input tokens cost 10% of regular input tokens. Actual results vary by project, model, and provider.

  | Token budget | Estimated history kept (pages) | Estimated total cost vs. no history |
  |---:|---:|---:|
  | `2048` | 3–4 | 1.65× |
  | `4096` | 6–9 | 1.79× |
  | `8192` | 12–19 | 2.10× |
  | `16384` | 23–38 | 2.66× |

  **Reusable glossaries**

  - Set **Glossary File** in the Run dialog to a UTF-8 `.json`, `.txt`, or `.tsv` file. The file is read-only and can be reused across projects.
  - **Matching** sends only entries whose source terms occur on the relevant page. **All** sends every entry and may use considerably more tokens.
  - Supported formats include:

    ```text
    # Sakura-style text
    source->translation # optional note

    # Tab-separated text
    source<TAB>translation<TAB>optional note
    ```

    ```json
    [
      {"src": "source", "dst": "translation", "info": "optional note"}
    ]
    ```

  - Matching is case-insensitive and literal. Conflicting entries, malformed files, unsupported formats, and missing files stop the translation before an LLM request is sent.
  - Prior-page context and glossary injection affect only `LLMTranslator`; other translators ignore these settings.

  </details>

# Installation

## On Windows

**Method A (One-Click Local Environment Setup, requires PowerShell)**:
The script installs `BallonsTranslator` in the directory where you run it:
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
Or run the following command in the Command Prompt (`cmd.exe`):
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```


**Method B (Download Pre-configured Package)**:
Download `Ballonstranslator_win_minium.zip` from [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases), extract it, and double-click `launch_win.bat` to launch the application.

These methods do not support Windows 7; Windows 7 users must install [Python 3.8](https://www.python.org/downloads/release/python-3810/) manually and run from source.

If you see errors involving `msvcp140.dll`, `c10.dll`, or `[WinError 1114]`, install or update the [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022; [official download notes](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).

## macOS / Linux

The script installs `BallonsTranslator` in the directory where you run it:
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

If `curl` is not available, download the script with `wget -O ...` instead. The app launches automatically after installation; later, use `cd BallonsTranslator && ./launch.sh` to start it again.

The app checks core dependencies at startup. When you select a module that needs extra libraries, the app will prompt you to install the missing optional dependencies (you can also enable automatic installation in Settings).


# Usage

**It is recommended to run the program in a terminal in case it crashed and left no information, see the following gif.**
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">
- The first time you run the application, please select the translator and set the source and target languages by clicking the settings icon.
- Open a folder containing images of a comic (manga/manhua/manhwa) that need translation by clicking the folder icon.
- Click the `Run` button and wait for the process to complete.

The font formats such as font size and color are determined by the program automatically in this process, you can predetermine those formats by change corresponding options from "decide by program" to "use global setting" in the config panel->Typesetting. (global settings are those formats shown by the right font format panel when you are not editing any textblock in the scene)
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">

## Image Editing

### Inpaint Tool
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
Image Editing Mode, Inpainting Tool
</p>

### rect tool
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
Rect Tool
</p>

To 'erase' unwanted inpainted results, use the inpainting tool or rect tool with your **right button** pressed.  
The result depends on how accurately the algorithm ("method 1" and "method 2" in the gif) extracts the text mask. It could perform worse on complex text & background.  

## Text editing
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
Text Editing Mode
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
Batch Text Formatting & Auto Layout
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
<p align=center>
OCR & Translate Selected Area
</p>

## Shortcuts
* ```A```/```D``` or ```pageUp```/```Down``` to turn the page
* ```Ctrl+Z```, ```Ctrl+Shift+Z``` to undo/redo most operations. (note the undo stack will be cleared after you turn the page)
* ```T``` to text-editting mode (or the "T" button on the bottom toolbar).
* ```W``` to activate text block creating mode, then drag the mouse on the canvas with the right button clicked to add a new text block. (see the text editing gif)
* ```P``` to image-editting mode.  
* In the image editing mode, use the slider on the right bottom to control the original image transparency.
* Disable or enable any automatic modules via titlebar->run, run with all modules disabled will re-letter and re-render all text according to corresponding settings.  
* Set parameters of automatic modules in the config panel.  
* ```Ctrl++```/```Ctrl+-``` (Also ```Ctrl+Shift+=```) to resize image.
* ```Ctrl+G```/```Ctrl+F``` to search globally/in current page.
* ```0-9``` to adjust opacity of text layer
* For text editing: bold - ```Ctrl+B```, underline - ```Ctrl+U```, Italics - ```Ctrl+I``` 
* Set text shadow and transparency in the text style panel -> Effect.  
* ```Alt+Arrow Keys``` or ```Alt+WASD``` (```pageDown``` or ```pageUp``` while in text editing mode) to switch between text blocks.
  
<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">

## Headless mode (Run without GUI)
``` python
python -m ballontranslator --headless --exec_dirs "[DIR_1],[DIR_2]..."
```
Note the configuration (source language, target language, inpaint model, etc) will load from config/config.json.  
If the rendered font size is not right, specify logical DPI manually via ```--ldpi ```, typical values are 96 and 72.


# Automation modules
This project is heavily dependent upon [manga-image-translator](https://github.com/zyddnys/manga-image-translator), online service and model training is not cheap, please consider to donate the project:  
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

[Sugoi translator](https://sugoitranslator.com/) is created by [mingshiba](https://www.patreon.com/mingshiba).
  
## Text detection
 * Support English and Japanese text detection, training code and more details can be found at [comic-text-detector](https://github.com/dmMaze/comic-text-detector)
 * Support using text detection from [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Username and password need to be filled in, and automatic login will be performed each time the program is launched.

   * For detailed instructions, see **Tuanzi OCR Instructions**: ([Chinese](doc/团子OCR说明.md) & [Brazilian Portuguese](doc/Manual_TuanziOCR_pt-BR.md) only)
 
 * `YSGDetector` models are trained by [lhj5426](https://github.com/lhj5426), these models would filter out onomatopoeia in CGs/Manga, download checkpoints from [YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector) and put into `data/models`. 


## OCR
 * All mit* models are from manga-image-translator, support English, Japanese and Korean recognition and text color extraction.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) is from [kha-white](https://github.com/kha-white), text recognition for Japanese, with the main focus being Japanese manga.
 * [PaddleOCRVLManga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga) finetuned on Japanese manga
 * Support using OCR from [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Username and password need to be filled in, and automatic login will be performed each time the program is launched.
   * The current implementation uses OCR on each textblock individually, resulting in slower speed and no significant improvement in accuracy. It is not recommended. If needed, please use the Tuanzi Detector instead.
   * When using the Tuanzi Detector for text detection, it is recommended to set OCR to none_ocr to directly read the text, saving time and reducing the number of requests.
   * For detailed instructions, see **Tuanzi OCR Instructions**: ([Chinese](doc/团子OCR说明.md) & [Brazilian Portuguese](doc/Manual_TuanziOCR_pt-BR.md) only)
* Added as an "optional" PaddleOCR module. In Debug mode you will see a message stating that it is not there. You can simply install it by following the instructions described there. If you don’t want to install the package yourself, just uncomment (remove the `#`) the lines with paddlepaddle(gpu) and paddleocr. Bet everything at your own peril andrisk. For me (bropines) and two testers, everything was installed fine, you may have an error. Write about it in issue and tag me.
* Added [OneOCR](https://github.com/b1tg/win11-oneocr). Local WINDOWS model taken from SnippingTOOL or Win.PHOTOS applications. To use it, you need to place the model and DLL files in the 'data/models/one-ocr' folder. Before running, it is better to throw the files at once. Read how to find and get DLL and model files here: https://github.com/dmMaze/BallonsTranslator/discussions/859#discussioncomment-12876757 . Thanks AuroraWright for the project [OneOCR](https://github.com/AuroraWright/oneocr)
 * OCR setting: Font recognition. Download the [Font Recognition Model (YuzuMarker.FontDetection)](https://github.com/JeffersonQin/YuzuMarker.FontDetection) and place it in the data\models\YuzuMarker.FontDetection directory.
  The three required files are: `data\models\YuzuMarker.FontDetection\font_dataset`, `data\models\YuzuMarker.FontDetection\name=4x-epoch=18-step=368676.ckpt`, and `data\font_demo_cache.bin`
  Font names with a recognition confidence rate greater than 60% will be saved in the `_detected_font_name` field of the JSON file. Currently, no visual display is provided. When exporting LabelPlus txt using the script [scripts/BTjson_to_LPtxt.pyw], you can optionally include font and font size information for importing into other software (such as Photoshop/InDesign) for text embedding.

## Inpainting
  * AOT is from [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
  * All lama* are finetuned using [LaMa](https://github.com/advimman/lama)
  * PatchMatch is an algorithm from [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), this program uses a [modified version](https://github.com/dmMaze/PyPatchMatchInpaint) by me. 
  
## Translators
* **You can find information about Translators modules [here.](doc/modules/translators.md)**

## FAQ & Misc
* If your computer has an Nvidia GPU or Apple silicon, the program will enable hardware acceleration. 
* Accelarate performance if you have a [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) or [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html) device as most modules uses [PyTorch](https://pytorch.org/get-started/locally/).
* Fonts are from your system's fonts.
* Thanks to [bropines](https://github.com/bropines) for the Russian localization.
* Added Export to photoshop JSX script by [bropines](https://github.com/bropines). </br> To read the instructions, improve the code and just poke around to see how it works, you can go to `scripts/export to photoshop` -> `install_manual.md`.

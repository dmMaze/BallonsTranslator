> [!IMPORTANT]  
> **If you're sharing the translated result publicly and no experienced human translator participated in a throughout translating or proofreading, please mark it as machine translation somewhere clear to see.**

# BallonTranslator
[简体中文](/README.md) | English | [pt-BR](doc/README_PT-BR.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md) | [Tiếng Việt](doc/README_VI.md) | [한국어](doc/README_KO.md) | [Español](doc/README_ES.md) | [Français](doc/README_FR.md)

Yet another computer-aided comic/manga translation tool powered by deep learning.

<img src="doc/src/ui0.jpg" div align=center>

<p align=center>
preview
</p>

# Features
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
  - Support search & replace
  - Support export/import to/from word documents

# Installation

## On Windows
If you don't want to install Python and Git by yourself and have access to the Internet:  
Download BallonsTranslator_dev_src_with_gitpython.7z from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing), unzip it and run launch_win.bat.   
Run scripts/local_gitpull.bat to get the latest update.
Note these provided packages cannot run on Windows 7, Win 7 users need to install [Python 3.8](https://www.python.org/downloads/release/python-3810/) and run the source code.

## Run the source code

Install [Python](https://www.python.org/downloads/release/python-31011) **<= 3.12** (dont use the one installed from microsoft store) and [Git](https://git-scm.com/downloads).

```bash
# Clone this repo
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# Launch app
$ python3 launch.py

# Update app
$ python3 launch.py --update
```

Note the first time you launch it will install the required libraries and download models automatically. If the downloads fail, you will need to download the **data** folder (or missing files mentioned in the terminal) from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) and save it to the corresponding path in source code folder.

## Build macOS application (compatible with both intel and apple silicon chips)
[Reference](doc/macOS_app.md)  
Some issues may occur, running the source code directly is the recommended way for now.

# Usage

**It is recommended to run the program in a terminal in case it crashed and left no information, see the following gif.**
<img src="doc/src/run.gif">  
- The first time you run the application, please select the translator and set the source and target languages by clicking the settings icon.
- Open a folder containing images of a comic (manga/manhua/manhwa) that need translation by clicking the folder icon.
- Click the `Run` button and wait for the process to complete.

The font formats such as font size and color are determined by the program automatically in this process, you can predetermine those formats by change corresponding options from "decide by program" to "use global setting" in the config panel->Typesetting. (global settings are those formats shown by the right font format panel when you are not editing any textblock in the scene)

## Image Editing

### Inpaint Tool
<img src="doc/src/imgedit_inpaint.gif">
<p align = "center">
Image Editing Mode, Inpainting Tool
</p>

### rect tool
<img src="doc/src/rect_tool.gif">
<p align = "center">
Rect Tool
</p>

To 'erase' unwanted inpainted results, use the inpainting tool or rect tool with your **right button** pressed.  
The result depends on how accurately the algorithm ("method 1" and "method 2" in the gif) extracts the text mask. It could perform worse on complex text & background.  

## Text editing
<img src="doc/src/textedit.gif">
<p align = "center">
Text Editing Mode
</p>

<img src="doc/src/multisel_autolayout.gif" div align=center>
<p align=center>
Batch Text Formatting & Auto Layout
</p>

<img src="doc/src/ocrselected.gif" div align=center>
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
  
<img src="doc/src/configpanel.png">

## Headless mode (Run without GUI)
``` python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
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
* Add support for [saladict](https://saladict.crimx.com) (*All-in-one professional pop-up dictionary and page translator*) in the mini menu on text selection. [Installation guide](doc/saladict.md)
* Accelarate performance if you have a [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) or [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html) device as most modules uses [PyTorch](https://pytorch.org/get-started/locally/).
* Fonts are from your system's fonts.
* Thanks to [bropines](https://github.com/bropines) for the Russian localization.
* Added Export to photoshop JSX script by [bropines](https://github.com/bropines). </br> To read the instructions, improve the code and just poke around to see how it works, you can go to `scripts/export to photoshop` -> `install_manual.md`.

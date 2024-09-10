> [!IMPORTANT]  
> **If you're sharing the translated result publicly and no experienced human translator participated in a throughout translating or proofreading, please mark it as machine translation somewhere clear to see.**

# BallonTranslator
[简体中文](/README.md) | English | [pt-BR](doc/README_PT-BR.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md) | [Tiếng Việt](doc/README_VI.md)

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

## Run the source code

Install [Python](https://www.python.org/downloads/release/python-31011) **< 3.12** (dont use the one installed from microsoft store) and [Git](https://git-scm.com/downloads).

```bash
# Clone this repo
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# Launch the app
$ python3 launch.py
```

Note the first time you launch it will install the required libraries and download models automatically. If the downloads fail, you will need to download the **data** folder (or missing files mentioned in the terminal) from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) and save it to the corresponding path in source code folder.

## Build macOS application (compatible with both intel and apple silicon chips)
<i>Note macOS can also run the source code if it didn't work.</i>  

![录屏2023-09-11 14 26 49](https://github.com/hyrulelinks/BallonsTranslator/assets/134026642/647c0fa0-ed37-49d6-bbf4-8a8697bc873e)

#### 1. Preparation
-   Download libs and models from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw "MEGA") or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)


<img width="1268" alt="截屏2023-09-08 13 44 55_7g32SMgxIf" src="https://github.com/dmMaze/BallonsTranslator/assets/134026642/40fbb9b8-a788-4a6e-8e69-0248abaee21a">

-  Put all the downloaded resources into a folder called data, the final directory tree structure should look like:

```
data
├── libs
│   └── patchmatch_inpaint.dll
└── models
    ├── aot_inpainter.ckpt
    ├── comictextdetector.pt
    ├── comictextdetector.pt.onnx
    ├── lama_mpe.ckpt
    ├── manga-ocr-base
    │   ├── README.md
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   ├── pytorch_model.bin
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    ├── mit32px_ocr.ckpt
    ├── mit48pxctc_ocr.ckpt
    └── pkuseg
        ├── postag
        │   ├── features.pkl
        │   └── weights.npz
        ├── postag.zip
        └── spacy_ontonotes
            ├── features.msgpack
            └── weights.npz

7 directories, 23 files
```

-  Install pyenv command line tool for managing Python versions. Recommend installing via Homebrew.
```
# Install via Homebrew
brew install pyenv

# Install via official script
curl https://pyenv.run | bash

# Set shell environment after install
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```


#### 2、Build the application
```
# Enter the `data` working directory
cd data

# Clone the `dev` branch of the repo
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git

# Enter the `BallonsTranslator` working directory
cd BallonsTranslator

# Run the build script, will ask for password at pyinstaller step, enter password and press enter
sh scripts/build-macos-app.sh
```
> 📌The packaged app is at ./data/BallonsTranslator/dist/BallonsTranslator.app, drag the app to macOS application folder to install. Ready to use out of box without extra Python config.


</details> 

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
## OCR
 * All mit* models are from manga-image-translator, support English, Japanese and Korean recognition and text color extraction.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) is from [kha-white](https://github.com/kha-white), text recognition for Japanese, with the main focus being Japanese manga.
 * Support using OCR from [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Username and password need to be filled in, and automatic login will be performed each time the program is launched.
   * The current implementation uses OCR on each textblock individually, resulting in slower speed and no significant improvement in accuracy. It is not recommended. If needed, please use the Tuanzi Detector instead.
   * When using the Tuanzi Detector for text detection, it is recommended to set OCR to none_ocr to directly read the text, saving time and reducing the number of requests.
   * For detailed instructions, see **Tuanzi OCR Instructions**: ([Chinese](doc/团子OCR说明.md) & [Brazilian Portuguese](doc/Manual_TuanziOCR_pt-BR.md) only)

## Inpainting
  * AOT is from [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
  * All lama* are finetuned using [LaMa](https://github.com/advimman/lama)
  * PatchMatch is an algorithm from [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), this program uses a [modified version](https://github.com/dmMaze/PyPatchMatchInpaint) by me. 
  

## Translators
Available translators: Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu. Papago, and Yandex.
 * Google shuts down translate service in China, please set corresponding 'url' in config panel to *.com.
 * [Caiyun](https://dashboard.caiyunapp.com/), [ChatGPT](https://platform.openai.com/playground), [Yandex](https://yandex.com/dev/translate/), [Baidu](http://developers.baidu.com/), and [DeepL](https://www.deepl.com/docs-api/api-access) translators needs to require a token or api key.
 * DeepL & Sugoi translator (and it's CT2 Translation conversion) thanks to [Snowad14](https://github.com/Snowad14).
 * Sugoi translates Japanese to English completely offline. Download [offline model](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm), move "sugoi_translator" into the BallonsTranslator/ballontranslator/data/models. 
 * [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame), check ```low vram mode``` in config panel if you\'re running it locally on a single device and encountered a crash due to vram OOM (enabled by default).
 * DeepLX: Please refer to [Vercel](https://github.com/bropines/Deeplx-vercel) or [deeplx](https://github.com/OwO-Network/DeepLX)
 * Added the [Translators](https://github.com/UlionTse/translators) library, which supports access to some translator services without api keys. You can find out about supported services [here](https://github.com/UlionTse/translators#supported-translation-services).

For other good offline English translators, please refer to this [thread](https://github.com/dmMaze/BallonsTranslator/discussions/515).  
To add a new translator, please reference [how_to_add_new_translator](doc/how_to_add_new_translator.md), it is simple as subclass a BaseClass and implementing two interfaces, then you can use it in the application, you are welcome to contribute to the project.  


## FAQ & Misc
* If your computer has an Nvidia GPU or Apple silicon, the program will enable hardware acceleration. 
* Add support for [saladict](https://saladict.crimx.com) (*All-in-one professional pop-up dictionary and page translator*) in the mini menu on text selection. [Installation guide](doc/saladict.md)
* Accelarate performance if you have a [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) or [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html) device as most modules uses [PyTorch](https://pytorch.org/get-started/locally/).
* Fonts are from your system's fonts.
* Thanks to [bropines](https://github.com/bropines) for the Russian localization.
* Added Export to photoshop JSX script by [bropines](https://github.com/bropines). </br> To read the instructions, improve the code and just poke around to see how it works, you can go to `scripts/export to photoshop` -> `install_manual.md`.

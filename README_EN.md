# BallonTranslator
[简体中文](README.md) | English | [Русский](README_RU.md) | [日本語](README_JA.md) | [Indonesia](README_ID.md)

Yet another computer-aided comic/manga translation tool powered by deep learning.

<img src="doc/src/ui0.jpg" div align=center>

<p align=center>
preview
</p>

# Features
* Fully automated translation  
  - Support automatic text-detection, recognition, removal, and translation. Overall performance is dependent upon these modules.
  - Lettering is based on the formatting estimation of the original text.
  - Works decently with manga and comics.
  - Improved manga->English, English->Chinese typesetting (based on the extraction of balloon regions.).
  
* Image editing  
  - Support mask editing & inpainting (something like spot healing brush tool in PS) 
  - Adapted to images with extreme aspect ratio such as webtoons
  
* Text editing  
  - Support rich text formatting and text style presets, translated texts can be edited interactively.
  - Support search & replace
  - Support export/import to/from word documents

# Installation


## Executable Binaries

**Windows users** can download Ballonstranslator-x.x.x-core.7z from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) (note: you also need to download latest Ballonstranslator-1.3.xx from GitHub release and extract it to overwrite **Ballontranslator-1.3.0-core** or older installation to get the app updated.)


## Auto installer

Everything is simple here, go to the repository, download the file and follow the instructions.

[Link](https://github.com/bropines/Ballon-translator-portable)

## Run the source code

If you're not on Windows or may want to run the latest development.

```bash
# Clone this repo
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# install requirements_macOS.txt on macOS
$ pip install -r requirements.txt
```

Notes: 
- To update run `git pull`
- `git clone -b dev` for dev branch, or git `checkout dev`

Install pytorch-cuda to enable GPU acceleration if you have a NVIDIA GPU.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

Download the **data** folder from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) and move it into BallonsTranslator/ballontranslator, finally run
```bash
python ballontranslator
```
For Linux or MacOS users, see [this script](ballontranslator/scripts/download_models.sh) and run to download ALL models

### Build macOS application (compatible with both intel and apple silicon chips)
![录屏2023-09-11 14 26 49](https://github.com/hyrulelinks/BallonsTranslator/assets/134026642/647c0fa0-ed37-49d6-bbf4-8a8697bc873e)

#### 1. Preparation
-   Download libs and models from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw "MEGA")

> 📌As of September 11, 2023, Google Drive resources have not been updated to the latest version, so do not download libs and models from Google Drive.
> 
<img width="1268" alt="截屏2023-09-08 13 44 55_7g32SMgxIf" src="https://github.com/dmMaze/BallonsTranslator/assets/134026642/40fbb9b8-a788-4a6e-8e69-0248abaee21a">

-  Download libopencv_world.4.4.0.dylib and libpatchmatch_inpaint.dylib.

> 📌The dylib files in the compressed package below are fat files, compatible with both intel and apple silicon chips for Mac devices.

[libopencv_world.4.4.0.dylib.zip](https://github.com/dmMaze/BallonsTranslator/files/12571658/libopencv_world.4.4.0.dylib.zip)

[libpatchmatch_inpaint.dylib.zip](https://github.com/dmMaze/BallonsTranslator/files/12571660/libpatchmatch_inpaint.dylib.zip)

-  Put all the downloaded resources into a folder called data, the final directory tree structure should look like:

```
data
├── libopencv_world.4.4.0.dylib
├── libpatchmatch_inpaint.dylib
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
sh build-macos-app.sh
```
> 📌The packaged app is at ./data/BallonsTranslator/dist/BallonsTranslator.app, drag the app to macOS application folder to install. Ready to use out of box without extra Python config.


</details>

To use Sugoi translator(Japanese-English only), download [offline model](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm), move "sugoi_translator" into the BallonsTranslator/ballontranslator/data/models.  

# Usage

**It is recommended to run the program in a terminal in case it crashed and left no information, see the following gif.**
<img src="doc/src/run.gif">  
- The first time you run the application, please select the translator and set the source and target languages by clicking the settings icon.
- Open a folder containing images of a comic (manga/manhua/manhwa) that need translation by clicking the folder icon.
- Click the `Run` button and wait for the process to complete.

The font formats such as font size and color are determined by the program automatically in this process, you can predetermine those formats by change corresponding options from "decide by program" to "use global setting" in the config panel->Lettering. (global settings are those formats shown by the right font format panel when you are not editing any textblock in the scene)

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
* The "OCR" and "A" button in the bottom toolbar controls whether to enable OCR and translation, if you disable them, the program will only do the text detection and removal.  
* Set parameters of automatic modules in the config panel.  
* ```Ctrl++```/```Ctrl+-``` (Also ```Ctrl+Shift+=```) to resize image.
* ```Ctrl+G```/```Ctrl+F``` to search globally/in current page.
* ```0-9``` to adjust opacity of lettering layer
  
<img src="doc/src/configpanel.png">  


# Automation modules
This project is heavily dependent upon [manga-image-translator](https://github.com/zyddnys/manga-image-translator), online service and model training is not cheap, please consider to donate the project:  
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

[Sugoi translator](https://sugoitranslator.com/) is created by [mingshiba](https://www.patreon.com/mingshiba).
  
## Text detection
Support English and Japanese text detection, training code and more details can be found at [comic-text-detector](https://github.com/dmMaze/comic-text-detector)

## OCR
 * mit_32px text recognition model is from manga-image-translator, support English and Japanese recognition and text color extraction.
 * mit_48px text recognition model is from manga-image-translator, support English, Japanese and Korean recognition and text color extraction.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) is from [kha-white](https://github.com/kha-white), text recognition for Japanese, with the main focus being Japanese manga.

## Inpainting
  * AOT is from [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
  * [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/advimman/lama)
  * PatchMatch is an algorithm from [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), this program uses a [modified version](https://github.com/dmMaze/PyPatchMatchInpaint) by me. 
  

## Translators
Available translators: Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu. Papago, and Yandex.
 * Google shuts down translate service in China, please set corresponding 'url' in config panel to *.com.
 * [Caiyun](https://dashboard.caiyunapp.com/), [ChatGPT](https://platform.openai.com/playground), [Yandex](https://yandex.com/dev/translate/), [Baidu](http://developers.baidu.com/), and [DeepL](https://www.deepl.com/docs-api/api-access) translators needs to require a token or api key.
 * DeepL & Sugoi translator (and it's CT2 Translation conversion) thanks to [Snowad14](https://github.com/Snowad14).
 * Sugoi translates Japanese to English completely offline.

 To add a new translator, please reference [how_to_add_new_translator](doc/how_to_add_new_translator.md), it is simple as subclass a BaseClass and implementing two interfaces, then you can use it in the application, you are welcome to contribute to the project.  


## FAQ & Misc
* If your computer has an Nvidia GPU, the program will enable cuda acceleration for all models by default, which requires around 6G GPU memory, you can turn down the inpaint_size in the config panel to avoid OOM. 
* Add support for [saladict](https://saladict.crimx.com) (*All-in-one professional pop-up dictionary and page translator*) in the mini menu on text selection. [Installation guide](doc/saladict.md)
* Accelarate performance if you have a [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) or [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html) device as most modules uses [PyTorch](https://pytorch.org/get-started/locally/).
* Fonts are from your system's fonts.
* Thanks to [bropines](https://github.com/bropines) for the Russian localization.

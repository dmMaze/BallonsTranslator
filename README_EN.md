# BallonTranslator
[简体中文](README.md) | English | [Русский](README_RU.md) | [日本語](README_JA.md) | [Indonesia](README_ID.md)

Yet another computer-aided comic/manga translation tool powered by deep learning.

<img src="doc/src/ui0.jpg" div align=center>

<p align=center>
preview
</p>

# Features
* Fully automated translation  
  - Support automatic text-detection, recognition, removal, and translation, overall performance is dependent upon these modules.
  - lettering is based on the formatting estimation of the original text.
  - Works decently with manga and comics.
  - Improved manga->English, English->Chinese typesetting (based on the extraction of balloon regions.).
  
* Image editing  
  - Support mask editing & inpainting (something like spot healing brush tool in PS) 
  - Adapted to images with extreme aspect ratio such as webtoons
  
* Text editing  
  - Support rich text formatting and text style presets, translated texts can be edited interactively.
  - Support search & replace
  - Support export/import to/from word documents

# Usage

Windows users can download Ballonstranslator-x.x.x-core.7z from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)(note: you also need to download latest Ballonstranslator-1.3.xx from GitHub release and extract it to overwrite **Ballontranslator-1.3.0-core** or older installation to get the app updated.)

## Run the source code

```bash

# Clone this repo
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# install requirements_macOS.txt on macOS
$ pip install -r requirements.txt
```

Install pytorch-cuda to enable GPU acceleration if you have a NVIDIA GPU.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

Download the **data** folder from [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) and move it into BallonsTranslator/ballontranslator, finally run
```bash
# For Linux or MacOS users, see [this script](https://github.com/dmMaze/BallonsTranslator/blob/master/ballontranslator/scripts/download_models.sh) and run to download ALL models
python ballontranslator
```

### Apple Silicon Mac native build .app application
```
### install python 3.9.13 virtual environment
brew install pyenv mecab
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.13
pyenv global 3.9.13
python3 -m venv ballonstranslator
source ballonstranslator/bin/activate

# Clone the repository
git clone https://github.com/dmMaze/BallonsTranslator.git
cd BallonsTranslator

# Install the dependencies
pip3 install -r requirements_macOS.txt

# Package the application
cd ballontranslator
sudo pyinstaller __main__.spec

# The packaged `BallonsTranslator.app` is in the `dist` folder
# Note that the app is not functional yet, you need to go to [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing), download `data` and overwrite it to `BallonsTranslator.app/Contents/Resources/data`.
# When overwriting select "``Merge``, after the overwrite is done, the application is finally packaged and complete, out of the box, just drag the application to the macOS application folder, no need to configure the Python environment again.
# Or see [this script](https://github.com/dmMaze/BallonsTranslator/blob/master/ballontranslator/scripts/download_models.sh)
```

To use Sugoi translator(Japanese-English only), download [offline model](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm), move "sugoi_translator" into the BallonsTranslator/ballontranslator/data/models.  

## Fully automated translation
**It is recommended to run the program in a terminal in case it crashed and left no information, see the following gif.**, Please select the desired translator and set the source and target languages the first time you run the application. Open a folder containing images that need translation, click the "Run" button and wait for the process to complete.  
<img src="doc/src/run.gif">  

The font formats such as font size and color are determined by the program automatically in this process, you can predetermine those formats by change corresponding options from "decide by program" to "use global setting" in the config panel->Lettering. (global settings are those formats shown by the right font format panel when you are not editing any textblock in the scene)

## Image editing

### inpaint tool
<img src="doc/src/imgedit_inpaint.gif">
<p align = "center">
Image editing mode, inpainting tool
</p>

### rect tool
<img src="doc/src/rect_tool.gif">
<p align = "center">
rect tool
</p>

To 'erase' unwanted inpainted results, use the inpainting tool or rect tool with your **right button** pressed.  
The result depends on how accurately the algorithm ("method 1" and "method 2" in the gif) extracts the text mask. It could perform worse on complex text & background.  

## Text editing
<img src="doc/src/textedit.gif">
<p align = "center">
Text editing mode
</p>

<img src="doc/src/multisel_autolayout.gif" div align=center>
<p align=center>
batch text formatting & auto layout
</p>

<img src="doc/src/ocrselected.gif" div align=center>
<p align=center>
ocr & translate selected area
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
* ```Ctrl++```/```Ctrl+-``` to resize image
* ```Ctrl+G```/```Ctrl+F``` to search globally/in current page.
  
<img src="doc/src/configpanel.png">  


# Automation modules
This project is heavily dependent upon [manga-image-translator](https://github.com/zyddnys/manga-image-translator), online service and model training is not cheap, please consider to donate the project:  
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

Sugoi translator is created by [mingshiba](https://www.patreon.com/mingshiba).
  
## Text detection
Support English and Japanese text detection, training code and more details can be found at [comic-text-detector](https://github.com/dmMaze/comic-text-detector)

## OCR
 * mit_32px text recognition model is from manga-image-translator, support English and Japanese recognition and text color extraction.
 * mit_48px text recognition model is from manga-image-translator, support English, Japanese and Korean recognition and text color extraction.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) is from [kha-white](https://github.com/kha-white), text recognition for Japanese, with the main focus being Japanese manga.

## Inpainting
  * AOT is from manga-image-translator.
  * PatchMatch is an algorithm from [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), this program use a [modified version](https://github.com/dmMaze/PyPatchMatchInpaint) by me. (Adobe uses this algorithm)
  

## Translators

 * <s> Please change the goolge translator url from *.cn to *.com if you are not blocked by GFW. </s> Google shuts down translate service in China, please set corresponding 'url' in config panel to *.com.
 * Caiyun translator need to require a [token](https://dashboard.caiyunapp.com/).
 * Papago.
 * DeepL & Sugoi translator(and it's CT2 Translation conversion) thanks to [Snowad14](https://github.com/Snowad14).

 To add a new translator, please reference [how_to_add_new_translator](doc/how_to_add_new_translator.md), it is simple as subclass a BaseClass and implementing two interfaces, then you can use it in the application, you are welcome to contribute to the project.  


## Misc
* If your computer has an Nvidia GPU, the program will enable cuda acceleration for all models by default, which requires around 6G GPU memory, you can turn down the inpaint_size in the config panel to avoid OOM. 
* Thanks to [bropines](https://github.com/bropines) for the Russian localisation.  
* Add support for [saladict](https://saladict.crimx.com) (*All-in-one professional pop-up dictionary and page translator*) in the mini menu on text selection. [Installation guide](doc/saladict.md)
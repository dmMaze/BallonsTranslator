# Clone repo
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git
cd BallonsTranslator

# Create and activate Python virtualenv
python_version=$(python -V 2>&1 | cut -d" " -f2 | cut -d. -f1-2)
if [[ "$python_version" == "3.11" ]]; then
  python3 -m venv 'venv'
  source 'venv/bin/activate'
else
  echo "Current Python version is $python_version, but 3.11 is required."
  echo "Install Python 3.11 via pyenv command."
  if command -v pyenv >/dev/null 2>&1; then
    pyenv install 3.11
    echo "Set current Python version to 3.11"
    pyenv global 3.11
    exec zsh
    python3 -m venv 'venv'
    source 'venv/bin/activate'
  else
    echo "pyenv command is not available"
    echo "Please ensure current Python version is 3.11, then run script again."
  fi
fi

# Install dependencies
pip3 install -r requirements.txt
pip3 install pyinstaller

# Create required folders
mkdir data/libs
mkdir data/models
mkdir data/models/manga-ocr-base
mkdir data/models/pkuseg
mkdir data/models/pkuseg/postag
mkdir data/models/pkuseg/spacy_ontonotes

# Downloads required files
## Download aot_inpainter.ckpt comictextdetector.pt comictextdetector.pt.onnx lama_mpe.ckpt
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt' -o data/models/aot_inpainter.ckpt
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt' -o data/models/comictextdetector.pt
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx' -o data/models/comictextdetector.pt.onnx
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt' -o data/models/lama_mpe.ckpt

## Download manga-ocr-base
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/README.md' -o data/models/manga-ocr-base/README.md
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/config.json' -o data/models/manga-ocr-base/config.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/preprocessor_config.json' -o data/models/manga-ocr-base/preprocessor_config.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/pytorch_model.bin' -o data/models/manga-ocr-base/pytorch_model.bin
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/special_tokens_map.json' -o data/models/manga-ocr-base/special_tokens_map.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/blob/main/tokenizer_config.json' -o data/models/manga-ocr-base/tokenizer_config.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/vocab.txt' -o data/models/manga-ocr-base/vocab.txt

## Download mit32px_ocr.ckpt 
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr.zip' -o ocr.zip
unzip 'ocr.zip' -d data/models
mv data/models/ocr.ckpt data/models/mit32px_ocr.ckpt
rm -rf ocr.zip data/models/alphabet-all-v5.txt

## Downloader mit48pxctc_ocr.ckpt
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip' -o ocr-ctc.zip
unzip 'ocr-ctc.zip' -d data/models
mv data/models/ocr-ctc.ckpt data/models/mit48pxctc_ocr.ckpt
rm -rf ocr-ctc.zip data/models/alphabet-all-v5.txt

## Download pkuseg
curl -L 'https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip' -o postag.zip
curl -L 'https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip' -o spacy_ontonotes.zip
unzip 'postag.zip' -d data/models/pkuseg/postag
unzip 'spacy_ontonotes.zip' -d data/models/pkuseg/spacy_ontonotes
rm -rf postag.zip spacy_ontonotes.zip

## Download libopencv_world.4.4.0.dylib libpatchmatch_inpaint.dylib
curl -L 'https://github.com/dmMaze/BallonsTranslator/files/12571658/libopencv_world.4.4.0.dylib.zip' -o libopencv_world.4.4.0.dylib.zip
curl -L 'https://github.com/dmMaze/BallonsTranslator/files/12571660/libpatchmatch_inpaint.dylib.zip' -o libpatchmatch_inpaint.dylib.zip
unzip 'libopencv_world.4.4.0.dylib.zip' -d data/libs
unzip 'libpatchmatch_inpaint.dylib.zip' -d data/libs
rm -rf libopencv_world.4.4.0.dylib.zip libpatchmatch_inpaint.dylib.zip

# Comment lines 213-229 of launch.py
cp launch.py launch.py.bak # backup launch.py
sed -i '' '213,229s|^| \#|' launch.py # comment specfied lines
sed -n '213,229p' launch.py # check if comment is successful

# Build macOS app via pyinstaller
sudo pyinstaller launch.spec

# Copy built app to Download directory
ditto dist/BallonsTranslator.app $HOME/Downloads/BallonsTranslator.app
echo "'BallonsTranslator.app' is in Downloads directory."
echo "Please manually drag 'BallonsTranslator.app' to Applications directory to finish install."

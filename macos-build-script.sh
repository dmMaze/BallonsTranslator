# Clone repository
echo "STEP 1: Clone repository."
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git
cd BallonsTranslator

# Create and activate Python virtual environment
echo "STEP 2: Create and activate Python virtual environment"
python_version=$(python3 -V 2>&1 | cut -d" " -f2 | cut -d"." -f1-2)

if ! which python3 >/dev/null 2>&1; then
    echo "ERROR: The 'python3' command not found. Please check the environment configuration."
    exit 1
else
    echo "INFO: The 'python3' command found." 
    if [ "$python_version" == "3.11" ]; then
        echo "INFO: The current Python version is 3.11"
        echo "INFO: Create Python virtual enviroment."
        python3 -m venv venv
        echo "INFO: Activate Python virtual enviroment."
        source venv/bin/activate
    else
        echo "ERROR: The current Python version is $python_version but 3.11 is required."
        echo "ERROR: Please switch to Python 3.11 before running this script."
        exit 1
    fi
fi

# OpenCV installation check
echo "STEP 3: Check installation of OpenCV."
python3 -c "import cv2" 2>/dev/null
if [ $? -eq 0 ]; then
    opencv_version=$(python3 -c "import cv2; print(cv2.__version__)")
    echo "INFO: OpenCV is installed. Version: $opencv_version"
else
    echo "ERROR: OpenCV is not installed."
    echo "ERROR: Please install OpenCV before running this script."
    echo "INFO: Recommand install via Homebrew with command 'brew install opencv'."
    exit 1
fi

# Create required directories
mkdir data/libs
mkdir data/models
mkdir data/models/manga-ocr-base
mkdir data/models/pkuseg
mkdir data/models/pkuseg/postag
mkdir data/models/pkuseg/spacy_ontonotes

# Download required data files
echo "STEP 4: Download required data files."

## Download models
echo "INFO: Download aot_inpainter model."
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt' -o data/models/aot_inpainter.ckpt
echo "INFO: Download comictextdetector model."
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt' -o data/models/comictextdetector.pt
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx' -o data/models/comictextdetector.pt.onnx
echo "INFO: Download lama_mpe model."
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt' -o data/models/lama_mpe.ckpt

## Download manga-ocr-base
echo "INFO: Download manga-ocr-base models."
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/README.md' -o data/models/manga-ocr-base/README.md
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/config.json' -o data/models/manga-ocr-base/config.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/preprocessor_config.json' -o data/models/manga-ocr-base/preprocessor_config.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/pytorch_model.bin' -o data/models/manga-ocr-base/pytorch_model.bin
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/special_tokens_map.json' -o data/models/manga-ocr-base/special_tokens_map.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/blob/main/tokenizer_config.json' -o data/models/manga-ocr-base/tokenizer_config.json
curl -L 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/vocab.txt' -o data/models/manga-ocr-base/vocab.txt

## Download OCR models
echo "INFO: Download ocr model."
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr.zip' -o ocr.zip
unzip 'ocr.zip' -d data/models
mv data/models/ocr.ckpt data/models/mit32px_ocr.ckpt
rm -rf ocr.zip data/models/alphabet-all-v5.txt

## Download OCR-CTC models
echo "INFO: Download ocr-ctc model."
curl -L 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip' -o ocr-ctc.zip
unzip 'ocr-ctc.zip' -d data/models
mv data/models/ocr-ctc.ckpt data/models/mit48pxctc_ocr.ckpt
rm -rf ocr-ctc.zip data/models/alphabet-all-v5.txt

## Download pkuseg
echo "INFO: Download pkuseg models."
curl -L 'https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip' -o postag.zip
curl -L 'https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip' -o spacy_ontonotes.zip
unzip 'postag.zip' -d data/models/pkuseg/postag
unzip 'spacy_ontonotes.zip' -d data/models/pkuseg/spacy_ontonotes
rm -rf postag.zip spacy_ontonotes.zip

## Download libraries
echo "INFO: Download patchmatch_inpaint libraries."
curl -L 'https://github.com/dmMaze/BallonsTranslator/files/12571658/libopencv_world.4.4.0.dylib.zip' -o libopencv_world.4.4.0.dylib.zip
curl -L 'https://github.com/dmMaze/BallonsTranslator/files/12571660/libpatchmatch_inpaint.dylib.zip' -o libpatchmatch_inpaint.dylib.zip
unzip 'libopencv_world.4.4.0.dylib.zip' -d data/libs
unzip 'libpatchmatch_inpaint.dylib.zip' -d data/libs
rm -rf libopencv_world.4.4.0.dylib.zip libpatchmatch_inpaint.dylib.zip
arch=$(uname -m)
if [ "$arch" = "arm64" ]; then
    ditto data/libs/libopencv_world.4.4.0.dylib data/libs/libopencv_world2.4.4.0.dylib --arch arm64
    ditto data/libs/libpatchmatch_inpaint.dylib data/libs/libpatchmatch_inpaint2.dylib --arch arm64
    rm -rf data/libs/libopencv_world.4.4.0.dylib libpatchmatch_inpaint.dylib
    mv data/libs/libopencv_world2.4.4.0.dylib data/libs/libopencv_world.4.4.0.dylib
    mv data/libs/libpatchmatch_inpaint2.dylib data/libs/libpatchmatch_inpaint.dylib
else
    ditto data/libs/libopencv_world.4.4.0.dylib data/libs/libopencv_world2.4.4.0.dylib --arch x86_64
    ditto data/libs/libpatchmatch_inpaint.dylib data/libs/libpatchmatch_inpaint2.dylib --arch x86_64
    rm -rf data/libs/libopencv_world.4.4.0.dylib libpatchmatch_inpaint.dylib
    mv data/libs/libopencv_world2.4.4.0.dylib data/libs/libopencv_world.4.4.0.dylib
    mv data/libs/libpatchmatch_inpaint2.dylib data/libs/libpatchmatch_inpaint.dylib
fi

# Checklist of required files
check_list="
data/alphabet-all-v5.txt
data/libs
data/libs/libopencv_world.4.4.0.dylib
data/libs/libpatchmatch_inpaint.dylib
data/models
data/models/aot_inpainter.ckpt
data/models/comictextdetector.pt
data/models/comictextdetector.pt.onnx
data/models/lama_mpe.ckpt
data/models/manga-ocr-base
data/models/manga-ocr-base/README.md
data/models/manga-ocr-base/config.json
data/models/manga-ocr-base/preprocessor_config.json
data/models/manga-ocr-base/pytorch_model.bin
data/models/manga-ocr-base/special_tokens_map.json
data/models/manga-ocr-base/tokenizer_config.json
data/models/manga-ocr-base/vocab.txt
data/models/mit32px_ocr.ckpt
data/models/mit48pxctc_ocr.ckpt
data/models/pkuseg
data/models/pkuseg/postag
data/models/pkuseg/postag/features.pkl
data/models/pkuseg/postag/weights.npz
data/models/pkuseg/spacy_ontonotes
data/models/pkuseg/spacy_ontonotes/features.msgpack
data/models/pkuseg/spacy_ontonotes/weights.npz
data/pkusegscores.json
"

# Validate required data files exist
echo "STEP 5: Validate required data files exist."
fail=false
for item in $check_list; do
    if [ ! -e "$item" ]; then
        echo "ERROR: $item not found"
        fail=true
    fi
done
 
if [ "$fail" = true ]; then
    echo "ERROR: Required files check failed, stopping script."
    exit 1
else
    echo "INFO: All required files exist, continuing script execution."
fi

# Install Python dependencies
echo "STEP 6: Install Python dependencies."
pip3 install -r requirements.txt
pip3 install pyinstaller

# Delete .DS_Store files 
echo "STEP 7: Delete .DS_Store files."
sudo find ./ -name '.DS_Store'
echo "INFO: All .DS_Store files found must be deleted."
sudo find ./ -name '.DS_Store' -delete

# Create packaged app
echo "STEP 8: Create packaged app."
echo "INFO: Use the pyinstaller spec file to bundle the app."
sudo pyinstaller launch.spec

# Copy app to Downloads
echo "INFO: Copy app to Downloads folder."
ditto dist/BallonsTranslator.app $HOME/Downloads/BallonsTranslator.app
echo "INFO: The app is now in your Downloads folder."
echo "INFO: Drag and drop the app icon into Applications folder to install it."
open $HOME/Downloads

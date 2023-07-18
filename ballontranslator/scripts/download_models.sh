#!/usr/bin/env bash


# Get the most of the models https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.3 here
# Place them in data/models

pushd $(dirname "$0") &> /dev/null

set -e 

PWD="$(pwd)"
MODELS_DIR="$PWD/../../data/models"
LIBS_DIR="$PWD/../../data/libs"

echo $PWD
echo $MODELS_DIR
echo $LIBS_DIR

mkdir -p $MODELS_DIR
cd $MODELS_DIR

# Comic Text Detector
wget -c "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt"

# Comic Text Detector for CPU
wget -c "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"

# AOT Inpainter
wget -c "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt" -O aot_inpainter.ckpt

# LaMa Inpainter
wget -c "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt" -O lama_mpe.ckpt

# Sugoi Translator
wget -c "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/sugoi-models.zip" ; unzip -d sugoi_translator sugoi-models.zip

# MIT_48PX_CTC OCR
wget -c "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip"; unzip ocr-ctc.zip; mv ocr-ctc.ckpt mit48pxctc_ocr.ckpt; rm alphabet-all-v5.txt

# Manga OCR
git lfs install; git clone "https://huggingface.co/kha-white/manga-ocr-base"

mkdir -p $LIBS_DIR
echo $LIBS_DIR

git clone --depth 1 https://github.com/vacancy/PyPatchMatch
cd PyPatchMatch

# TODO
# idk how to detect if 'pkg-config --cflags opencv' fails because mine does (Arch BTW), 
# but there's opencv4 on my system and it compiles.
# an idea is to 'ls opencv*' these paths 'pkg-config --variable pc_path pkg-config' but... to do.

make -j$(nproc)
mv libpatchmatch.so $LIBS_DIR
cd ..; rm -rf PyPatchMatch


popd &> /dev/null
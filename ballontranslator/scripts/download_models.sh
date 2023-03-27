#!/usr/bin/env bash


CTD_MODEL_LINK="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt"
CTD_ONNX_MODEL_LINK="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"

AOT_INPAINTER_MODEL_LINK="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt"
LAMA_MPE_INPAINTER_MODEL_LINK="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt"

SUGOI_TRANSLATOR_MODEL_LINK="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/sugoi-models.zip"

MANGA_OCR_MODEL_LINK="https://huggingface.co/kha-white/manga-ocr-base"
MIT48PX_OCR_MODEL_LINK="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip"


pushd $(dirname "$0") &> /dev/null

set -e 

PWD="$(pwd)"
MODELS_DIR="$PWD/../data/models"
LIBS_DIR="$PWD/../data/libs"

mkdir -p $MODELS_DIR
cd $MODELS_DIR

wget -c $CTD_MODEL_LINK

wget -c $CTD_ONNX_MODEL_LINK

wget -c $AOT_INPAINTER_MODEL_LINK -O aot_inpainter.ckpt

wget -c $LAMA_MPE_INPAINTER_MODEL_LINK -O lama_mpe.ckpt

wget -c $SUGOI_TRANSLATOR_MODEL_LINK ; unzip -d sugoi_translator sugoi-models.zip

wget -c $MIT48PX_OCR_MODEL_LINK; unzip ocr-ctc.zip; mv ocr-ctc.ckpt mit48pxctc_ocr.ckpt; rm alphabet-all-v5.txt

git lfs install; git clone $MANGA_OCR_MODEL_LINK

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
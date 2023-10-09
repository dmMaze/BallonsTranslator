# Clone repository
echo "STEP 1: Clone repository."
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git
cd BallonsTranslator

# Define directories
DATA_DIR='data'
LIBS_DIR='data/libs'
MODELS_DIR='data/models'
MANGA_OCR_BASE_DIR='data/models/manga-ocr-base'
PKUSEG_DIR='data/models/pkuseg'
POSTAG_DIR='data/models/pkuseg/postag'
SPACY_ONTONOTES_DIR='data/models/pkuseg/spacy_ontonotes'

# Check and make directories
mkdir -p "$DATA_DIR"
mkdir -p "$LIBS_DIR"
mkdir -p "$MODELS_DIR" 
mkdir -p "$MANGA_OCR_BASE_DIR"
mkdir -p "$PKUSEG_DIR"
mkdir -p "$POSTAG_DIR"
mkdir -p "$SPACY_ONTONOTES_DIR"

# Create and activate Python virtual environment
echo "STEP 2: Create and activate Python virtual environment"
python_version=$(python3 -V 2>&1 | cut -d" " -f2 | cut -d"." -f1-2)

if ! which python3 >/dev/null 2>&1; then
    echo "ERROR: The 'python3' command not found. Please check the environment configuration."
    exit 1
else
    echo "INFO: The 'python3' command found." 
    if [ "$python_version" == "3.11" ]; then
        echo "INFO: ✅ The current Python version is 3.11"
        python3 -m venv venv
        echo "INFO: ✅ Python virtual enviroment created."
        source venv/bin/activate
        echo "INFO: ✅ Python virtual enviroment activated."
    else
        echo "ERROR: ❌ The current Python version is $python_version but 3.11 is required."
        echo "ERROR: Please switch to Python 3.11 before running this script."
        exit 1
    fi
fi

# OpenCV installation check
echo "STEP 3: Check installation of OpenCV."
echo "INFO: Install OpenCV Python package in virtual environment."
pip3 install opencv-python
echo "INFO: Checking OpenCV installation..."
python3 -c "import cv2" 2>/dev/null
if [ $? -eq 0 ]; then
    opencv_version=$(python3 -c "import cv2; print(cv2.__version__)")
    echo "INFO: ✅ OpenCV is installed. Version: $opencv_version"
else
    echo "ERROR: ❌ OpenCV is not installed."
    echo "ERROR: Please install OpenCV before running this script."
    echo "INFO: Recommand install via Homebrew with command 'brew install opencv'."
    exit 1
fi

# Download extra data files
echo "STEP 4: Download extra data files."

# Function to calculate file hash
calculate_hash() {
    local file_path=$1
    shasum -a 256 "$file_path" | cut -d ' ' -f 1
}

# Function to download and process files
download_and_process_files() {
    echo "INFO: STEP 4: Downloading data files..."
    local files=(
        'inpainting.ckpt|https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3|no_zip||aot_inpainter.ckpt|data/models|878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f'
        'comictextdetector.pt|https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3|no_zip||comictextdetector.pt|data/models|1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb'
        'comictextdetector.pt.onnx|https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3|no_zip||comictextdetector.pt.onnx|data/models|1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'
        'inpainting_lama_mpe.ckpt|https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3|no_zip||lama_mpe.ckpt|data/models|d625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc'
        'README.md|https://huggingface.co/kha-white/manga-ocr-base/raw/main|no_zip||README.md|data/models/manga-ocr-base|32f413afcc4295151e77d25202c5c5d81ef621b46f947da1c3bde13256dc0d5f'
        'config.json|https://huggingface.co/kha-white/manga-ocr-base/raw/main|no_zip||config.json|data/models/manga-ocr-base|8c0e395de8fa699daaac21aee33a4ba9bd1309cfbff03147813d2a025f39f349'
        'preprocessor_config.json|https://huggingface.co/kha-white/manga-ocr-base/raw/main|no_zip||preprocessor_config.json|data/models/manga-ocr-base|af4eb4d79cf61b47010fc0bc9352ee967579c417423b4917188d809b7e048948'
        'pytorch_model.bin|https://huggingface.co/kha-white/manga-ocr-base/raw/main|no_zip||pytorch_model.bin|data/models/manga-ocr-base|c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb'
        'special_tokens_map.json|https://huggingface.co/kha-white/manga-ocr-base/raw/main|no_zip||special_tokens_map.json|data/models/manga-ocr-base|303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3'
        'tokenizer_config.json|https://huggingface.co/kha-white/manga-ocr-base/raw/main|no_zip||tokenizer_config.json|data/models/manga-ocr-base|d775ad1deac162dc56b84e9b8638f95ed8a1f263d0f56f4f40834e26e205e266'
        'vocab.txt|https://huggingface.co/kha-white/manga-ocr-base/raw/main|no_zip||vocab.txt|data/models/manga-ocr-base|344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d'
        'ocr.zip|https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3|zip|ocr.ckpt|mit32px_ocr.ckpt|data/models|d9f619a9dccce8ce88357d1b17d25f07806f225c033ea42c64e86c45446cfe71'
        'ocr-ctc.zip|https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3|zip|ocr-ctc.ckpt|mit48pxctc_ocr.ckpt|data/models|8b0837a24da5fde96c23ca47bb7abd590cd5b185c307e348c6e0b7238178ed89'
        'postag.zip|https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16|zip|features.pkl|features.pkl|data/models/pkuseg/postag|17d734c186a0f6e76d15f4990e766a00eed5f72bea099575df23677435ee749d'
        'postag.zip|https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16|zip|weights.npz|weights.npz|data/models/pkuseg/postag|2bbd53b366be82a1becedb4d29f76296b36ad7560b6a8c85d54054900336d59a'
        'spacy_ontonotes.zip|https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26|zip|features.msgpack|features.msgpack|data/models/pkuseg/spacy_ontonotes|fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18'
        'spacy_ontonotes.zip|https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26|zip|weights.npz|weights.npz|data/models/pkuseg/spacy_ontonotes|5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e'
        'libopencv_world.4.4.0.dylib.zip|https://github.com/dmMaze/BallonsTranslator/files/12571658|zip|libopencv_world.4.4.0.dylib|libopencv_world.4.4.0.dylib|data/libs|58bc785e410389a473a2a130c3134f5721a25bc5456072cb8efd405bf132487e'
        'libpatchmatch_inpaint.dylib.zip|https://github.com/dmMaze/BallonsTranslator/files/12571660|zip|libpatchmatch_inpaint.dylib|libpatchmatch_inpaint.dylib|data/libs|3d318885b3f440af0d837c3716a055e31b98b8eb7e50d1b6142d3cdc46c04e75'
        )
        
    # Iterate through file information
    for file_info in "${files[@]}"; do
        IFS='|' read -r -a file_data <<< "$file_info"
        source_file="${file_data[0]}"
        source_file_base_url="${file_data[1]}"
        is_zip="${file_data[2]}"
        unzip_file="${file_data[3]}"
        target_file="${file_data[4]}"
        target_dir="${file_data[5]}"
        target_file_expected_hash="${file_data[6]}"
        
        # Combine source file and base URL to get download URL
        local download_url="$source_file_base_url/$source_file"
        
        # Check if target_file exists and verify hash if it does
        if [ -e "$target_dir/$target_file" ]; then
            echo "INFO: $target_file already exists, verifying hash..."
            computed_hash=$(calculate_hash "$target_dir/$target_file")
            if [ "$computed_hash" == "$target_file_expected_hash" ]; then
                echo "INFO: ✅ Existing $target_file hash verification passed."
                continue
            else
                echo "WARNING: ❌ Existing $target_file hash verification failed."
                rm -rf "$target_dir/$target_file"
            fi
        fi
            
        # Download and process accordingly based on is_zip and unzip_file
        echo "INFO: Downloading $target_file..."
        if [[ "$is_zip" == "zip" ]]; then
            curl -L "$download_url" -o "$source_file"
            unzip -j "$source_file" "$unzip_file" -d "$target_dir"
            if [ "$unzip_file" != "$target_file" ]; then
                # Rename the file
                mv "$target_dir/$unzip_file" "$target_dir/$target_file"
            fi
            rm -rf "$source_file"
        else
            curl -L "$download_url" -o "$target_dir/$target_file"
        fi
        
        # Calculate hash after download and processing
        downloaded_file_hash=$(calculate_hash "$target_dir/$target_file")
    
        # Check if hash matches expected hash
        if [ "$downloaded_file_hash" == "$target_file_expected_hash" ]; then
            echo "INFO: ✅ Downloaded $target_file hash verification passed."
            continue
        else
            echo "WARNING: ❌ Downloaded $target_file hash verification failed."
            # Remove the existing file
            rm -f "$target_dir/$target_file"

            # Redownload the file
            if [[ "$is_zip" == "zip" ]]; then
                curl -L "$download_url" -o "$source_file"
                unzip -j "$source_file" "$unzip_file" -d "$target_dir"
                if [ "$unzip_file" != "$target_file" ]; then
                    mv "$target_dir/$unzip_file" "$target_dir/$target_file"
                fi
                rm -f "$source_file"
            else
                curl -L "$download_url" -o "$target_dir/$target_file"
            fi
            
            # Calculate hash after re-download
            redownloaded_file_hash=$(calculate_hash "$target_dir/$target_file")
            
            # Check if hash matches expected hash after re-download
            if [ "$redownloaded_file_hash" == "$target_file_expected_hash" ]; then
                echo "INFO: ✅ Re-downloaded $target_file hash verification passed."
                continue
            else
                echo "WARNING: ❌ Re-downloaded $target_file hash verification failed."
                echo "ERROR: ❌ Unable to download $target_file. Exiting."
                exit 1
            fi
        fi
    done
}

# Call the download functions
download_and_process_files

# Extract libraries based on architecture
local arch=$(uname -m)
if [ "$arch" = "arm64" ]; then
    ditto --arch arm64 "$LIBS_DIR/libopencv_world.4.4.0.dylib" "$LIBS_DIR/libopencv_world2.4.4.0.dylib"
    ditto --arch arm64 "$LIBS_DIR/libpatchmatch_inpaint.dylib" "$LIBS_DIR/libpatchmatch_inpaint2.dylib"
else
    ditto --arch x86_64 "$LIBS_DIR/libopencv_world.4.4.0.dylib" "$LIBS_DIR/libopencv_world2.4.4.0.dylib"
    ditto --arch x86_64 "$LIBS_DIR/libpatchmatch_inpaint.dylib" "$LIBS_DIR/libpatchmatch_inpaint2.dylib"
fi

# Remove unnecessary libraries
rm "$LIBS_DIR/libopencv_world.4.4.0.dylib" "$LIBS_DIR/libpatchmatch_inpaint.dylib"
mv "$LIBS_DIR/libopencv_world2.4.4.0.dylib" "$LIBS_DIR/libopencv_world.4.4.0.dylib"
mv "$LIBS_DIR/libpatchmatch_inpaint2.dylib" "$LIBS_DIR/libpatchmatch_inpaint.dylib"

# Checklist of extra data files
check_list="
data/alphabet-all-v5.txt
$LIBS_DIR/libopencv_world.4.4.0.dylib
$LIBS_DIR/libpatchmatch_inpaint.dylib
$MODELS_DIR/aot_inpainter.ckpt
$MODELS_DIR/comictextdetector.pt
$MODELS_DIR/comictextdetector.pt.onnx
$MODELS_DIR/lama_mpe.ckpt
$MANGA_OCR_BASE_DIR/README.md
$MANGA_OCR_BASE_DIR/config.json
$MANGA_OCR_BASE_DIR/preprocessor_config.json
$MANGA_OCR_BASE_DIR/pytorch_model.bin
$MANGA_OCR_BASE_DIR/special_tokens_map.json
$MANGA_OCR_BASE_DIR/tokenizer_config.json
$MANGA_OCR_BASE_DIR/vocab.txt
$MODELS_DIR/mit32px_ocr.ckpt
$MODELS_DIR/mit48pxctc_ocr.ckpt
$POSTAG_DIR/features.pkl
$POSTAG_DIR/weights.npz
$SPACY_ONTONOTES_DIR
$SPACY_ONTONOTES_DIR/features.msgpack
$SPACY_ONTONOTES_DIR/weights.npz
data/pkusegscores.json
"

# Validate extra data files exist
echo "STEP 5: Validate data files exist."
fail=false
for item in $check_list; do
    if [ ! -e "$item" ]; then
        echo "ERROR: ❌ $item not found"
        fail=true
    fi
done
 
if [ "$fail" = true ]; then
    echo "ERROR: ❌ Data files check failed. Exiting."
    exit 1
else
    echo "INFO: ✅ Data files all exist."
fi

# Install Python dependencies
echo "STEP 6: Install Python dependencies."
pip3 install -r requirements.txt
pip3 install pyinstaller

# Delete .DS_Store files 
echo "STEP 7: Delete .DS_Store files."
echo "INFO: Permission required to delete .DS_Store files."
sudo find ./ -name '.DS_Store'
sudo find ./ -name '.DS_Store' -delete
echo "INFO: ✅ .DS_Store files all deleted."

# Create packaged app
echo "STEP 8: Create packaged app."
echo "INFO: Use the pyinstaller spec file to bundle the app."
sudo pyinstaller launch.spec

# Check if app exists
app_path="dist/BallonsTranslator.app"
if [ -e "$app_path" ]; then
    # Copy app to Downloads folder
    echo "INFO: Copying app to Downloads folder..."
    ditto "$app_path" "$HOME/Downloads/BallonsTranslator.app"
    echo "INFO: ✅ The app is now in your Downloads folder."
    echo "INFO: Drag and drop the app icon into Applications folder to install it."
    open $HOME/Downloads
else
    echo "ERROR: ❌ App not found. Please build the app first."
fi

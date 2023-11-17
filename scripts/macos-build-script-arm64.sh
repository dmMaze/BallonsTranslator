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
    echo "ERROR: ❌ The 'python3' command not found."
    echo "ERROR: Please check the Python environment configuration."
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

# Check data file hash
echo "STEP 4: Check data file hash."

# Function to calculate file hash
calculate_hash() {
    local file_path=$1
    shasum -a 256 "$file_path" | cut -d ' ' -f 1
}

# Function to check file hash
check_file_hash() {
    local files=(
        'alphabet-all-v5.txt|data|c1295ae1962e69e35b5b225a0405d1f3432e368c9941d23bfd3acda12654da33'
        'alphabet-all-v7.txt|data|f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a'
        'macos_libopencv_world.4.8.0.dylib|data/libs|843704ab096d3afd8709abe2a2c525ce3a836bb0a629ed1ee9b8f5cee9938310'
        'macos_libpatchmatch_inpaint.dylib|data/libs|849ca84759385d410c9587d69690e668822a3fc376ce2219e583e7e0be5b5e9a'
        'aot_inpainter.ckpt|data/models|878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f'
        'comictextdetector.pt|data/models|1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb'
        'comictextdetector.pt.onnx|data/models|1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'
        'lama_large_512px.ckpt|data/models|11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935'
        'lama_mpe.ckpt|data/models|d625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc'
        'config.json|data/models/manga-ocr-base|8c0e395de8fa699daaac21aee33a4ba9bd1309cfbff03147813d2a025f39f349'
        'preprocessor_config.json|data/models/manga-ocr-base|af4eb4d79cf61b47010fc0bc9352ee967579c417423b4917188d809b7e048948'
        'pytorch_model.bin|data/models/manga-ocr-base|c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb'
        'README.md|data/models/manga-ocr-base|32f413afcc4295151e77d25202c5c5d81ef621b46f947da1c3bde13256dc0d5f'
        'special_tokens_map.json|data/models/manga-ocr-base|303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3'
        'tokenizer_config.json|data/models/manga-ocr-base|d775ad1deac162dc56b84e9b8638f95ed8a1f263d0f56f4f40834e26e205e266'
        'vocab.txt|data/models/manga-ocr-base|344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d'
        'mit32px_ocr.ckpt|data/models|d9f619a9dccce8ce88357d1b17d25f07806f225c033ea42c64e86c45446cfe71'
        'mit48pxctc_ocr.ckpt|data/models|8b0837a24da5fde96c23ca47bb7abd590cd5b185c307e348c6e0b7238178ed89'
        'ocr_ar_48px.ckpt|data/models|29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d'
        'features.pkl|data/models/pkuseg/postag|17d734c186a0f6e76d15f4990e766a00eed5f72bea099575df23677435ee749d'
        'weights.npz|data/models/pkuseg/postag|2bbd53b366be82a1becedb4d29f76296b36ad7560b6a8c85d54054900336d59a'
        'features.msgpack|data/models/pkuseg/spacy_ontonotes|fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18'
        'weights.npz|data/models/pkuseg/spacy_ontonotes|5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e'
        'pkusegscores.json|data|ca6b8c6b8ba70d4370b0f2de6bd128ebb0f5f64ff06f01ba6358e49a776b0c3f'
    )
        
    # Iterate through file information
    for file_info in "${files[@]}"; do
        IFS='|' read -r -a file_data <<< "$file_info"
        target_file="${file_data[0]}"
        target_dir="${file_data[1]}"
        target_precalculated_hash="${file_data[2]}"
        target_file_path="$target_dir/$target_file"

        # Check if $target_file exists
        if [ -e "$target_file_path" ]; then
            target_computed_hash=$(calculate_hash "$target_file_path")
            
            # Compare hashes
            if [ "$target_computed_hash" == "$target_precalculated_hash" ]; then
                echo "INFO: ✅ $target_file found and hash matches."
            else
                echo "WARNING: ❌ $target_file found but hash mismatches."
                echo "INFO: Expected hash: $target_precalculated_hash"
                echo "INFO: Computed hash: $target_computed_hash"
                exit 1
            fi
        else
            echo "WARNING: ❌ $target_file not found at $target_file_path."
            exit 1
        fi
    done
}

# Call functions
check_file_hash

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

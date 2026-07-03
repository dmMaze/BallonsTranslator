#!/usr/bin/env sh

set -eu

SOURCE_URL="https://github.com/dmMaze/BallonsTranslator/archive/refs/heads/dev.zip"
APP_NAME="BallonsTranslator"
ARCHIVE_DIR="BallonsTranslator-dev"
VENV_NAME=".venv"

info() {
    printf '%s\n' "$*"
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [ -n "${TMP_DIR:-}" ] && [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
    fi
}

make_temp_dir() {
    mktemp -d 2>/dev/null || mktemp -d -t ballontranslator
}

download_file() {
    url=$1
    output=$2

    if command -v curl >/dev/null 2>&1; then
        curl -fL "$url" -o "$output"
        return
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -O "$output" "$url"
        return
    fi

    fail "curl or wget is required to download ${url}"
}

extract_archive() {
    archive=$1
    output_dir=$2

    # macOS /usr/bin/unzip can fail on GitHub source zips with UTF-8 filenames.
    if command -v bsdtar >/dev/null 2>&1; then
        bsdtar -xf "$archive" -C "$output_dir"
        return
    fi
    if command -v unzip >/dev/null 2>&1; then
        unzip -q "$archive" -d "$output_dir"
        return
    fi

    fail "bsdtar or unzip is required to extract ${archive}"
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    info "uv was not found. Installing uv for the current user..."
    installer="$TMP_DIR/uv-install.sh"
    download_file "https://astral.sh/uv/install.sh" "$installer"
    sh "$installer"

    PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    export PATH

    command -v uv >/dev/null 2>&1 || fail "uv install completed, but uv is still not on PATH"
}

backup_existing_app() {
    app_dir=$1

    if [ ! -e "$app_dir" ]; then
        return
    fi

    timestamp=$(date +%Y%m%d%H%M%S)
    backup_dir="${app_dir}.backup.${timestamp}"
    suffix=1

    while [ -e "$backup_dir" ]; do
        backup_dir="${app_dir}.backup.${timestamp}.${suffix}"
        suffix=$((suffix + 1))
    done

    info "Existing ${APP_NAME} found. Moving it to ${backup_dir}"
    mv "$app_dir" "$backup_dir"
}

INSTALL_PARENT=$(pwd)
APP_DIR="${INSTALL_PARENT}/${APP_NAME}"
TMP_DIR=$(make_temp_dir)
trap cleanup EXIT INT TERM

ensure_uv

archive_path="${TMP_DIR}/dev.zip"
extract_dir="${TMP_DIR}/extract"

info "Downloading ${APP_NAME} source..."
download_file "$SOURCE_URL" "$archive_path"

mkdir "$extract_dir"
info "Extracting source..."
extract_archive "$archive_path" "$extract_dir"

extracted_app="${extract_dir}/${ARCHIVE_DIR}"
[ -d "$extracted_app" ] || fail "Expected ${ARCHIVE_DIR} in downloaded archive"

backup_existing_app "$APP_DIR"
mv "$extracted_app" "$APP_DIR"

cd "$APP_DIR"

[ -f "launch.sh" ] || fail "Downloaded source does not include launch.sh"
chmod +x "launch.sh"

info "Creating ${VENV_NAME} with Python 3.12..."
uv venv --python 3.12 "$VENV_NAME"

venv_python="${APP_DIR}/${VENV_NAME}/bin/python"
[ -x "$venv_python" ] || fail "Virtual environment Python was not created at ${venv_python}"

info ""
info "${APP_NAME} is ready."
info "Launching ${APP_NAME}..."
cleanup
trap - EXIT INT TERM
exec ./launch.sh "$@"

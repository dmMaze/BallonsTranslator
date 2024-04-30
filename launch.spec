# 导入模块
import os
import sys
from PyInstaller.utils.hooks import collect_data_files
import subprocess

# 获取提交哈希值
commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()  

# 构造带提交哈希值的版本号
version = "1.4.0.dev." + commit_hash

block_cipher = None

a = Analysis([
        'launch.py',
        ],
    pathex=[
        './scripts',
        ],
    binaries=[],
    datas=[
        ('.btrans_cache', './.btrans_cache'),
        ('config', './config'),
        ('data', './data'),
        ('doc', './doc'),
        ('fonts', './fonts'),
        ('icons', './icons'),
        ('modules', './modules'),
        ('scripts', './scripts'),
        ('translate', './translate'),
        ('ui', './ui'),
        ('utils', './utils'),
        ('venv/lib/python3.12/site-packages/spacy_pkuseg', './spacy_pkuseg'),
        ('venv/lib/python3.12/site-packages/torchvision', './torchvision'),
        ('venv/lib/python3.12/site-packages/translators', './translators'),
        ('venv/lib/python3.12/site-packages/cryptography', './cryptography'),
        ],
    hiddenimports=[
        'PyQt6',
        'numpy',
        'urllib3',
        'jaconv',
        'torch',
        'torchvision',
        'transformers',
        'fugashi',
        'unidic_lite',
        'tqdm',
        'shapely',
        'pyclipper',
        'einops',
        'termcolor',
        'bs4',
        'deepl',
        'qtpy',
        'sentencepiece',
        'ctranslate2',
        'docx2txt',
        'piexif',
        'keyboard',
        'requests',
        'colorama',
        'openai',
        'httpx',
        'langdetect',
        'srsly',
        'execjs',
        'pathos',
        ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='launch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='launch',
)
app = BUNDLE(
    coll,
    name='BallonsTranslator.app',
    icon='icons/icon.icns',
    bundle_identifier=None,
    info_plist={
        'CFBundleDisplayName': 'BallonsTranslator',
        'CFBundleName': 'BallonsTranslator',
        'CFBundlePackageType': 'APPL',
        'CFBundleSignature': 'BATR',
        'CFBundleShortVersionString': version,
        'CFBundleVersion': version,
        'CFBundleExecutable': 'launch',
        'CFBundleIconFile': 'icon.icns',
        'CFBundleIdentifier': 'dev.dmmaze.batr',
        'CFBundleInfoDictionaryVersion': '6.0',
        'LSApplicationCategoryType': 'public.app-category.graphics-design',
        'LSEnvironment': {'LANG': 'zh_CN.UTF-8'},
      }
)

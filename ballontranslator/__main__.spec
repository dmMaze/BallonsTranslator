# -*- mode: python ; coding: utf-8 -*-

# macOS pyinstaller 打包

block_cipher = None


a = Analysis(
    ['__main__.py'],
    pathex=[
        './', # 当前目录是***必须***的 
        './dl', 
        './dl/inpaint', 
        './dl/ocr', 
        './dl/textdetector', 
        './dl/textdetector/ctd', 
        './dl/textdetector/yolov5', 
        './dl/translators', 
        './scripts', 
        './tests', 
        './ui', 
        './ui/framelesswindow', 
        './ui/framelesswindow/fw_qt6', 
        './ui/framelesswindow/fw_qt6/linux', 
        './ui/framelesswindow/fw_qt6/mac', 
        './ui/framelesswindow/fw_qt6/utils', 
        './ui/framelesswindow/fw_qt6/windows', 
        './utils'],
    binaries=[],
    datas=[('data', './data')], # data表示data文件夹内所有目录和文件，./data表示这些目录和文件放到打包项目data文件夹内
    hiddenimports=[],
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
    name='__main__',
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
    name='__main__',
)
app = BUNDLE(
    coll,
    name='__main__.app',
    icon=None,
    bundle_identifier=None,
)

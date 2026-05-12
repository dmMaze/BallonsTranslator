from pathlib import Path
import sys
import argparse
import os.path as osp
import os
import importlib
import subprocess
from platform import platform

BRANCH = 'dev'
VERSION = '1.4.0'

python = sys.executable
git = os.environ.get('GIT', "git")
skip_install = False
index_url = os.environ.get('INDEX_URL', "")
QT_APIS = ['pyqt6', 'pyside6', 'pyqt5', 'pyside2']
stored_commit_hash = None

REQ_WIN = [
    'pywin32'
]

PATH_ROOT=Path(__file__).parent
PATH_FONTS=str(PATH_ROOT/'fonts')
FONT_EXTS = {'.ttf','.otf','.ttc','.pfb'}

IS_WIN7 = "Windows-7" in platform()

import utils.shared as shared # Earlier import of shared to use default for config_path argument

parser = argparse.ArgumentParser()
parser.add_argument("--reinstall-torch", action='store_true', help="launch.py argument: install the appropriate version of torch even if you have some version already installed")
parser.add_argument("--proj-dir", default='', type=str, help='Open project directory on startup')
if IS_WIN7:
    parser.add_argument("--qt-api", default='pyqt5', choices=QT_APIS, help='Set qt api')
else:
    parser.add_argument("--qt-api", default='pyqt6', choices=QT_APIS, help='Set qt api')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--requirements", default='requirements.txt')
parser.add_argument("--headless", action='store_true', help='run without GUI')
parser.add_argument("--headless_continuous", action='store_true', help='like headless but will not exit after finishing translation, prompts the user for new exec_dirs until user exits the program')
parser.add_argument("--exec_dirs", default='', help='translation queue (project directories) separated by comma')
parser.add_argument("--ldpi", default=None, type=float, help='logical dots perinch')
parser.add_argument("--export-translation-txt", action='store_true', help='save translation to txt file once RUN completed')
parser.add_argument("--export-source-txt", action='store_true', help='save source to txt file once RUN completed')
parser.add_argument("--frozen", action='store_true', help='run without checking requirements')
parser.add_argument("--update", action='store_true', help="Update the repository before launching") # Add argument --update
parser.add_argument("--config_path", default=shared.CONFIG_PATH, help='Config file to use for translation') # Named config_path to avoid conflict with existing name config
parser.add_argument('--nightly', action='store_true', help="Enable AMD Nightly ROCm")
args, _ = parser.parse_known_args()


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)

    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")

        return ""

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(args, desc=None):
    if skip_install:
        return

    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line} --disable-pip-version-check --no-warn-script-location', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=True)


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


BT = None
APP = None

def restart():
    global BT
    print('restarting...\n')
    if BT:
        BT.close()
    os.execv(sys.executable, ['python'] + sys.argv)


def setup_locks():
    from utils.lock import RUNTIME_LOCKS
    from qtpy.QtCore import QMutex
    RUNTIME_LOCKS['model_loading'] = QMutex()


def main():

    if args.debug:
        os.environ['BALLOONTRANS_DEBUG'] = '1'

    os.environ['QT_API'] = args.qt_api

    commit = commit_hash()

    print('Python version: ', sys.version)
    print('Python executable: ', sys.executable)
    print(f'Version: {VERSION}')
    print(f'Branch: {BRANCH}')
    print(f"Commit hash: {commit}")

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(APP_DIR)

    prepare_environment()

    from utils.zluda_config import enable_zluda_config
    enable_zluda_config()

    if args.update:
        if getattr(sys, 'frozen', False):
            print('Running as app, skipping update.')
        else:
            print('Checking for updates...')
            try:
                current_commit = commit_hash()
                run(f"{git} fetch origin {BRANCH}", desc="Fetching updates from git...", errdesc="Failed to fetch updates.")
                latest_commit = run(f"{git} rev-parse origin/{BRANCH}").strip()

                if current_commit != latest_commit:
                    print("New updates found. Updating repository...")
                    run(f"{git} pull origin {BRANCH}", desc="Updating repository...", errdesc="Failed to update repository.")
                    print("Repository updated. Restarting to apply updates...")
                    restart()
                    return
                else:
                    print("No updates found.")
            except Exception as e:
                print(f"Update check failed: {e}")
                print("Continuing with the current version.")


    from utils.logger import setup_logging, logger as LOGGER
    from utils.io_utils import find_all_files_recursive
    from utils import config as program_config

    from qtpy.QtCore import QTranslator, QLocale, Qt
    shared.args = args
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name().replace('en_CN', 'zh_CN')
    shared.HEADLESS = args.headless
    shared.HEADLESS_CONTINUOUS = args.headless_continuous
    shared.load_cache()
    program_config.load_config(args.config_path)
    config = program_config.pcfg

    if args.headless or args.headless_continuous:
        config.module.load_model_on_demand = True
        config.module.empty_runcache = False

    if sys.platform == 'win32':
        import ctypes
        myappid = u'BalloonsTranslator' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    import qtpy
    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QIcon, QFontDatabase, QGuiApplication, QFont
    from qtpy import API, QT_VERSION

    LOGGER.info(f'QT_API: {API}, QT Version: {QT_VERSION}')

    shared.DEBUG = args.debug
    shared.USE_PYSIDE6 = API == 'pyside6'
    if qtpy.API_NAME[-1] == '6':
        shared.FLAG_QT6 = True
    else:
        shared.FLAG_QT6 = False
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable high dpi scaling
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use high dpi icons
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    os.chdir(shared.PROGRAM_PATH)

    setup_logging(shared.LOGGING_PATH)

    app_args = sys.argv
    if args.headless or args.headless_continuous:
        app_args = sys.argv + ['-platform', 'offscreen']
    app = QApplication(app_args)
    app.setApplicationName('BalloonsTranslator')
    app.setApplicationVersion(VERSION)

    # import msl.loadlib (required by translators/trans_eztrans) before init QApplication
    # yield QWindowsContext: OleInitialize() failed on py3.10, 
    from modules.base import init_module_registries, TORCH_AVAILABLE
    from modules.prepare_local_files import prepare_local_files_forall
    init_module_registries()
    prepare_local_files_forall()

    # Check for Blackwell GPU incompatibility
    if TORCH_AVAILABLE:
        from modules.base import torch as _torch
        if hasattr(_torch, 'cuda') and not _torch.cuda.is_available():
            try:
                _nvsmi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                _gpu_name = _nvsmi.stdout.strip()
                if any(name in _gpu_name for name in ["RTX 5090", "RTX 5080", "RTX 5070", "RTX 5060", "RTX 50"]):
                    print("\n" + "=" * 60)
                    print(f"WARNING: Detected Blackwell GPU ({_gpu_name}) but CUDA is not available!")
                    print("The installed PyTorch was compiled for an older CUDA version")
                    print("that does not support Blackwell (RTX 50 series) GPUs.")
                    print("")
                    print("To fix, reinstall PyTorch with CUDA 12.8+ support:")
                    print("  pip uninstall torch torchvision torchaudio ultralytics -y")
                    print("  python launch.py --reinstall-torch")
                    print("=" * 60 + "\n")
            except Exception:
                pass

    if not args.headless and not args.headless_continuous:
        ps = QGuiApplication.primaryScreen()
        shared.LDPI = ps.logicalDotsPerInch()
        shared.SCREEN_W = ps.geometry().width()
        shared.SCREEN_H = ps.geometry().height()

    lang = config.display_lang
    langp = osp.join(shared.TRANSLATE_DIR, lang + '.qm')
    if osp.exists(langp):
        translator = QTranslator()
        translator.load(lang, osp.dirname(osp.abspath(__file__)) + "/translate")
        app.installTranslator(translator)
    elif lang not in ('en_US', 'English'):
        LOGGER.warning(f'target display language file {langp} doesnt exist.')
    LOGGER.info(f'set display language to {lang}')

    # Fonts
    # Load custom fonts if they exist
    if shared.FLAG_QT6:
        families_before = set(QFontDatabase.families())
    else:
        fdb = QFontDatabase()
        families_before = set(fdb.families())
    # 2. 加载自定义字体
    if osp.exists(PATH_FONTS):
        for fp in find_all_files_recursive(PATH_FONTS, FONT_EXTS):
            fnt_idx = QFontDatabase.addApplicationFont(fp)
            # 无需处理 applicationFontFamilies，让 Qt 内部自行归类合并
    if sys.platform == 'win32' and args.headless:
        # font database does not initialise on windows with qpa -offscreen:
        # whttps://github.com/dmMaze/BallonsTranslator/issues/519
        from qtpy.QtCore import QStandardPaths
        font_dir_list = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.FontsLocation)
        for fd in font_dir_list:
            fp_list = find_all_files_recursive(fd, FONT_EXTS)
            for fp in fp_list:
                fnt_idx = QFontDatabase.addApplicationFont(fp)
    # 3. 记录加载后的字体家族，通过差集精准提取自定义字体家族
    if shared.FLAG_QT6:
        families_after = set(QFontDatabase.families())
    else:
        families_after = set(fdb.families())
    shared.FONT_FAMILIES = families_after
    
    raw_custom_families = sorted(list(families_after - families_before))
     # ===== 智能归并 + 权重排序（不替换别名） =====
    import re
    weight_suffixes = [
        'Thin', 'ExtraLight', 'UltraLight', 'Light', 
        'Regular', 'Medium', 'SemiBold', 'DemiBold',
        'Bold', 'ExtraBold', 'UltraBold', 'Black', 'Heavy'
    ]
    style_suffixes = ['Italic', 'Oblique']
    
    # 用于排序的权重字典（数值越小越靠前，不改变原名称）
    style_order = {
        "Thin": 0, "ExtraLight": 10, "UltraLight": 10, "Light": 20, 
        "Regular": 40, "Normal": 40, "Book": 40, "Medium": 50, 
        "SemiBold": 60, "DemiBold": 60, "Bold": 70, "ExtraBold": 80, 
        "UltraBold": 80, "Black": 90, "Heavy": 90,
        "Italic": 100, "Oblique": 100
    }
    
    def get_sort_key(s):
        base = s.split()[0]
        return style_order.get(base, 99)
    family_alias = {}
    for raw_fam in raw_custom_families:
        canonical = raw_fam
        found = False
        for w in weight_suffixes:
            for s in style_suffixes:
                suffix = f"{w} {s}"
                if raw_fam.endswith(f" {suffix}"):
                    canonical = raw_fam[:-len(suffix)-1]
                    found = True
                    break
            if found: break
        if not found:
            for w in weight_suffixes:
                if raw_fam.endswith(f" {w}"):
                    canonical = raw_fam[:-len(w)-1]
                    found = True
                    break
        if not found:
            for s in style_suffixes:
                if raw_fam.endswith(f" {s}"):
                    canonical = raw_fam[:-len(s)-1]
                    found = True
                    break
        family_alias.setdefault(canonical, []).append(raw_fam)
    
    merged_custom_families = []
    for canonical, raw_list in family_alias.items():
        merged_custom_families.append(canonical)
        shared.FONT_FAMILY_ALIAS[canonical] = raw_list
        
        merged_styles = []
        # 1. 收集 Qt 返回的原始样式（不修改名称）
        for raw_fam in raw_list:
            if shared.FLAG_QT6:
                styles = QFontDatabase.styles(raw_fam)
            else:
                styles = fdb.styles(raw_fam)
            for st in styles:
                if st not in merged_styles:
                    merged_styles.append(st)
        
        # 2. 从子家族名推断缺失的样式（保持原名，不映射别名）
        for raw_fam in raw_list:
            for w in weight_suffixes:
                if raw_fam.endswith(f" {w}"):
                    if w not in merged_styles:
                        merged_styles.append(w)
            for s in style_suffixes:
                if raw_fam.endswith(f" {s}"):
                    if s not in merged_styles:
                        merged_styles.append(s)
        
        # 3. 按权重排序
        merged_styles.sort(key=get_sort_key)
        shared.FONT_STYLES[canonical] = merged_styles
    shared.CUSTOM_FONT_FAMILIES = sorted(merged_custom_families)
    shared.ALL_FONT_FAMILIES = sorted(list((families_after - set(raw_custom_families)) | set(merged_custom_families)))
    # 4. 为VF字体补充虚拟样式并标记
    _weight_keywords = ["Thin", "Light", "Bold", "Black", "Medium", "Heavy", 
                        "SemiBold", "DemiBold", "ExtraBold", "UltraLight", "ExtraLight"]
    
    standard_vf_styles = ["Thin", "ExtraLight", "Light", "Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Black"]
    
    for family in shared.CUSTOM_FONT_FAMILIES:
        styles = shared.FONT_STYLES.get(family, [])
        # 判断条件：样式很少 + 家族名无字重后缀 -> 可能是VF字体
        if len(styles) <= 2 and not any(family.endswith(f" {w}") for w in _weight_keywords):
            virtual_set = set()
            supplemented = list(styles)
            for std_s in standard_vf_styles:
                if std_s not in supplemented:
                    supplemented.append(std_s)
                    virtual_set.add(std_s)
            
            supplemented.sort(key=get_sort_key)
            shared.FONT_STYLES[family] = supplemented
            if virtual_set:
                shared.VIRTUAL_FONT_STYLES[family] = virtual_set
    # ===== 归并排序结束 =====

    #  生成 Family 和 Style 映射
    for family in shared.ALL_FONT_FAMILIES:
        if family not in shared.FONT_STYLES:
            if shared.FLAG_QT6:
                styles = QFontDatabase.styles(family)
            else:
                styles = fdb.styles(family)
            shared.FONT_STYLES[family] = styles

        # ===== 新增：检测可变字体轴并补充虚拟样式 =====
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
    except ImportError:
        TTFont = None
        LOGGER.warning("fontTools not installed, variable font support disabled")
    if TTFont is not None and osp.exists(PATH_FONTS):
        _font_file_to_family = {}  # 文件路径 -> FamilyName 映射
        # 先建立文件到家族的反向映射
        for fp in find_all_files_recursive(PATH_FONTS, FONT_EXTS):
            try:
                temp_idx = QFontDatabase.addApplicationFont(fp)
                temp_families = QFontDatabase.applicationFontFamilies(temp_idx)
                for fam in temp_families:
                    if fam in shared.CUSTOM_FONT_FAMILIES:
                        _font_file_to_family[fp] = fam
            except:
                pass
        for fp, family in _font_file_to_family.items():
            try:
                tt = TTFont(fp)
                if 'fvar' in tt:
                    # 这是一个可变字体
                    axes = {}
                    for axis in tt['fvar'].axes:
                        axes[axis.axisTag] = (axis.minValue, axis.maxValue, axis.defaultValue)
                    shared.FONT_VARIABLE_AXES[family] = axes
                    # 如果 Qt 只返回了空或单一样式，为其生成虚拟样式
                    current_styles = shared.FONT_STYLES.get(family, [])
                    if len(current_styles) <= 1 and 'wght' in axes:
                        wght_min, wght_max, wght_default = axes['wght']
                        # 生成常见的字重节点
                        virtual_styles = []
                        weight_names = [
                            (100, "Thin"), (200, "ExtraLight"), (300, "Light"),
                            (400, "Regular"), (500, "Medium"), (600, "SemiBold"),
                            (700, "Bold"), (800, "ExtraBold"), (900, "Black")
                        ]
                        for wval, wname in weight_names:
                            if wght_min <= wval <= wght_max:
                                virtual_styles.append(wname)
                        if virtual_styles:
                            shared.FONT_STYLES[family] = virtual_styles
                    tt.close()
            except Exception as e:
                pass
    # ===== 新增结束 =====

    app_font = QFont('Microsoft YaHei UI')
    if not app_font.exactMatch() or sys.platform == 'darwin':
        app_font = app.font()
    app_font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    QGuiApplication.setFont(app_font)
    shared.DEFAULT_FONT_FAMILY = app_font.family()
    shared.APP_DEFAULT_FONT = app_font.family()
    
    if args.ldpi:
        shared.LDPI = args.ldpi

    setup_locks()

    from ui.mainwindow import MainWindow
    ballontrans = MainWindow(app, config, open_dir=args.proj_dir, **vars(args))
    global BT
    BT = ballontrans
    BT.restart_signal.connect(restart)

    if not args.headless and not args.headless_continuous:
        if shared.SCREEN_W > 1707 and sys.platform == 'win32':   # higher than 2560 (1440p) / 1.5
            # https://github.com/dmMaze/BallonsTranslator/issues/220
            BT.comicTransSplitter.setHandleWidth(7)

        ballontrans.setWindowIcon(QIcon(shared.ICON_PATH))
        ballontrans.show()
        ballontrans.resetStyleSheet()
    sys.exit(app.exec())

def is_amd_gpu():
    try:
        if sys.platform == 'win32':
            # Windows: use wmic
            cmd = 'wmic path win32_VideoController get name'
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            return any(keyword in output for keyword in ["AMD", "Radeon"])

        else:
            return False

    except Exception:
        return False

def supported_amd_nightly_gpu():
    try:
        if sys.platform == 'win32':
            # Windows: use wmic
            cmd = 'wmic path win32_VideoController get name'
            output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)

            if any(keyword in output for keyword in
                   ["RX 7900", "RX 7800", "RX 7700", "RX 7600", "PRO W7900", "PRO W7800", "PRO W7700"]):
                return "RDNA3"
            if any(keyword in output for keyword in
                   ["RX 9070", "RX 9060"]):
                return "RDNA4"
        else:
            return "None"

    except Exception:
        return "None"

def prepare_environment():

    try:
        import packaging
    except ModuleNotFoundError:
        run_pip(f"install packaging", "install packaging")

    from utils.package import check_req_file, check_reqs

    if getattr(sys, 'frozen', False):
        print('Running as app, skip dependency installation')
        return

    if args.frozen:
        return

    req_updated = False
    if sys.platform == 'win32':
        for req in REQ_WIN:
            if not check_reqs([req]):
                run_pip(f"install {req}", req)
                req_updated = True

    if is_amd_gpu():
        print('AMD GPU: Yes')
        if args.nightly:
            amd_nightly_gpu = supported_amd_nightly_gpu()
            if amd_nightly_gpu == "None":
                Exception("No AMD Nightly GPU supported")
            if amd_nightly_gpu == "RDNA3":
                torch_command = os.environ.get('TORCH_COMMAND',
                                               "pip install https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0%2Bgitfc14c65-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0%2Bc85f008-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchaudio-2.6.0a0%2B1a8f621-cp312-cp312-win_amd64.whl")
            if amd_nightly_gpu == "RDNA4":
                torch_command = os.environ.get('TORCH_COMMAND',
                                               "pip install https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0%2Bgitfc14c65-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0%2Bc85f008-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchaudio-2.6.0a0%2B1a8f621-cp312-cp312-win_amd64.whl")
        else:
            # AMD GPU: Cuda 11.8, Pytorch 2.2.2
            torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 --disable-pip-version-check")
    else:
        # Detect NVIDIA GPU architecture to pick the right CUDA version
        _torch_index = "https://download.pytorch.org/whl/cu124"
        try:
            _nvsmi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            _gpu_name = _nvsmi.stdout.strip()
            # Blackwell (RTX 50 series) needs CUDA 12.8+
            if any(name in _gpu_name for name in ["RTX 5090", "RTX 5080", "RTX 5070", "RTX 5060", "RTX 50"]):
                _torch_index = "https://download.pytorch.org/whl/nightly/cu128"
                print(f"Detected Blackwell GPU ({_gpu_name}), using CUDA 12.8+ PyTorch")
            else:
                print(f"Detected GPU: {_gpu_name}")
        except Exception:
            pass
        if "nightly" in _torch_index:
            torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch torchvision torchaudio --index-url {_torch_index} --disable-pip-version-check")
        else:
            torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url {_torch_index} --disable-pip-version-check")
    if args.reinstall_torch:
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
        req_updated = True

    if not check_req_file(args.requirements):
        run_pip(f"install -r {args.requirements}", "requirements")
        req_updated = True

    if req_updated:
        import site
        importlib.reload(site)





if __name__ == '__main__':
    main()

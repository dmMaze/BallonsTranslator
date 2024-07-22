from pathlib import Path
import sys
import argparse
import os.path as osp
import os
import importlib
import re
import subprocess
import importlib.util
import pkg_resources
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
parser.add_argument("--exec_dirs", default='', help='translation queue (project directories) separated by comma')
parser.add_argument("--ldpi", default=None, type=float, help='logical dots perinch')
parser.add_argument("--exec-subfolders", default='', type=str, help='Folder containing subdirectories to be added to --exec_dirs')
args, _ = parser.parse_known_args()
if args.exec_subfolders:
    subfolders = [os.path.join(args.exec_subfolders, d) for d in os.listdir(args.exec_subfolders) if os.path.isdir(os.path.join(args.exec_subfolders, d))]
    args.exec_dirs = ','.join(subfolders)


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
    return run(f'"{python}" -m pip {args} --prefer-binary{index_url_line} --disable-pip-version-check', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=True)


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def load_modules():

    def _load_module(module_dir: str, module_pattern: str):
        modules = os.listdir(module_dir)
        pattern = re.compile(module_pattern)
        module_path = module_dir.replace('/', '.')
        if not module_path.endswith('.'):
            module_path += '.'
        for module_name in modules:
            if pattern.match(module_name) is not None:
                importlib.import_module(module_path + module_name.replace('.py', ''))

    for kwargs in [
        {'module_dir': 'modules/translators', 'module_pattern': r'trans_(.*?).py'},
        {'module_dir': 'modules/textdetector', 'module_pattern': r'detector_(.*?).py'},
        {'module_dir': 'modules/inpaint', 'module_pattern': r'inpaint_(.*?).py'},
        {'module_dir': 'modules/ocr', 'module_pattern': r'ocr_(.*?).py'},
    ]:
        _load_module(**kwargs)

BT = None
APP = None

def restart():
    global BT
    print('restarting...\n')
    BT.close()
    os.execv(sys.executable, ['python'] + sys.argv)


def main():

    if args.debug:
        os.environ['BALLOONTRANS_DEBUG'] = '1'

    os.environ['QT_API'] = args.qt_api

    commit = commit_hash()

    print('py version: ', sys.version)
    print('py executable: ', sys.executable)
    print(f'version: {VERSION}')
    print(f'branch: {BRANCH}')
    print(f"Commit hash: {commit}")

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(APP_DIR)

    prepare_environment()

    from utils.logger import setup_logging, logger as LOGGER
    import utils.shared as shared
    from utils.io_utils import find_all_files_recursive
    from utils import config as program_config

    from qtpy.QtCore import QTranslator, QLocale, Qt
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name().replace('en_CN', 'zh_CN')
    shared.HEADLESS = args.headless
    shared.load_cache()
    program_config.load_config()
    config = program_config.pcfg

    if args.headless:
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
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    os.chdir(shared.PROGRAM_PATH)

    setup_logging(shared.LOGGING_PATH)

    load_modules()
    from modules.prepare_local_files import prepare_local_files_forall
    prepare_local_files_forall()

    app_args = sys.argv
    if args.headless:
        app_args = sys.argv + ['-platform', 'offscreen']
    app = QApplication(app_args)

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
    if osp.exists(PATH_FONTS):
        for fp in find_all_files_recursive(PATH_FONTS, FONT_EXTS):
            fnt_idx = QFontDatabase.addApplicationFont(fp)
            if fnt_idx >= 0:
                shared.CUSTOM_FONTS.append(QFontDatabase.applicationFontFamilies(fnt_idx)[0])
    
    if sys.platform == 'win32' and args.headless:
        # font database does not initialise on windows with qpa -offscreen: 
        # whttps://github.com/dmMaze/BallonsTranslator/issues/519
        from qtpy.QtCore import QStandardPaths
        font_dir_list = QStandardPaths.standardLocations(QStandardPaths.FontsLocation)
        for fd in font_dir_list:
            fp_list = find_all_files_recursive(fd, FONT_EXTS)
            for fp in fp_list:
                fnt_idx = QFontDatabase.addApplicationFont(fp)

    if shared.FLAG_QT6:
        shared.FONT_FAMILIES = set(f for f in QFontDatabase.families())
    else:
        fdb = QFontDatabase()
        shared.FONT_FAMILIES = set(fdb.families())

    yahei = QFont('Microsoft YaHei UI')
    if yahei.exactMatch() and not sys.platform == 'darwin':
        QGuiApplication.setFont(yahei)
        shared.DEFAULT_FONT_FAMILY = 'Microsoft YaHei UI'
        shared.APP_DEFAULT_FONT = 'Microsoft YaHei UI'
    else:
        app_font = app.font().family()
        shared.DEFAULT_FONT_FAMILY = app_font
        shared.APP_DEFAULT_FONT = app_font

    shared.APP_DEFAULT_FONT = app.font().defaultFamily()
    if args.ldpi:
        shared.LDPI = args.ldpi

    from ui.mainwindow import MainWindow

    ballontrans = MainWindow(app, config, open_dir=args.proj_dir, **vars(args))
    global BT
    BT = ballontrans
    BT.restart_signal.connect(restart)

    if not args.headless:
        ps = QGuiApplication.primaryScreen()
        shared.LDPI = ps.logicalDotsPerInch()
        shared.SCREEN_W = ps.geometry().width()
        shared.SCREEN_H = ps.geometry().height()
        if shared.SCREEN_W > 1707 and sys.platform == 'win32':   # higher than 2560 (1440p) / 1.5
            # https://github.com/dmMaze/BallonsTranslator/issues/220
            BT.comicTransSplitter.setHandleWidth(10)

        ballontrans.setWindowIcon(QIcon(shared.ICON_PATH))
        ballontrans.show()
        ballontrans.resetStyleSheet()
    sys.exit(app.exec())

def prepare_environment():
    if getattr(sys, 'frozen', False):
        print('Running as app, skip dependency installation')
        return

    req_updated = False
    if sys.platform == 'win32':
        for req in REQ_WIN:
            try:
                pkg_resources.require(req)
            except Exception:
                run_pip(f"install {req}", req)
                req_updated = True

    torch_command = os.environ.get('TORCH_COMMAND', "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --disable-pip-version-check")
    if args.reinstall_torch or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)
        req_updated = True
    try:
        pkg_resources.require(open(args.requirements,mode='r', encoding='utf8'))
    except Exception as e:
        print(e)
        run_pip(f"install -r {args.requirements}", "requirements")
        req_updated = True

    if req_updated:
        import site
        importlib.reload(site)

if __name__ == '__main__':
    main()

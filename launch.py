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
import time

T_LAUNCH = time.time()

python = sys.executable
git = os.environ.get('GIT', "git")
skip_install = False
index_url = os.environ.get('INDEX_URL', "")
QT_APIS = ['pyqt5', 'pyqt6', 'pyside6']
stored_commit_hash = None

REQ_WIN = [
    'pywin32'
]

PATH_ROOT=Path(__file__).parent  
PATH_FONTS=PATH_ROOT/'fonts'


parser = argparse.ArgumentParser()
parser.add_argument("--reinstall-torch", action='store_true', help="launch.py argument: install the appropriate version of torch even if you have some version already installed")
parser.add_argument("--proj-dir", default='', type=str, help='Open project directory on startup')
parser.add_argument("--qt-api", default='', choices=QT_APIS, help='Set qt api')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--requirements", default='requirements.txt')
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


TRANSLATOR_DIR = 'modules/translators'
TRANSLATOR_PATTERN = re.compile(r'trans_(.*?).py') 
def load_translators(translators = None):
    if translators is None:
        translators = os.listdir(TRANSLATOR_DIR)

    for translator in translators:
        if TRANSLATOR_PATTERN.match(translator) is not None:
            importlib.import_module('modules.translators.' + translator.replace('.py', ''))


def load_modules():
    load_translators()

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

    from utils import appinfo

    commit = commit_hash()

    print('py version: ', sys.version)
    print('py executable: ', sys.executable)
    print(f'version: {appinfo.version}')
    print(f'branch: {appinfo.branch}')
    print(f"Commit hash: {commit}")

    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(APP_DIR)

    prepare_environment()

    from utils.logger import setup_logging, logger as LOGGER
    import utils.shared as shared
    from utils import config as program_config
    from utils.config import ProgramConfig

    from qtpy.QtCore import QTranslator, QLocale, Qt
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name()

    shared.load_cache()

    if osp.exists(shared.CONFIG_PATH):
        try:
            config = ProgramConfig.load(shared.CONFIG_PATH)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            config = ProgramConfig()
    else:
        LOGGER.info(f'{shared.CONFIG_PATH} does not exist, new config file will be created.')
        config = ProgramConfig()
    program_config.pcfg = config

    from modules.prepare_local_files import prepare_local_files_forall

    prepare_local_files_forall()

    if not args.qt_api in QT_APIS:
        os.environ['QT_API'] = 'pyqt6'
    else:
        os.environ['QT_API'] = args.qt_api

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

    app = QApplication(sys.argv)

    lang = config.display_lang
    langp = osp.join(shared.TRANSLATE_DIR, lang + '.qm')
    if osp.exists(langp):
        translator = QTranslator()
        translator.load(lang, osp.dirname(osp.abspath(__file__)) + "/translate")
        app.installTranslator(translator)
    elif lang not in ('en_US', 'English'):
        LOGGER.warning(f'target display language file {langp} doesnt exist.')
    LOGGER.info(f'set display language to {lang}')

    ps = QGuiApplication.primaryScreen()
    shared.LDPI = ps.logicalDotsPerInch()
    shared.SCREEN_W = ps.geometry().width()
    shared.SCREEN_H = ps.geometry().height()

    # Fonts
    # Load custom fonts if they exist
    for font in os.listdir(PATH_FONTS):
        if font.endswith(('.ttf','.otf')):
            QFontDatabase.addApplicationFont((PATH_FONTS/font).as_posix())
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

    from ui.mainwindow import MainWindow

    ballontrans = MainWindow(app, config, open_dir=args.proj_dir)
    global BT
    BT = ballontrans
    BT.restart_signal.connect(restart)

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

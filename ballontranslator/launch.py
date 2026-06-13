from pathlib import Path
import sys
import argparse
import os.path as osp
import os
import subprocess
from platform import platform

BRANCH = 'dev'
VERSION = '1.4.0'

git = os.environ.get('GIT', "git")
QT_APIS = ['pyqt6', 'pyside6', 'pyqt5', 'pyside2']
stored_commit_hash = None

FONT_EXTS = {'.ttf','.otf','.ttc','.pfb'}

IS_WIN7 = "Windows-7" in platform()

import ballontranslator.utils.shared as shared # Earlier import of shared to use default for config_path argument

PATH_ROOT = Path(shared.PROGRAM_PATH)
PATH_FONTS = str(PATH_ROOT / 'fonts')

parser = argparse.ArgumentParser()
parser.add_argument("--proj-dir", default='', type=str, help='Open project directory on startup')
if IS_WIN7:
    parser.add_argument("--qt-api", default='pyqt5', choices=QT_APIS, help='Set qt api')
else:
    parser.add_argument("--qt-api", default='pyqt6', choices=QT_APIS, help='Set qt api')
parser.add_argument("--debug", action='store_true')
parser.add_argument("--system_hf_cache", action='store_true', help="use system huggingface cache directory instead of ./data/models")
parser.add_argument("--headless", action='store_true', help='run without GUI and prompt for new exec_dirs after finishing until user exits the program')
parser.add_argument("--exec_dirs", default='', help='translation queue (project directories) separated by comma')
parser.add_argument("--ldpi", default=None, type=float, help='logical dots perinch')
parser.add_argument("--export-translation-txt", action='store_true', help='save translation to txt file once RUN completed')
parser.add_argument("--export-source-txt", action='store_true', help='save source to txt file once RUN completed')
parser.add_argument("--config_path", default=shared.CONFIG_PATH, help='Config file to use for translation') # Named config_path to avoid conflict with existing name config
if "--headless_continuous" in sys.argv[1:]:
    parser.error("--headless_continuous has been renamed to --headless")
args, _ = parser.parse_known_args()


BT = None
APP = None

def restart():
    global BT
    print('restarting...\n')
    if BT:
        BT.close()
    argv = list(sys.argv)
    try:
        main_path = Path(__file__).resolve().parents[0] / '__main__.py'
        if Path(argv[0]).resolve() == main_path:
            argv = ['-m', 'ballontranslator', *argv[1:]]
    except Exception:
        pass
    os.execv(sys.executable, [sys.executable] + argv)


def setup_locks():
    from ballontranslator.utils.lock import RUNTIME_LOCKS
    from qtpy.QtCore import QMutex
    RUNTIME_LOCKS['model_loading'] = QMutex()


def preload_msvc_runtime():
    """Best-effort preload of the MSVC runtime before Qt alters DLL lookup.

    PyQt6 registers its bundled Qt bin directory with ``AddDllDirectory`` on
    import. On Windows this can make later PyTorch DLL initialization resolve
    ``msvcp140.dll`` from PyQt6's older bundled copy instead of the system
    runtime, so this must run before any ``qtpy``/``PyQt6`` import.

    >>> preload_msvc_runtime() in (True, False)
    True
    """

    if sys.platform != 'win32':
        return False

    import ctypes

    loaded = False
    for dll_name in ('vcruntime140.dll', 'msvcp140.dll', 'vcruntime140_1.dll'):
        try:
            ctypes.CDLL(dll_name)
            loaded = True
        except OSError:
            if dll_name == 'msvcp140.dll':
                print(
                    'Microsoft Visual C++ Redistributable is not installed or '
                    'is not visible to this process. Deep learning modules may '
                    'fail to load until the x64 VC runtime is installed.'
                )
    return loaded


def core_requirements_env(config_path: str) -> dict:
    """Return the environment used by launch-time core dependency repair.

    >>> env = core_requirements_env('/path/that/does/not/exist')
    >>> isinstance(env, dict)
    True
    """

    from ballontranslator.utils.network_mirrors import (
        installer_env_with_pypi_mirror,
        read_saved_pypi_mirror,
    )

    return installer_env_with_pypi_mirror(os.environ.copy(), read_saved_pypi_mirror(config_path))


def setup_network_mirrors(config, config_path: str, qt_locale_name: str, program_config_module, logger) -> list:
    """Backfill and apply network mirror settings after config loading.

    >>> class Mirrors:
    ...     huggingface = None
    ...     pypi = None
    >>> class Config:
    ...     mirrors = Mirrors()
    >>> class ProgramConfig:
    ...     @staticmethod
    ...     def save_config():
    ...         return True
    >>> setup_network_mirrors(Config(), '/path/that/does/not/exist', 'en_US', ProgramConfig, logger=None)
    []
    """

    from ballontranslator.utils.network_mirrors import (
        backfill_missing_mirror_defaults,
        collect_system_locale_names,
        collect_system_timezone_names,
        missing_mirror_fields,
        normalize_mirror_value,
        should_use_china_mirrors,
    )

    def log_info(message: str):
        if logger is not None:
            logger.info(message)

    missing_mirrors = missing_mirror_fields(config_path)
    locale_names = collect_system_locale_names(qt_locale_name)
    timezone_names = collect_system_timezone_names()
    log_info(
        'Checking network mirror defaults. Missing mirror fields: '
        f'{", ".join(sorted(missing_mirrors)) if missing_mirrors else "none"}'
    )
    if missing_mirrors:
        use_china_mirrors = should_use_china_mirrors(locale_names, timezone_names)
        log_info(f'Network mirror heuristic locale hints: {locale_names}')
        log_info(f'Network mirror heuristic timezone hints: {timezone_names}')
        log_info(
            'Network mirror heuristic result: '
            f'{"mainland China detected" if use_china_mirrors else "mainland China not detected"}'
        )
    else:
        log_info('Network mirror config fields are present; skipping automatic mirror selection.')

    updated_mirrors = backfill_missing_mirror_defaults(
        config.mirrors,
        missing_mirrors,
        locale_names=locale_names,
        timezone_names=timezone_names,
    )
    if updated_mirrors:
        log_info(f'Automatically selected network mirrors for: {", ".join(updated_mirrors)}')
    elif missing_mirrors:
        log_info('No network mirrors were selected automatically.')
    if missing_mirrors:
        program_config_module.save_config()

    huggingface_mirror = normalize_mirror_value(config.mirrors.huggingface)
    if huggingface_mirror:
        os.environ['HF_ENDPOINT'] = huggingface_mirror
        log_info(f'Using Hugging Face mirror endpoint: {huggingface_mirror}')
    else:
        log_info('Hugging Face mirror endpoint: none')
    pypi_mirror = normalize_mirror_value(config.mirrors.pypi)
    if pypi_mirror:
        log_info(f'Using PyPI package mirror: {pypi_mirror}')
    else:
        log_info('PyPI package mirror: none')
    return updated_mirrors


def main():

    if args.debug:
        os.environ['BALLOONTRANS_DEBUG'] = '1'

    os.environ['QT_API'] = args.qt_api
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    APP_DIR = shared.PROGRAM_PATH
    os.chdir(APP_DIR)

    print('Python version: ', sys.version)
    print('Python executable: ', sys.executable)
    print(f'Version: {VERSION}')
    print(f'Branch: {BRANCH}')

    if not args.system_hf_cache:
        os.environ['HF_HOME'] = osp.join(APP_DIR, 'data/models')

    preload_msvc_runtime()

    from ballontranslator.utils.core_requirements import ensure_core_requirements
    if ensure_core_requirements(APP_DIR, env=core_requirements_env(args.config_path)):
        print('Core requirements updated. Restarting...')
        restart()
        return

    from ballontranslator.utils.logger import setup_logging, logger as LOGGER
    from ballontranslator.utils.io_utils import find_all_files_recursive
    from ballontranslator.utils import config as program_config

    from qtpy.QtCore import QTranslator, QLocale, Qt
    setup_logging(shared.LOGGING_PATH)
    shared.args = args
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name().replace('en_CN', 'zh_CN')
    shared.HEADLESS = args.headless
    shared.load_cache()
    program_config.load_config(args.config_path)
    config = program_config.pcfg

    if args.headless:
        config.module.empty_runcache = False

    updated_mirrors = setup_network_mirrors(
        config,
        args.config_path,
        QLocale.system().name(),
        program_config,
        LOGGER,
    )

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

    app_args = sys.argv
    if args.headless:
        app_args = sys.argv + ['-platform', 'offscreen']
    app = QApplication(app_args)
    app.setApplicationName('BalloonsTranslator')
    app.setApplicationVersion(VERSION)

    if not args.headless:
        ps = QGuiApplication.primaryScreen()
        shared.LDPI = ps.logicalDotsPerInch()
        shared.SCREEN_W = ps.geometry().width()
        shared.SCREEN_H = ps.geometry().height()

    lang = config.display_lang
    langp = osp.join(shared.TRANSLATE_DIR, lang + '.qm')
    if osp.exists(langp):
        translator = QTranslator()
        translator.load(lang, shared.TRANSLATE_DIR)
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
        font_dir_list = QStandardPaths.standardLocations(QStandardPaths.StandardLocation.FontsLocation)
        for fd in font_dir_list:
            fp_list = find_all_files_recursive(fd, FONT_EXTS)
            for fp in fp_list:
                fnt_idx = QFontDatabase.addApplicationFont(fp)

    if shared.FLAG_QT6:
        shared.FONT_FAMILIES = set(f for f in QFontDatabase.families())
    else:
        fdb = QFontDatabase()
        shared.FONT_FAMILIES = set(fdb.families())

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

    from ballontranslator.ui.mainwindow import MainWindow
    from ballontranslator.utils.message import create_info_dialog
    ballontrans = MainWindow(app, config, open_dir=args.proj_dir, **vars(args))
    global BT
    BT = ballontrans
    BT.restart_signal.connect(restart)

    if not args.headless:
        if shared.SCREEN_W > 1707 and sys.platform == 'win32':   # higher than 2560 (1440p) / 1.5
            # https://github.com/dmMaze/BallonsTranslator/issues/220
            BT.comicTransSplitter.setHandleWidth(7)

        ballontrans.setWindowIcon(QIcon(shared.ICON_PATH))
        ballontrans.show()
        ballontrans.resetStyleSheet()
    if updated_mirrors:
        create_info_dialog(QApplication.translate(
            'NetworkMirrors',
            'Network mirrors were selected automatically for better access to dependencies and model downloads.',
        ))
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

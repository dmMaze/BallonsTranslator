from pathlib import Path
import sys
import argparse
import os.path as osp
import os
import shutil
import subprocess
from platform import platform


git = os.environ.get('GIT', "git")
QT_APIS = ['pyqt6', 'pyside6', 'pyqt5', 'pyside2']
stored_commit_hash = None

FONT_EXTS = {'.ttf','.otf','.ttc','.pfb'}

IS_WIN7 = "Windows-7" in platform()

def disable_bundled_windows_user_site() -> list:
    """Remove per-user packages from the bundled Windows Python runtime.

    >>> isinstance(disable_bundled_windows_user_site(), list)
    True
    """

    if sys.platform != 'win32':
        return []

    executable_dir = Path(sys.executable).parent
    if executable_dir.name.lower() != 'ballontrans_pylibs_win':
        return []

    os.environ['PYTHONNOUSERSITE'] = '1'
    try:
        import site
    except Exception:
        return []

    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = getattr(site, 'USER_SITE', None)
    if not user_site:
        return []

    user_site_paths = user_site if isinstance(user_site, (list, tuple)) else [user_site]
    blocked_paths = {
        osp.normcase(osp.abspath(path))
        for path in user_site_paths
        if path
    }
    removed = []
    remaining = []
    for path in sys.path:
        if path and osp.normcase(osp.abspath(path)) in blocked_paths:
            removed.append(path)
        else:
            remaining.append(path)

    if removed:
        sys.path[:] = remaining

    # Keep later imports and subprocess restarts from re-enabling AppData packages.
    site.ENABLE_USER_SITE = False
    return removed


disable_bundled_windows_user_site()

import ballontranslator.utils.shared as shared # Earlier import of shared to use default for config_path argument
from ballontranslator.utils.version import APP_VERSION

os.environ['NUMBA_CACHE_DIR'] = osp.join(shared.cache_dir, 'numba')

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
parser.add_argument(
    "--show-release-info",
    "--show_release_info",
    dest="show_release_info",
    action='store_true',
    help='show cached GitHub release information on startup without making an API request',
)
parser.add_argument(
    "--config",
    "--config_path",
    dest="config_path",
    default=shared.CONFIG_PATH,
    help='Config file to use for translation',
)
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


def ensure_resource_theme_files(program_path: str = None, logger=None) -> list:
    """Copy moved stylesheet/theme files from the old config location if needed.

    >>> ensure_resource_theme_files('/path/that/does/not/exist')
    []
    """

    root = Path(program_path or shared.PROGRAM_PATH)
    copied = []
    for filename in ('stylesheet.css', 'themes.json'):
        target_path = root / 'resources' / filename
        if target_path.exists():
            continue

        source_path = root / 'config' / filename
        if not source_path.exists():
            continue

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied.append(filename)
            if logger is not None:
                logger.info(f'Copied missing resource file from old config path: {filename}')
        except OSError as e:
            if logger is not None:
                logger.warning(f'Failed to copy missing resource file {filename}: {e}')
    return copied


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


def main():

    if args.debug:
        os.environ['BALLOONTRANS_DEBUG'] = '1'

    os.environ['QT_API'] = args.qt_api
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    APP_DIR = shared.PROGRAM_PATH
    os.chdir(APP_DIR)

    print('Python version: ', sys.version)
    print('Python executable: ', sys.executable)
    print(f'Version: {APP_VERSION}')

    if not args.system_hf_cache:
        os.environ['HF_HOME'] = osp.join(APP_DIR, 'data/models')

    preload_msvc_runtime()

    from ballontranslator.utils.logger import setup_logging, logger as LOGGER
    from ballontranslator.utils.network_mirrors import auto_fill_network_mirrors
    setup_logging(shared.LOGGING_PATH)
    ensure_resource_theme_files(APP_DIR, LOGGER)
    updated_mirrors = auto_fill_network_mirrors(args.config_path, LOGGER)

    from ballontranslator.utils.core_requirements import ensure_core_requirements
    if ensure_core_requirements(APP_DIR, env=core_requirements_env(args.config_path)):
        print('Core requirements updated. Restarting...')
        restart()
        return

    from ballontranslator.utils.io_utils import find_all_files_recursive
    from ballontranslator.utils import config as program_config

    from qtpy.QtCore import QTranslator, QLocale, Qt, QTimer
    shared.args = args
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name().replace('en_CN', 'zh_CN')
    shared.HEADLESS = args.headless
    shared.load_cache()
    program_config.load_config(args.config_path)
    config = program_config.pcfg

    if args.headless:
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

    if sys.platform == 'win32':
        application_attribute = getattr(Qt, 'ApplicationAttribute', Qt)
        QApplication.setAttribute(
            application_attribute.AA_DontCreateNativeWidgetSiblings,
            True,
        )

    os.chdir(shared.PROGRAM_PATH)

    app_args = sys.argv
    if args.headless:
        app_args = sys.argv + ['-platform', 'offscreen']
    app = QApplication(app_args)
    app.setApplicationName('BalloonsTranslator')
    app.setApplicationVersion(APP_VERSION)

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
        font_database = QFontDatabase
    else:
        font_database = QFontDatabase()
    shared.FONT_FAMILIES = set(font_database.families())

    from ballontranslator.ui.text_engine.font_family import (
        register_qt_font_family_aliases,
    )
    font_aliases = register_qt_font_family_aliases(
        shared.FONT_FAMILIES,
        font_database.styles,
    )
    if font_aliases:
        LOGGER.info(
            'Registered Qt-safe aliases for %d font families.',
            len(font_aliases),
        )

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
    delete_on_close = getattr(Qt, 'WidgetAttribute', Qt).WA_DeleteOnClose
    # Destroy the Qt window tree before SIP performs interpreter-exit cleanup.
    ballontrans.setAttribute(delete_on_close, True)
    global BT
    BT = ballontrans
    BT.restart_signal.connect(restart)

    if not args.headless:
        # if shared.SCREEN_W > 1707 and sys.platform == 'win32':   # higher than 2560 (1440p) / 1.5
        #     # https://github.com/dmMaze/BallonsTranslator/issues/220
        BT.comicTransSplitter.setHandleWidth(7)

        ballontrans.setWindowIcon(QIcon(shared.ICON_PATH))
        ballontrans.show()
        if shared.ON_WINDOWS:
            from ballontranslator.ui.framelesswindow import FramelessMoveResize
            # SC_MAXIMIZE animates only after the normal window is visible.
            QTimer.singleShot(
                0,
                lambda: FramelessMoveResize.maximize(ballontrans),
            )
    if updated_mirrors:
        create_info_dialog(QApplication.translate(
            'NetworkMirrors',
            'Network mirrors were selected automatically for better access to dependencies and model downloads.',
        ))
    # Let this frame release Qt objects before SIP's interpreter-exit cleanup.
    return app.exec()


if __name__ == '__main__':
    main()

import sys
import argparse
import os.path as osp
import os
# from utils.logger import logger as LOGGER
from utils.logger import setup_logging, logger as LOGGER

QT_APIS = ['pyqt5', 'pyqt6']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj-dir", default='', type=str, help='Open project directory on startup')
    parser.add_argument("--qt-api", default='', choices=QT_APIS, help='Set qt api')
    args = parser.parse_args()

    if not args.qt_api in QT_APIS:
        os.environ['QT_API'] = 'pyqt6'
    else:
        os.environ['QT_API'] = args.qt_api

    if sys.platform == 'darwin':
        os.environ['QT_API'] = 'pyqt6'
        LOGGER.info('running on macOS, set QT_API to pyqt6')

    if sys.platform == 'win32':
        import ctypes
        myappid = u'BalloonsTranslator' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    import qtpy
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import QTranslator, QLocale, Qt
    from qtpy.QtGui import QIcon
    from qtpy.QtGui import  QGuiApplication, QIcon, QFont

    from ui import constants as C
    if qtpy.API_NAME[-1] == '6':
        C.FLAG_QT6 = True
    else:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    os.chdir(C.PROGRAM_PATH)

    setup_logging(C.LOGGING_PATH)

    app = QApplication(sys.argv)
    translator = QTranslator()
    translator.load(
        QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + "/data/translate",
    )
    app.installTranslator(translator)

    C.LDPI = QGuiApplication.primaryScreen().logicalDotsPerInch()
    yahei = QFont('Microsoft YaHei UI')
    if yahei.exactMatch():
        QGuiApplication.setFont(yahei)

    from ui.mainwindow import MainWindow
    ballontrans = MainWindow(app, open_dir=args.proj_dir)
    ballontrans.setWindowIcon(QIcon(C.ICON_PATH))
    ballontrans.show()
    ballontrans.resetStyleSheet()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

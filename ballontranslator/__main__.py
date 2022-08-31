import sys
import argparse
import os.path as osp
import os

QT_APIS = ['pyqt5', 'pyqt6']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj-dir", default='', type=str, help='Open project directory on startup')
    parser.add_argument("--qt-api", default='', choices=QT_APIS, help='Set qt api')
    args = parser.parse_args()

    if not args.qt_api in QT_APIS:
        os.environ['QT_API'] = 'pyqt5'
    else:
        os.environ['QT_API'] = args.qt_api

    if sys.platform == 'win32':
        import ctypes
        myappid = u'BalloonsTranslator' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    import qtpy
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import QTranslator, QLocale
    from qtpy.QtGui import QIcon

    from ui import constants
    if qtpy.API_NAME[-1] == '6':
        constants.FLAG_QT6 = True

    from ui.mainwindow import MainWindow
    
    os.chdir(constants.PROGRAM_PATH)
    app = QApplication(sys.argv)
    translator = QTranslator()
    translator.load(
        QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + "/data/translate",
    )
    app.installTranslator(translator)

    ballontrans = MainWindow(app, open_dir=args.proj_dir)
    ballontrans.setWindowIcon(QIcon(constants.ICON_PATH))
    ballontrans.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

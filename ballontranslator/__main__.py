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

    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import QTranslator, QLocale

    from ui.mainwindow import MainWindow
    from ui.constants import PROGRAM_PATH

    os.chdir(PROGRAM_PATH)
    app = QApplication(sys.argv)
    translator = QTranslator()
    translator.load(
        QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + "/data/translate",
    )
    app.installTranslator(translator)
    ballontrans = MainWindow(app, open_dir=args.proj_dir)

    ballontrans.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

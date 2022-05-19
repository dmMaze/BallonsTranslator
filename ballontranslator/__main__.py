import sys
import argparse
import os.path as osp
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTranslator, QLocale
from ui.mainwindow import MainWindow
from ui.constants import PROGRAM_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj-dir", default='', type=str, help='Open project directory on startup')
    args = parser.parse_args()

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

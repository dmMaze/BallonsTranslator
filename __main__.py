import sys
import os.path as osp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTranslator, QLocale
from ui.mainwindow import MainWindow
from ui.constants import PROGRAM_PATH, LIBS_PATH

if __name__ == '__main__':
    import os
    os.chdir(PROGRAM_PATH)
    app = QApplication(sys.argv)
    translator = QTranslator()
    translator.load(
        QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + "/translate",
    )
    app.installTranslator(translator)
    ballontrans = MainWindow(app)

    # ballontrans.openDir(r'data/testpacks/manga2')
    ballontrans.show()
    sys.exit(app.exec())
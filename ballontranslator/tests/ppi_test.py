import sys, os

from qtpy.QtCore import Qt, QRectF, QRect
from qtpy.QtGui import QPixmap, QImage, QPainter, QFont, QColor
from qtpy.QtWidgets import QApplication, QWidget

def pt2px(pt):
    print(LDPI)
    return int(round(pt * LDPI / 72.))

def px2pt(px):
    return px / LDPI * 72.

class MyWidget(QWidget):
    def __init__(self):
        global LDPI, DPI
        super().__init__()
        image = QImage(1000, 1000, QImage.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        from qtpy.QtGui import QGuiApplication
        DPI = QGuiApplication.primaryScreen().physicalDotsPerInch()
        LDPI = QGuiApplication.primaryScreen().logicalDotsPerInch()
        print(f'DPI: {DPI}, LDPI: {LDPI}')
        
        p = QPainter(image)
        p.setPen(Qt.GlobalColor.black)
        font = QFont("华文彩云")
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        px = 20
        font.setPixelSize(px)
        p.setFont(font)
        p.drawText(QRectF(0, 0, 1000, 1000), 'Hello淦')
        print(px2pt(px))
        font.setPointSizeF(px2pt(px))
        p.setFont(font)
        p.setPen(QColor(255, 0, 0, 127))
        p.drawText(QRectF(0, 0, 1000, 1000), 'Hello淦')
        image.save('data/px2pt.png', 'PNG')

        image.fill(Qt.GlobalColor.transparent)
        p.setPen(Qt.GlobalColor.black)
        pt = 200
        font.setPointSizeF(pt)
        p.setFont(font)
        p.drawText(QRectF(0, 0, 1000, 1000), 'Hello淦')
        font.setPixelSize(pt2px(pt))
        p.setFont(font)
        p.setPen(QColor(255, 0, 0, 127))
        p.drawText(QRectF(0, 0, 1000, 1000), 'Hello淦')
        image.save('data/pt2px.png', 'PNG')

        p.end()

if __name__ == '__main__':
 
    app = QApplication(sys.argv)
    W = MyWidget()
    W.show()
    sys.exit(app.exec())
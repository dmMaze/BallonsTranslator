"""Render vertical Roman descender overflow through the production scene."""

from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import QColor, QFont, QImage, QPainter, QPen
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


OUTPUT_DIR = Path('tests/ui/artifacts/vertical_roman_overflow')


def _item(origin: tuple[int, int], *, standard: bool) -> TextBlkItem:
    width, height = 180, 520
    bounds = [
        origin[0],
        origin[1],
        origin[0] + width,
        origin[1] + height,
    ]
    block = TextBlock(bounds, text_layout_version=TEXT_LAYOUT_VERSION)
    block._bounding_rect = list(bounds)
    block.translation = '一般abcg'
    block.fontformat.vertical = True
    block.fontformat.standard_vertical_roman_alignment = standard
    block.fontformat.font_family = 'Noto Sans CJK SC'
    block.fontformat.font_size = 96
    block.fontformat.frgb = [26, 29, 34]
    block.fontformat.stroke_width = 0.04
    block.fontformat.shadow_strength = 0.65
    block.fontformat.shadow_radius = 0.035
    block.fontformat.shadow_offset = [0.035, 0.035]
    item = TextBlkItem(block, 0)
    item.set_ui_guide_suppressed(True)
    return item


def render(scale: int) -> QImage:
    width, height = 650, 650
    image = QImage(
        width * scale,
        height * scale,
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(QColor('#f2efe9'))
    scene = QGraphicsScene()
    scene.setSceneRect(QRectF(0, 0, width, height))
    title = scene.addSimpleText(
        'Vertical Roman ink overflow — rotated / upright',
        QFont('Sans Serif', 13),
    )
    title.setBrush(QColor('#5e5952'))
    title.setPos(22, 10)

    items = (
        _item((85, 78), standard=False),
        _item((390, 78), standard=True),
    )
    labels = ('rotated lowercase', 'standard upright')
    for item, label_text in zip(items, labels):
        scene.addItem(item)
        item.repaint_background(render_scale=float(scale))
        logical = item.mapRectToScene(item.logical_unpadded_rect())
        guide = scene.addRect(
            logical,
            QPen(QColor('#cc5b4f'), 1.0, Qt.PenStyle.DashLine),
        )
        guide.setZValue(10)
        label = scene.addSimpleText(label_text, QFont('Sans Serif', 10))
        label.setBrush(QColor('#756e65'))
        label.setPos(logical.left(), logical.top() - 24)

    painter = QPainter(image)
    try:
        scene.render(
            painter,
            QRectF(0, 0, image.width(), image.height()),
            scene.sceneRect(),
        )
    finally:
        painter.end()
    return image


def main() -> None:
    _app = QApplication.instance() or QApplication([])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scale in (1, 3):
        path = OUTPUT_DIR / f'vertical_roman_overflow_fixture_{scale}x.png'
        if not render(scale).save(str(path)):
            raise RuntimeError(f'unable to save {path}')
        print(path.resolve())


if __name__ == '__main__':
    main()

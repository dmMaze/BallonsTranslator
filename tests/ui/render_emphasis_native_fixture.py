"""Render native-document emphasis marks through the production scene path."""

from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QFont, QImage, QPainter, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.text_engine.annotations import (
    apply_emphasis,
    apply_ruby,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.fontformat import TextTransformStack
from ballontranslator.utils.textblock import TextBlock


OUTPUT_DIR = Path('tests/ui/artifacts/emphasis_native')


def _item(
    text: str,
    *,
    vertical: bool,
    style: str,
    position: str,
    origin: tuple[int, int],
    gradient: tuple[list[int], list[int]],
    ruby: bool = False,
) -> TextBlkItem:
    width, height = ((530, 150) if not vertical else (230, 480))
    bounds = [
        origin[0],
        origin[1],
        origin[0] + width,
        origin[1] + height,
    ]
    block = TextBlock(bounds)
    block._bounding_rect = list(bounds)
    block.translation = text
    block.vertical = vertical
    block.fontformat.font_family = 'DejaVu Sans'
    block.fontformat.font_size = 48
    block.fontformat.stroke_width = 0.18
    block.fontformat.srgb = [34, 38, 48]
    block.fontformat.shadow_strength = 0.75
    block.fontformat.shadow_radius = 0.08
    block.fontformat.shadow_offset = [0.07, 0.07]
    block.fontformat.gradient_enabled = True
    block.fontformat.gradient_start_color = gradient[0]
    block.fontformat.gradient_end_color = gradient[1]
    block.fontformat.text_transform = TextTransformStack((), 11.0)
    item = TextBlkItem(block, 0)
    cursor = QTextCursor(item.document())
    cursor.select(QTextCursor.SelectionType.Document)
    apply_emphasis(cursor, style, position)
    if ruby:
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_ruby(cursor, 'group', 'えん', 'over')
    item.layout.reLayoutEverything()
    item.set_ui_guide_suppressed(True)
    return item


def render(scale: int) -> QImage:
    width, height = 1120, 700
    image = QImage(
        width * scale,
        height * scale,
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(QColor('#eeeae2'))
    scene = QGraphicsScene()
    scene.setSceneRect(QRectF(0, 0, width, height))
    title = scene.addSimpleText(
        'Native emphasis documents — thick stroke at 1x / 3x',
        QFont('Sans Serif', 13),
    )
    title.setBrush(QColor('#5d584f'))
    title.setPos(24, 10)
    fixtures = (
        _item(
            'OPEN CIRCLE',
            vertical=False,
            style='open circle',
            position='over right',
            origin=(30, 75),
            gradient=([225, 65, 45], [45, 95, 225]),
        ),
        _item(
            'FILLED CIRCLE',
            vertical=False,
            style='filled circle',
            position='under left',
            origin=(30, 285),
            gradient=([35, 160, 100], [125, 55, 200]),
            ruby=True,
        ),
        _item(
            '開放円形',
            vertical=True,
            style='open circle',
            position='over right',
            origin=(650, 80),
            gradient=([220, 70, 35], [35, 105, 220]),
        ),
        _item(
            '塗潰円形',
            vertical=True,
            style='filled circle',
            position='under left',
            origin=(890, 80),
            gradient=([30, 155, 95], [135, 55, 205]),
            ruby=True,
        ),
    )
    for item in fixtures:
        scene.addItem(item)
        item.repaint_background(render_scale=float(scale))
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
        path = OUTPUT_DIR / f'emphasis_native_fixture_{scale}x.png'
        if not render(scale).save(str(path)):
            raise RuntimeError(f'unable to save {path}')
        print(path.resolve())


if __name__ == '__main__':
    main()

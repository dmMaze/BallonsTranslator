"""Render the Ruby/furigana visual acceptance fixture."""

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

from ballontranslator.ui.text_engine.annotations import apply_emphasis, apply_ruby
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.fontformat import TextTransformStack
from ballontranslator.utils.textblock import TextBlock


OUTPUT_DIR = Path('tests/ui/artifacts/ruby_furigana')


def _item(
    text: str,
    *,
    vertical: bool,
    position: str,
    ruby_type: str,
    reading: str,
    origin: tuple[int, int],
    base_length: int = 2,
    emphasis: bool = True,
    effects: bool = True,
    stroke_width: float = 0.12,
    fill_color: list[int] | None = None,
) -> TextBlkItem:
    width, height = ((510, 180) if not vertical else (250, 470))
    block = TextBlock([origin[0], origin[1], origin[0] + width, origin[1] + height])
    block._bounding_rect = [origin[0], origin[1], origin[0] + width, origin[1] + height]
    block.translation = text
    block.vertical = vertical
    block.fontformat.font_size = 34
    if fill_color is not None:
        block.fontformat.frgb = fill_color
    if effects:
        block.fontformat.stroke_width = stroke_width
        block.fontformat.shadow_strength = 0.8
        block.fontformat.shadow_radius = 0.1
        block.fontformat.shadow_offset = [0.08, 0.08]
    block.fontformat.text_transform = TextTransformStack((), 14.0)
    item = TextBlkItem(block, 0)
    cursor = QTextCursor(item.document())
    cursor.setPosition(0)
    cursor.setPosition(base_length, QTextCursor.MoveMode.KeepAnchor)
    apply_ruby(cursor, ruby_type, reading, position)
    if ruby_type == 'group' and emphasis:
        apply_emphasis(cursor, 'filled sesame', 'over right')
    item.layout.reLayoutEverything()
    item.set_ui_guide_suppressed(True)
    return item


def render(scale: int) -> QImage:
    width, height = 1140, 720
    image = QImage(
        width * scale,
        height * scale,
        QImage.Format.Format_ARGB32_Premultiplied,
    )
    image.fill(QColor('#f1eee8'))
    scene = QGraphicsScene()
    scene.setSceneRect(QRectF(0, 0, width, height))
    title = scene.addSimpleText(
        'Ruby / Furigana — native half-width stroke, space-around, H + V',
        QFont('Sans Serif', 12),
    )
    title.setBrush(QColor('#665f57'))
    title.setPos(24, 8)
    fixtures = (
        _item('東京、Kana A!', vertical=False, position='over', ruby_type='group', reading='とてもながいとうきょう', origin=(25, 50)),
        _item('日本。Latin B?', vertical=False, position='under', ruby_type='mono', reading='に ほん', origin=(25, 255)),
        _item(
            '哈尔滨佛学院、A!',
            vertical=True,
            position='over',
            ruby_type='group',
            reading='哈佛',
            origin=(590, 50),
            base_length=6,
            emphasis=False,
            stroke_width=0.18,
            fill_color=[242, 242, 242],
        ),
        _item('日本。B?', vertical=True, position='under', ruby_type='mono', reading='にっ ぽん', origin=(860, 50)),
    )
    for item in fixtures:
        scene.addItem(item)
        # Exercise the same scale-sensitive effect cache used by page export.
        item.repaint_background(render_scale=float(scale))
    painter = QPainter(image)
    scene.render(
        painter,
        QRectF(0, 0, image.width(), image.height()),
        scene.sceneRect(),
    )
    painter.end()
    return image


def main() -> None:
    _app = QApplication.instance() or QApplication([])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scale in (1, 3):
        path = OUTPUT_DIR / f'ruby_furigana_fixture_{scale}x.png'
        if not render(scale).save(str(path)):
            raise RuntimeError(f'unable to save {path}')
        print(path.resolve())


if __name__ == '__main__':
    main()

import unittest
from types import SimpleNamespace

from qtpy.QtCore import QPointF, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.ui.text_engine.item import TextBlkItem, _ConflictBadgeItem
from ballontranslator.utils.multi_extract import has_conflict, resolve
from ballontranslator.utils.textblock import TextBlock

# Module-level reference: a local QApplication is destroyed on return, which
# deletes every widget created under it.
APP = None


def get_app():
    global APP
    if APP is None:
        APP = QApplication.instance() or QApplication([])
    return APP


def make_block(payload=None):
    block = TextBlock([0, 0, 300, 120])
    block._bounding_rect = [0, 0, 300, 120]
    block.vertical = True
    block.text = ['そういえば']
    if payload is not None:
        block.multi_extract = payload
    return block


def make_item(payload=None):
    item = TextBlkItem(make_block(payload), 0)
    scene = QGraphicsScene()
    scene.addItem(item)
    # The scene owns the items: keep it alive for as long as the item is used.
    item._test_scene = scene
    get_app().processEvents()
    return item


CONFLICT = {
    'chosen': 0,
    'support': [0, 1],
    'candidates': [{'pass': 1, 'text': ['そういえは']}],
}


class ConflictBadgeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_block_without_payload_has_no_badge(self):
        item = make_item()

        self.assertFalse(item._conflict_badge_item.isVisible())

    def test_block_with_payload_shows_the_badge(self):
        item = make_item(CONFLICT)

        self.assertTrue(item._conflict_badge_item.isVisible())

    def test_badge_is_a_fixed_size_dot(self):
        badge = make_item(CONFLICT)._conflict_badge_item

        self.assertEqual(badge.boundingRect().size().width(), 15.6)
        self.assertEqual(badge.boundingRect().size().height(), 15.6)
        self.assertTrue(
            badge.flags()
            & badge.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self.assertEqual(
            badge.acceptedMouseButtons(),
            Qt.MouseButton.LeftButton,
        )

    def test_badge_sits_on_the_bottom_right_corner(self):
        item = make_item(CONFLICT)
        badge = item._conflict_badge_item
        outline = item.geometry_controller.visual_outline_in_item()
        corner = outline.boundingRect().bottomRight()
        half = _ConflictBadgeItem.DIAMETER / 2

        self.assertAlmostEqual(badge.pos().x(), corner.x() - half, places=1)
        self.assertAlmostEqual(badge.pos().y(), corner.y() - half, places=1)
        self.assertGreater(
            badge.pos().x(),
            outline.boundingRect().center().x(),
        )
        self.assertGreater(
            badge.pos().y(),
            outline.boundingRect().center().y(),
        )

    def test_resolving_the_block_hides_the_badge(self):
        item = make_item(CONFLICT)

        resolve(item.blk, 1)
        item.refresh_conflict_badge()

        self.assertFalse(has_conflict(item.blk))
        self.assertFalse(item._conflict_badge_item.isVisible())
        self.assertEqual(item.blk.get_text(), 'そういえは')

    def test_editing_hides_the_badge_and_leaving_editing_restores_it(self):
        item = make_item(CONFLICT)

        item.startEdit()
        get_app().processEvents()
        self.assertFalse(item._conflict_badge_item.isVisible())

        item.endEdit()
        get_app().processEvents()
        self.assertTrue(item._conflict_badge_item.isVisible())

    def test_suppressing_ui_guides_hides_the_badge(self):
        item = make_item(CONFLICT)

        item.set_ui_guide_suppressed(True)
        self.assertFalse(item._conflict_badge_item.isVisible())

        item.set_ui_guide_suppressed(False)
        self.assertTrue(item._conflict_badge_item.isVisible())

    def test_badge_does_not_shift_the_block_position(self):
        plain = make_item()
        marked = make_item(CONFLICT)

        self.assertEqual(plain.pos(), QPointF(0, 0))
        self.assertEqual(marked.pos(), QPointF(0, 0))
        self.assertEqual(
            plain.geometry_controller.visual_outline_in_item().boundingRect(),
            marked.geometry_controller.visual_outline_in_item().boundingRect(),
        )


class _FakeMouseEvent:
    """PyQt6 refuses to build scene mouse events, so mimic the two members
    the badge reads: the local position and ``accept()``."""

    def __init__(self, pos=QPointF(6.0, 6.0)):
        self._pos = QPointF(pos)
        self.accepted = False

    def pos(self):
        return self._pos

    def accept(self):
        self.accepted = True


class ConflictBadgeClickTest(unittest.TestCase):
    """The badge is the in-canvas entry point of the review dialog."""

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def test_badge_is_clickable(self):
        badge = make_item(CONFLICT)._conflict_badge_item

        self.assertEqual(
            badge.acceptedMouseButtons(),
            Qt.MouseButton.LeftButton,
        )
        self.assertEqual(
            badge.cursor().shape(),
            Qt.CursorShape.PointingHandCursor,
        )
        self.assertTrue(badge.toolTip())

    def test_clicking_the_badge_requests_a_review(self):
        item = make_item(CONFLICT)
        requests = []
        item.review_multi_extract.connect(lambda: requests.append(True))
        badge = item._conflict_badge_item
        press = _FakeMouseEvent()
        release = _FakeMouseEvent()

        badge.mousePressEvent(press)
        badge.mouseReleaseEvent(release)

        self.assertTrue(press.accepted)
        self.assertTrue(release.accepted)
        self.assertEqual(requests, [True])

    def test_pressing_requests_nothing(self):
        item = make_item(CONFLICT)
        requests = []
        item.review_multi_extract.connect(lambda: requests.append(True))

        item._conflict_badge_item.mousePressEvent(_FakeMouseEvent())

        self.assertEqual(requests, [])

    def test_releasing_outside_the_badge_requests_nothing(self):
        item = make_item(CONFLICT)
        requests = []
        item.review_multi_extract.connect(lambda: requests.append(True))

        item._conflict_badge_item.mouseReleaseEvent(
            _FakeMouseEvent(QPointF(-40.0, -40.0))
        )

        self.assertEqual(requests, [])


class ConflictBadgeCanvasClickTest(unittest.TestCase):
    """The canvas must hand the badge its own pixels.

    The badge sits on the block's bottom-right reshape corner, where the shape
    handles and the drawing/transform paths also claim the press.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def _canvas(self):
        from ballontranslator.ui.canvas import Canvas

        canvas = Canvas()
        canvas.imgtrans_proj = SimpleNamespace(
            img_valid=True,
            current_img='a.jpg',
            pages={'a.jpg': []},
        )
        canvas.gv.resize(600, 400)
        canvas.gv.show()
        self.addCleanup(self._destroy, canvas)
        self.app.processEvents()
        return canvas

    @staticmethod
    def _destroy(canvas):
        canvas.gv.setParent(None)
        canvas.gv.deleteLater()
        get_app().processEvents()

    def test_clicking_a_selected_blocks_badge_requests_review(self):
        canvas = self._canvas()
        item = TextBlkItem(make_block(CONFLICT), 0)
        canvas.addItem(item)
        item.setSelected(True)
        canvas.txtblkShapeControl.setBlkItem(item)
        self.app.processEvents()
        requests = []
        item.review_multi_extract.connect(lambda: requests.append(True))
        badge = item._conflict_badge_item
        center = badge.mapToScene(badge.boundingRect().center())

        QTest.mouseClick(
            canvas.gv.viewport(),
            Qt.MouseButton.LeftButton,
            pos=canvas.gv.mapFromScene(center),
        )
        self.app.processEvents()

        self.assertEqual(requests, [True])
        # The press must not start a reshape or move the block.
        self.assertFalse(item.reshaping)
        self.assertEqual(item.pos(), QPointF(0, 0))

    def test_clicking_away_does_not_request_review(self):
        canvas = self._canvas()
        item = TextBlkItem(make_block(CONFLICT), 0)
        canvas.addItem(item)
        self.app.processEvents()
        requests = []
        item.review_multi_extract.connect(lambda: requests.append(True))

        QTest.mouseClick(
            canvas.gv.viewport(),
            Qt.MouseButton.LeftButton,
            pos=canvas.gv.mapFromScene(QPointF(20.0, 20.0)),
        )
        self.app.processEvents()

        self.assertEqual(requests, [])


if __name__ == '__main__':
    unittest.main()

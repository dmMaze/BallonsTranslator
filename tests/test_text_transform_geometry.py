import math
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF

from ballontranslator.ui.text_transform import (
    mapped_rect_polygon,
    rect_polygon,
    text_transform_matrix,
    text_transform_point,
)
from ballontranslator.utils.fontformat import FontFormat, normalize_text_transform


class TextTransformNormalizationTest(unittest.TestCase):

    def test_fontformat_uses_canonical_neutral_defaults(self):
        fontformat = FontFormat()

        self.assertEqual(fontformat.text_transform, (1.0, 1.0, 0.0))
        self.assertEqual(fontformat.horizontal_scale, 1.0)
        self.assertEqual(fontformat.vertical_scale, 1.0)
        self.assertEqual(fontformat.slant_angle, 0.0)

    def test_normalization_clamps_and_uses_six_decimal_precision(self):
        self.assertEqual(
            normalize_text_transform(1.23456789, 0.01, 90.0),
            (1.234568, 0.1, 45.0),
        )
        self.assertEqual(
            normalize_text_transform(9.0, 4.000001, -90.0),
            (4.0, 4.0, -45.0),
        )

    def test_normalization_canonicalizes_negative_zero(self):
        angle = normalize_text_transform(1.0, 1.0, -0.0)[2]

        self.assertEqual(angle, 0.0)
        self.assertEqual(math.copysign(1.0, angle), 1.0)

    def test_normalization_rejects_each_nonfinite_component(self):
        for component in range(3):
            for value in (math.nan, math.inf, -math.inf):
                values = [1.0, 1.0, 0.0]
                values[component] = value
                with self.subTest(component=component, value=value):
                    with self.assertRaisesRegex(ValueError, 'finite numbers'):
                        normalize_text_transform(*values)


class TextTransformMatrixTest(unittest.TestCase):
    CASES = (
        ('identity', 1.0, 1.0, 0.0),
        ('minimum_horizontal', 0.1, 1.0, 0.0),
        ('maximum_horizontal', 4.0, 1.0, 0.0),
        ('minimum_vertical', 1.0, 0.1, 0.0),
        ('maximum_vertical', 1.0, 4.0, 0.0),
        ('wide_and_short', 4.0, 0.1, 0.0),
        ('narrow_and_tall', 0.1, 4.0, 0.0),
        ('negative_slant_limit', 1.0, 1.0, -45.0),
        ('positive_slant_limit', 1.0, 1.0, 45.0),
        ('nonuniform_with_slant', 2.75, 0.4, 33.25),
    )

    def assertPointAlmostEqual(self, actual, expected, places=9):
        self.assertAlmostEqual(actual.x(), expected.x(), places=places)
        self.assertAlmostEqual(actual.y(), expected.y(), places=places)

    @staticmethod
    def direct_formula(point, pivot, horizontal_scale, vertical_scale, slant_angle):
        k = -math.tan(math.radians(slant_angle))
        dx = point.x() - pivot.x()
        dy = point.y() - pivot.y()
        return QPointF(
            pivot.x() + horizontal_scale * dx + k * vertical_scale * dy,
            pivot.y() + vertical_scale * dy,
        )

    def test_identity_matrix_is_exactly_neutral(self):
        matrix = text_transform_matrix(1.0, 1.0, 0.0, QPointF(8.0, -3.0))

        self.assertTrue(matrix.isIdentity())

    def test_matrix_matches_direct_formula_and_keeps_pivot_fixed(self):
        pivot = QPointF(-3.25, 7.5)
        points = (
            pivot,
            QPointF(0.0, 0.0),
            QPointF(12.5, -4.75),
            QPointF(-10000.0, 10000.0),
        )

        for name, horizontal_scale, vertical_scale, slant_angle in self.CASES:
            with self.subTest(case=name):
                k = -math.tan(math.radians(slant_angle))
                matrix = text_transform_matrix(
                    horizontal_scale, vertical_scale, slant_angle, pivot
                )
                self.assertAlmostEqual(matrix.m11(), horizontal_scale)
                self.assertAlmostEqual(matrix.m12(), 0.0)
                self.assertAlmostEqual(matrix.m21(), k * vertical_scale)
                self.assertAlmostEqual(matrix.m22(), vertical_scale)
                self.assertAlmostEqual(
                    matrix.dx(),
                    pivot.x()
                    - horizontal_scale * pivot.x()
                    - k * vertical_scale * pivot.y(),
                )
                self.assertAlmostEqual(
                    matrix.dy(), pivot.y() - vertical_scale * pivot.y()
                )
                self.assertPointAlmostEqual(matrix.map(pivot), pivot)

                for point in points:
                    expected = self.direct_formula(
                        point,
                        pivot,
                        horizontal_scale,
                        vertical_scale,
                        slant_angle,
                    )
                    self.assertPointAlmostEqual(matrix.map(point), expected)
                    self.assertPointAlmostEqual(
                        text_transform_point(
                            point,
                            pivot,
                            horizontal_scale,
                            vertical_scale,
                            slant_angle,
                        ),
                        expected,
                    )

    def test_forward_inverse_round_trip_at_transform_extremes(self):
        pivot = QPointF(14.0, -9.0)
        points = (
            QPointF(-2500.0, -1750.0),
            QPointF(0.0, 0.0),
            QPointF(9876.5, 4321.25),
        )

        for name, horizontal_scale, vertical_scale, slant_angle in self.CASES:
            with self.subTest(case=name):
                matrix = text_transform_matrix(
                    horizontal_scale, vertical_scale, slant_angle, pivot
                )
                inverse, invertible = matrix.inverted()
                self.assertTrue(invertible)
                for point in points:
                    self.assertPointAlmostEqual(
                        inverse.map(matrix.map(point)), point, places=7
                    )

    def test_mapped_polygon_preserves_corner_order_and_shear(self):
        rect = QRectF(-2.0, 3.0, 5.0, 7.0)
        pivot = QPointF(1.25, -4.5)
        matrix = text_transform_matrix(2.0, 0.5, 30.0, pivot)
        source = rect_polygon(rect)
        mapped = mapped_rect_polygon(rect, matrix)
        expected_source = (
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft(),
        )

        self.assertEqual(len(source), 4)
        self.assertEqual(len(mapped), 4)
        for index, corner in enumerate(expected_source):
            self.assertPointAlmostEqual(source[index], corner)
            self.assertPointAlmostEqual(
                mapped[index],
                self.direct_formula(corner, pivot, 2.0, 0.5, 30.0),
            )

        self.assertNotAlmostEqual(mapped[0].x(), mapped[3].x())


if __name__ == '__main__':
    unittest.main()

"""Regressions for the source-off ratio recheck repair.

These cases pin the three defects that let every report return the whole real
line or a merged punctured line:

1. the joint-domain projection discarded off-diagonal channel covariance;
2. the ratio-line intersection ignored singular (rank zero / rank one) shapes;
3. a full-rank grazer on the zero-inlet plane was reported as an attained ratio,
   and two open components touching at that point were merged into one line.
"""
import sqlite3
import unittest
from contextlib import closing
from fractions import Fraction

from evidence.channels import ReadingPair
from evidence.codec import decode_artifact
from review.algebraic_set import ExactInterval, normalize_ratio_set
from review.request import BoundRatioRequest
from review.ratios import compute_ratio_set
from tests.fixtures import disk_artifact, initialize_bench, issue_current, sample_batch


def _set(pair):
    return compute_ratio_set(BoundRatioRequest(artifact=None, request=None, pair=pair))


def _pair(sxx, sxz, szz, mx, mz):
    return ReadingPair(Fraction(mx), Fraction(mz), Fraction(sxx), Fraction(sxz), Fraction(szz))


class ShapeProjectionTests(unittest.TestCase):
    def test_published_shape_keeps_channel_cross_covariance(self):
        _, _, text = disk_artifact()
        readings = decode_artifact(text).readings
        index = readings.catalog.index_of
        # in and out-a share the drive_x coefficient row: their covariance is nonzero.
        self.assertEqual(readings.shape[index('in')][index('out-a')], Fraction(1))
        # in follows drive_x, out-b follows drive_y: independent drives, zero covariance.
        self.assertEqual(readings.shape[index('in')][index('out-b')], Fraction(0))
        # twice = 2x, sum = x+y => cross term 2, diagonal variances 4 and 2.
        self.assertEqual(readings.shape[index('twice')][index('sum')], Fraction(2))
        self.assertEqual(readings.shape[index('sum')][index('sum')], Fraction(2))

    def test_private_parameter_names_never_leave_the_bench(self):
        revision, _, text = disk_artifact()
        for private_name in revision.cohort.columns:
            self.assertNotIn(private_name, text)
        for raw in ('design', 'measurement', 'drive'):
            self.assertNotIn(raw, text)


class RankAwareRatioTests(unittest.TestCase):
    def test_rank_zero_domain_is_center_singleton_or_empty(self):
        singleton = _set(_pair(0, 0, 0, 3, 6))
        self.assertEqual(singleton.components,
                         (ExactInterval(Fraction(2), Fraction(2), True, True),))
        # Center reading has a zero inlet: no division is supported.
        self.assertEqual(_set(_pair(0, 0, 0, 0, 5)).components, ())

    def test_fixed_inlet_interval_is_closed_at_both_ends(self):
        result = _set(_pair(0, 0, 4, 2, 1))
        part, = result.components
        self.assertEqual((part.lower, part.upper), (Fraction(-1, 2), Fraction(3, 2)))
        self.assertTrue(part.lower_closed and part.upper_closed)
        self.assertEqual(_set(_pair(0, 0, 4, 0, 1)).components, ())

    def test_rank_one_constant_mobius_map_is_a_singleton(self):
        # shifted = 1+x, proportional = 2+2x share one drive: ratio is always 2.
        result = _set(_pair(1, 2, 4, 1, 2))
        self.assertEqual(result.components,
                         (ExactInterval(Fraction(2), Fraction(2), True, True),))

    def test_rank_one_disconnected_rays_meet_origin_case_a(self):
        result = _set(_pair(1, 1, 1, 0, 1))
        encoded = [
            (part.lower, part.upper, part.lower_closed, part.upper_closed)
            for part in result.components
        ]
        self.assertEqual(encoded, [
            (None, Fraction(0), False, True),
            (Fraction(2), None, True, False),
        ])

    def test_full_rank_origin_tangent_punctures_the_real_line_case_b(self):
        # Unit disk centered at (inlet 0, outlet 1): slope 0 only touches at the origin.
        result = _set(_pair(1, 0, 1, 0, 1))
        encoded = [
            (part.lower, part.upper, part.lower_closed, part.upper_closed)
            for part in result.components
        ]
        self.assertEqual(encoded, [
            (None, Fraction(0), False, False),
            (Fraction(0), None, False, False),
        ])
        self.assertFalse(result.contains(Fraction(0)))
        self.assertTrue(result.contains(Fraction(1)) and result.contains(Fraction(-1)))

    def test_full_rank_closed_cuts_are_still_included(self):
        # Unit disk centered at (1, 1): the feasible ratios form the closed ray
        # r >= 0.  The boundary r = 0 grazes the disk at the nonzero inlet x* = 1, so
        # the endpoint is closed and r = 0 is an attained ratio; negative slopes miss.
        result = _set(_pair(1, 0, 1, 1, 1))
        part, = result.components
        self.assertEqual(part.lower, Fraction(0))
        self.assertIsNone(part.upper)
        self.assertTrue(part.lower_closed)
        self.assertFalse(part.upper_closed)
        self.assertTrue(result.contains(Fraction(0)))
        self.assertTrue(result.contains(Fraction(2)))
        self.assertFalse(result.contains(Fraction(-1)))

    def test_infeasible_ratio_line_returns_empty_set(self):
        # Unit disk centered at (inlet 0, outlet 2): the room polynomial is 1 - 3 r^2,
        # so attainable ratios are the two rays beyond +/-sqrt(3); ordinary values such
        # as 0 and 1 are absent while 2 is reached through a nonzero inlet.
        result = _set(_pair(1, 0, 1, 0, 2))
        self.assertFalse(result.contains(Fraction(0)))
        self.assertFalse(result.contains(Fraction(1)))
        self.assertTrue(result.contains(Fraction(2)))
        lower, upper = result.components
        self.assertIsNone(lower.lower)
        self.assertIsNone(upper.upper)
        self.assertTrue(lower.upper_closed and upper.lower_closed)

    def test_correlated_full_rank_grazer_uses_general_gradient(self):
        # Non-diagonal full-rank shape with det = 1 and center (inlet 0, outlet 1).
        # The exact solver returns two closed unbounded rays (-inf, 0] U [1, +inf);
        # both finite cuts graze the ellipsoid at nonzero-inlet points and therefore
        # stay closed (no zero-inlet puncture).  Checked against the canonical
        # completing-the-square oracle rather than a hand-expanded polynomial.
        result = _set(_pair(2, 1, 1, 0, 1))
        encoded = [
            (part.lower, part.upper, part.lower_closed, part.upper_closed)
            for part in result.components
        ]
        self.assertEqual(encoded, [
            (None, Fraction(0), False, True),
            (Fraction(1), None, True, False),
        ])
        for value in (Fraction(0), Fraction(1), Fraction(-7), Fraction(9)):
            self.assertTrue(result.contains(value))
        for value in (Fraction(1, 2), Fraction(2, 3)):
            self.assertFalse(result.contains(value))


class NormalizationTests(unittest.TestCase):
    def test_open_components_touching_at_one_point_do_not_merge(self):
        merged = normalize_ratio_set([
            ExactInterval(None, Fraction(0), False, False),
            ExactInterval(Fraction(0), None, False, False),
        ])
        self.assertEqual(len(merged.components), 2)

    def test_closed_components_meeting_at_endpoint_still_merge(self):
        merged = normalize_ratio_set([
            ExactInterval(None, Fraction(0), False, True),
            ExactInterval(Fraction(0), None, True, False),
        ])
        self.assertEqual(len(merged.components), 1)


class EndToEndShapeTests(unittest.TestCase):
    def test_budget_saturation_collapses_the_domain_to_its_center(self):
        with closing(sqlite3.connect(':memory:')) as connection:
            ledger, register = initialize_bench(connection)
            ledger.accept_batch(**sample_batch())
            _, artifact, text = issue_current(ledger, register, budget=0)
        readings = decode_artifact(text).readings
        # Budget equal to the minimum residual leaves zero slack: shape vanishes while
        # the calibrated centers are still published.
        self.assertTrue(all(value == 0 for row in readings.shape for value in row))
        self.assertEqual(readings.center[readings.catalog.index_of('out-a')], Fraction(1))
        self.assertTrue(artifact.artifact_id)


if __name__ == '__main__':
    unittest.main()

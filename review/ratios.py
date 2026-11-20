from fractions import Fraction

from review.algebraic_set import ExactInterval, RatioSet, normalize_ratio_set
from review.quadratics import compare_exact, polynomial_sign, rational_between, solve_real_quadratic
from review.request import BoundRatioRequest


def _empty():
    return RatioSet(())


def _singleton(value):
    return RatioSet((ExactInterval(value, value, True, True),))


def _linear_sign(coefficients, value):
    slope, intercept = coefficients
    if slope == 0:
        return (intercept > 0) - (intercept < 0)
    return ((slope > 0) - (slope < 0)) * compare_exact(value, -intercept / slope)


def _feasible_pieces(room, cuts, excluded):
    """Build the attainable set from a room quadratic with boundary cuts.

    Feasibility is uniformly ``room(r) >= 0``; each region between consecutive cuts is
    classified by an exact interior sign probe, so the construction is independent of
    which way the quadratic opens.  An included cut contributes a closed singleton that
    merges into the adjacent region; an excluded cut (a zero-inlet grazer) stays open on
    both sides and never gets a singleton, so the puncture survives normalization.
    """
    pieces = []
    left = None
    for right in (*cuts, None):
        if polynomial_sign(room, rational_between(left, right)) >= 0:
            pieces.append(ExactInterval(
                left, right,
                left is not None and left not in excluded,
                right is not None and right not in excluded,
            ))
        if right is not None and right not in excluded:
            pieces.append(ExactInterval(right, right, True, True))
        left = right
    return normalize_ratio_set(pieces)


def compute_ratio_set(bound: BoundRatioRequest):
    """Ratios r = outlet/inlet witnessed by an actual joint reading with inlet != 0.

    The published joint reading domain is the closed ellipsoid
    (w - mu)^T S^{-1} (w - mu) <= 1 with rational center mu and a rational symmetric
    positive semidefinite pair shape S (at most 2x2).  A ratio r is admissible exactly
    when the ratio line outlet = r * inlet meets the ellipsoid at a point whose inlet
    coordinate is nonzero.
    """
    pair = bound.pair
    mx, mz = Fraction(pair.inlet_center), Fraction(pair.outlet_center)
    sxx, sxz, szz = (
        Fraction(pair.inlet_shape), Fraction(pair.cross_shape), Fraction(pair.outlet_shape),
    )

    # No slack: the domain collapses to the single center reading.
    if sxx == 0 and sxz == 0 and szz == 0:
        return _empty() if mx == 0 else _singleton(mz / mx)

    determinant = sxx * szz - sxz * sxz

    # Rank-one pair shape.  Along its single drive direction the outlet is an affine
    # function of the inlet, so ratios are a Möbius map of one closed interval.
    if determinant == 0:
        if sxx == 0:
            # PSD forces sxz == 0 here; the inlet is fixed at mx and the outlet ranges
            # over [mz - sqrt(szz), mz + sqrt(szz)].
            if mx == 0:
                return _empty()
            room = (-mx * mx, 2 * mx * mz, szz - mz * mz)
            cuts = solve_real_quadratic(*room) or ()
            lower, upper = (cuts[0], cuts[0]) if len(cuts) == 1 else (cuts[0], cuts[1])
            return RatioSet((ExactInterval(lower, upper, True, True),))
        k = sxz / sxx
        if mz == k * mx:
            # The Möbius map is constant: every feasible reading divides to the same
            # ratio, even though room(r) >= 0 alone would advertise a whole interval.
            return _singleton(k)
        # Adjugate form of the intersection condition; room(r) >= 0 means feasible.
        room = (sxx - mx * mx, 2 * (mx * mz - sxz), szz - mz * mz)
        return _feasible_pieces(room, solve_real_quadratic(*room) or (), set())

    # Full-rank shape.  On the line outlet = r * inlet, expand the canonical ellipsoid
    # numerator (w - mu)^T M (w - mu), M = [[szz, -sxz], [-sxz, sxx]], as a quadratic
    # P(r) x^2 + Q(r) x + R(r) in the inlet x (R already subtracts det).  Since P(r)>0,
    # completing the square makes the line feasible exactly when Q(r)^2 - 4 P R >= 0;
    # this numerator is quadratic in r (the cubic/quartic terms cancel).  Collect its
    # three exact coefficients by evaluation at r = 0, 1, 2, keeping "room >= 0
    # feasible" uniform with the rank-one branch.
    def _components(value):
        p_coeff = szz - 2 * sxz * value + sxx * value * value
        q_coeff = -2 * szz * mx + 2 * sxz * (value * mx + mz) - 2 * sxx * value * mz
        r_coeff = szz * mx * mx - 2 * sxz * mx * mz + sxx * mz * mz - determinant
        return p_coeff, q_coeff, r_coeff

    def _room(value):
        p_coeff, q_coeff, r_coeff = _components(value)
        return q_coeff * q_coeff - 4 * p_coeff * r_coeff

    p0, p1, p2 = _room(0), _room(1), _room(2)
    c2 = (p2 - 2 * p1 + p0) / 2
    c1 = p1 - p0 - c2
    # The quadratics module orders coefficients as (quadratic, linear, constant).
    room = (c2, c1, p0)
    cuts = solve_real_quadratic(*room) or ()

    # A cut r0 where the line merely grazes the ellipsoid at a point with zero inlet
    # supports no actual outlet/inlet division and must be punctured.  The minimizer
    # inlet is x* = -Q/(2P), which vanishes exactly when Q(r0) = 0; Q is linear in r.
    q_linear = (
        2 * (sxz * mx - sxx * mz),
        2 * (sxz * mz - szz * mx),
    )
    excluded = {cut for cut in cuts if _linear_sign(q_linear, cut) == 0}
    return _feasible_pieces(room, cuts, excluded)

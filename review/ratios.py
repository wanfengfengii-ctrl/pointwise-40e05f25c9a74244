from review.algebraic_set import ExactInterval, RatioSet, normalize_ratio_set
from review.quadratics import polynomial_sign, rational_between, solve_real_quadratic
from review.request import BoundRatioRequest


def compute_ratio_set(bound: BoundRatioRequest):
    pair = bound.pair
    inlet = pair.inlet_center
    outlet = pair.outlet_center
    inlet_shape, cross, outlet_shape = pair.inlet_shape, pair.cross_shape, pair.outlet_shape
    if inlet == 0 and inlet_shape == 0:
        return RatioSet(())

    # Use the intersection of the requested ratio line with the closed reading domain.
    room = (inlet_shape - inlet**2, 2 * (inlet * outlet - cross), outlet_shape - outlet**2)
    cuts = solve_real_quadratic(*room) or ()
    pieces = []
    left = None
    for right in (*cuts, None):
        if polynomial_sign(room, rational_between(left, right)) >= 0:
            pieces.append(ExactInterval(left, right))
        if right is not None:
            pieces.append(ExactInterval(right, right, True, True))
        left = right
    return normalize_ratio_set(pieces)

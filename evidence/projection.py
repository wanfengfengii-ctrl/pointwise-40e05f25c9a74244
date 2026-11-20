from fractions import Fraction

from calibration.rational_linalg import multiply_matrices, solve_linear_system
from evidence.channels import ChannelCatalog, PublicReadings


def project_joint_domain(fitted, wiring):
    catalog = ChannelCatalog(tuple(channel.name for channel in wiring))
    least_squares = fitted.least_squares
    coefficients = tuple(channel.coefficients for channel in wiring)
    fitted_column = tuple((value,) for value in least_squares.center)
    linear_centers = multiply_matrices(coefficients, fitted_column)
    center = tuple(
        linear[0] + channel.offset
        for linear, channel in zip(linear_centers, wiring, strict=True)
    )
    slack = fitted.budget.slack
    zero = Fraction(0)
    size = len(center)
    # The joint feasible domain is the closed ellipsoid (w-mu)^T C^{-1} (w-mu) <= slack,
    # whose full shape matrix C = slack * A * gram^{-1} * A^T carries the cross-channel
    # covariances.  Dropping the off-diagonal entries disconnects channels that share a
    # private drive and corrupts every outlet/inlet ratio below.
    dual_rows = tuple(
        solve_linear_system(least_squares.gram, row) if slack else tuple(zero for _ in row)
        for row in coefficients
    )

    def _entry(index, column):
        if not slack:
            return zero
        return slack * sum(
            (coefficient * value for coefficient, value in zip(coefficients[index], dual_rows[column], strict=True)),
            zero,
        )

    shape = tuple(
        tuple(_entry(index, column) for column in range(size)) for index in range(size)
    )
    return PublicReadings(catalog, center, shape)

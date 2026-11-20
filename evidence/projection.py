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
    variances = [zero] * len(center)
    if slack:
        for index, row in enumerate(coefficients):
            dual = solve_linear_system(least_squares.gram, row)
            variances[index] = slack * sum(
                (coefficient * value for coefficient, value in zip(row, dual, strict=True)), zero
            )
    shape = tuple(
        tuple(variance if index == column else zero for column in range(len(center)))
        for index, variance in enumerate(variances)
    )
    return PublicReadings(catalog, center, shape)

import numpy as np
from numcodecs_safeguards.safeguards.elementwise.abs_findiff import (
    _finite_difference_coefficients,
)


def test_centred_zero_order():
    np.testing.assert_array_equal(
        _finite_difference_coefficients(0, np.array([0])),
        np.array([1]),
    )

    np.testing.assert_array_equal(
        _finite_difference_coefficients(0, np.array([0, 1, -1, 2, -2, 3, -3, 4, -4])),
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
    )


def test_centred_first_order():
    np.testing.assert_array_equal(
        _finite_difference_coefficients(1, np.array([0, 1, -1])),
        np.array([0, 0.5, -0.5]),
    )

    np.testing.assert_array_equal(
        _finite_difference_coefficients(1, np.array([0, 1, -1, 2, -2])),
        np.array([0, 2 / 3, -2 / 3, -1 / 12, 1 / 12]),
    )

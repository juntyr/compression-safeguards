from fractions import Fraction as F

from numcodecs_safeguards.safeguards.elementwise.abs_findiff import (
    _finite_difference_coefficients,
)


def test_centred_zero_order():
    assert _finite_difference_coefficients(0, [0]) == [1]

    assert _finite_difference_coefficients(0, [0, 1, -1, 2, -2, 3, -3, 4, -4]) == [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


def test_centred_first_order():
    assert _finite_difference_coefficients(1, [0, 1, -1]) == [0, F(1, 2), F(-1, 2)]

    assert _finite_difference_coefficients(1, [0, 1, -1, 2, -2]) == [
        0,
        F(2, 3),
        F(-2, 3),
        F(-1, 12),
        F(1, 12),
    ]

    assert _finite_difference_coefficients(1, [0, 1, -1, 2, -2, 3, -3]) == [
        0,
        F(3, 4),
        F(-3, 4),
        F(-3, 20),
        F(3, 20),
        F(1, 60),
        F(-1, 60),
    ]

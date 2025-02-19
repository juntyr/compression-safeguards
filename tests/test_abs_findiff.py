"""
Finite difference coefficients from:

Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
spaced grids. Mathematics of Computation, 51(184), 699-706. Available from:
https://doi.org/10.1090/s0025-5718-1988-0935077-0.
"""

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

    assert _finite_difference_coefficients(1, [0, 1, -1, 2, -2, 3, -3, 4, -4]) == [
        0,
        F(4, 5),
        F(-4, 5),
        F(-1, 5),
        F(1, 5),
        F(4, 105),
        F(-4, 105),
        F(-1, 280),
        F(1, 280),
    ]


def test_centred_second_order():
    assert _finite_difference_coefficients(2, [0, 1, -1]) == [-2, 1, 1]

    assert _finite_difference_coefficients(2, [0, 1, -1, 2, -2]) == [
        F(-5, 2),
        F(4, 3),
        F(4, 3),
        F(-1, 12),
        F(-1, 12),
    ]

    assert _finite_difference_coefficients(2, [0, 1, -1, 2, -2, 3, -3]) == [
        F(-49, 18),
        F(3, 2),
        F(3, 2),
        F(-3, 20),
        F(-3, 20),
        F(1, 90),
        F(1, 90),
    ]

    assert _finite_difference_coefficients(2, [0, 1, -1, 2, -2, 3, -3, 4, -4]) == [
        F(-205, 72),
        F(8, 5),
        F(8, 5),
        F(-1, 5),
        F(-1, 5),
        F(8, 315),
        F(8, 315),
        F(-1, 560),
        F(-1, 560),
    ]


def test_centred_third_order():
    assert _finite_difference_coefficients(3, [0, 1, -1, 2, -2]) == [
        0,
        -1,
        1,
        F(1, 2),
        F(-1, 2),
    ]

    assert _finite_difference_coefficients(3, [0, 1, -1, 2, -2, 3, -3]) == [
        0,
        F(-13, 8),
        F(13, 8),
        1,
        -1,
        F(-1, 8),
        F(1, 8),
    ]

    assert _finite_difference_coefficients(3, [0, 1, -1, 2, -2, 3, -3, 4, -4]) == [
        0,
        F(-61, 30),
        F(61, 30),
        F(169, 120),
        F(-169, 120),
        F(-3, 10),
        F(3, 10),
        F(7, 240),
        F(-7, 240),
    ]


def test_centred_fourth_order():
    assert _finite_difference_coefficients(4, [0, 1, -1, 2, -2]) == [
        6,
        -4,
        -4,
        1,
        1,
    ]

    assert _finite_difference_coefficients(4, [0, 1, -1, 2, -2, 3, -3]) == [
        F(28, 3),
        F(-13, 2),
        F(-13, 2),
        2,
        2,
        F(-1, 6),
        F(-1, 6),
    ]

    assert _finite_difference_coefficients(4, [0, 1, -1, 2, -2, 3, -3, 4, -4]) == [
        F(91, 8),
        F(-122, 15),
        F(-122, 15),
        F(169, 60),
        F(169, 60),
        F(-2, 5),
        F(-2, 5),
        F(7, 240),
        F(7, 240),
    ]

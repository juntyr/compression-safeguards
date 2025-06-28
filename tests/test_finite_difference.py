"""
Finite difference coefficients from:

Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
spaced grids. Mathematics of Computation, 51(184), 699-706. Available from:
https://doi.org/10.1090/s0025-5718-1988-0935077-0.
"""

from sympy import Rational as F

from compression_safeguards.safeguards._qois.finite_difference import (
    _finite_difference_coefficients,
)


def test_central_zeroth_order():
    assert _finite_difference_coefficients(0, F(0), (F(0),)) == (1,)

    assert _finite_difference_coefficients(
        0, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3), F(4), F(-4))
    ) == (
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_central_first_order():
    assert _finite_difference_coefficients(1, F(0), (F(0), F(1), F(-1))) == (
        0,
        F(1, 2),
        F(-1, 2),
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(0), F(1), F(-1), F(2), F(-2))
    ) == (
        0,
        F(2, 3),
        F(-2, 3),
        F(-1, 12),
        F(1, 12),
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3))
    ) == (
        0,
        F(3, 4),
        F(-3, 4),
        F(-3, 20),
        F(3, 20),
        F(1, 60),
        F(-1, 60),
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3), F(4), F(-4))
    ) == (
        0,
        F(4, 5),
        F(-4, 5),
        F(-1, 5),
        F(1, 5),
        F(4, 105),
        F(-4, 105),
        F(-1, 280),
        F(1, 280),
    )


def test_central_second_order():
    assert _finite_difference_coefficients(2, F(0), (F(0), F(1), F(-1))) == (-2, 1, 1)

    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(1), F(-1), F(2), F(-2))
    ) == (
        F(-5, 2),
        F(4, 3),
        F(4, 3),
        F(-1, 12),
        F(-1, 12),
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3))
    ) == (
        F(-49, 18),
        F(3, 2),
        F(3, 2),
        F(-3, 20),
        F(-3, 20),
        F(1, 90),
        F(1, 90),
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3), F(4), F(-4))
    ) == (
        F(-205, 72),
        F(8, 5),
        F(8, 5),
        F(-1, 5),
        F(-1, 5),
        F(8, 315),
        F(8, 315),
        F(-1, 560),
        F(-1, 560),
    )


def test_central_third_order():
    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(1), F(-1), F(2), F(-2))
    ) == (
        0,
        -1,
        1,
        F(1, 2),
        F(-1, 2),
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3))
    ) == (
        0,
        F(-13, 8),
        F(13, 8),
        1,
        -1,
        F(-1, 8),
        F(1, 8),
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3), F(4), F(-4))
    ) == (
        0,
        F(-61, 30),
        F(61, 30),
        F(169, 120),
        F(-169, 120),
        F(-3, 10),
        F(3, 10),
        F(7, 240),
        F(-7, 240),
    )


def test_central_fourth_order():
    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(1), F(-1), F(2), F(-2))
    ) == (
        6,
        -4,
        -4,
        1,
        1,
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3))
    ) == (
        F(28, 3),
        F(-13, 2),
        F(-13, 2),
        2,
        2,
        F(-1, 6),
        F(-1, 6),
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3), F(4), F(-4))
    ) == (
        F(91, 8),
        F(-122, 15),
        F(-122, 15),
        F(169, 60),
        F(169, 60),
        F(-2, 5),
        F(-2, 5),
        F(7, 240),
        F(7, 240),
    )


def test_forward_zeroth_order():
    assert _finite_difference_coefficients(0, F(0), (F(0),)) == (1,)

    assert _finite_difference_coefficients(0, F(0), (F(0), F(1), F(2), F(3), F(4))) == (
        1,
        0,
        0,
        0,
        0,
    )


def test_forward_first_order():
    assert _finite_difference_coefficients(1, F(0), (F(0), F(1))) == (-1, 1)

    assert _finite_difference_coefficients(1, F(0), (F(0), F(1), F(2))) == (
        F(-3, 2),
        2,
        F(-1, 2),
    )

    assert _finite_difference_coefficients(1, F(0), (F(0), F(1), F(2), F(3))) == (
        F(-11, 6),
        3,
        F(-3, 2),
        F(1, 3),
    )

    assert _finite_difference_coefficients(1, F(0), (F(0), F(1), F(2), F(3), F(4))) == (
        F(-25, 12),
        4,
        -3,
        F(4, 3),
        F(-1, 4),
    )


def test_forward_second_order():
    assert _finite_difference_coefficients(2, F(0), (F(0), F(1), F(2))) == (1, -2, 1)

    assert _finite_difference_coefficients(2, F(0), (F(0), F(1), F(2), F(3))) == (
        2,
        -5,
        4,
        -1,
    )

    assert _finite_difference_coefficients(2, F(0), (F(0), F(1), F(2), F(3), F(4))) == (
        F(35, 12),
        F(-26, 3),
        F(19, 2),
        F(-14, 3),
        F(11, 12),
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(1), F(2), F(3), F(4), F(5))
    ) == (
        F(15, 4),
        F(-77, 6),
        F(107, 6),
        -13,
        F(61, 12),
        F(-5, 6),
    )


def test_forward_third_order():
    assert _finite_difference_coefficients(3, F(0), (F(0), F(1), F(2), F(3))) == (
        -1,
        3,
        -3,
        1,
    )

    assert _finite_difference_coefficients(3, F(0), (F(0), F(1), F(2), F(3), F(4))) == (
        F(-5, 2),
        9,
        -12,
        7,
        F(-3, 2),
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(1), F(2), F(3), F(4), F(5))
    ) == (
        F(-17, 4),
        F(71, 4),
        F(-59, 2),
        F(49, 2),
        F(-41, 4),
        F(7, 4),
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(1), F(2), F(3), F(4), F(5), F(6))
    ) == (
        F(-49, 8),
        29,
        F(-461, 8),
        62,
        F(-307, 8),
        13,
        F(-15, 8),
    )


def test_forward_fourth_order():
    assert _finite_difference_coefficients(4, F(0), (F(0), F(1), F(2), F(3), F(4))) == (
        1,
        -4,
        6,
        -4,
        1,
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(1), F(2), F(3), F(4), F(5))
    ) == (
        3,
        -14,
        26,
        -24,
        11,
        -2,
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(1), F(2), F(3), F(4), F(5), F(6))
    ) == (
        F(35, 6),
        -31,
        F(137, 2),
        F(-242, 3),
        F(107, 2),
        -19,
        F(17, 6),
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7))
    ) == (
        F(28, 3),
        F(-111, 2),
        142,
        F(-1219, 6),
        176,
        F(-185, 2),
        F(82, 3),
        F(-7, 2),
    )


def test_backward_zeroth_order():
    assert _finite_difference_coefficients(0, F(0), (F(0),)) == (1,)

    assert _finite_difference_coefficients(
        0, F(0), (F(0), F(-1), F(-2), F(-3), F(-4))
    ) == (
        1,
        0,
        0,
        0,
        0,
    )


def test_backward_first_order():
    assert _finite_difference_coefficients(1, F(0), (F(0), F(-1))) == (1, -1)

    assert _finite_difference_coefficients(1, F(0), (F(0), F(-1), F(-2))) == (
        F(3, 2),
        -2,
        F(1, 2),
    )

    assert _finite_difference_coefficients(1, F(0), (F(0), F(-1), F(-2), F(-3))) == (
        F(11, 6),
        -3,
        F(3, 2),
        F(-1, 3),
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(0), F(-1), F(-2), F(-3), F(-4))
    ) == (
        F(25, 12),
        -4,
        3,
        F(-4, 3),
        F(1, 4),
    )


def test_backward_second_order():
    assert _finite_difference_coefficients(2, F(0), (F(0), F(-1), F(-2))) == (1, -2, 1)

    assert _finite_difference_coefficients(2, F(0), (F(0), F(-1), F(-2), F(-3))) == (
        2,
        -5,
        4,
        -1,
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(-1), F(-2), F(-3), F(-4))
    ) == (
        F(35, 12),
        F(-26, 3),
        F(19, 2),
        F(-14, 3),
        F(11, 12),
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(-1), F(-2), F(-3), F(-4), F(-5))
    ) == (
        F(15, 4),
        F(-77, 6),
        F(107, 6),
        -13,
        F(61, 12),
        F(-5, 6),
    )


def test_backward_third_order():
    assert _finite_difference_coefficients(3, F(0), (F(0), F(-1), F(-2), F(-3))) == (
        1,
        -3,
        3,
        -1,
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(-1), F(-2), F(-3), F(-4))
    ) == (
        F(5, 2),
        -9,
        12,
        -7,
        F(3, 2),
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(-1), F(-2), F(-3), F(-4), F(-5))
    ) == (
        F(17, 4),
        F(-71, 4),
        F(59, 2),
        F(-49, 2),
        F(41, 4),
        F(-7, 4),
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(0), F(-1), F(-2), F(-3), F(-4), F(-5), F(-6))
    ) == (
        F(49, 8),
        -29,
        F(461, 8),
        -62,
        F(307, 8),
        -13,
        F(15, 8),
    )


def test_backward_fourth_order():
    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(-1), F(-2), F(-3), F(-4))
    ) == (1, -4, 6, -4, 1)

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(-1), F(-2), F(-3), F(-4), F(-5))
    ) == (
        3,
        -14,
        26,
        -24,
        11,
        -2,
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(-1), F(-2), F(-3), F(-4), F(-5), F(-6))
    ) == (
        F(35, 6),
        -31,
        F(137, 2),
        F(-242, 3),
        F(107, 2),
        -19,
        F(17, 6),
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(0), F(-1), F(-2), F(-3), F(-4), F(-5), F(-6), F(-7))
    ) == (
        F(28, 3),
        F(-111, 2),
        142,
        F(-1219, 6),
        176,
        F(-185, 2),
        F(82, 3),
        F(-7, 2),
    )

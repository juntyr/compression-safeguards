"""
Finite difference coefficients from:

Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily
spaced grids. Mathematics of Computation, 51(184), 699-706. Available from:
[doi:10.1090/s0025-5718-1988-0935077-0](https://doi.org/10.1090/s0025-5718-1988-0935077-0).
"""

from functools import partial

from sympy import Rational as F

from compression_safeguards.safeguards._qois.finite_difference import (
    _finite_difference_coefficients,
)
from compression_safeguards.safeguards._qois.symfunc import symmetric_modulo


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


def test_central_half_way_zeroth_order():
    assert _finite_difference_coefficients(0, F(0), (F(1, 2), F(-1, 2))) == (
        F(1, 2),
        F(1, 2),
    )

    assert _finite_difference_coefficients(
        0, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2))
    ) == (F(9, 16), F(9, 16), F(-1, 16), F(-1, 16))

    assert _finite_difference_coefficients(
        0, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2))
    ) == (F(75, 128), F(75, 128), F(-25, 256), F(-25, 256), F(3, 256), F(3, 256))

    assert _finite_difference_coefficients(
        0,
        F(0),
        (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2), F(7, 2), F(-7, 2)),
    ) == (
        F(1225, 2048),
        F(1225, 2048),
        F(-245, 2048),
        F(-245, 2048),
        F(49, 2048),
        F(49, 2048),
        F(-5, 2048),
        F(-5, 2048),
    )


def test_central_half_way_first_order():
    assert _finite_difference_coefficients(1, F(0), (F(1, 2), F(-1, 2))) == (
        1,
        -1,
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2))
    ) == (F(9, 8), F(-9, 8), F(-1, 24), F(1, 24))

    assert _finite_difference_coefficients(
        1, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2))
    ) == (F(75, 64), F(-75, 64), F(-25, 384), F(25, 384), F(3, 640), F(-3, 640))

    assert _finite_difference_coefficients(
        1,
        F(0),
        (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2), F(7, 2), F(-7, 2)),
    ) == (
        F(1225, 1024),
        F(-1225, 1024),
        F(-245, 3072),
        F(245, 3072),
        F(49, 5120),
        F(-49, 5120),
        F(-5, 7168),
        F(5, 7168),
    )


def test_central_half_way_second_order():
    assert _finite_difference_coefficients(
        2, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2))
    ) == (F(-1, 2), F(-1, 2), F(1, 2), F(1, 2))

    assert _finite_difference_coefficients(
        2, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2))
    ) == (F(-17, 24), F(-17, 24), F(13, 16), F(13, 16), F(-5, 48), F(-5, 48))

    assert _finite_difference_coefficients(
        2,
        F(0),
        (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2), F(7, 2), F(-7, 2)),
    ) == (
        F(-1891, 2304),
        F(-1891, 2304),
        F(1299, 1280),
        F(1299, 1280),
        F(-499, 2304),
        F(-499, 2304),
        F(259, 11520),
        F(259, 11520),
    )


def test_central_half_way_third_order():
    assert _finite_difference_coefficients(
        3, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2))
    ) == (-3, 3, 1, -1)

    assert _finite_difference_coefficients(
        3, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2))
    ) == (F(-17, 4), F(17, 4), F(13, 8), F(-13, 8), F(-1, 8), F(1, 8))

    assert _finite_difference_coefficients(
        3,
        F(0),
        (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2), F(7, 2), F(-7, 2)),
    ) == (
        F(-1891, 384),
        F(1891, 384),
        F(1299, 640),
        F(-1299, 640),
        F(-499, 1920),
        F(499, 1920),
        F(37, 1920),
        F(-37, 1920),
    )


def test_central_half_way_fourth_order():
    assert _finite_difference_coefficients(
        4, F(0), (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2))
    ) == (1, 1, F(-3, 2), F(-3, 2), F(1, 2), F(1, 2))

    assert _finite_difference_coefficients(
        4,
        F(0),
        (F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(5, 2), F(-5, 2), F(7, 2), F(-7, 2)),
    ) == (
        F(83, 48),
        F(83, 48),
        F(-45, 16),
        F(-45, 16),
        F(59, 48),
        F(59, 48),
        F(-7, 48),
        F(-7, 48),
    )


def test_forward_half_way_zeroth_order():
    assert _finite_difference_coefficients(0, F(0), (F(-1, 2),)) == (1,)

    assert _finite_difference_coefficients(0, F(0), (F(-1, 2), F(1, 2))) == (
        F(1, 2),
        F(1, 2),
    )

    assert _finite_difference_coefficients(0, F(0), (F(-1, 2), F(1, 2), F(3, 2))) == (
        F(3, 8),
        F(3, 4),
        F(-1, 8),
    )

    assert _finite_difference_coefficients(
        0, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2))
    ) == (
        F(5, 16),
        F(15, 16),
        F(-5, 16),
        F(1, 16),
    )

    assert _finite_difference_coefficients(
        0, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2))
    ) == (
        F(35, 128),
        F(35, 32),
        F(-35, 64),
        F(7, 32),
        F(-5, 128),
    )

    assert _finite_difference_coefficients(
        0, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2))
    ) == (
        F(63, 256),
        F(315, 256),
        F(-105, 128),
        F(63, 128),
        F(-45, 256),
        F(7, 256),
    )

    assert _finite_difference_coefficients(
        0, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2))
    ) == (
        F(231, 1024),
        F(693, 512),
        F(-1155, 1024),
        F(231, 256),
        F(-495, 1024),
        F(77, 512),
        F(-21, 1024),
    )

    assert _finite_difference_coefficients(
        0,
        F(0),
        (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2), F(13, 2)),
    ) == (
        F(429, 2048),
        F(3003, 2048),
        F(-3003, 2048),
        F(3003, 2048),
        F(-2145, 2048),
        F(1001, 2048),
        F(-273, 2048),
        F(33, 2048),
    )

    assert _finite_difference_coefficients(
        0,
        F(0),
        (
            F(-1, 2),
            F(1, 2),
            F(3, 2),
            F(5, 2),
            F(7, 2),
            F(9, 2),
            F(11, 2),
            F(13, 2),
            F(15, 2),
        ),
    ) == (
        F(6435, 32768),
        F(6435, 4096),
        F(-15015, 8192),
        F(9009, 4096),
        F(-32175, 16384),
        F(5005, 4096),
        F(-4095, 8192),
        F(495, 4096),
        F(-429, 32768),
    )


def test_forward_half_way_first_order():
    assert _finite_difference_coefficients(1, F(0), (F(-1, 2), F(1, 2))) == (
        -1,
        1,
    )

    assert _finite_difference_coefficients(1, F(0), (F(-1, 2), F(1, 2), F(3, 2))) == (
        -1,
        1,
        0,
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2))
    ) == (
        F(-23, 24),
        F(7, 8),
        F(1, 8),
        F(-1, 24),
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2))
    ) == (
        F(-11, 12),
        F(17, 24),
        F(3, 8),
        F(-5, 24),
        F(1, 24),
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2))
    ) == (
        F(-563, 640),
        F(67, 128),
        F(143, 192),
        F(-37, 64),
        F(29, 128),
        F(-71, 1920),
    )

    assert _finite_difference_coefficients(
        1, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2))
    ) == (
        F(-1627, 1920),
        F(211, 640),
        F(59, 48),
        F(-235, 192),
        F(91, 128),
        F(-443, 1920),
        F(31, 960),
    )

    assert _finite_difference_coefficients(
        1,
        F(0),
        (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2), F(13, 2)),
    ) == (
        F(-88069, 107520),
        F(2021, 15360),
        F(28009, 15360),
        F(-6803, 3072),
        F(5227, 3072),
        F(-12673, 15360),
        F(3539, 15360),
        F(-3043, 107520),
    )

    assert _finite_difference_coefficients(
        1,
        F(0),
        (
            F(-1, 2),
            F(1, 2),
            F(3, 2),
            F(5, 2),
            F(7, 2),
            F(9, 2),
            F(11, 2),
            F(13, 2),
            F(15, 2),
        ),
    ) == (
        F(-1423, 1792),
        F(-491, 7168),
        F(7753, 3072),
        F(-18509, 5120),
        F(3535, 1024),
        F(-2279, 1024),
        F(953, 1024),
        F(-1637, 7168),
        F(2689, 107520),
    )


def test_forward_half_way_second_order():
    assert _finite_difference_coefficients(2, F(0), (F(-1, 2), F(1, 2), F(3, 2))) == (
        1,
        -2,
        1,
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2))
    ) == (
        F(3, 2),
        F(-7, 2),
        F(5, 2),
        F(-1, 2),
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2))
    ) == (
        F(43, 24),
        F(-14, 3),
        F(17, 4),
        F(-5, 3),
        F(7, 24),
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2))
    ) == (
        F(95, 48),
        F(-269, 48),
        F(49, 8),
        F(-85, 24),
        F(59, 48),
        F(-3, 16),
    )

    assert _finite_difference_coefficients(
        2, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2))
    ) == (
        F(12139, 5760),
        F(-6119, 960),
        F(3091, 384),
        F(-1759, 288),
        F(1211, 384),
        F(-919, 960),
        F(739, 5760),
    )

    assert _finite_difference_coefficients(
        2,
        F(0),
        (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2), F(13, 2)),
    ) == (
        F(25333, 11520),
        F(-80813, 11520),
        F(2553, 256),
        F(-21457, 2304),
        F(14651, 2304),
        F(-3687, 1280),
        F(8863, 11520),
        F(-211, 2304),
    )

    assert _finite_difference_coefficients(
        2,
        F(0),
        (
            F(-1, 2),
            F(1, 2),
            F(3, 2),
            F(5, 2),
            F(7, 2),
            F(9, 2),
            F(11, 2),
            F(13, 2),
            F(15, 2),
        ),
    ) == (
        F(81227, 35840),
        F(-67681, 8960),
        F(34151, 2880),
        F(-16747, 1280),
        F(5669, 512),
        F(-76621, 11520),
        F(1699, 640),
        F(-5647, 8960),
        F(21719, 322560),
    )


def test_forward_half_way_third_order():
    assert _finite_difference_coefficients(
        3, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2))
    ) == (
        -1,
        3,
        -3,
        1,
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2))
    ) == (
        -2,
        7,
        -9,
        5,
        -1,
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2))
    ) == (
        F(-23, 8),
        F(91, 8),
        F(-71, 4),
        F(55, 4),
        F(-43, 8),
        F(7, 8),
    )

    assert _finite_difference_coefficients(
        3, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2))
    ) == (
        F(-29, 8),
        F(127, 8),
        -29,
        F(115, 4),
        F(-133, 8),
        F(43, 8),
        F(-3, 4),
    )

    assert _finite_difference_coefficients(
        3,
        F(0),
        (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2), F(13, 2)),
    ) == (
        F(-8197, 1920),
        F(39139, 1920),
        F(-27219, 640),
        F(19699, 384),
        F(-15043, 384),
        F(12099, 640),
        F(-10099, 1920),
        F(1237, 1920),
    )

    assert _finite_difference_coefficients(
        3,
        F(0),
        (
            F(-1, 2),
            F(1, 2),
            F(3, 2),
            F(5, 2),
            F(7, 2),
            F(9, 2),
            F(11, 2),
            F(13, 2),
            F(15, 2),
        ),
    ) == (
        F(-2317, 480),
        F(47707, 1920),
        F(-7443, 128),
        F(158471, 1920),
        F(-30037, 384),
        F(32091, 640),
        F(-40087, 1920),
        F(1961, 384),
        F(-357, 640),
    )


def test_forward_half_way_fourth_order():
    assert _finite_difference_coefficients(
        4, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2))
    ) == (1, -4, 6, -4, 1)

    assert _finite_difference_coefficients(
        4, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2))
    ) == (
        F(5, 2),
        F(-23, 2),
        21,
        -19,
        F(17, 2),
        F(-3, 2),
    )

    assert _finite_difference_coefficients(
        4, F(0), (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2))
    ) == (
        F(101, 24),
        F(-87, 4),
        F(373, 8),
        F(-319, 6),
        F(273, 8),
        F(-47, 4),
        F(41, 24),
    )

    assert _finite_difference_coefficients(
        4,
        F(0),
        (F(-1, 2), F(1, 2), F(3, 2), F(5, 2), F(7, 2), F(9, 2), F(11, 2), F(13, 2)),
    ) == (
        F(287, 48),
        F(-1639, 48),
        F(1341, 16),
        F(-5527, 48),
        F(4613, 48),
        F(-783, 16),
        F(677, 48),
        F(-85, 48),
    )

    assert _finite_difference_coefficients(
        4,
        F(0),
        (
            F(-1, 2),
            F(1, 2),
            F(3, 2),
            F(5, 2),
            F(7, 2),
            F(9, 2),
            F(11, 2),
            F(13, 2),
            F(15, 2),
        ),
    ) == (
        F(14861, 1920),
        F(-1447, 30),
        F(21299, 160),
        F(-25651, 120),
        F(42119, 192),
        F(-2951, 20),
        F(30437, 480),
        F(-1903, 120),
        F(1127, 640),
    )


def test_central_second_order_with_offset():
    # +3/4
    assert _finite_difference_coefficients(
        2, F(3, 4), (F(3, 4), F(7, 4), F(-1, 4))
    ) == (-2, 1, 1)

    # +1/27
    assert _finite_difference_coefficients(
        2, F(1, 27), (F(1, 27), F(28, 27), F(-26, 27), F(55, 27), F(-53, 27))
    ) == (
        F(-5, 2),
        F(4, 3),
        F(4, 3),
        F(-1, 12),
        F(-1, 12),
    )

    # -1
    assert _finite_difference_coefficients(
        2, F(-1), (F(-1), F(0), F(-2), F(1), F(-3), F(2), F(-4))
    ) == (
        F(-49, 18),
        F(3, 2),
        F(3, 2),
        F(-3, 20),
        F(-3, 20),
        F(1, 90),
        F(1, 90),
    )

    # +42
    assert _finite_difference_coefficients(
        2, F(42), (F(42), F(43), F(41), F(44), F(40), F(45), F(39), F(46), F(38))
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


def test_central_second_order_with_spacing():
    # /4
    assert _finite_difference_coefficients(2, F(0), (F(0), F(1, 4), F(-1, 4))) == (
        -32,
        16,
        16,
    )

    # *4
    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(4), F(-4), F(8), F(-8))
    ) == (
        F(-5, 32),
        F(4, 48),
        F(4, 48),
        F(-1, 192),
        F(-1, 192),
    )

    # *-2
    assert _finite_difference_coefficients(
        2, F(0), (F(0), F(-2), F(2), F(-4), F(4), F(-6), F(6))
    ) == (
        F(-49, 72),
        F(3, 8),
        F(3, 8),
        F(-3, 80),
        F(-3, 80),
        F(1, 360),
        F(1, 360),
    )

    # +1, /2
    assert _finite_difference_coefficients(
        2, F(1), (F(1), F(3, 2), F(1, 2), F(2), F(0), F(5, 2), F(-1, 2), F(3), F(-1))
    ) == (
        F(-205, 18),
        F(32, 5),
        F(32, 5),
        F(-4, 5),
        F(-4, 5),
        F(32, 315),
        F(32, 315),
        F(-4, 560),
        F(-4, 560),
    )


def test_central_second_order_with_periodic_transform():
    def delta_transform(x, period):
        return symmetric_modulo(x, period)

    # period must be >= 2*coefficient range to allow proper sampling (no aliasing)
    assert _finite_difference_coefficients(
        2,
        F(0),
        (F(0), F(1), F(-1)),
        delta_transform=partial(delta_transform, period=F(4)),
    ) == (-2, 1, 1)

    assert _finite_difference_coefficients(
        2,
        F(0),
        (F(0), F(1), F(9), F(2), F(8)),
        delta_transform=partial(delta_transform, period=F(10)),
    ) == (
        F(-5, 2),
        F(4, 3),
        F(4, 3),
        F(-1, 12),
        F(-1, 12),
    )

    assert _finite_difference_coefficients(
        2,
        F(0),
        (F(0), F(1), F(11), F(2), F(10), F(3), F(9)),
        delta_transform=partial(delta_transform, period=F(12)),
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
        2,
        F(0),
        (F(0), F(1), F(-1), F(2), F(-2), F(3), F(-3), F(4), F(-4)),
        delta_transform=partial(delta_transform, period=F(16)),
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

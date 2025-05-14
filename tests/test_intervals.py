import numpy as np

from numcodecs_safeguards.cast import as_bits
from numcodecs_safeguards.intervals import (
    Interval,
    IntervalUnion,
    Lower,
    Upper,
    _maximum,
    _minimum,
)
from numcodecs_safeguards.safeguards.pointwise.abs import AbsoluteErrorBoundSafeguard
from numcodecs_safeguards.safeguards.pointwise.sign import SignPreservingSafeguard
from numcodecs_safeguards.safeguards.pointwise.zero import ZeroIsZeroSafeguard


def test_sign():
    safeguard = SignPreservingSafeguard()

    intervals = safeguard.compute_safe_intervals(np.arange(0, 10, dtype=np.uint8))
    np.testing.assert_equal(
        intervals._lower,
        np.array([[0] + [1] * 9], dtype=np.uint8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[0] + [255] * 9], dtype=np.uint8),
    )

    intervals = safeguard.compute_safe_intervals(np.arange(-9, 10, dtype=np.int8))
    np.testing.assert_equal(
        intervals._lower,
        np.array([[-128] * 9 + [0] + [1] * 9], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-1] * 9 + [0] + [127] * 9], dtype=np.int8),
    )

    intervals = safeguard.compute_safe_intervals(
        np.array([-np.nan, -np.inf, -5.0, -0.0, +0.0, 5.0, np.inf, np.nan])
    )
    np.testing.assert_equal(
        as_bits(intervals._lower),
        as_bits(
            np.array(
                [
                    [
                        _minimum(np.dtype(float)),
                        _minimum(np.dtype(float)),
                        _minimum(np.dtype(float)),
                        -0.0,
                        0.0,
                        np.finfo(float).smallest_subnormal,
                        np.finfo(float).smallest_subnormal,
                        np.finfo(float).smallest_subnormal,
                    ]
                ]
            )
        ),
    )
    np.testing.assert_equal(
        as_bits(intervals._upper),
        as_bits(
            np.array(
                [
                    [
                        -np.finfo(float).smallest_subnormal,
                        -np.finfo(float).smallest_subnormal,
                        -np.finfo(float).smallest_subnormal,
                        -0.0,
                        0.0,
                        _maximum(np.dtype(float)),
                        _maximum(np.dtype(float)),
                        _maximum(np.dtype(float)),
                    ]
                ]
            )
        ),
    )


def test_abs():
    safeguard = AbsoluteErrorBoundSafeguard(eb_abs=2)

    intervals = safeguard.compute_safe_intervals(np.arange(0, 10, dtype=np.uint8))
    np.testing.assert_equal(
        intervals._lower,
        np.array([[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.uint8),
    )

    intervals = safeguard.compute_safe_intervals(np.arange(-9, 10, dtype=np.int8))
    np.testing.assert_equal(
        intervals._lower,
        np.arange(-11, 8, dtype=np.int8).reshape(1, -1),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.arange(-7, 12, dtype=np.int8).reshape(1, -1),
    )

    intervals = safeguard.compute_safe_intervals(
        np.array([-128, -127, -126, -125, 124, 125, 126, 127], dtype=np.int8)
    )
    np.testing.assert_equal(
        intervals._lower,
        np.array([[-128, -128, -128, -127, 122, 123, 124, 125]], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-126, -125, -124, -123, 126, 127, 127, 127]], dtype=np.int8),
    )


def test_sign_abs():
    data = np.arange(-4, 5, dtype=np.int8)

    sign_intervals = SignPreservingSafeguard().compute_safe_intervals(data)
    abs_intervals = AbsoluteErrorBoundSafeguard(eb_abs=2).compute_safe_intervals(data)

    intervals = sign_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals._lower,
        np.array([[-6, -5, -4, -3, 0, 1, 1, 1, 2]], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-2, -1, -1, -1, 0, 3, 4, 5, 6]], dtype=np.int8),
    )

    data = np.arange(-4, 5, dtype=float)

    sign_intervals = SignPreservingSafeguard().compute_safe_intervals(data)
    abs_intervals = AbsoluteErrorBoundSafeguard(eb_abs=2.0).compute_safe_intervals(data)

    intervals = sign_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals._lower,
        np.array(
            [
                [
                    -6.0,
                    -5.0,
                    -4.0,
                    -3.0,
                    0.0,
                    np.finfo(float).smallest_subnormal,
                    np.finfo(float).smallest_subnormal,
                    1.0,
                    2.0,
                ]
            ]
        ),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array(
            [
                [
                    -2.0,
                    -1.0,
                    -np.finfo(float).smallest_subnormal,
                    -np.finfo(float).smallest_subnormal,
                    0.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ]
            ]
        ),
    )


def test_zero_abs():
    data = np.arange(-4, 5, dtype=np.int8)

    zero_intervals = ZeroIsZeroSafeguard(zero=-1).compute_safe_intervals(data)
    abs_intervals = AbsoluteErrorBoundSafeguard(eb_abs=2).compute_safe_intervals(data)

    intervals = zero_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals._lower,
        np.array([[-6, -5, -4, -1, -2, -1, 0, 1, 2]], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-2, -1, 0, -1, 2, 3, 4, 5, 6]], dtype=np.int8),
    )

    data = np.arange(-4, 5, dtype=float)

    zero_intervals = ZeroIsZeroSafeguard(zero=-1.0).compute_safe_intervals(data)
    abs_intervals = AbsoluteErrorBoundSafeguard(eb_abs=2.0).compute_safe_intervals(data)

    intervals = zero_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals._lower,
        np.array([[-6.0, -5.0, -4.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0]]),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-2.0, -1.0, 0.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )


def generate_random_interval_union() -> tuple[IntervalUnion, set]:
    n = np.random.randint(1, 5)

    pivots = np.sort(np.random.randint(0, 100, n * 2))

    intervals = IntervalUnion.empty(np.dtype(int), 1, 1)
    elems = set()

    for i in range(n):
        low, high = pivots[i * 2], pivots[i * 2 + 1]
        interval = Interval.empty(np.dtype(int), 1)
        Lower(np.array(low)) <= interval[:] <= Upper(np.array(high))
        intervals = intervals.union(interval.into_union())
        elems = elems.union(range(low, high + 1))

    return (intervals, elems)


def test_union_no_overlap():
    z = Interval.empty(np.dtype(int), 1)
    Lower(np.array(0)) <= z[:] <= Upper(np.array(0))

    a = Interval.empty(np.dtype(int), 1)
    Lower(np.array(53)) <= a[:] <= Upper(np.array(53))

    az = a.union(z)

    np.testing.assert_array_equal(az._lower, np.array([[0], [53]]))
    np.testing.assert_array_equal(az._upper, np.array([[0], [53]]))

    b = Interval.empty(np.dtype(int), 1)
    Lower(np.array(0)) <= b[:] <= Upper(np.array(53))
    b = b.into_union()

    abz = az.union(b)

    np.testing.assert_array_equal(abz._lower, np.array([[0]]))
    np.testing.assert_array_equal(abz._upper, np.array([[53]]))


def test_union_adjacent():
    a = Interval.empty(np.dtype(int), 1)
    Lower(np.array(1)) <= a[:] <= Upper(np.array(3))

    b = Interval.empty(np.dtype(int), 1)
    Lower(np.array(4)) <= b[:] <= Upper(np.array(5))

    ab = a.union(b)

    np.testing.assert_array_equal(ab._lower, np.array([[1]]))
    np.testing.assert_array_equal(ab._upper, np.array([[5]]))

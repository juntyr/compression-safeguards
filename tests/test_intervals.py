import numpy as np

from compression_safeguards.safeguards.pointwise.eb import ErrorBoundSafeguard
from compression_safeguards.safeguards.pointwise.same import SameValueSafeguard
from compression_safeguards.safeguards.pointwise.sign import SignPreservingSafeguard
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.cast import as_bits
from compression_safeguards.utils.intervals import (
    Interval,
    Lower,
    Upper,
    _maximum,
    _minimum,
)


def test_sign():
    safeguard = SignPreservingSafeguard()

    intervals = safeguard.compute_safe_intervals(
        np.arange(0, 10, dtype=np.uint8), late_bound=Bindings.EMPTY
    )
    np.testing.assert_equal(
        intervals._lower,
        np.array([[0] + [1] * 9], dtype=np.uint8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[0] + [255] * 9], dtype=np.uint8),
    )

    intervals = safeguard.compute_safe_intervals(
        np.arange(-9, 10, dtype=np.int8), late_bound=Bindings.EMPTY
    )
    np.testing.assert_equal(
        intervals._lower,
        np.array([[-128] * 9 + [0] + [1] * 9], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-1] * 9 + [0] + [127] * 9], dtype=np.int8),
    )

    intervals = safeguard.compute_safe_intervals(
        np.array([-np.nan, -np.inf, -5.0, -0.0, +0.0, 5.0, np.inf, np.nan]),
        late_bound=Bindings.EMPTY,
    )
    np.testing.assert_equal(
        as_bits(intervals._lower),
        as_bits(
            np.array(
                [
                    [
                        _minimum(np.dtype(float)),
                        -np.inf,
                        -np.inf,
                        -0.0,
                        0.0,
                        np.finfo(float).smallest_subnormal,
                        np.finfo(float).smallest_subnormal,
                        np.array(as_bits(np.array(np.inf, dtype=float)) + 1).view(
                            float
                        ),
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
                        -np.array(as_bits(np.array(np.inf, dtype=float)) + 1).view(
                            float
                        ),
                        -np.finfo(float).smallest_subnormal,
                        -np.finfo(float).smallest_subnormal,
                        -0.0,
                        0.0,
                        np.inf,
                        np.inf,
                        _maximum(np.dtype(float)),
                    ]
                ]
            )
        ),
    )


def test_abs():
    safeguard = ErrorBoundSafeguard(type="abs", eb=2)

    intervals = safeguard.compute_safe_intervals(
        np.arange(0, 10, dtype=np.uint8), late_bound=Bindings.EMPTY
    )
    np.testing.assert_equal(
        intervals._lower,
        np.array([[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.uint8),
    )

    intervals = safeguard.compute_safe_intervals(
        np.arange(-9, 10, dtype=np.int8), late_bound=Bindings.EMPTY
    )
    np.testing.assert_equal(
        intervals._lower,
        np.arange(-11, 8, dtype=np.int8).reshape(1, -1),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.arange(-7, 12, dtype=np.int8).reshape(1, -1),
    )

    intervals = safeguard.compute_safe_intervals(
        np.array([-128, -127, -126, -125, 124, 125, 126, 127], dtype=np.int8),
        late_bound=Bindings.EMPTY,
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

    sign_intervals = SignPreservingSafeguard().compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )
    abs_intervals = ErrorBoundSafeguard(type="abs", eb=2).compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )

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

    sign_intervals = SignPreservingSafeguard().compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )
    abs_intervals = ErrorBoundSafeguard(type="abs", eb=2.0).compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )

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


def test_same_abs():
    data = np.arange(-4, 5, dtype=np.int8)

    same_intervals = SameValueSafeguard(value=-1).compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )
    abs_intervals = ErrorBoundSafeguard(type="abs", eb=2).compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )

    intervals = same_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals._lower,
        np.array([[-6, -5, -4, -1, -2, -1, 0, 1, 2]], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-2, -1, 0, -1, 2, 3, 4, 5, 6]], dtype=np.int8),
    )

    data = np.arange(-4, 5, dtype=float)

    same_intervals = SameValueSafeguard(value=-1.0).compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )
    abs_intervals = ErrorBoundSafeguard(type="abs", eb=2.0).compute_safe_intervals(
        data, late_bound=Bindings.EMPTY
    )

    intervals = same_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals._lower,
        np.array([[-6.0, -5.0, -4.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0]]),
    )
    np.testing.assert_equal(
        intervals._upper,
        np.array([[-2.0, -1.0, 0.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )


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


def test_preserve_inf():
    data = np.array([-np.nan, -np.inf, -42, -0.0, 0.0, 42, np.inf, np.nan])

    empty = Interval.empty_like(data)
    preserved = Interval.empty_like(data).preserve_inf(data)

    assert np.all(
        as_bits(preserved._lower)
        == as_bits(np.where(np.isinf(data), data, empty._lower))
    )
    assert np.all(
        as_bits(preserved._upper)
        == as_bits(np.where(np.isinf(data), data, empty._upper))
    )


def test_preserve_signed_nan():
    data = np.array(
        [-np.nan, -np.inf, -42, -0.0, 0.0, 42, np.inf, np.nan], dtype=np.float64
    )

    empty = Interval.empty_like(data)
    preserved = Interval.empty_like(data).preserve_signed_nan(data, equal_nan=False)

    assert np.all(
        as_bits(preserved._lower)
        == as_bits(np.where(np.isnan(data), data, empty._lower))
    )
    assert np.all(
        as_bits(preserved._upper)
        == as_bits(np.where(np.isnan(data), data, empty._upper))
    )

    preserved = Interval.empty_like(data).preserve_signed_nan(data, equal_nan=True)

    assert (
        as_bits(preserved._lower[0]) == 0xFFFFFFFFFFFFFFFF
    )  # negative NaN with largest mantissa
    assert (
        as_bits(preserved._upper[0]) == 0xFFF0000000000001
    )  # negative NaN with smallest mantissa
    assert np.all(as_bits(preserved._lower[1:-1]) == as_bits(empty._lower[1:-1]))
    assert np.all(as_bits(preserved._upper[1:-1]) == as_bits(empty._upper[1:-1]))
    assert (
        as_bits(preserved._lower[-1]) == 0x7FF0000000000001
    )  # positive NaN with smallest mantissa
    assert (
        as_bits(preserved._upper[-1]) == 0x7FFFFFFFFFFFFFFF
    )  # positive NaN with largest mantissa


def test_preserve_any_nan():
    data = np.array(
        [-np.nan, -np.inf, -42, -0.0, 0.0, 42, np.inf, np.nan], dtype=np.float64
    )

    empty = Interval.empty_like(data)
    preserved = Interval.empty_like(data).preserve_any_nan(data, equal_nan=False)

    assert np.all(
        as_bits(preserved._lower)
        == as_bits(np.where(np.isnan(data), data, empty._lower))
    )
    assert np.all(
        as_bits(preserved._upper)
        == as_bits(np.where(np.isnan(data), data, empty._upper))
    )

    preserved = Interval.empty_like(data).preserve_any_nan(data, equal_nan=True)

    assert (
        as_bits(preserved._lower[0, 0]) == 0xFFFFFFFFFFFFFFFF
    )  # negative NaN with largest mantissa
    assert (
        as_bits(preserved._upper[0, 0]) == 0xFFF0000000000001
    )  # negative NaN with smallest mantissa
    assert (
        as_bits(preserved._lower[1, 0]) == 0x7FF0000000000001
    )  # positive NaN with smallest mantissa
    assert (
        as_bits(preserved._upper[1, 0]) == 0x7FFFFFFFFFFFFFFF
    )  # positive NaN with largest mantissa
    assert np.all(as_bits(preserved._lower[:, 1:-1]) == as_bits(empty._lower[1:-1]))
    assert np.all(as_bits(preserved._upper[:, 1:-1]) == as_bits(empty._upper[1:-1]))
    assert (
        as_bits(preserved._lower[0, -1]) == 0xFFFFFFFFFFFFFFFF
    )  # negative NaN with largest mantissa
    assert (
        as_bits(preserved._upper[0, -1]) == 0xFFF0000000000001
    )  # negative NaN with smallest mantissa
    assert (
        as_bits(preserved._lower[1, -1]) == 0x7FF0000000000001
    )  # positive NaN with smallest mantissa
    assert (
        as_bits(preserved._upper[1, -1]) == 0x7FFFFFFFFFFFFFFF
    )  # positive NaN with largest mantissa


def test_preserve_finite():
    data = np.array([-np.nan, -np.inf, -42, -0.0, 0.0, 42, np.inf, np.nan])

    empty = Interval.empty_like(data)
    preserved = Interval.empty_like(data).preserve_finite(data)

    fin = np.finfo(data.dtype).max

    assert np.all(
        as_bits(preserved._lower)
        == as_bits(np.where(np.isfinite(data), -fin, empty._lower))
    )
    assert np.all(
        as_bits(preserved._upper)
        == as_bits(np.where(np.isfinite(data), fin, empty._upper))
    )


def test_preserve_non_nan():
    data = np.array([-np.nan, -np.inf, -42, -0.0, 0.0, 42, np.inf, np.nan])

    empty = Interval.empty_like(data)
    preserved = Interval.empty_like(data).preserve_non_nan(data)

    assert np.all(
        as_bits(preserved._lower)
        == as_bits(np.where(np.isnan(data), empty._lower, -np.inf))
    )
    assert np.all(
        as_bits(preserved._upper)
        == as_bits(np.where(np.isnan(data), empty._upper, np.inf))
    )

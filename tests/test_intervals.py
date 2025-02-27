import numpy as np
from numcodecs_safeguards.safeguards.elementwise.abs import AbsoluteErrorBoundSafeguard
from numcodecs_safeguards.safeguards.elementwise.sign import SignPreservingSafeguard
from numcodecs_safeguards.safeguards.elementwise.zero import ZeroIsZeroSafeguard


def test_sign():
    safeguard = SignPreservingSafeguard()

    intervals = safeguard._compute_intervals(np.arange(0, 10, dtype=np.uint8))
    np.testing.assert_equal(
        intervals.lower,
        np.array([[0] + [1] * 9], dtype=np.uint8),
    )
    np.testing.assert_equal(
        intervals.upper,
        np.array([[0] + [255] * 9], dtype=np.uint8),
    )

    intervals = safeguard._compute_intervals(np.arange(-9, 10, dtype=np.int8))
    np.testing.assert_equal(
        intervals.lower,
        np.array([[-128] * 9 + [0] + [1] * 9], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals.upper,
        np.array([[-1] * 9 + [0] + [127] * 9], dtype=np.int8),
    )


def test_abs():
    safeguard = AbsoluteErrorBoundSafeguard(eb_abs=2)

    intervals = safeguard._compute_intervals(np.arange(0, 10, dtype=np.uint8))
    np.testing.assert_equal(
        intervals.lower,
        np.array([[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint8),
    )
    np.testing.assert_equal(
        intervals.upper,
        np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.uint8),
    )

    intervals = safeguard._compute_intervals(np.arange(-9, 10, dtype=np.int8))
    np.testing.assert_equal(
        intervals.lower,
        np.arange(-11, 8, dtype=np.int8).reshape(1, -1),
    )
    np.testing.assert_equal(
        intervals.upper,
        np.arange(-7, 12, dtype=np.int8).reshape(1, -1),
    )

    intervals = safeguard._compute_intervals(
        np.array([-128, -127, -126, -125, 124, 125, 126, 127], dtype=np.int8)
    )
    np.testing.assert_equal(
        intervals.lower,
        np.array([[-128, -128, -128, -127, 122, 123, 124, 125]], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals.upper,
        np.array([[-126, -125, -124, -123, 126, 127, 127, 127]], dtype=np.int8),
    )


def test_sign_abs():
    data = np.arange(-4, 5, dtype=np.int8)

    sign_intervals = SignPreservingSafeguard()._compute_intervals(data)
    abs_intervals = AbsoluteErrorBoundSafeguard(eb_abs=2)._compute_intervals(data)

    intervals = sign_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals.lower,
        np.array([[-6, -5, -4, -3, 0, 1, 1, 1, 2]], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals.upper,
        np.array([[-2, -1, -1, -1, 0, 3, 4, 5, 6]], dtype=np.int8),
    )


def test_zero_abs():
    data = np.arange(-4, 5, dtype=np.int8)

    zero_intervals = ZeroIsZeroSafeguard(zero=-1)._compute_intervals(data)
    abs_intervals = AbsoluteErrorBoundSafeguard(eb_abs=2)._compute_intervals(data)

    intervals = zero_intervals.intersect(abs_intervals)

    np.testing.assert_equal(
        intervals.lower,
        np.array([[-6, -5, -4, -1, -2, -1, 0, 1, 2]], dtype=np.int8),
    )
    np.testing.assert_equal(
        intervals.upper,
        np.array([[-2, -1, 0, -1, 2, 3, 4, 5, 6]], dtype=np.int8),
    )

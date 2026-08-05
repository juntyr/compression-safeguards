import itertools

import numpy as np

from compression_safeguards.safeguards.pointwise.same import EquivalentValueSafeguard
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.cast import to_total_order

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    values = [0, 42, -1024, np.finfo(float).min]
    if np.issubdtype(data.dtype, np.floating):
        values += [-0.0, +0.0, -np.inf, +np.inf, -np.nan, +np.nan]

    for exclusive in [True, False]:
        for value in values:
            safeguard = dict(kind="equivalent", value=value, exclusive=exclusive)

            decoded = encode_decode_zero(data, safeguards=[safeguard])
            if np.issubdtype(data.dtype, np.floating) and np.isnan(value):
                assert np.all(~np.isnan(data) | np.isnan(decoded))
            else:
                assert np.all((data != value) | (decoded == value))

            decoded = encode_decode_neg(data, safeguards=[safeguard])
            if np.issubdtype(data.dtype, np.floating) and np.isnan(value):
                assert np.all(~np.isnan(data) | np.isnan(decoded))
            else:
                assert np.all((data != value) | (decoded == value))

            decoded = encode_decode_identity(data, safeguards=[safeguard])
            if np.issubdtype(data.dtype, np.floating) and np.isnan(value):
                assert np.all(~np.isnan(data) | np.isnan(decoded))
            else:
                assert np.all((data != value) | (decoded == value))

            decoded = encode_decode_noise(data, safeguards=[safeguard])
            if np.issubdtype(data.dtype, np.floating) and np.isnan(value):
                assert np.all(~np.isnan(data) | np.isnan(decoded))
            else:
                assert np.all((data != value) | (decoded == value))


def test_empty():
    check_all_codecs(np.empty(0))


def test_dimensions():
    check_all_codecs(np.array(42.0))
    check_all_codecs(np.array([42.0]))
    check_all_codecs(np.array([[42.0]]))
    check_all_codecs(np.array([[[42.0]]]))


def test_arange():
    check_all_codecs(np.arange(100, dtype=float))


def test_linspace():
    check_all_codecs(np.linspace(-1024, 1024, 2831))


def test_edge_cases():
    check_all_codecs(
        np.array(
            [
                np.inf,
                np.nan,
                -np.inf,
                -np.nan,
                np.finfo(float).min,
                np.finfo(float).max,
                np.finfo(float).smallest_normal,
                -np.finfo(float).smallest_normal,
                np.finfo(float).smallest_subnormal,
                -np.finfo(float).smallest_subnormal,
                0.0,
                -0.0,
            ]
        )
    )


def test_nan_intervals():
    imin = neg_nan_min = np.iinfo(np.uint64).min
    imax = pos_nan_max = np.iinfo(np.uint64).max

    non_nan_min = to_total_order(np.array(np.float64(-np.inf)))
    non_nan_max = to_total_order(np.array(np.float64(+np.inf)))

    neg_nan_max = non_nan_min - 1
    pos_nan_min = non_nan_max + 1

    for value in [np.nan, -np.nan]:
        intervals = EquivalentValueSafeguard(
            value=np.float64(value), exclusive=False
        ).compute_safe_intervals(
            data=np.array(np.float64(0.0)), late_bound=Bindings.EMPTY
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._lower), np.array([[imin]])
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._upper), np.array([[imax]])
        )

    for value, data in itertools.product([np.nan, -np.nan], [np.nan, -np.nan]):
        intervals = EquivalentValueSafeguard(
            value=np.float64(value), exclusive=False
        ).compute_safe_intervals(
            data=np.array(np.float64(data)), late_bound=Bindings.EMPTY
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._lower), np.array([[neg_nan_min], [pos_nan_min]])
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._upper), np.array([[neg_nan_max], [pos_nan_max]])
        )

    for value in [np.nan, -np.nan]:
        intervals = EquivalentValueSafeguard(
            value=np.float64(value), exclusive=True
        ).compute_safe_intervals(
            data=np.array(np.float64(0.0)), late_bound=Bindings.EMPTY
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._lower), np.array([[non_nan_min]])
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._upper), np.array([[non_nan_max]])
        )

    for value, data in itertools.product([np.nan, -np.nan], [np.nan, -np.nan]):
        intervals = EquivalentValueSafeguard(
            value=np.float64(value), exclusive=True
        ).compute_safe_intervals(
            data=np.array(np.float64(data)), late_bound=Bindings.EMPTY
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._lower), np.array([[neg_nan_min], [pos_nan_min]])
        )
        np.testing.assert_array_equal(
            to_total_order(intervals._upper), np.array([[neg_nan_max], [pos_nan_max]])
        )


def test_fuzzer_found_exclusive_negative_nan_value():
    data = np.array([[-np.nan], [6.214029e27]], dtype=np.float32)
    decoded = np.array([[1.4012985e-45], [1.7591685e22]], dtype=np.float32)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="equivalent", value="$x", exclusive=True),
        ],
    )

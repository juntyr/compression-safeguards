import numpy as np

from .codecs import (
    encode_decode_identity,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    values = [0, 42, -1024, np.finfo(float).min]
    if np.issubdtype(data.dtype, np.floating):
        values += [-0.0, +0.0, -np.inf, +np.inf, -np.nan, +np.nan]

    for value in values:
        safeguard = dict(kind="equivalent", value=value)

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

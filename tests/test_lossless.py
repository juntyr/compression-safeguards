import numpy as np

from compression_safeguards.utils.cast import as_bits

from .codecs import (
    encode_decode_identity,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    safeguard = dict(kind="lossless")

    decoded = encode_decode_zero(data, safeguards=[safeguard])
    assert np.all(as_bits(data) == as_bits(decoded))

    decoded = encode_decode_neg(data, safeguards=[safeguard])
    assert np.all(as_bits(data) == as_bits(decoded))

    decoded = encode_decode_identity(data, safeguards=[safeguard])
    assert np.all(as_bits(data) == as_bits(decoded))

    decoded = encode_decode_noise(data, safeguards=[safeguard])
    assert np.all(as_bits(data) == as_bits(decoded))


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

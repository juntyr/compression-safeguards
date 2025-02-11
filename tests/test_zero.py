import numpy as np


from .codecs import (
    encode_decode_zero,
    encode_decode_neg,
    encode_decode_identity,
    encode_decode_noise,
)


def check_all_codecs(data: np.ndarray):
    for zero in [None, 0, 42, -1024, np.finfo(float).min]:
        safeguard = dict(kind="zero")
        if zero is not None:
            safeguard["zero"] = zero
        else:
            zero = 0

        decoded = encode_decode_zero(data, safeguards=[safeguard])
        assert np.all((data != zero) | (decoded == zero))

        decoded = encode_decode_neg(data, safeguards=[safeguard])
        assert np.all((data != zero) | (decoded == zero))

        decoded = encode_decode_identity(data, safeguards=[safeguard])
        assert np.all((data != zero) | (decoded == zero))

        decoded = encode_decode_noise(data, safeguards=[safeguard])
        assert np.all((data != zero) | (decoded == zero))


def test_empty():
    check_all_codecs(np.empty(0))


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
                np.finfo(float).tiny,
                -np.finfo(float).tiny,
            ]
        )
    )

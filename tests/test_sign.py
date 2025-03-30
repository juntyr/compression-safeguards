import numpy as np

from numcodecs_safeguards.safeguards.elementwise.sign import SignPreservingSafeguard

from .codecs import (
    encode_decode_zero,
    encode_decode_neg,
    encode_decode_identity,
    encode_decode_noise,
)


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(data, safeguards=[dict(kind="sign")])
    assert np.all(
        SignPreservingSafeguard()._sign(data)
        == SignPreservingSafeguard()._sign(decoded)
    )

    decoded = encode_decode_neg(data, safeguards=[dict(kind="sign")])
    assert np.all(
        SignPreservingSafeguard()._sign(data)
        == SignPreservingSafeguard()._sign(decoded)
    )

    decoded = encode_decode_identity(data, safeguards=[dict(kind="sign")])
    assert np.all(
        SignPreservingSafeguard()._sign(data)
        == SignPreservingSafeguard()._sign(decoded)
    )

    decoded = encode_decode_noise(data, safeguards=[dict(kind="sign")])
    assert np.all(
        SignPreservingSafeguard()._sign(data)
        == SignPreservingSafeguard()._sign(decoded)
    )


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
                -0.0,
                +0.0,
            ]
        )
    )

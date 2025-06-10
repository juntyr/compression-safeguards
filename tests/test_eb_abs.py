import numpy as np

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(data, safeguards=[dict(kind="eb", type="abs", eb=0.1)])
    np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.1)

    decoded = encode_decode_neg(data, safeguards=[dict(kind="eb", type="abs", eb=0.1)])
    np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.1)

    decoded = encode_decode_identity(
        data, safeguards=[dict(kind="eb", type="abs", eb=0.1)]
    )
    np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.0)

    decoded = encode_decode_noise(
        data, safeguards=[dict(kind="eb", type="abs", eb=0.1)]
    )
    np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.1)


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


def test_fuzzer_rounding_error():
    data = np.array([3785425606792480904, 9838263505978427528], dtype=np.uint64)
    decoded = np.array([506381766090041480, 4722174329285543943], dtype=np.uint64)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="eb", type="abs", eb=2.2250738585072014e-308, equal_nan=True),
        ],
    )

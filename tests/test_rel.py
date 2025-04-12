import numpy as np

from .codecs import (
    encode_decode_zero,
    encode_decode_neg,
    encode_decode_identity,
    encode_decode_noise,
    encode_decode_mock,
)


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(
        data,
        safeguards=[dict(kind="rel", eb_rel=0.1)],
    )
    np.testing.assert_allclose(decoded, data, rtol=0.1)

    decoded = encode_decode_neg(
        data,
        safeguards=[dict(kind="rel", eb_rel=0.1)],
    )
    np.testing.assert_allclose(decoded, data, rtol=0.1)

    decoded = encode_decode_identity(
        data,
        safeguards=[dict(kind="rel", eb_rel=0.1)],
    )
    np.testing.assert_allclose(decoded, data, rtol=0.0)

    decoded = encode_decode_noise(
        data,
        safeguards=[dict(kind="rel", eb_rel=0.1)],
    )
    np.testing.assert_allclose(decoded, data, rtol=0.1)


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


def test_fuzzer_inf_times_zero():
    data = np.array(
        [
            0,
            0,
        ],
        dtype=np.uint16,
    )
    decoded = np.array(
        [
            0,
            0,
        ],
        dtype=np.uint16,
    )

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="zero", zero=0.0),
            dict(kind="rel", eb_rel=4.287938165015999e290, equal_nan=False),
        ],
    )

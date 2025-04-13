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
        safeguards=[dict(kind="ratio", eb_ratio=10**1.0)],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(decoded, data, rtol=10.0, atol=0.0)

    decoded = encode_decode_neg(
        data,
        safeguards=[dict(kind="ratio", eb_ratio=10**1.0)],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(decoded, data, rtol=10.0, atol=0.0)

    decoded = encode_decode_identity(
        data,
        safeguards=[dict(kind="ratio", eb_ratio=10**1.0)],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(decoded, data, rtol=10.0, atol=0.0)

    decoded = encode_decode_noise(
        data,
        safeguards=[dict(kind="ratio", eb_ratio=10**1.0)],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(decoded, data, rtol=10.0, atol=0.0)


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


def test_fuzzer_overcorrect():
    data = np.array([255, 0], dtype=np.uint8)
    decoded = np.array([0, 4], dtype=np.uint8)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="zero", zero=-1),
            dict(kind="ratio", eb_ratio=3.567564553293311e293),
        ],
    )


def test_fuzzer_overflow():
    data = np.array([506, 0, 64000, 57094, 65535, 255, 65321, 65535], dtype=np.uint16)
    decoded = np.array(
        [65535, 65535, 1535, 1285, 64215, 64250, 9731, 10499], dtype=np.uint16
    )

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="ratio", eb_ratio=10**61, equal_nan=True),
        ],
    )


def test_fuzzer_rounding_error():
    data = np.array(
        [5723915, 0, 1460076544, -43177, -1, -1, -1, -1, -1, -1], dtype=np.int32
    )
    decoded = np.array(
        [-1, -1, -1, -1, -1, -1, 33554431, -16777216, -1, -1], dtype=np.int32
    )

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="zero", zero=1.6989962568688874e308),
            dict(kind="ratio", eb_ratio=2.5924625501554395e303, equal_nan=False),
        ],
    )


def test_fuzzer_int_to_float_precision():
    data = np.array([71789313200750347])
    decoded = np.array([2821266740684990247])

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="ratio", eb_ratio=10**2.2250738585072014e-308, equal_nan=True),
        ],
    )


def test_fuzzer_issue_9():
    data = np.array(
        [
            1499027801,
            1499027801,
            117922137,
            482048,
            117901063,
            2080835335,
            117901063,
            117900551,
        ],
        dtype=np.int32,
    )
    decoded = np.array(
        [
            117901063,
            117901063,
            117901063,
            117901063,
            117901063,
            117901092,
            -1761147129,
            -1751672937,
        ],
        dtype=np.int32,
    )

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="ratio", eb_ratio=1.0645163269184692e308, equal_nan=True),
            # neither the findiff-abs nor monotonicity safeguards apply since
            #  they require windows larger than the data
        ],
    )

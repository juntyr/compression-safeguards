import numpy as np

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_none,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(
        data,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="forward",
                order=1,
                accuracy=1,
                dx=1,
                eb_abs=0.1,
            )
        ],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(np.diff(decoded), np.diff(data), rtol=0.0, atol=0.1)

    decoded = encode_decode_neg(
        data,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="forward",
                order=1,
                accuracy=1,
                dx=1,
                eb_abs=0.1,
            )
        ],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(np.diff(decoded), np.diff(data), rtol=0.0, atol=0.1)

    decoded = encode_decode_identity(
        data,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="forward",
                order=1,
                accuracy=1,
                dx=1,
                eb_abs=0.1,
            )
        ],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(np.diff(decoded), np.diff(data), rtol=0.0, atol=0.1)

    decoded = encode_decode_noise(
        data,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="forward",
                order=1,
                accuracy=1,
                dx=1,
                eb_abs=0.1,
            )
        ],
    )
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        np.testing.assert_allclose(np.diff(decoded), np.diff(data), rtol=0.0, atol=0.1)


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


def test_fuzzer_int_iter():
    data = np.array([65373], dtype=np.uint16)
    decoded = np.array([42246], dtype=np.uint16)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="backwards",
                order=0,
                accuracy=1,
                dx=2.2250738585072014e-308,
                eb_abs=2.2250738585072014e-308,
                axis=0,
            ),
        ],
    )


def test_fuzzer_fraction_overflow():
    data = np.array([7], dtype=np.int8)
    decoded = np.array([0], dtype=np.int8)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="backwards",
                order=7,
                accuracy=6,
                dx=7.215110354450764e305,
                eb_abs=2.2250738585072014e-308,
                axis=0,
            ),
        ],
    )


def test_fuzzer_fraction_compare():
    data = np.array([1978047305655558])

    encode_decode_none(
        data,
        safeguards=[
            dict(kind="zero", zero=7),
            dict(
                kind="findiff_abs",
                type="backwards",
                order=7,
                accuracy=7,
                dx=2.2250738585072014e-308,
                eb_abs=0,
                axis=None,
            ),
            dict(kind="sign"),
        ],
    )


def test_fuzzer_eb_abs():
    data = np.array([[-27, 8, 8], [8, 8, 8], [8, 8, 8]], dtype=np.int8)
    decoded = np.array([[8, 8, 8], [8, 8, 8], [8, 8, 8]], dtype=np.int8)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="forward",
                order=8,
                accuracy=8,
                dx=8,
                eb_abs=8,
                axis=8,
            ),
            dict(kind="sign"),
        ],
    )


def test_fuzzer_fraction_float_overflow():
    data = np.array([[0], [0], [7], [0], [4], [0], [59], [199]], dtype=np.uint16)
    decoded = np.array(
        [[1], [1], [0], [30720], [124], [32768], [16427], [3797]], dtype=np.uint16
    )

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="findiff_abs",
                type="forward",
                order=1,
                accuracy=3,
                dx=59,
                eb_abs=8.812221249325077e307,
            ),
            dict(kind="sign"),
        ],
    )

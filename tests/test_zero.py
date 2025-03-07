import numpy as np

from numcodecs_safeguards.intervals import _as_bits

from .codecs import (
    encode_decode_zero,
    encode_decode_neg,
    encode_decode_identity,
    encode_decode_noise,
    encode_decode_mock,
)


def check_all_codecs(data: np.ndarray):
    for zero in [None, 0, 42, -1024, np.finfo(float).min]:
        safeguard = dict(kind="zero")
        if zero is not None:
            safeguard["zero"] = zero
        else:
            zero = 0
        zero = _as_bits(np.full((), zero, dtype=data.dtype))

        decoded = encode_decode_zero(data, safeguards=[safeguard])
        assert np.all((_as_bits(data) != zero) | (_as_bits(decoded) == zero))

        decoded = encode_decode_neg(data, safeguards=[safeguard])
        assert np.all((_as_bits(data) != zero) | (_as_bits(decoded) == zero))

        decoded = encode_decode_identity(data, safeguards=[safeguard])
        assert np.all((_as_bits(data) != zero) | (_as_bits(decoded) == zero))

        decoded = encode_decode_noise(data, safeguards=[safeguard])
        assert np.all((_as_bits(data) != zero) | (_as_bits(decoded) == zero))


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


def test_fuzzer_invalid_cast():
    data = np.array([115491483746327])
    decoded = np.array([150740651871305728])

    encode_decode_mock(
        data,
        decoded,
        safeguards=[dict(kind="zero", zero=5.760455112138539e292)],
    )

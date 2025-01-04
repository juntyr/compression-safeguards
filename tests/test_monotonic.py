import numpy as np


from .codecs import (
    encode_decode_zero,
    encode_decode_neg,
    encode_decode_identity,
    encode_decode_noise,
)


def check_all_codecs(data: np.ndarray):
    for window in range(1, 3 + 1):
        encode_decode_zero(data, guardrails=[dict(kind="monotonic", window=window)])
        encode_decode_neg(data, guardrails=[dict(kind="monotonic", window=window)])
        encode_decode_identity(data, guardrails=[dict(kind="monotonic", window=window)])
        encode_decode_noise(data, guardrails=[dict(kind="monotonic", window=window)])


def test_empty():
    check_all_codecs(np.empty(0))


def test_arange():
    check_all_codecs(np.arange(100, dtype=float))


def test_linspace():
    check_all_codecs(np.linspace(-1024, 1024, 2831))


def test_edge_cases():
    check_all_codecs(
        np.array(
            [np.inf, np.nan, -np.inf, -np.nan, np.finfo(float).min, np.finfo(float).max]
        )
    )


def test_rounded_cos():
    x = np.linspace(0.0, np.pi * 4.0, 100)
    data = np.round(np.cos(x) / 0.1) * 0.1

    check_all_codecs(data)


def test_cos_sin():
    x = np.linspace(0.0, np.pi * 4.0, 100)
    x, y = np.meshgrid(x, x)
    data = np.stack([np.cos(x), np.sin(y)], axis=-1)

    check_all_codecs(data)


def test_cos_sin_cos():
    x = np.linspace(0.0, np.pi * 2.0, 10)
    x, y, z = np.meshgrid(x, x, x)
    z += np.pi
    data = np.stack([np.cos(x), np.sin(y), np.cos(z)], axis=-1)

    check_all_codecs(data)

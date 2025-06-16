import numpy as np

from compression_safeguards.safeguards.pointwise.sign import SignPreservingSafeguard
from compression_safeguards.utils.bindings import Bindings

from .codecs import (
    encode_decode_identity,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(data, safeguards=[dict(kind="sign")])
    assert np.all(
        np.where(
            np.isnan(data),
            np.isnan(decoded) & (np.signbit(data) == np.signbit(decoded)),
            np.sign(data) == np.sign(decoded),
        )
    )

    decoded = encode_decode_neg(data, safeguards=[dict(kind="sign")])
    assert np.all(
        np.where(
            np.isnan(data),
            np.isnan(decoded) & (np.signbit(data) == np.signbit(decoded)),
            np.sign(data) == np.sign(decoded),
        )
    )

    decoded = encode_decode_identity(data, safeguards=[dict(kind="sign")])
    assert np.all(
        np.where(
            np.isnan(data),
            np.isnan(decoded) & (np.signbit(data) == np.signbit(decoded)),
            np.sign(data) == np.sign(decoded),
        )
    )

    decoded = encode_decode_noise(data, safeguards=[dict(kind="sign")])
    assert np.all(
        np.where(
            np.isnan(data),
            np.isnan(decoded) & (np.signbit(data) == np.signbit(decoded)),
            np.sign(data) == np.sign(decoded),
        )
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


def test_late_bound():
    safeguard = SignPreservingSafeguard(offset="offset")

    data = np.arange(6, dtype=np.uint8).reshape(2, 3)

    vmin, vmax = np.iinfo(data.dtype).min, np.iinfo(data.dtype).max

    late_bound = Bindings(
        offset=np.array([0, 2, 2, 2, 2, 5], dtype=np.uint8).reshape(2, 3),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    print(valid)
    assert np.all(
        valid._lower
        == np.array(
            [0, vmin, 2, 2 + 1, 2 + 1, 5],
            dtype=np.uint8,
        )
    )
    assert np.all(
        valid._upper
        == np.array(
            [0, 2 - 1, 2, vmax, vmax, 5],
            dtype=np.uint8,
        )
    )

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    # -data wraps around for uint8
    assert np.all(ok == np.array([True, False, False, True, True, False]).reshape(2, 3))

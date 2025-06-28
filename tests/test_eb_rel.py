import numpy as np

from compression_safeguards.safeguards.pointwise.eb import ErrorBoundSafeguard
from compression_safeguards.utils.bindings import Bindings

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    decoded = encode_decode_zero(
        data,
        safeguards=[dict(kind="eb", type="rel", eb=0.1)],
    )
    np.testing.assert_allclose(decoded, data, rtol=0.1)

    decoded = encode_decode_neg(
        data,
        safeguards=[dict(kind="eb", type="rel", eb=0.1)],
    )
    np.testing.assert_allclose(decoded, data, rtol=0.1)

    decoded = encode_decode_identity(
        data,
        safeguards=[dict(kind="eb", type="rel", eb=0.1)],
    )
    np.testing.assert_allclose(decoded, data, rtol=0.0)

    decoded = encode_decode_noise(
        data,
        safeguards=[dict(kind="eb", type="rel", eb=0.1)],
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
            dict(kind="same", value=0),
            dict(kind="eb", type="rel", eb=4.287938165015999e290, equal_nan=False),
        ],
    )


def test_late_bound_eb():
    safeguard = ErrorBoundSafeguard(type="rel", eb="eb")
    assert safeguard.late_bound == {"eb"}

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(
        eb=np.array([5, 4, 3, 2, 1, 0]).reshape(2, 3),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(valid._lower == (data.flatten() - np.array([0, 4, 6, 6, 4, 0])))
    assert np.all(valid._upper == (data.flatten() + np.array([0, 4, 6, 6, 4, 0])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, True, False, False]).reshape(2, 3))

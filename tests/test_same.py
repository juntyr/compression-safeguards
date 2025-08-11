import numpy as np
import pytest

from compression_safeguards.safeguards.pointwise.same import SameValueSafeguard
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.cast import as_bits

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    for value in [0, 42, -1024, np.finfo(float).min]:
        safeguard = dict(kind="same", value=value)
        value = as_bits(np.full((), value, dtype=data.dtype))

        decoded = encode_decode_zero(data, safeguards=[safeguard])
        assert np.all((as_bits(data) != value) | (as_bits(decoded) == value))

        decoded = encode_decode_neg(data, safeguards=[safeguard])
        assert np.all((as_bits(data) != value) | (as_bits(decoded) == value))

        decoded = encode_decode_identity(data, safeguards=[safeguard])
        assert np.all((as_bits(data) != value) | (as_bits(decoded) == value))

        decoded = encode_decode_noise(data, safeguards=[safeguard])
        assert np.all((as_bits(data) != value) | (as_bits(decoded) == value))


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
                np.finfo(float).smallest_normal,
                -np.finfo(float).smallest_normal,
                np.finfo(float).smallest_subnormal,
                -np.finfo(float).smallest_subnormal,
                0.0,
                -0.0,
            ]
        )
    )


def test_fuzzer_invalid_cast():
    data = np.array([115491483746327])
    decoded = np.array([150740651871305728])

    with pytest.raises(
        TypeError,
        match="cannot losslessly cast same safeguard value from float64 to int64",
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[dict(kind="same", value=5.760455112138539e292)],
        )


def test_late_bound_inclusive():
    safeguard = SameValueSafeguard(value="same")
    assert safeguard.late_bound == {"same"}

    data = np.arange(6).reshape(2, 3)

    vmin, vmax = np.iinfo(data.dtype).min, np.iinfo(data.dtype).max

    late_bound = Bindings(
        same=np.array([1, 0, 4, 3, 2, 5]).reshape(2, 3),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(valid._lower == np.array([vmin, vmin, vmin, 3, vmin, 5]))
    assert np.all(valid._upper == np.array([vmax, vmax, vmax, 3, vmax, 5]))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, False, True, False]).reshape(2, 3))


def test_late_bound_exclusive():
    safeguard = SameValueSafeguard(value="value", exclusive=True)
    assert safeguard.late_bound == {"value"}

    data = np.arange(6, dtype=np.uint8).reshape(2, 3)

    vmin, vmax = np.iinfo(data.dtype).min, np.iinfo(data.dtype).max

    late_bound = Bindings(
        value=np.array([1, 0, 4, 3, 2, 5], dtype=np.uint8).reshape(2, 3),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(
        valid._lower
        == np.array(
            [[vmin, 0 + 1, vmin, 3, vmin, 5], [1 + 1, vmax, 4 + 1, vmax, 2 + 1, vmax]],
            dtype=np.uint8,
        )
    )
    assert np.all(
        valid._upper
        == np.array(
            [[1 - 1, vmax, 4 - 1, 3, 2 - 1, 5], [vmax, vmin, vmax, vmin, vmax, vmin]],
            dtype=np.uint8,
        )
    )

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, False, True, False]).reshape(2, 3))


def test_fuzzer_pick_negative_shape_mismatch():
    data = np.array([[801]], dtype=np.int16)
    decoded = np.array([[15934]], dtype=np.int16)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="same", value=49, exclusive=True),
            dict(kind="eb", type="rel", eb=1, equal_nan=True),
        ],
    )

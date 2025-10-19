import re
from itertools import product

import numpy as np
import pytest

from compression_safeguards import Safeguards
from compression_safeguards.safeguards.stencil import BoundaryCondition
from compression_safeguards.safeguards.stencil.monotonicity import (
    Monotonicity,
    MonotonicityPreservingSafeguard,
)
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.cast import as_bits

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray, constant_boundary=4.2):
    for monotonicity, window, boundary in product(
        Monotonicity,
        range(1, 3 + 1),
        BoundaryCondition,
    ):
        safeguards = [
            dict(
                kind="monotonicity",
                monotonicity=monotonicity,
                window=window,
                boundary=boundary,
                constant_boundary=constant_boundary
                if boundary == BoundaryCondition.constant
                else None,
            )
        ]

        encode_decode_zero(data, safeguards=safeguards)
        encode_decode_neg(data, safeguards=safeguards)
        encode_decode_identity(data, safeguards=safeguards)
        encode_decode_noise(data, safeguards=safeguards)


def test_empty():
    check_all_codecs(np.empty(0))


def test_dimensions():
    check_all_codecs(np.array(42.0))
    check_all_codecs(np.array(42, dtype=np.int64), constant_boundary=42)
    check_all_codecs(np.array([42.0]))
    check_all_codecs(np.array([[42.0]]))
    check_all_codecs(np.array([[[42.0]]]))


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


def test_monotonicity():
    windows = dict(
        si=np.array([1.0, 2, 3]),  # strictly increasing
        sd=np.array([3.0, 2, 1]),  # strictly decreasing
        co=np.array([2.0, 2, 2]),  # constant
        wi=np.array([1.0, 1, 3]),  # weakly increasing
        wd=np.array([3.0, 1, 1]),  # weakly decreasing
        ns=np.array([3.0, 1, 2]),  # noise
        nf=np.array([1.0, np.nan, 3]),  # non-finite
    )

    # mapping for each monotonicity
    # - keys: which windows activate the safeguard
    # - values: which windows validate without correction
    monotonicities = {
        Monotonicity.strict: dict(
            si=("si",),
            sd=("sd",),
        ),
        Monotonicity.strict_with_consts: dict(
            si=("si",),
            sd=("sd",),
            co=("co",),
        ),
        Monotonicity.strict_to_weak: dict(
            si=("si", "wi", "co"),
            sd=("sd", "wd", "co"),
        ),
        Monotonicity.weak: dict(
            si=("si", "wi", "co"),
            sd=("sd", "wd", "co"),
            wi=("si", "wi", "co"),
            wd=("sd", "wd", "co"),
        ),
    }

    # test for all monotonicities
    for monotonicity, active_allowed in monotonicities.items():
        safeguard = MonotonicityPreservingSafeguard(
            monotonicity, window=1, boundary="valid"
        )

        # test for all possible window combinations
        for data_window, prediction_window in product(windows, windows):
            data = windows[data_window]
            prediction = windows[prediction_window]

            # the constant window needs to adjusted for the weak monotonicities
            #  since the implementation also checks that no overlap with
            #  adjacent elements occurs, which the weak windows have for 1.0
            if prediction_window == "co" and data_window in ("wi", "wd"):
                prediction = np.array([1.0, 1, 1])

            # if the window activates the safeguard ...
            if data_window in active_allowed:
                # the check has to return the expected result
                assert safeguard.check(data, prediction, late_bound=Bindings.EMPTY) == (
                    prediction_window in active_allowed[data_window]
                )

                # correcting the data must pass both checks
                corrected = safeguard.compute_safe_intervals(
                    data, late_bound=Bindings.EMPTY
                ).pick(prediction)
                assert safeguard.check(data, corrected, late_bound=Bindings.EMPTY)
            else:
                # the window doesn't activate the safeguard so the checks must
                #  succeed
                assert safeguard.check(data, prediction, late_bound=Bindings.EMPTY)

                # the window doesn't activate the safeguard so the corrected
                #  array should be bit-equivalent to the prediction array
                assert np.array_equal(
                    as_bits(
                        safeguard.compute_safe_intervals(
                            data, late_bound=Bindings.EMPTY
                        ).pick(prediction)
                    ),
                    as_bits(prediction),
                )


def test_fuzzer_sign_flip():
    data = np.array([14, 47, 0, 0, 254, 255, 255, 255, 0, 0], dtype=np.uint8)
    decoded = np.array([73, 0, 0, 0, 0, 27, 49, 14, 14, 50], dtype=np.uint8)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="monotonicity",
                monotonicity="strict_to_weak",
                window=1,
                boundary="valid",
            ),
            dict(kind="sign"),
        ],
    )


def test_fuzzer_padding_overflow():
    data = np.array([[0.0]], dtype=np.float32)
    decoded = np.array([[-9.444733e21]], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=r"cannot losslessly cast \(some\) values from float64 to float32",
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="monotonicity",
                    monotonicity="weak",
                    window=108,
                    boundary="constant",
                    constant_boundary=-1.7976657042415566e308,
                    axis=None,
                ),
                dict(kind="sign"),
            ],
        )


def test_late_bound_constant_boundary():
    data = np.array([14, 47, 0, 0, 254, 255, 255, 255, 0, 0], dtype=np.uint8)
    decoded = np.array([73, 0, 0, 0, 0, 27, 49, 14, 14, 50], dtype=np.uint8)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="monotonicity",
                monotonicity="strict_to_weak",
                window=1,
                boundary="constant",
                constant_boundary=4,
            ),
            dict(kind="sign"),
        ],
    )

    for c in ["$x", "$X"]:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"monotonicity.constant_boundary: must be scalar but late-bound constant data {c} may not be"
            ),
        ):
            safeguards = Safeguards(
                safeguards=[
                    dict(
                        kind="monotonicity",
                        monotonicity="strict_to_weak",
                        window=1,
                        boundary="constant",
                        constant_boundary=c,
                    ),
                    dict(kind="sign"),
                ],
            )

    safeguards = Safeguards(
        safeguards=[
            dict(
                kind="monotonicity",
                monotonicity="strict_to_weak",
                window=1,
                boundary="constant",
                constant_boundary="const",
            ),
            dict(kind="sign"),
        ],
    )
    assert safeguards.late_bound == {"const"}

    correction = safeguards.compute_correction(
        data, decoded, late_bound=Bindings(const=4)
    )
    safeguards.apply_correction(decoded, correction)

    with pytest.raises(
        ValueError,
        match=r"cannot losslessly cast \(some\) values from int64 to uint8",
    ):
        correction = safeguards.compute_correction(
            data, decoded, late_bound=Bindings(const=-1)
        )


def test_fuzzer_found_broadcast():
    data = np.array([], dtype=np.int8)
    decoded = np.array([], dtype=np.int8)

    with pytest.raises(
        ValueError,
        match=r"monotonicity.constant_boundary=䣿䡈: cannot broadcast from shape \(0,\) to shape \(\)",
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="monotonicity",
                    monotonicity="strict",
                    window=33,
                    boundary="constant",
                    constant_boundary="䣿䡈",
                ),
                dict(kind="sign"),
            ],
            fixed_constants={"䣿䡈": np.array([], dtype=np.int64)},
        )

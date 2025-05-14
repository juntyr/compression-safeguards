from itertools import product

import numpy as np

from numcodecs_safeguards.cast import as_bits
from numcodecs_safeguards.safeguards.stencil.monotonicity import (
    Monotonicity,
    MonotonicityPreservingSafeguard,
)

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    for monotonicity, window, boundary in product(
        Monotonicity, range(1, 3 + 1), ["valid", "edge"]
    ):
        encode_decode_zero(
            data,
            safeguards=[
                dict(
                    kind="monotonicity",
                    monotonicity=monotonicity,
                    window=window,
                    boundary=boundary,
                )
            ],
        )
        encode_decode_neg(
            data,
            safeguards=[
                dict(
                    kind="monotonicity",
                    monotonicity=monotonicity,
                    window=window,
                    boundary=boundary,
                )
            ],
        )
        encode_decode_identity(
            data,
            safeguards=[
                dict(
                    kind="monotonicity",
                    monotonicity=monotonicity,
                    window=window,
                    boundary=boundary,
                )
            ],
        )
        encode_decode_noise(
            data,
            safeguards=[
                dict(
                    kind="monotonicity",
                    monotonicity=monotonicity,
                    window=window,
                    boundary=boundary,
                )
            ],
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
        for data_window, decoded_window in product(windows, windows):
            data = windows[data_window]
            decoded = windows[decoded_window]

            # the constant window needs to adjusted for the weak monotonicities
            #  since the implementation also checks that no overlap with
            #  adjacent elements occurs, which the weak windows have for 1.0
            if decoded_window == "co" and data_window in ("wi", "wd"):
                decoded = np.array([1.0, 1, 1])

            # if the window activates the safeguard ...
            if data_window in active_allowed:
                # the check has to return the expected result
                assert safeguard.check(data, decoded) == (
                    decoded_window in active_allowed[data_window]
                )

                # correcting the data must pass both checks
                corrected = safeguard.compute_safe_intervals(data).pick(decoded)
                assert safeguard.check(data, corrected)
            else:
                # the window doesn't activate the safeguard so the checks must
                #  succeed
                assert safeguard.check(data, decoded)

                # the window doesn't activate the safeguard so the corrected
                #  array should be bit-equivalent to the decoded array
                assert np.array_equal(
                    as_bits(safeguard.compute_safe_intervals(data).pick(decoded)),
                    as_bits(decoded),
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

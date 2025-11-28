import re
from itertools import product

import numpy as np
import pytest

from compression_safeguards import Safeguards
from compression_safeguards.safeguards.stencil import BoundaryCondition
from compression_safeguards.safeguards.stencil.monotonicity import (
    MonotonicityPreservingSafeguard,
)
from compression_safeguards.safeguards.stencil.qoi.eb import (
    StencilQuantityOfInterestErrorBoundSafeguard,
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

MONOTONICITY_QOIS: dict[str, str] = dict(
    strict="""
        all([
            # strictly decreasing sequences stay strictly decreasing
            all(X[1:] < X[:-1]) == all(C["$X"][1:] < C["$X"][:-1]),
            # strictly increasing sequences stay strictly increasing
            all(X[1:] > X[:-1]) == all(C["$X"][1:] > C["$X"][:-1]),
        ])
    """,
    strict_with_consts="""
        all([
            # strictly decreasing sequences stay strictly decreasing
            all(X[1:] < X[:-1]) == all(C["$X"][1:] < C["$X"][:-1]),
            # constant sequences stay constant
            all(X[1:] == X[:-1]) == all(C["$X"][1:] == C["$X"][:-1]),
            # strictly increasing sequences stay strictly increasing
            all(X[1:] > X[:-1]) == all(C["$X"][1:] > C["$X"][:-1]),
        ])
    """,
    strict_to_weak="""
        all([
            # strictly decreasing sequences become weakly decreasing
            #  (1) x strictly decreasing -> x' weakly decreasing or const
            any([all(X[1:] <= X[:-1]), not(all(C["$X"][1:] < C["$X"][:-1]))]),
            #  (2) x' strictly decreasing -> x strictly decreasing
            #      (not x' w -> x s since that fails for x'=x if x is w)
            any([not(all(X[1:] < X[:-1])), all(C["$X"][1:] < C["$X"][:-1])]),
            # strictly increasing sequences become weakly increasing
            #  (1) x strictly increasing -> x' weakly increasing or const
            any([all(X[1:] >= X[:-1]), not(all(C["$X"][1:] > C["$X"][:-1]))]),
            #  (2) x' strictly increasing -> x strictly increasing
            #      (not x' w -> x s since that fails for x'=x if x is w)
            any([not(all(X[1:] > X[:-1])), all(C["$X"][1:] > C["$X"][:-1])]),
        ])
    """,
    weak="""
        all([
            # weakly decreasing & not constant sequences stay weakly decreasing
            #  (1) x weakly decreasing & not constant -> x' weakly decreasing
            any([all(X[1:] <= X[:-1]), not(all([
                all(C["$X"][1:] <= C["$X"][:-1]),
                not(all(C["$X"][1:] == C["$X"][:-1])),
            ]))]),
            #  (2) x' weakly decreasing & not constant -> x weakly decreasing
            #                                              & not constant
            any([not(all([
                all(X[1:] <= X[:-1]),
                not(all(X[1:] == X[:-1])),
            ])), all([
                all(C["$X"][1:] <= C["$X"][:-1]),
                not(all(C["$X"][1:] == C["$X"][:-1])),
            ])]),
            # weakly increasing & not constant sequences stay weakly increasing
            #  (1) x weakly increasing & not constant -> x' weakly increasing
            any([all(X[1:] >= X[:-1]), not(all([
                all(C["$X"][1:] >= C["$X"][:-1]),
                not(all(C["$X"][1:] == C["$X"][:-1])),
            ]))]),
            #  (2) x' weakly increasing & not constant -> x weakly increasing
            #                                              & not constant
            any([not(all([
                all(X[1:] >= X[:-1]),
                not(all(X[1:] == X[:-1])),
            ])), all([
                all(C["$X"][1:] >= C["$X"][:-1]),
                not(all(C["$X"][1:] == C["$X"][:-1])),
            ])]),
        ])
    """,
)


# a -> b is equivalent to any(not(a), b)
MONOTONICITY_ALLOW_SPURIOUS_QOIS: dict[str, str] = dict(
    strict="""
        all([
            # strictly decreasing sequences stay strictly decreasing
            any([all(X[1:] < X[:-1]), not(all(C["$X"][1:] < C["$X"][:-1]))]),
            # strictly increasing sequences stay strictly increasing
            any([all(X[1:] > X[:-1]), not(all(C["$X"][1:] > C["$X"][:-1]))]),
        ])
    """,
    strict_with_consts="""
        all([
            # strictly decreasing sequences stay strictly decreasing
            any([all(X[1:] < X[:-1]), not(all(C["$X"][1:] < C["$X"][:-1]))]),
            # constant sequences stay constant
            any([all(X[1:] == X[:-1]), not(all(C["$X"][1:] == C["$X"][:-1]))]),
            # strictly increasing sequences stay strictly increasing
            any([all(X[1:] > X[:-1]), not(all(C["$X"][1:] > C["$X"][:-1]))]),
        ])
    """,
    strict_to_weak="""
        all([
            # strictly decreasing sequences become weakly decreasing
            any([all(X[1:] <= X[:-1]), not(all(C["$X"][1:] < C["$X"][:-1]))]),
            # strictly increasing sequences become weakly increasing
            any([all(X[1:] >= X[:-1]), not(all(C["$X"][1:] > C["$X"][:-1]))]),
        ])
    """,
    weak="""
        all([
            # weakly decreasing & not constant sequences stay weakly decreasing
            any([all(X[1:] <= X[:-1]), not(all([
                all(C["$X"][1:] <= C["$X"][:-1]),
                not(all(C["$X"][1:] == C["$X"][:-1])),
            ]))]),
            # weakly increasing & not constant sequences stay weakly increasing
            any([all(X[1:] >= X[:-1]), not(all([
                all(C["$X"][1:] >= C["$X"][:-1]),
                not(all(C["$X"][1:] == C["$X"][:-1])),
            ]))]),
        ])
    """,
)


def check_all_codecs(data: np.ndarray, constant_boundary=4.2):
    for qois, monotonicity, window, boundary in product(
        [MONOTONICITY_QOIS, MONOTONICITY_ALLOW_SPURIOUS_QOIS],
        ["strict", "strict_with_consts", "strict_to_weak", "weak"],
        range(1, 3 + 1),
        BoundaryCondition,
    ):
        safeguards = [
            dict(
                kind="qoi_eb_stencil",
                qoi=qois[monotonicity],
                neighbourhood=[
                    dict(
                        axis=axis,
                        before=window,
                        after=window,
                        boundary=boundary,
                        constant_boundary=constant_boundary
                        if boundary == BoundaryCondition.constant
                        else None,
                    )
                ],
                type="abs",
                eb=0,
            )
            for axis in range(data.ndim)
        ]

        sanity_safeguards = Safeguards(
            safeguards=[
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
        )

        corrected = encode_decode_zero(data, safeguards=safeguards)
        assert sanity_safeguards.check(data, corrected)

        corrected = encode_decode_neg(data, safeguards=safeguards)
        assert sanity_safeguards.check(data, corrected)

        corrected = encode_decode_identity(data, safeguards=safeguards)
        assert sanity_safeguards.check(data, corrected)

        corrected = encode_decode_noise(data, safeguards=safeguards)
        assert sanity_safeguards.check(data, corrected)


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


def test_monotonicities_spurious():
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
    monotonicities = dict(
        strict=dict(
            si=("si",),
            sd=("sd",),
        ),
        strict_with_consts=dict(
            si=("si",),
            sd=("sd",),
            co=("co",),
        ),
        strict_to_weak=dict(
            si=("si", "wi", "co"),
            sd=("sd", "wd", "co"),
        ),
        weak=dict(
            si=("si", "wi", "co"),
            sd=("sd", "wd", "co"),
            wi=("si", "wi", "co"),
            wd=("sd", "wd", "co"),
        ),
    )

    # test for all monotonicities
    for monotonicity, active_allowed in monotonicities.items():
        safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
            qoi=MONOTONICITY_ALLOW_SPURIOUS_QOIS[monotonicity],
            neighbourhood=[
                dict(
                    axis=0,
                    before=1,
                    after=1,
                    boundary="valid",
                )
            ],
            type="abs",
            eb=0,
        )
        sanity_safeguard = MonotonicityPreservingSafeguard(
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
                assert safeguard.check(
                    data, prediction, late_bound=Bindings(**{"$X": data})
                ) == (prediction_window in active_allowed[data_window])
                assert sanity_safeguard.check(
                    data, prediction, late_bound=Bindings.EMPTY
                ) == (prediction_window in active_allowed[data_window])

                # correcting the data must pass both checks
                corrected = safeguard.compute_safe_intervals(
                    data, late_bound=Bindings(**{"$X": data})
                ).pick(prediction)
                assert safeguard.check(
                    data, corrected, late_bound=Bindings(**{"$X": data})
                )
                assert sanity_safeguard.check(
                    data, corrected, late_bound=Bindings.EMPTY
                )
            else:
                # the window doesn't activate the safeguard so the checks must
                #  succeed
                assert safeguard.check(
                    data, prediction, late_bound=Bindings(**{"$X": data})
                )
                assert sanity_safeguard.check(
                    data, prediction, late_bound=Bindings.EMPTY
                )

                # the window doesn't activate the safeguard so the corrected
                #  array should be bit-equivalent to the prediction array
                corrected = safeguard.compute_safe_intervals(
                    data, late_bound=Bindings(**{"$X": data})
                ).pick(prediction)
                if not np.all(
                    (as_bits(corrected) == as_bits(prediction))
                    | np.isnan(data)
                    | np.isnan(prediction)
                ):
                    assert False, (
                        f"{corrected} != {prediction} for {data} and {monotonicity}"
                    )


def test_monotonicities_non_spurious():
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
    # - keys: which data windows activate the safeguard
    # - values: which prediction windows validate without correction
    # and set of all prediction windows that can activate the safeguard
    monotonicities = dict(
        strict=(
            dict(
                si=("si",),
                sd=("sd",),
            ),
            ["si", "sd"],
        ),
        strict_with_consts=(
            dict(
                si=("si",),
                sd=("sd",),
                co=("co",),
            ),
            ["si", "sd", "co"],
        ),
        strict_to_weak=(
            dict(
                si=("si", "wi", "co"),
                sd=("sd", "wd", "co"),
            ),
            ["si", "sd"],
        ),
        weak=(
            dict(
                si=("si", "wi", "co"),
                sd=("sd", "wd", "co"),
                wi=("si", "wi", "co"),
                wd=("sd", "wd", "co"),
            ),
            ["si", "sd", "wi", "wd"],
        ),
    )

    # test for all monotonicities
    for monotonicity, (active_allowed, trigger) in monotonicities.items():
        safeguard = StencilQuantityOfInterestErrorBoundSafeguard(
            qoi=MONOTONICITY_QOIS[monotonicity],
            neighbourhood=[
                dict(
                    axis=0,
                    before=1,
                    after=1,
                    boundary="valid",
                )
            ],
            type="abs",
            eb=0,
        )
        sanity_safeguard = MonotonicityPreservingSafeguard(
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
            if (data_window in active_allowed) or (prediction_window in trigger):
                # the check has to return the expected result
                assert safeguard.check(
                    data, prediction, late_bound=Bindings(**{"$X": data})
                ) == (
                    (prediction_window in active_allowed[data_window])
                    if data_window in active_allowed
                    else False
                )
                assert sanity_safeguard.check(
                    data, prediction, late_bound=Bindings.EMPTY
                ) == (
                    (prediction_window in active_allowed[data_window])
                    if data_window in active_allowed
                    else True
                )

                # correcting the data must pass both checks
                corrected = safeguard.compute_safe_intervals(
                    data, late_bound=Bindings(**{"$X": data})
                ).pick(prediction)
                assert safeguard.check(
                    data, corrected, late_bound=Bindings(**{"$X": data})
                )
                assert sanity_safeguard.check(
                    data, corrected, late_bound=Bindings.EMPTY
                )
            else:
                # the window doesn't activate the safeguard so the checks must
                #  succeed
                assert safeguard.check(
                    data, prediction, late_bound=Bindings(**{"$X": data})
                )
                assert sanity_safeguard.check(
                    data, prediction, late_bound=Bindings.EMPTY
                )

                # the window doesn't activate the safeguard, but the stencil
                #  QoIs still impose requirements
                corrected = safeguard.compute_safe_intervals(
                    data, late_bound=Bindings(**{"$X": data})
                ).pick(prediction)
                assert safeguard.check(
                    data, corrected, late_bound=Bindings(**{"$X": data})
                )
                assert sanity_safeguard.check(
                    data, corrected, late_bound=Bindings.EMPTY
                )


def test_fuzzer_sign_flip():
    data = np.array([14, 47, 0, 0, 254, 255, 255, 255, 0, 0], dtype=np.uint8)
    decoded = np.array([73, 0, 0, 0, 0, 27, 49, 14, 14, 50], dtype=np.uint8)

    corrected = encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi=MONOTONICITY_QOIS["strict_to_weak"],
                neighbourhood=[
                    dict(
                        axis=axis,
                        before=1,
                        after=1,
                        boundary="valid",
                    )
                ],
                type="abs",
                eb=0,
            )
            for axis in range(data.ndim)
        ]
        + [dict(kind="sign")],
    )

    assert Safeguards(
        safeguards=[
            dict(
                kind="monotonicity",
                monotonicity="strict_to_weak",
                window=1,
                boundary="valid",
            ),
            dict(kind="sign"),
        ]
    ).check(data, corrected)


def test_fuzzer_padding_overflow():
    data = np.array([[0.0]], dtype=np.float32)
    decoded = np.array([[-9.444733e21]], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=r"qoi_eb_stencil\.neighbourhood\[0\]\.constant_boundary: cannot losslessly cast \(some\) values from float64 to float32",
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi=MONOTONICITY_QOIS["weak"],
                    neighbourhood=[
                        dict(
                            axis=axis,
                            before=108,
                            after=108,
                            boundary="constant",
                            constant_boundary=-1.7976657042415566e308,
                        )
                    ],
                    type="abs",
                    eb=0,
                )
                for axis in range(data.ndim)
            ]
            + [dict(kind="sign")],
        )


def test_late_bound_constant_boundary():
    data = np.array([14, 47, 0, 0, 254, 255, 255, 255, 0, 0], dtype=np.uint8)
    decoded = np.array([73, 0, 0, 0, 0, 27, 49, 14, 14, 50], dtype=np.uint8)

    corrected = encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi=MONOTONICITY_QOIS["strict_to_weak"],
                neighbourhood=[
                    dict(
                        axis=axis,
                        before=1,
                        after=1,
                        boundary="constant",
                        constant_boundary=4,
                    )
                ],
                type="abs",
                eb=0,
            )
            for axis in range(data.ndim)
        ]
        + [dict(kind="sign")],
    )

    assert Safeguards(
        safeguards=[
            dict(
                kind="monotonicity",
                monotonicity="strict_to_weak",
                window=1,
                boundary="constant",
                constant_boundary=4,
            ),
            dict(kind="sign"),
        ]
    ).check(data, corrected)

    for c in ["$x", "$X"]:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"safeguards[0].qoi_eb_stencil.neighbourhood[0].constant_boundary: must be a scalar but late-bound constant data {c} may not be"
            ),
        ):
            safeguards = Safeguards(
                safeguards=[
                    dict(
                        kind="qoi_eb_stencil",
                        qoi=MONOTONICITY_QOIS["strict_to_weak"],
                        neighbourhood=[
                            dict(
                                axis=axis,
                                before=1,
                                after=1,
                                boundary="constant",
                                constant_boundary=c,
                            )
                        ],
                        type="abs",
                        eb=0,
                    )
                    for axis in range(data.ndim)
                ]
            )

    safeguards = Safeguards(
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi=MONOTONICITY_QOIS["strict_to_weak"],
                neighbourhood=[
                    dict(
                        axis=axis,
                        before=1,
                        after=1,
                        boundary="constant",
                        constant_boundary="const",
                    )
                ],
                type="abs",
                eb=0,
            )
            for axis in range(data.ndim)
        ],
    )
    assert (safeguards.late_bound - safeguards.builtin_late_bound) == {"const"}

    correction = safeguards.compute_correction(
        data, decoded, late_bound=Bindings(const=4)
    )
    corrected = safeguards.apply_correction(decoded, correction)

    assert Safeguards(
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
    ).check(data, corrected, late_bound=dict(const=4))

    with pytest.raises(
        ValueError,
        match=r"qoi_eb_stencil\.neighbourhood\[0\]\.constant_boundary=const: cannot losslessly cast \(some\) values from int64 to uint8",
    ):
        correction = safeguards.compute_correction(
            data, decoded, late_bound=Bindings(const=-1)
        )


def test_fuzzer_found_broadcast():
    data = np.array([0], dtype=np.int8)
    decoded = np.array([0], dtype=np.int8)

    with pytest.raises(
        ValueError,
        match=r"qoi_eb_stencil\.neighbourhood\[0\]\.constant_boundary=䣿䡈: cannot broadcast from shape \(0,\) to shape \(\)",
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="qoi_eb_stencil",
                    qoi=MONOTONICITY_QOIS["strict"],
                    neighbourhood=[
                        dict(
                            axis=axis,
                            before=33,
                            after=33,
                            boundary="constant",
                            constant_boundary="䣿䡈",
                        )
                    ],
                    type="abs",
                    eb=0,
                )
                for axis in range(data.ndim)
            ]
            + [dict(kind="sign")],
            fixed_constants={"䣿䡈": np.array([], dtype=np.int64)},
        )


def test_fuzzer_found_slice_indexing():
    data = np.array([[1431655765]], dtype=np.uint32)

    corrected = encode_decode_zero(
        data,
        safeguards=[
            dict(kind="same", value="$x_max", exclusive=True),
            dict(
                kind="select",
                selector="UU",
                safeguards=[
                    dict(
                        kind="qoi_eb_stencil",
                        qoi=MONOTONICITY_QOIS["strict_with_consts"],
                        neighbourhood=[
                            dict(
                                axis=axis,
                                before=85,
                                after=85,
                                boundary="constant",
                                constant_boundary=85,
                            )
                        ],
                        type="abs",
                        eb=0,
                    )
                    for axis in range(data.ndim)
                ],
            ),
        ],
        fixed_constants=dict(UU=0),
    )

    assert Safeguards(
        safeguards=[
            dict(kind="same", value="$x_max", exclusive=True),
            dict(
                kind="select",
                selector="UU",
                safeguards=[
                    dict(
                        kind="monotonicity",
                        monotonicity="strict_with_consts",
                        window=85,
                        boundary="constant",
                        constant_boundary=85,
                        axis=None,
                    )
                ],
            ),
        ]
    ).check(data, corrected, late_bound={"UU": 0, "$x_max": np.nanmax(data)})


def test_fuzzer_found_wrapping_const_sequence():
    data = np.array(
        [
            [135, 135, 1, 44, 1, 0],
            [43, 1, 26, 0, 255, 255],
            [112, 255, 255, 112, 112, 112],
        ],
        dtype=np.uint8,
    )

    corrected = encode_decode_zero(
        data,
        safeguards=[
            dict(
                kind="qoi_eb_stencil",
                qoi=MONOTONICITY_QOIS["strict_with_consts"],
                neighbourhood=[dict(axis=axis, before=1, after=1, boundary="wrap")],
                type="abs",
                eb=0,
            )
            for axis in range(data.ndim)
        ],
    )

    assert Safeguards(
        safeguards=[
            dict(
                kind="monotonicity",
                monotonicity="strict_with_consts",
                window=1,
                boundary="wrap",
                axis=None,
            )
        ]
    ).check(data, corrected)

from collections.abc import Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.pointwise.abc import PointwiseSafeguard
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard
from compression_safeguards.utils.bindings import Bindings, Parameter
from compression_safeguards.utils.intervals import Interval, IntervalUnion
from compression_safeguards.utils.typing import JSON, S, T

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray):
    for combinator in ["any", "all"]:
        decoded = encode_decode_zero(
            data,
            safeguards=[
                dict(
                    kind=combinator,
                    safeguards=[
                        dict(kind="eb", type="ratio", eb=1.1),
                        dict(kind="eb", type="abs", eb=0.1),
                    ],
                )
            ],
        )
        np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)

        decoded = encode_decode_neg(
            data,
            safeguards=[
                dict(
                    kind=combinator,
                    safeguards=[
                        dict(kind="eb", type="ratio", eb=1.1),
                        dict(kind="eb", type="abs", eb=0.1),
                    ],
                )
            ],
        )
        np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)

        decoded = encode_decode_identity(
            data,
            safeguards=[
                dict(
                    kind=combinator,
                    safeguards=[
                        dict(kind="eb", type="ratio", eb=1.1),
                        dict(kind="eb", type="abs", eb=0.1),
                    ],
                )
            ],
        )
        np.testing.assert_allclose(decoded, data, rtol=0.0, atol=0.0)

        decoded = encode_decode_noise(
            data,
            safeguards=[
                dict(
                    kind=combinator,
                    safeguards=[
                        dict(kind="eb", type="ratio", eb=1.1),
                        dict(kind="eb", type="abs", eb=0.1),
                    ],
                )
            ],
        )
        np.testing.assert_allclose(decoded, data, rtol=0.1, atol=0.1)


def test_empty():
    check_all_codecs(np.empty(0))


def test_dimensions():
    check_all_codecs(np.array(42.0))
    check_all_codecs(np.array(42, dtype=np.int64))
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


def test_inheritance():
    pointwise_config = dict(kind="eb", type="abs", eb=1)
    stencil_config = dict(
        kind="qoi_eb_stencil",
        qoi="finite_difference(x, order=1, accuracy=1, type=1, axis=0, grid_spacing=1)",
        neighbourhood=[
            dict(
                axis=0,
                before=0,
                after=1,
                boundary="valid",
            )
        ],
        type="abs",
        eb=1,
    )

    for combinator in ["any", "all"]:
        safeguards = Safeguards(
            safeguards=[
                dict(
                    kind=combinator,
                    safeguards=[pointwise_config, pointwise_config],
                )
            ]
        )
        assert len(safeguards.safeguards) == 1
        assert isinstance(safeguards.safeguards[0], PointwiseSafeguard)
        assert not isinstance(safeguards.safeguards[0], StencilSafeguard)

        safeguards = Safeguards(
            safeguards=[
                dict(
                    kind=combinator,
                    safeguards=[stencil_config, stencil_config],
                ),
            ]
        )
        assert len(safeguards.safeguards) == 1
        assert not isinstance(safeguards.safeguards[0], PointwiseSafeguard)
        assert isinstance(safeguards.safeguards[0], StencilSafeguard)

        safeguards = Safeguards(
            safeguards=[
                dict(
                    kind=combinator,
                    safeguards=[
                        pointwise_config,
                        stencil_config,
                    ],
                ),
            ]
        )
        assert len(safeguards.safeguards) == 1
        assert not isinstance(safeguards.safeguards[0], PointwiseSafeguard)
        assert isinstance(safeguards.safeguards[0], StencilSafeguard)


def test_fuzzer_found_any_selector_shape():
    data = np.array(
        [
            [-1.849731611000141e095, np.nan, np.nan],
            [np.nan, np.nan, 5.915260930833873e-270],
            [5.686073566141173e-270, 5.686073566141173e-270, 5.686073566141173e-270],
        ],
        dtype=np.float64,
    )
    decoded = np.array(
        [
            [5.686073566141173e-270, 5.686073566141173e-270, 5.686073566141173e-270],
            [5.686073566141173e-270, 5.686073566141173e-270, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(kind="sign", offset=43),
            dict(
                kind="any",
                safeguards=[
                    dict(
                        kind="monotonicity",
                        monotonicity="weak",
                        window=1,
                        boundary="valid",
                        axis=None,
                    ),
                ],
            ),
        ],
    )


def test_all_stencil_requirements():
    data = np.array([256, 256], dtype=np.int16)
    decoded = np.array([0, 0], dtype=np.int16)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="all",
                safeguards=[
                    dict(kind="assume_safe"),
                    dict(
                        kind="qoi_eb_stencil",
                        qoi="sum(X)",
                        neighbourhood=[
                            dict(
                                axis=0,
                                before=1,
                                after=1,
                                boundary="constant",
                                constant_boundary="$x_max",
                            ),
                        ],
                        type="abs",
                        eb=1,
                    ),
                ],
            ),
        ],
    )


def test_any_unsafely_shadowed_stencil_requirements():
    data = np.array([256, 256], dtype=np.int16)
    decoded = np.array([0, 0], dtype=np.int16)

    # prior to https://github.com/juntyr/compression-safeguards/pull/48,
    #  the any safeguard would union safe intervals per-point and the
    #  stencil safety could thus be shadowed by neighbouring points
    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="any",
                safeguards=[
                    SometimesSafeguard(is_safe="is_safe"),
                    dict(
                        kind="qoi_eb_stencil",
                        qoi="sum(X)",
                        neighbourhood=[
                            dict(
                                axis=0,
                                before=1,
                                after=1,
                                boundary="constant",
                                constant_boundary="$x_max",
                            ),
                        ],
                        type="abs",
                        eb=1,
                    ),
                ],
            ),
        ],
        fixed_constants=dict(is_safe=np.array([True, False])),
    )


# a safeguard that sometimes reports always safe and sometimes always unsafe
# this is just for testing since we require that a safeguard must always have a
#  non-empty safe interval
class SometimesSafeguard(PointwiseSafeguard):
    __slots__: tuple[str, ...] = ("_is_safe",)
    _is_safe: Parameter

    kind: ClassVar[str] = "sometimes_safe"

    def __init__(self, *, is_safe: str | Parameter) -> None:
        self._is_safe = (
            is_safe if isinstance(is_safe, Parameter) else Parameter(is_safe)
        )

    @property
    @override
    def late_bound(self) -> Set[Parameter]:
        return frozenset([self._is_safe])

    @override
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        is_safe = late_bound.resolve_ndarray_with_lossless_cast(
            self._is_safe, data.shape, np.dtype(np.intp)
        )

        return (is_safe != 0) | where

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        is_safe = late_bound.resolve_ndarray_with_lossless_cast(
            self._is_safe, data.shape, np.dtype(np.intp)
        )

        return (
            Interval.empty_like(data)
            .preserve_only_where(is_safe == 0)
            .preserve_only_where(True if where is True else where.flatten())
            .into_union()
        )

    @override
    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        is_safe = late_bound.resolve_ndarray_with_lossless_cast(
            self._is_safe, foot.shape, np.dtype(np.intp)
        )

        return (is_safe == 0) & where

    @override
    def get_config(self) -> dict[str, JSON]:
        return dict(kind=type(self).kind, is_safe=str(self._is_safe))

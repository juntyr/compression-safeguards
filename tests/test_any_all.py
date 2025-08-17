import numpy as np

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.pointwise.abc import PointwiseSafeguard
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard

from .codecs import (
    encode_decode_identity,
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

import numpy as np
import pytest

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.combinators.select import SelectSafeguard
from compression_safeguards.safeguards.pointwise.abc import PointwiseSafeguard
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard
from compression_safeguards.utils.binding import Bindings

from .codecs import encode_decode_mock


def test_select():
    safeguard = SelectSafeguard(
        selector="s",
        safeguards=[
            dict(kind="abs", eb_abs=100),
            dict(kind="abs", eb_abs=10),
            dict(kind="abs", eb_abs=1),
        ],
    )

    data = np.arange(10).reshape(2, 5)

    late_bound = Bindings(
        s=np.array([0, 1, 2, 0, 1, 2, 2, 1, 0, 1]).reshape(2, 5),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(
        valid._lower
        == (data.flatten() - np.array([100, 10, 1, 100, 10, 1, 1, 10, 100, 10]))
    )
    assert np.all(
        valid._upper
        == (data.flatten() + np.array([100, 10, 1, 100, 10, 1, 1, 10, 100, 10]))
    )

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(
        ok
        == np.array(
            [True, True, False, True, True, False, False, False, True, False]
        ).reshape(2, 5)
    )


def test_inheritance():
    pointwise_config = dict(kind="abs", eb_abs=1)
    stencil_config = dict(
        kind="qoi_abs_stencil",
        qoi="findiff(x, order=1, accuracy=1, type=1, dx=1, axis=0)",
        neighbourhood=[
            dict(
                axis=0,
                before=0,
                after=1,
                boundary="valid",
            )
        ],
        eb_abs=1,
    )

    safeguards = Safeguards(
        safeguards=[
            dict(
                kind="select",
                selector="s",
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
                kind="select",
                selector="s",
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
                kind="select",
                selector="s",
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


def test_parameter_validation():
    with pytest.raises(
        AssertionError,
        match=r"parameter `` must be a valid identifier",
    ):
        Safeguards(
            safeguards=[
                dict(
                    kind="select",
                    selector="",
                    safeguards=[dict(kind="zero"), dict(kind="sign")],
                )
            ]
        )


def test_numcodecs_validation():
    data = np.array([], dtype=np.uint8)
    decoded = np.array([], dtype=np.uint8)

    with pytest.raises(
        AssertionError,
        match=r"SafeguardsCodec does not \(yet\) support late-bound parameters",
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="select",
                    selector="param",
                    safeguards=[
                        dict(kind="qoi_abs_pw", qoi="(-(-x))", eb_abs=0),
                        dict(kind="zero"),
                    ],
                ),
            ],
        )

    for combinator in ["any", "all"]:
        with pytest.raises(
            AssertionError,
            match=r"SafeguardsCodec does not \(yet\) support late-bound parameters",
        ):
            encode_decode_mock(
                data,
                decoded,
                safeguards=[
                    dict(
                        kind=combinator,
                        safeguards=[
                            dict(
                                kind="select",
                                selector="param",
                                safeguards=[
                                    dict(kind="qoi_abs_pw", qoi="(-(-x))", eb_abs=0),
                                    dict(kind="zero"),
                                ],
                            ),
                            dict(kind="sign"),
                        ],
                    ),
                ],
            )

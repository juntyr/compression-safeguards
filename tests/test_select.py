import numpy as np
import pytest

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.combinators.select import SelectSafeguard
from compression_safeguards.safeguards.pointwise.abc import PointwiseSafeguard
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.error import LateBoundParameterResolutionError

from .codecs import encode_decode_mock


def test_select():
    safeguard = SelectSafeguard(
        selector="s",
        safeguards=[
            dict(kind="eb", type="abs", eb=100),
            dict(kind="eb", type="abs", eb=10),
            dict(kind="eb", type="abs", eb=1),
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


def test_select_literal():
    safeguard = SelectSafeguard(
        selector=1,
        safeguards=[
            dict(kind="eb", type="abs", eb=100),
            dict(kind="eb", type="abs", eb=10),
            dict(kind="eb", type="abs", eb=1),
        ],
    )

    data = np.arange(10).reshape(2, 5)

    valid = safeguard.compute_safe_intervals(data, late_bound=Bindings.EMPTY)
    assert np.all(valid._lower == (data.flatten() - 10))
    assert np.all(valid._upper == (data.flatten() + 10))

    ok = safeguard.check_pointwise(data, -data, late_bound=Bindings.EMPTY)
    assert np.all(
        ok
        == np.array(
            [True, True, True, True, True, True, False, False, False, False]
        ).reshape(2, 5)
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
        ValueError,
        match=r"safeguards\[0\]\.select\.selector: parameter `` is not a valid identifier",
    ):
        Safeguards(
            safeguards=[
                dict(
                    kind="select",
                    selector="",
                    safeguards=[dict(kind="same", value=0), dict(kind="sign")],
                )
            ]
        )


def test_numcodecs_validation():
    data = np.array([], dtype=np.uint8)
    decoded = np.array([], dtype=np.uint8)

    with pytest.raises(
        LateBoundParameterResolutionError,
        match="missing late-bound parameter `param`",
    ):
        encode_decode_mock(
            data,
            decoded,
            safeguards=[
                dict(
                    kind="select",
                    selector="param",
                    safeguards=[
                        dict(kind="qoi_eb_pw", qoi="(-(-x))", type="abs", eb=0),
                        dict(kind="same", value=0),
                    ],
                ),
            ],
        )
    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="select",
                selector="param",
                safeguards=[
                    dict(kind="qoi_eb_pw", qoi="(-(-x))", type="abs", eb=0),
                    dict(kind="same", value=0),
                ],
            ),
        ],
        fixed_constants=dict(param=0),
    )

    for combinator in ["any", "all"]:
        with pytest.raises(
            LateBoundParameterResolutionError,
            match="missing late-bound parameter `param`",
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
                                    dict(
                                        kind="qoi_eb_pw",
                                        qoi="(-(-x))",
                                        type="abs",
                                        eb=0,
                                    ),
                                    dict(kind="same", value=0),
                                ],
                            ),
                            dict(kind="sign"),
                        ],
                    ),
                ],
            )

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
                                dict(
                                    kind="qoi_eb_pw",
                                    qoi="(-(-x))",
                                    type="abs",
                                    eb=0,
                                ),
                                dict(kind="same", value=0),
                            ],
                        ),
                        dict(kind="sign"),
                    ],
                ),
            ],
            fixed_constants=dict(param=-1),
        )


def test_fuzzer_found_select_shape_mismatch():
    data = np.array([[256], [0]], dtype=np.int16)
    decoded = np.array([[0], [0]], dtype=np.int16)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            dict(
                kind="select",
                selector="$x_min",
                safeguards=[
                    dict(kind="same", value="$x_min", exclusive=True),
                ],
            ),
        ],
    )

import numpy as np

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.combinators.select import SelectSafeguard
from compression_safeguards.safeguards.pointwise.abc import PointwiseSafeguard
from compression_safeguards.safeguards.stencil.abc import StencilSafeguard


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
    selection = np.array([0, 1, 2, 0, 1, 2, 2, 1, 0, 1]).reshape(2, 5)

    valid = safeguard.compute_safe_intervals(data, late_bound=dict(s=selection))
    assert np.all(
        valid._lower
        == (data.flatten() - np.array([100, 10, 1, 100, 10, 1, 1, 10, 100, 10]))
    )
    assert np.all(
        valid._upper
        == (data.flatten() + np.array([100, 10, 1, 100, 10, 1, 1, 10, 100, 10]))
    )

    ok = safeguard.check_pointwise(data, -data, late_bound=dict(s=selection))
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

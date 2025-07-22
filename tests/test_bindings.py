import json

import numcodecs.registry
import numpy as np
import pytest
from numcodecs_safeguards import SafeguardsCodec

from compression_safeguards import Safeguards
from compression_safeguards.utils.bindings import Bindings


def test_missing_extraneous_bindings():
    data = np.array([1.0, 2.0, 3.0])
    prediction = np.zeros(3)

    safeguards = Safeguards(
        safeguards=[
            dict(kind="qoi_eb_pw", qoi='x * c["zero"] * c["$x"]', type="abs", eb="eb")
        ]
    )

    with pytest.raises(AssertionError, match=r"missing bindings.+eb.+,.+zero"):
        safeguards.compute_correction(data, prediction, late_bound=Bindings.empty())

    with pytest.raises(AssertionError, match=r"extraneous bindings.+\$x"):
        safeguards.compute_correction(
            data, prediction, late_bound=Bindings(eb=0.1, zero=0, **{"$x": prediction})
        )

    safeguards.compute_correction(data, prediction, late_bound=Bindings(eb=0.1, zero=0))


def test_numcodecs_missing_extraneous_bindings():
    data = np.array([1.0, 2.0, 3.0])

    codec = SafeguardsCodec(
        codec=dict(id="zero"),
        safeguards=[
            dict(
                kind="qoi_eb_pw",
                qoi='x * c["$x"] / (c["$x_max"] - c["$x_min"])',
                type="abs",
                eb=0.1,
            )
        ],
    )

    codec.decode(codec.encode(data))


@pytest.mark.parametrize(
    "eb",
    [
        1,
        1.0,
        np.float32(1.0),
        np.float64(1.0),
        np.linspace(0.0, 1.0),
        np.array([[1], [2]]),
    ],
)
def test_numcodecs_config(eb):
    codec = SafeguardsCodec(
        codec=dict(id="zero"),
        safeguards=[
            dict(
                kind="eb",
                type="abs",
                eb="eb",
            )
        ],
        fixed_constants=dict(eb=eb),
    )

    config = json.loads(json.dumps(codec.get_config()))

    assert type(config["fixed_constants"]["eb"]) in (int, float, str)

    codec2 = numcodecs.registry.get_codec(config)

    assert isinstance(codec2, SafeguardsCodec)
    assert codec2.late_bound == codec.late_bound
    np.testing.assert_array_equal(codec2._late_bound._bindings["eb"], eb)

    assert codec2.get_config() == codec.get_config()

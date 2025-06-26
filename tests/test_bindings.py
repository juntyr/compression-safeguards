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

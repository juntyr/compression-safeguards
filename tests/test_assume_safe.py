import numpy as np
import pytest

from compression_safeguards.api import Safeguards
from compression_safeguards.safeguards.combinators.assume_safe import (
    AssumeAlwaysSafeguard,
)
from compression_safeguards.utils.bindings import Bindings
from compression_safeguards.utils.cast import as_bits
from compression_safeguards.utils.intervals import _maximum, _minimum


@pytest.mark.parametrize("dtype", sorted(d.name for d in Safeguards.supported_dtypes()))
def test_dtypes(dtype):
    dtype = np.dtype(dtype)

    safeguard = AssumeAlwaysSafeguard()

    intervals = safeguard.compute_safe_intervals(
        np.zeros(1, dtype=dtype), late_bound=Bindings.empty()
    )

    np.testing.assert_equal(
        as_bits(intervals._lower),
        as_bits(np.array([[_minimum(dtype)]])),
    )
    np.testing.assert_equal(
        as_bits(intervals._upper),
        as_bits(np.array([[_maximum(dtype)]])),
    )

    assert safeguard.check(
        np.array([24], dtype=dtype),
        np.array([42], dtype=dtype),
        late_bound=Bindings.empty(),
    )

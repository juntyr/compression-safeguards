from itertools import product

import numpy as np
import pytest

from compression_safeguards.utils._compat import (
    _ceil_modulo,
    _floor_modulo,
    _round_ties_even_modulo,
    _trunc_modulo,
)

VALS = [-np.nan, -np.inf, -1.0, -0.5, -0.0, +0.0, +0.5, +1.0, +np.inf, +np.nan]


@pytest.mark.parametrize("p,q", product(VALS, VALS))
def test_floor_modulo(p, q):
    r = _floor_modulo(p, q)

    if np.isnan(p) or np.isnan(q):
        assert np.isnan(r)
        assert np.signbit(r) == np.signbit(q)
        return

    if q == 0:
        assert np.isnan(r)
        assert np.signbit(r) == np.signbit(q)
        return

    if np.isinf(p):
        assert np.isnan(r)
        assert np.signbit(r) == np.signbit(q)
        return

    if p == 0:
        assert r == 0
        assert np.signbit(r) == np.signbit(q)
        return

    if np.isinf(q):
        if np.signbit(p) == np.signbit(q):
            assert r == p
        else:
            assert r == q
        return

    if q < 0:
        assert r <= 0
        assert np.signbit(r) == np.signbit(-1)
        assert r > q
    else:
        assert r >= 0
        assert np.signbit(r) == np.signbit(+1)
        assert r < q


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
@pytest.mark.parametrize("p,q", product(VALS, VALS))
def test_ceil_modulo(p, q):
    r = _ceil_modulo(p, q)

    if np.isnan(p) or np.isnan(q):
        assert np.isnan(r)
        assert np.signbit(r) != np.signbit(q)
        return

    if q == 0:
        assert np.isnan(r)
        assert np.signbit(r) != np.signbit(q)
        return

    if np.isinf(p):
        assert np.isnan(r)
        assert np.signbit(r) != np.signbit(q)
        return

    if p == 0:
        assert r == 0
        assert np.signbit(r) != np.signbit(q)
        return

    if np.isinf(q):
        if np.signbit(p) == np.signbit(q):
            assert r == -q
        else:
            assert r == p
        return

    if q < 0:
        assert r >= 0
        assert np.signbit(r) == np.signbit(+1)
        assert r < -q
    else:
        assert r <= 0
        assert np.signbit(r) == np.signbit(-1)
        assert r > -q


@pytest.mark.parametrize("p,q", product(VALS, VALS))
def test_trunc_modulo(p, q):
    r = _trunc_modulo(p, q)

    if np.isnan(p) or np.isnan(q):
        assert np.isnan(r)
        assert np.signbit(r) == np.signbit(p)
        return

    if q == 0:
        assert np.isnan(r)
        assert np.signbit(r) == np.signbit(p)
        return

    if np.isinf(p):
        assert np.isnan(r)
        assert np.signbit(r) == np.signbit(p)
        return

    if p == 0:
        assert r == 0
        assert np.signbit(r) == np.signbit(p)
        return

    if np.isinf(q):
        assert r == p
        return

    if p < 0:
        assert r <= 0
        assert np.signbit(r) == np.signbit(-1)
        assert r > np.copysign(q, -1)
    else:
        assert r >= 0
        assert np.signbit(r) == np.signbit(+1)
        assert r < np.copysign(q, +1)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
@pytest.mark.parametrize("p,q", product(VALS, VALS))
def test_round_ties_even_modulo(p, q):
    r = _round_ties_even_modulo(p, q)

    if np.isnan(p) or np.isnan(q):
        assert np.isnan(r)  # any NaN
        return

    if q == 0:
        assert np.isnan(r)  # any NaN
        return

    if np.isinf(p):
        assert np.isnan(r)  # any NaN
        return

    if p == 0:
        assert r == 0
        assert np.signbit(r) == np.signbit(p)
        return

    if np.isinf(q):
        assert r == p
        return

    assert r >= np.copysign(q / 2, -1)
    assert r <= np.copysign(q / 2, +1)

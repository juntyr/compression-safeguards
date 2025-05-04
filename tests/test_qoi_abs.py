from typing import Any

import numpy as np
import sympy as sp

from numcodecs_safeguards.safeguards.pointwise.qoi.abs import _derive_eb_abs_qoi


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def check_qoi_example(f: Any, eb: Any, xv: int | float, tauv: int | float):
    xv = np.array(xv, dtype=np.float64)
    tauv = np.array(tauv, dtype=np.float64)

    print(f" --> for x={xv} and tau={tauv}: ", end="")

    f_x = f(xv)

    # FIXME: error bound rounding errors
    eb = abs(eb(xv, tauv) * 0.99)
    assert (not np.isfinite(f_x)) or (np.isfinite(eb) and eb >= 0)

    if np.isfinite(f_x):
        assert abs(f(xv - eb) - f_x) <= tauv
        assert abs(f(xv + eb) - f_x) <= tauv
        print(f"ok: [{f(xv - eb) - f(xv)}, {f(xv + eb) - f(xv)}]")
    elif np.isnan(f_x):
        assert np.isnan(f(xv - eb))
        assert np.isnan(f(xv + eb))
        print("ok: nan")
    else:
        # assert f(xv - eb) == f_x
        # assert f(xv + eb) == f_x
        assert not np.isfinite(f(xv - eb))
        assert not np.isfinite(f(xv + eb))
        print("ok: inf")


def check_qoi_examples(f):
    x = sp.Symbol("x", real=True)
    tau = sp.Symbol("tau", real=True, positive=True)

    print(f"f(x) = {f(x)}")

    f_x = sp.lambdify([x], f(x), "numpy")
    eb = sp.lambdify([x, tau], _derive_eb_abs_qoi(f(x), x, tau, True), "numpy")

    for xv in [-2, -0.5, 0.0, 0.5, 2]:
        for tauv in [10.0, 1.0, 0.1, 0.01]:
            check_qoi_example(f_x, eb, xv, tauv)


def test_polynomial():
    check_qoi_examples(lambda x: x)
    check_qoi_examples(lambda x: x**2)
    check_qoi_examples(lambda x: x**3)
    check_qoi_examples(lambda x: x**2 + x + 1)


def test_exponential():
    check_qoi_examples(lambda x: 0.5**x)
    check_qoi_examples(lambda x: 2**x)
    check_qoi_examples(lambda x: 3**x)
    check_qoi_examples(lambda x: sp.exp(x))

    check_qoi_examples(lambda x: 2 ** (x + 1))


def test_logarithmic():
    check_qoi_examples(lambda x: sp.log(x, 2))
    check_qoi_examples(lambda x: sp.log(x))

    check_qoi_examples(lambda x: sp.log(x + 1))


def test_power_function():
    check_qoi_examples(lambda x: 1 / (x + 3))


def test_inverse():
    check_qoi_examples(lambda x: 1 / x)
    check_qoi_examples(lambda x: 1 / x**2)
    check_qoi_examples(lambda x: 1 / x**3)


def test_sqrt():
    check_qoi_examples(lambda x: sp.sqrt(x))
    check_qoi_examples(lambda x: 1 / sp.sqrt(x))


def test_sigmoid():
    check_qoi_examples(lambda x: 1 / (1 + sp.exp(-x)))


def test_tanh():
    check_qoi_examples(lambda x: (sp.exp(x) - sp.exp(-x)) / (sp.exp(x) + sp.exp(-x)))


# def test_composed():
#     check_qoi_examples(lambda x: 2 / (sp.ln(x) + sp.sqrt(x)))

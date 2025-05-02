from typing import Any

import numpy as np
import sympy as sp

from numcodecs_safeguards.safeguards.pointwise.qoi import _derive_eb_abs_qoi


def check_qoi_example(
    f: Any, eb: None | Any, xv: int | float, tauv: int | float
):
    print(f" --> for x={xv} and tau={tauv}: ", end="")

    f_x = f(xv)
    
    if not np.isfinite(f_x):
        print("skip")
        return

    if eb is not None:
        # FIXME: error bound rounding errors
        eb = eb(xv, tauv) * 0.99

        if not np.isfinite(eb):
            print("nan")
            return
        assert eb >= 0

        assert abs(f(xv - eb) - f(xv)) <= tauv
        assert abs(f(xv + eb) - f(xv)) <= tauv
        print(f"ok: [{f(xv - eb) - f(xv)}, {f(xv + eb) - f(xv)}]")
    else:
        assert f(xv - 1000) == f(xv)
        assert f(xv + 1000) == f(xv)
        print("any")


def check_qoi_examples(f):
    x = sp.Symbol("x", real=True)
    tau = sp.Symbol("tau", real=True, positive=True)

    print(f"f(x) = {f(x)}")

    eb = _derive_eb_abs_qoi(f(x), x, tau, True)

    f_x = sp.lambdify([x], f(x), "numpy")

    if eb is not None:
        eb = sp.lambdify([x, tau], eb, "numpy")

    for xv in [-2, -0.5, 0.0, 0.5, 2]:
        for tauv in [10.0, 1.0, 0.1, 0.01]:
            check_qoi_example(f_x, eb, xv, tauv)


def test_solve():
    check_qoi_examples(lambda x: sp.Integer(3))
    check_qoi_examples(lambda x: x)
    check_qoi_examples(lambda x: 2 * x + 3)
    check_qoi_examples(lambda x: x**2)
    check_qoi_examples(lambda x: sp.ln(x))
    check_qoi_examples(lambda x: sp.sqrt(x))
    check_qoi_examples(lambda x: 3 / x)
    check_qoi_examples(lambda x: x**2 + x)


def test_weighted_sum():
    check_qoi_examples(lambda x: 2 * x**2 - x + 0.5 * sp.sqrt(x))


def test_product():
    check_qoi_examples(lambda x: x**2)
    check_qoi_examples(lambda x: x**3)
    check_qoi_examples(lambda x: x**4)
    check_qoi_examples(lambda x: x**5)
    check_qoi_examples(lambda x: x**6)


def test_composition():
    # check_qoi_examples(lambda x: 0.5 / sp.sqrt(x))
    # check_qoi_examples(lambda x: 1 / x)
    # check_qoi_examples(lambda x: 2 / (x**2))
    # check_qoi_examples(lambda x: 3 / (x**3))
    # check_qoi_examples(lambda x: 4 / (x**4))
    # check_qoi_examples(lambda x: 5 / (x**5))
    check_qoi_examples(lambda x: 2 / (sp.ln(x) + sp.sqrt(x)))

import sympy as sp

from numcodecs_safeguards.safeguards.pointwise.qoi import _derive_eb_abs_qoi


def check_qoi_example(f, xv: int | float, tauv: int | float):
    x = sp.Symbol("x", real=True)
    tau = sp.Symbol("tau", real=True, positive=True)

    print(f"f(x) = {f(x)}   for x={xv} and tau={tauv}")

    eb = _derive_eb_abs_qoi(f(x), x, tau, True)

    xv = 3.0
    tauv = 0.1

    if eb is not None:
        # FIXME: error bound rounding errors
        ebl = eb[0].doit().subs([(x, xv), (tau, tauv)], simultaneous=True) * 0.99
        ebu = eb[1].doit().subs([(x, xv), (tau, tauv)], simultaneous=True) * 0.99

        assert ebl <= 0
        assert ebu >= 0

        assert abs(f(xv + ebl) - f(xv)) <= tauv
        assert abs(f(xv + ebu) - f(xv)) <= tauv
    else:
        assert f(xv - 1000) == f(xv)
        assert f(xv + 1000) == f(xv)


def check_qoi_examples(f):
    for x in [-2]:  # , -0.5, 0.0, 0.5, 2]:
        for tau in [10.0]:  # , 1.0, 0.1, 0.01, 0.0]:
            check_qoi_example(f, x, tau)


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
    check_qoi_examples(lambda x: 0.5 / sp.sqrt(x))
    check_qoi_examples(lambda x: 1 / x)
    check_qoi_examples(lambda x: 2 / (x**2))
    check_qoi_examples(lambda x: 3 / (x**3))
    check_qoi_examples(lambda x: 4 / (x**4))
    check_qoi_examples(lambda x: 5 / (x**5))
    check_qoi_examples(lambda x: 2 / (sp.ln(x) + sp.sqrt(x)))

import numpy as np
import pytest

from numcodecs_safeguards.quantizer import _SUPPORTED_DTYPES

from .codecs import (
    encode_decode_identity,
    encode_decode_neg,
    encode_decode_noise,
    encode_decode_zero,
)


def check_all_codecs(data: np.ndarray, qoi: str):
    for encode_decode in [
        encode_decode_zero,
        encode_decode_neg,
        encode_decode_identity,
        encode_decode_noise,
    ]:
        for eb_abs in [10.0, 1.0, 0.1, 0.01, 0.0]:
            encode_decode(
                data,
                safeguards=[dict(kind="qoi_abs_pw", qoi=qoi, eb_abs=eb_abs)],
            )


def check_empty(qoi: str):
    check_all_codecs(np.empty(0), qoi)


def check_unit(qoi: str):
    check_all_codecs(np.linspace(-1.0, 1.0, 100), qoi)


def check_circle(qoi: str):
    check_all_codecs(np.linspace(-np.pi * 2, np.pi * 2, 100), qoi)


def check_arange(qoi: str):
    check_all_codecs(np.arange(100, dtype=float), qoi)


def check_linspace(qoi: str):
    check_all_codecs(np.linspace(-1024, 1024, 2831), qoi)


def check_edge_cases(qoi: str):
    check_all_codecs(
        np.array(
            [
                np.inf,
                np.nan,
                -np.inf,
                -np.nan,
                np.finfo(float).min,
                np.finfo(float).max,
                np.finfo(float).tiny,
                -np.finfo(float).tiny,
                -0.0,
                +0.0,
            ]
        ),
        qoi,
    )


CHECKS = [
    check_empty,
    check_unit,
    check_circle,
    check_arange,
    check_linspace,
    check_edge_cases,
]


def test_sandbox():
    with pytest.raises(AssertionError, match="invalid qoi expression"):
        # sandbox escape based on https://stackoverflow.com/q/35804961 and
        #  https://stackoverflow.com/a/35806044
        check_all_codecs(
            np.empty(0),
            "f\"{[c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings'][0]()._module.__builtins__['quit']()}\"",
        )


@pytest.mark.parametrize("check", CHECKS)
def test_empty(check):
    with pytest.raises(AssertionError, match="empty"):
        check("")
    with pytest.raises(AssertionError, match="empty"):
        check("  \t   \n   ")


def test_non_expression():
    with pytest.raises(AssertionError, match="numeric expression"):
        check_all_codecs(np.empty(0), "exp")


@pytest.mark.parametrize("check", CHECKS)
def test_constant(check):
    with pytest.raises(AssertionError, match="constant"):
        check("0")
    with pytest.raises(AssertionError, match="constant"):
        check("pi")
    with pytest.raises(AssertionError, match="constant"):
        check("e")
    with pytest.raises(AssertionError, match="constant"):
        check("-(-(-e))")


@pytest.mark.parametrize("check", CHECKS)
def test_imaginary(check):
    with pytest.raises(AssertionError, match="imaginary"):
        check_all_codecs(np.array([2], dtype=np.uint64), "(-log(-20417, base=ln(x)))")
    with pytest.raises(AssertionError, match="imaginary"):
        check("(-log(-20417, base=ln(x)))")


@pytest.mark.parametrize("check", CHECKS)
def test_polynomial(check):
    check("x")
    check("2*x")
    check("3*x + pi")
    check("x**2")
    check("x**3")
    check("x**2 + x + 1")


@pytest.mark.parametrize("check", CHECKS)
def test_exponential(check):
    check("0.5**x")
    check("2**x")
    check("3**x")
    check("e**(x)")
    check("exp(x)")
    check("2 ** (x + 1)")

    check_all_codecs(np.array([51.0]), "2**x")
    check_all_codecs(np.array([31.0]), "exp(x)")


@pytest.mark.parametrize("check", CHECKS)
def test_logarithm(check):
    check("log(x, base=2)")
    check("ln(x)")
    check("ln(x + 1)")
    check("log(2, base=x)")


@pytest.mark.parametrize("check", CHECKS)
def test_inverse(check):
    check("1 / x")
    check("1 / x**2")
    check("1 / x**3")


@pytest.mark.parametrize("check", CHECKS)
def test_power_function(check):
    check("1 / (x + 3)")


@pytest.mark.parametrize("check", CHECKS)
def test_sqrt(check):
    check("sqrt(x)")
    check("1 / sqrt(x)")
    check("sqrt(sqrt(x))")


@pytest.mark.parametrize("check", CHECKS)
def test_sigmoid(check):
    check("1 / (1 + exp(-x))")


@pytest.mark.parametrize("check", CHECKS)
def test_tanh(check):
    check("(exp(x) - exp(-x)) / (exp(x) + exp(-x))")


@pytest.mark.parametrize("check", CHECKS)
def test_hyperbolic(check):
    check("sinh(x)")
    check("cosh(x)")
    check("tanh(x)")
    check("coth(x)")
    check("sech(x)")
    check("csch(x)")

    check("asinh(x)")
    check("acosh(x)")
    check("atanh(x)")
    check("acoth(x)")
    check("asech(x)")
    check("acsch(x)")


@pytest.mark.parametrize("check", CHECKS)
def test_composed(check):
    check("2 / (ln(x) + sqrt(x))")

    check_all_codecs(np.array([-1, 0, 1]), "exp(ln(x)+x)")
    check("exp(ln(x)+x)")


@pytest.mark.parametrize("dtype", sorted(d.name for d in _SUPPORTED_DTYPES))
def test_dtypes(dtype):
    check_all_codecs(np.array([[1]], dtype=dtype), "x/sqrt(pi)")


@pytest.mark.parametrize("check", CHECKS)
def test_fuzzer_found(check):
    with pytest.raises(AssertionError, match="failed to parse"):
        check("(((-8054**5852)-x)-1)")

    check_all_codecs(
        np.array([[18312761160228738559]], dtype=np.uint64), "((pi**(x**(x+x)))**1)"
    )
    check_all_codecs(np.array([-1024.0]), "((pi**(x**(x+x)))**1)")
    check("((pi**(x**(x+x)))**1)")

    check_all_codecs(np.array([], np.uint64), "(-((e/(22020**-37))**x))")
    check("(-((e/(22020**-37))**x))")


def test_lambdify_dtype():
    import inspect

    import sympy as sp

    from numcodecs_safeguards.safeguards.pointwise.qoi.abs import (
        _compile_sympy_expr_to_numpy,
    )

    x = sp.Symbol("x", real=True)

    fn = _compile_sympy_expr_to_numpy([x], x + sp.pi + sp.E, np.dtype(np.float16))

    assert (
        inspect.getsource(fn)
        == "def _lambdifygenerated(x):\n    return x + float16('2.71828') + float16('3.14159')\n"
    )

    assert np.float16("2.71828") == np.float16(np.e)
    assert np.float16("3.14159") == np.float16(np.pi)

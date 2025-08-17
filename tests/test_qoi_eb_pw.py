import numpy as np
import pytest

from compression_safeguards import Safeguards
from compression_safeguards.safeguards.pointwise.qoi.eb import (
    PointwiseQuantityOfInterestErrorBoundSafeguard,
)
from compression_safeguards.utils.bindings import Bindings

from .codecs import (
    encode_decode_identity,
    encode_decode_mock,
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
        for type, eb in [
            ("abs", 10.0),
            ("abs", 1.0),
            ("abs", 0.1),
            ("abs", 0.01),
            ("abs", 0.0),
            ("rel", 10.0),
            ("rel", 1.0),
            ("rel", 0.1),
            ("rel", 0.01),
            ("rel", 0.0),
            ("ratio", 10.0),
            ("ratio", 2.0),
            ("ratio", 1.1),
            ("ratio", 1.01),
            ("ratio", 1.0),
        ]:
            try:
                encode_decode(
                    data,
                    safeguards=[dict(kind="qoi_eb_pw", qoi=qoi, type=type, eb=eb)],
                )
            except Exception as err:
                print(encode_decode, qoi, type, eb)
                raise err


def check_empty(qoi: str):
    check_all_codecs(np.empty(0), qoi)


def check_dimensions(qoi: str):
    check_all_codecs(np.array(42.0), qoi)
    check_all_codecs(np.array(42, dtype=np.int64), qoi)
    check_all_codecs(np.array([42.0]), qoi)
    check_all_codecs(np.array([[42.0]]), qoi)
    check_all_codecs(np.array([[[42.0]]]), qoi)


def check_unit(qoi: str):
    check_all_codecs(np.linspace(-1.0, 1.0, 100, dtype=np.float16), qoi)


def check_circle(qoi: str):
    check_all_codecs(np.linspace(-np.pi * 2, np.pi * 2, 100, dtype=np.int64), qoi)


def check_arange(qoi: str):
    check_all_codecs(np.arange(100, dtype=float), qoi)


def check_linspace(qoi: str):
    check_all_codecs(np.linspace(-1024, 1024, 2831, dtype=np.float32), qoi)


# FIXME: could we also test the float128 edge cases, somehow?
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
                np.finfo(float).smallest_normal,
                -np.finfo(float).smallest_normal,
                np.finfo(float).smallest_subnormal,
                -np.finfo(float).smallest_subnormal,
                0.0,
                -0.0,
            ]
        ),
        qoi,
    )


CHECKS = [
    check_empty,
    check_dimensions,
    check_unit,
    check_circle,
    check_arange,
    check_linspace,
    check_edge_cases,
]


def test_sandbox():
    with pytest.raises(
        AssertionError, match="failed to parse pointwise QoI expression"
    ):
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
    with pytest.raises(AssertionError, match="empty"):
        check(" # just a comment ")


def test_non_expression():
    with pytest.raises(AssertionError, match="EOF"):
        check_all_codecs(np.empty(0), "exp")
    with pytest.raises(AssertionError, match="unexpected token `x`"):
        check_all_codecs(np.empty(0), "e x p")


def test_whitespace():
    check_all_codecs(np.array([]), "  \n \t x   \t\n  ")
    check_all_codecs(np.array([]), "  \n \t x \t \n  - \t \n 3  \t\n  ")
    check_all_codecs(np.array([]), "x    -    3")
    check_all_codecs(np.array([]), "sqrt   \n (x)")
    check_all_codecs(np.array([]), "log ( x , base \t = \n 2 )")


def test_comment():
    check_all_codecs(np.array([]), "x # great variable")
    check_all_codecs(np.array([]), "# great variable\nx")
    check_all_codecs(np.array([]), "x # nothing 3+4 really matters 1/0")
    check_all_codecs(
        np.array([]), "log #1\n ( #2\n x #3\n , #4\n base #5\n = #6\n 2 #7\n )"
    )


def test_variables():
    with pytest.raises(
        AssertionError,
        match=r'cannot assign to identifier `a`, assign to a variable v\["a"\] instead',
    ):
        check_all_codecs(np.array([]), "a = 4; return a")
    with pytest.raises(
        AssertionError, match="pointwise QoI variables use lower-case `v`"
    ):
        check_all_codecs(np.array([]), 'V["a"]')
    with pytest.raises(
        AssertionError, match='invalid string literal with missing closing `"`'
    ):
        check_all_codecs(np.array([]), 'v["a]')
    with pytest.raises(AssertionError, match="invalid quoted parameter"):
        check_all_codecs(np.array([]), 'v["123"]')
    with pytest.raises(AssertionError, match="invalid quoted parameter"):
        check_all_codecs(np.array([]), 'v["a 123"]')
    with pytest.raises(
        AssertionError, match=r"variable name must not be built-in \(start with `\$`\)"
    ):
        check_all_codecs(np.array([]), 'v["$a"]')
    with pytest.raises(AssertionError, match=r'undefined variable v\["a"\]'):
        check_all_codecs(np.array([]), 'v["a"]')
    with pytest.raises(AssertionError, match=r'undefined variable v\["b"\]'):
        check_all_codecs(np.array([]), 'v["a"] = 3; return x + v["b"];')
    with pytest.raises(AssertionError, match="unexpected token `=`"):
        check_all_codecs(np.array([]), "1 = x")
    with pytest.raises(AssertionError, match=r"expected `\(`"):
        check_all_codecs(np.array([]), 'v["a"] = log; return x + v["a"];')
    with pytest.raises(
        AssertionError, match=r'cannot override already-defined variable v\["a"\]'
    ):
        check_all_codecs(
            np.array([]), 'v["a"] = x + 1; v["a"] = v["a"]; return v["a"];'
        )
    check_all_codecs(np.array([]), 'v["a"] = 3; return x + v["a"];')
    check_all_codecs(
        np.array([]), 'v["a"] = 3; v["b"] = v["a"] - 1; return x + (v["b"] * 2);'
    )
    check_all_codecs(
        np.array([]), 'v["a"] = x + 1; v["b"] = x - 1; return v["a"] + v["b"];'
    )
    check_all_codecs(np.array([]), 'c["$x"] * x')


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
def test_negate(check):
    check("-x")
    check("--x")
    check("--(-x)")


@pytest.mark.parametrize("check", CHECKS)
def test_abs(check):
    check("abs(x)")
    check("abs(-x)")
    check("abs(abs(x))")


@pytest.mark.parametrize("check", CHECKS)
def test_polynomial(check):
    # identities
    check("x")
    check("x + 0")
    check("x * 1")

    # x * 0
    check("x * 0")

    # x + (inf / -inf, NaN)
    check("x + (1/0)")
    check("x + (-1/0)")
    check("x + (0/0)")

    # x * (inf / -inf, NaN)
    check("x * (1/0)")
    check("x * (-1/0)")
    check("x * (0/0)")

    # sum(x * (inf / -inf, NaN), 0)
    check("x * (1/0) + 0")
    check("x * (-1/0) + 0")
    check("x * (0/0) + 0")

    # some simple polynomials
    check("2*x")
    check("x + x")
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
    check("exp2(x)")
    check("exp10(x)")
    check("2 ** (x + 1)")

    check_all_codecs(np.array([51.0]), "2**x")
    check_all_codecs(np.array([31.0]), "exp(x)")
    check_all_codecs(np.array([42.0]), "exp2(x)")
    check_all_codecs(np.array([42.0]), "exp10(x)")


@pytest.mark.parametrize("check", CHECKS)
def test_logarithm(check):
    check("log(x, base=2)")
    check("ln(x)")
    check("ln(x + 1)")
    check("log2(x)")
    check("log2(x + 1)")
    check("log10(x)")
    check("log10(x + 1)")
    check("log(2, base=x)")


@pytest.mark.parametrize("check", CHECKS)
def test_sign(check):
    check("sign(x)")
    check("sign(x + 1)")
    check("sign(x * 2)")
    check("sign(e**x)")


@pytest.mark.parametrize("check", CHECKS)
def test_rounding(check):
    check("floor(x)")
    check("ceil(x)")
    check("trunc(x)")
    check("round_ties_even(x)")

    check("floor(x) * floor(1.5)")
    check("ceil(x) * ceil(1.5)")
    check("trunc(x) * trunc(1.5)")
    check("round_ties_even(x) * round_ties_even(1.5)")


@pytest.mark.parametrize("check", CHECKS)
def test_inverse(check):
    check("1 / x")
    check("1 / x**2")
    check("1 / x**3")

    check("reciprocal(x)")
    check("reciprocal(x - 1)")
    check("reciprocal(x**2)")
    check("reciprocal(x**3)")
    check("reciprocal(reciprocal(x))")


@pytest.mark.parametrize("check", CHECKS)
def test_power_function(check):
    check("1 / (x + 3)")


@pytest.mark.parametrize("check", CHECKS)
def test_sqrt(check):
    check("sqrt(x)")
    check("1 / sqrt(x)")
    check("reciprocal(sqrt(x))")
    check("sqrt(sqrt(x))")


@pytest.mark.parametrize("check", CHECKS)
def test_square(check):
    check("square(x)")
    check("1 / square(x)")
    check("square(square(x))")


@pytest.mark.parametrize("check", CHECKS)
def test_sigmoid(check):
    check("1 / (1 + exp(-x))")


@pytest.mark.parametrize("check", CHECKS)
def test_tanh(check):
    check("(exp(x) - exp(-x)) / (exp(x) + exp(-x))")


@pytest.mark.parametrize("check", CHECKS)
def test_trigonometric(check):
    check("sin(x)")
    check("cos(x)")
    check("tan(x)")
    check("cot(x)")
    check("sec(x)")
    check("csc(x)")

    check("asin(x)")
    check("acos(x)")
    check("atan(x)")
    check("acot(x)")
    check("asec(x)")
    check("acsc(x)")


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

    check("sinh(x) * sinh(1.5)")
    check("asinh(x) * asinh(1.5)")


@pytest.mark.parametrize("check", CHECKS)
def test_classification(check):
    check("isfinite(x)")
    check("isinf(x)")
    check("isnan(x)")


@pytest.mark.parametrize("check", CHECKS)
def test_where(check):
    check("where(x, 1, 0)")
    check("where(0, x*2, 0)")
    check("where(0, 1, x/2)")
    check("where(1, x*2, 0)")
    check("where(1, 1, x/2)")
    check("where(0, x*2, x/2)")
    check("where(1, x*2, x/2)")
    check("where(x, x*2, x/2)")


@pytest.mark.parametrize("check", CHECKS)
def test_composed(check):
    check("2 / (ln(x) + sqrt(x))")

    check_all_codecs(np.array([-1, 0, 1]), "exp(ln(x)+x)")
    check("exp(ln(x)+x)")


@pytest.mark.parametrize("dtype", sorted(d.name for d in Safeguards.supported_dtypes()))
def test_dtypes(dtype):
    check_all_codecs(np.array([[1]], dtype=dtype), "x/sqrt(pi)")


@pytest.mark.parametrize("check", CHECKS)
def test_fuzzer_found(check):
    with pytest.warns(UserWarning, match="symbolic integer evaluation"):
        check_all_codecs(np.array([42.0], np.float16), "(((-8054**5852)-x)-1)")
    with pytest.warns(UserWarning, match="symbolic integer evaluation"):
        check("(((-8054**5852)-x)-1)")

    check_all_codecs(
        np.array([[18312761160228738559]], dtype=np.uint64), "((pi**(x**(x+x)))**1)"
    )
    check_all_codecs(np.array([-1024.0]), "((pi**(x**(x+x)))**1)")
    check("((pi**(x**(x+x)))**1)")

    check_all_codecs(np.array([], np.uint64), "(-((e/(22020**-37))**x))")
    check("(-((e/(22020**-37))**x))")


def test_evaluate_expr_with_dtype():
    from compression_safeguards.safeguards._qois.expr.addsub import ScalarAdd
    from compression_safeguards.safeguards._qois.expr.data import Data
    from compression_safeguards.safeguards._qois.expr.literal import Euler, Pi

    expr = ScalarAdd(ScalarAdd(Data(()), Pi()), Euler())

    assert f"{expr!r}" == "x + pi + e"

    assert expr.eval((), np.float16("4.2"), dict()) == np.float16("4.2") + np.float16(
        np.e
    ) + np.float16(np.pi)


def test_late_bound_eb_abs():
    safeguard = PointwiseQuantityOfInterestErrorBoundSafeguard(
        qoi="x", type="abs", eb="eb"
    )
    assert safeguard.late_bound == {"eb"}

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(
        eb=np.array([5, 4, 3, 2, 1, 0]).reshape(2, 3),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(valid._lower == (data.flatten() - np.array([5, 4, 3, 2, 1, 0])))
    assert np.all(valid._upper == (data.flatten() + np.array([5, 4, 3, 2, 1, 0])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(
        ok == np.array([True, True, False, False, False, False]).reshape(2, 3)
    )


def test_late_bound_eb_rel():
    safeguard = PointwiseQuantityOfInterestErrorBoundSafeguard(
        qoi="x", type="rel", eb="eb"
    )
    assert safeguard.late_bound == {"eb"}

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(
        eb=np.array([5, 4, 3, 2, 1, 0]).reshape(2, 3),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(valid._lower == (data.flatten() - np.array([0, 4, 6, 6, 4, 0])))
    assert np.all(valid._upper == (data.flatten() + np.array([0, 4, 6, 6, 4, 0])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, True, False, False]).reshape(2, 3))


def test_late_bound_eb_ratio():
    safeguard = PointwiseQuantityOfInterestErrorBoundSafeguard(
        qoi="x", type="ratio", eb="eb"
    )
    assert safeguard.late_bound == {"eb"}

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(
        eb=np.array([6, 5, 4, 3, 2, 1]).reshape(2, 3),
    )

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(valid._lower == (data.flatten() - np.array([0, 0, 1, 2, 2, 0])))
    assert np.all(valid._upper == (data.flatten() + np.array([0, 4, 6, 6, 4, 0])))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(
        ok == np.array([True, False, False, False, False, False]).reshape(2, 3)
    )


def test_late_bound_constant():
    safeguard = PointwiseQuantityOfInterestErrorBoundSafeguard(
        qoi='x / c["f"]', type="abs", eb=1
    )
    assert safeguard.late_bound == {"f"}

    data = np.arange(6).reshape(2, 3)

    late_bound = Bindings(
        f=np.array([16, 8, 4, 2, 1, 0]).reshape(2, 3),
    )

    imax = np.iinfo(data.dtype).max

    valid = safeguard.compute_safe_intervals(data, late_bound=late_bound)
    assert np.all(valid._lower == np.array([0 - 16, 1 - 8, 2 - 4, 3 - 2, 4 - 1, 1]))
    assert np.all(valid._upper == np.array([0 + 16, 1 + 8, 2 + 4, 3 + 2, 4 + 1, imax]))

    ok = safeguard.check_pointwise(data, -data, late_bound=late_bound)
    assert np.all(ok == np.array([True, True, True, False, False, False]).reshape(2, 3))


@pytest.mark.parametrize("check", CHECKS)
def test_pointwise_normalised_absolute_error(check):
    # pointwise normalised / range-relative absolute error bound
    check('(x - c["$x_min"]) / (c["$x_max"] - c["$x_min"])')


@pytest.mark.parametrize("check", CHECKS)
def test_pointwise_histogram_index(check):
    check('round_ties_even(100 * (x - c["$x_min"]) / (c["$x_max"] - c["$x_min"]))')


def test_gaussian_kernel():
    safeguard = PointwiseQuantityOfInterestErrorBoundSafeguard(
        qoi="""
        v["sigma"] = 1.5;
        v["i"] = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
        v["k"] = 1/(sqrt(2*pi)*v["sigma"]) * exp((-square(v["i"])) / (2*square(v["sigma"])));
        v["kern"] = matmul([v["k"]].T, [v["k"]]);
        v["kernel"] = v["kern"] / sum(v["kern"]);
        return x * v["kernel"][5,5];
        """,
        type="abs",
        eb=1,
    )

    # alternative implementation for a 2D Gaussian kernel from
    #  https://stackoverflow.com/a/43346070
    i = np.arange(11).astype(np.float64) - 5
    k = np.exp(-0.5 * np.square(i) / np.square(1.5))
    kern = np.outer(k, k)
    kernel = kern / np.sum(kern)

    assert (
        safeguard.evaluate_qoi(
            np.array(1.0, dtype=np.float64), late_bound=Bindings.empty()
        )
        == kernel[5, 5]
    )


def test_constant_fold():
    from compression_safeguards.safeguards._qois.expr.data import Data
    from compression_safeguards.safeguards._qois.expr.literal import Number
    from compression_safeguards.safeguards._qois.expr.logexp import (
        ScalarLogWithBase,
    )

    expr = ScalarLogWithBase(
        Number.from_symbolic_int(100), Number.from_symbolic_int(10)
    )
    assert f"{expr!r}" == "log(100, base=10)"
    assert expr.eval((), np.empty(0, dtype=np.float64), {}) == 2

    assert expr.constant_fold(np.dtype(np.float64)) == 2

    expr = ScalarLogWithBase(Data(index=()), Number.from_symbolic_int(10))
    assert f"{expr!r}" == "log(x, base=10)"
    assert expr.eval((), np.array(100, dtype=np.float64), {}) == 2

    expr = expr.constant_fold(np.dtype(np.float64))
    assert f"{expr!r}" == f"ln(x) / ({np.log(np.float64(10))!r})"
    assert expr.eval((), np.array(100, dtype=np.float64), {}) == 2


def test_fuzzer_found_sign_constant_fold():
    data = np.array([], dtype=np.int64)
    decoded = np.array([], dtype=np.int64)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            PointwiseQuantityOfInterestErrorBoundSafeguard(
                qoi="sign(e) - x", type="rel", eb=2
            ),
        ],
    )


def test_fuzzer_found_logarithm_noise():
    from numcodecs_combinators.framed import FramedCodecStack
    from numcodecs_safeguards import SafeguardsCodec

    from .codecs import NoiseCodec

    data = np.linspace(-1.0, 1.0, 100, dtype=np.float16)

    codec = SafeguardsCodec(
        codec=FramedCodecStack(NoiseCodec(seed=258957826)),
        safeguards=[
            PointwiseQuantityOfInterestErrorBoundSafeguard(
                qoi="log2(x + 1)", type="ratio", eb=10
            ),
        ],
    )

    encoded = codec.encode(data)
    codec.decode(encoded)


def test_fuzzer_found_classification():
    data = np.array([], dtype=np.int16)
    decoded = np.array([], dtype=np.int16)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            PointwiseQuantityOfInterestErrorBoundSafeguard(
                qoi="isinf(log(exp2(exp10(isnan(exp10(e)))), base=x))",
                type="rel",
                eb=0,
            ),
        ],
    )

    data = np.array([], dtype=np.int64)
    decoded = np.array([], dtype=np.int64)

    encode_decode_mock(
        data,
        decoded,
        safeguards=[
            PointwiseQuantityOfInterestErrorBoundSafeguard(
                qoi="exp10(exp10(isinf(exp2(1.5298016088828333e+308))) - x)",
                type="rel",
                eb=0,
            ),
        ],
    )

import numpy as np
import pytest

from compression_safeguards.safeguards._qois.expr.abs import ScalarAbs
from compression_safeguards.safeguards._qois.expr.addsub import ScalarSubtract
from compression_safeguards.safeguards._qois.expr.classification import (
    ScalarIsFinite,
    ScalarIsInf,
    ScalarIsNaN,
)
from compression_safeguards.safeguards._qois.expr.constfold import ScalarFoldedConstant
from compression_safeguards.safeguards._qois.expr.data import Data
from compression_safeguards.safeguards._qois.expr.divmul import (
    ScalarDivide,
    ScalarMultiply,
)
from compression_safeguards.safeguards._qois.expr.hyperbolic import (
    ScalarAcosh,
    ScalarAsinh,
    ScalarTanh,
)
from compression_safeguards.safeguards._qois.expr.literal import Euler, Number, Pi
from compression_safeguards.safeguards._qois.expr.logexp import (
    Exponential,
    Logarithm,
    ScalarExp,
    ScalarLog,
    ScalarLogWithBase,
)
from compression_safeguards.safeguards._qois.expr.neg import ScalarNegate
from compression_safeguards.safeguards._qois.expr.power import ScalarPower
from compression_safeguards.safeguards._qois.expr.reciprocal import ScalarReciprocal
from compression_safeguards.safeguards._qois.expr.round import ScalarRoundTiesEven
from compression_safeguards.safeguards._qois.expr.sign import ScalarSign
from compression_safeguards.safeguards._qois.expr.square import ScalarSqrt, ScalarSquare
from compression_safeguards.safeguards._qois.expr.trigonometric import (
    ScalarAcos,
    ScalarAsin,
    ScalarAtan,
    ScalarCos,
    ScalarSin,
    ScalarTan,
)
from compression_safeguards.safeguards._qois.expr.where import ScalarWhere
from compression_safeguards.safeguards._qois.interval import (
    compute_safe_data_lower_upper_interval_union,
)
from compression_safeguards.utils._compat import _is_negative_zero, _is_positive_zero
from compression_safeguards.utils._float128 import _float128, _float128_dtype

np.set_printoptions(floatmode="unique")


def test_abs():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarAbs(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.abs(X) - 1,
            np.abs(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -43.0,
                    -3.0,
                    -2.0,
                    -1.5,
                    -1.0,
                    -1.0,
                    -1.5,
                    -2.0,
                    1.0,
                    41.0,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -41.0,
                    -1.0,
                    2.0,
                    1.5,
                    1.0,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    43.0,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_sign_same():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSign(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sign(X),
            np.sign(X),
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max
    fmin = np.finfo(X.dtype).smallest_subnormal

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -0.0,
                    -0.0,
                    fmin,
                    fmin,
                    fmin,
                    fmin,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmin,
                    -fmin,
                    -fmin,
                    -fmin,
                    +0.0,
                    +0.0,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_sign_one_off():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSign(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sign(X) - 1,
            np.sign(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -0.0,
                    -0.0,
                    -0.0,
                    -0.0,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    +0.0,
                    +0.0,
                    +0.0,
                    +0.0,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_sign_any():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSign(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sign(X) - 2,
            np.sign(X) + 2,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max

    assert np.all(
        (
            valid._lower[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    -fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._lower[0]))
    )
    assert np.all(np.isnan(valid._lower[1]))
    assert np.all(
        (
            valid._upper[0]
            == np.array(
                [
                    -np.nan,
                    -np.inf,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    fmax,
                    np.inf,
                    np.nan,
                ]
            )
        )
        | (np.isnan(X) & np.isnan(valid._upper[0]))
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_square():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSquare(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.square(X) - 1,
            np.square(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -np.sqrt(np.square(42.0) + 1),
                -np.sqrt(5.0),
                -np.sqrt(2.0),
                -np.sqrt(1.25),
                -1.0,
                -1.0,
                -np.sqrt(1.25),
                -np.sqrt(2.0),
                np.sqrt(3.0),
                np.sqrt(np.square(42.0) - 1),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -np.sqrt(np.square(42.0) - 1),
                -np.sqrt(3.0),
                np.sqrt(2.0),
                np.sqrt(1.25),
                1.0,
                1.0,
                np.sqrt(1.25),
                np.sqrt(2.0),
                np.sqrt(5.0),
                np.sqrt(np.square(42.0) + 1),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_reciprocal():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarReciprocal(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.reciprocal(X) - 1,
            np.reciprocal(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        assert np.all(
            (np.abs(np.reciprocal(X) - np.reciprocal(valid._lower[0])) <= 1.0)
            | (np.reciprocal(X) == np.reciprocal(valid._lower[0]))
            | (np.isnan(X) & np.isnan(valid._lower[0]))
        )
    assert np.all(np.isnan(valid._lower[1]))
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        assert np.all(
            (np.abs(np.reciprocal(X) - np.reciprocal(valid._upper[0])) <= 1.0)
            | (np.reciprocal(X) == np.reciprocal(valid._upper[0]))
            | (np.isnan(X) & np.isnan(valid._upper[0]))
        )
    assert np.all(np.isnan(valid._upper[1]))


def test_sin():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -42.0,
            -2.0,
            -1.0,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            1.0,
            2.0,
            42.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarSin(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.sin(X) - 1,
            np.sin(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max

    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -42.0 - (np.asin(1.0) - np.asin(np.sin(-42.0))),
                -2.0 - (np.asin(np.sin(-2.0) + 1.0) - np.asin(np.sin(-2.0))),
                -np.pi / 2,
                -np.pi / 2,
                -fmax,
                -fmax,
                0.5 + (np.asin(np.sin(0.5) - 1) - np.asin(np.sin(0.5))),
                1.0 + (np.asin(np.sin(1.0) - 1) - np.asin(np.sin(1.0))),
                np.pi / 2,
                42.0 - (np.asin(np.sin(42.0) + 1) - np.asin(np.sin(42.0))),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -42.0 - (np.asin(np.sin(-42.0) - 1) - np.asin(np.sin(-42.0))),
                -np.pi / 2,
                -1.0 + (np.asin(np.sin(-1.0) + 1) - np.asin(np.sin(-1.0))),
                -0.5 + (np.asin(np.sin(-0.5) + 1) - np.asin(np.sin(-0.5))),
                fmax,
                fmax,
                np.pi / 2,
                np.pi / 2,
                2.0 - (np.asin(np.sin(2.0) - 1.0) - np.asin(np.sin(2.0))),
                42.0 - (np.asin(-1.0) - np.asin(np.sin(42.0))),
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._upper[1]))


def test_asin():
    X = np.array(
        [
            -np.nan,
            -np.inf,
            -2.0,
            -1.0,
            -0.75,
            -0.5,
            -0.0,
            +0.0,
            0.5,
            0.75,
            1.0,
            2.0,
            np.inf,
            np.nan,
        ]
    )

    expr = ScalarAsin(Data.SCALAR)

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        X_lower, X_upper = expr.compute_data_bounds(
            np.asin(X) - 1,
            np.asin(X) + 1,
            X,
            X,
            dict(),
        )
        valid = compute_safe_data_lower_upper_interval_union(
            X,
            X_lower,
            X_upper,
        )

    fmax = np.finfo(X.dtype).max
    one_eps = np.nextafter(X.dtype.type(1), 2)

    np.testing.assert_allclose(
        valid._lower[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -fmax,
                -1.0,
                -1.0,
                np.sin(np.asin(-0.5) - 1.0),
                np.sin(-1.0),
                np.sin(-1.0),
                np.sin(np.asin(0.5) - 1.0),
                np.sin(np.asin(0.75) - 1.0),
                np.sin(np.asin(1.0) - 1.0),
                one_eps,
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._lower[1]))
    np.testing.assert_allclose(
        valid._upper[0],
        np.array(
            [
                -np.nan,
                -np.inf,
                -one_eps,
                np.sin(np.asin(-1.0) + 1.0),
                np.sin(np.asin(-0.75) + 1.0),
                np.sin(np.asin(-0.5) + 1.0),
                np.sin(1.0),
                np.sin(1.0),
                np.sin(np.asin(0.5) + 1.0),
                1.0,
                1.0,
                fmax,
                np.inf,
                np.nan,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
        equal_nan=True,
    )
    assert np.all(np.isnan(valid._upper[1]))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_bounded_hang():
    X = np.array(-1.79769313e308, dtype=np.float64)

    expr = ScalarPower(
        ScalarSign(Data.SCALAR),
        ScalarReciprocal(ScalarTanh(Data.SCALAR)),  # coth(x)
    )

    assert expr.eval((), X, dict()) == np.float64(-1.0)

    expr_lower = np.array(-1, dtype=np.float64)
    expr_upper = np.array(0, dtype=np.float64)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower <= X
    assert X_upper == X

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float64(-1.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float64(-1.0))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_subtract_inf():
    X = np.array(0.0, dtype=np.float32)

    expr = ScalarSubtract(Data.SCALAR, Data.SCALAR)

    assert expr.eval((), X, dict()) == np.array(np.float32(0.0))

    expr_lower = np.array(-np.inf, dtype=np.float32)
    expr_upper = np.array(np.inf, dtype=np.float32)

    fmax = np.finfo(X.dtype).max

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    # bad: -inf - -inf = NaN and +inf - +inf = NaN
    assert X_lower == np.array(-fmax, dtype=np.float32)
    assert X_upper == np.array(fmax, dtype=np.float32)

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float32(0.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float32(0.0))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_times_zero():
    X = np.array(-np.inf, dtype=np.float16)

    expr = ScalarMultiply(Number("-9.58497987659779e+300"), ScalarAbs(Data.SCALAR))

    assert expr.eval((), X, dict()) == np.array(np.float16(-np.inf))

    expr_lower = np.array(-np.inf, dtype=np.float16)
    expr_upper = np.array(np.inf, dtype=np.float16)

    smallest_subnormal = np.finfo(X.dtype).smallest_subnormal

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(-np.inf, dtype=np.float16)
    assert X_upper == np.array(-smallest_subnormal, dtype=np.float16)

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float16(-np.inf))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float16(-np.inf))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_where():
    X = np.array(0.0, dtype=np.float16)

    expr = ScalarWhere(
        ScalarAbs(Data.SCALAR),
        ScalarAcos(Data.SCALAR),
        ScalarAcosh(ScalarReciprocal(Data.SCALAR)),
    )

    assert expr.eval((), X, dict()) == np.array(np.float16(np.inf))

    expr_lower = np.array(0.0, dtype=np.float16)
    expr_upper = np.array(np.inf, dtype=np.float16)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(0.0, dtype=np.float16) and not np.signbit(X_lower)
    assert X_upper == np.array(0.0, dtype=np.float16) and not np.signbit(X_upper)

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float16(np.inf))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float16(np.inf))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_inconsistent_where():
    X = np.array(_float128("-1.797693134862315708145274237317044e+308"))

    expr = ScalarWhere(
        ScalarSubtract(Number.ONE, ScalarIsNaN(ScalarAcos(Data.SCALAR))),
        ScalarRoundTiesEven(Data.SCALAR),
        ScalarAtan(ScalarReciprocal(Data.SCALAR)),  # acot(x)
    )

    assert expr.eval((), X, dict()) == np.array(
        _float128("-5.562684646268004075307639094889258e-309")
    )

    expr_lower = np.array(_float128("-5.562684646268004075307639094889258e-309"))
    expr_upper = np.array(_float128("0.0e+000"))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(_float128("-inf"))
    assert X_upper == np.array(_float128("-1.797693134862315708145274237317044e+308"))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(_float128("-0.0e+000"))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(
        _float128("-5.562684646268004075307639094889258e-309")
    )


def test_fuzzer_found_cosine_monotonicity():
    X = np.array(0.1133, dtype=np.float16)

    expr = ScalarCos(Data.SCALAR)

    assert expr.eval((), X, dict()) == np.array(0.9937, dtype=np.float16)

    expr_lower = np.array(7.3e-06, dtype=np.float16)
    expr_upper = np.array(0.9937, dtype=np.float16)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(0.1133, dtype=np.float16)
    assert X_upper == np.array(1.57, dtype=np.float16)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_asinh_overflow():
    X = np.array(_float128("-4.237431194812790058760014731131757e+4778"))

    expr = ScalarAsinh(Data.SCALAR)
    assert np.rint(expr.eval((), X, dict())) == np.array(_float128("-11004"))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_invalid_divide_rewrite():
    X = np.array(2.9e-322, dtype=np.float64)

    expr = ScalarDivide(ScalarIsInf(Data.SCALAR), ScalarAbs(Data.SCALAR))

    assert expr.eval((), X, dict()) == np.array(0.0, dtype=np.float64)

    expr_lower = np.array(0.0, dtype=np.float64)
    expr_upper = np.array(0.0, dtype=np.float64)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(2.9e-322, dtype=np.float64)
    assert X_upper == np.array(1.7976931348623157e308, dtype=np.float64)

    assert expr.eval((), np.array(X_lower), dict()) == np.array(0.0, dtype=np.float64)
    assert expr.eval((), np.array(X_upper), dict()) == np.array(0.0, dtype=np.float64)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_power_nan_hang():
    X = np.array(-1.26836425e-30, dtype=np.float64)

    expr = ScalarPower(ScalarSign(Data.SCALAR), Data.SCALAR)

    assert np.isnan(expr.eval((), X, dict()))

    expr_lower = np.array(np.nan, dtype=np.float64)
    expr_upper = np.array(np.nan, dtype=np.float64)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(-1.26836425e-30, dtype=np.float64)
    assert X_upper == np.array(-1.26836425e-30, dtype=np.float64)

    assert np.isnan(expr.eval((), np.array(X_lower), dict()))
    assert np.isnan(expr.eval((), np.array(X_upper), dict()))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_divide_tiny_hang():
    X = np.array(2.81944e-319, dtype=np.float64)

    expr = ScalarDivide(ScalarMultiply(Data.SCALAR, Data.SCALAR), Data.SCALAR)

    assert expr.eval((), X, dict()) == np.array(0.0, dtype=np.float64)

    expr_lower = np.array(0.0, dtype=np.float64)
    expr_upper = np.array(0.0, dtype=np.float64)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(2.81944e-319, dtype=np.float64)
    assert X_upper == np.array(2.81944e-319, dtype=np.float64)

    assert expr.eval((), np.array(X_lower), dict()) == np.array(0.0, dtype=np.float64)
    assert expr.eval((), np.array(X_upper), dict()) == np.array(0.0, dtype=np.float64)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_asinh_excessive_nudging():
    X = np.array(_float128(0.0))

    expr = ScalarDivide(ScalarMultiply(ScalarAsinh(Data.SCALAR), Data.SCALAR), Euler())

    assert expr.eval((), X, dict()) == np.array(_float128(0.0))

    expr_lower = np.array(_float128(0.0))
    expr_upper = np.array(_float128(0.0))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(_float128(0.0))
    assert X_upper == np.array(_float128(0.0))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_excessive_nudging_log_power():
    X = np.array(0.0, dtype=np.float16)

    expr = ScalarPower(ScalarLog(Logarithm.log10, Data.SCALAR), Euler())

    assert expr.eval((), X, dict()) == np.array(np.inf, dtype=np.float16)

    expr_lower = np.array(0.0, dtype=np.float16)
    expr_upper = np.array(np.inf, dtype=np.float16)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(0.0, dtype=np.float16)
    assert X_upper == np.array(0.0, dtype=np.float16)


def test_fuzzer_found_negative_power_symbolic_const_propagation():
    a = ScalarPower(Number("2"), Number("2"))
    assert isinstance(a, Number)
    assert a.as_int() == 4

    a = ScalarPower(Number("-2"), Number("3"))
    assert isinstance(a, Number)
    assert a.as_int() == -8

    a = ScalarPower(Number("-5"), Number("0"))
    assert isinstance(a, Number)
    assert a.as_int() == 1

    a = ScalarPower(Number("0"), Number("0"))
    assert isinstance(a, Number)
    assert a.as_int() == 1

    a = ScalarPower(Number("-5"), Number("-1"))
    assert isinstance(a, ScalarPower)

    a = ScalarPower(Number("0"), Number("-1"))
    assert isinstance(a, ScalarPower)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_nan_power_zero():
    X = np.array(_float128("1.65e-4963"))

    expr = ScalarPower(
        Number("nan"),
        ScalarRoundTiesEven(Data.SCALAR),
    )

    assert expr.eval((), X, dict()) == np.array(_float128(1.0))

    expr_lower = np.array(_float128(0.0))
    expr_upper = np.array(_float128(1.0))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(_float128(0.0))
    assert X_upper == np.array(_float128(0.5))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(_float128(1.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(_float128(1.0))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_addsub_bound_overflow():
    X = np.array(np.float16(6.0e-08))

    expr = ScalarMultiply(
        ScalarDivide(
            ScalarIsFinite(Data.SCALAR),
            ScalarAbs(Data.SCALAR),
        ),
        ScalarAtan(Data.SCALAR),
    )

    assert expr.eval((), X, dict()) == np.array(np.float16(np.inf))

    expr_lower = np.array(np.float16(-57376.0))
    expr_upper = np.array(np.float16(np.inf))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float16(6.0e-08))
    assert X_upper == np.array(np.float16(6.0e-08))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_log_upper_bound_negative_zero():
    X = np.array(np.float32(7.47605e-27))

    expr = ScalarLogWithBase(
        Number("-0.0"),
        ScalarTanh(Data.SCALAR),
    )

    assert expr.eval((), X, dict()) == np.array(np.float32(np.inf))

    expr_lower = np.array(np.float32(7.47605e-27))
    expr_upper = np.array(np.float32(np.inf))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float32(1.0e-45))
    assert X_upper == np.array(np.float32(8.66434))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float32(np.inf))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float32(np.inf))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_excessive_nudging_one_power_nan():
    X = np.array(np.float16(0.0))

    expr = ScalarPower(
        ScalarExp(Exponential.exp2, Data.SCALAR),
        Number("nan"),
    )

    assert expr.eval((), X, dict()) == np.array(np.float16(1.0))

    expr_lower = np.array(np.float16(0.0))
    expr_upper = np.array(np.float16(1.0))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float16(0.0))
    assert X_upper == np.array(np.float16(0.0))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float16(1.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float16(1.0))


def test_fuzzer_found_excessive_nudging_atan_product():
    X = np.array(np.float32(33556004.0))

    expr = ScalarMultiply(
        ScalarMultiply(
            ScalarAtan(Data.SCALAR),
            ScalarAtan(Data.SCALAR),
        ),
        ScalarAtan(Data.SCALAR),
    )

    assert np.round(expr.eval((), X, dict()), 3) == np.array(np.float32(3.876))

    expr_lower = expr.eval((), X, dict())
    expr_upper = np.array(np.float32(5.0197614e33))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower <= np.array(np.float32(33556004.0))
    assert X_upper == np.array(np.float32(33556004.0))

    assert np.round(expr.eval((), np.array(X_lower), dict()), 3) == np.array(
        np.float32(3.876)
    )
    assert np.round(expr.eval((), np.array(X_upper), dict()), 3) == np.array(
        np.float32(3.876)
    )


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_oh_no_multiplication():
    X = np.array(np.float64(-1.4568159901474651e144))

    expr = ScalarMultiply(
        ScalarReciprocal(Data.SCALAR),
        ScalarMultiply(
            ScalarReciprocal(Data.SCALAR),
            ScalarSquare(Data.SCALAR),
        ),
    )

    assert expr.eval((), X, dict()) == np.array(np.float64(1.0))

    expr_lower = np.array(np.float64(-1.4568159901474629e144))
    expr_upper = np.array(np.float64(1.0))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float64(-1.4568159901474651e144))
    assert X_upper == np.array(np.float64(-1.4568159901474651e144))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float64(1.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float64(1.0))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_excessive_nudging_product():
    X = np.array(np.float16(255.9))

    expr = ScalarMultiply(
        ScalarAbs(Data.SCALAR),
        ScalarMultiply(
            ScalarIsFinite(Data.SCALAR),
            ScalarRoundTiesEven(Data.SCALAR),
        ),
    )

    assert expr.eval((), X, dict()) == np.array(np.float16(65500.0))

    expr_lower = np.array(np.float16(0.0))
    expr_upper = np.array(np.float16(65504.0))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float16(0.0))
    assert X_upper == np.array(np.float16(255.9))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float16(0.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float16(65500.0))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_excessive_nudging_zero_product():
    X = np.array(np.float32(1.2830058e-22))

    expr = ScalarMultiply(
        ScalarNegate(Data.SCALAR),
        ScalarMultiply(
            ScalarFoldedConstant(np.float32(1.283019e-22)),
            ScalarNegate(Data.SCALAR),
        ),
    )

    assert expr.eval((), X, dict()) == np.array(np.float32(0.0))

    expr_lower = np.array(np.float32(0.0))
    expr_upper = np.array(np.float32(1.2630801e-38))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float32(0.0))
    assert X_upper == np.array(np.float32(3.8519333e-19))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float32(0.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float32(0.0))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_excessive_nudging_division():
    X = np.array(np.float64(3.7921287488073535e146))

    expr = ScalarDivide(
        ScalarMultiply(
            ScalarAbs(Data.SCALAR),
            ScalarAcosh(Data.SCALAR),
        ),
        ScalarAbs(Data.SCALAR),
    )

    assert expr.eval((), X, dict()) == np.array(np.float64(338.2034982942516))

    expr_lower = np.array(np.float64(338.2034982942516))
    expr_upper = np.array(np.float64(3.7921287488073793e146))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float64(3.7921287488073535e146))
    assert X_upper == np.array(np.float64(3.7921287488073535e146))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(
        np.float64(338.2034982942516)
    )
    assert expr.eval((), np.array(X_upper), dict()) == np.array(
        np.float64(338.2034982942516)
    )


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_tan_upper_negative_zero():
    X = np.array(np.float16(-0.841))

    expr = ScalarMultiply(
        ScalarDivide(
            Number("-56809"),
            ScalarTan(Data.SCALAR),
        ),
        Number("255"),
    )

    assert expr.eval((), X, dict()) == np.array(np.float16(np.inf))

    expr_lower = np.array(np.float16(-0.841))
    expr_upper = np.array(np.float16(np.inf))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float16(-1.57))
    assert X_upper == np.array(np.float16(-0.0))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float16(7012.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float16(np.inf))


def test_fuzzer_found_power_excessive_nudging():
    X = np.array(np.float32(13323083.0))

    expr = ScalarPower(
        Number("1263225675"),
        ScalarAtan(Data.SCALAR),
    )

    assert expr.eval((), X, dict()) == np.array(np.float32(197957600000000.0))

    expr_lower = np.array(np.float32(13323083.0))
    expr_upper = np.array(np.float32(1.979576e14))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float32(0.9948097))
    assert X_upper == np.array(np.float32(13323083.0))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float32(13323100.0))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(
        np.float32(197957600000000.0)
    )


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_sign_negative_zero_bound():
    X = np.array(np.float16(-0.0004613))

    expr = ScalarDivide(
        ScalarMultiply(
            Number("-1886417009"),
            Pi(),
        ),
        ScalarSign(Data.SCALAR),
    )

    assert expr.eval((), X, dict()) == np.array(np.float16(np.inf))

    expr_lower = np.array(np.float16(-0.0004613))
    expr_upper = np.array(np.float16(np.inf))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float16(-np.inf))
    assert X_upper == np.array(np.float16(-6.0e-08))

    assert expr.eval((), np.array(X_lower), dict()) == np.array(np.float16(np.inf))
    assert expr.eval((), np.array(X_upper), dict()) == np.array(np.float16(np.inf))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "float128"])
def test_negative_zero_ops(dtype):
    dtype = _float128_dtype if dtype == "float128" else np.dtype(dtype)
    ty = dtype.type

    assert _is_negative_zero(ty(-0.0))
    assert not _is_positive_zero(ty(-0.0))
    assert not _is_negative_zero(ty(+0.0))
    assert _is_positive_zero(ty(+0.0))

    assert _is_negative_zero(np.positive(ty(-0.0)))
    assert _is_positive_zero(np.positive(ty(+0.0)))

    assert _is_positive_zero(np.negative(ty(-0.0)))
    assert _is_negative_zero(np.negative(ty(+0.0)))

    assert _is_negative_zero(np.add(ty(-0.0), ty(-0.0)))
    assert _is_positive_zero(np.add(ty(-0.0), ty(+0.0)))
    assert _is_positive_zero(np.add(ty(+0.0), ty(-0.0)))
    assert _is_positive_zero(np.add(ty(+0.0), ty(+0.0)))

    assert _is_positive_zero(np.subtract(ty(-0.0), ty(-0.0)))
    assert _is_negative_zero(np.subtract(ty(-0.0), ty(+0.0)))
    assert _is_positive_zero(np.subtract(ty(+0.0), ty(-0.0)))
    assert _is_positive_zero(np.subtract(ty(+0.0), ty(+0.0)))

    assert _is_positive_zero(np.multiply(ty(-0.0), ty(-0.0)))
    assert _is_negative_zero(np.multiply(ty(-0.0), ty(+0.0)))
    assert _is_negative_zero(np.multiply(ty(+0.0), ty(-0.0)))
    assert _is_positive_zero(np.multiply(ty(+0.0), ty(+0.0)))
    assert _is_negative_zero(np.multiply(ty(-0.0), ty(1.0)))
    assert _is_positive_zero(np.multiply(ty(+0.0), ty(1.0)))

    assert np.isnan(np.divide(ty(-0.0), ty(-0.0)))
    assert np.isnan(np.divide(ty(-0.0), ty(+0.0)))
    assert np.isnan(np.divide(ty(+0.0), ty(-0.0)))
    assert np.isnan(np.divide(ty(+0.0), ty(+0.0)))
    assert _is_negative_zero(np.divide(ty(-0.0), ty(1.0)))
    assert _is_positive_zero(np.divide(ty(+0.0), ty(1.0)))
    assert np.isneginf(np.divide(ty(1.0), ty(-0.0)))
    assert np.isposinf(np.divide(ty(1.0), ty(+0.0)))

    assert np.power(ty(-0.0), ty(-0.0)) == ty(1.0)
    assert np.power(ty(-0.0), ty(+0.0)) == ty(1.0)
    assert np.power(ty(+0.0), ty(-0.0)) == ty(1.0)
    assert np.power(ty(+0.0), ty(+0.0)) == ty(1.0)
    assert np.power(ty(1.0), ty(-0.0)) == ty(1.0)
    assert np.power(ty(1.0), ty(+0.0)) == ty(1.0)
    assert _is_negative_zero(np.power(ty(-0.0), ty(1.0)))
    assert _is_positive_zero(np.power(ty(+0.0), ty(1.0)))
    assert _is_positive_zero(np.power(ty(-0.0), ty(2.0)))
    assert _is_positive_zero(np.power(ty(+0.0), ty(2.0)))
    assert _is_negative_zero(np.power(ty(-0.0), ty(3.0)))
    assert _is_positive_zero(np.power(ty(+0.0), ty(3.0)))
    assert _is_positive_zero(np.power(ty(-0.0), ty(3.6)))
    assert _is_positive_zero(np.power(ty(+0.0), ty(3.6)))
    assert np.isneginf(np.power(ty(-0.0), ty(-1.0)))
    assert np.isposinf(np.power(ty(+0.0), ty(-1.0)))
    assert np.isposinf(np.power(ty(-0.0), ty(-2.0)))
    assert np.isposinf(np.power(ty(+0.0), ty(-2.0)))
    assert np.isneginf(np.power(ty(-0.0), ty(-3.0)))
    assert np.isposinf(np.power(ty(+0.0), ty(-3.0)))
    assert np.isposinf(np.power(ty(-0.0), ty(-3.6)))
    assert np.isposinf(np.power(ty(+0.0), ty(-3.6)))

    assert np.isneginf(np.log(ty(-0.0)))
    assert np.isneginf(np.log(ty(+0.0)))
    assert np.isneginf(np.log2(ty(-0.0)))
    assert np.isneginf(np.log2(ty(+0.0)))
    assert np.isneginf(np.log10(ty(-0.0)))
    assert np.isneginf(np.log10(ty(+0.0)))

    assert np.exp(ty(-0.0)) == ty(1.0)
    assert np.exp(ty(+0.0)) == ty(1.0)
    assert np.exp2(ty(-0.0)) == ty(1.0)
    assert np.exp2(ty(+0.0)) == ty(1.0)
    assert np.power(ty(10.0), ty(-0.0)) == ty(1.0)
    assert np.power(ty(10.0), ty(+0.0)) == ty(1.0)

    assert _is_negative_zero(np.sqrt(ty(-0.0)))
    assert _is_positive_zero(np.sqrt(ty(+0.0)))

    assert _is_positive_zero(np.square(ty(-0.0)))
    assert _is_positive_zero(np.square(ty(+0.0)))

    assert np.isneginf(np.reciprocal(ty(-0.0)))
    assert np.isposinf(np.reciprocal(ty(+0.0)))

    assert _is_positive_zero(np.abs(ty(-0.0)))
    assert _is_positive_zero(np.abs(ty(+0.0)))

    assert _is_positive_zero(np.sign(ty(-0.0)))
    assert _is_positive_zero(np.sign(ty(+0.0)))

    assert _is_negative_zero(np.floor(ty(-0.0)))
    assert _is_positive_zero(np.floor(ty(+0.0)))
    assert _is_negative_zero(np.ceil(ty(-0.0)))
    assert _is_positive_zero(np.ceil(ty(+0.0)))
    assert _is_negative_zero(np.trunc(ty(-0.0)))
    assert _is_positive_zero(np.trunc(ty(+0.0)))
    assert _is_negative_zero(np.rint(ty(-0.0)))
    assert _is_positive_zero(np.rint(ty(+0.0)))

    assert _is_negative_zero(np.sin(ty(-0.0)))
    assert _is_positive_zero(np.sin(ty(+0.0)))

    assert np.cos(ty(-0.0)) == ty(1.0)
    assert np.cos(ty(+0.0)) == ty(1.0)

    assert _is_negative_zero(np.tan(ty(-0.0)))
    assert _is_positive_zero(np.tan(ty(+0.0)))

    assert _is_negative_zero(np.asin(ty(-0.0)))
    assert _is_positive_zero(np.asin(ty(+0.0)))

    assert np.rint(np.acos(ty(-0.0)) * 100) == ty(157)  # pi/2 * 100
    assert np.rint(np.acos(ty(+0.0)) * 100) == ty(157)  # pi/2 * 100

    assert _is_negative_zero(np.atan(ty(-0.0)))
    assert _is_positive_zero(np.atan(ty(+0.0)))

    assert _is_negative_zero(np.sinh(ty(-0.0)))
    assert _is_positive_zero(np.sinh(ty(+0.0)))

    assert np.cosh(ty(-0.0)) == ty(1.0)
    assert np.cosh(ty(+0.0)) == ty(1.0)

    assert _is_negative_zero(np.tanh(ty(-0.0)))
    assert _is_positive_zero(np.tanh(ty(+0.0)))

    assert _is_negative_zero(np.asin(ty(-0.0)))
    assert _is_positive_zero(np.asin(ty(+0.0)))

    assert np.isnan(np.acosh(ty(-0.0)))
    assert np.isnan(np.acosh(ty(+0.0)))

    assert _is_negative_zero(np.atan(ty(-0.0)))
    assert _is_positive_zero(np.atan(ty(+0.0)))

    assert np.isfinite(ty(-0.0))
    assert np.isfinite(ty(+0.0))
    assert np.array(np.isfinite(ty(-0.0))).astype(dtype) == ty(1.0)
    assert np.array(np.isfinite(ty(+0.0))).astype(dtype) == ty(1.0)

    assert not np.isinf(ty(-0.0))
    assert not np.isinf(ty(+0.0))
    assert _is_positive_zero(np.array(np.isinf(ty(-0.0))).astype(dtype))
    assert _is_positive_zero(np.array(np.isinf(ty(+0.0))).astype(dtype))

    assert not np.isnan(ty(-0.0))
    assert not np.isnan(ty(+0.0))
    assert _is_positive_zero(np.array(np.isnan(ty(-0.0))).astype(dtype))
    assert _is_positive_zero(np.array(np.isnan(ty(+0.0))).astype(dtype))

    assert ty(-0.0) == 0
    assert ty(+0.0) == 0


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_power_nan_zero_in_bounds():
    X = np.array(np.float16(-15920.0))

    expr = ScalarPower(
        ScalarAcosh(Data.SCALAR),
        Data.SCALAR,
    )

    assert np.isnan(expr.eval((), X, dict()))

    expr_lower = np.array(np.float16(np.nan))
    expr_upper = np.array(np.float16(np.nan))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float16(-15920.0))
    assert X_upper == np.array(np.float16(-15920.0))

    assert np.isnan(expr.eval((), np.array(X_lower), dict()))
    assert np.isnan(expr.eval((), np.array(X_upper), dict()))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_power_nan_excessive_nudging():
    X = np.array(np.float64(-1.9794576248699151e-106))

    expr = ScalarPower(
        ScalarSqrt(Data.SCALAR),
        ScalarExp(Exponential.exp, Data.SCALAR),
    )

    assert np.isnan(expr.eval((), X, dict()))

    expr_lower = np.array(np.float64(np.nan))
    expr_upper = np.array(np.float64(np.nan))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float64(-1.9794576248699151e-106))
    assert X_upper == np.array(np.float64(-1.9794576248699151e-106))

    assert np.isnan(expr.eval((), np.array(X_lower), dict()))
    assert np.isnan(expr.eval((), np.array(X_upper), dict()))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_asinh_power_excessive_nudging():
    X = np.array(np.float32(2.52e-43))

    expr = ScalarPower(
        ScalarAsinh(Data.SCALAR),
        ScalarFoldedConstant(np.float32(3.2749045e-10)),
    )

    assert expr.eval((), X, dict()) == np.array(np.float32(0.99999994))

    expr_lower = np.array(np.float32(0.0))
    expr_upper = np.array(np.float32(0.99999994))

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(np.float32(0.0))
    assert X_upper == np.array(np.float32(2.52e-43))

    assert expr.eval((), X_lower, dict()) == np.array(np.float32(0.0))
    assert expr.eval((), X_upper, dict()) == np.array(np.float32(0.99999994))

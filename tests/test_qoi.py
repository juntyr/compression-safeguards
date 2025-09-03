import numpy as np

from compression_safeguards.safeguards._qois.expr.abs import ScalarAbs
from compression_safeguards.safeguards._qois.expr.addsub import ScalarSubtract
from compression_safeguards.safeguards._qois.expr.classification import (
    ScalarIsInf,
    ScalarIsNaN,
)
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
from compression_safeguards.safeguards._qois.expr.literal import Number
from compression_safeguards.safeguards._qois.expr.power import ScalarPower
from compression_safeguards.safeguards._qois.expr.reciprocal import ScalarReciprocal
from compression_safeguards.safeguards._qois.expr.round import ScalarRoundTiesEven
from compression_safeguards.safeguards._qois.expr.sign import ScalarSign
from compression_safeguards.safeguards._qois.expr.square import ScalarSquare
from compression_safeguards.safeguards._qois.expr.trigonometric import (
    ScalarAcos,
    ScalarAsin,
    ScalarAtan,
    ScalarCos,
    ScalarSin,
)
from compression_safeguards.safeguards._qois.expr.where import ScalarWhere
from compression_safeguards.safeguards._qois.interval import (
    compute_safe_data_lower_upper_interval_union,
)
from compression_safeguards.utils._float128 import _float128


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

    expr = ScalarAbs(Data(index=()))

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

    expr = ScalarSign(Data(index=()))

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

    expr = ScalarSign(Data(index=()))

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

    expr = ScalarSign(Data(index=()))

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

    expr = ScalarSquare(Data(index=()))

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

    expr = ScalarReciprocal(Data(index=()))

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

    expr = ScalarSin(Data(index=()))

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

    expr = ScalarAsin(Data(index=()))

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
        ScalarSign(Data(index=())),
        ScalarReciprocal(ScalarTanh(Data(index=()))),  # coth(x)
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

    expr = ScalarSubtract(Data(index=()), Data(index=()))

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

    expr = ScalarMultiply(Number("-9.58497987659779e+300"), ScalarAbs(Data(index=())))

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
        ScalarAbs(Data(index=())),
        ScalarAcos(Data(index=())),
        ScalarAcosh(ScalarReciprocal(Data(index=()))),
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
        ScalarSubtract(Number.ONE, ScalarIsNaN(ScalarAcos(Data(index=())))),
        ScalarRoundTiesEven(Data(index=())),
        ScalarAtan(ScalarReciprocal(Data(index=()))),  # acot(x)
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

    expr = ScalarCos(Data(index=()))

    assert expr.eval((), X, dict()) == np.array(0.9937, dtype=np.float16)

    expr_lower = np.array(7.3e-06, dtype=np.float16)
    expr_upper = np.array(0.9937, dtype=np.float16)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(0.1133, dtype=np.float16)
    assert X_upper == np.array(1.57, dtype=np.float16)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_asinh_overflow():
    X = np.array(_float128("-4.237431194812790058760014731131757e+4778"))

    expr = ScalarAsinh(Data(index=()))
    assert np.rint(expr.eval((), X, dict())) == np.array(_float128("-11004"))


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_invalid_divide_rewrite():
    X = np.array(2.9e-322, dtype=np.float64)

    expr = ScalarDivide(ScalarIsInf(Data(index=())), ScalarAbs(Data(index=())))

    assert expr.eval((), X, dict()) == np.array(0.0, dtype=np.float64)

    expr_lower = np.array(0.0, dtype=np.float64)
    expr_upper = np.array(0.0, dtype=np.float64)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(2.9e-322, dtype=np.float64)
    assert X_upper == np.array(2.9e-322, dtype=np.float64)

    assert expr.eval((), np.array(X_lower), dict()) == np.array(0.0, dtype=np.float64)
    assert expr.eval((), np.array(X_upper), dict()) == np.array(0.0, dtype=np.float64)


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def test_fuzzer_found_power_nan_hang():
    X = np.array(-1.26836425e-30, dtype=np.float64)

    expr = ScalarPower(ScalarSign(Data(index=())), Data(index=()))

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

    expr = ScalarDivide(ScalarMultiply(Data(index=()), Data(index=())), Data(index=()))

    assert expr.eval((), X, dict()) == np.array(0.0, dtype=np.float64)

    expr_lower = np.array(0.0, dtype=np.float64)
    expr_upper = np.array(0.0, dtype=np.float64)

    X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    assert X_lower == np.array(2.81944e-319, dtype=np.float64)
    assert X_upper == np.array(2.81944e-319, dtype=np.float64)

    assert expr.eval((), np.array(X_lower), dict()) == np.array(0.0, dtype=np.float64)
    assert expr.eval((), np.array(X_upper), dict()) == np.array(0.0, dtype=np.float64)


# def test_fuzzer_found_excessive_nudging():
#     X = np.array(_float128(0.0))

#     expr = ScalarDivide(
#         ScalarMultiply(ScalarAsinh(Data(index=())), Data(index=())), Euler()
#     )

#     assert expr.eval((), X, dict()) == np.array(_float128(0.0))

#     expr_lower = np.array(_float128(0.0))
#     expr_upper = np.array(_float128(0.0))

#     X_lower, X_upper = expr.compute_data_bounds(expr_lower, expr_upper, X, X, dict())


# ===

# dtype = dtype('float16')
# X = array(0., dtype=float16)
# expr = log10(x) ** e
# exprv = np.float16(inf)
# expr_lower = array(0., dtype=float16)
# expr_upper = array(inf, dtype=float16)

# ===

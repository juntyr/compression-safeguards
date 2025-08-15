import atheris

with atheris.instrument_imports():
    import sys
    from typing import Callable

    import numpy as np

    from compression_safeguards.safeguards._qois.expr.abc import Expr
    from compression_safeguards.safeguards._qois.expr.abs import ScalarAbs
    from compression_safeguards.safeguards._qois.expr.classification import (
        ScalarIsFinite,
        ScalarIsInf,
        ScalarIsNaN,
    )
    from compression_safeguards.safeguards._qois.expr.data import Data
    from compression_safeguards.safeguards._qois.expr.group import Group
    from compression_safeguards.safeguards._qois.expr.hyperbolic import (
        Hyperbolic,
        ScalarAsinh,
        ScalarHyperbolic,
        ScalarSinh,
    )
    from compression_safeguards.safeguards._qois.expr.logexp import (
        Exponential,
        Logarithm,
        ScalarExp,
        ScalarLog,
    )
    from compression_safeguards.safeguards._qois.expr.neg import ScalarNegate
    from compression_safeguards.safeguards._qois.expr.reciprocal import ScalarReciprocal
    from compression_safeguards.safeguards._qois.expr.round import (
        ScalarCeil,
        ScalarFloor,
        ScalarRoundTiesEven,
        ScalarTrunc,
    )
    from compression_safeguards.safeguards._qois.expr.sign import ScalarSign
    from compression_safeguards.safeguards._qois.expr.square import (
        ScalarSqrt,
        ScalarSquare,
    )
    from compression_safeguards.safeguards._qois.expr.trigonometric import (
        ScalarAsin,
        ScalarSin,
        ScalarTrigonometric,
        Trigonometric,
    )
    from compression_safeguards.utils.cast import _float128_dtype, _isnan
    # from compression_safeguards.safeguards._qois.expr.addsub import ScalarAdd, ScalarSubtract
    # from compression_safeguards.safeguards._qois.expr.array import Array
    # from compression_safeguards.safeguards._qois.expr.divmul import ScalarDivide, ScalarMultiply
    # from compression_safeguards.safeguards._qois.expr.logexp import ScalarLogWithBase
    # from compression_safeguards.safeguards._qois.expr.power import ScalarPower
    # from compression_safeguards.safeguards._qois.expr.where import ScalarWhere


DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    _float128_dtype,
]

EXPRESSIONS: list[Callable[[Expr], Expr]] = [
    ScalarAbs,
    ScalarIsFinite,
    ScalarIsInf,
    ScalarIsNaN,
    Group,
    ScalarSinh,
    lambda a: ScalarHyperbolic(Hyperbolic.cosh, a),
    lambda a: ScalarHyperbolic(Hyperbolic.tanh, a),
    lambda a: ScalarHyperbolic(Hyperbolic.coth, a),
    lambda a: ScalarHyperbolic(Hyperbolic.sech, a),
    lambda a: ScalarHyperbolic(Hyperbolic.csch, a),
    ScalarAsinh,
    lambda a: ScalarHyperbolic(Hyperbolic.acosh, a),
    lambda a: ScalarHyperbolic(Hyperbolic.atanh, a),
    lambda a: ScalarHyperbolic(Hyperbolic.acoth, a),
    lambda a: ScalarHyperbolic(Hyperbolic.asech, a),
    lambda a: ScalarHyperbolic(Hyperbolic.acsch, a),
    lambda a: ScalarExp(Exponential.exp, a),
    lambda a: ScalarExp(Exponential.exp2, a),
    lambda a: ScalarExp(Exponential.exp10, a),
    lambda a: ScalarLog(Logarithm.ln, a),
    lambda a: ScalarLog(Logarithm.log2, a),
    lambda a: ScalarLog(Logarithm.log10, a),
    ScalarNegate,
    ScalarReciprocal,
    ScalarCeil,
    ScalarFloor,
    ScalarRoundTiesEven,
    ScalarTrunc,
    ScalarSign,
    ScalarSqrt,
    ScalarSquare,
    ScalarSin,
    lambda a: ScalarTrigonometric(Trigonometric.cos, a),
    lambda a: ScalarTrigonometric(Trigonometric.tan, a),
    lambda a: ScalarTrigonometric(Trigonometric.cot, a),
    lambda a: ScalarTrigonometric(Trigonometric.sec, a),
    lambda a: ScalarTrigonometric(Trigonometric.csc, a),
    ScalarAsin,
    lambda a: ScalarTrigonometric(Trigonometric.acos, a),
    lambda a: ScalarTrigonometric(Trigonometric.atan, a),
    lambda a: ScalarTrigonometric(Trigonometric.acot, a),
    lambda a: ScalarTrigonometric(Trigonometric.asec, a),
    lambda a: ScalarTrigonometric(Trigonometric.acsc, a),
]


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def check_one_input(data) -> None:
    data = atheris.FuzzedDataProvider(data)

    # select the dtype and expression to test
    dtype = DTYPES[data.ConsumeIntInRange(0, len(DTYPES) - 1)]
    expr = EXPRESSIONS[data.ConsumeIntInRange(0, len(EXPRESSIONS) - 1)]

    # generate the data
    X = np.array(data.ConsumeFloat(), dtype=dtype)

    # evaluate the expression on the data
    e: Expr = expr(Data(index=()))
    exprv = e.eval((), X, late_bound=dict())

    # generate the lower and upper bounds on the expression
    expr_lower = np.array(data.ConsumeFloat(), dtype=dtype)
    expr_upper = np.array(data.ConsumeFloat(), dtype=dtype)

    # skip if exprv and its bounds disagree on NaNs
    if (_isnan(exprv) != _isnan(expr_lower)) or (_isnan(exprv) != _isnan(expr_upper)):
        return

    # ensure that expr_lower <= exprv <= expr_upper
    # and that -0.0 bounds are handled correctly
    expr_lower = np.minimum(expr_lower, exprv)
    expr_lower = np.where(expr_lower == exprv, exprv, expr_lower)

    expr_upper = np.maximum(exprv, expr_upper)
    expr_upper = np.where(expr_upper == exprv, exprv, expr_upper)

    # compute the lower and upper bounds on the data
    # and evaluate the expression for them
    X_lower, X_upper = e.compute_data_bounds(expr_lower, expr_upper, X, X, dict())
    exprv_X_lower = e.eval((), X_lower, late_bound=dict())
    exprv_X_upper = e.eval((), X_upper, late_bound=dict())

    # generate a test data value
    X_test = np.array(data.ConsumeFloat(), dtype=dtype)

    # ensure that X and X_test agree on NaN
    X_test = np.where(_isnan(X) != _isnan(X_test), X, X_test)

    # ensure that X_lower <= X_test <= X_upper
    # and that -0.0 bounds are handled correctly
    X_test = np.maximum(X_lower, np.minimum(X_test, X_upper))
    X_test = np.where(X_test == X, X, X_test)
    X_test = np.where(X_test == X_lower, X_lower, X_test)
    X_test = np.where(X_test == X_upper, X_upper, X_test)

    # evaluate the expression on X_test
    exprv_X_test = e.eval((), X_test, late_bound=dict())

    try:
        # ASSERT: X bounds must only be NaN if X is NaN
        #         (they can be of any value if X is NaN)
        assert (not _isnan(X_lower)) or _isnan(X), "X_lower isnan mismatch"
        assert (not _isnan(X_upper)) or _isnan(X), "X_upper isnan mismatch"

        # ASSERT: X must be within the computed X bounds
        assert _isnan(X) or ((X >= X_lower) and (X <= X_upper)), "X outside bounds"

        # ASSERT: exprv bounds must be NaN iff exprv is NaN
        assert _isnan(exprv) == _isnan(exprv_X_lower), "exprv_X_lower isnan mismatch"
        assert _isnan(exprv) == _isnan(exprv_X_upper), "exprv_X_upper isnan mismatch"

        # ASSERT: exprv, exprv_X_lower, and exprv_X_upper must be within the
        #         expr bounds
        assert _isnan(exprv) or ((exprv >= expr_lower) and (exprv <= expr_upper)), (
            "exprv outside bounds"
        )
        assert _isnan(exprv) or (
            (exprv_X_lower >= expr_lower) and (exprv_X_lower <= expr_upper)
        ), "exprv_X_lower outside bounds"
        assert _isnan(exprv) or (
            (exprv_X_upper >= expr_lower) and (exprv_X_upper <= expr_upper)
        ), "exprv_X_upper outside bounds"

        # ASSERT: exprv_X_test bounds must be NaN iff exprv is NaN
        assert _isnan(exprv) == _isnan(exprv_X_test), "exprv_X_test isnan mismatch"

        # ASSERT: exprv_X_test must be within the expr bounds
        assert _isnan(exprv) or (
            (exprv_X_test >= expr_lower) and (exprv_X_test <= expr_upper)
        ), "exprv_X_test outside bounds"
    except Exception as err:
        print(
            "\n===\n\n"
            + "\n".join(
                [
                    f"dtype = {dtype!r}",
                    f"X = {X!r}",
                    f"e = {e!r}",
                    f"exprv = {exprv!r}",
                    f"expr_lower = {expr_lower!r}",
                    f"expr_upper = {expr_upper!r}",
                    f"X_lower = {X_lower!r}",
                    f"X_upper = {X_upper!r}",
                    f"exprv_X_lower = {exprv_X_lower!r}",
                    f"exprv_X_upper = {exprv_X_upper!r}",
                    f"X_test = {X_test!r}",
                    f"exprv_X_test = {exprv_X_test!r}",
                ]
            )
            + "\n\n===\n"
        )
        raise err


atheris.Setup(sys.argv, check_one_input)
atheris.Fuzz()


"""

Bugs to fix:
    
=== trigonometric functions on infinite inputs ===

dtype = <class 'numpy.float32'>
X = array(-inf, dtype=float32)
e = cot(x)
exprv = np.float32(nan)
expr_lower = array(nan, dtype=float32)
expr_upper = array(nan, dtype=float32)
X_lower = array(nan, dtype=float32)
X_upper = array(nan, dtype=float32)
exprv_X_lower = np.float32(nan)
exprv_X_upper = np.float32(nan)
X_test = array(nan, dtype=float32)
exprv_X_test = np.float32(nan)

===

"""

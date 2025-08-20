import atheris
from timeoutcontext import timeout

with atheris.instrument_imports():
    import sys
    import warnings
    from typing import Any, Callable

    import numpy as np

    from compression_safeguards.safeguards._qois.expr.abc import Expr
    from compression_safeguards.safeguards._qois.expr.abs import ScalarAbs
    from compression_safeguards.safeguards._qois.expr.addsub import (
        ScalarAdd,
        ScalarSubtract,
    )
    from compression_safeguards.safeguards._qois.expr.classification import (
        ScalarIsFinite,
        ScalarIsInf,
        ScalarIsNaN,
    )
    from compression_safeguards.safeguards._qois.expr.constfold import (
        ScalarFoldedConstant,
    )
    from compression_safeguards.safeguards._qois.expr.data import Data
    from compression_safeguards.safeguards._qois.expr.divmul import (
        ScalarDivide,
        ScalarMultiply,
    )
    from compression_safeguards.safeguards._qois.expr.group import Group
    from compression_safeguards.safeguards._qois.expr.hyperbolic import (
        Hyperbolic,
        ScalarAsinh,
        ScalarHyperbolic,
        ScalarSinh,
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
    from compression_safeguards.safeguards._qois.expr.where import ScalarWhere
    from compression_safeguards.utils._compat import _isnan
    from compression_safeguards.utils._float128 import _float128_dtype


DTYPES = [
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
    _float128_dtype,
]

NULLARY_EXPRESSIONS: list[Callable[[Any, np.dtype], Expr]] = [
    lambda data, dtype: Euler(),
    lambda data, dtype: Pi(),
    lambda data, dtype: Number(f"{data.ConsumeInt(4)}"),
    lambda data, dtype: Number(f"{data.ConsumeRegularFloat()}"),
    lambda data, dtype: ScalarFoldedConstant(dtype.type(data.ConsumeFloat())),
    lambda data, dtype: Data(index=()),
]
UNARY_EXPRESSIONS: list[Callable[[Expr], Expr]] = [
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
BINARY_EXPRESSIONS: list[Callable[[Expr, Expr], Expr]] = [
    ScalarAdd,
    ScalarSubtract,
    ScalarDivide,
    ScalarMultiply,
    ScalarLogWithBase,
    ScalarPower,
]
TERNARY_EXPRESSIONS: list[Callable[[Expr, Expr, Expr], Expr]] = [
    ScalarWhere,
    lambda a, b, c: ScalarAdd(ScalarAdd(a, b), c),
    lambda a, b, c: ScalarAdd(a, ScalarAdd(b, c)),
    lambda a, b, c: ScalarAdd(ScalarSubtract(a, b), c),
    lambda a, b, c: ScalarSubtract(ScalarAdd(a, b), c),
    lambda a, b, c: ScalarMultiply(ScalarMultiply(a, b), c),
    lambda a, b, c: ScalarMultiply(a, ScalarMultiply(b, c)),
    lambda a, b, c: ScalarMultiply(ScalarDivide(a, b), c),
    lambda a, b, c: ScalarDivide(ScalarMultiply(a, b), c),
]


warnings.filterwarnings("error")


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def check_one_input(data) -> None:
    data = atheris.FuzzedDataProvider(data)

    # select the dtype to test
    dtype = DTYPES[data.ConsumeIntInRange(0, len(DTYPES) - 1)]

    # build the shallow unary/binary/ternary expression to test
    try:
        with timeout(1):
            dataexpr = Data(index=())
            exprid = data.ConsumeIntInRange(
                0,
                len(UNARY_EXPRESSIONS)
                + len(BINARY_EXPRESSIONS)
                + len(TERNARY_EXPRESSIONS)
                - 1,
            )
            if exprid < len(UNARY_EXPRESSIONS):
                expr: Expr = (UNARY_EXPRESSIONS[exprid])(dataexpr)
            elif exprid < (len(UNARY_EXPRESSIONS) + len(BINARY_EXPRESSIONS)):
                exprs = []
                for _ in range(2):
                    i = data.ConsumeIntInRange(
                        0, len(NULLARY_EXPRESSIONS) + len(UNARY_EXPRESSIONS) - 1
                    )
                    if i < len(NULLARY_EXPRESSIONS):
                        exprs.append((NULLARY_EXPRESSIONS[i])(data, dtype))
                    else:
                        exprs.append(
                            (UNARY_EXPRESSIONS[i - len(NULLARY_EXPRESSIONS)])(dataexpr)
                        )
                [expra, exprb] = exprs
                expr = (BINARY_EXPRESSIONS[exprid - len(UNARY_EXPRESSIONS)])(
                    expra, exprb
                )
            else:
                exprs = []
                for _ in range(3):
                    i = data.ConsumeIntInRange(
                        0, len(NULLARY_EXPRESSIONS) + len(UNARY_EXPRESSIONS) - 1
                    )
                    if i < len(NULLARY_EXPRESSIONS):
                        exprs.append((NULLARY_EXPRESSIONS[i])(data, dtype))
                    else:
                        exprs.append(
                            (UNARY_EXPRESSIONS[i - len(NULLARY_EXPRESSIONS)])(dataexpr)
                        )
                [expra, exprb, exprc] = exprs
                expr = (
                    TERNARY_EXPRESSIONS[
                        exprid - len(UNARY_EXPRESSIONS) - len(BINARY_EXPRESSIONS)
                    ]
                )(expra, exprb, exprc)
    except TimeoutError:
        # skip expressions that take too long just to build
        return
    except Warning as err:
        # skip expressions that try to perform a**b with excessive digits
        if ("symbolic integer evaluation" in str(err)) and (
            "excessive number of digits" in str(err)
        ):
            return
        raise err

    # skip if the expression is constant
    if not expr.has_data:
        return

    # generate the data
    X = np.array(data.ConsumeFloat(), dtype=dtype)

    # NaN data values are always preserved externally by the QoI safeguards, so
    #  the expressions don't need to handle them
    if _isnan(X):
        return

    # evaluate the expression on the data
    exprv = expr.eval((), X, late_bound=dict())

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
    expr_lower, expr_upper = np.array(expr_lower), np.array(expr_upper)

    # compute the lower and upper bounds on the data
    # and evaluate the expression for them
    try:
        with timeout(1):
            X_lower, X_upper = expr.compute_data_bounds(
                expr_lower, expr_upper, X, X, dict()
            )
    except TimeoutError as err:
        print(
            "\n===\n\n"
            + "\n".join(
                [
                    f"dtype = {dtype!r}",
                    f"X = {X!r}",
                    f"expr = {expr!r}",
                    f"exprv = {exprv!r}",
                    f"expr_lower = {expr_lower!r}",
                    f"expr_upper = {expr_upper!r}",
                ]
            )
            + "\n\n===\n"
        )
        raise err
    X_lower, X_upper = np.array(X_lower), np.array(X_upper)
    exprv_X_lower = expr.eval((), X_lower, late_bound=dict())
    exprv_X_upper = expr.eval((), X_upper, late_bound=dict())

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
    X_test = np.array(X_test)

    # evaluate the expression on X_test
    exprv_X_test = expr.eval((), X_test, late_bound=dict())

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
                    f"expr = {expr!r}",
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

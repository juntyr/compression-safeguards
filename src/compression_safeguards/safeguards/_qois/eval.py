from math import prod

import numpy as np
import sympy as sp
import sympy.tensor.array.expressions  # noqa: F401

from ...utils.cast import (
    _float128_dtype,
    _float128_precision,
    _sign,
)
from ...utils.typing import F
from .associativity import NonAssociativeAdd, NonAssociativeMul, rewrite_qoi_expr
from .symfunc import round_ties_even as sp_round_ties_even
from .symfunc import sign as sp_sign
from .symfunc import symmetric_modulo as sp_symmetric_modulo
from .symfunc import trunc as sp_trunc


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def evaluate_sympy_expr_to_numpy(
    expr: sp.Basic,
    vals: dict[
        sp.Symbol | sp.tensor.array.expressions.ArraySymbol,
        np.ndarray[tuple[int, ...], np.dtype[F]],
    ],
    dtype: np.dtype[F],
) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
    if expr is sp.S.ImaginaryUnit:
        raise ValueError("cannot evaluate an expression containing an imaginary number")

    if expr.is_Integer:
        return dtype.type(f"{expr.p}")

    if expr.is_Rational:
        return dtype.type(f"{expr.p}") / dtype.type(f"{expr.q}")

    if expr.is_Float:
        # TODO: check if this is the correct way to print at full precision
        return dtype.type(f"{expr}")

    if expr.is_NumberSymbol:
        precision = (
            _float128_precision
            if dtype == _float128_dtype
            else np.finfo(dtype).precision
        ) * 2
        return dtype.type(f"{sp.N(expr, precision)}")

    if expr is sp.S.Infinity:
        return dtype.type("inf")

    if expr is sp.S.NaN:
        return dtype.type("nan")

    if expr.is_Symbol:
        return vals[expr]

    if expr.func is sp.tensor.array.expressions.ArrayElement and len(expr.args) == 2:
        if expr.args[0] in vals:
            # data symbols are keyed directly
            return vals[expr.args[0]][tuple([...] + [int(i) for i in expr.indices])]
        # late-bound constants are keyed by the symbol
        return vals[expr.args[0].args[0]][tuple([...] + [int(i) for i in expr.indices])]

    if expr.func is sp.Abs and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.abs(argv)

    if expr.func is sp.log and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.log(argv)

    if expr.func is sp.exp and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.exp(argv)

    if expr.is_Pow and len(expr.args) == 2:
        a, b = expr.args
        av = evaluate_sympy_expr_to_numpy(a, vals, dtype)
        bv = evaluate_sympy_expr_to_numpy(b, vals, dtype)
        return np.power(av, bv)

    if expr.func is sp_sign and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return _sign(argv)

    if expr.func is sp.floor and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.floor(argv)

    if expr.func is sp.ceiling and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.ceil(argv)

    if expr.func is sp_trunc and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.trunc(argv)

    if expr.func is sp_round_ties_even and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.rint(argv)

    if expr.func is sp_symmetric_modulo and len(expr.args) == 2:
        (p, q) = expr.args
        pv = evaluate_sympy_expr_to_numpy(p, vals, dtype)
        qv = evaluate_sympy_expr_to_numpy(q, vals, dtype)
        q2v = qv / 2
        res = np.mod(pv + q2v, qv)
        if dtype == _float128_dtype:
            res = np.mod(res + qv, qv)
        return res - q2v

    if expr.func is sp.sin and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.sin(argv)

    if expr.func is sp.asin and len(expr.args) == 1:
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg, vals, dtype)
        return np.asin(argv)

    TRIGONOMETRIC = {
        # derived trigonometric functions
        sp.cos: lambda x: sp.sin(rewrite_qoi_expr(x + (sp.pi / 2)), evaluate=False),
        sp.tan: lambda x: rewrite_qoi_expr(sp.sin(x) / sp.cos(x)),
        sp.csc: lambda x: rewrite_qoi_expr(1 / sp.sin(x)),
        sp.sec: lambda x: rewrite_qoi_expr(1 / sp.cos(x)),
        sp.cot: lambda x: rewrite_qoi_expr(sp.cos(x) / sp.sin(x)),
        # inverse trigonometric functions
        sp.acos: lambda x: rewrite_qoi_expr((sp.pi / 2) - sp.asin(x)),
        sp.atan: lambda x: rewrite_qoi_expr(
            ((sp.pi / 2) - sp.asin(1 / sp.sqrt(x**2 + 1))) * (sp.Abs(x) / x)
        ),
        sp.acsc: lambda x: rewrite_qoi_expr(sp.asin(1 / x)),
        sp.asec: lambda x: rewrite_qoi_expr(sp.acos(1 / x)),
        sp.acot: lambda x: rewrite_qoi_expr(sp.atan(1 / x)),
    }

    # rewrite derived trigonometric functions using sin and asin
    if expr.func in TRIGONOMETRIC and len(expr.args) == 1:
        (arg,) = expr.args
        return evaluate_sympy_expr_to_numpy(
            (TRIGONOMETRIC[expr.func])(arg), vals, dtype
        )

    HYPERBOLIC = {
        # basic hyperbolic functions
        sp.sinh: lambda x: rewrite_qoi_expr((sp.exp(x) - sp.exp(-x)) / 2),
        sp.cosh: lambda x: rewrite_qoi_expr((sp.exp(x) + sp.exp(-x)) / 2),
        # derived hyperbolic functions
        sp.tanh: lambda x: rewrite_qoi_expr((sp.exp(x * 2) - 1) / (sp.exp(x * 2) + 1)),
        sp.csch: lambda x: rewrite_qoi_expr(2 / (sp.exp(x) - sp.exp(-x))),
        sp.sech: lambda x: rewrite_qoi_expr(2 / (sp.exp(x) + sp.exp(-x))),
        sp.coth: lambda x: rewrite_qoi_expr((sp.exp(x * 2) + 1) / (sp.exp(x * 2) - 1)),
        # inverse hyperbolic functions
        sp.asinh: lambda x: rewrite_qoi_expr(sp.ln(x + sp.sqrt(x**2 + 1))),
        sp.acosh: lambda x: rewrite_qoi_expr(sp.ln(x + sp.sqrt(x**2 - 1))),
        sp.atanh: lambda x: rewrite_qoi_expr(sp.ln((1 + x) / (1 - x)) / 2),
        sp.acsch: lambda x: rewrite_qoi_expr(sp.ln((1 / x) + sp.sqrt(x ** (-2) + 1))),
        sp.asech: lambda x: rewrite_qoi_expr(sp.ln((1 + sp.sqrt(1 - x**2)) / x)),
        sp.acoth: lambda x: rewrite_qoi_expr(sp.ln((x + 1) / (x - 1)) / 2),
    }

    # rewrite hyperbolic functions using their exponential definitions
    if expr.func in HYPERBOLIC and len(expr.args) == 1:
        (arg,) = expr.args
        return evaluate_sympy_expr_to_numpy((HYPERBOLIC[expr.func])(arg), vals, dtype)

    if expr.func is NonAssociativeAdd and len(expr.args) == 2:
        (a, b) = expr.args
        av = evaluate_sympy_expr_to_numpy(a, vals, dtype)
        bv = evaluate_sympy_expr_to_numpy(b, vals, dtype)
        return av + bv

    if expr.is_Add:
        arg, *args = expr.args
        return sum(
            (evaluate_sympy_expr_to_numpy(a, vals, dtype) for a in args),
            start=evaluate_sympy_expr_to_numpy(arg, vals, dtype),
        )

    if expr.func is NonAssociativeMul and len(expr.args) == 2:
        (a, b) = expr.args
        av = evaluate_sympy_expr_to_numpy(a, vals, dtype)
        bv = evaluate_sympy_expr_to_numpy(b, vals, dtype)
        return av * bv

    if expr.is_Mul:
        arg, *args = expr.args
        return prod(
            (evaluate_sympy_expr_to_numpy(a, vals, dtype) for a in args),
            start=evaluate_sympy_expr_to_numpy(arg, vals, dtype),
        )

    raise ValueError(f"unsupported expression kind {expr} (= {sp.srepr(expr)} =)")

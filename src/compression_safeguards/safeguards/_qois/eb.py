from typing import Callable

import numpy as np
import sympy as sp
import sympy.tensor.array.expressions  # noqa: F401

from ...utils.cast import (
    _float128_dtype,
    _float128_smallest_subnormal,
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero_inf_to_finite,
    _nextafter,
    _sign,
)
from ...utils.typing import F, S
from .array import NumPyLikeArray
from .associativity import NonAssociativeAdd, NonAssociativeMul, rewrite_qoi_expr
from .symfunc import round_ties_even as sp_round_ties_even
from .symfunc import sign as sp_sign
from .symfunc import trunc as sp_trunc
from .vars import LateBoundConstant


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def compute_data_eb_for_stencil_qoi_eb_unchecked(
    expr: sp.Basic,
    xv: np.ndarray[S, np.dtype[F]],
    eb_expr_lower: np.ndarray[S, np.dtype[F]],
    eb_expr_upper: np.ndarray[S, np.dtype[F]],
    check_is_x: Callable[[sp.Basic], bool],
    evaluate_sympy_expr_to_numpy: Callable[[sp.Basic], np.ndarray],
    late_bound_constants: frozenset[LateBoundConstant],
    compute_data_eb_for_stencil_qoi_eb: Callable[
        [
            # expr
            sp.Basic,
            # xv
            np.ndarray[S, np.dtype[F]],
            # eb_expr_lower
            np.ndarray[S, np.dtype[F]],
            # eb_expr_upper
            np.ndarray[S, np.dtype[F]],
        ],
        tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]],
    ],
) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
    """
    Translate an error bound on a derived quantity of interest (QoI) into an
    error bound on the input data.

    This function does not check the returned error bound on the input data,
    use `_compute_data_eb_for_qoi_eb` instead.

    Parameters
    ----------
    expr : sp.Basic
        Symbolic SymPy expression that defines the QoI.
    xv : np.ndarray[S, np.dtype[F]]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[S, np.dtype[F]]
        Finite pointwise lower bound on the QoI error, must be negative or zero.
    eb_expr_upper : np.ndarray[S, np.dtype[F]]
        Finite pointwise upper bound on the QoI error, must be positive or zero.
    check_is_x : Callable[[sp.Basic], bool]
        Check if an expression is equal to `x` the data symbol.
    late_bound_constants : frozenset[LateBoundConstant]
        Set of late-bound constants that are not counted as symbols.
    evaluate_sympy_expr_to_numpy : Callable[[sp.Basic], np.ndarray]
        Evaluate a sympy expression to a numpy array.
    compute_data_eb_for_stencil_qoi_eb : Callable[
        [
            # expr
            sp.Basic,
            # xv
            np.ndarray[S, np.dtype[F]],
            # eb_expr_lower
            np.ndarray[S, np.dtype[F]],
            # eb_expr_upper
            np.ndarray[S, np.dtype[F]],
        ],
        tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]],
    ]
        Callback for the outer checked version of this function.

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    > Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. *Proceedings of the VLDB Endowment*.
    16, 4 (December 2022), 697-710. Available from:
    [doi:10.14778/3574245.3574255](https://doi.org/10.14778/3574245.3574255).
    """

    assert len(expr.free_symbols - late_bound_constants) > 0, (
        "constants have no error bounds"
    )

    zero = np.array(0, dtype=xv.dtype)

    # x
    if check_is_x(expr):
        return (eb_expr_lower, eb_expr_upper)

    # array
    if expr.func in (sp.Array, NumPyLikeArray):
        raise ValueError("expression must evaluate to a scalar not an array")

    # abs(...) is only used internally in exp(ln(abs(...)))
    if expr.func is sp.Abs and len(expr.args) == 1:
        # evaluate arg
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        # flip the lower/upper error bound if the arg is negative
        eql = np.where(argv < 0, -eb_expr_upper, eb_expr_lower)
        equ = np.where(argv < 0, -eb_expr_lower, eb_expr_upper)
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eql,  # type: ignore
            equ,  # type: ignore
        )

    # ln(...)
    # sympy automatically transforms log(..., base) into ln(...)/ln(base)
    if expr.func is sp.log and len(expr.args) == 1:
        # evaluate arg and ln(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.log(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(np.exp(exprv + eb_expr_lower) - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(0, np.exp(exprv + eb_expr_upper) - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in ln(e^(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.log(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.log(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # e^(...)
    if expr.func is sp.exp and len(expr.args) == 1:
        # evaluate arg and e^arg
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.exp(argv)

        # update the error bounds
        # ensure that ln is not passed a negative argument
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(np.log(np.maximum(0, exprv + eb_expr_lower)) - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(0, np.log(np.maximum(0, exprv + eb_expr_upper)) - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in e^(ln(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.exp(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.exp(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # rewrite a ** b as e^(b*ln(abs(a)))
    # this is mathematically incorrect for a <= 0 but works for deriving error bounds
    if expr.is_Pow and len(expr.args) == 2:
        a, b = expr.args
        return compute_data_eb_for_stencil_qoi_eb(
            sp.exp(rewrite_qoi_expr(b * sp.ln(sp.Abs(a))), evaluate=False),
            xv,
            eb_expr_lower,
            eb_expr_upper,
        )

    # sign(...)
    if expr.func is sp_sign and len(expr.args) == 1:
        # evaluate arg and sign(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = _sign(argv)

        # evaluate the lower and upper sign bounds that satisfy the error bound
        exprv_lower = np.maximum(-1, exprv + np.maximum(-2, np.ceil(eb_expr_lower)))
        exprv_upper = np.minimum(exprv + np.minimum(np.floor(eb_expr_upper), +2), +1)

        if xv.dtype == _float128_dtype:
            smallest_subnormal = _float128_smallest_subnormal
        else:
            smallest_subnormal = np.finfo(xv.dtype).smallest_subnormal

        # compute the lower and upper arg bounds that produce the sign bounds
        argv_lower = np.where(
            _isnan(exprv),
            exprv,
            np.where(
                exprv_lower == 0,
                zero,
                np.where(
                    exprv_lower < 0,
                    np.array(-np.inf, dtype=xv.dtype),
                    smallest_subnormal,
                ),
            ),
        )
        argv_upper = np.where(
            _isnan(exprv),
            exprv,
            np.where(
                exprv_upper == 0,
                zero,
                np.where(
                    exprv_upper < 0,
                    -smallest_subnormal,
                    np.array(np.inf, dtype=xv.dtype),
                ),
            ),
        )

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(argv_lower - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(0, argv_upper - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in sign(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: _sign(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: _sign(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # floor(...): round down, towards negative infinity
    if expr.func is sp.floor and len(expr.args) == 1:
        # evaluate arg and floor(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.floor(argv)

        # compute the rounded result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will round to meet the error bounds
        argv_lower = exprv_lower
        argv_upper = _nextafter(exprv_upper + 1, exprv_upper)

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in floor(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.floor(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.floor(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # ceil(...): round up, towards positive infinity
    if expr.func is sp.ceiling and len(expr.args) == 1:
        # evaluate arg and ceil(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.ceil(argv)

        # compute the rounded result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will round to meet the error bounds
        argv_lower = _nextafter(exprv_lower - 1, exprv_lower)
        argv_upper = exprv_upper

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in ceil(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.ceil(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.ceil(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # trunc(...): round towards zero
    if expr.func is sp_trunc and len(expr.args) == 1:
        # evaluate arg and trunc(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.trunc(argv)

        # compute the truncated result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will truncate to meet the error bounds
        argv_lower = np.where(
            exprv_lower <= 0, _nextafter(exprv_lower - 1, exprv_lower), exprv_lower
        )
        argv_upper = np.where(
            exprv_upper >= 0, _nextafter(exprv_upper + 1, exprv_upper), exprv_upper
        )

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in trunc(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.trunc(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.trunc(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # round_ties_even(...)
    if expr.func is sp_round_ties_even and len(expr.args) == 1:
        # evaluate arg and round_ties_even(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.rint(argv)

        # compute the rounded result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will round to meet the error bounds
        argv_lower = exprv_lower - 0.5
        argv_upper = exprv_upper + 0.5

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in round_ties_even(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.rint(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.rint(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # sin(...)
    if expr.func is sp.sin and len(expr.args) == 1:
        # evaluate arg and sin(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.sin(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            # we need to compare to asin(sin(...)) instead of ... to account
            #  for asin's output domain
            np.minimum(
                np.asin(np.maximum(-1, exprv + eb_expr_lower)) - np.asin(exprv), 0
            ),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(
                0, np.asin(np.minimum(exprv + eb_expr_upper, 1)) - np.asin(exprv)
            ),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # np.asin maps to [-pi/2, +pi/2] where sin is monotonically increasing
        # flip the argument error bounds where sin is monotonically decreasing
        eal, eau = (
            np.where(np.sin(argv + eal) > exprv, -eau, eal),
            np.where(np.sin(argv + eau) < exprv, -eal, eau),
        )

        # handle rounding errors in sin(asin(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.sin(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.sin(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # asin(...)
    if expr.func is sp.asin and len(expr.args) == 1:
        # evaluate arg and asin(arg)
        (arg,) = expr.args
        argv = evaluate_sympy_expr_to_numpy(arg)
        exprv = np.asin(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            # np.sin(max(-np.pi/2, ...)) might not be precise, so explicitly
            #  bound lower bounds to be <= 0
            np.minimum(np.sin(np.maximum(-np.pi / 2, exprv + eb_expr_lower)) - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            # np.sin(min(..., np.pi/2)) might not be precise, so explicitly
            #  bound upper bounds to be >= 0
            np.maximum(0, np.sin(np.minimum(exprv + eb_expr_upper, np.pi / 2)) - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in asin(sin(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.asin(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.asin(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            arg,
            xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

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
        return compute_data_eb_for_stencil_qoi_eb(
            (TRIGONOMETRIC[expr.func])(arg),
            xv,
            eb_expr_lower,
            eb_expr_upper,
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
        return compute_data_eb_for_stencil_qoi_eb(
            (HYPERBOLIC[expr.func])(arg),
            xv,
            eb_expr_lower,
            eb_expr_upper,
        )

    # handle the constant term c of a sum (t..) + c
    if expr.func is NonAssociativeAdd and len(expr.args) == 2:
        # ensure the post-conditions provided by rewrite_qoi_expr are met
        (term, const) = expr.args
        assert len(term.free_symbols - late_bound_constants) > 0
        assert len(const.free_symbols - late_bound_constants) == 0

        # evaluate the non-constant and constant term and their sum
        termv = evaluate_sympy_expr_to_numpy(term)
        constv = evaluate_sympy_expr_to_numpy(const)
        exprv = termv + constv

        # handle rounding errors in the addition
        eb_term_lower = ensure_bounded_derived_error(
            lambda etl: (termv + etl) + constv,
            exprv,
            termv,
            eb_expr_lower,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_term_upper = ensure_bounded_derived_error(
            lambda etu: (termv + etu) + constv,
            exprv,
            termv,
            eb_expr_upper,
            eb_expr_lower,
            eb_expr_upper,
        )

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            term,
            xv,
            eb_term_lower,
            eb_term_upper,
        )

    # a_1 * e_1 + ... + a_n * e_n (weighted sum of non-constant terms)
    # using Corollary 2 and Lemma 4 from Jiao et al.
    if expr.is_Add:
        # ensure the post-conditions provided by rewrite_qoi_expr are met
        assert all(
            len(arg.free_symbols - late_bound_constants) > 0 for arg in expr.args
        )

        terms = list(expr.args[:])

        # peek-hole optimisation to extract weighting factors of the terms
        termvs = []
        factorvs = []
        for i, term in enumerate(terms):
            # extract the weighting factor of the term
            if term.func is NonAssociativeMul:
                (t, f) = term.args
                assert len(t.free_symbols - late_bound_constants) > 0
                assert len(f.free_symbols - late_bound_constants) == 0

                termvs.append(evaluate_sympy_expr_to_numpy(t))
                factorvs.append(evaluate_sympy_expr_to_numpy(f))
                terms[i] = t
            else:
                termvs.append(evaluate_sympy_expr_to_numpy(term))
                factorvs.append(np.array(1, dtype=xv.dtype))

        # evaluate the total expression sum
        exprv = sum(
            (termvs[i] * factorvs[i] for i in range(1, len(terms))),
            start=(termvs[0] * factorvs[0]),
        )

        # compute the sum of absolute factors
        # unrolled loop in case a factor is a late-bound constant array
        total_abs_factor = np.abs(factorvs[0])
        for factor in factorvs[1:]:
            total_abs_factor += np.abs(factor)

        etl = _nan_to_zero_inf_to_finite(eb_expr_lower / total_abs_factor)
        etu = _nan_to_zero_inf_to_finite(eb_expr_upper / total_abs_factor)

        # stack the lower and upper bounds for each term factor
        # flip the lower/upper error bound sign if the factor is negative
        # here we only flip the sign but not the bounds since we're combining
        #  a worst case lower/upper bound over several terms
        etl_stack = np.stack([np.where(factorv < 0, -etl, etl) for factorv in factorvs])
        etu_stack = np.stack([np.where(factorv < 0, -etu, etu) for factorv in factorvs])

        # handle rounding errors in the total absolute factor early
        etl_stack = ensure_bounded_derived_error(
            lambda etl_stack: sum(
                (
                    (termvs[i] + etl_stack[i]) * factorvs[i]
                    for i in range(1, len(terms))
                ),
                start=((termvs[0] + etl_stack[0]) * factorvs[0]),
            ),
            exprv,
            np.stack(termvs),
            etl_stack,
            eb_expr_lower,
            eb_expr_upper,
        )
        etu_stack = ensure_bounded_derived_error(
            lambda etu_stack: sum(
                (
                    (termvs[i] + etu_stack[i]) * factorvs[i]
                    for i in range(1, len(terms))
                ),
                start=((termvs[0] + etu_stack[0]) * factorvs[0]),
            ),
            exprv,
            np.stack(termvs),
            etu_stack,
            eb_expr_lower,
            eb_expr_upper,
        )

        eb_x_lower, eb_x_upper = None, None
        for i, (term, factorv) in enumerate(zip(terms, factorvs)):
            # recurse into the terms with a weighted error bound
            exl, exu = compute_data_eb_for_stencil_qoi_eb(
                term,
                xv,
                # flip the lower/upper error bound if the factor is negative
                # here we do *not* flip the sign since we already flipped it above
                np.where(factorv < 0, etu_stack[i], etl_stack[i]),  # type: ignore
                np.where(factorv < 0, etl_stack[i], etu_stack[i]),  # type: ignore
            )
            # combine the inner error bounds
            if eb_x_lower is None:
                eb_x_lower = exl
            else:
                eb_x_lower = np.maximum(eb_x_lower, exl)
            if eb_x_upper is None:
                eb_x_upper = exu
            else:
                eb_x_upper = np.minimum(eb_x_upper, exu)

        return eb_x_lower, eb_x_upper  # type: ignore

    # handle the constant factor f of a product (t..) * f
    if expr.func is NonAssociativeMul and len(expr.args) == 2:
        # ensure the post-conditions provided by rewrite_qoi_expr are met
        (term, const) = expr.args
        assert len(term.free_symbols - late_bound_constants) > 0
        assert len(const.free_symbols - late_bound_constants) == 0

        # evaluate the non-constant and constant term and their sum
        termv = evaluate_sympy_expr_to_numpy(term)
        factorv = evaluate_sympy_expr_to_numpy(const)
        exprv = termv * factorv

        efl = _nan_to_zero_inf_to_finite(eb_expr_lower / np.abs(factorv))
        efu = _nan_to_zero_inf_to_finite(eb_expr_upper / np.abs(factorv))

        # flip the lower/upper error bound if the factor is negative
        etl = np.where(factorv < 0, -efu, efl)
        etu = np.where(factorv < 0, -efl, efu)

        # handle rounding errors in the multiplication
        eb_term_lower = ensure_bounded_derived_error(
            lambda etl: (termv + etl) * factorv,
            exprv,
            termv,
            etl,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_term_upper = ensure_bounded_derived_error(
            lambda etu: (termv + etu) * factorv,
            exprv,
            termv,
            etu,
            eb_expr_lower,
            eb_expr_upper,
        )

        # composition using Lemma 3 from Jiao et al.
        return compute_data_eb_for_stencil_qoi_eb(
            term,
            xv,
            eb_term_lower,
            eb_term_upper,
        )

    # rewrite e_1 * ... * e_n (product of non-constant terms) as
    #  e^(ln(abs(e_1) + ... + ln(abs(e_n)))
    # this is mathematically incorrect if the product is non-positive,
    #  but works for deriving error bounds
    if expr.is_Mul:
        # ensure the post-conditions provided by rewrite_qoi_expr are met
        assert all(
            len(arg.free_symbols - late_bound_constants) > 0 for arg in expr.args
        )

        return compute_data_eb_for_stencil_qoi_eb(
            sp.exp(
                rewrite_qoi_expr(sp.Add(*[sp.log(sp.Abs(arg)) for arg in expr.args])),
                evaluate=False,
            ),
            xv,
            eb_expr_lower,
            eb_expr_upper,
        )

    raise ValueError(f"unsupported expression kind {expr} (= {sp.srepr(expr)} =)")


def ensure_bounded_derived_error(
    expr: Callable[[np.ndarray[S, np.dtype[F]]], np.ndarray[S, np.dtype[F]]],
    exprv: np.ndarray[S, np.dtype[F]],
    xv: np.ndarray[S, np.dtype[F]],
    eb_x_guess: np.ndarray[S, np.dtype[F]],
    eb_expr_lower: np.ndarray[S, np.dtype[F]],
    eb_expr_upper: np.ndarray[S, np.dtype[F]],
) -> np.ndarray[S, np.dtype[F]]:
    """
    Ensure that an error bound on an expression is met by an error bound on
    the input data by nudging the provided guess.

    Parameters
    ----------
    expr : Callable[[np.ndarray[S, np.dtype[F]]], np.ndarray[S, np.dtype[F]]]
        Expression over which the error bound will be ensured.

        The expression takes in the error bound guess and returns the value of
        the expression for this error.
    exprv : np.ndarray[S, np.dtype[F]]
        Evaluation of the expression for the zero-error case.
    xv : np.ndarray[S, np.dtype[F]]
        Actual values of the input data, which are only used for better
        refinement of the error bound guess.
    eb_x_guess : np.ndarray[S, np.dtype[F]]
        Provided guess for the error bound on the initial data.
    eb_expr_lower : np.ndarray[S, np.dtype[F]]
        Finite pointwise lower bound on the expression error, must be negative
        or zero.
    eb_expr_upper : np.ndarray[S, np.dtype[F]]
        Finite pointwise upper bound on the expression error, must be positive
        or zero.

    Returns
    -------
    eb_x : np.ndarray[S, np.dtype[F]]
        Finite pointwise error bound on the input data.
    """

    # check if any derived expression exceeds the error bound
    # this check matches the QoI safeguard's validity check
    def is_eb_exceeded(eb_x_guess):
        return ~np.where(
            _isfinite(exprv),
            ((expr(eb_x_guess) - exprv) >= eb_expr_lower)
            & ((expr(eb_x_guess) - exprv) <= eb_expr_upper),
            np.where(
                _isinf(exprv),
                expr(eb_x_guess) == exprv,
                _isnan(expr(eb_x_guess)),
            ),
        ) & (eb_x_guess != 0)

    eb_exceeded = is_eb_exceeded(eb_x_guess)

    if not np.any(eb_exceeded):
        return eb_x_guess

    # first try to nudge the error bound itself
    # we can nudge with nextafter since the expression values are floating
    #  point
    eb_x_guess = np.where(eb_exceeded, _nextafter(eb_x_guess, 0), eb_x_guess)  # type: ignore

    # check again
    eb_exceeded = is_eb_exceeded(eb_x_guess)

    if not np.any(eb_exceeded):
        return eb_x_guess

    # second try to nudge it with respect to the data
    eb_x_guess = np.where(eb_exceeded, _nextafter(xv + eb_x_guess, xv) - xv, eb_x_guess)  # type: ignore

    # check again
    eb_exceeded = is_eb_exceeded(eb_x_guess)

    if not np.any(eb_exceeded):
        return eb_x_guess

    while True:
        # finally fall back to repeatedly cutting it in half
        eb_x_guess = np.where(eb_exceeded, eb_x_guess * 0.5, eb_x_guess)  # type: ignore

        eb_exceeded = is_eb_exceeded(eb_x_guess)

        if not np.any(eb_exceeded):
            return eb_x_guess

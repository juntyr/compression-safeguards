"""
Stencil quantity of interest (QoI) absolute error bound safeguard.
"""

__all__ = ["QuantityOfInterestAbsoluteErrorBoundSafeguard"]

import functools
from typing import Callable

import numpy as np
import sympy as sp
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.tensor.array.expressions import ArrayElement, ArraySymbol

from ....cast import (
    F,
    _float128,
    _float128_dtype,
    _float128_precision,
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero,
    _nextafter,
    to_finite_float,
)
from ....intervals import IntervalUnion
from ...pointwise.qoi import Expr
from .. import BoundaryCondition
from ..abc import S, StencilSafeguard, T


class QuantityOfInterestAbsoluteErrorBoundSafeguard(StencilSafeguard):
    """
    The `QuantityOfInterestAbsoluteErrorBoundSafeguard` guarantees that the
    pointwise absolute error on a derived quantity of interest (QoI) over a
    neighbourhood of data points is less than or equal to the provided bound
    `eb_abs`.

    The quantity of interest is specified as a non-constant expression, in
    string form, on the neighbourhood tensor `X` that is centred on the
    pointwise value `x`. For example, to bound the error on the four-neighbour
    box mean in a 3x3 neighbourhood (where `x = X[0,0]`), set
    `qoi=Expr("(X[-1,0]+X[+1,0]+X[0,+1]+X[-1,0])/4")`. Note that `X` uses
    indexing relative to the centred data point `x`.

    If the derived quantity of interest for an element evaluates to an infinite
    value, this safeguard guarantees that the quantity of interest on the
    decoded value produces the exact same infinite value. For a NaN quantity of
    interest, this safeguard guarantees that the quantity of interest on the
    decoded value is also NaN, but does not guarantee that it has the same
    bit pattern.

    The qoi expression is written using the following EBNF grammar for `expr`:

    ```ebnf
    expr    =
        literal
      | const
      | var
      | unary
      | binary
    ;

    literal =
        int
      | float
    ;

    int     = ? integer literal ?;
    float   = ? floating point literal ?;

    const   =
        "e"                               (* Euler's number *)
      | "pi"                              (* pi *)
    ;

    var     =
        "x";                              (* pointwise data value *)
      | "X", "[", int, [ ",", int ], "]"  (* neighbourhood data value *)
    ;

    unary   =
        "(", expr, ")"                    (* parenthesis *)
      | "-", expr                         (* negation *)
      | "sqrt", "(", expr, ")"            (* square root *)
      | "ln", "(", expr, ")"              (* natural logarithm *)
      | "exp", "(", expr, ")"             (* exponential e^x *)
    ;

    binary  =
        expr, "+", expr                   (* addition *)
      | expr, "-", expr                   (* subtraction *)
      | expr, "*", expr                   (* multiplication *)
      | expr, "/", expr                   (* division *)
      | expr, "**", expr                  (* exponentiation *)
      | "log", "(", expr, ",", expr, ")"  (* logarithm log(a, base) *)
    ;
    ```

    Parameters
    ----------
    qoi : Expr
        The non-constant expression for computing the derived quantity of
        interest for a neighbourhood tensor `X`.
    shape : tuple[tuple[int, int], ...]
        The shape of the data neighbourhood, expressed as (before, after)
        tuples, where before is non-positive and after is non-negative.

        e.g. a neighbourhood of shape `((-1, 2), (-2, 0))` is 2D and contains
        one element before and two elements after the current one on the first
        axis, and two elements before on the second axis.
    axes : tuple[int, ...]
        The axes that the neighbourhood is collected from. The neighbourhood
        window is applied independently to any additional axes. The number of
        axes must match the number of dimensions in the shape.

        e.g. for a 3d data array, 2d shape `((-1, 1), (-1, 1))`, and axes
        `(0, -1)`, the neighbourhood is created over the first and last axis,
        and applied independently along the middle axis.
    boundary : str | BoundaryCondition
        Boundary condition for evaluating the quantity of interest near the
        data domain boundaries, e.g. by extending values.
    eb_abs : int | float
        The non-negative absolute error bound on the quantity of interest that
        is enforced by this safeguard.
    constant_boundary : None | int | float
        Optional constant value with which the data domain is extended for a
        constant boundary.
    """

    __slots__ = (
        "_qoi",
        "_shape",
        "_axes",
        "_boundary",
        "_eb_abs",
        "_constant_boundary",
        "_qoi_expr",
        "_X",
    )
    _qoi: Expr
    _shape: tuple[tuple[int, int], ...]
    _axes: tuple[int, ...]
    _boundary: BoundaryCondition
    _eb_abs: int | float
    _constant_boundary: None | int | float
    _qoi_expr: sp.Basic
    _X: ArraySymbol

    kind = "qoi_abs_stencil"

    def __init__(
        self,
        qoi: Expr,
        shape: tuple[tuple[int, int], ...],
        axes: tuple[int, ...],
        boundary: str | BoundaryCondition,
        eb_abs: int | float,
        constant_boundary: None | int | float = None,
    ):
        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert isinstance(eb_abs, int) or _isfinite(eb_abs), "eb_abs must be finite"

        self._qoi = qoi
        self._eb_abs = eb_abs

        s = []
        c = []
        assert len(shape) > 0, "shape must not be empty"
        for b, a in shape:
            assert b <= 0, "shape's before must be non-positive"
            assert a >= 0, "shape's after must be non-negative"
            s.append(abs(b) + 1 + abs(a))
            c.append(abs(b))
        self._shape = shape

        print(shape)
        print(s)
        print(c)

        self._X = ArraySymbol("X", s)
        X = self._X.as_explicit()
        X.__class__ = _ImmutableDenseNDimArrayWithRelativeIndexing
        x = X.__getitem__(c)

        print(X)
        print(x)

        assert len(axes) == len(shape), (
            "number of axes must match the number of shape dimensions"
        )

        self._boundary = (
            boundary
            if isinstance(boundary, BoundaryCondition)
            else BoundaryCondition[boundary]
        )
        assert (self._boundary != BoundaryCondition.constant) == (
            constant_boundary is None
        ), (
            "constant_boundary must be provided if and only if the constant boundary condition is used"
        )
        self._constant_boundary = constant_boundary

        assert len(qoi.strip()) > 0, "qoi expression must not be empty"
        try:
            qoi_expr = sp.parse_expr(
                self._qoi,
                local_dict=dict(x=x, X=X),
                global_dict=dict(
                    # literals
                    Integer=sp.Integer,
                    Float=sp.Float,
                    Rational=sp.Rational,
                    # constants
                    pi=sp.pi,
                    e=sp.E,
                    # operators
                    sqrt=sp.sqrt,
                    exp=sp.exp,
                    ln=sp.ln,
                    log=sp.log,
                ),
                transformations=(sp.parsing.sympy_parser.auto_number,),
            )
            # check if the expression is well-formed (e.g. no int's that cannot
            #  be printed) and if an error bound can be computed
            _canary_repr = str(qoi_expr)
            _canary_eb_abs = _compute_data_eb_for_qoi_eb(
                qoi_expr,
                self._X,
                np.zeros(s),
                np.zeros(s),
                np.zeros(s),
            )
        except Exception as err:
            raise AssertionError(
                f"failed to parse qoi expression {qoi!r}: {err}"
            ) from err
        assert len(qoi_expr.free_symbols) > 0, "qoi expression must not be constant"
        assert not qoi_expr.has(sp.I), (
            "qoi expression must not contain imaginary numbers"
        )

        self._qoi_expr = qoi_expr

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        raise NotImplementedError

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        raise NotImplementedError

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        config = dict(
            kind=type(self).kind,
            qoi=self._qoi,
            shape=self._shape,
            axes=self._axes,
            boundary=self._boundary.name,
            eb_abs=self._eb_abs,
            constant_boundary=self._constant_boundary,
        )

        if self._constant_boundary is None:
            del config["constant_boundary"]

        return config


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def _compute_data_eb_for_qoi_eb(
    expr: sp.Basic,
    x: sp.Symbol,
    xv: np.ndarray,
    tauv_lower: np.ndarray,
    tauv_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Translate an error bound on a derived quantity of interest (QoI) into an
    error bound on the input data.

    This function checks the computed error bound before returning to correct
    any rounding errors.

    Parameters
    ----------
    expr : sp.Basic
        Symbolic SymPy expression that defines the QoI.
    x : sp.Symbol
        Symbol for the pointwise input data.
    xv : np.ndarray[S, F]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[S, F]
        Finite pointwise lower bound on the QoI error, must be negative or zero.
    eb_expr_upper : np.ndarray[S, F]
        Finite pointwise upper bound on the QoI error, must be positive or zero.

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[S, F], np.ndarray[S, F]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    tl, tu = _compute_data_eb_for_qoi_eb_unchecked(expr, x, xv, tauv_lower, tauv_upper)

    exprl = _compile_sympy_expr_to_numpy([x], expr, xv.dtype)
    exprv = (exprl)(xv)

    # handle rounding errors in the lower error bound computation
    tl = _ensure_bounded_derived_error(
        lambda tl: np.where(tl == 0, exprv, (exprl)(xv + tl)),
        exprv,
        xv,
        tl,
        tauv_lower,
        tauv_upper,
    )
    tu = _ensure_bounded_derived_error(
        lambda tu: np.where(tu == 0, exprv, (exprl)(xv + tu)),
        exprv,
        xv,
        tu,
        tauv_lower,
        tauv_upper,
    )

    return tl, tu


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def _compute_data_eb_for_qoi_eb_unchecked(
    expr: sp.Basic,
    x: sp.Symbol,
    xv: np.ndarray[S, F],
    eb_expr_lower: np.ndarray[S, F],
    eb_expr_upper: np.ndarray[S, F],
) -> tuple[np.ndarray[S, F], np.ndarray[S, F]]:
    """
    Translate an error bound on a derived quantity of interest (QoI) into an
    error bound on the input data.

    This function does not check the returned error bound on the input data,
    use `_compute_data_eb_for_qoi_eb` instead.

    Parameters
    ----------
    expr : sp.Basic
        Symbolic SymPy expression that defines the QoI.
    x : sp.Symbol
        Symbol for the pointwise input data.
    xv : np.ndarray[S, F]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[S, F]
        Finite pointwise lower bound on the QoI error, must be negative or zero.
    eb_expr_upper : np.ndarray[S, F]
        Finite pointwise upper bound on the QoI error, must be positive or zero.

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[S, F], np.ndarray[S, F]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    assert len(expr.free_symbols) > 0, "constants have no error bounds"

    zero = np.array(0, dtype=xv.dtype)

    # x
    if expr.func is ArrayElement and len(expr.args) == 2 and expr.args[0] == x:
        return (eb_expr_lower, eb_expr_upper)

    # abs(...) is only used internally in exp(ln(abs(...)))
    if expr.func is sp.Abs and len(expr.args) == 1:
        # evaluate arg
        (arg,) = expr.args
        argv = _compile_sympy_expr_to_numpy([x], arg, xv.dtype)(xv)
        # flip the lower/upper error bound if the arg is negative
        eql = np.where(argv < 0, -eb_expr_upper, eb_expr_lower)
        equ = np.where(argv < 0, -eb_expr_lower, eb_expr_upper)
        return _compute_data_eb_for_qoi_eb(arg, x, xv, eql, equ)

    # ln(...)
    # sympy automatically transforms log(..., base) into ln(...)/ln(base)
    if expr.func is sp.log and len(expr.args) == 1:
        # evaluate arg and ln(arg)
        (arg,) = expr.args
        argv = _compile_sympy_expr_to_numpy([x], arg, xv.dtype)(xv)
        exprv = np.log(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.exp(exprv + eb_expr_lower) - argv,
        )
        eal = _nan_to_zero(to_finite_float(eal, xv.dtype))

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.exp(exprv + eb_expr_upper) - argv,
        )
        eau = _nan_to_zero(to_finite_float(eau, xv.dtype))

        # handle rounding errors in ln(e^(...)) early
        eal = _ensure_bounded_derived_error(
            lambda eal: np.log(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = _ensure_bounded_derived_error(
            lambda eau: np.log(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return _compute_data_eb_for_qoi_eb(arg, x, xv, eb_arg_lower, eb_arg_upper)

    # e^(...)
    if expr.func is sp.exp and len(expr.args) == 1:
        # evaluate arg and e^arg
        (arg,) = expr.args
        argv = _compile_sympy_expr_to_numpy([x], arg, xv.dtype)(xv)
        exprv = np.exp(argv)

        # update the error bounds
        # ensure that ln is not passed a negative argument
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.log(np.maximum(zero, exprv + eb_expr_lower)) - argv,
        )
        eal = _nan_to_zero(to_finite_float(eal, xv.dtype))

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.log(np.maximum(zero, exprv + eb_expr_upper)) - argv,
        )
        eau = _nan_to_zero(to_finite_float(eau, xv.dtype))

        # handle rounding errors in e^(ln(...)) early
        eal = _ensure_bounded_derived_error(
            lambda eal: np.exp(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = _ensure_bounded_derived_error(
            lambda eau: np.exp(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return _compute_data_eb_for_qoi_eb(arg, x, xv, eb_arg_lower, eb_arg_upper)

    # rewrite a ** b as e^(b*ln(abs(a)))
    # this is mathematically incorrect for a <= 0 but works for deriving error bounds
    if expr.is_Pow and len(expr.args) == 2:
        a, b = expr.args
        return _compute_data_eb_for_qoi_eb(
            sp.exp(b * sp.ln(sp.Abs(a)), evaluate=False),
            x,
            xv,
            eb_expr_lower,
            eb_expr_upper,
        )

    # a_1 * e_1 + ... + a_n * e_n + c (weighted sum)
    # using Corollary 2 and Lemma 4 from Jiao et al.
    if expr.is_Add:
        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        factors = []
        for i, term in enumerate(terms):
            # extract the weighting factor of the term
            if term.is_Mul:
                factors.append(
                    _compile_sympy_expr_to_numpy(
                        [],
                        sp.Mul(
                            *[arg for arg in term.args if len(arg.free_symbols) == 0]  # type: ignore
                        ),
                        xv.dtype,
                    )()
                )
                terms[i] = sp.Mul(
                    *[arg for arg in term.args if len(arg.free_symbols) > 0]  # type: ignore
                )
            else:
                factors.append(np.array(1))
        total_abs_factor = np.sum(np.abs(factors))

        etl: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_lower / total_abs_factor, xv.dtype)
        )
        etu: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_upper / total_abs_factor, xv.dtype)
        )

        # handle rounding errors in the total absolute factor early
        etl = _ensure_bounded_derived_error(
            lambda etl: etl * total_abs_factor,
            np.zeros_like(xv),
            None,
            etl,
            eb_expr_lower,
            eb_expr_upper,
        )
        etu = _ensure_bounded_derived_error(
            lambda etu: etu * total_abs_factor,
            np.zeros_like(xv),
            None,
            etu,
            eb_expr_lower,
            eb_expr_upper,
        )

        eb_x_lower, eb_x_upper = None, None
        for term, factor in zip(terms, factors):
            # recurse into the terms with a weighted error bound
            exl, exu = _compute_data_eb_for_qoi_eb(
                term,
                x,
                xv,
                # flip the lower/upper error bound if the factor is negative
                -etu if factor < 0 else etl,
                -etl if factor < 0 else etu,
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

    # rewrite f * e_1 * ... * e_n (product) as f * e^(ln(abs(e_1) + ... + ln(abs(e_n)))
    # this is mathematically incorrect if the product is non-positive,
    #  but works for deriving error bounds
    if expr.is_Mul:
        # extract the constant factor and reduce tauv
        factor = _compile_sympy_expr_to_numpy(
            [],
            sp.Mul(*[arg for arg in expr.args if len(arg.free_symbols) == 0]),  # type: ignore
            xv.dtype,
        )()

        efl: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_lower / np.abs(factor), xv.dtype)
        )
        efu: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_upper / np.abs(factor), xv.dtype)
        )

        # handle rounding errors in the factor early
        efl = _ensure_bounded_derived_error(
            lambda efl: efl * np.abs(factor),
            np.zeros_like(xv),
            None,
            efl,
            eb_expr_lower,
            eb_expr_upper,
        )
        efu = _ensure_bounded_derived_error(
            lambda efu: efu * np.abs(factor),
            np.zeros_like(xv),
            None,
            efu,
            eb_expr_lower,
            eb_expr_upper,
        )

        # flip the lower/upper error bound if the factor is negative
        eb_factor_lower = -efu if factor < 0 else efl
        eb_factor_upper = -efl if factor < 0 else efu

        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        if len(terms) == 1:
            return _compute_data_eb_for_qoi_eb(
                terms[0], x, xv, eb_factor_lower, eb_factor_upper
            )

        return _compute_data_eb_for_qoi_eb(
            sp.exp(sp.Add(*[sp.log(sp.Abs(term)) for term in terms]), evaluate=False),
            x,
            xv,
            eb_factor_lower,
            eb_factor_upper,
        )

    raise ValueError(f"unsupported expression kind {expr} (= {sp.srepr(expr)} =)")


def _ensure_bounded_derived_error(
    expr: Callable[[np.ndarray[S, F]], np.ndarray[S, F]],
    exprv: np.ndarray[S, F],
    xv: None | np.ndarray[S, F],
    eb_x_guess: np.ndarray[S, F],
    eb_expr_lower: np.ndarray[S, F],
    eb_expr_upper: np.ndarray[S, F],
) -> np.ndarray[S, F]:
    """
    Ensure that an error bound on an expression is met by an error bound on
    the input data by nudging the provided guess.

    Parameters
    ----------
    expr : Callable[[np.ndarray[S, F]], np.ndarray[S, F]]
        Expression over which the error bound will be ensured.

        The expression takes in the error bound guess and returns the value of
        the expression for this error.
    exprv : np.ndarray[S, F]
        Evaluation of the expression for the zero-error case.
    xv : None | np.ndarray[S, F]
        Actual values of the input data, which are only used for better
        refinement of the error bound guess.
    eb_x_guess : np.ndarray[S, F]
        Provided guess for the error bound on the initial data.
    eb_expr_lower : np.ndarray[S, F]
        Finite pointwise lower bound on the expression error, must be negative
        or zero.
    eb_expr_upper : np.ndarray[S, F]
        Finite pointwise upper bound on the expression error, must be positive
        or zero.

    Returns
    -------
    eb_x : np.ndarray[S, F]
        Finite pointwise error bound on the input data.
    """

    # check if any derived expression exceeds the error bound
    # this check matches the qoi safeguard's validity check
    is_eb_exceeded = lambda eb_x_guess: ~np.where(
        _isfinite(exprv),
        ((expr(eb_x_guess) - exprv) >= eb_expr_lower)
        & ((expr(eb_x_guess) - exprv) <= eb_expr_upper),
        np.where(
            _isinf(exprv),
            expr(eb_x_guess) == exprv,
            _isnan(expr(eb_x_guess)),
        ),
    )
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

    if xv is not None:
        # second try to nudge it with respect to the data
        eb_x_guess = np.where(
            eb_exceeded, _nextafter(xv + eb_x_guess, xv) - xv, eb_x_guess
        )  # type: ignore

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


def _create_ndarray_with_relative_indexing(
    array: ImmutableDenseNDimArray, center: tuple[int, ...]
) -> ImmutableDenseNDimArray:
    class _ImmutableDenseNDimArrayWithRelativeIndexing(ImmutableDenseNDimArray):
        def __getitem__(self, index):
            if isinstance(index, (int, sp.Integer)):
                print("index-int", index)
            elif isinstance(index, (tuple, list)):
                for i in index:
                    if isinstance(i, (int, sp.Integer)):
                        print("index-tuple-int", i)
                    elif isinstance(i, slice):
                        assert isinstance(i.start, (None, int, sp.Integer))
                        assert isinstance(i.stop, (None, int, sp.Integer))
                        assert isinstance(i.step, (None, int, sp.Integer))
                    else:
                        raise ValueError(f"unsupported index kind {i} {sp.srepr(i)}")
            else:
                raise ValueError(f"unsupported index kind {index}")
            return super().__getitem__(index)


def _compile_sympy_expr_to_numpy(
    symbols: list[sp.Symbol],
    expr: sp.Basic,
    dtype: np.dtype,
) -> Callable[..., np.ndarray]:
    """
    Compile the SymPy expression `expr` over a list of `variables` into a
    function that uses NumPy for numerical evaluation.

    The function evaluates to a numpy array of the provided `dtype` if all
    variable inputs are numpy arrays of the same `dtype`.
    """

    return sp.lambdify(
        symbols,
        expr,
        modules=["numpy"]
        + ([{_float128_dtype.name: _float128}] if dtype == _float128_dtype else []),
        printer=_create_sympy_numpy_printer_class(dtype),
        docstring_limit=0,
    )


@functools.cache
def _create_sympy_numpy_printer_class(
    dtype: np.dtype,
) -> type[sp.printing.numpy.NumPyPrinter]:
    """
    Create a SymPy to NumPy printer class that outputs numerical values and
    constants with the provided `dtype` and sufficient precision.
    """

    class NumPyDtypePrinter(sp.printing.numpy.NumPyPrinter):
        __slots__ = ("_dtype",)

        # remove default printing of known constants
        _kc = dict()

        def __init__(self, settings=None):
            self._dtype = dtype.name
            if settings is None:
                settings = dict()
            if dtype == _float128_dtype:
                settings["precision"] = _float128_precision * 2
            else:
                settings["precision"] = np.finfo(dtype).precision * 2
            super().__init__(settings)

        def _print_Integer(self, expr):
            return str(expr.p)

        def _print_Rational(self, expr):
            return f"{self._dtype}({expr.p}) / {self._dtype}({expr.q})"

        def _print_Float(self, expr):
            # explicitly create the float from its string representation
            #  e.g. 1.2 -> float16('1.2')
            s = super()._print_Float(expr)
            return f"{self._dtype}({s!r})"

        def _print_Exp1(self, expr):
            return self._print_NumberSymbol(expr)

        def _print_Pi(self, expr):
            return self._print_NumberSymbol(expr)

        def _print_NaN(self, expr):
            return f"{self._dtype}(nan)"

        def _print_Infinity(self, expr):
            return f"{self._dtype}(inf)"

        def _print_ImaginaryUnit(self, expr):
            raise ValueError(
                "cannot evaluate an expression containing an imaginary number"
            )

    return NumPyDtypePrinter

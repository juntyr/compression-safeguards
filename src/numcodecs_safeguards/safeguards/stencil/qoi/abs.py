"""
Stencil quantity of interest (QoI) absolute error bound safeguard.
"""

__all__ = ["QuantityOfInterestAbsoluteErrorBoundSafeguard"]

import functools
import re
from itertools import product
from typing import Callable, TypeVar

import numpy as np
import sympy as sp
import sympy.tensor.array.expressions as _
from numpy.lib.stride_tricks import sliding_window_view

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
    as_bits,
    to_finite_float,
    to_float,
)
from ....intervals import Interval, IntervalUnion
from ...pointwise.abs import _compute_safe_eb_diff_interval
from ...pointwise.qoi import Expr
from ...pointwise.qoi.abs import _ensure_bounded_derived_error
from .. import BoundaryCondition, _pad_with_boundary
from ..abc import S, StencilSafeguard, T

Qs = TypeVar("Qs", bound=tuple[int, ...])
Ns = TypeVar("Ns", bound=tuple[int, ...])


class QuantityOfInterestAbsoluteErrorBoundSafeguard(StencilSafeguard):
    """
    The `QuantityOfInterestAbsoluteErrorBoundSafeguard` guarantees that the
    pointwise absolute error on a derived quantity of interest (QoI) over a
    neighbourhood of data points is less than or equal to the provided bound
    `eb_abs`.

    The quantity of interest is specified as a non-constant expression, in
    string form, on the neighbourhood tensor `X` that is centred on the
    pointwise value `x`. For example, to bound the error on the four-neighbour
    box mean in a 3x3 neighbourhood (where `x = X[I]`), set
    `qoi=Expr("(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4")`.
    Note that `X` can be indexed absolute or relative to the centred data point
    `x` using the index array `I`.

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
      | array
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
      | "X"                               (* data neighbourhood *)
    ;

    array   =
        "A", "[", [                       (* n-dimensional array *)
            expr, { ",", expr }, [","]
        ], "]"
    ;

    unary   =
        "(", expr, ")"                    (* parenthesis *)
      | "-", expr                         (* negation *)
      | "sqrt", "(", expr, ")"            (* square root *)
      | "ln", "(", expr, ")"              (* natural logarithm *)
      | "exp", "(", expr, ")"             (* exponential e^x *)
      | "asum", "(", expr, ")"            (* sum over an array *)
    ;

    binary  =
        expr, "+", expr                   (* addition *)
      | expr, "-", expr                   (* subtraction *)
      | expr, "*", expr                   (* multiplication *)
      | expr, "/", expr                   (* division *)
      | expr, "**", expr                  (* exponentiation *)
      | "log", "(", expr, ",", expr, ")"  (* logarithm log(a, base) *)
      | expr, "[", indices "]"            (* array indexing *)
    ;

    indices =
        index, { ",", index }, [","]
    ;

    index   =
        "I"                               (* index of the neighbourhood centre *)
      | expr                              (* index expression *)
      | [expr], ":", [expr]               (* slicing *)
    ;

    ```

    Parameters
    ----------
    qoi : Expr
        The non-constant expression for computing the derived quantity of
        interest for a neighbourhood tensor `X`.
    shape : tuple[tuple[int, int], ...]
        The shape of the data neighbourhood, expressed as (before, after)
        tuples, where before (non-positive) and after (non-negative)
        specify the range of values relative to the data point to include.

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
    _X: sp.tensor.array.expressions.ArraySymbol

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
        I = []
        assert len(shape) > 0, "shape must not be empty"
        for b, a in shape:
            assert type(b) is int, "shape's before must be an integer"
            assert b <= 0, "shape's before must be non-positive"
            assert type(a) is int, "shape's after must be an integer"
            assert a >= 0, "shape's after must be non-negative"
            s.append(-b + 1 + abs(a))
            I.append(-b)
        self._shape = shape

        self._X = sp.tensor.array.expressions.ArraySymbol("X", s)
        X = self._X.as_explicit()
        X.__class__ = _NumPyLikeArray
        x = X.__getitem__(I)

        assert len(axes) == len(shape), (
            "number of axes must match the number of shape dimensions"
        )
        self._axes = axes

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
        assert _QOI_PATTERN.fullmatch(qoi) is not None, "invalid qoi expression"
        try:

            def sqrt(x):
                return x ** sp.Rational(1, 2)

            def exp(x):
                return sp.E**x

            def ln(x):
                if isinstance(x, _NumPyLikeArray):
                    return x.applyfunc(sp.ln)
                return sp.ln(x)

            def log(x, base):
                if isinstance(x, _NumPyLikeArray):
                    ln_x = x.applyfunc(sp.ln)
                else:
                    ln_x = sp.ln(x)
                if isinstance(base, _NumPyLikeArray):
                    ln_base = base.applyfunc(sp.ln)
                else:
                    ln_base = sp.ln(base)
                return ln_x / ln_base

            def asum(x):
                assert isinstance(x, _NumPyLikeArray), (
                    "can only compute the sum over an array"
                )
                return sum(sp.tensor.array.arrayop.Flatten(x), sp.Integer(0))

            qoi_expr = sp.parse_expr(
                self._qoi,
                local_dict=dict(x=x, X=X, I=_NumPyLikeArray(I)),
                global_dict=dict(
                    # literals
                    Integer=sp.Integer,
                    Float=sp.Float,
                    Rational=sp.Rational,
                    # arrays
                    A=_ArrayConstructor,
                    # constants
                    pi=sp.pi,
                    e=sp.E,
                    # operators
                    sqrt=sqrt,
                    exp=exp,
                    ln=ln,
                    log=log,
                    asum=asum,
                ),
                transformations=(sp.parsing.sympy_parser.auto_number,),
            )
            assert isinstance(qoi_expr, sp.Basic), (
                "qoi expression must evaluate to a numeric expression"
            )
            # check if the expression is well-formed (e.g. no int's that cannot
            #  be printed) and if an error bound can be computed
            _canary_repr = str(qoi_expr)
            _canary_eb_abs = _compute_data_eb_for_stencil_qoi_eb(
                qoi_expr,
                self._X,
                np.zeros(s),
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
        """
        Check which elements in the `decoded` array satisfy the absolute error
        bound for the quantity of interest over a neighbourhood on the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Pointwise, `True` if the check succeeded for this element.
        """

        all_axes = []
        for axis in self._axes:
            if axis >= data.ndim or axis < -data.ndim:
                raise IndexError(
                    f"axis index {axis} is out of bounds for array of shape {data.shape}"
                )
            naxis = data.ndim - axis if axis < 0 else axis
            if naxis in all_axes:
                raise IndexError(
                    f"duplicate axis index {axis}, normalised to {naxis}, for array of shape {data.shape}"
                )
            all_axes.append(naxis)

        window = tuple(-b + 1 + a for b, a in self._shape)

        data_boundary = _pad_with_boundary(
            data,
            self._boundary,
            tuple(-b for b, a in self._shape),
            tuple(a for b, a in self._shape),
            self._constant_boundary,
            self._axes,
        )
        decoded_boundary = _pad_with_boundary(
            decoded,
            self._boundary,
            tuple(-b for b, a in self._shape),
            tuple(a for b, a in self._shape),
            self._constant_boundary,
            self._axes,
        )

        data_windows_float: np.ndarray = to_float(
            sliding_window_view(data_boundary, window, axis=self._axes, writeable=False)  # type: ignore
        )
        decoded_windows_float: np.ndarray = to_float(
            sliding_window_view(
                decoded_boundary, window, axis=self._axes, writeable=False
            )  # type: ignore
        )

        qoi_lambda = _compile_sympy_expr_to_numpy(
            [self._X], self._qoi_expr, data_windows_float.dtype
        )

        qoi_data = (qoi_lambda)(data_windows_float)
        qoi_decoded = (qoi_lambda)(to_float(decoded_windows_float))

        absolute_bound = (
            np.where(
                qoi_data > qoi_decoded,
                qoi_data - qoi_decoded,
                qoi_decoded - qoi_data,
            )
            <= self._eb_abs
        )
        same_bits = as_bits(qoi_data, kind="V") == as_bits(qoi_decoded, kind="V")
        both_nan = _isnan(qoi_data) & _isnan(qoi_decoded)

        windows_ok = np.where(
            _isfinite(qoi_data),
            absolute_bound,
            np.where(
                _isinf(qoi_data),
                same_bits,
                both_nan,
            ),
        )

        ok = np.ones_like(data, dtype=np.bool)

        if self._boundary == BoundaryCondition.valid:
            sl = [slice(None)] * data.ndim
            for axis, (b, a) in zip(self._axes, self._shape):
                start = None if b == 0 else -b
                end = None if a == 0 else -a
                sl[axis] = slice(start, end)
            s = tuple(sl)
        else:
            s = tuple([slice(None)] * data.ndim)

        ok = np.ones_like(data, dtype=np.bool)
        ok[s] = windows_ok

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the absolute error bound is upheld with
        respect to the quantity of interest over a neighbourhood on the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the absolute error bound is upheld.
        """

        all_axes = []
        for axis in self._axes:
            if axis >= data.ndim or axis < -data.ndim:
                raise IndexError(
                    f"axis index {axis} is out of bounds for array of shape {data.shape}"
                )
            naxis = data.ndim - axis if axis < 0 else axis
            if naxis in all_axes:
                raise IndexError(
                    f"duplicate axis index {axis}, normalised to {naxis}, for array of shape {data.shape}"
                )
            all_axes.append(naxis)

        window = tuple(-b + 1 + a for b, a in self._shape)

        data_boundary = _pad_with_boundary(
            data,
            self._boundary,
            tuple(-b for b, a in self._shape),
            tuple(a for b, a in self._shape),
            self._constant_boundary,
            self._axes,
        )

        data_windows_float: np.ndarray = to_float(
            sliding_window_view(data_boundary, window, axis=self._axes, writeable=False)  # type: ignore
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs: np.ndarray = to_finite_float(self._eb_abs, data_windows_float.dtype)
        assert eb_abs >= 0

        qoi_lambda = _compile_sympy_expr_to_numpy(
            [self._X], self._qoi_expr, data_windows_float.dtype
        )

        # ensure the error bounds are representable in QoI space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # compute the error-bound adjusted QoIs
            data_qoi = (qoi_lambda)(data_windows_float)
            qoi_lower = data_qoi - eb_abs
            qoi_upper = data_qoi + eb_abs

            # check if they're representable within the error bound
            qoi_lower_outside_eb_abs = (data_qoi - qoi_lower) > eb_abs
            qoi_upper_outside_eb_abs = (qoi_upper - data_qoi) > eb_abs

            # otherwise nudge the error-bound adjusted QoIs
            # we can nudge with nextafter since the QoIs are floating point and
            #  only finite QoIs are nudged
            qoi_lower = np.where(
                qoi_lower_outside_eb_abs & _isfinite(data_qoi),
                _nextafter(qoi_lower, data_qoi),
                qoi_lower,
            )
            qoi_upper = np.where(
                qoi_upper_outside_eb_abs & _isfinite(data_qoi),
                _nextafter(qoi_upper, data_qoi),
                qoi_upper,
            )

            # compute the adjusted error bound
            eb_qoi_lower = _nan_to_zero(qoi_lower - data_qoi)
            eb_qoi_upper = _nan_to_zero(qoi_upper - data_qoi)

        # check that the adjusted error bounds fulfil all requirements
        assert eb_qoi_lower.ndim == data.ndim
        assert eb_qoi_lower.dtype == data_windows_float.dtype
        assert eb_qoi_upper.ndim == data.ndim
        assert eb_qoi_upper.dtype == data_windows_float.dtype
        assert np.all(
            (eb_qoi_lower <= 0) & (eb_qoi_lower >= -eb_abs) & _isfinite(eb_qoi_lower)
        )
        assert np.all(
            (eb_qoi_upper >= 0) & (eb_qoi_upper <= eb_abs) & _isfinite(eb_qoi_upper)
        )

        # if no error bounds are imposed, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, allow all values
        if eb_qoi_lower.size == 0:
            assert data.size == 0 or self._boundary == BoundaryCondition.valid
            return Interval.full_like(data).into_union()  # type: ignore

        if self._boundary == BoundaryCondition.valid:
            sl = [slice(None)] * data.ndim
            for axis, (b, a) in zip(self._axes, self._shape):
                start = None if b == 0 else -b
                end = None if a == 0 else -a
                sl[axis] = slice(start, end)
            s = tuple(sl)
        else:
            s = tuple([slice(None)] * data.ndim)

        data_float: np.ndarray = to_float(data)

        # compute the error bound in data space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_x_lower, eb_x_upper = _compute_data_eb_for_stencil_qoi_eb(
                self._qoi_expr,
                self._X,
                data_windows_float,
                data_float[s],
                eb_qoi_lower,
                eb_qoi_upper,
            )
        assert np.all((eb_x_lower <= 0) & _isfinite(eb_x_lower))
        assert np.all((eb_x_upper >= 0) & _isfinite(eb_x_upper))

        # FIXME: need to account for boundary conditions, adapt from monotonicity
        eb_x_orig_lower = np.full_like(data_float, -np.inf)
        eb_x_orig_upper = np.full_like(data_float, np.inf)
        for offset in product(*[range(-b + a + 1) for b, a in self._shape]):
            sl = [slice(None)] * data.ndim
            for axis, o in zip(self._axes, offset):
                sl[axis] = slice(o)
            eb_x_orig_lower[tuple(sl)] = np.maximum(
                eb_x_orig_lower[tuple(sl)], eb_x_lower
            )
            eb_x_orig_upper[tuple(sl)] = np.minimum(
                eb_x_upper, eb_x_orig_upper[tuple(sl)]
            )
        assert np.all((eb_x_orig_lower <= 0) & _isfinite(eb_x_orig_lower))
        assert np.all((eb_x_orig_upper >= 0) & _isfinite(eb_x_orig_upper))

        return _compute_safe_eb_diff_interval(
            data,
            data_float,
            eb_x_orig_lower,
            eb_x_orig_upper,
            equal_nan=True,
        ).into_union()  # type: ignore

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
def _compute_data_eb_for_stencil_qoi_eb(
    expr: sp.Basic,
    X: sp.tensor.array.expressions.ArraySymbol,
    XvN: np.ndarray[tuple[int, ...], F],  # np.ndarray[tuple[*Qs, *Ns], F],
    Xv: np.ndarray[Qs, F],
    tauv_lower: np.ndarray[Qs, F],
    tauv_upper: np.ndarray[Qs, F],
) -> tuple[np.ndarray[Qs, F], np.ndarray[Qs, F]]:
    """
    Translate an error bound on a derived quantity of interest (QoI) into an
    error bound on the input data.

    This function checks the computed error bound before returning to correct
    any rounding errors.

    Parameters
    ----------
    expr : sp.Basic
        Symbolic SymPy expression that defines the QoI.
    X : sp.tensor.array.expressions.ArraySymbol
        Symbol for the input data neighbourhood.
    XvN : np.ndarray[tuple[*Qs, *Ns], F]
        Actual values of the input data, with the neighbourhood on the last axes.
    Xv : np.ndarray[Qs, F]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[Qs, F]
        Finite pointwise lower bound on the QoI error, must be negative or zero.
    eb_expr_upper : np.ndarray[Qs, F]
        Finite pointwise upper bound on the QoI error, must be positive or zero.

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[Qs, F], np.ndarray[Qs, F]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    tl, tu = _compute_data_eb_for_stencil_qoi_eb_unchecked(
        expr, X, XvN, Xv, tauv_lower, tauv_upper
    )

    exprl = _compile_sympy_expr_to_numpy([X], expr, Xv.dtype)
    exprv = (exprl)(XvN)

    # handle rounding errors in the lower error bound computation
    tl = _ensure_bounded_derived_error(
        # tl has shape Qs and has XvN (*Qs, *Ns), so their sum has (*Qs, *Ns)
        #  and evaluating the expression brings us back to Qs
        lambda tl: np.where(
            tl == 0,
            exprv,
            (exprl)(XvN + tl.reshape(list(tl.shape) + [1] * (XvN.ndim - tl.ndim))),
        ),  # type: ignore
        exprv,
        Xv,
        tl,
        tauv_lower,
        tauv_upper,
    )
    tu = _ensure_bounded_derived_error(
        # tu has shape Qs and has XvN (*Qs, *Ns), so their sum has (*Qs, *Ns)
        #  and evaluating the expression brings us back to Qs
        lambda tu: np.where(
            tu == 0,
            exprv,
            (exprl)(XvN + tu.reshape(list(tu.shape) + [1] * (XvN.ndim - tu.ndim))),
        ),  # type: ignore
        exprv,
        Xv,
        tu,
        tauv_lower,
        tauv_upper,
    )

    return tl, tu


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def _compute_data_eb_for_stencil_qoi_eb_unchecked(
    expr: sp.Basic,
    X: sp.tensor.array.expressions.ArraySymbol,
    XvN: np.ndarray[tuple[int, ...], F],  # np.ndarray[tuple[*Qs, *Ns], F],
    Xv: np.ndarray[Qs, F],
    eb_expr_lower: np.ndarray[Qs, F],
    eb_expr_upper: np.ndarray[Qs, F],
) -> tuple[np.ndarray[Qs, F], np.ndarray[Qs, F]]:
    """
    Translate an error bound on a derived quantity of interest (QoI) into an
    error bound on the input data.

    This function does not check the returned error bound on the input data,
    use `_compute_data_eb_for_qoi_eb` instead.

    Parameters
    ----------
    expr : sp.Basic
        Symbolic SymPy expression that defines the QoI.
    X : sp.tensor.array.expressions.ArraySymbol
        Symbol for the input data neighbourhood.
    XvN : np.ndarray[tuple[*Qs, *Ns], F]
        Actual values of the input data, with the neighbourhood on the last axes.
    Xv : np.ndarray[Qs, F]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[Qs, F]
        Finite pointwise lower bound on the QoI error, must be negative or zero.
    eb_expr_upper : np.ndarray[Qs, F]
        Finite pointwise upper bound on the QoI error, must be positive or zero.

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[Qs, F], np.ndarray[Qs, F]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    assert len(expr.free_symbols) > 0, "constants have no error bounds"

    zero = np.array(0, dtype=Xv.dtype)

    # X[...]
    if (
        expr.func is sp.tensor.array.expressions.ArrayElement
        and len(expr.args) == 2
        and expr.args[0] == X
    ):
        return (eb_expr_lower, eb_expr_upper)

    # array
    if expr.func in (sp.Array, _NumPyLikeArray):
        raise ValueError("expression must evaluate to a scalar not an array")

    # abs(...) is only used internally in exp(ln(abs(...)))
    if expr.func is sp.Abs and len(expr.args) == 1:
        # evaluate arg
        (arg,) = expr.args
        argv = _compile_sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
        # flip the lower/upper error bound if the arg is negative
        eql = np.where(argv < 0, -eb_expr_upper, eb_expr_lower)
        equ = np.where(argv < 0, -eb_expr_lower, eb_expr_upper)
        return _compute_data_eb_for_stencil_qoi_eb(arg, X, XvN, Xv, eql, equ)  # type: ignore

    # ln(...)
    # sympy automatically transforms log(..., base) into ln(...)/ln(base)
    if expr.func is sp.log and len(expr.args) == 1:
        # evaluate arg and ln(arg)
        (arg,) = expr.args
        argv = _compile_sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
        exprv = np.log(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.exp(exprv + eb_expr_lower) - argv,
        )
        eal = _nan_to_zero(to_finite_float(eal, Xv.dtype))

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.exp(exprv + eb_expr_upper) - argv,
        )
        eau = _nan_to_zero(to_finite_float(eau, Xv.dtype))

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
        return _compute_data_eb_for_stencil_qoi_eb(
            arg,
            X,
            XvN,
            Xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # e^(...)
    if expr.func is sp.exp and len(expr.args) == 1:
        # evaluate arg and e^arg
        (arg,) = expr.args
        argv = _compile_sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
        exprv = np.exp(argv)

        # update the error bounds
        # ensure that ln is not passed a negative argument
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.log(np.maximum(zero, exprv + eb_expr_lower)) - argv,
        )
        eal = _nan_to_zero(to_finite_float(eal, Xv.dtype))

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.log(np.maximum(zero, exprv + eb_expr_upper)) - argv,
        )
        eau = _nan_to_zero(to_finite_float(eau, Xv.dtype))

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
        return _compute_data_eb_for_stencil_qoi_eb(
            arg,
            X,
            XvN,
            Xv,
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
        )

    # rewrite a ** b as e^(b*ln(abs(a)))
    # this is mathematically incorrect for a <= 0 but works for deriving error bounds
    if expr.is_Pow and len(expr.args) == 2:
        a, b = expr.args
        return _compute_data_eb_for_stencil_qoi_eb(
            sp.exp(b * sp.ln(sp.Abs(a)), evaluate=False),
            X,
            XvN,
            Xv,
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
                        Xv.dtype,
                    )()
                )
                terms[i] = sp.Mul(
                    *[arg for arg in term.args if len(arg.free_symbols) > 0]  # type: ignore
                )
            else:
                factors.append(np.array(1))
        total_abs_factor = np.sum(np.abs(factors))

        etl: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_lower / total_abs_factor, Xv.dtype)
        )
        etu: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_upper / total_abs_factor, Xv.dtype)
        )

        # handle rounding errors in the total absolute factor early
        etl = _ensure_bounded_derived_error(
            lambda etl: etl * total_abs_factor,
            np.zeros_like(Xv),
            None,
            etl,
            eb_expr_lower,
            eb_expr_upper,
        )
        etu = _ensure_bounded_derived_error(
            lambda etu: etu * total_abs_factor,
            np.zeros_like(Xv),
            None,
            etu,
            eb_expr_lower,
            eb_expr_upper,
        )

        eb_x_lower, eb_x_upper = None, None
        for term, factor in zip(terms, factors):
            # recurse into the terms with a weighted error bound
            exl, exu = _compute_data_eb_for_stencil_qoi_eb(
                term,
                X,
                XvN,
                Xv,
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
            Xv.dtype,
        )()

        efl: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_lower / np.abs(factor), Xv.dtype)
        )
        efu: np.ndarray = _nan_to_zero(
            to_finite_float(eb_expr_upper / np.abs(factor), Xv.dtype)
        )

        # handle rounding errors in the factor early
        efl = _ensure_bounded_derived_error(
            lambda efl: efl * np.abs(factor),
            np.zeros_like(Xv),
            None,
            efl,
            eb_expr_lower,
            eb_expr_upper,
        )
        efu = _ensure_bounded_derived_error(
            lambda efu: efu * np.abs(factor),
            np.zeros_like(Xv),
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
            return _compute_data_eb_for_stencil_qoi_eb(
                terms[0], X, XvN, Xv, eb_factor_lower, eb_factor_upper
            )

        return _compute_data_eb_for_stencil_qoi_eb(
            sp.exp(sp.Add(*[sp.log(sp.Abs(term)) for term in terms]), evaluate=False),
            X,
            XvN,
            Xv,
            eb_factor_lower,
            eb_factor_upper,
        )

    raise ValueError(f"unsupported expression kind {expr} (= {sp.srepr(expr)} =)")


def _compile_sympy_expr_to_numpy(
    symbols: list[sp.tensor.array.expressions.ArraySymbol],
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

        def _print_ArrayElement(self, expr):
            return (
                f"{expr.name}[..., {', '.join([self._print(i) for i in expr.indices])}]"
            )

    return NumPyDtypePrinter


class _NumPyLikeArray(sp.Array):
    __slots__ = ()

    def __add__(self, other):
        if isinstance(other, _NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i + j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i + other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __radd__(self, other):
        other = sp.sympify(other)
        result_list = [other + i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __sub__(self, other):
        if isinstance(other, _NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i - j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i - other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rsub__(self, other):
        other = sp.sympify(other)
        result_list = [other - i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __mul__(self, other):
        if isinstance(other, _NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i * j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i * other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rmul__(self, other):
        other = sp.sympify(other)
        result_list = [other * i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __truediv__(self, other):
        if isinstance(other, _NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i / j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i / other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rtruediv__(self, other):
        other = sp.sympify(other)
        result_list = [other / i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __pow__(self, other):
        if isinstance(other, _NumPyLikeArray):
            if self.shape != other.shape:
                raise ValueError("array shape mismatch")
            result_list = [
                i**j
                for i, j in zip(
                    sp.tensor.array.arrayop.Flatten(self),
                    sp.tensor.array.arrayop.Flatten(other),
                )
            ]
            return type(self)(result_list, self.shape)
        other = sp.sympify(other)
        result_list = [i**other for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rpow__(self, other):
        other = sp.sympify(other)
        result_list = [other**i for i in sp.tensor.array.arrayop.Flatten(self)]
        return type(self)(result_list, self.shape)

    # TODO: also support log
    # TODO: also support "matrix" multiplication


class _ArrayConstructor:
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("cannot call array constructor")

    def __class_getitem__(cls, index):
        return _NumPyLikeArray(index)


# pattern of syntactically weakly valid expressions
# we only check against forbidden tokens, not for semantic validity
#  i.e. just enough that it's safe to eval afterwards
_QOI_PATTERN = re.compile(
    r"(?:"
    r"(?:"
    r"(?:[0-9]+)"
    r"|(?:[0-9]+\.[0-9]+)"
    r"|(?:e)"
    r"|(?:pi)"
    r"|(?:x)"
    r"|(?:X)"
    r"|(?:I)"
    r"|(?:A)"
    r"|(?:sqrt)"
    r"|(?:ln)"
    r"|(?:log)"
    r"|(?:exp)"
    r"|(?:asum)"
    r")?"
    r"|(?:[ \t\n\(\)\[\],:\+\-\*/])"
    r")*"
)

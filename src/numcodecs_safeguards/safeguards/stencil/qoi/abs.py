"""
Stencil quantity of interest (QoI) absolute error bound safeguard.
"""

__all__ = ["StencilQuantityOfInterestAbsoluteErrorBoundSafeguard"]

import re
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import sympy as sp
import sympy.tensor.array.expressions  # noqa: F401
from numpy.lib.stride_tricks import sliding_window_view

from ....cast import (
    F,
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
from ... import _qois
from ...pointwise.abs import _compute_safe_eb_diff_interval
from ...pointwise.qoi.abs import _ensure_bounded_derived_error
from .. import (
    BoundaryCondition,
    NeighbourhoodAxis,
    NeighbourhoodBoundaryAxis,
    _pad_with_boundary,
)
from ..abc import S, StencilSafeguard, T
from . import StencilExpr

Qs = TypeVar("Qs", bound=tuple[int, ...])
Ns = TypeVar("Ns", bound=tuple[int, ...])


class StencilQuantityOfInterestAbsoluteErrorBoundSafeguard(StencilSafeguard):
    """
    The `StencilQuantityOfInterestAbsoluteErrorBoundSafeguard` guarantees that
    the pointwise absolute error on a derived quantity of interest (QoI) over a
    neighbourhood of data points is less than or equal to the provided bound
    `eb_abs`.

    The quantity of interest is specified as a non-constant expression, in
    string form, on the neighbourhood tensor `X` that is centred on the
    pointwise value `x`. For example, to bound the error on the four-neighbour
    box mean in a 3x3 neighbourhood (where `x = X[I]`), set
    `qoi="(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4"`.
    Note that `X` can be indexed absolute or relative to the centred data point
    `x` using the index array `I`.

    The stencil QoI safeguard can also be used to bound the pointwise absolute
    error of the finite-difference-approximated derivative over the data.

    If the derived quantity of interest for an element evaluates to an infinite
    value, this safeguard guarantees that the quantity of interest on the
    decoded value produces the exact same infinite value. For a NaN quantity of
    interest, this safeguard guarantees that the quantity of interest on the
    decoded value is also NaN, but does not guarantee that it has the same
    bit pattern.

    The qoi expression is written using the following EBNF grammar[^1] for
    `expr`:

    [^1]: You can visualise the EBNF grammar at <https://matthijsgroen.github.io/ebnf2railroad/try-yourself.html>.

    ```ebnf
    expr    =
        literal
      | const
      | var
      | array
      | unary
      | binary
      | findiff
    ;

    literal =
        int
      | float
    ;

    int     =                             (* integer literal *)
        [ sign ], digit, { digit }
    ;
    float   =                             (* floating point literal *)
        [ sign ], digit, { digit }, ".", digit, { digit }, [
            "e", [ sign ], digit, { digit }
        ]
    ;

    sign    =
        "+" | "-"
    ;
    digit   =
        "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    ;

    const   =
        "e"                               (* Euler's number *)
      | "pi"                              (* pi *)
    ;

    var     =
        "x"                               (* pointwise data value *)
      | "X"                               (* data neighbourhood *)
    ;

    array   =
        "A", "[", [
            expr, { ",", expr }, [","]    (* n-dimensional array *)
        ], "]"
    ;

    unary   =
        "(", expr, ")"                    (* parenthesis *)
      | "-", expr                         (* negation *)
      | "sqrt", "(", expr, ")"            (* square root *)
      | "ln", "(", expr, ")"              (* natural logarithm *)
      | "exp", "(", expr, ")"             (* exponential e^x *)
      | "asum", "(", expr, ")"            (* sum over an array *)
      | "tr", "(", expr, ")"              (* transpose of a matrix (2d array) *)
      | "sin", "(", expr, ")"             (* sine sin(x) *)
      | "cos", "(", expr, ")"             (* cosine cos(x) *)
      | "tan", "(", expr, ")"             (* tangent tan(x) *)
      | "cot", "(", expr, ")"             (* cotangent cot(x) *)
      | "sec", "(", expr, ")"             (* secant sec(x) *)
      | "csc", "(", expr, ")"             (* cosecant csc(x) *)
      | "asin", "(", expr, ")"            (* inverse sine asin(x) *)
      | "acos", "(", expr, ")"            (* inverse cosine acos(x) *)
      | "atan", "(", expr, ")"            (* inverse tangent atan(x) *)
      | "acot", "(", expr, ")"            (* inverse cotangent acot(x) *)
      | "asec", "(", expr, ")"            (* inverse secant asec(x) *)
      | "acsc", "(", expr, ")"            (* inverse cosecant acsc(x) *)
      | "sinh", "(", expr, ")"            (* hyperbolic sine sinh(x) *)
      | "cosh", "(", expr, ")"            (* hyperbolic cosine cosh(x) *)
      | "tanh", "(", expr, ")"            (* hyperbolic tangent tanh(x) *)
      | "coth", "(", expr, ")"            (* hyperbolic cotangent coth(x) *)
      | "sech", "(", expr, ")"            (* hyperbolic secant sech(x) *)
      | "csch", "(", expr, ")"            (* hyperbolic cosecant csch(x) *)
      | "asinh", "(", expr, ")"           (* inverse hyperbolic sine asinh(x) *)
      | "acosh", "(", expr, ")"           (* inverse hyperbolic cosine acosh(x) *)
      | "atanh", "(", expr, ")"           (* inverse hyperbolic tangent atanh(x) *)
      | "acoth", "(", expr, ")"           (* inverse hyperbolic cotangent acoth(x) *)
      | "asech", "(", expr, ")"           (* inverse hyperbolic secant asech(x) *)
      | "acsch", "(", expr, ")"           (* inverse hyperbolic cosecant acsch(x) *)
    ;

    binary  =
        expr, "+", expr                   (* addition *)
      | expr, "-", expr                   (* subtraction *)
      | expr, "*", expr                   (* multiplication *)
      | expr, "/", expr                   (* division *)
      | expr, "**", expr                  (* exponentiation *)
      | "log", "(",
            expr, ","                     (* logarithm with explicit base *)
          , "base", "=", expr,
        ")"
      | expr, "[", indices, "]"           (* array indexing *)
      | "matmul", "("                     (* matrix (2d array) multiplication *)
          , expr, ",", expr,
        ")"
    ;

    indices =
        index, { ",", index }, [","]
    ;

    index   =
        "I"                               (* index of the neighbourhood centre *)
      | expr                              (* index expression *)
      | [expr], ":", [expr]               (* slicing *)
    ;

    findiff =
        "findiff", "("                    (* finite difference over an expression *)
          , expr, ","
          , "order", "=", int, ","           (* order of the derivative *)
          , "accuracy", "=", int, ","        (* order of accuracy of the approximation *)
          , "type", "=", (
                "-1" | "0" | "1"             (* backwards | central | forward difference *)
            ), ","
          , "dx", "=", ( int | float ), ","  (* uniform grid spacing *)
          , "axis", "=", int                 (* axis, relative to the neighbourhood *)
      , ")"
    ;

    ```

    Parameters
    ----------
    qoi : Expr
        The non-constant expression for computing the derived quantity of
        interest for a neighbourhood tensor `X`.
    neighbourhood : Sequence[dict | NeighbourhoodAxis]
        The non-empty axes of the data neighbourhood over which the quantity of
        interest is computed. The neighbourhood window is applied independently
        over any additional axes in the data.
    eb_abs : int | float
        The non-negative absolute error bound on the quantity of interest that
        is enforced by this safeguard.
    """

    __slots__ = (
        "_qoi",
        "_neighbourhood",
        "_eb_abs",
        "_qoi_expr",
        "_X",
    )
    _qoi: StencilExpr
    _neighbourhood: tuple[NeighbourhoodBoundaryAxis, ...]
    _eb_abs: int | float
    _qoi_expr: sp.Basic
    _X: sp.tensor.array.expressions.ArraySymbol

    kind = "qoi_abs_stencil"

    def __init__(
        self,
        qoi: StencilExpr,
        neighbourhood: Sequence[dict | NeighbourhoodBoundaryAxis],
        eb_abs: int | float,
    ):
        self._neighbourhood = tuple(
            axis
            if isinstance(axis, NeighbourhoodBoundaryAxis)
            else NeighbourhoodBoundaryAxis.from_config(axis)
            for axis in neighbourhood
        )
        assert len(self._neighbourhood) > 0, "neighbourhood must not be empty"
        assert len(set(axis.axis for axis in self._neighbourhood)) == len(
            self._neighbourhood
        ), "neighbourhood axes must be unique"

        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert isinstance(eb_abs, int) or _isfinite(eb_abs), "eb_abs must be finite"
        self._eb_abs = eb_abs

        shapel, Il = [], []  # noqa: E741
        for axis in self._neighbourhood:
            shapel.append(axis.before + 1 + axis.after)
            Il.append(axis.before)
        shape, I = tuple(shapel), tuple(Il)  # noqa: E741

        self._X = sp.tensor.array.expressions.ArraySymbol("X", shape)
        X = self._X.as_explicit()
        X.__class__ = _qois.array.NumPyLikeArray

        assert len(qoi.strip()) > 0, "qoi expression must not be empty"
        assert _QOI_PATTERN.fullmatch(qoi) is not None, "invalid qoi expression"
        try:
            qoi_expr = sp.parse_expr(
                qoi,
                local_dict=dict(
                    # === data ===
                    # data neighbourhood
                    X=X,
                    x=X.__getitem__(I),
                    # neighbourhood index
                    I=_qois.array.NumPyLikeArray(I),
                    # === constants ===
                    **_qois.math.CONSTANTS,
                    # === operators ===
                    # poinwise math
                    **_qois.math.FUNCTIONS,
                    # array math
                    **_qois.amath.CONSTRUCTORS,
                    **_qois.amath.FUNCTIONS,
                    # finite difference
                    findiff=_qois.findiff.create_findiff_for_neighbourhood(
                        self._X, shape, I
                    ),
                ),
                global_dict=dict(
                    # literals
                    Integer=sp.Integer,
                    Float=sp.Float,
                    Rational=sp.Rational,
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
                np.zeros(shape),
                np.zeros(shape),
                np.zeros(shape),
                np.zeros(shape),
            )
        except Exception as err:
            raise AssertionError(
                f"failed to parse qoi expression {qoi!r}: {err}"
            ) from err
        assert len(qoi_expr.free_symbols) > 0, "qoi expression must not be constant"
        assert not qoi_expr.has(sp.I), (
            "qoi expression must not contain imaginary numbers"
        )

        self._qoi = qoi
        self._qoi_expr = qoi_expr

    def compute_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[None | NeighbourhoodAxis, ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        [`None`][None] is returned along dimensions for which there is no data
        neighbourhood.

        This method also checks that the data shape is compatible with this
        stencil safeguard.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the data.

        Returns
        -------
        neighbourhood_shape : tuple[None | NeighbourhoodAxis, ...]
            The shape of the data neighbourhood.
        """

        neighbourhood: list[None | NeighbourhoodAxis] = [None] * len(data_shape)

        all_axes = []
        for axis in self._neighbourhood:
            if (axis.axis >= len(data_shape)) or (axis.axis < -len(data_shape)):
                raise IndexError(
                    f"axis index {axis.axis} is out of bounds for array of shape {data_shape}"
                )
            naxis = len(data_shape) - axis.axis if axis.axis < 0 else axis.axis
            if naxis in all_axes:
                raise IndexError(
                    f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {data_shape}"
                )
            all_axes.append(naxis)

            neighbourhood[naxis] = axis.shape

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape,
        # return an empty neighbourhood
        for axis, w in zip(self._neighbourhood, window):
            if data_shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return (None,) * len(data_shape)
        if np.prod(data_shape) == 0:
            return (None,) * len(data_shape)

        return tuple(neighbourhood)

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
        for axis in self._neighbourhood:
            if (axis.axis >= data.ndim) or (axis.axis < -data.ndim):
                raise IndexError(
                    f"axis index {axis.axis} is out of bounds for array of shape {data.shape}"
                )
            naxis = data.ndim - axis.axis if axis.axis < 0 else axis.axis
            if naxis in all_axes:
                raise IndexError(
                    f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {data.shape}"
                )
            all_axes.append(naxis)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, allow all values
        for axis, w in zip(self._neighbourhood, window):
            if data.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return np.ones_like(data, dtype=np.bool)  # type: ignore
        if data.size == 0:
            return np.ones_like(data, dtype=np.bool)  # type: ignore

        data_boundary, decoded_boundary = data, decoded
        for axis in self._neighbourhood:
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis.constant_boundary,
                axis.axis,
            )
            decoded_boundary = _pad_with_boundary(
                decoded_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis.constant_boundary,
                axis.axis,
            )

        data_windows_float: np.ndarray = to_float(
            sliding_window_view(
                data_boundary,
                window,
                axis=tuple(axis.axis for axis in self._neighbourhood),
                writeable=False,
            )  # type: ignore
        )
        decoded_windows_float: np.ndarray = to_float(
            sliding_window_view(
                decoded_boundary,
                window,
                axis=tuple(axis.axis for axis in self._neighbourhood),
                writeable=False,
            )  # type: ignore
        )

        qoi_lambda = _qois.compile.sympy_expr_to_numpy(
            [self._X], self._qoi_expr, data_windows_float.dtype
        )

        qoi_data = (qoi_lambda)(data_windows_float)
        qoi_decoded = (qoi_lambda)(decoded_windows_float)

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

        s = [slice(None)] * data.ndim
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                start = None if axis.before == 0 else axis.before
                end = None if axis.after == 0 else -axis.after
                s[axis.axis] = slice(start, end)

        ok = np.ones_like(data, dtype=np.bool)
        ok[tuple(s)] = windows_ok

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
        for axis in self._neighbourhood:
            if (axis.axis >= data.ndim) or (axis.axis < -data.ndim):
                raise IndexError(
                    f"axis index {axis.axis} is out of bounds for array of shape {data.shape}"
                )
            naxis = data.ndim - axis.axis if axis.axis < 0 else axis.axis
            if naxis in all_axes:
                raise IndexError(
                    f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {data.shape}"
                )
            all_axes.append(naxis)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, allow all values
        for axis, w in zip(self._neighbourhood, window):
            if data.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return Interval.full_like(data).into_union()  # type: ignore
        if data.size == 0:
            return Interval.full_like(data).into_union()  # type: ignore

        data_boundary = data
        for axis in self._neighbourhood:
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis.constant_boundary,
                axis.axis,
            )

        data_windows_float: np.ndarray = to_float(
            sliding_window_view(
                data_boundary,
                window,
                axis=tuple(axis.axis for axis in self._neighbourhood),
                writeable=False,
            )  # type: ignore
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs: np.ndarray = to_finite_float(self._eb_abs, data_windows_float.dtype)
        assert eb_abs >= 0

        qoi_lambda = _qois.compile.sympy_expr_to_numpy(
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

        s = [slice(None)] * data.ndim
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                start = None if axis.before == 0 else axis.before
                end = None if axis.after == 0 else -axis.after
                s[axis.axis] = slice(start, end)

        data_float: np.ndarray = to_float(data)

        # compute the error bound in data space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_x_lower, eb_x_upper = _compute_data_eb_for_stencil_qoi_eb(
                self._qoi_expr,
                self._X,
                data_windows_float,
                data_float[tuple(s)],
                eb_qoi_lower,
                eb_qoi_upper,
            )
        assert np.all((eb_x_lower <= 0) & _isfinite(eb_x_lower))
        assert np.all((eb_x_upper >= 0) & _isfinite(eb_x_upper))

        # compute how the data indices are distributed into windows
        # i.e. for each qoi element, which data does it depend on
        indices_boundary = np.arange(data.size).reshape(data.shape)
        for axis in self._neighbourhood:
            indices_boundary = _pad_with_boundary(
                indices_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                None if axis.constant_boundary is None else data.size,
                axis.axis,
            )
        indices_windows = sliding_window_view(  # type: ignore
            indices_boundary,
            window,
            axis=tuple(axis.axis for axis in self._neighbourhood),
            writeable=False,
        ).reshape((-1, np.prod(window)))

        # compute the reverse: for each data element, which windows is it in
        # i.e. for each data element, which qoi elements does it contribute to
        #      and thus which error bounds affect it
        reverse_indices_windows = np.full(
            (data.size, np.prod(window)), indices_windows.shape[0]
        )
        reverse_indices_counter = np.zeros(data.size, dtype=int)
        for i in range(np.prod(window)):
            # manual loop to account for potential aliasing:
            # with a wrapping boundary, more than one j for the same window
            # position j could refer back to the same data element
            for j in range(indices_windows.shape[0]):
                idx = indices_windows[j, i]
                if idx != data.size:
                    # lazily allocate more to account for all possible edge cases
                    if reverse_indices_counter[idx] >= reverse_indices_windows.shape[1]:
                        new_reverse_indices_windows = np.full(
                            (data.size, reverse_indices_windows.shape[1] * 2),
                            indices_windows.shape[0],
                        )
                        new_reverse_indices_windows[
                            :, : reverse_indices_windows.shape[1]
                        ] = reverse_indices_windows
                        reverse_indices_windows = new_reverse_indices_windows
                    # update the reverse mapping
                    reverse_indices_windows[idx][reverse_indices_counter[idx]] = j
                    reverse_indices_counter[idx] += 1

        # flatten the qoi error bounds and append an infinite value,
        # which is indexed if an element did not contribute to the maximum
        # number of windows
        with np.errstate(invalid="ignore"):
            eb_x_lower_flat = np.full(eb_x_lower.size + 1, -np.inf, data_float.dtype)
            eb_x_lower_flat[:-1] = eb_x_lower.flatten()
            eb_x_upper_flat = np.full(eb_x_upper.size + 1, np.inf, data_float.dtype)
            eb_x_upper_flat[:-1] = eb_x_upper.flatten()

        # for each data element, reduce over the error bounds that affect it
        eb_x_orig_lower = np.amax(
            eb_x_lower_flat[reverse_indices_windows], axis=1
        ).reshape(data.shape)
        eb_x_orig_upper = np.amin(
            eb_x_upper_flat[reverse_indices_windows], axis=1
        ).reshape(data.shape)
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

        return dict(
            kind=type(self).kind,
            qoi=self._qoi,
            neighbourhood=[axis.get_config() for axis in self._neighbourhood],
            eb_abs=self._eb_abs,
        )


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

    exprl = _qois.compile.sympy_expr_to_numpy([X], expr, Xv.dtype)
    exprv = (exprl)(XvN)

    # handle rounding errors in the lower error bound computation
    tl = _ensure_bounded_derived_error(
        # tl has shape Qs and has XvN (*Qs, *Ns), so their sum has (*Qs, *Ns)
        #  and evaluating the expression brings us back to Qs
        lambda tl: np.where(  # type: ignore
            tl == 0,
            exprv,
            (exprl)(XvN + tl.reshape(list(tl.shape) + [1] * (XvN.ndim - tl.ndim))),
        ),
        exprv,
        Xv,
        tl,
        tauv_lower,
        tauv_upper,
    )
    tu = _ensure_bounded_derived_error(
        # tu has shape Qs and has XvN (*Qs, *Ns), so their sum has (*Qs, *Ns)
        #  and evaluating the expression brings us back to Qs
        lambda tu: np.where(  # type: ignore
            tu == 0,
            exprv,
            (exprl)(XvN + tu.reshape(list(tu.shape) + [1] * (XvN.ndim - tu.ndim))),
        ),
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
    if expr.func in (sp.Array, _qois.array.NumPyLikeArray):
        raise ValueError("expression must evaluate to a scalar not an array")

    # abs(...) is only used internally in exp(ln(abs(...)))
    if expr.func is sp.Abs and len(expr.args) == 1:
        # evaluate arg
        (arg,) = expr.args
        argv = _qois.compile.sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
        # flip the lower/upper error bound if the arg is negative
        eql = np.where(argv < 0, -eb_expr_upper, eb_expr_lower)
        equ = np.where(argv < 0, -eb_expr_lower, eb_expr_upper)
        return _compute_data_eb_for_stencil_qoi_eb(arg, X, XvN, Xv, eql, equ)  # type: ignore

    # ln(...)
    # sympy automatically transforms log(..., base) into ln(...)/ln(base)
    if expr.func is sp.log and len(expr.args) == 1:
        # evaluate arg and ln(arg)
        (arg,) = expr.args
        argv = _qois.compile.sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
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
        argv = _qois.compile.sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
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

    # sin(...)
    if expr.func is sp.sin and len(expr.args) == 1:
        # evaluate arg and sin(arg)
        (arg,) = expr.args
        argv = _qois.compile.sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
        exprv = np.sin(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            # we need to compare to asin(sin(...)) instead of ... to account
            #  for asin's output domain
            np.asin(np.maximum(-1, exprv + eb_expr_lower)) - np.asin(exprv),
        )
        eal = _nan_to_zero(to_finite_float(eal, Xv.dtype))

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.asin(np.minimum(exprv + eb_expr_upper, 1)) - np.asin(exprv),
        )
        eau = _nan_to_zero(to_finite_float(eau, Xv.dtype))

        # np.asin maps to [-pi/2, +pi/2] where sin is monotonically increasing
        # flip the argument error bounds where sin is monotonically decreasing
        eal, eau = (
            np.where(np.sin(argv + eal) > exprv, -eau, eal),
            np.where(np.sin(argv + eau) < exprv, -eal, eau),
        )

        # handle rounding errors in sin(asin(...)) early
        eal = _ensure_bounded_derived_error(
            lambda eal: np.sin(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = _ensure_bounded_derived_error(
            lambda eau: np.sin(argv + eau),
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

    # asin(...)
    if expr.func is sp.asin and len(expr.args) == 1:
        # evaluate arg and asin(arg)
        (arg,) = expr.args
        argv = _qois.compile.sympy_expr_to_numpy([X], arg, Xv.dtype)(XvN)
        exprv = np.asin(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            # np.sin(max(-np.pi/2, ...)) might not be precise, so explicitly
            #  bound lower bounds to be <= 0
            np.minimum(np.sin(np.maximum(-np.pi / 2, exprv + eb_expr_lower)) - argv, 0),
        )
        eal = _nan_to_zero(to_finite_float(eal, Xv.dtype))

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            # np.sin(min(..., np.pi/2)) might not be precise, so explicitly
            #  bound upper bounds to be >= 0
            np.maximum(0, np.sin(np.minimum(exprv + eb_expr_upper, np.pi / 2)) - argv),
        )
        eau = _nan_to_zero(to_finite_float(eau, Xv.dtype))

        # handle rounding errors in asin(sin(...)) early
        eal = _ensure_bounded_derived_error(
            lambda eal: np.asin(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = _ensure_bounded_derived_error(
            lambda eau: np.asin(argv + eau),
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

    TRIGONOMETRIC = {
        # derived trigonometric functions
        sp.cos: lambda x: sp.sin(x + (sp.pi / 2), evaluate=False),
        sp.tan: lambda x: sp.sin(x) / sp.cos(x),
        sp.csc: lambda x: 1 / sp.sin(x),
        sp.sec: lambda x: 1 / sp.cos(x),
        sp.cot: lambda x: sp.cos(x) / sp.sin(x),
        # inverse trigonometric functions
        sp.acos: lambda x: (sp.pi / 2) - sp.asin(x),
        sp.atan: lambda x: ((sp.pi / 2) - sp.asin(1 / sp.sqrt(x**2 + 1)))
        * (sp.Abs(x) / x),
        sp.acsc: lambda x: sp.asin(1 / x),
        sp.asec: lambda x: sp.acos(1 / x),
        sp.acot: lambda x: sp.atan(1 / x),
    }

    # rewrite derived trigonometric functions using sin
    if expr.func in TRIGONOMETRIC and len(expr.args) == 1:
        (arg,) = expr.args
        return _compute_data_eb_for_stencil_qoi_eb(
            (TRIGONOMETRIC[expr.func])(arg),
            X,
            XvN,
            Xv,
            eb_expr_lower,
            eb_expr_upper,
        )

    HYPERBOLIC = {
        # basic hyperbolic functions
        sp.sinh: lambda x: (sp.exp(x) - sp.exp(-x)) / 2,
        sp.cosh: lambda x: (sp.exp(x) + sp.exp(-x)) / 2,
        # derived hyperbolic functions
        sp.tanh: lambda x: (sp.exp(x * 2) - 1) / (sp.exp(x * 2) + 1),
        sp.csch: lambda x: 2 / (sp.exp(x) - sp.exp(-x)),
        sp.sech: lambda x: 2 / (sp.exp(x) + sp.exp(-x)),
        sp.coth: lambda x: (sp.exp(x * 2) + 1) / (sp.exp(x * 2) - 1),
        # inverse hyperbolic functions
        sp.asinh: lambda x: sp.ln(x + sp.sqrt(x**2 + 1)),
        sp.acosh: lambda x: sp.ln(x + sp.sqrt(x**2 - 1)),
        sp.atanh: lambda x: sp.ln((1 + x) / (1 - x)) / 2,
        sp.acsch: lambda x: sp.ln((1 / x) + sp.sqrt(x ** (-2) + 1)),
        sp.asech: lambda x: sp.ln((1 + sp.sqrt(1 - x**2)) / x),
        sp.acoth: lambda x: sp.ln((x + 1) / (x - 1)) / 2,
    }

    # rewrite hyperbolic functions using their exponential definitions
    if expr.func in HYPERBOLIC and len(expr.args) == 1:
        (arg,) = expr.args
        return _compute_data_eb_for_stencil_qoi_eb(
            (HYPERBOLIC[expr.func])(arg),
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
                    _qois.compile.sympy_expr_to_numpy(
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
        factor = _qois.compile.sympy_expr_to_numpy(
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


# pattern of syntactically weakly valid expressions
# we only check against forbidden tokens, not for semantic validity
#  i.e. just enough that it's safe to eval afterwards
_QOI_KWARG_PATTERN = (
    r"(?:"
    r"(?:base[ \t]*=[ \t]*)"
    r"|(?:order[ \t]*=[ \t]*)"
    r"|(?:accuracy[ \t]*=[ \t]*)"
    r"|(?:type[ \t]*=[ \t]*)"
    r"|(?:dx[ \t]*=[ \t]*)"
    r"|(?:axis[ \t]*=[ \t]*)"
    r")"
)
_QOI_ATOM_PATTERN = (
    r"(?:"
    + r"(?:[+-]?[0-9]+)"
    + r"|(?:[+-]?[0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?)"
    + r"|(?:x)"
    + r"|(?:X)"
    + r"|(?:I)"
    + r"".join(rf"|(?:{c})" for c in _qois.math.CONSTANTS)
    + r"".join(rf"|(?:{c})" for c in _qois.math.FUNCTIONS)
    + r"".join(rf"|(?:{c})" for c in _qois.amath.CONSTRUCTORS)
    + r"".join(rf"|(?:{c})" for c in _qois.amath.FUNCTIONS)
    + r"|(?:findiff)"
    + r")"
)
_QOI_SEPARATOR_PATTERN = r"(?:[ \t\n\(\)\[\],:\+\-\*/])"
_QOI_PATTERN = re.compile(
    rf"{_QOI_SEPARATOR_PATTERN}*{_QOI_ATOM_PATTERN}(?:{_QOI_SEPARATOR_PATTERN}+{_QOI_KWARG_PATTERN}?{_QOI_ATOM_PATTERN})*{_QOI_SEPARATOR_PATTERN}*"
)

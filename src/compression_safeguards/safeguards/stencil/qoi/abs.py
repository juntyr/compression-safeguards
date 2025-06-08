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

from ....utils.binding import Bindings
from ....utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero,
    _nextafter,
    as_bits,
    to_finite_float,
    to_float,
)
from ....utils.intervals import Interval, IntervalUnion
from ....utils.typing import F, S, T
from ..._qois.amath import CONSTRUCTORS as AMATH_CONSTRUCTORS
from ..._qois.amath import FUNCTIONS as AMATH_FUNCTIONS
from ..._qois.array import NumPyLikeArray
from ..._qois.compile import sympy_expr_to_numpy as compile_sympy_expr_to_numpy
from ..._qois.eb import (
    compute_data_eb_for_stencil_qoi_eb_unchecked,
    ensure_bounded_derived_error,
)
from ..._qois.findiff import create_findiff_for_neighbourhood
from ..._qois.math import CONSTANTS as MATH_CONSTANTS
from ..._qois.math import FUNCTIONS as MATH_FUNCTIONS
from ..._qois.re import (
    QOI_COMMENT_PATTERN,
    QOI_FLOAT_LITERAL_PATTERN,
    QOI_INT_LITERAL_PATTERN,
    QOI_WHITESPACE_PATTERN,
)
from ...pointwise.abs import _compute_safe_eb_diff_interval
from .. import (
    BoundaryCondition,
    NeighbourhoodAxis,
    NeighbourhoodBoundaryAxis,
    _pad_with_boundary,
)
from ..abc import StencilSafeguard
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
    string form, over the neighbourhood tensor `X` that is centred on the
    pointwise value `x`. For example, to bound the error on the four-neighbour
    box mean in a 3x3 neighbourhood (where `x = X[I]`), set
    `qoi="(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4"`.
    Note that `X` can be indexed absolute or relative to the centred data point
    `x` using the index array `I`.

    The stencil QoI safeguard can also be used to bound the pointwise absolute
    error of the finite-difference-approximated derivative over the data by
    using the `findiff` function in the `qoi` expression.

    The shape of the data neighbourhood is specified as an ordered list of
    unique data axes and boundary conditions that are applied to these axes.
    If the safeguard is applied to data with an insufficient number of
    dimensions, it raises an exception. If the safeguard is applied to data
    with additional dimensions, it is indendently applied along these extra
    axes. For instance, a 2d QoI is applied to independently to all 2d slices
    in a 3d data cube.

    If the data neighbourhood uses the
    [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid]
    boundary condition along an axis, the safeguard is only applied to data
    neighbourhoods centred on data points that have sufficient points before
    and after to satisfy the neighbourhood shape, i.e. it is not applied to
    all data points. If the axis is smaller than required by the neighbourhood
    along this axis, the safeguard is not applied. Using a different
    [`BoundaryCondition`][compression_safeguards.safeguards.stencil.BoundaryCondition]
    ensures that the safeguard is always applied to all data points.

    If the derived quantity of interest for a data neighbourhood evaluates to
    an infinite value, this safeguard guarantees that the quantity of interest
    on the decoded data neighbourhood produces the exact same infinite value.
    For a NaN quantity of interest, this safeguard guarantees that the quantity
    of interest on the decoded data neighbourhood is also NaN, but does not
    guarantee that it has the same bit pattern.

    The QoI expression is written using the following EBNF grammar[^1] for
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
      | array
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

    array   =
        "A", "[", [
            expr, { ",", expr }, [","]    (* n-dimensional array *)
        ], "]"
    ;

    const   =
        "e"                               (* Euler's number *)
      | "pi"                              (* pi *)
    ;

    var     =
        "x"                               (* pointwise data value *)
      | "X"                               (* data neighbourhood *)
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

    The QoI expression can also contain whitespaces (space ` `, tab `\\t`,
    newline `\\n`) and single-line inline comments starting with a hash `#`.

    The implementation of the absolute error bound on pointwise quantities of
    interest is inspired by:

    > Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. *Proceedings of the VLDB Endowment*.
    16, 4 (December 2022), 697-710. Available from:
    <https://doi.org/10.14778/3574245.3574255>.

    Parameters
    ----------
    qoi : StencilExpr
        The non-constant expression for computing the derived quantity of
        interest over a neighbourhood tensor `X`.
    neighbourhood : Sequence[dict | NeighbourhoodBoundaryAxis]
        The non-empty axes of the data neighbourhood for which the quantity of
        interest is computed. The neighbourhood window is applied independently
        over any additional axes in the data.

        The per-axis boundary conditions are applied to the data in their order
        in the neighbourhood, i.e. earlier boundary extensions can influence
        later ones.
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

        shape = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)
        I = tuple(axis.before for axis in self._neighbourhood)  # noqa: E741

        self._X = sp.tensor.array.expressions.ArraySymbol("X", shape)
        X = self._X.as_explicit()
        X.__class__ = NumPyLikeArray

        qoi_stripped = QOI_WHITESPACE_PATTERN.sub(
            " ", QOI_COMMENT_PATTERN.sub(" ", qoi)
        ).strip()

        assert len(qoi_stripped) > 0, "QoI expression must not be empty"
        assert _QOI_PATTERN.fullmatch(qoi) is not None, "invalid QoI expression"
        try:
            qoi_expr = sp.parse_expr(
                qoi_stripped,
                local_dict=dict(
                    # === data ===
                    # data neighbourhood
                    X=X,
                    x=X.__getitem__(I),
                    # neighbourhood index
                    I=NumPyLikeArray(I),
                    # === constants ===
                    **MATH_CONSTANTS,
                    # === operators ===
                    # poinwise math
                    **MATH_FUNCTIONS,
                    # array math
                    **AMATH_CONSTRUCTORS,
                    **AMATH_FUNCTIONS,
                    # finite difference
                    findiff=create_findiff_for_neighbourhood(self._X, shape, I),
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
                "QoI expression must evaluate to a numeric expression"
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
                f"failed to parse QoI expression {qoi!r}: {err}"
            ) from err
        assert len(qoi_expr.free_symbols) > 0, "QoI expression must not be constant"
        assert not qoi_expr.has(sp.I), (
            "QoI expression must not contain imaginary numbers"
        )

        self._qoi = qoi
        self._qoi_expr = qoi_expr

    def compute_check_neighbourhood_for_data_shape(
        self,
        data_shape: tuple[int, ...],
    ) -> tuple[None | NeighbourhoodAxis, ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        [`None`][None] is returned along dimensions for which the stencil QoI
        safeguard does not need to look at adjacent data points.

        This method also checks that the data shape is compatible with the
        stencil QoI safeguard.

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
            naxis = axis.axis if axis.axis >= 0 else len(data_shape) + axis.axis
            if naxis in all_axes:
                raise IndexError(
                    f"duplicate axis index {axis.axis}, normalised to {naxis}, for array of shape {data_shape}"
                )
            all_axes.append(naxis)

            neighbourhood[naxis] = axis.shape

        if np.prod(data_shape) == 0:
            return (None,) * len(data_shape)

        return tuple(neighbourhood)

    def evaluate_qoi(
        self, data: np.ndarray[S, np.dtype[T]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
        """
        Evaluate the derived quantity of interest on the `data`.

        The quantity of interest may have a different shape if the
        [valid][compression_safeguards.safeguards.stencil.BoundaryCondition.valid]
        boundary condition is used along any axis.

        If the `data` is of integer dtype, the quantity of interest is
        evaluated in floating point with sufficient precision to represent all
        integer values.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the quantity of interest is evaluated.

        Returns
        -------
        qoi : np.ndarray[tuple[int, ...], np.dtype[F]]
            Evaluated quantity of interest, in floating point.
        """

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        empty_shape = list(data.shape)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, return empty
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                empty_shape[axis.axis] = max(
                    0, empty_shape[axis.axis] - axis.before - axis.after
                )

        if any(s == 0 for s in empty_shape):
            float_dtype: np.dtype[F] = to_float(
                np.array([0, 1], dtype=data.dtype)
            ).dtype
            return np.zeros(empty_shape, dtype=float_dtype)

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

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        data_windows_float: np.ndarray = to_float(
            sliding_window_view(
                data_boundary,
                window,
                axis=tuple(axis.axis for axis in self._neighbourhood),
                writeable=False,
            )  # type: ignore
        )

        qoi_lambda = compile_sympy_expr_to_numpy(
            [self._X], self._qoi_expr, data_windows_float.dtype
        )

        return (qoi_lambda)(data_windows_float)

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
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

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

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

        qoi_lambda = compile_sympy_expr_to_numpy(
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
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
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

        # check that the data shape is compatible with the neighbourhood shape
        self.compute_check_neighbourhood_for_data_shape(data.shape)

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        # if the neighbourhood is empty, e.g. because we in valid mode and the
        #  neighbourhood shape exceeds the data shape, allow all values
        for axis, w in zip(self._neighbourhood, window):
            if data.shape[axis.axis] < w and axis.boundary == BoundaryCondition.valid:
                return Interval.full_like(data).into_union()
        if data.size == 0:
            return Interval.full_like(data).into_union()

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

        qoi_lambda = compile_sympy_expr_to_numpy(
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
                # not all data points are (valid) data neighbourhood centres
                data_float[tuple(s)],
                eb_qoi_lower,
                eb_qoi_upper,
            )
        assert np.all((eb_x_lower <= 0) & _isfinite(eb_x_lower))
        assert np.all((eb_x_upper >= 0) & _isfinite(eb_x_upper))

        # compute how the data indices are distributed into windows
        # i.e. for each QoI element, which data does it depend on
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
        # i.e. for each data element, which QoI elements does it contribute to
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

        # flatten the QoI error bounds and append an infinite value,
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
        ).into_union()

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
    XvN: np.ndarray[
        tuple[int, ...], np.dtype[F]
    ],  # np.ndarray[tuple[*Qs, *Ns], np.dtype[F]],
    Xv: np.ndarray[Qs, np.dtype[F]],
    tauv_lower: np.ndarray[Qs, np.dtype[F]],
    tauv_upper: np.ndarray[Qs, np.dtype[F]],
) -> tuple[np.ndarray[Qs, np.dtype[F]], np.ndarray[Qs, np.dtype[F]]]:
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
    XvN : np.ndarray[tuple[*Qs, *Ns], np.dtype[F]]
        Actual values of the input data, with the neighbourhood on the last axes.
    Xv : np.ndarray[Qs, np.dtype[F]]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[Qs, np.dtype[F]]
        Finite pointwise lower bound on the QoI error, must be negative or zero.
    eb_expr_upper : np.ndarray[Qs, np.dtype[F]]
        Finite pointwise upper bound on the QoI error, must be positive or zero.

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[Qs, np.dtype[F]], np.ndarray[Qs, np.dtype[F]]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    tl, tu = compute_data_eb_for_stencil_qoi_eb_unchecked(
        expr,
        Xv,
        tauv_lower,
        tauv_upper,
        check_is_x=lambda expr: (
            expr.func is sp.tensor.array.expressions.ArrayElement
            and len(expr.args) == 2
            and expr.args[0] == X
        ),
        evaluate_sympy_expr_to_numpy=lambda expr: compile_sympy_expr_to_numpy(
            [X], expr, Xv.dtype
        )(XvN),
        compute_data_eb_for_stencil_qoi_eb=lambda expr,
        Xv,
        tauv_lower,
        tauv_upper: _compute_data_eb_for_stencil_qoi_eb(
            expr,
            X,
            XvN,
            Xv,
            tauv_lower,
            tauv_upper,
        ),
    )

    exprl = compile_sympy_expr_to_numpy([X], expr, Xv.dtype)
    exprv = (exprl)(XvN)

    # handle rounding errors in the lower error bound computation
    tl = ensure_bounded_derived_error(
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
    tu = ensure_bounded_derived_error(
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


# pattern of syntactically weakly valid expressions
# we only check against forbidden tokens, not for semantic validity
#  i.e. just enough that it's safe to eval afterwards
_QOI_KWARG_PATTERN = (
    r"(?:"
    + r"|".join(
        rf"(?:{k}(?:{QOI_COMMENT_PATTERN.pattern}|(?:[ \t\n]))*=(?:{QOI_COMMENT_PATTERN.pattern}|(?:[ \t\n]))*)"
        for k in ("base", "order", "accuracy", "type", "dx", "axis")
    )
    + r")"
)
_QOI_ATOM_PATTERN = (
    r"(?:"
    + r"".join(
        rf"|(?:{l})"
        for l in (QOI_INT_LITERAL_PATTERN, QOI_FLOAT_LITERAL_PATTERN)  # noqa: E741
    )
    + r"|(?:x)"
    + r"|(?:X)"
    + r"|(?:I)"
    + r"".join(rf"|(?:{c})" for c in MATH_CONSTANTS)
    + r"".join(rf"|(?:{f})" for f in MATH_FUNCTIONS)
    + r"".join(rf"|(?:{c})" for c in AMATH_CONSTRUCTORS)
    + r"".join(rf"|(?:{f})" for f in AMATH_FUNCTIONS)
    + r"|(?:findiff)"
    + r")"
)
_QOI_SEPARATOR_PATTERN = (
    rf"(?:{QOI_COMMENT_PATTERN.pattern}|(?:[ \t\n\(\)\[\],:\+\-\*/]))"
)
_QOI_PATTERN = re.compile(
    rf"{_QOI_SEPARATOR_PATTERN}*{_QOI_ATOM_PATTERN}(?:{_QOI_SEPARATOR_PATTERN}+{_QOI_KWARG_PATTERN}?{_QOI_ATOM_PATTERN})*{_QOI_SEPARATOR_PATTERN}*"
)

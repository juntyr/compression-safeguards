"""
Stencil quantity of interest (QoI) error bound safeguard.
"""

__all__ = ["StencilQuantityOfInterestErrorBoundSafeguard"]

import re
from collections.abc import Sequence, Set
from typing import TypeVar

import numpy as np
import sympy as sp
import sympy.tensor.array.expressions  # noqa: F401
from numpy.lib.stride_tricks import sliding_window_view

from ....utils.bindings import Bindings, Parameter
from ....utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero,
    _nan_to_zero_inf_to_finite,
    _nextafter,
    as_bits,
    lossless_cast,
    saturating_finite_float_cast,
    to_float,
)
from ....utils.intervals import Interval, IntervalUnion
from ....utils.typing import F, S, T
from ..._qois.amath import CONSTRUCTORS as AMATH_CONSTRUCTORS
from ..._qois.amath import FUNCTIONS as AMATH_FUNCTIONS
from ..._qois.array import NumPyLikeArray
from ..._qois.associativity import rewrite_qoi_expr
from ..._qois.eb import (
    compute_data_eb_for_stencil_qoi_eb_unchecked,
    ensure_bounded_derived_error,
)
from ..._qois.eval import evaluate_sympy_expr_to_numpy
from ..._qois.finite_difference import create_finite_difference_for_neighbourhood
from ..._qois.interval import compute_safe_eb_lower_upper_interval_union
from ..._qois.math import CONSTANTS as MATH_CONSTANTS
from ..._qois.math import FUNCTIONS as MATH_FUNCTIONS
from ..._qois.re import (
    QOI_COMMENT_PATTERN,
    QOI_FLOAT_LITERAL_PATTERN,
    QOI_IDENTIFIER_PATTERN,
    QOI_INT_LITERAL_PATTERN,
    QOI_WHITESPACE_PATTERN,
)
from ..._qois.vars import FUNCTIONS as VARS_FUNCTIONS
from ..._qois.vars import (
    LateBoundConstant,
    LateBoundConstantEnvironment,
    UnresolvedVariable,
    VariableEnvironment,
)
from ...eb import (
    ErrorBound,
    _apply_finite_qoi_error_bound,
    _check_error_bound,
    _compute_finite_absolute_error,
    _compute_finite_absolute_error_bound,
)
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


class StencilQuantityOfInterestErrorBoundSafeguard(StencilSafeguard):
    """
    The `StencilQuantityOfInterestErrorBoundSafeguard` guarantees that the
    pointwise error `type` on a derived quantity of interest (QoI) over a
    neighbourhood of data points is less than or equal to the provided bound
    `eb`.

    The quantity of interest is specified as a non-constant expression, in
    string form, over the neighbourhood tensor `X` that is centred on the
    pointwise value `x`. For example, to bound the error on the four-neighbour
    box mean in a 3x3 neighbourhood (where `x = X[I]`), set
    `qoi="(X[I+A[-1,0]]+X[I+A[+1,0]]+X[I+A[0,-1]]+X[I+A[0,+1]])/4"`.
    Note that `X` can be indexed absolute or relative to the centred data point
    `x` using the index array `I`.

    The stencil QoI safeguard can also be used to bound the pointwise error of
    the finite-difference-approximated derivative[^1] over the data by using the
    `finite_difference` function in the `qoi` expression.

    [^1]: The finite difference coefficients for arbitrary orders, accuracies,
    and grid spacings are derived using the algorithm from: Fornberg, B. (1988).
    Generation of finite difference formulas on arbitrarily spaced grids.
    *Mathematics of Computation*, 51(184), 699-706. Available from:
    [doi:10.1090/s0025-5718-1988-0935077-0](https://doi.org/10.1090/s0025-5718-1988-0935077-0).

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

    The error bound can be verified by evaluating the QoI using the
    [`evaluate_qoi`][compression_safeguards.safeguards.stencil.qoi.eb.StencilQuantityOfInterestErrorBoundSafeguard.evaluate_qoi]
    method, which returns the the QoI in a sufficiently large floating point
    type (keeps the same dtype for floating point data, chooses a dtype with a
    mantissa that has at least as many bits as / for the integer dtype).

    The QoI expression is written using the following EBNF grammar[^2] for
    `expr`:

    [^2]: You can visualise the EBNF grammar at <https://matthijsgroen.github.io/ebnf2railroad/try-yourself.html>.

    ```ebnf
    expr    =
        literal
      | array
      | const
      | data
      | var
      | let
      | unary
      | binary
      | finite_difference
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
      | "c", "[", '"', ident, '"', "]"    (* late-bound constant value *)
      | "C", "[", '"', ident, '"', "]"    (* late-bound constant neighbourhood *)
      | "c", "[",
            '"', "$", ident, '"'          (* late-bound built-in constant value *)
        , "]"
      | "C", "[",
            '"', "$", ident, '"'          (* late-bound built-in constant neighbourhood *)
        , "]"
    ;

    data    =
        "x"                               (* pointwise data value *)
      | "X"                               (* data neighbourhood *)
    ;

    var     =
        "V", "[", '"', ident, '"', "]"    (* variable *)
    ;

    ident   =
        ( letter | "_" )                  (* identifier *)
      , { letter | digit | "_" }
    ;
    letter  =
        "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k"
      | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v"
      | "w" | "x" | "y" | "z" | "A" | "B" | "C" | "D" | "E" | "F" | "G"
      | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R"
      | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
    ;

    let     =
        "let", "(",
            var, ",", expr, {
                ",", var, ",", expr       (* let var=expr in expr scope *)
            },
        ")", "(", expr, ")"
    ;

    unary   =
        "(", expr, ")"                    (* parenthesis *)
      | "-", expr                         (* negation *)
      | "sqrt", "(", expr, ")"            (* square root *)
      | "ln", "(", expr, ")"              (* natural logarithm *)
      | "exp", "(", expr, ")"             (* exponential e^x *)
      | "sign", "(", expr, ")"            (* sign function, signed NaN for NaNs *)
      | "floor", "(", expr, ")"           (* round down, towards negative infinity *)
      | "ceil", "(", expr, ")"            (* round up, towards positive infinity *)
      | "trunc", "(", expr, ")"           (* round towards zero *)
      | "round_ties_even", "(", expr, ")" (* round to nearest integer, ties to even *)
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

    finite_difference =
        "finite_difference", "("          (* finite difference over an expression *)
          , expr, ","
          , "order", "=", int, ","           (* order of the derivative *)
          , "accuracy", "=", int, ","        (* order of accuracy of the approximation *)
          , "type", "=", (
                "-1" | "0" | "1"             (* backwards | central | forward difference *)
            ), ","
          , "axis", "=", int, ","            (* axis, relative to the neighbourhood *)
          , (
                "grid_spacing", "=", expr    (* scalar uniform grid spacing along the axis *)
              | "grid_centre", "=", expr     (* centre of an arbitrary grid along the axis *)
            )
          , [
                ",",
                "grid_period", "=", expr     (* optional grid period, e.g. 2*pi or 360 *)
            ]
      , ")"
    ;
    ```

    The QoI expression can also contain whitespaces (space ` `, tab `\\t`,
    newline `\\n`) and single-line inline comments starting with a hash `#`.

    The implementation of the error bound on pointwise quantities of interest
    is inspired by:

    > Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. *Proceedings of the VLDB Endowment*.
    16, 4 (December 2022), 697-710. Available from:
    [doi:10.14778/3574245.3574255](https://doi.org/10.14778/3574245.3574255).

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
    type : str | ErrorBound
        The type of error bound on the quantity of interest that is enforced by
        this safeguard.
    eb : int | float | str | Parameter
        The value of or late-bound parameter name for the error bound on the
        quantity of interest that is enforced by this safeguard.
    """

    __slots__ = (
        "_qoi",
        "_neighbourhood",
        "_type",
        "_eb",
        "_qoi_expr",
        "_X",
        "_late_bound_constants",
    )
    _qoi: StencilExpr
    _neighbourhood: tuple[NeighbourhoodBoundaryAxis, ...]
    _type: ErrorBound
    _eb: int | float | Parameter
    _qoi_expr: sp.Basic
    _X: sp.tensor.array.expressions.ArraySymbol
    _late_bound_constants: frozenset[LateBoundConstant]

    kind = "qoi_eb_stencil"

    def __init__(
        self,
        qoi: StencilExpr,
        neighbourhood: Sequence[dict | NeighbourhoodBoundaryAxis],
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
    ) -> None:
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

        self._type = type if isinstance(type, ErrorBound) else ErrorBound[type]

        if isinstance(eb, Parameter):
            self._eb = eb
        elif isinstance(eb, str):
            self._eb = Parameter(eb)
        else:
            _check_error_bound(self._type, eb)
            self._eb = eb

        shape = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)
        I = tuple(axis.before for axis in self._neighbourhood)  # noqa: E741

        self._X = sp.tensor.array.expressions.ArraySymbol(
            sp.Symbol("X", extended_real=True), shape
        )
        X = self._X.as_explicit()
        X.__class__ = NumPyLikeArray

        qoi_stripped = QOI_WHITESPACE_PATTERN.sub(
            " ", QOI_COMMENT_PATTERN.sub(" ", qoi)
        ).strip()

        def create_late_bound_array_symbol(name: str) -> NumPyLikeArray:
            C = sp.tensor.array.expressions.ArraySymbol(
                LateBoundConstant(name, extended_real=True), shape
            ).as_explicit()
            C.__class__ = NumPyLikeArray
            return C

        def create_late_bound_value_symbol(name: str):
            return create_late_bound_array_symbol(name)[I]

        assert len(qoi_stripped) > 0, "QoI expression must not be empty"
        assert _QOI_PATTERN.fullmatch(qoi_stripped) is not None, (
            "invalid QoI expression"
        )
        try:
            qoi_expr = sp.parse_expr(
                qoi_stripped,
                local_dict=dict(
                    # === data ===
                    # data neighbourhood
                    X=X,
                    x=X[I],
                    # neighbourhood index
                    I=NumPyLikeArray(I),
                    # === constants ===
                    **MATH_CONSTANTS,
                    c=LateBoundConstantEnvironment(
                        "c",
                        create_late_bound_value_symbol,  # type: ignore
                    ),
                    C=LateBoundConstantEnvironment(
                        "C",
                        create_late_bound_array_symbol,  # type: ignore
                    ),
                    # === variables ===
                    V=VariableEnvironment("V"),
                    **VARS_FUNCTIONS,
                    # === operators ===
                    # poinwise math
                    **MATH_FUNCTIONS,
                    # array math
                    **AMATH_CONSTRUCTORS,
                    **AMATH_FUNCTIONS,
                    # finite difference
                    finite_difference=create_finite_difference_for_neighbourhood(
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
            assert not isinstance(qoi_expr, UnresolvedVariable), (
                f'unresolved variable {qoi_expr._env._symbol}["{qoi_expr._name}"], perhaps you forgot to define it within a let expression'
            )
            assert isinstance(qoi_expr, sp.Basic), (
                "QoI expression must evaluate to a numeric expression"
            )
            qoi_expr = rewrite_qoi_expr(qoi_expr)
            self._late_bound_constants = frozenset(
                s for s in qoi_expr.free_symbols if isinstance(s, LateBoundConstant)
            )
            # check if the expression is well-formed (e.g. no int's that cannot
            #  be printed) and if an error bound can be computed
            _canary_repr = str(qoi_expr)
            _canary_data_eb = _compute_data_eb_for_stencil_qoi_eb(
                qoi_expr,
                self._X,
                np.zeros(shape),
                np.zeros(shape),
                np.zeros(shape),
                np.zeros(shape),
                {c: np.zeros(shape) for c in self._late_bound_constants},
            )
        except Exception as err:
            raise AssertionError(
                f"failed to parse QoI expression {qoi!r}: {err}"
            ) from err
        assert len(qoi_expr.free_symbols - self._late_bound_constants) > 0, (
            "QoI expression must not be constant"
        )
        assert not qoi_expr.has(sp.I), (
            "QoI expression must not contain imaginary numbers"
        )

        self._qoi = qoi
        self._qoi_expr = qoi_expr

    @property
    def late_bound(self) -> Set[Parameter]:
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        parameters = [c.parameter for c in self._late_bound_constants]

        if isinstance(self._eb, Parameter):
            parameters.append(self._eb)

        for axis in self._neighbourhood:
            if isinstance(axis.constant_boundary, Parameter):
                parameters.append(axis.constant_boundary)

        return frozenset(parameters)

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
        self,
        data: np.ndarray[S, np.dtype[T]],
        late_bound: Bindings,
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
        late_bound : Bindings
            Bindings for late-bound constants in the quantity of interest.

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

        constant_boundaries = [
            None
            if axis.constant_boundary is None
            else late_bound.resolve_ndarray_with_lossless_cast(
                axis.constant_boundary, (), data.dtype
            )
            if isinstance(axis.constant_boundary, Parameter)
            else lossless_cast(
                axis.constant_boundary,
                data.dtype,
                "stencil QoI safeguard constant boundary",
            )
            for axis in self._neighbourhood
        ]

        data_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = data
        for axis, axis_constant_boundary in zip(
            self._neighbourhood, constant_boundaries
        ):
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )

        window = tuple(axis.before + 1 + axis.after for axis in self._neighbourhood)

        data_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            to_float(
                sliding_window_view(
                    data_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
        )

        late_bound_constants = dict()
        for c in self._late_bound_constants:
            late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(
                    c.parameter, data.shape, data.dtype
                )
            )
            for axis, axis_constant_boundary in zip(
                self._neighbourhood, constant_boundaries
            ):
                late_boundary = _pad_with_boundary(
                    late_boundary,
                    axis.boundary,
                    axis.before,
                    axis.after,
                    axis_constant_boundary,
                    axis.axis,
                )
            late_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
                to_float(
                    sliding_window_view(
                        late_boundary,
                        window,
                        axis=tuple(axis.axis for axis in self._neighbourhood),
                        writeable=False,
                    )  # type: ignore
                )
            )
            late_bound_constants[c] = late_windows_float

        return evaluate_sympy_expr_to_numpy(
            self._qoi_expr,
            {self._X: data_windows_float, **late_bound_constants},
            data_windows_float.dtype,
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the error bound for
        the quantity of interest over a neighbourhood on the `data`.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data to be encoded.
        decoded : np.ndarray[S, np.dtype[T]]
            Decoded data.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
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

        constant_boundaries = [
            None
            if axis.constant_boundary is None
            else late_bound.resolve_ndarray_with_lossless_cast(
                axis.constant_boundary, (), data.dtype
            )
            if isinstance(axis.constant_boundary, Parameter)
            else lossless_cast(
                axis.constant_boundary,
                data.dtype,
                "stencil QoI safeguard constant boundary",
            )
            for axis in self._neighbourhood
        ]

        data_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = data
        decoded_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = decoded
        for axis, axis_constant_boundary in zip(
            self._neighbourhood, constant_boundaries
        ):
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )
            decoded_boundary = _pad_with_boundary(
                decoded_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )

        data_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            to_float(
                sliding_window_view(
                    data_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
        )
        decoded_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            to_float(
                sliding_window_view(
                    decoded_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
        )

        late_bound_constants = dict()
        for c in self._late_bound_constants:
            late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(
                    c.parameter, data.shape, data.dtype
                )
            )
            for axis, axis_constant_boundary in zip(
                self._neighbourhood, constant_boundaries
            ):
                late_boundary = _pad_with_boundary(
                    late_boundary,
                    axis.boundary,
                    axis.before,
                    axis.after,
                    axis_constant_boundary,
                    axis.axis,
                )
            late_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
                to_float(
                    sliding_window_view(
                        late_boundary,
                        window,
                        axis=tuple(axis.axis for axis in self._neighbourhood),
                        writeable=False,
                    )  # type: ignore
                )
            )
            late_bound_constants[c] = late_windows_float

        qoi_data = evaluate_sympy_expr_to_numpy(
            self._qoi_expr,
            {self._X: data_windows_float, **late_bound_constants},
            data_windows_float.dtype,
        )
        qoi_decoded = evaluate_sympy_expr_to_numpy(
            self._qoi_expr,
            {self._X: decoded_windows_float, **late_bound_constants},
            data_windows_float.dtype,
        )

        eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                qoi_data.shape,
                qoi_data.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, qoi_data.dtype, "stencil QoI error bound safeguard eb"
            )
        )
        _check_error_bound(self._type, eb)

        finite_ok = _compute_finite_absolute_error(
            self._type, qoi_data, qoi_decoded
        ) <= _compute_finite_absolute_error_bound(self._type, eb, qoi_data)

        same_bits = as_bits(qoi_data, kind="V") == as_bits(qoi_decoded, kind="V")
        both_nan = _isnan(qoi_data) & _isnan(qoi_decoded)

        windows_ok = np.where(
            _isfinite(qoi_data),
            finite_ok,
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
        Compute the intervals in which the error bound is upheld with respect
        to the quantity of interest over a neighbourhood on the `data`.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the error bound is upheld.
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

        constant_boundaries = [
            None
            if axis.constant_boundary is None
            else late_bound.resolve_ndarray_with_lossless_cast(
                axis.constant_boundary, (), data.dtype
            )
            if isinstance(axis.constant_boundary, Parameter)
            else lossless_cast(
                axis.constant_boundary,
                data.dtype,
                "stencil QoI safeguard constant boundary",
            )
            for axis in self._neighbourhood
        ]

        data_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = data
        for axis, axis_constant_boundary in zip(
            self._neighbourhood, constant_boundaries
        ):
            data_boundary = _pad_with_boundary(
                data_boundary,
                axis.boundary,
                axis.before,
                axis.after,
                axis_constant_boundary,
                axis.axis,
            )

        data_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
            to_float(
                sliding_window_view(
                    data_boundary,
                    window,
                    axis=tuple(axis.axis for axis in self._neighbourhood),
                    writeable=False,
                )  # type: ignore
            )
        )

        late_bound_constants = dict()
        for c in self._late_bound_constants:
            late_boundary: np.ndarray[tuple[int, ...], np.dtype[T]] = (
                late_bound.resolve_ndarray_with_lossless_cast(
                    c.parameter, data.shape, data.dtype
                )
            )
            for axis, axis_constant_boundary in zip(
                self._neighbourhood, constant_boundaries
            ):
                late_boundary = _pad_with_boundary(
                    late_boundary,
                    axis.boundary,
                    axis.before,
                    axis.after,
                    axis_constant_boundary,
                    axis.axis,
                )
            late_windows_float: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = (
                to_float(
                    sliding_window_view(
                        late_boundary,
                        window,
                        axis=tuple(axis.axis for axis in self._neighbourhood),
                        writeable=False,
                    )  # type: ignore
                )
            )
            late_bound_constants[c] = late_windows_float

        data_qoi = evaluate_sympy_expr_to_numpy(
            self._qoi_expr,
            {self._X: data_windows_float, **late_bound_constants},
            data_windows_float.dtype,
        )

        eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                data_qoi.shape,
                data_qoi.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, data_qoi.dtype, "stencil QoI error bound safeguard eb"
            )
        )
        _check_error_bound(self._type, eb)

        qoi_lower_upper: tuple[
            np.ndarray[S, np.dtype[np.floating]], np.ndarray[S, np.dtype[np.floating]]
        ] = _apply_finite_qoi_error_bound(
            self._type,
            eb,
            data_qoi,
        )
        qoi_lower, qoi_upper = qoi_lower_upper

        # ensure the error bounds are representable in QoI space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # compute the adjusted error bound
            eb_qoi_lower = _nan_to_zero(qoi_lower - data_qoi)
            eb_qoi_upper = _nan_to_zero(qoi_upper - data_qoi)

            # check if they're representable within the error bound
            eb_qoi_lower_outside = (data_qoi + eb_qoi_lower) < qoi_lower
            eb_qoi_upper_outside = (data_qoi + eb_qoi_upper) > qoi_upper

            # otherwise nudge the error-bound adjusted QoIs
            # we can nudge with nextafter since the QoIs are floating point
            eb_qoi_lower = np.where(
                eb_qoi_lower_outside & _isfinite(qoi_lower),
                _nextafter(eb_qoi_lower, 0),
                eb_qoi_lower,
            )
            eb_qoi_upper = np.where(
                eb_qoi_upper_outside & _isfinite(qoi_upper),
                _nextafter(eb_qoi_upper, 0),
                eb_qoi_upper,
            )

        # check that the adjusted error bounds fulfil all requirements
        assert eb_qoi_lower.ndim == data.ndim
        assert eb_qoi_lower.dtype == data_windows_float.dtype
        assert eb_qoi_upper.ndim == data.ndim
        assert eb_qoi_upper.dtype == data_windows_float.dtype
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            assert np.all(
                (eb_qoi_lower <= 0)
                & (((data_qoi + eb_qoi_lower) >= qoi_lower) | ~_isfinite(data_qoi))
                & _isfinite(eb_qoi_lower)
            )
            assert np.all(
                (eb_qoi_upper >= 0)
                & (((data_qoi + eb_qoi_upper) <= qoi_upper) | ~_isfinite(data_qoi))
                & _isfinite(eb_qoi_upper)
            )

        s = [slice(None)] * data.ndim
        for axis in self._neighbourhood:
            if axis.boundary == BoundaryCondition.valid:
                start = None if axis.before == 0 else axis.before
                end = None if axis.after == 0 else -axis.after
                s[axis.axis] = slice(start, end)

        data_float: np.ndarray[S, np.dtype[np.floating]] = to_float(data)

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
                late_bound_constants,
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
                None if axis.constant_boundary is None else np.array(data.size),  # type: ignore
                axis.axis,
            )
        indices_windows = sliding_window_view(  # type: ignore
            indices_boundary,
            window,
            axis=tuple(axis.axis for axis in self._neighbourhood),
            writeable=False,
        ).reshape((-1, np.prod(window)))

        # only contribute window elements that are used in the QoI
        window_used = np.zeros(window, dtype=bool)
        for x in self._qoi_expr.find(sp.tensor.array.expressions.ArrayElement):
            name, idxs = x.args
            if name == self._X:
                window_used[tuple(idxs)] = True

        # compute the reverse: for each data element, which windows is it in
        # i.e. for each data element, which QoI elements does it contribute to
        #      and thus which error bounds affect it
        reverse_indices_windows = np.full(
            (data.size, np.sum(window_used)), indices_windows.shape[0]
        )
        reverse_indices_counter = np.zeros(data.size, dtype=int)
        for i, u in enumerate(window_used.flat):
            # skip window indices that are not used in the QoI
            if not u:
                continue
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
        # since some data elements may have no error bounds that affect them,
        #  e.g. because of the valid boundary condition, they may have infinite
        #  bounds that we need to map back to huge finite bounds
        eb_x_orig_lower: np.ndarray[S, np.dtype[np.floating]] = (
            _nan_to_zero_inf_to_finite(  # type: ignore
                np.amax(eb_x_lower_flat[reverse_indices_windows], axis=1)
            ).reshape(data.shape)
        )
        eb_x_orig_upper: np.ndarray[S, np.dtype[np.floating]] = (
            _nan_to_zero_inf_to_finite(  # type: ignore
                np.amin(eb_x_upper_flat[reverse_indices_windows], axis=1)
            ).reshape(data.shape)
        )

        return compute_safe_eb_lower_upper_interval_union(
            data,
            data_float,
            eb_x_orig_lower,
            eb_x_orig_upper,
        )

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
            type=self._type.name,
            eb=self._eb,
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
    late_bound_constants: dict[
        LateBoundConstant, np.ndarray[tuple[int, ...], np.dtype[F]]
    ],
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
        Actual values of the input data, with the neighbourhood on the last
        axis.
    Xv : np.ndarray[Qs, np.dtype[F]]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[Qs, np.dtype[F]]
        Finite pointwise lower bound on the QoI error, must be negative or
        zero.
    eb_expr_upper : np.ndarray[Qs, np.dtype[F]]
        Finite pointwise upper bound on the QoI error, must be positive or
        zero.
    late_bound_constants : dict[LateBoundConstant, np.ndarray[tuple[*Qs, *Ns], np.dtype[F]]]
        Values of the late-bound constants that are not counted as symbols,
        with the neighbourhood on the last axis.

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[Qs, np.dtype[F]], np.ndarray[Qs, np.dtype[F]]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    > Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. *Proceedings of the VLDB Endowment*.
    16, 4 (December 2022), 697-710. Available from:
    [doi:10.14778/3574245.3574255](https://doi.org/10.14778/3574245.3574255).
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
        evaluate_sympy_expr_to_numpy=lambda expr: evaluate_sympy_expr_to_numpy(
            expr,
            {X: XvN, **late_bound_constants},
            Xv.dtype,
        ),
        late_bound_constants=frozenset(late_bound_constants.keys()),
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
            late_bound_constants,
        ),
    )

    exprv = evaluate_sympy_expr_to_numpy(
        expr,
        {X: XvN, **late_bound_constants},
        Xv.dtype,
    )

    # handle rounding errors in the lower error bound computation
    tl = ensure_bounded_derived_error(
        # tl has shape Qs and has XvN (*Qs, *Ns), so their sum has (*Qs, *Ns)
        #  and evaluating the expression brings us back to Qs
        lambda tl: np.where(  # type: ignore
            tl == 0,
            exprv,
            evaluate_sympy_expr_to_numpy(
                expr,
                {
                    X: XvN + tl.reshape(list(tl.shape) + [1] * (XvN.ndim - tl.ndim)),
                    **late_bound_constants,
                },
                Xv.dtype,
            ),
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
            evaluate_sympy_expr_to_numpy(
                expr,
                {
                    X: XvN + tu.reshape(list(tu.shape) + [1] * (XvN.ndim - tu.ndim)),
                    **late_bound_constants,
                },
                Xv.dtype,
            ),
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
        rf"(?:{k}[ ]?=[ ]?)"
        for k in (
            "base",
            "order",
            "accuracy",
            "type",
            "axis",
            "grid_spacing",
            "grid_centre",
            "grid_period",
        )
    )
    + r")"
)
_QOI_ATOM_PATTERN = (
    r"(?:"
    + r"|".join(
        rf"(?:{l})"
        for l in (QOI_INT_LITERAL_PATTERN, QOI_FLOAT_LITERAL_PATTERN)  # noqa: E741
    )
    + r"|(?:x)"
    + r"|(?:X)"
    + r"|(?:I)"
    + r"".join(rf"|(?:{c})" for c in MATH_CONSTANTS)
    + r"".join(rf"|(?:{f})" for f in MATH_FUNCTIONS)
    + r"".join(rf"|(?:{c})" for c in AMATH_CONSTRUCTORS)
    + r"".join(rf"|(?:{f})" for f in AMATH_FUNCTIONS)
    + r"|(?:finite_difference)"
    + r"".join(rf"|(?:{v})" for v in VARS_FUNCTIONS)
    + r"".join(
        rf'|(?:{v}[ ]?\[[ ]?"[\$]?{QOI_IDENTIFIER_PATTERN}"[ ]?\])'
        for v in ("c", "C", "V")
    )
    + r")"
)
_QOI_SEPARATOR_PATTERN = r"(?:[ \(\)\[\],:\+\-\*/])"
_QOI_PATTERN = re.compile(
    rf"{_QOI_SEPARATOR_PATTERN}*{_QOI_ATOM_PATTERN}(?:{_QOI_SEPARATOR_PATTERN}+{_QOI_KWARG_PATTERN}?{_QOI_ATOM_PATTERN})*{_QOI_SEPARATOR_PATTERN}*"
)

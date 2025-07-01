"""
Pointwise quantity of interest (QoI) error bound safeguard.
"""

__all__ = ["PointwiseQuantityOfInterestErrorBoundSafeguard"]

import re
from collections.abc import Set

import numpy as np
import sympy as sp

from ....utils.bindings import Bindings, Parameter
from ....utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero,
    _nextafter,
    as_bits,
    saturating_finite_float_cast,
    to_float,
)
from ....utils.intervals import IntervalUnion
from ....utils.typing import F, S, T
from ..._qois.associativity import rewrite_qoi_expr
from ..._qois.compile import sympy_expr_to_numpy as compile_sympy_expr_to_numpy
from ..._qois.eb import (
    compute_data_eb_for_stencil_qoi_eb_unchecked,
    ensure_bounded_derived_error,
)
from ..._qois.interval import compute_safe_eb_lower_upper_interval
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
from ..abc import PointwiseSafeguard
from . import PointwiseExpr


class PointwiseQuantityOfInterestErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `PointwiseQuantityOfInterestErrorBoundSafeguard` guarantees that the
    pointwise error `type` on a derived pointwise quantity of interest (QoI)
    is less than or equal to the provided bound `eb`.

    The quantity of interest is specified as a non-constant expression, in
    string form, over the pointwise value `x`. For example, to bound the error
    on the square of `x`, set `qoi="x**2"`.

    If the derived quantity of interest for an element evaluates to an infinite
    value, this safeguard guarantees that the quantity of interest on the
    decoded value produces the exact same infinite value. For a NaN quantity of
    interest, this safeguard guarantees that the quantity of interest on the
    decoded value is also NaN, but does not guarantee that it has the same
    bit pattern.

    The error bound can be verified by evaluating the QoI using the
    [`evaluate_qoi`][compression_safeguards.safeguards.pointwise.qoi.eb.PointwiseQuantityOfInterestErrorBoundSafeguard.evaluate_qoi]
    method, which returns the the QoI in a sufficiently large floating point
    type (keeps the same dtype for floating point data, chooses a dtype with a
    mantissa that has at least as many bits as / for the integer dtype).

    The QoI expression is written using the following EBNF grammar[^1] for
    `expr`:

    [^1]: You can visualise the EBNF grammar at <https://matthijsgroen.github.io/ebnf2railroad/try-yourself.html>.

    ```ebnf
    expr    =
        literal
      | const
      | data
      | var
      | let
      | unary
      | binary
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
      | "c", "[", '"', ident, '"', "]"    (* late-bound constant value *)
      | "c", "[",
            '"', "$", ident, '"'          (* late-bound built-in constant value *)
      , "]"
    ;

    data    = "x";                        (* pointwise data value *)

    var     =
        "v", "[", '"', ident, '"', "]"    (* variable *)
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
    qoi : PointwiseExpr
        The non-constant expression for computing the derived quantity of
        interest over a pointwise value `x`.
    type : str | ErrorBound
        The type of error bound on the quantity of interest that is enforced by
        this safeguard.
    eb : int | float | str | Parameter
        The value of or late-bound parameter name for the error bound on the
        quantity of interest that is enforced by this safeguard.
    """

    __slots__ = (
        "_qoi",
        "_type",
        "_eb",
        "_qoi_expr",
        "_x",
        "_late_bound_constants",
    )
    _qoi: PointwiseExpr
    _type: ErrorBound
    _eb: int | float | Parameter
    _qoi_expr: sp.Basic
    _x: sp.Symbol
    _late_bound_constants: frozenset[LateBoundConstant]

    kind = "qoi_eb_pw"

    def __init__(
        self,
        qoi: PointwiseExpr,
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
    ) -> None:
        self._type = type if isinstance(type, ErrorBound) else ErrorBound[type]

        if isinstance(eb, Parameter):
            self._eb = eb
        elif isinstance(eb, str):
            self._eb = Parameter(eb)
        else:
            _check_error_bound(self._type, eb)
            self._eb = eb

        self._x = sp.Symbol("x", extended_real=True)

        qoi_stripped = QOI_WHITESPACE_PATTERN.sub(
            " ", QOI_COMMENT_PATTERN.sub(" ", qoi)
        ).strip()

        assert len(qoi_stripped) > 0, "QoI expression must not be empty"
        assert _QOI_PATTERN.fullmatch(qoi_stripped) is not None, (
            "invalid QoI expression"
        )
        try:
            qoi_expr = sp.parse_expr(
                qoi_stripped,
                local_dict=dict(
                    # === data ===
                    # pointwise data
                    x=self._x,
                    # === constants ===
                    **MATH_CONSTANTS,
                    c=LateBoundConstantEnvironment(
                        "c", lambda name: LateBoundConstant(name, extended_real=True)
                    ),
                    # === variables ===
                    v=VariableEnvironment("v"),
                    **VARS_FUNCTIONS,
                    # === operators ===
                    # poinwise math
                    **MATH_FUNCTIONS,
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
            _canary_data_eb = _compute_data_eb_for_qoi_eb(
                qoi_expr,
                self._x,
                np.empty(0),
                np.empty(0),
                np.empty(0),
                {c: np.empty(0) for c in self._late_bound_constants},
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

        return frozenset(parameters)

    def evaluate_qoi(
        self,
        data: np.ndarray[S, np.dtype[T]],
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[F]]:
        """
        Evaluate the derived quantity of interest on the `data`.

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
        qoi : np.ndarray[S, np.dtype[F]]
            Evaluated quantity of interest, in floating point.
        """

        data_float: np.ndarray[S, np.dtype[F]] = to_float(data)

        late_bound_constants = {
            c: late_bound.resolve_ndarray_with_lossless_cast(
                c.parameter, data.shape, data_float.dtype
            )
            for c in self._late_bound_constants
        }

        qoi_lambda = compile_sympy_expr_to_numpy(
            [self._x, *late_bound_constants.keys()], self._qoi_expr, data_float.dtype
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            return (qoi_lambda)(data_float, *late_bound_constants.values())

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
        the quantity of interest on the `data`.

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

        data_float: np.ndarray[S, np.dtype[np.floating]] = to_float(data)

        late_bound_constants = {
            c: late_bound.resolve_ndarray_with_lossless_cast(
                c.parameter, data.shape, data_float.dtype
            )
            for c in self._late_bound_constants
        }

        qoi_lambda = compile_sympy_expr_to_numpy(
            [self._x, *late_bound_constants.keys()], self._qoi_expr, data_float.dtype
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            qoi_data = (qoi_lambda)(data_float, *late_bound_constants.values())
            qoi_decoded = (qoi_lambda)(
                to_float(decoded), *late_bound_constants.values()
            )

        eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                qoi_data.shape,
                qoi_data.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, qoi_data.dtype, "pointwise QoI error bound safeguard eb"
            )
        )
        _check_error_bound(self._type, eb)

        finite_ok = _compute_finite_absolute_error(
            self._type, qoi_data, qoi_decoded
        ) <= _compute_finite_absolute_error_bound(self._type, eb, qoi_data)

        same_bits = as_bits(qoi_data, kind="V") == as_bits(qoi_decoded, kind="V")
        both_nan = _isnan(qoi_data) & _isnan(qoi_decoded)

        ok = np.where(
            _isfinite(qoi_data),
            finite_ok,
            np.where(
                _isinf(qoi_data),
                same_bits,
                both_nan,
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the error bound is upheld with
        respect to the quantity of interest on the `data`.

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

        data_float: np.ndarray[S, np.dtype[np.floating]] = to_float(data)

        late_bound_constants = {
            c: late_bound.resolve_ndarray_with_lossless_cast(
                c.parameter, data.shape, data_float.dtype
            )
            for c in self._late_bound_constants
        }

        qoi_lambda = compile_sympy_expr_to_numpy(
            [self._x, *late_bound_constants.keys()], self._qoi_expr, data_float.dtype
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            data_qoi: np.ndarray[S, np.dtype[np.floating]] = (qoi_lambda)(
                data_float, *late_bound_constants.values()
            )

        eb: np.ndarray[tuple[()] | S, np.dtype[np.floating]] = (
            late_bound.resolve_ndarray_with_saturating_finite_float_cast(
                self._eb,
                data_qoi.shape,
                data_qoi.dtype,
            )
            if isinstance(self._eb, Parameter)
            else saturating_finite_float_cast(
                self._eb, data_qoi.dtype, "pointwise QoI error bound safeguard eb"
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
            eb_qoi_lower: np.ndarray[S, np.dtype[np.floating]] = _nan_to_zero(
                qoi_lower - data_qoi
            )
            eb_qoi_upper: np.ndarray[S, np.dtype[np.floating]] = _nan_to_zero(
                qoi_upper - data_qoi
            )

            # check if they're representable within the error bound
            eb_qoi_lower_outside = (data_qoi + eb_qoi_lower) < qoi_lower
            eb_qoi_upper_outside = (data_qoi + eb_qoi_upper) > qoi_upper

            # otherwise nudge the error-bound adjusted QoIs
            # we can nudge with nextafter since the QoIs are floating point
            eb_qoi_lower = np.where(
                eb_qoi_lower_outside & _isfinite(qoi_lower),
                _nextafter(eb_qoi_lower, 0),
                eb_qoi_lower,
            )  # type: ignore
            eb_qoi_upper = np.where(
                eb_qoi_upper_outside & _isfinite(qoi_upper),
                _nextafter(eb_qoi_upper, 0),
                eb_qoi_upper,
            )  # type: ignore

        # check that the adjusted error bounds fulfil all requirements
        assert eb_qoi_lower.shape == data.shape
        assert eb_qoi_lower.dtype == data_float.dtype
        assert eb_qoi_upper.shape == data.shape
        assert eb_qoi_upper.dtype == data_float.dtype
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

        late_bound_constants = {
            c: late_bound.resolve_ndarray_with_lossless_cast(
                c.parameter, data.shape, data_float.dtype
            )
            for c in self._late_bound_constants
        }

        # compute the error bound in data space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_x_lower_upper: tuple[
                np.ndarray[S, np.dtype[np.floating]],
                np.ndarray[S, np.dtype[np.floating]],
            ] = _compute_data_eb_for_qoi_eb(
                self._qoi_expr,
                self._x,
                data_float,
                eb_qoi_lower,
                eb_qoi_upper,
                late_bound_constants,
            )
            eb_x_lower, eb_x_upper = eb_x_lower_upper

        return compute_safe_eb_lower_upper_interval(
            data,
            data_float,
            eb_x_lower,
            eb_x_upper,
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
            kind=type(self).kind, qoi=self._qoi, type=self._type.name, eb=self._eb
        )


@np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
def _compute_data_eb_for_qoi_eb(
    expr: sp.Basic,
    x: sp.Symbol,
    xv: np.ndarray[S, np.dtype[F]],
    tauv_lower: np.ndarray[S, np.dtype[F]],
    tauv_upper: np.ndarray[S, np.dtype[F]],
    late_bound_constants: dict[LateBoundConstant, np.ndarray[S, np.dtype[F]]],
) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
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
    xv : np.ndarray[S, np.dtype[F]]
        Actual values of the input data.
    eb_expr_lower : np.ndarray[S, np.dtype[F]]
        Finite pointwise lower bound on the QoI error, must be negative or zero.
    eb_expr_upper : np.ndarray[S, np.dtype[F]]
        Finite pointwise upper bound on the QoI error, must be positive or zero.
    late_bound_constants : dict[LateBoundConstant, np.ndarray[S, np.dtype[F]]]
        Values of the late-bound constants that are not counted as symbols.

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

    tl, tu = compute_data_eb_for_stencil_qoi_eb_unchecked(
        expr,
        xv,
        tauv_lower,
        tauv_upper,
        check_is_x=lambda expr: expr == x,
        evaluate_sympy_expr_to_numpy=lambda expr: compile_sympy_expr_to_numpy(
            [x, *late_bound_constants.keys()], expr, xv.dtype
        )(xv, *late_bound_constants.values()),
        late_bound_constants=frozenset(late_bound_constants.keys()),
        compute_data_eb_for_stencil_qoi_eb=lambda expr,
        xv,
        tauv_lower,
        tauv_upper: _compute_data_eb_for_qoi_eb(
            expr,
            x,
            xv,
            tauv_lower,
            tauv_upper,
            late_bound_constants,
        ),
    )

    exprl = compile_sympy_expr_to_numpy(
        [x, *late_bound_constants.keys()], expr, xv.dtype
    )
    exprv = (exprl)(xv, *late_bound_constants.values())

    # handle rounding errors in the lower error bound computation
    tl = ensure_bounded_derived_error(
        lambda tl: np.where(  # type: ignore
            tl == 0, exprv, (exprl)(xv + tl, *late_bound_constants.values())
        ),
        exprv,
        xv,
        tl,
        tauv_lower,
        tauv_upper,
    )
    tu = ensure_bounded_derived_error(
        lambda tu: np.where(  # type: ignore
            tu == 0, exprv, (exprl)(xv + tu, *late_bound_constants.values())
        ),
        exprv,
        xv,
        tu,
        tauv_lower,
        tauv_upper,
    )

    return tl, tu


# pattern of syntactically weakly valid expressions
# we only check against forbidden tokens, not for semantic validity
#  i.e. just enough that it's safe to eval afterwards
_QOI_KWARG_PATTERN = r"(?:" + r"|".join(rf"(?:{k}[ ]?=[ ]?)" for k in ("base",)) + r")"
_QOI_ATOM_PATTERN = (
    r"(?:"
    + r"|".join(
        rf"(?:{l})"
        for l in (QOI_INT_LITERAL_PATTERN, QOI_FLOAT_LITERAL_PATTERN)  # noqa: E741
    )
    + r"|(?:x)"
    + r"".join(rf"|(?:{c})" for c in MATH_CONSTANTS)
    + r"".join(rf"|(?:{f})" for f in MATH_FUNCTIONS)
    + r"".join(rf"|(?:{v})" for v in VARS_FUNCTIONS)
    + r"".join(
        rf'|(?:{v}[ ]?\[[ ]?"[\$]?{QOI_IDENTIFIER_PATTERN}"[ ]?\])' for v in ("c", "v")
    )
    + r")"
)
_QOI_SEPARATOR_PATTERN = r"(?:[ \(\),\+\-\*/])"
_QOI_PATTERN = re.compile(
    rf"{_QOI_SEPARATOR_PATTERN}*{_QOI_ATOM_PATTERN}(?:{_QOI_SEPARATOR_PATTERN}+{_QOI_KWARG_PATTERN}?{_QOI_ATOM_PATTERN})*{_QOI_SEPARATOR_PATTERN}*"
)

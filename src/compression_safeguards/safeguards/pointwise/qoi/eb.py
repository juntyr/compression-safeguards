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
    to_float,
)
from ....utils.intervals import IntervalUnion
from ....utils.typing import F, S, T
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
    QOI_INT_LITERAL_PATTERN,
    QOI_WHITESPACE_PATTERN,
)
from ..._qois.vars import CONSTRUCTORS as VARS_CONSTRUCTORS
from ..._qois.vars import FUNCTIONS as VARS_FUNCTIONS
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
    ;

    data    = "x";                        (* pointwise data value *)

    var     =
        "V", "[", '"', ident, '"', "]"    (* variable *)
    ;

    let     =
        "let", "(",
            var, ",", expr, ",", expr     (* let var=expr in expr scope *)
      , ")"
    ;

    unary   =
        "(", expr, ")"                    (* parenthesis *)
      | "-", expr                         (* negation *)
      | "sqrt", "(", expr, ")"            (* square root *)
      | "ln", "(", expr, ")"              (* natural logarithm *)
      | "exp", "(", expr, ")"             (* exponential e^x *)
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
    <https://doi.org/10.14778/3574245.3574255>.

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
    )
    _qoi: PointwiseExpr
    _type: ErrorBound
    _eb: int | float | Parameter
    _qoi_expr: sp.Basic
    _x: sp.Symbol

    kind = "qoi_eb_pw"

    def __init__(
        self,
        qoi: PointwiseExpr,
        type: str | ErrorBound,
        eb: int | float | str | Parameter,
    ):
        self._type = type if isinstance(type, ErrorBound) else ErrorBound[type]

        if isinstance(eb, Parameter):
            self._eb = eb
        elif isinstance(eb, str):
            self._eb = Parameter(eb)
        else:
            _check_error_bound(self._type, eb)
            self._eb = eb

        self._x = sp.Symbol("x", real=True)

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
                    # pointwise data
                    x=self._x,
                    # === constants ===
                    **MATH_CONSTANTS,
                    # === variables ===
                    **VARS_CONSTRUCTORS,
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
            assert isinstance(qoi_expr, sp.Basic), (
                "QoI expression must evaluate to a numeric expression"
            )
            # check if the expression is well-formed (e.g. no int's that cannot
            #  be printed) and if an error bound can be computed
            _canary_repr = str(qoi_expr)
            _canary_data_eb = _compute_data_eb_for_qoi_eb(
                qoi_expr, self._x, np.empty(0), np.empty(0), np.empty(0)
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

        return frozenset([self._eb]) if isinstance(self._eb, Parameter) else frozenset()

    def evaluate_qoi(
        self, data: np.ndarray[S, np.dtype[T]]
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

        Returns
        -------
        qoi : np.ndarray[S, np.dtype[F]]
            Evaluated quantity of interest, in floating point.
        """

        data_float: np.ndarray = to_float(data)

        qoi_lambda = compile_sympy_expr_to_numpy(
            [self._x], self._qoi_expr, data_float.dtype
        )

        return (qoi_lambda)(data_float)

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
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : np.ndarray
            Pointwise, `True` if the check succeeded for this element.
        """

        data_float: np.ndarray = to_float(data)

        qoi_lambda = compile_sympy_expr_to_numpy(
            [self._x], self._qoi_expr, data_float.dtype
        )

        qoi_data = (qoi_lambda)(data_float)
        qoi_decoded = (qoi_lambda)(to_float(decoded))

        eb = (
            late_bound.resolve_ndarray(
                self._eb,
                qoi_data.shape,
                qoi_data.dtype,
            )
            if isinstance(self._eb, Parameter)
            else self._eb
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
        data : np.ndarray
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the error bound is upheld.
        """

        data_float: np.ndarray = to_float(data)

        qoi_lambda = compile_sympy_expr_to_numpy(
            [self._x], self._qoi_expr, data_float.dtype
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            data_qoi = (qoi_lambda)(data_float)

        eb = (
            late_bound.resolve_ndarray(
                self._eb,
                data_qoi.shape,
                data_qoi.dtype,
            )
            if isinstance(self._eb, Parameter)
            else self._eb
        )
        _check_error_bound(self._type, eb)

        qoi_lower_upper: tuple[np.ndarray, np.ndarray] = _apply_finite_qoi_error_bound(
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

        # compute the error bound in data space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_x_lower, eb_x_upper = _compute_data_eb_for_qoi_eb(
                self._qoi_expr,
                self._x,
                data_float,
                eb_qoi_lower,
                eb_qoi_upper,
            )

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

    Returns
    -------
    eb_x_lower, eb_x_upper : tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]
        Finite pointwise lower and upper error bound on the input data `x`.

    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    tl, tu = compute_data_eb_for_stencil_qoi_eb_unchecked(
        expr,
        xv,
        tauv_lower,
        tauv_upper,
        check_is_x=lambda expr: expr == x,
        evaluate_sympy_expr_to_numpy=lambda expr: compile_sympy_expr_to_numpy(
            [x], expr, xv.dtype
        )(xv),
        compute_data_eb_for_stencil_qoi_eb=lambda expr,
        xv,
        tauv_lower,
        tauv_upper: _compute_data_eb_for_qoi_eb(
            expr,
            x,
            xv,
            tauv_lower,
            tauv_upper,
        ),
    )

    exprl = compile_sympy_expr_to_numpy([x], expr, xv.dtype)
    exprv = (exprl)(xv)

    # handle rounding errors in the lower error bound computation
    tl = ensure_bounded_derived_error(
        lambda tl: np.where(tl == 0, exprv, (exprl)(xv + tl)),  # type: ignore
        exprv,
        xv,
        tl,
        tauv_lower,
        tauv_upper,
    )
    tu = ensure_bounded_derived_error(
        lambda tu: np.where(tu == 0, exprv, (exprl)(xv + tu)),  # type: ignore
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
_QOI_KWARG_PATTERN = (
    r"(?:"
    + r"|".join(
        rf"(?:{k}(?:{QOI_COMMENT_PATTERN.pattern}|(?:[ \t\n]))*=(?:{QOI_COMMENT_PATTERN.pattern}|(?:[ \t\n]))*)"
        for k in ("base",)
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
    + r"".join(rf"|(?:{c})" for c in MATH_CONSTANTS)
    + r"".join(rf"|(?:{f})" for f in MATH_FUNCTIONS)
    + r"".join(rf"|(?:{v})" for v in VARS_FUNCTIONS)
    + r"".join(
        rf'|(?:{v}{QOI_WHITESPACE_PATTERN.pattern}*\[{QOI_WHITESPACE_PATTERN.pattern}*"[a-zA-Z_][a-zA-Z0-9]*"{QOI_WHITESPACE_PATTERN.pattern}*\])'
        for v in VARS_CONSTRUCTORS
    )
    + r")"
)
_QOI_SEPARATOR_PATTERN = rf"(?:{QOI_COMMENT_PATTERN.pattern}|(?:[ \t\n\(\),\+\-\*/]))"
_QOI_PATTERN = re.compile(
    rf"{_QOI_SEPARATOR_PATTERN}*{_QOI_ATOM_PATTERN}(?:{_QOI_SEPARATOR_PATTERN}+{_QOI_KWARG_PATTERN}?{_QOI_ATOM_PATTERN})*{_QOI_SEPARATOR_PATTERN}*"
)

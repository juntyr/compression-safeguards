"""
Quantity of interest (QOI) absolute error bound safeguard.
"""

__all__ = ["QuantityOfInterestAbsoluteErrorBoundSafeguard"]

from typing import Callable

import numpy as np
import sympy as sp

from . import Expr
from ..abc import PointwiseSafeguard, S, T
from ..abs import _compute_safe_eb_abs_interval
from ....cast import (
    as_bits,
    to_float,
    to_finite_float,
    from_total_order,
    to_total_order,
)
from ....intervals import IntervalUnion


class QuantityOfInterestAbsoluteErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `QuantityOfInterestAbsoluteErrorBoundSafeguard` guarantees that the
    pointwise absolute error on a derived quantity of interest (QOI) is less
    than or equal to the provided bound `eb_abs`.

    The quantity of interest is specified as a non-constant expression, in
    string form, on the pointwise value `x`. For example, to bound the error on
    the square of `x`, set `qoi=Expr("x**2")`.

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

    var     = "x";                        (* pointwise data value *)

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

    The implementation of the absolute error bound on pointwise quantities of
    interest is based on:

    > Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. *Proceedings of the VLDB Endowment*.
    16, 4 (December 2022), 697-710. Available from:
    <https://doi.org/10.14778/3574245.3574255>.

    Parameters
    ----------
    qoi : Expr
        The non-constant expression for computing the derived quantity of
        interest for a pointwise value `x`.
    eb_abs : int | float
        The non-negative absolute error bound on the quantity of interest that
        is enforced by this safeguard.
    """

    __slots__ = (
        "_qoi",
        "_eb_abs",
        "_qoi_lambda",
        "_eb_abs_qoi_lambda",
    )
    _qoi: Expr
    _eb_abs: int | float
    _qoi_lambda: Callable[[np.ndarray], np.ndarray]
    _eb_abs_qoi_lambda: Callable[[np.ndarray, np.ndarray], np.ndarray]

    kind = "qoi_abs"

    def __init__(self, qoi: Expr, eb_abs: int | float):
        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert isinstance(eb_abs, int) or np.isfinite(eb_abs), "eb_abs must be finite"

        self._qoi = qoi
        self._eb_abs = eb_abs

        x = sp.Symbol("x", real=True)

        try:
            qoi_expr = sp.parse_expr(
                self._qoi,
                local_dict=dict(x=x),
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
            self._qoi_lambda = sp.lambdify(x, qoi_expr, modules="numpy", cse=True)
        except Exception as err:
            raise AssertionError(f"failed to parse qoi expression {qoi!r}") from err

        tau = sp.Symbol("tau", real=True, nonnegative=True)

        try:
            eb_abs_qoi = _derive_eb_abs_qoi(qoi_expr, x, tau)
            self._eb_abs_qoi_lambda = sp.lambdify(
                [x, tau],
                eb_abs_qoi,
                modules="numpy",
                cse=True,
            )
        except Exception as err:
            raise AssertionError(
                f"failed to derive error bound for qoi expression {qoi!r}"
            ) from err

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the absolute error
        bound for the quantity of interest on the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Per-element, `True` if the check succeeded for this element.
        """

        qoi_data = (self._qoi_lambda)(to_float(data))
        qoi_decoded = (self._qoi_lambda)(to_float(decoded))

        absolute_bound = (
            np.where(
                qoi_data > qoi_decoded,
                qoi_data - qoi_decoded,
                qoi_decoded - qoi_data,
            )
            <= self._eb_abs
        )
        same_bits = as_bits(qoi_data, kind="V") == as_bits(qoi_decoded, kind="V")
        both_nan = np.isnan(qoi_data) & np.isnan(qoi_decoded)

        ok = np.where(
            np.isfinite(qoi_data),
            absolute_bound,
            np.where(
                np.isinf(qoi_data),
                same_bits,
                both_nan,
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the absolute error bound is upheld with
        respect to the quantity of interest on the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the absolute error bound is upheld.
        """

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs: np.ndarray = to_finite_float(self._eb_abs, data_float.dtype)
        assert eb_abs >= 0.0

        # ensure the error bounds are representable in QOI space
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # compute the error-bound adjusted QOIs
            data_qoi = (self._qoi_lambda)(data_float)
            qoi_lower = data_qoi - eb_abs
            qoi_upper = data_qoi + eb_abs

            # check if they're representable within the error bound
            qoi_lower_outside_eb_abs = (data_qoi - qoi_lower) > eb_abs
            qoi_upper_outside_eb_abs = (qoi_upper - data_qoi) > eb_abs

            # otherwise nudge the error-bound adjusted QOIs
            qoi_lower = from_total_order(
                to_total_order(qoi_lower) + qoi_lower_outside_eb_abs,
                data_float.dtype,
            )
            qoi_upper = from_total_order(
                to_total_order(qoi_upper) - qoi_upper_outside_eb_abs,
                data_float.dtype,
            )

            # compute the adjusted error bound
            eb_abs_lower = data_qoi - qoi_lower
            eb_abs_upper = qoi_upper - data_qoi
            eb_abs = np.minimum(eb_abs_lower, eb_abs_upper)
        eb_abs = np.nan_to_num(eb_abs, nan=0.0, posinf=0.0, neginf=None)
        assert np.all(eb_abs >= 0.0)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi = (self._eb_abs_qoi_lambda)(data_float, eb_abs)
        eb_abs_qoi = np.nan_to_num(eb_abs_qoi, nan=0.0, posinf=0.0, neginf=None)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi_float: np.ndarray = to_finite_float(eb_abs_qoi, data_float.dtype)
        assert np.all(eb_abs_qoi_float >= 0.0)

        return _compute_safe_eb_abs_interval(
            data.flatten(),
            data_float.flatten(),
            eb_abs_qoi_float.flatten(),  # type: ignore
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

        return dict(kind=type(self).kind, qoi=self._qoi, eb_abs=self._eb_abs)


def _derive_eb_abs_qoi(
    expr: sp.Basic,
    x: sp.Basic,
    tau: sp.Basic,
) -> sp.Basic:
    """
    Inspired by:

    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697-710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    assert len(expr.free_symbols) > 0, "constants have no error bounds"

    # x
    if expr == x:
        return tau

    # ln(...)
    # sympy automatically transforms log(..., base) into ln(...)/ln(base)
    if expr.func is sp.log and len(expr.args) == 1:
        (arg,) = expr.args
        if arg == x:
            # base case ln(x), derived using sympy
            return sp.Abs(x) * sp.Min(
                sp.Abs(sp.exp(tau) - 1), sp.Abs(sp.exp(tau * sp.Integer(-1)) - 1)
            )
        else:
            # composition using Lemma 3 from Jiao et al.
            tau = _derive_eb_abs_qoi(expr, arg, tau)
            return _derive_eb_abs_qoi(arg, x, tau)

    # e^(...)
    if expr.func is sp.exp and len(expr.args) == 1:
        (arg,) = expr.args
        if arg == x:
            # base case exp(x), derived using sympy
            return sp.Min(
                sp.Abs(sp.log(tau * sp.exp(x * sp.Integer(-1)) + 1)),
                sp.Abs(sp.log(tau * sp.Integer(-1) * sp.exp(x * sp.Integer(-1)) + 1)),
            )
        else:
            # composition using Lemma 3 from Jiao et al.
            tau = _derive_eb_abs_qoi(expr, arg, tau)
            return _derive_eb_abs_qoi(arg, x, tau)

    # rewrite a ** b as e^(b*log(a))
    if expr.is_Pow and len(expr.args) == 2:
        a, b = expr.args
        return _derive_eb_abs_qoi(sp.exp(b * sp.ln(a), evaluate=False), x, tau)

    # a_1 * e_1 + ... + a_n * e_n + c (weighted sum)
    # using Corollary 2 and Lemma 4 from Jiao et al.
    if expr.is_Add:
        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        abs_factors = []
        for term in terms:
            # extract the weighting factor of the term
            # and take its absolute value
            if term.is_Mul:
                abs_factors.append(
                    sp.Abs(
                        sp.Mul(
                            *[arg for arg in term.args if len(arg.free_symbols) == 0]  # type: ignore
                        )
                    )
                )
            else:
                abs_factors.append(sp.Integer(1))
        total_abs_factor = sp.Add(*abs_factors)

        ebs = []
        for term, factor in zip(terms, abs_factors):
            # recurse into the terms with a weighted error bound
            # we have already checked that the terms are non-const,
            #  so the returned error bound must not be None
            ebs.append(
                _derive_eb_abs_qoi(
                    term,
                    x,
                    tau * factor / total_abs_factor,
                )
            )

        # combine the inner error bounds
        return sp.Min(*ebs)

    # e_1 * ... * e_n (product) using Corollary 3 from Jiao et al.
    if expr.is_Mul:
        # extract the constant factor and reduce tau
        factor = sp.Mul(*[arg for arg in expr.args if len(arg.free_symbols) == 0])  # type: ignore
        tau = tau / sp.Abs(factor)

        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        if len(terms) == 1:
            return _derive_eb_abs_qoi(terms[0], x, tau)

        # recurse as if the multiplication was a binary tree
        tleft, tright = terms[: len(terms) // 2], terms[len(terms) // 2 :]
        fp = sp.Abs(sp.Mul(*tleft)) + sp.Abs(sp.Mul(*tright))  # type: ignore

        # conservative error bound for multiplication (Jiao et al.)
        tau = sp.Abs((-fp + sp.sqrt(sp.Integer(4) * tau + sp.Pow(fp, 2))) / 2)

        ebs = []
        for tbranch in (tleft, tright):
            # recurse into the terms with the adapted error bound
            # we have already checked that the terms are non-const,
            #  so the returned error bound must not be None
            ebs.append(_derive_eb_abs_qoi(sp.Mul(*tbranch), x, tau))  # type: ignore

        # combine the inner error bounds
        return sp.Min(*ebs)

    raise ValueError(f"unsupported expression kind {expr} (= {sp.srepr(expr)} =)")

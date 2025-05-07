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
        "_qoi_expr",
        "_qoi_lambda",
    )
    _qoi: Expr
    _eb_abs: int | float
    _qoi_expr: sp.Basic
    _qoi_lambda: Callable[[np.ndarray], np.ndarray]

    kind = "qoi_abs"

    def __init__(self, qoi: Expr, eb_abs: int | float):
        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert isinstance(eb_abs, int) or np.isfinite(eb_abs), "eb_abs must be finite"

        self._qoi = qoi
        self._eb_abs = eb_abs

        x = sp.Symbol("x", real=True)

        try:
            self._qoi_expr = sp.parse_expr(
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
            self._qoi_lambda = sp.lambdify(x, self._qoi_expr, modules="numpy", cse=True)
        except Exception as err:
            raise AssertionError(
                f"failed to parse qoi expression {qoi!r}: {err}"
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
            eb_abs_lower = qoi_lower - data_qoi
            eb_abs_upper = qoi_upper - data_qoi
        eb_abs_lower = np.nan_to_num(eb_abs_lower, nan=0.0, posinf=None, neginf=0.0)
        eb_abs_upper = np.nan_to_num(eb_abs_upper, nan=0.0, posinf=0.0, neginf=None)
        assert np.all(eb_abs_lower <= 0.0)
        assert np.all(eb_abs_upper >= 0.0)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi_lower, eb_abs_qoi_upper = _compute_eb_abs_qoi(
                self._qoi_expr, sp.Symbol("x", real=True), data_float, eb_abs_lower, eb_abs_upper,
            )
        eb_abs_qoi_lower = np.nan_to_num(eb_abs_qoi_lower, nan=0.0, posinf=None, neginf=0.0)
        eb_abs_qoi_upper = np.nan_to_num(eb_abs_qoi_upper, nan=0.0, posinf=0.0, neginf=None)
        print(self._eb_abs, eb_abs_qoi_lower, eb_abs_qoi_upper, np.abs((self._qoi_lambda)(data_float + eb_abs_qoi_lower) - (self._qoi_lambda)(data_float)), np.abs((self._qoi_lambda)(data_float + eb_abs_qoi_upper) - (self._qoi_lambda)(data_float)))

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi_lower_float: np.ndarray = to_finite_float(eb_abs_qoi_lower, data_float.dtype)
            eb_abs_qoi_upper_float: np.ndarray = to_finite_float(eb_abs_qoi_upper, data_float.dtype)
        assert np.all(eb_abs_qoi_lower_float <= 0.0)
        assert np.all(eb_abs_qoi_upper_float >= 0.0)

        return _compute_safe_eb_abs_interval(
            data.flatten(),
            data_float.flatten(),
            np.minimum(np.abs(eb_abs_qoi_lower_float), np.abs(eb_abs_qoi_upper_float)).flatten(),  # type: ignore
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


def _compute_eb_abs_qoi(
    expr: sp.Basic,
    x: sp.Basic,
    xv: np.ndarray,
    tauv_lower: np.ndarray,
    tauv_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Compute for {expr} with x={xv} for {tauv_lower} <= e <= {tauv_upper}")

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
        return (tauv_lower, tauv_upper)
    
    if expr.func is sp.Abs and len(expr.args) == 1:
        (arg,) = expr.args
        return _compute_eb_abs_qoi(arg, x, xv, tauv_lower, tauv_upper)

    # ln(...)
    # sympy automatically transforms log(..., base) into ln(...)/ln(base)
    if expr.func is sp.log and len(expr.args) == 1:
        (arg,) = expr.args
        argv = sp.lambdify([x], arg, "numpy")(xv)
        # base case ln(x), derived using sympy
        # FIXME: we need to also ensure that x doesn't become an invalid non-positive input
        tauv_lower = -np.abs(argv * (np.exp(-tauv_lower) - 1))
        tauv_upper = np.abs(argv * (np.exp(tauv_upper) - 1))
        if arg == x:
            return tauv_lower, tauv_upper
        return _compute_eb_abs_qoi(arg, x, xv, tauv_lower, tauv_upper)

    # e^(...)
    if expr.func is sp.exp and len(expr.args) == 1:
        (arg,) = expr.args
        argv = sp.lambdify([x], arg, "numpy")(xv)
        # base case exp(x), derived using sympy
        tauv_lower = -np.abs(np.log(-tauv_lower * np.exp(-argv) + 1))
        tauv_upper = np.abs(np.log(tauv_upper * np.exp(-argv) + 1))
        if arg == x:
            return tauv_lower, tauv_upper
        return _compute_eb_abs_qoi(arg, x, xv, tauv_lower, tauv_upper)

    # rewrite a ** b as e^(b*log(a))
    if expr.is_Pow and len(expr.args) == 2:
        a, b = expr.args
        return _compute_eb_abs_qoi(sp.exp(b * sp.ln(sp.Abs(a)), evaluate=False), x, xv, tauv_lower, tauv_upper)

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
                    sp.lambdify(
                        [],
                        sp.Abs(
                            sp.Mul(
                                *[
                                    arg
                                    for arg in term.args
                                    if len(arg.free_symbols) == 0
                                ]  # type: ignore
                            )
                        ),
                        "numpy",
                    )()
                )
            else:
                abs_factors.append(1)
        total_abs_factor = np.sum(abs_factors)

        ebs_lower, ebs_upper = None, None
        for term, factor in zip(terms, abs_factors):
            # recurse into the terms with a weighted error bound
            eb_lower, eb_upper = _compute_eb_abs_qoi(
                term,
                x,
                xv,
                tauv_lower * factor / total_abs_factor,
                tauv_upper * factor / total_abs_factor,
            )
            if ebs_lower is None:
                ebs_lower = eb_lower
                ebs_upper = eb_upper
            else:
                ebs_lower = np.maximum(ebs_lower, eb_lower)
                ebs_upper = np.minimum(ebs_upper, eb_upper)

        # combine the inner error bounds
        return ebs_lower, ebs_upper

    # e_1 * ... * e_n (product) using Corollary 3 from Jiao et al.
    if expr.is_Mul:
        # extract the constant factor and reduce tau
        factor = sp.lambdify(
            [],
            sp.Mul(*[arg for arg in expr.args if len(arg.free_symbols) == 0]),
            "numpy",
        )()  # type: ignore
        tauv_lower = tauv_lower / np.abs(factor)
        tauv_upper = tauv_upper / np.abs(factor)

        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        if len(terms) == 1:
            return _compute_eb_abs_qoi(terms[0], x, xv, tauv_lower, tauv_upper)

        return _compute_eb_abs_qoi(
            sp.exp(sp.Add(*[sp.log(term) for term in terms]), evaluate=False),
            x,
            xv,
            tauv_lower,
            tauv_upper,
        )

    raise ValueError(f"unsupported expression kind {expr} (= {sp.srepr(expr)} =)")

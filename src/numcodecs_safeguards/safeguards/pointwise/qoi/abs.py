"""
Quantity of interest (QOI) absolute error bound safeguard.
"""

__all__ = ["QuantityOfInterestAbsoluteErrorBoundSafeguard"]

from typing import Callable

import numpy as np
import sympy as sp

from ..abc import PointwiseSafeguard, S, T
from ..abs import _compute_safe_eb_abs_interval
from ....cast import as_bits, to_float, to_finite_float
from ....intervals import IntervalUnion


class QuantityOfInterestAbsoluteErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `QuantityOfInterestAbsoluteErrorBoundSafeguard` guarantees that the
    pointwise absolute error for a derived quantity of interest (QOI) is less
    than or equal to the provided bound `eb_abs`.

    The quantity of interest is specified as a non-constant expression, in
    string form, on the pointwise value `x`. For example, to bound the error on
    the square of `x`, set `qoi="x**2"`. The following operations are
    supported, where `...` denotes any expression:

    - integer and floating point constants
    - pointwise value `x`
    - addition `(...) + (...)`
    - multiplication `(...) * (...)`
    - division `(...) / (...)`
    - square root `sqrt(...)`
    - exponentiation `(...) ** (...)`
      - either the base or the exponent must be constant
      - if the exponent is constant, it must be an integer (or 1/2 for `sqrt`)
    - exponential `exp(...)`
    - natural logarithm `log(...)`
      - optionally a different base can be provided second in `log(..., ...)`

    If the derived quantity of interest for an element evaluates to an infinite
    value, this safeguard guarantees that the quantity of interest on the
    decoded value produces the exact same infinite value. For a NaN quantity of
    interest, this safeguard guarantees that the quantity of interest on the
    decoded value is also NaN, but does not guarantee that it has the same
    bitpattern.

    Parameters
    ----------
    qoi : str
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
    _qoi: str
    _eb_abs: int | float
    _qoi_lambda: Callable[[np.ndarray], np.ndarray]
    _eb_abs_qoi_lambda: Callable[[np.ndarray, np.ndarray], np.ndarray]

    kind = "qoi_abs"

    def __init__(self, qoi: str, eb_abs: int | float):
        self._qoi = qoi
        self._eb_abs = eb_abs

        x = sp.Symbol("x", real=True)
        tau = sp.Symbol("tau", real=True, positive=True)

        qoi_expr = sp.parse_expr(
            self._qoi,
            local_dict=dict(x=x),
            transformations=(sp.parsing.sympy_parser.auto_number,),
        ).simplify()
        self._qoi_lambda = sp.lambdify(x, qoi_expr, modules="numpy", cse=True)

        eb_abs_qoi = _derive_eb_abs_qoi(qoi_expr, x, tau, True).simplify()
        self._eb_abs_qoi_lambda = sp.lambdify(
            [x, tau],
            eb_abs_qoi,
            modules="numpy",
            cse=True,
        )

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

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi = (self._eb_abs_qoi_lambda)(data_float, eb_abs)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi_float: np.ndarray = to_finite_float(eb_abs_qoi, data_float.dtype)
        assert eb_abs_qoi_float >= 0.0

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
    allow_composition: bool,
) -> sp.Basic:
    """
    Inspired by:
    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697–710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    assert len(expr.free_symbols) > 0, "constants have no error bounds"

    if expr == x:
        return tau

    # support sqrt(x) using Jiao et al.
    # modified s.t. we never have sqrt(-...)
    if expr.is_Pow and expr.args == (x, sp.Rational(1, 2)):
        return sp.Min(
            sp.Abs(sp.Pow(tau, 2) - sp.Integer(2) * tau * sp.sqrt(x)), sp.Abs(x)
        )

    # support ln(x) using Jiao et al.
    if expr.func is sp.functions.elementary.exponential.log and expr.args == (x,):
        return sp.Abs(x) * sp.Min(1 - sp.exp(tau * sp.Integer(-1)), sp.exp(tau) - 1)

    # support exp(x), derived using sympy
    if expr.func is sp.functions.elementary.exponential.exp and expr.args == (x,):
        return sp.log(tau * sp.exp(x * sp.Integer(-1)) + 1)

    # support (const)**(x), derived using sympy
    if (
        expr.is_Pow
        and len(expr.args) == 2
        and len(expr.args[0].free_symbols) == 0
        and expr.args[1] == x
    ):
        b = expr.args[0]
        return sp.Abs(x * sp.Integer(-1) + sp.log(sp.Pow(b, x) + tau, b))

    # support 1/x, derived using sympy
    if expr.is_Pow and expr.args == (x, sp.Integer(-1)):
        return sp.Min(
            sp.Abs(tau * sp.Pow(x, 2) / (tau * x - 1)),  # type: ignore
            sp.Abs(tau * sp.Pow(x, 2) / (tau * x + 1)),  # type: ignore
        )

    # support weighted sums through recursion
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
                    True,
                )
            )

        # combine the inner error bounds
        return sp.Min(*ebs)

    # support multiplication through recursion
    if expr.is_Mul:
        # extract the constant factor and reduce tau
        factor = sp.Mul(*[arg for arg in expr.args if len(arg.free_symbols) == 0])  # type: ignore
        tau = tau / sp.Abs(factor)

        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        if len(terms) == 1:
            return _derive_eb_abs_qoi(terms[0], x, tau, True)

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
            ebs.append(_derive_eb_abs_qoi(sp.Mul(*tbranch), x, tau, True))  # type: ignore

        # combine the inner error bounds
        return sp.Min(*ebs)

    # support positive integer powers via multiplication
    if (
        expr.is_Pow
        and len(expr.args) == 2
        and expr.args[1].is_Integer
        and expr.args[1] > sp.Integer(1)
    ):
        # split the power into the product of two terms
        return _derive_eb_abs_qoi(
            sp.Mul(
                sp.Pow(expr.args[0], expr.args[1] // sp.Integer(2)),
                sp.Pow(expr.args[0], expr.args[1] - (expr.args[1] // sp.Integer(2))),
                evaluate=False,
            ),
            x,
            tau,
            True,
        )

    # support negative powers via inverse
    if (
        expr.is_Pow
        and len(expr.args) == 2
        and expr.args[1].is_Number
        and expr.args[1] < sp.Integer(0)
        and expr.args[1] != -1
    ):
        # split the power into its inverse
        return _derive_eb_abs_qoi(
            sp.Pow(
                sp.Pow(expr.args[0], expr.args[1] * sp.Integer(-1)), -1, evaluate=False
            ),
            x,
            tau,
            True,
        )

    if allow_composition:
        if (
            expr.func
            in (
                sp.functions.elementary.exponential.exp,
                sp.functions.elementary.exponential.log,
            )
            and len(expr.args) == 1
        ):
            expr_inner = expr.args[0]

            eb = _derive_eb_abs_qoi(expr, expr_inner, tau, False)

            return _derive_eb_abs_qoi(expr_inner, x, eb, True)

        if (
            expr.is_Pow
            and len(expr.args) == 2
            and (
                len(expr.args[0].free_symbols) == 0
                or len(expr.args[1].free_symbols) == 0
            )
        ):
            expr_inner = expr.args[int(len(expr.args[0].free_symbols) == 0)]

            eb = _derive_eb_abs_qoi(expr, expr_inner, tau, False)

            return _derive_eb_abs_qoi(expr_inner, x, eb, True)

    raise TypeError(f"unsupported expression kind {expr} ({sp.srepr(expr)})")

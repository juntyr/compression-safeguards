"""
Quantity of interest (QOI) safeguard.
"""

__all__ = ["QuantityOfInterestSafeguard"]

import numpy as np
import sympy as sp
from sympy import Symbol, Basic
from sympy.parsing.sympy_parser import parse_expr, auto_number
from sympy.utilities.lambdify import lambdify

from .abc import PointwiseSafeguard, S, T
from .abs import _compute_safe_eb_abs_interval
from ...cast import as_bits, to_float, to_finite_float
from ...intervals import IntervalUnion


class QuantityOfInterestSafeguard(PointwiseSafeguard):
    __slots__ = (
        "_qoi",
        "_eb_abs",
        "_qoi_lambda",
        "_eb_abs_qoi_lambda",
    )
    _qoi: str
    _eb_abs: int | float

    kind = "qoi"

    def __init__(self, qoi: str, eb_abs: int | float):
        self._qoi = qoi
        self._eb_abs = eb_abs

        x = Symbol("x", real=True)
        tau = Symbol("tau", real=True, positive=True)

        qoi_expr = parse_expr(
            self._qoi, local_dict=dict(x=x), transformations=(auto_number,)
        ).simplify()
        print(qoi_expr)
        self._qoi_lambda = lambdify(x, qoi_expr, modules="numpy", cse=True)

        eb_abs_qoi = _derive_eb_abs_qoi(qoi_expr, x, tau)
        print(eb_abs_qoi)
        if eb_abs_qoi is None:
            self._eb_abs_qoi_lambda = lambda x: np.full_like(x, None)
        else:
            eb_abs_qoi_abs = sp.Min(*[abs(eb) for eb in eb_abs_qoi])  # type: ignore
            self._eb_abs_qoi_lambda = lambdify(
                x,
                eb_abs_qoi_abs.subs(tau, self._eb_abs).simplify(),
                modules="numpy",
                cse=True,
            )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
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
        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs_qoi = (self._eb_abs_qoi_lambda)(data_float)

        # TODO: handle None error bounds

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
    expr: Basic, x: Symbol, tau: Basic
) -> None | tuple[Basic, Basic]:
    """
    Based on:
    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697–710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    # constants have no error bounds
    if len(expr.free_symbols) == 0:
        return None

    # first try to solve symbolically
    eb_abs_sym = _try_solve_eb_abs_qoi(expr, x, tau)
    if eb_abs_sym is not None:
        return eb_abs_sym

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
                    abs(
                        sp.Mul(
                            *[arg for arg in term.args if len(arg.free_symbols) == 0]
                        )
                    )
                )
            else:
                abs_factors.append(sp.Integer(1))
        total_abs_factor = sp.Add(*abs_factors)

        ebls, ebus = [], []
        for term, factor in zip(terms, abs_factors):
            # recurse into the terms with a weighted error bound
            # we have already checked that the terms are non-const,
            #  so the returned error bound must not be None
            ebl, ebu = _derive_eb_abs_qoi(
                term,
                x,
                tau * factor / total_abs_factor,
            )  # type: ignore
            ebls.append(ebl)
            ebus.append(ebu)

        # combine the inner error bounds:
        # - maximum of all (non-positive) lower bounds
        # - minimum of all (non-negative) upper bounds
        return sp.Max(*ebls), sp.Min(*ebus)

    raise TypeError(f"unsupported expression kind {expr} ({sp.srepr(expr)})")


def _try_solve_eb_abs_qoi(
    expr: Basic, x: Symbol, tau: Basic
) -> None | tuple[Basic, Basic]:
    # symbol for the error bound on the raw data
    e = sp.Symbol("e", real=True)

    with sp.evaluate(False):
        f_x = expr
        f_xe = expr.subs(x, x + e)

    # try to solve |f(x+e) - f(x)| <= tau
    #  using squares since sympy better supports them
    # if solving fails, return None
    try:
        ebs: list[Basic] = sp.solve((f_xe - f_x) ** 2 - tau**2, e)  # type: ignore
    except NotImplementedError:
        return None

    # if there are no solutions, return None
    if len(ebs) == 0:
        return None

    # bail if any solution contains imaginary numbers
    if any(eb.has(sp.I) for eb in ebs):
        return None

    # lower eb: largest non-positive error bound, or zero
    eb_lower_inf = sp.Max(
        *[
            sp.Piecewise(
                (eb, eb <= 0),  # type: ignore
                (-sp.oo, True),
            )
            for eb in ebs
        ]
    )
    eb_lower = sp.Piecewise((eb_lower_inf, eb_lower_inf > (-sp.oo)), (0, True))

    # upper eb: smallest non-negative error bound, or zero
    eb_upper_inf = sp.Min(
        *[
            sp.Piecewise(
                (eb, eb >= 0),  # type: ignore
                (sp.oo, True),
            )
            for eb in ebs
        ]
    )
    eb_upper = sp.Piecewise((eb_upper_inf, eb_upper_inf < sp.oo), (0, True))

    return (eb_lower, eb_upper)

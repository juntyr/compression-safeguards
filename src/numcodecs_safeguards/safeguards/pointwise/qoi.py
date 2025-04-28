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

        x = Symbol("x")

        qoi_expr = parse_expr(
            self._qoi, local_dict=dict(x=x), transformations=(auto_number,)
        ).simplify()
        print(qoi_expr)
        self._qoi_lambda = lambdify(x, qoi_expr, modules="numpy", cse=True)

        eb_abs_qoi = _derive_eb_abs_qoi(qoi_expr, x, sp.sympify(self._eb_abs))
        print(eb_abs_qoi)
        if eb_abs_qoi is None:
            self._eb_abs_qoi_lambda = lambda x: np.full_like(x, None)
        else:
            self._eb_abs_qoi_lambda = lambdify(
                x, eb_abs_qoi.simplify(), modules="numpy", cse=True
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


def _derive_eb_abs_qoi(expr: Basic, x: Symbol, eb_abs: Basic) -> None | Basic:
    """
    Based on:
    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697–710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    # Q(const, tau, x) = any
    if len(expr.free_symbols) == 0:
        return None
    # Q(x, tau, x) = tau
    if expr.is_Symbol and expr == x:
        return eb_abs
    # Q(sum(...), tau, x)
    elif expr.is_Add:
        # optimisation: only count the entries without x
        num_non_const = sum(len(arg.free_symbols) > 0 for arg in expr.args)
        eb_abs = eb_abs / sp.Integer(num_non_const)
        res = list(
            filter(
                lambda x: x is not None,
                (_derive_eb_abs_qoi(arg, x, eb_abs) for arg in expr.args),
            )
        )
        return sp.Min(*res) if len(res) > 0 else None  # type: ignore
    # TODO: handle ax+b better
    # TODO: should we try to decompose polynomials into linear QoIs?
    # elif expr.is_Mul and len(expr.args) == 2 and expr.args[0].is_Number and expr.args[1] == x:
    #     if expr.args[0] == 0:
    #         return None
    #     return eb_abs / abs(expr.args[0])
    elif expr.is_Mul:
        if len(expr.args) == 0:
            return None
        if len(expr.args) == 1:
            return _derive_eb_abs_qoi(expr.args[0], x, eb_abs)
        left, right = expr.args[: len(expr.args) // 2], expr.args[len(expr.args) // 2 :]
        fp = abs(sp.Mul(*left)) + abs(sp.Mul(*right))  # type: ignore
        eb_abs = (-fp + sp.sqrt(eb_abs * sp.Integer(4) + fp * fp)) / sp.Integer(2)
        res = list(
            filter(
                lambda x: x is not None,
                (
                    _derive_eb_abs_qoi(sp.Mul(*args), x, eb_abs)
                    for args in (left, right)
                ),
            )
        )
        return sp.Min(*res) if len(res) > 0 else None  # type: ignore
    elif (
        expr.is_Pow
        and len(expr.args) == 2
        and expr.args[1].is_Integer
        and expr.args[1] >= 0  # type: ignore
    ):
        if expr.args[1] == 0:
            return None
        return _derive_eb_abs_qoi(
            sp.Mul(
                *[expr.args[0]] * int(expr.args[1]),  # type: ignore
                evaluate=False,
            ),
            x,
            eb_abs,
        )
    # Q(sqrt(x), tau, x) = tau^2 - 2*tau*sqrt(x)
    elif (
        expr.is_Pow
        and len(expr.args) == 2
        and expr.args[0] == x
        and expr.args[1] == sp.Rational(1, 2)
    ):
        return eb_abs**2 - 2 * eb_abs * sp.sqrt(x)  # type: ignore
    elif (
        expr.func is sp.functions.elementary.exponential.log
        and (len(expr.args) >= 1)
        and (len(expr.args) <= 2)
        and expr.args[0] == x
        and (
            len(expr.args) == 1
            or (
                expr.args[1].is_Number and expr.args[1] > 1  # type: ignore
            )
        )
    ):
        b = sp.E if len(expr.args) == 1 else expr.args[1]
        return abs(x) * sp.Min(1 - b ** (-eb_abs), b**eb_abs - 1)  # type: ignore
    else:
        raise TypeError(f"unsupported expression kind {expr} ({sp.srepr(expr)})")

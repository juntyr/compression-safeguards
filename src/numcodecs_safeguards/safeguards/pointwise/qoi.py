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

        eb_abs_qoi = _derive_eb_abs_qoi(qoi_expr, x, tau, True)
        print(eb_abs_qoi)
        if eb_abs_qoi is None:
            self._eb_abs_qoi_lambda = lambda x: np.full_like(x, None)
        else:
            self._eb_abs_qoi_lambda = lambdify(
                x,
                eb_abs_qoi.subs(tau, self._eb_abs).simplify(),
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
    expr: Basic,
    x: Symbol,
    tau: Basic,
    allow_composition: bool,
) -> None | Basic:
    """
    Inspired by:
    Pu Jiao, Sheng Di, Hanqi Guo, Kai Zhao, Jiannan Tian, Dingwen Tao, Xin
    Liang, and Franck Cappello. (2022). Toward Quantity-of-Interest Preserving
    Lossy Compression for Scientific Data. Proc. VLDB Endow. 16, 4 (December
    2022), 697–710. Available from: https://doi.org/10.14778/3574245.3574255.
    """

    # constants have no error bounds
    if len(expr.free_symbols) == 0:
        return None

    if expr == x:
        return tau

    # first try to solve symbolically
    # but only for simple expressions
    if (len(expr.args) == 1 and expr.args[0] == x) or (
        len(expr.args) == 2
        and (
            (expr.args[0] == x and len(expr.args[1].free_symbols) == 0)
            or (expr.args[1] == x and len(expr.args[0].free_symbols) == 0)
        )
        and (not expr.is_Pow or not expr.args[1].is_Number or abs(expr.args[1]) <= 1)
    ):
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
        factor = sp.Mul(*[arg for arg in expr.args if len(arg.free_symbols) == 0])
        tau = tau / abs(factor)  # type: ignore

        # find all non-constant terms
        terms = [arg for arg in expr.args if len(arg.free_symbols) > 0]

        if len(terms) == 1:
            return _derive_eb_abs_qoi(terms[0], x, tau, True)

        # recurse as if the multiplication was a binary tree
        tleft, tright = terms[: len(terms) // 2], terms[len(terms) // 2 :]
        fp = abs(sp.Mul(*tleft)) + abs(sp.Mul(*tright))  # type: ignore

        # conservative error bound for multiplication (Jiao et al.)
        tau = abs((-fp + sp.sqrt(4 * tau + fp**2)) / 2)  # type: ignore

        ebs = []
        for tbranch in (tleft, tright):
            # recurse into the terms with the adapted error bound
            # we have already checked that the terms are non-const,
            #  so the returned error bound must not be None
            ebs.append(_derive_eb_abs_qoi(sp.Mul(*tbranch), x, tau, True))

        # combine the inner error bounds
        return sp.Min(*ebs)

    # support positive integer powers via multiplication
    if (
        expr.is_Pow
        and len(expr.args) == 2
        and expr.args[1].is_Integer
        and abs(expr.args[1]) > 1  # type: ignore
    ):
        # split the power into the product of two terms
        return _derive_eb_abs_qoi(
            sp.Mul(
                sp.Pow(expr.args[0], expr.args[1] // 2),  # type: ignore
                sp.Pow(expr.args[0], expr.args[1] - (expr.args[1] // 2)),  # type: ignore
                evaluate=False,
            ),
            x,
            tau,
            True,
        )

    if allow_composition:
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


_SOLVE_CACHE: dict[Basic, None | list[Basic]] = dict()


def _try_solve_eb_abs_qoi(
    expr: Basic,
    x: Basic,
    tau: Basic,
) -> None | Basic:
    global _SOLVE_CACHE

    # symbol for tau without including tau's complexity in the solve
    t = sp.Symbol("t", real=True, positive=True)

    if expr in _SOLVE_CACHE:
        ebs = _SOLVE_CACHE[expr]
    else:
        ebs = _try_solve_eb_abs_qoi_inner(expr, x, t)
        # print("\n".join(f" = {eb} [{eb.is_real}]" for eb in ebs))
        _SOLVE_CACHE[expr] = ebs

    # if there are no solutions, return None
    if ebs is None or len(ebs) == 0:
        return None

    ebs = [eb.subs(t, sp.Abs(tau)).simplify() for eb in ebs]

    return sp.Min(*[abs(eb) for eb in ebs])


def _try_solve_eb_abs_qoi_inner(
    expr: Basic,
    x: Basic,
    t: Symbol,
) -> None | list[Basic]:
    # symbol for the error bound on the raw data
    e = sp.Symbol("e", real=True)

    with sp.evaluate(False):
        f_x = expr
        f_xe = expr.subs(x, x + e)

    print(f"Try solve {f_x} with x=({x}) for {f_xe}")

    # try to solve |f(x+e) - f(x)| <= tau
    #  using squares since sympy better supports them
    # if solving fails, return None
    try:
        ebs: list[Basic] = sp.solve((f_xe - f_x) ** 2 - t**2, e)  # type: ignore
    except NotImplementedError:
        print("nope")
        return None

    # print(ebs)
    # print(sp.solveset((f_xe - f_x) ** 2 - t**2, e, sp.S.Reals))

    if len(ebs) == 0:
        print("nope")

    return ebs

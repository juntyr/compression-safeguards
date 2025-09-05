from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds
from .abc import Expr
from .literal import Number
from .typing import F, Ns, Ps, PsI


class Group(Expr[Expr]):
    __slots__ = ("_expr",)
    _expr: Expr

    def __new__(cls, expr: Expr):
        if isinstance(expr, (Number, Group)):
            return expr
        this = super(Group, cls).__new__(cls)
        this._expr = expr
        return this

    @property
    def args(self) -> tuple[Expr]:
        return (self._expr,)

    def with_args(self, expr: Expr) -> "Group":
        return Group(expr)

    def constant_fold(self, dtype: np.dtype[F]) -> F | "Expr":
        fexpr = self._expr.constant_fold(dtype)
        # partially / not constant folded -> stop further folding
        if isinstance(fexpr, Expr):
            return Group(fexpr)
        # fully constant folded -> allow further folding
        return fexpr

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return self._expr.eval(x, Xs, late_bound)

    @data_bounds(DataBounds.infallible)
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        return self._expr.compute_data_bounds(expr_lower, expr_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"({self._expr!r})"

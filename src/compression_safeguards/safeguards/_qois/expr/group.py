from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds
from .abc import AnyExpr, Expr
from .literal import Number
from .typing import F, Ns, Ps, PsI


class Group(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_expr",)
    _expr: AnyExpr

    def __init__(self, expr: AnyExpr) -> None:
        self._expr = expr

    def __new__(cls, expr: AnyExpr) -> "Group | Number":  # type: ignore[misc]
        if isinstance(expr, Number | Group):
            return expr
        return super().__new__(cls)

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._expr,)

    @override
    def with_args(self, expr: AnyExpr) -> "Group | Number":
        return Group(expr)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        fexpr = self._expr.constant_fold(dtype)
        # partially / not constant folded -> stop further folding
        if isinstance(fexpr, Expr):
            return Group(fexpr)
        # fully constant folded -> allow further folding
        return fexpr

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return self._expr.eval(x, Xs, late_bound)

    @data_bounds(DataBounds.infallible)
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        return self._expr.compute_data_bounds(expr_lower, expr_upper, X, Xs, late_bound)

    @override
    def __repr__(self) -> str:
        return f"({self._expr!r})"

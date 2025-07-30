from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.typing import F, S
from .abc import Expr


class Group(Expr):
    __slots__ = ("_expr",)
    _expr: Expr

    def __init__(self, expr: Expr):
        self._expr = expr._expr if isinstance(expr, Group) else expr

    @property
    def has_data(self) -> bool:
        return self._expr.has_data

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._expr.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | "Expr":
        fexpr = self._expr.constant_fold(dtype)
        # partially / not constant folded -> stop further folding
        if isinstance(fexpr, Expr):
            return Group(fexpr)
        # fully constant folded -> allow further folding
        return fexpr

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return self._expr.eval(X, late_bound)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        return self._expr.compute_data_error_bound(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # group just passes on the arguments
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"({self._expr!r})"

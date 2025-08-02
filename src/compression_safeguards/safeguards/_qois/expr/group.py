from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from .abc import Expr
from .literal import Number
from .typing import F, Ns, Ps, PsI


class Group(Expr):
    __slots__ = ("_expr",)
    _expr: Expr

    def __new__(cls, expr: Expr):
        if isinstance(expr, (Number, Group)):
            return expr
        this = super(Group, cls).__new__(cls)
        this._expr = expr
        return this

    @property
    def has_data(self) -> bool:
        return self._expr.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._expr.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return Group(self._expr.apply_array_element_offset(axis, offset))

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
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return self._expr.eval(x, Xs, late_bound)

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        return self._expr.compute_data_error_bound(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # group just passes on the arguments
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"({self._expr!r})"

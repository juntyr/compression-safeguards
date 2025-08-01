from collections.abc import Mapping
from typing import Callable

import numpy as np

from .....utils.bindings import Parameter
from .abc import Expr
from .typing import F, Ns, Ps, PsI


class FoldedScalarConst(Expr):
    __slots__ = ("_const",)
    _const: np.number

    def __init__(self, const: np.number):
        self._const = const

    @property
    def has_data(self) -> bool:
        return False

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset()

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        assert isinstance(self._const, dtype.type)
        return self._const  # type: ignore

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        assert isinstance(self._const, Xs.dtype.type)
        return self._const  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        assert False, "folded constants have no error bounds"

    @staticmethod
    def constant_fold_unary(
        expr: Expr, dtype: np.dtype[F], m: Callable[[F], F], rm: Callable[[Expr], Expr]
    ) -> F | Expr:
        fexpr = expr.constant_fold(dtype)
        if isinstance(fexpr, Expr):
            return rm(fexpr)
        return m(fexpr)

    @staticmethod
    def constant_fold_binary(
        left: Expr,
        right: Expr,
        dtype: np.dtype[F],
        m: Callable[[F, F], F],
        rm: Callable[[Expr, Expr], Expr],
    ) -> F | Expr:
        fleft = left.constant_fold(dtype)
        fright = right.constant_fold(dtype)
        if isinstance(fleft, Expr):
            if isinstance(fright, Expr):
                return rm(fleft, fright)
            return rm(fleft, FoldedScalarConst(fright))
        if isinstance(fright, Expr):
            return rm(FoldedScalarConst(fleft), fright)
        return m(fleft, fright)

    @staticmethod
    def constant_fold_expr(expr: Expr, dtype: np.dtype[F]) -> "Expr":
        fexpr = expr.constant_fold(dtype)
        if isinstance(fexpr, Expr):
            return fexpr
        return FoldedScalarConst(fexpr)

    def __repr__(self) -> str:
        return f"({self._const!r})"

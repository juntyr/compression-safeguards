from collections.abc import Mapping
from typing import Callable

import numpy as np

from ....utils.bindings import Parameter
from .abc import Expr
from .typing import F, Ns, Ps, PsI


class ScalarFoldedConstant(Expr):
    __slots__ = ("_const",)
    _const: np.number

    def __init__(self, const: np.number):
        self._const = const[()] if isinstance(const, np.ndarray) else const  # type: ignore

    @property
    def has_data(self) -> bool:
        return False

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return frozenset()

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return self

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return frozenset()

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        assert isinstance(self._const, dtype.type)
        const: F = self._const  # type: ignore
        return const

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        assert isinstance(self._const, Xs.dtype.type)
        const: F = self._const  # type: ignore
        return np.broadcast_to(const, x)  # type: ignore

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        assert False, "folded constants have no data bounds"

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

        if not (isinstance(fleft, Expr) or isinstance(fright, Expr)):
            return m(fleft, fright)

        fleft = fleft if isinstance(fleft, Expr) else ScalarFoldedConstant(fleft)
        fright = fright if isinstance(fright, Expr) else ScalarFoldedConstant(fright)

        return rm(fleft, fright)

    # FIXME: more general constant_fold_ternary is blocked on not being able to
    #        relate on TypeVarTuple to another, here *Expr to *F, see e.g.
    #        https://github.com/python/typing/issues/1216
    @staticmethod
    def constant_fold_ternary(
        left: Expr,
        middle: Expr,
        right: Expr,
        dtype: np.dtype[F],
        m: Callable[[F, F, F], F],
        rm: Callable[[Expr, Expr, Expr], Expr],
    ) -> F | Expr:
        fleft = left.constant_fold(dtype)
        fmiddle = middle.constant_fold(dtype)
        fright = right.constant_fold(dtype)

        if not (
            isinstance(fleft, Expr)
            or isinstance(fmiddle, Expr)
            or isinstance(fright, Expr)
        ):
            return m(fleft, fmiddle, fright)

        fleft = fleft if isinstance(fleft, Expr) else ScalarFoldedConstant(fleft)
        fmiddle = (
            fmiddle if isinstance(fmiddle, Expr) else ScalarFoldedConstant(fmiddle)
        )
        fright = fright if isinstance(fright, Expr) else ScalarFoldedConstant(fright)

        return rm(fleft, fmiddle, fright)

    @staticmethod
    def constant_fold_expr(expr: Expr, dtype: np.dtype[F]) -> "Expr":
        fexpr = expr.constant_fold(dtype)
        if isinstance(fexpr, Expr):
            return fexpr
        return ScalarFoldedConstant(fexpr)

    def __repr__(self) -> str:
        return f"({self._const!r})"

from collections.abc import Callable, Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _broadcast_to
from ....utils.bindings import Parameter
from .abc import AnyExpr, EmptyExpr, Expr
from .typing import F, Ns, Ps, np_sndarray


class ScalarFoldedConstant(EmptyExpr):
    __slots__: tuple[str, ...] = ("_const",)
    _const: np.number

    def __init__(self, const: np.number) -> None:
        self._const = const[()] if isinstance(const, np.ndarray) else const  # type: ignore

    @property
    @override
    def args(self) -> tuple[()]:
        return ()

    @override
    def with_args(self) -> "ScalarFoldedConstant":
        return ScalarFoldedConstant(self._const)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        assert isinstance(self._const, dtype.type)
        return self._const

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        assert isinstance(self._const, Xs.dtype.type)
        return _broadcast_to(self._const, Xs.shape[:1])

    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        assert False, "folded constants have no data bounds"

    @staticmethod
    def constant_fold_unary(
        expr: AnyExpr,
        dtype: np.dtype[F],
        m: Callable[[F], F],
        rm: Callable[[AnyExpr], AnyExpr],
    ) -> F | AnyExpr:
        fexpr: F | AnyExpr = expr.constant_fold(dtype)
        if isinstance(fexpr, Expr):
            return rm(fexpr)
        return m(fexpr)

    @staticmethod
    def constant_fold_binary(
        left: AnyExpr,
        right: AnyExpr,
        dtype: np.dtype[F],
        m: Callable[[F, F], F],
        rm: Callable[[AnyExpr, AnyExpr], AnyExpr],
    ) -> F | AnyExpr:
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
        left: AnyExpr,
        middle: AnyExpr,
        right: AnyExpr,
        dtype: np.dtype[F],
        m: Callable[[F, F, F], F],
        rm: Callable[[AnyExpr, AnyExpr, AnyExpr], AnyExpr],
    ) -> F | AnyExpr:
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
    def constant_fold_expr(expr: AnyExpr, dtype: np.dtype[F]) -> AnyExpr:
        fexpr = expr.constant_fold(dtype)
        if isinstance(fexpr, Expr):
            return fexpr
        return ScalarFoldedConstant(fexpr)

    @override
    def __repr__(self) -> str:
        return f"({self._const!r})"

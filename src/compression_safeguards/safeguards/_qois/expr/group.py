from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds
from ..context import Callback, Context
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .literal import Number


class Group(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_expr", "_cache")
    _expr: AnyExpr
    _cache: None | tuple[int, int, np.ndarray]

    def __new__(cls, expr: AnyExpr) -> "Group | Number":  # type: ignore[misc]
        if isinstance(expr, Number | Group):
            return expr
        # we have to also assign inside __new__ since the double Group
        #  unwrapping would still call __init__ and create a reference loop
        this = super().__new__(cls)
        this._expr = expr
        this._cache = None
        return this

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._expr,)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

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
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        # if self._cache is not None:
        #     Xs_id, late_bound_id, cached = self._cache
        #     if (id(Xs) == Xs_id) and (id(late_bound) == late_bound_id):
        #         return _ensure_array(cached, copy=True)

        result = self._expr.eval(Xs, late_bound)

        # self._cache = id(Xs), id(late_bound), _ensure_array(result, copy=True)

        return result

    @data_bounds(DataBounds.infallible)
    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context[Ps, Ns, F],
        callback: Callback[Ps, Ns, F],
    ) -> None:
        return self._expr.deferred_compute_data_bounds(
            expr_lower, expr_upper, Xs, late_bound, ctx, callback
        )

    @override
    def __repr__(self) -> str:
        return f"({self._expr!r})"

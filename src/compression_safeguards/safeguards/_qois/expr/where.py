from collections.abc import Mapping
from functools import partial

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _broadcast_to, _ensure_array, _where
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..context import Callback, Context, DataBoundsAccumulator
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarWhere(Expr[AnyExpr, AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_condition", "_a", "_b")
    _condition: AnyExpr
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, condition: AnyExpr, a: AnyExpr, b: AnyExpr):
        self._condition = condition
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr, AnyExpr]:
        return (self._condition, self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, condition: AnyExpr, a: AnyExpr, b: AnyExpr) -> "ScalarWhere":
        return ScalarWhere(condition, a, b)

    @override  # type: ignore
    def eval_has_data(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[np.bool]]:
        has_data = self._condition.eval_has_data(Xs, late_bound)
        has_data |= _where(
            self._condition.eval(Xs, late_bound) != 0,
            self._a.eval_has_data(Xs, late_bound),
            self._b.eval_has_data(Xs, late_bound),
        )
        return has_data

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        cond = self._condition.constant_fold(dtype)
        a = self._a.constant_fold(dtype)
        b = self._b.constant_fold(dtype)

        if not isinstance(cond, Expr):
            if cond != 0 and not isinstance(a, Expr):
                return a

            if cond == 0 and not isinstance(b, Expr):
                return b

        return ScalarWhere(
            ScalarFoldedConstant.from_folded(cond),
            ScalarFoldedConstant.from_folded(a),
            ScalarFoldedConstant.from_folded(b),
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _where(
            self._condition.eval(Xs, late_bound) != 0,
            self._a.eval(Xs, late_bound),
            self._b.eval(Xs, late_bound),
        )

    @checked_data_bounds
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
        # evaluate the condition, a, and b
        cond, a, b = self._condition, self._a, self._b
        condv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            cond.eval(Xs, late_bound)
        )
        condvb_Ps: np.ndarray[tuple[Ps], np.dtype[np.bool]] = condv != 0
        condvb_Ns: np_sndarray[Ps, Ns, np.dtype[np.bool]] = _broadcast_to(
            _ensure_array(condvb_Ps).reshape(Xs.shape[:1] + (1,) * (Xs.ndim - 1)),
            Xs.shape,
        )

        # FIXME: could this be done in a less hacky way?
        def prune_pre_visit(e: AnyExpr) -> None:
            ctx._context[e]._dependents -= 1
            if ctx._context[e]._dependents > 0:
                return
            for a in e.args:
                prune_pre_visit(a)

        if not (np.any(condvb_Ps) and a.has_data):
            prune_pre_visit(a)
        if not ((not np.all(condvb_Ps)) and b.has_data):
            prune_pre_visit(b)

        wrapped_callback: DataBoundsAccumulator[Ps, Ns, F] = DataBoundsAccumulator(
            Xs=Xs, terms=3, callback=callback
        )

        if cond.has_data:
            # for simplicity, we assume that the condition must always evaluate
            #  to the same boolean when compared to 0
            cond_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                condv.shape, Xs.dtype.type(-np.inf)
            )
            cond_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                condv.shape, Xs.dtype.type(np.inf)
            )

            # zero condition values must remain zero
            cond_lower[condv == 0] = -0.0
            cond_upper[condv == 0] = +0.0

            smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

            # non-zero condition values must remain non-zero
            # TODO: an interval union could represent the two disjoint
            #       intervals in the future
            cond_lower[condv > 0] = smallest_subnormal
            cond_upper[condv < 0] = -smallest_subnormal

            cond.deferred_compute_data_bounds(
                cond_lower,
                cond_upper,
                Xs,
                late_bound,
                ctx,
                partial(wrapped_callback.on_complete_term, term=0),
            )
        else:
            wrapped_callback.complete_term(0)

        if np.any(condvb_Ps) and a.has_data:
            # pass on the data bounds to a but only use its bounds on Xs if
            #  chosen by the condition
            a_lower = _ensure_array(expr_lower, copy=True)
            a_lower[~condvb_Ps] = Xs.dtype.type(-np.inf)

            a_upper = _ensure_array(expr_upper, copy=True)
            a_upper[~condvb_Ps] = Xs.dtype.type(np.inf)

            a.deferred_compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
                ctx,
                partial(wrapped_callback.on_complete_term, term=1, where=condvb_Ns),
            )
        else:
            wrapped_callback.complete_term(1)

        if (not np.all(condvb_Ps)) and b.has_data:
            # pass on the data bounds to b but only use its bounds on Xs if
            #  chosen by the condition
            b_lower = _ensure_array(expr_lower, copy=True)
            b_lower[condvb_Ps] = Xs.dtype.type(-np.inf)

            b_upper = _ensure_array(expr_upper, copy=True)
            b_upper[condvb_Ps] = Xs.dtype.type(np.inf)

            b.deferred_compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
                ctx,
                partial(wrapped_callback.on_complete_term, term=2, where=~condvb_Ns),
            )
        else:
            wrapped_callback.complete_term(2)

    @override
    def __repr__(self) -> str:
        return f"where({self._condition!r}, {self._a!r}, {self._b!r})"

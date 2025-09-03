from collections.abc import Mapping

import numpy as np

from ....utils._compat import _broadcast_to, _maximum, _minimum, _where
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Fi, Ns, Ps, PsI


class ScalarWhere(Expr[Expr, Expr, Expr]):
    __slots__ = ("_condition", "_a", "_b")
    _condition: Expr
    _a: Expr
    _b: Expr

    def __init__(self, condition: Expr, a: Expr, b: Expr):
        self._condition = condition
        self._a = a
        self._b = b

    @property
    def args(self) -> tuple[Expr, Expr, Expr]:
        return (self._condition, self._a, self._b)

    def with_args(self, condition: Expr, a: Expr, b: Expr) -> "ScalarWhere":
        return ScalarWhere(condition, a, b)

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_ternary(
            self._condition,
            self._a,
            self._b,
            dtype,
            lambda cond, a, b: _where(cond != 0, a, b),
            ScalarWhere,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _where(
            self._condition.eval(x, Xs, late_bound) != 0,
            self._a.eval(x, Xs, late_bound),
            self._b.eval(x, Xs, late_bound),
        )

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate the condition, a, and b
        cond, a, b = self._condition, self._a, self._b
        condv: np.ndarray[Ps, np.dtype[F]] = cond.eval(X.shape, Xs, late_bound)
        condvb_Ps: np.ndarray[Ps, np.dtype[np.bool]] = condv != 0
        condvb_Ns: np.ndarray[Ns, np.dtype[np.bool]] = _broadcast_to(
            np.array(condvb_Ps, copy=None).reshape(X.shape + (1,) * (Xs.ndim - X.ndim)),
            Xs.shape,
        )
        av = a.eval(X.shape, Xs, late_bound)
        bv = b.eval(X.shape, Xs, late_bound)

        Xs_lower: np.ndarray[Ns, np.dtype[F]] = np.full(Xs.shape, X.dtype.type(-np.inf))
        Xs_upper: np.ndarray[Ns, np.dtype[F]] = np.full(Xs.shape, X.dtype.type(np.inf))

        if cond.has_data:
            # for simplicity, we assume that the condition must always be
            #  preserved exactly
            cl, cu = cond.compute_data_bounds(condv, condv, X, Xs, late_bound)
            Xs_lower = _maximum(Xs_lower, cl)
            Xs_upper = _minimum(Xs_upper, cu)

        if np.any(condvb_Ps) and a.has_data:
            # pass on the data bounds to a but only use its bounds on Xs if
            #  chosen by the condition
            al, au = a.compute_data_bounds(
                _where(condvb_Ps, expr_lower, av),
                _where(condvb_Ps, expr_upper, av),
                X,
                Xs,
                late_bound,
            )

            # combine the data bounds
            np.copyto(Xs_lower, _maximum(Xs_lower, al), where=condvb_Ns, casting="no")
            np.copyto(Xs_upper, _minimum(Xs_upper, au), where=condvb_Ns, casting="no")

        if (not np.all(condvb_Ps)) and b.has_data:
            # pass on the data bounds to b but only use its bounds on Xs if
            #  chosen by the condition
            bl, bu = b.compute_data_bounds(
                _where(condvb_Ps, bv, expr_lower),
                _where(condvb_Ps, bv, expr_upper),
                X,
                Xs,
                late_bound,
            )

            # combine the data bounds
            np.copyto(Xs_lower, _maximum(Xs_lower, bl), where=~condvb_Ns, casting="no")
            np.copyto(Xs_upper, _minimum(Xs_upper, bu), where=~condvb_Ns, casting="no")

        return Xs_lower, Xs_upper

    def __repr__(self) -> str:
        return f"where({self._condition!r}, {self._a!r}, {self._b!r})"

from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _broadcast_to,
    _ensure_array,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
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
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
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
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            cond_lower[condv > 0] = smallest_subnormal
            cond_upper[condv < 0] = -smallest_subnormal

            cl, cu = cond.compute_data_bounds(cond_lower, cond_upper, Xs, late_bound)
            Xs_lower = _maximum_zero_sign_sensitive(Xs_lower, cl)
            Xs_upper = _minimum_zero_sign_sensitive(Xs_upper, cu)

        if np.any(condvb_Ps) and a.has_data:
            # pass on the data bounds to a but only use its bounds on Xs if
            #  chosen by the condition
            al, au = a.compute_data_bounds(
                _where(condvb_Ps, expr_lower, av),
                _where(condvb_Ps, expr_upper, av),
                Xs,
                late_bound,
            )

            # combine the data bounds
            np.copyto(
                Xs_lower,
                _maximum_zero_sign_sensitive(Xs_lower, al),
                where=condvb_Ns,
                casting="no",
            )
            np.copyto(
                Xs_upper,
                _minimum_zero_sign_sensitive(Xs_upper, au),
                where=condvb_Ns,
                casting="no",
            )

        if (not np.all(condvb_Ps)) and b.has_data:
            # pass on the data bounds to b but only use its bounds on Xs if
            #  chosen by the condition
            bl, bu = b.compute_data_bounds(
                _where(condvb_Ps, bv, expr_lower),
                _where(condvb_Ps, bv, expr_upper),
                Xs,
                late_bound,
            )

            # combine the data bounds
            np.copyto(
                Xs_lower,
                _maximum_zero_sign_sensitive(Xs_lower, bl),
                where=~condvb_Ns,
                casting="no",
            )
            np.copyto(
                Xs_upper,
                _minimum_zero_sign_sensitive(Xs_upper, bu),
                where=~condvb_Ns,
                casting="no",
            )

        return Xs_lower, Xs_upper

    @override
    def __repr__(self) -> str:
        return f"where({self._condition!r}, {self._a!r}, {self._b!r})"

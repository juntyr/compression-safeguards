from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _is_negative_zero,
    _is_positive_zero,
    _is_sign_negative_number,
    _is_sign_positive_number,
    _nextafter,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..context import Callback, Context
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarNextafter(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_from", "_to")
    _from: AnyExpr
    _to: AnyExpr

    def __init__(self, from_: AnyExpr, to: AnyExpr) -> None:
        self._from = from_
        self._to = to

        if self._to.has_data:
            raise NotImplementedError(
                "`nextafter(from, to)` with non-constant direction `to`"
            )

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._from, self._to)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, from_: AnyExpr, to: AnyExpr) -> "ScalarNextafter":
        return ScalarNextafter(from_, to)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._from, self._to, dtype, _nextafter, ScalarNextafter
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _nextafter(
            self._from.eval(Xs, late_bound), self._to.eval(Xs, late_bound)
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
        from_const = not self._from.has_data
        to_const = not self._to.has_data
        assert to_const, (
            "cannot compute the data bounds for nextafter(from, to) with non-constant to"
        )
        assert not (from_const and to_const), "constant nextafter has no data bounds"

        # evaluate to
        to = self._to
        tov = to.eval(Xs, late_bound)

        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # nextafter(fromv, NaN) = NaN for any fromv
        # nextafter(NaN, tov) = NaN for any tov
        # otherwise
        # - if a bound is equal to tov, nudge the bound in both direction,
        #   since both will be back to tov with nextafter
        # - if nextafter would nudge a bound down, nudge the bound down
        # - if nextafter would nudge a bound up, nudge the bound up
        # we need to be very very careful around -0.0 and +0.0
        #  nextafter(+0.0, -Inf) = -smallest_subnormal
        #  nextafter(-0.0, +Inf) = +smallest_subnormal
        # and have to move a bound across zero if
        # - expr_lower = +0.0 and tov is sign negative
        #   -> from_lower = +smallest_subnormal
        # - expr_upper = -0.0 and tov is sign positive
        #   -> from_upper = -smallest_subnormal
        # and we need to be careful to include -0.0 and +0.0 in edge cases
        # - expr_lower = smallest_subnormal and expr_lower <= tov
        #   -> from_lower = -0.0
        # - expr_upper = -smallest_subnormal and expr_upper >= tov
        #   -> from_upper = +0.0
        from_lower = _ensure_array(expr_lower, copy=True)
        _nextafter(
            expr_lower,
            Xs.dtype.type(-np.inf),
            out=from_lower,
            where=(expr_lower <= tov),
        )
        from_lower[_is_positive_zero(expr_lower) & (expr_lower <= tov)] = Xs.dtype.type(
            -0.0
        )
        from_lower[_is_positive_zero(expr_lower) & _is_sign_negative_number(tov)] = (
            smallest_subnormal
        )
        from_lower[(expr_lower == smallest_subnormal) & (expr_lower <= tov)] = (
            Xs.dtype.type(-0.0)
        )
        _nextafter(
            expr_lower, Xs.dtype.type(np.inf), out=from_lower, where=(expr_lower > tov)
        )
        from_lower[_is_negative_zero(expr_lower) & (expr_lower > tov)] = (
            smallest_subnormal
        )

        from_upper = _ensure_array(expr_upper, copy=True)
        _nextafter(
            expr_upper, Xs.dtype.type(-np.inf), out=from_upper, where=(expr_upper < tov)
        )
        from_upper[
            _is_positive_zero(expr_upper) & (expr_upper < tov)
        ] = -smallest_subnormal
        _nextafter(
            expr_upper, Xs.dtype.type(np.inf), out=from_upper, where=(expr_upper >= tov)
        )
        from_upper[_is_negative_zero(expr_upper) & (expr_upper >= tov)] = Xs.dtype.type(
            0.0
        )
        from_upper[
            _is_negative_zero(expr_upper) & _is_sign_positive_number(tov)
        ] = -smallest_subnormal
        from_upper[(expr_upper == -smallest_subnormal) & (expr_upper >= tov)] = (
            Xs.dtype.type(0.0)
        )

        return self._from.deferred_compute_data_bounds(
            from_lower, from_upper, Xs, late_bound, ctx, callback
        )

    @override
    def __repr__(self) -> str:
        return f"nextafter({self._from!r}, {self._to!r})"

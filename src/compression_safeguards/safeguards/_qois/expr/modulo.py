from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ceil_modulo,
    _ensure_array,
    _euclidean_modulo,
    _floor_modulo,
    _is_sign_negative_number,
    _is_sign_positive_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _round_ties_even_modulo,
    _trunc_modulo,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from ..context import Callback, Context
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarFloorModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._p, self._q)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, p: AnyExpr, q: AnyExpr) -> "ScalarFloorModulo":
        return ScalarFloorModulo(p, q)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._p,
            self._q,
            dtype,
            _floor_modulo,
            ScalarFloorModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _floor_modulo(self._p.eval(Xs, late_bound), self._q.eval(Xs, late_bound))

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
        p_const = not self._p.has_data
        q_const = not self._q.has_data
        assert q_const, (
            "cannot compute the data bounds for floor_modulo(p, q) with non-constant q"
        )
        assert not (p_const and q_const), "constant floor_modulo has no data bounds"

        # evaluate p, q and floor_modulo(p, q)
        p, q = self._p, self._q
        pv = p.eval(Xs, late_bound)
        qv = q.eval(Xs, late_bound)
        exprv = _floor_modulo(pv, qv)

        fl: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(qv, copy=True)
        fl[qv > 0] = Xs.dtype.type(+0.0)
        fu: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(qv, copy=True)
        fu[qv < 0] = Xs.dtype.type(-0.0)

        # ensure that the bounds on floor_modulo(...) are in
        #  - (q, -0.0] if q < 0
        #  - [+0.0, q) if q > 0
        efl: np.ndarray[tuple[Ps], np.dtype[F]] = _maximum_zero_sign_sensitive(
            fl, expr_lower
        )
        efu: np.ndarray[tuple[Ps], np.dtype[F]] = _minimum_zero_sign_sensitive(
            expr_upper, fu
        )

        # floor_modulo(...) is periodic, so we need to drop to difference
        #  bounds before applying the difference to argv to stay in the
        #  same period
        p_lower_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(efl, exprv)
        p_upper_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(efu, exprv)

        # check for the case where any finite value would work
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, fl
        ) & np.greater_equal(expr_upper, fu)

        fmax = np.finfo(Xs.dtype).max

        # if qv is NaN, anything is allowed for pv
        # if pv is NaN, it should stay NaN
        # if qv is 0, anything is allowed for pv
        # if pv is inf, it must stay inf
        # if qv is inf and the signbits of pv and qv match, use expr bounds within the same signbit
        # if qv is inf anf the signbits of pv and qv don't match, anything is allowed within as long as the mismatch stays
        # if the full domain is ok, allow the full finite domain
        # otherwise, apply the bounds to the current repetition
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper

        p_lower = _ensure_array(p_lower_diff, copy=True)
        np.add(p_lower, pv, out=p_lower)
        p_lower[full_domain] = -fmax
        p_lower[np.isposinf(qv) & _is_sign_negative_number(pv)] = -fmax
        p_lower[np.isneginf(qv) & _is_sign_positive_number(pv)] = Xs.dtype.type(+0.0)
        _maximum_zero_sign_sensitive(
            p_lower,
            Xs.dtype.type(+0.0),
            out=p_lower,
            where=(np.isposinf(qv) & _is_sign_positive_number(pv)),
        )
        np.copyto(p_lower, pv, where=np.isinf(pv))
        p_lower[qv == 0] = Xs.dtype.type(-np.inf)
        np.copyto(p_lower, pv, where=np.isnan(pv))
        p_lower[np.isnan(qv)] = Xs.dtype.type(-np.inf)
        _minimum_zero_sign_sensitive(pv, p_lower, out=p_lower)

        p_upper = _ensure_array(p_upper_diff, copy=True)
        np.add(p_upper, pv, out=p_upper)
        p_upper[full_domain] = fmax
        p_upper[np.isposinf(qv) & _is_sign_negative_number(pv)] = Xs.dtype.type(-0.0)
        p_upper[np.isneginf(qv) & _is_sign_positive_number(pv)] = fmax
        _minimum_zero_sign_sensitive(
            p_upper,
            Xs.dtype.type(-0.0),
            out=p_upper,
            where=(np.isneginf(qv) & _is_sign_negative_number(pv)),
        )
        np.copyto(p_upper, pv, where=np.isinf(pv))
        p_upper[qv == 0] = Xs.dtype.type(np.inf)
        np.copyto(p_upper, pv, where=np.isnan(pv))
        p_upper[np.isnan(qv)] = Xs.dtype.type(np.inf)
        _maximum_zero_sign_sensitive(pv, p_upper, out=p_upper)

        # we need to force pv if expr_lower == expr_upper
        np.copyto(p_lower, pv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(p_upper, pv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in floor_modulo early
        p_lower = guarantee_arg_within_expr_bounds(
            lambda p_lower: _floor_modulo(p_lower, qv),
            exprv,
            pv,
            p_lower,
            expr_lower,
            expr_upper,
        )
        p_upper = guarantee_arg_within_expr_bounds(
            lambda p_upper: _floor_modulo(p_upper, qv),
            exprv,
            pv,
            p_upper,
            expr_lower,
            expr_upper,
        )

        return p.deferred_compute_data_bounds(
            p_lower,
            p_upper,
            Xs,
            late_bound,
            ctx,
            callback,
        )

    @override
    def __repr__(self) -> str:
        return f"floor_modulo({self._p!r}, {self._q!r})"


class ScalarCeilModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._p, self._q)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, p: AnyExpr, q: AnyExpr) -> "ScalarCeilModulo":
        return ScalarCeilModulo(p, q)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._p,
            self._q,
            dtype,
            _ceil_modulo,
            ScalarCeilModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _ceil_modulo(self._p.eval(Xs, late_bound), self._q.eval(Xs, late_bound))

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
        assert False, "cannot compute the data bounds for ceil_modulo"

    @override
    def __repr__(self) -> str:
        return f"ceil_modulo({self._p!r}, {self._q!r})"


class ScalarTruncModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._p, self._q)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, p: AnyExpr, q: AnyExpr) -> "ScalarTruncModulo":
        return ScalarTruncModulo(p, q)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._p,
            self._q,
            dtype,
            _trunc_modulo,
            ScalarTruncModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _trunc_modulo(self._p.eval(Xs, late_bound), self._q.eval(Xs, late_bound))

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
        assert False, "cannot compute the data bounds for trunc_modulo"

    @override
    def __repr__(self) -> str:
        return f"trunc_modulo({self._p!r}, {self._q!r})"


class ScalarRoundTiesEvenModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._p, self._q)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, p: AnyExpr, q: AnyExpr) -> "ScalarRoundTiesEvenModulo":
        return ScalarRoundTiesEvenModulo(p, q)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._p,
            self._q,
            dtype,
            _round_ties_even_modulo,
            ScalarRoundTiesEvenModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _round_ties_even_modulo(
            self._p.eval(Xs, late_bound), self._q.eval(Xs, late_bound)
        )

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
        assert False, "cannot compute the data bounds for round_ties_even_modulo"

    @override
    def __repr__(self) -> str:
        return f"round_ties_even_modulo({self._p!r}, {self._q!r})"


class ScalarEuclideanModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._p, self._q)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, p: AnyExpr, q: AnyExpr) -> "ScalarEuclideanModulo":
        return ScalarEuclideanModulo(p, q)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._p,
            self._q,
            dtype,
            _euclidean_modulo,
            ScalarEuclideanModulo,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return _euclidean_modulo(
            self._p.eval(Xs, late_bound), self._q.eval(Xs, late_bound)
        )

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
        assert False, "cannot compute the data bounds for euclidean_modulo"

    @override
    def __repr__(self) -> str:
        return f"euclidean_modulo({self._p!r}, {self._q!r})"

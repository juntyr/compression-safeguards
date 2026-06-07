from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ceil_modulo,
    _ensure_array,
    _euclidean_modulo,
    _floor_modulo,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _round_ties_even_modulo,
    _trunc_modulo,
    _where,
    _zeros,
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

        if self._q.has_data:
            raise NotImplementedError(
                "`floor_modulo(p, q)` with non-constant divisor `q`"
            )

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

        # the bounds on floor_modulo(...) are
        #  - (q, -0.0] if q < 0
        #  - [+0.0, q) if q > 0
        rem_lower = _ensure_array(qv, copy=True)
        rem_lower[qv > 0] = Xs.dtype.type(+0.0)
        rem_upper = _ensure_array(qv, copy=True)
        rem_upper[qv < 0] = Xs.dtype.type(-0.0)

        rem_expr_lower = _maximum_zero_sign_sensitive(rem_lower, expr_lower)
        rem_expr_upper = _minimum_zero_sign_sensitive(expr_upper, rem_upper)

        # floor_modulo(...) is periodic, so we need to drop to difference
        #  bounds before applying the difference to pv to stay in the
        #  same period
        p_lower_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_lower, exprv
        )
        p_upper_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_upper, exprv
        )

        # check for the case where any finite value would work for pv
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, rem_lower
        ) & np.greater_equal(expr_upper, rem_upper)

        fmax = np.finfo(Xs.dtype).max
        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # floor_modulo(pv, NaN) = NaN for any pv
        # floor_modulo(NaN, qv) = NaN, keep pv NaN
        # floor_modulo(pv, 0) = NaN for any pv
        # floor_modulo(+-Inf, qv) = NaN, keep pv infinite
        # floor_modulo(pv, +-Inf) = ??
        #  - if the bounds include Inf, sign(pv) != sign(+-Inf) is allowed
        #    - if the bounds exclude zero, only allow sign(pv) != sign(+-Inf)
        #  - if the bounds only include zero, restrict pv to zero
        #  - if the bounds exclude Inf, only sign(pv) == sign(+-Inf) is allowed
        # if the full domain is ok, allow the full finite domain for pv
        # otherwise, apply the bounds to the current repetition
        # if p_lower == pv and pv == -0.0, we need to guarantee that
        #  p_lower is also -0.0, same for p_upper

        p_lower = _ensure_array(pv, copy=True)
        np.add(p_lower, p_lower_diff, out=p_lower)
        np.copyto(  # exact bounds around zero if signs match
            p_lower,
            rem_expr_lower,
            where=((np.abs(pv) < np.abs(qv)) & ((pv == 0) | ((pv > 0) == (qv > 0)))),
            casting="no",
        )
        _maximum_zero_sign_sensitive(  # we don't allow repetition slips
            p_lower, Xs.dtype.type(-0.0), out=p_lower, where=(pv >= 0)
        )
        exact_p_lower = _zeros(pv.shape, np.dtype(np.bool))
        p_lower[full_domain] = -fmax
        exact_p_lower |= full_domain
        p_lower[np.isposinf(qv) & np.isposinf(rem_expr_upper)] = -fmax
        exact_p_lower |= np.isposinf(qv) & np.isposinf(rem_expr_upper)
        p_lower[np.isposinf(qv) & (rem_expr_upper == 0)] = Xs.dtype.type(-0.0)
        exact_p_lower |= np.isposinf(qv) & (rem_expr_upper == 0)
        p_lower[np.isneginf(qv) & (rem_expr_lower == 0)] = Xs.dtype.type(-0.0)
        exact_p_lower |= np.isneginf(qv) & (rem_expr_lower == 0)
        p_lower[np.isneginf(qv) & (pv > 0) & (rem_expr_upper < 0)] = smallest_subnormal
        exact_p_lower |= np.isneginf(qv) & (pv > 0) & (rem_expr_upper < 0)
        _maximum_zero_sign_sensitive(
            p_lower,
            Xs.dtype.type(+0.0),
            out=p_lower,
            where=(np.isposinf(qv) & ~np.isposinf(rem_expr_upper)),
        )
        # if the zero boundary was within the approximate bounds, it will be
        # exact since zero is an exact boundary and the bounds would not be
        # inaccurate in magnitude beyond their size
        exact_p_lower |= np.isposinf(qv) & ~np.isposinf(rem_expr_upper) & (p_lower == 0)
        np.copyto(p_lower, pv, where=np.isinf(pv), casting="no")
        exact_p_lower |= np.isinf(pv)
        p_lower[qv == 0] = Xs.dtype.type(-np.inf)
        exact_p_lower |= qv == 0
        np.copyto(p_lower, pv, where=np.isnan(pv), casting="no")
        exact_p_lower |= np.isnan(pv)
        p_lower[np.isnan(qv)] = Xs.dtype.type(-np.inf)
        exact_p_lower |= np.isnan(qv)
        _minimum_zero_sign_sensitive(pv, p_lower, out=p_lower)

        p_upper = _ensure_array(pv, copy=True)
        np.add(p_upper, p_upper_diff, out=p_upper)
        np.copyto(  # exact bounds around zero if signs match
            p_upper,
            rem_expr_upper,
            where=((np.abs(pv) < np.abs(qv)) & ((pv == 0) | ((pv > 0) == (qv > 0)))),
            casting="no",
        )
        _minimum_zero_sign_sensitive(  # we don't allow repetition slips
            p_upper, Xs.dtype.type(+0.0), out=p_upper, where=(pv <= 0)
        )
        exact_p_upper = _zeros(pv.shape, np.dtype(np.bool))
        p_upper[full_domain] = fmax
        exact_p_upper |= full_domain
        p_upper[np.isposinf(qv) & (rem_expr_upper == 0)] = Xs.dtype.type(+0.0)
        exact_p_upper |= np.isposinf(qv) & (rem_expr_upper == 0)
        p_upper[np.isposinf(qv) & (pv < 0) & (rem_expr_lower > 0)] = -smallest_subnormal
        exact_p_upper |= np.isposinf(qv) & (pv < 0) & (rem_expr_lower > 0)
        p_upper[np.isneginf(qv) & np.isneginf(rem_expr_lower)] = fmax
        exact_p_upper |= np.isneginf(qv) & np.isneginf(rem_expr_lower)
        p_upper[np.isneginf(qv) & (rem_expr_lower == 0)] = Xs.dtype.type(+0.0)
        exact_p_upper |= np.isneginf(qv) & (rem_expr_lower == 0)
        _minimum_zero_sign_sensitive(
            p_upper,
            Xs.dtype.type(-0.0),
            out=p_upper,
            where=(np.isneginf(qv) & ~np.isneginf(rem_expr_lower)),
        )
        # if the zero boundary was within the approximate bounds, it will be
        # exact since zero is an exact boundary and the bounds would not be
        # inaccurate in magnitude beyond their size
        exact_p_upper |= np.isneginf(qv) & ~np.isneginf(rem_expr_lower) & (p_upper == 0)
        np.copyto(p_upper, pv, where=np.isinf(pv), casting="no")
        exact_p_upper |= np.isinf(pv)
        p_upper[qv == 0] = Xs.dtype.type(np.inf)
        exact_p_upper |= qv == 0
        np.copyto(p_upper, pv, where=np.isnan(pv), casting="no")
        exact_p_upper |= np.isnan(pv)
        p_upper[np.isnan(qv)] = Xs.dtype.type(np.inf)
        exact_p_upper |= np.isnan(qv)
        _maximum_zero_sign_sensitive(pv, p_upper, out=p_upper)

        # if p_lower/p_upper is not exact, it may have rounding errors,
        # so we guard against slipping into the next repetition
        p_lower_expr_upper = _where(
            exact_p_lower,
            expr_upper,
            exprv,
        )
        p_upper_expr_lower = _where(
            exact_p_upper,
            expr_lower,
            exprv,
        )

        # handle rounding errors in floor_modulo early
        p_lower = guarantee_arg_within_expr_bounds(
            lambda p_lower: _floor_modulo(p_lower, qv),
            exprv,
            pv,
            p_lower,
            expr_lower,
            p_lower_expr_upper,
        )
        p_upper = guarantee_arg_within_expr_bounds(
            lambda p_upper: _floor_modulo(p_upper, qv),
            exprv,
            pv,
            p_upper,
            p_upper_expr_lower,
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

        if self._q.has_data:
            raise NotImplementedError(
                "`ceil_modulo(p, q)` with non-constant divisor `q`"
            )

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
            "cannot compute the data bounds for ceil_modulo(p, q) with non-constant q"
        )
        assert not (p_const and q_const), "constant ceil_modulo has no data bounds"

        # evaluate p, q and ceil_modulo(p, q)
        p, q = self._p, self._q
        pv = p.eval(Xs, late_bound)
        qv = q.eval(Xs, late_bound)
        exprv = _ceil_modulo(pv, qv)

        # the bounds on ceil_modulo(...) are
        #  - [+0.0, -q) if q < 0
        #  - (-q, -0.0] if q > 0
        rem_lower = _ensure_array(-qv, copy=True)
        rem_lower[qv < 0] = Xs.dtype.type(+0.0)
        rem_upper = _ensure_array(-qv, copy=True)
        rem_upper[qv > 0] = Xs.dtype.type(-0.0)

        rem_expr_lower = _maximum_zero_sign_sensitive(rem_lower, expr_lower)
        rem_expr_upper = _minimum_zero_sign_sensitive(expr_upper, rem_upper)

        # ceil_modulo(...) is periodic, so we need to drop to difference
        #  bounds before applying the difference to pv to stay in the
        #  same period
        p_lower_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_lower, exprv
        )
        p_upper_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_upper, exprv
        )

        # check for the case where any finite value would work for pv
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, rem_lower
        ) & np.greater_equal(expr_upper, rem_upper)

        fmax = np.finfo(Xs.dtype).max
        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # ceil_modulo(pv, NaN) = NaN for any pv
        # ceil_modulo(NaN, qv) = NaN, keep pv NaN
        # ceil_modulo(pv, 0) = NaN for any pv
        # ceil_modulo(+-Inf, qv) = NaN, keep pv infinite
        # ceil_modulo(pv, +-Inf) = ??
        #  - if the bounds include Inf, sign(pv) == sign(+-Inf) is allowed
        #    - if the bounds exclude zero, only allow sign(pv) == sign(+-Inf)
        #  - if the bounds only include zero, restrict pv to zero
        #  - if the bounds exclude Inf, only sign(pv) != sign(+-Inf) is allowed
        # if the full domain is ok, allow the full finite domain for pv
        # otherwise, apply the bounds to the current repetition
        # if p_lower == pv and pv == -0.0, we need to guarantee that
        #  p_lower is also -0.0, same for p_upper

        p_lower = _ensure_array(pv, copy=True)
        np.add(p_lower, p_lower_diff, out=p_lower)
        np.copyto(  # exact bounds around zero if signs mismatch
            p_lower,
            rem_expr_lower,
            where=((np.abs(pv) < np.abs(qv)) & ((pv == 0) | ((pv > 0) != (qv > 0)))),
            casting="no",
        )
        _maximum_zero_sign_sensitive(  # we don't allow repetition slips
            p_lower, Xs.dtype.type(-0.0), out=p_lower, where=(pv >= 0)
        )
        exact_p_lower = _zeros(pv.shape, np.dtype(np.bool))
        p_lower[full_domain] = -fmax
        exact_p_lower |= full_domain
        p_lower[np.isneginf(qv) & np.isposinf(rem_expr_lower)] = -fmax
        exact_p_lower |= np.isneginf(qv) & np.isposinf(rem_expr_lower)
        p_lower[np.isneginf(qv) & (rem_expr_upper == 0)] = Xs.dtype.type(-0.0)
        exact_p_lower |= np.isneginf(qv) & (rem_expr_upper == 0)
        p_lower[np.isposinf(qv) & (rem_expr_lower == 0)] = Xs.dtype.type(-0.0)
        exact_p_lower |= np.isposinf(qv) & (rem_expr_lower == 0)
        p_lower[np.isposinf(qv) & (pv > 0) & (rem_expr_upper < 0)] = smallest_subnormal
        exact_p_lower |= np.isposinf(qv) & (pv > 0) & (rem_expr_upper < 0)
        _maximum_zero_sign_sensitive(
            p_lower,
            Xs.dtype.type(+0.0),
            out=p_lower,
            where=(np.isneginf(qv) & ~np.isposinf(rem_expr_upper)),
        )
        # if the zero boundary was within the approximate bounds, it will be
        # exact since zero is an exact boundary and the bounds would not be
        # inaccurate in magnitude beyond their size
        exact_p_lower |= np.isneginf(qv) & ~np.isposinf(rem_expr_upper) & (p_lower == 0)
        np.copyto(p_lower, pv, where=np.isinf(pv), casting="no")
        exact_p_lower |= np.isinf(pv)
        p_lower[qv == 0] = Xs.dtype.type(-np.inf)
        exact_p_lower |= qv == 0
        np.copyto(p_lower, pv, where=np.isnan(pv), casting="no")
        exact_p_lower |= np.isnan(pv)
        p_lower[np.isnan(qv)] = Xs.dtype.type(-np.inf)
        exact_p_lower |= np.isnan(qv)
        _minimum_zero_sign_sensitive(pv, p_lower, out=p_lower)

        p_upper = _ensure_array(pv, copy=True)
        np.add(p_upper, p_upper_diff, out=p_upper)
        np.copyto(  # exact bounds around zero if signs mismatch
            p_upper,
            rem_expr_upper,
            where=((np.abs(pv) < np.abs(qv)) & ((pv == 0) | ((pv > 0) != (qv > 0)))),
            casting="no",
        )
        _minimum_zero_sign_sensitive(  # we don't allow repetition slips
            p_upper, Xs.dtype.type(+0.0), out=p_upper, where=(pv <= 0)
        )
        exact_p_upper = _zeros(pv.shape, np.dtype(np.bool))
        p_upper[full_domain] = fmax
        exact_p_upper |= full_domain
        p_upper[np.isneginf(qv) & (rem_expr_upper == 0)] = Xs.dtype.type(+0.0)
        exact_p_upper |= np.isneginf(qv) & (rem_expr_upper == 0)
        p_upper[np.isneginf(qv) & (pv < 0) & (rem_expr_lower > 0)] = -smallest_subnormal
        exact_p_upper |= np.isneginf(qv) & (pv < 0) & (rem_expr_lower > 0)
        p_upper[np.isposinf(qv) & np.isneginf(rem_expr_lower)] = fmax
        exact_p_upper |= np.isposinf(qv) & np.isneginf(rem_expr_lower)
        p_upper[np.isposinf(qv) & (rem_expr_lower == 0)] = Xs.dtype.type(+0.0)
        exact_p_upper |= np.isposinf(qv) & (rem_expr_lower == 0)
        _minimum_zero_sign_sensitive(
            p_upper,
            Xs.dtype.type(-0.0),
            out=p_upper,
            where=(np.isposinf(qv) & ~np.isneginf(rem_expr_lower)),
        )
        # if the zero boundary was within the approximate bounds, it will be
        # exact since zero is an exact boundary and the bounds would not be
        # inaccurate in magnitude beyond their size
        exact_p_upper |= np.isposinf(qv) & ~np.isneginf(rem_expr_lower) & (p_upper == 0)
        np.copyto(p_upper, pv, where=np.isinf(pv), casting="no")
        exact_p_upper |= np.isinf(pv)
        p_upper[qv == 0] = Xs.dtype.type(np.inf)
        exact_p_upper |= qv == 0
        np.copyto(p_upper, pv, where=np.isnan(pv), casting="no")
        exact_p_upper |= np.isnan(pv)
        p_upper[np.isnan(qv)] = Xs.dtype.type(np.inf)
        exact_p_upper |= np.isnan(qv)
        _maximum_zero_sign_sensitive(pv, p_upper, out=p_upper)

        # if p_lower/p_upper is not exact, it may have rounding errors,
        # so we guard against slipping into the next repetition
        p_lower_expr_upper = _where(
            exact_p_lower,
            expr_upper,
            exprv,
        )
        p_upper_expr_lower = _where(
            exact_p_upper,
            expr_lower,
            exprv,
        )

        # handle rounding errors in ceil_modulo early
        p_lower = guarantee_arg_within_expr_bounds(
            lambda p_lower: _ceil_modulo(p_lower, qv),
            exprv,
            pv,
            p_lower,
            expr_lower,
            p_lower_expr_upper,
        )
        p_upper = guarantee_arg_within_expr_bounds(
            lambda p_upper: _ceil_modulo(p_upper, qv),
            exprv,
            pv,
            p_upper,
            p_upper_expr_lower,
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
        return f"ceil_modulo({self._p!r}, {self._q!r})"


class ScalarTruncModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

        if self._q.has_data:
            raise NotImplementedError(
                "`trunc_modulo(p, q)` with non-constant divisor `q`"
            )

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
            "cannot compute the data bounds for trunc_modulo(p, q) with non-constant q"
        )
        assert not (p_const and q_const), "constant trunc_modulo has no data bounds"

        # evaluate p, q and trunc_modulo(p, q)
        p, q = self._p, self._q
        pv = p.eval(Xs, late_bound)
        qv = q.eval(Xs, late_bound)
        exprv = _trunc_modulo(pv, qv)

        qv_neg: np.ndarray[tuple[Ps], np.dtype[F]] = np.copysign(qv, -1)
        qv_pos: np.ndarray[tuple[Ps], np.dtype[F]] = np.copysign(qv, +1)

        # the bounds on trunc_modulo(...) are
        #  - (-|q|, +|q|) if -|q| < pv < +|q|
        #  - (-|q|, -0.0] if pv < -|q|
        #  - [+0.0, +|q|) if pv > +|q|
        rem_lower = _ensure_array(qv_neg, copy=True)
        rem_lower[pv >= qv_pos] = Xs.dtype.type(+0.0)
        rem_upper = _ensure_array(qv_pos, copy=True)
        rem_upper[pv <= qv_neg] = Xs.dtype.type(-0.0)

        rem_expr_lower = _maximum_zero_sign_sensitive(rem_lower, expr_lower)
        rem_expr_upper = _minimum_zero_sign_sensitive(expr_upper, rem_upper)

        # trunc_modulo(...) is periodic, so we need to drop to difference bounds
        #  before applying the difference to pv to stay in the same period
        p_lower_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_lower, exprv
        )
        p_upper_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_upper, exprv
        )

        # check for the case where any finite value would work for pv
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, qv_neg
        ) & np.greater_equal(expr_upper, qv_pos)

        fmax = np.finfo(Xs.dtype).max

        # trunc_modulo(pv, NaN) = NaN for any pv
        # trunc_modulo(NaN, qv) = NaN, keep pv NaN
        # trunc_modulo(pv, 0) = NaN for any pv
        # trunc_modulo(+-Inf, qv) = NaN, keep pv infinite
        # trunc_modulo(pv, +-Inf) = pv, so use normal pv bounds
        # if the full domain is ok, allow the full finite domain for pv
        # if the positive/negative domain is ok, allow the finite domain with
        #  the matching sign for pv
        # otherwise, apply the bounds to the current repetition
        # propagate -0.0 and +0.0 bounds on pv to avoid nudging
        # if p_lower == pv and pv == -0.0, we need to guarantee that
        #  p_lower is also -0.0, same for p_upper

        p_lower = _ensure_array(pv, copy=True)
        np.add(p_lower, p_lower_diff, out=p_lower)
        np.copyto(  # exact bounds around zero
            p_lower,
            rem_expr_lower,
            where=(np.abs(pv) < np.abs(qv)),
            casting="no",
        )
        _maximum_zero_sign_sensitive(  # we don't allow repetition slips
            p_lower, Xs.dtype.type(+0.0), out=p_lower, where=(pv >= qv_pos)
        )
        exact_p_lower = _zeros(pv.shape, np.dtype(np.bool))
        p_lower[full_domain] = -fmax
        exact_p_lower |= full_domain
        p_lower[(expr_lower <= qv_neg) & (expr_upper >= 0) & (pv < qv_pos)] = -fmax
        exact_p_lower |= (expr_lower <= qv_neg) & (expr_upper >= 0) & (pv < qv_pos)
        np.copyto(p_lower, pv, where=np.isinf(pv), casting="no")
        exact_p_lower |= np.isinf(pv)
        p_lower[qv == 0] = Xs.dtype.type(-np.inf)
        exact_p_lower |= qv == 0
        np.copyto(p_lower, pv, where=np.isnan(pv), casting="no")
        exact_p_lower |= np.isnan(pv)
        p_lower[np.isnan(qv)] = Xs.dtype.type(-np.inf)
        exact_p_lower |= np.isnan(qv)
        _minimum_zero_sign_sensitive(pv, p_lower, out=p_lower)

        p_upper = _ensure_array(pv, copy=True)
        np.add(p_upper, p_upper_diff, out=p_upper)
        np.copyto(  # exact bounds around zero
            p_upper,
            rem_expr_upper,
            where=(np.abs(pv) < np.abs(qv)),
            casting="no",
        )
        _minimum_zero_sign_sensitive(  # we don't allow repetition slips
            p_upper, Xs.dtype.type(-0.0), out=p_upper, where=(pv <= qv_neg)
        )
        exact_p_upper = _zeros(pv.shape, np.dtype(np.bool))
        p_upper[full_domain] = fmax
        exact_p_upper |= full_domain
        p_upper[(expr_lower <= 0) & (expr_upper >= qv_pos) & (pv > qv_neg)] = fmax
        exact_p_upper |= (expr_lower <= 0) & (expr_upper >= qv_pos) & (pv > qv_neg)
        np.copyto(p_upper, pv, where=np.isinf(pv), casting="no")
        exact_p_upper |= np.isinf(pv)
        p_upper[qv == 0] = Xs.dtype.type(np.inf)
        exact_p_upper |= qv == 0
        np.copyto(p_upper, pv, where=np.isnan(pv), casting="no")
        exact_p_upper |= np.isnan(pv)
        p_upper[np.isnan(qv)] = Xs.dtype.type(np.inf)
        exact_p_upper |= np.isnan(qv)
        _maximum_zero_sign_sensitive(pv, p_upper, out=p_upper)

        # if p_lower/p_upper is not exact, it may have rounding errors,
        # so we guard against slipping into the next repetition
        p_lower_expr_upper = _where(
            exact_p_lower,
            expr_upper,
            exprv,
        )
        p_upper_expr_lower = _where(
            exact_p_upper,
            expr_lower,
            exprv,
        )

        # handle rounding errors in trunc_modulo early
        p_lower = guarantee_arg_within_expr_bounds(
            lambda p_lower: _trunc_modulo(p_lower, qv),
            exprv,
            pv,
            p_lower,
            expr_lower,
            p_lower_expr_upper,
        )
        p_upper = guarantee_arg_within_expr_bounds(
            lambda p_upper: _trunc_modulo(p_upper, qv),
            exprv,
            pv,
            p_upper,
            p_upper_expr_lower,
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
        return f"trunc_modulo({self._p!r}, {self._q!r})"


class ScalarRoundTiesEvenModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

        if self._q.has_data:
            raise NotImplementedError(
                "`round_ties_even_modulo(p, q)` with non-constant divisor `q`"
            )

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
            "cannot compute the data bounds for round_ties_even_modulo(p, q) with non-constant q"
        )
        assert not (p_const and q_const), (
            "constant round_ties_even_modulo has no data bounds"
        )

        # evaluate p, q and round_ties_even_modulo(p, q)
        p, q = self._p, self._q
        pv = p.eval(Xs, late_bound)
        qv = q.eval(Xs, late_bound)
        exprv = _round_ties_even_modulo(pv, qv)

        qv2: np.ndarray[tuple[Ps], np.dtype[F]] = np.divide(np.abs(qv), 2)

        # the bounds on round_ties_even_modulo(...) are [-|q/2|, +|q/2|]
        rem_expr_lower = _maximum_zero_sign_sensitive(-qv2, expr_lower)
        rem_expr_upper = _minimum_zero_sign_sensitive(expr_upper, qv2)

        # round_ties_even_modulo(...) is periodic, so we need to drop to
        #  difference bounds before applying the difference to pv to stay in
        #  the same period
        p_lower_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_lower, exprv
        )
        p_upper_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_upper, exprv
        )

        # check for the case where any finite value would work for pv
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, -qv2
        ) & np.greater_equal(expr_upper, qv2)

        fmax = np.finfo(Xs.dtype).max

        # round_ties_even_modulo(pv, NaN) = NaN for any pv
        # round_ties_even_modulo(NaN, qv) = NaN, keep pv NaN
        # round_ties_even_modulo(pv, 0) = NaN for any pv
        # round_ties_even_modulo(+-Inf, qv) = NaN, keep pv infinite
        # round_ties_even_modulo(pv, +-Inf) = pv, so use normal pv bounds
        # if the full domain is ok, allow the full finite domain for pv
        # otherwise, apply the bounds to the current repetition
        # propagate -0.0 and +0.0 bounds on pv to avoid nudging
        # if p_lower == pv and pv == -0.0, we need to guarantee that
        #  p_lower is also -0.0, same for p_upper

        p_lower = _ensure_array(pv, copy=True)
        np.add(p_lower, p_lower_diff, out=p_lower)
        np.copyto(
            p_lower,
            rem_expr_lower,
            where=((p_lower == 0) & (rem_expr_lower == 0)),
            casting="no",
        )
        p_lower[full_domain] = -fmax
        np.copyto(p_lower, pv, where=np.isinf(pv), casting="no")
        p_lower[qv == 0] = Xs.dtype.type(-np.inf)
        np.copyto(p_lower, pv, where=np.isnan(pv), casting="no")
        p_lower[np.isnan(qv)] = Xs.dtype.type(-np.inf)
        _minimum_zero_sign_sensitive(pv, p_lower, out=p_lower)

        p_upper = _ensure_array(pv, copy=True)
        np.add(p_upper, p_upper_diff, out=p_upper)
        np.copyto(
            p_upper,
            rem_expr_upper,
            where=((p_upper == 0) & (rem_expr_upper == 0)),
            casting="no",
        )
        p_upper[full_domain] = fmax
        np.copyto(p_upper, pv, where=np.isinf(pv), casting="no")
        p_upper[qv == 0] = Xs.dtype.type(np.inf)
        np.copyto(p_upper, pv, where=np.isnan(pv), casting="no")
        p_upper[np.isnan(qv)] = Xs.dtype.type(np.inf)
        _maximum_zero_sign_sensitive(pv, p_upper, out=p_upper)

        # handle rounding errors in round_ties_even_modulo early
        p_lower = guarantee_arg_within_expr_bounds(
            lambda p_lower: _round_ties_even_modulo(p_lower, qv),
            exprv,
            pv,
            p_lower,
            expr_lower,
            expr_upper,
        )
        p_upper = guarantee_arg_within_expr_bounds(
            lambda p_upper: _round_ties_even_modulo(p_upper, qv),
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
        return f"round_ties_even_modulo({self._p!r}, {self._q!r})"


class ScalarEuclideanModulo(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_p", "_q")
    _p: AnyExpr
    _q: AnyExpr

    def __init__(self, p: AnyExpr, q: AnyExpr) -> None:
        self._p = p
        self._q = q

        if self._q.has_data:
            raise NotImplementedError(
                "`euclidean_modulo(p, q)` with non-constant divisor `q`"
            )

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
            "cannot compute the data bounds for euclidean_modulo(p, q) with non-constant q"
        )
        assert not (p_const and q_const), "constant euclidean_modulo has no data bounds"

        # evaluate p, q and euclidean_modulo(p, q)
        p, q = self._p, self._q
        pv = p.eval(Xs, late_bound)
        qv = q.eval(Xs, late_bound)
        exprv = _euclidean_modulo(pv, qv)

        # the bounds on euclidean_modulo(...) are [+0.0, |q|)
        rem_expr_lower = _maximum_zero_sign_sensitive(Xs.dtype.type(+0.0), expr_lower)
        rem_expr_upper = _minimum_zero_sign_sensitive(expr_upper, np.abs(qv))

        # euclidean_modulo(...) is periodic, so we need to drop to difference
        #  bounds before applying the difference to pv to stay in the
        #  same period
        p_lower_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_lower, exprv
        )
        p_upper_diff: np.ndarray[tuple[Ps], np.dtype[F]] = np.subtract(
            rem_expr_upper, exprv
        )

        # check for the case where any finite value would work for pv
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, 0
        ) & np.greater_equal(expr_upper, np.abs(qv))

        fmax = np.finfo(Xs.dtype).max
        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # euclidean_modulo(pv, NaN) = NaN for any pv
        # euclidean_modulo(NaN, qv) = NaN, keep pv NaN
        # euclidean_modulo(pv, 0) = NaN for any pv
        # euclidean_modulo(+-Inf, qv) = NaN, keep pv infinite
        # euclidean_modulo(pv, +-Inf) = ??
        #  - if the bounds include +Inf, negative pv is allowed
        #    - if the bounds exclude zero, only allow negative pv
        #  - if the bounds only include zero, restrict pv to zero
        #  - if the bounds exclude +Inf, only positive pv is allowed
        # if the full domain is ok, allow the full finite domain for pv
        # otherwise, apply the bounds to the current repetition
        # if p_lower == pv and pv == -0.0, we need to guarantee that
        #  p_lower is also -0.0, same for p_upper

        p_lower = _ensure_array(pv, copy=True)
        np.add(p_lower, p_lower_diff, out=p_lower)
        np.copyto(  # exact bounds around zero if positive
            p_lower,
            rem_expr_lower,
            where=((np.abs(pv) < np.abs(qv)) & (pv >= 0)),
            casting="no",
        )
        _maximum_zero_sign_sensitive(  # we don't allow repetition slips
            p_lower, Xs.dtype.type(-0.0), out=p_lower, where=(pv >= 0)
        )
        exact_p_lower = _zeros(pv.shape, np.dtype(np.bool))
        p_lower[full_domain] = -fmax
        exact_p_lower |= full_domain
        p_lower[np.isinf(qv) & np.isposinf(rem_expr_upper)] = -fmax
        exact_p_lower |= np.isinf(qv) & np.isposinf(rem_expr_upper)
        p_lower[np.isinf(qv) & (rem_expr_upper == 0)] = Xs.dtype.type(-0.0)
        exact_p_lower |= np.isinf(qv) & (rem_expr_upper == 0)
        _maximum_zero_sign_sensitive(
            p_lower,
            Xs.dtype.type(+0.0),
            out=p_lower,
            where=(np.isinf(qv) & ~np.isposinf(rem_expr_upper)),
        )
        # if the zero boundary was within the approximate bounds, it will be
        # exact since zero is an exact boundary and the bounds would not be
        # inaccurate in magnitude beyond their size
        exact_p_lower |= np.isinf(qv) & ~np.isposinf(rem_expr_upper) & (p_lower == 0)
        np.copyto(p_lower, pv, where=np.isinf(pv), casting="no")
        exact_p_lower |= np.isinf(pv)
        p_lower[qv == 0] = Xs.dtype.type(-np.inf)
        exact_p_lower |= qv == 0
        np.copyto(p_lower, pv, where=np.isnan(pv), casting="no")
        exact_p_lower |= np.isnan(pv)
        p_lower[np.isnan(qv)] = Xs.dtype.type(-np.inf)
        exact_p_lower |= np.isnan(qv)
        _minimum_zero_sign_sensitive(pv, p_lower, out=p_lower)

        p_upper = _ensure_array(pv, copy=True)
        np.add(p_upper, p_upper_diff, out=p_upper)
        np.copyto(  # exact bounds around zero if positive
            p_upper,
            rem_expr_upper,
            where=((np.abs(pv) < np.abs(qv)) & (pv >= 0)),
            casting="no",
        )
        _minimum_zero_sign_sensitive(  # we don't allow repetition slips
            p_upper, Xs.dtype.type(+0.0), out=p_upper, where=(pv <= 0)
        )
        exact_p_upper = _zeros(pv.shape, np.dtype(np.bool))
        p_upper[full_domain] = fmax
        exact_p_upper |= full_domain
        p_upper[np.isinf(qv) & (rem_expr_upper == 0)] = Xs.dtype.type(+0.0)
        exact_p_upper |= np.isinf(qv) & (rem_expr_upper == 0)
        p_upper[np.isinf(qv) & (pv < 0) & (rem_expr_lower > 0)] = -smallest_subnormal
        exact_p_upper |= np.isinf(qv) & (pv < 0) & (rem_expr_lower > 0)
        np.copyto(p_upper, pv, where=np.isinf(pv), casting="no")
        exact_p_upper |= np.isinf(pv)
        p_upper[qv == 0] = Xs.dtype.type(np.inf)
        exact_p_upper |= qv == 0
        np.copyto(p_upper, pv, where=np.isnan(pv), casting="no")
        exact_p_upper |= np.isnan(pv)
        p_upper[np.isnan(qv)] = Xs.dtype.type(np.inf)
        exact_p_upper |= np.isnan(qv)
        _maximum_zero_sign_sensitive(pv, p_upper, out=p_upper)

        # if p_lower/p_upper is not exact, it may have rounding errors,
        # so we guard against slipping into the next repetition
        p_lower_expr_upper = _where(
            exact_p_lower,
            expr_upper,
            exprv,
        )
        p_upper_expr_lower = _where(
            exact_p_upper,
            expr_lower,
            exprv,
        )

        # handle rounding errors in euclidean_modulo early
        p_lower = guarantee_arg_within_expr_bounds(
            lambda p_lower: _euclidean_modulo(p_lower, qv),
            exprv,
            pv,
            p_lower,
            expr_lower,
            p_lower_expr_upper,
        )
        p_upper = guarantee_arg_within_expr_bounds(
            lambda p_upper: _euclidean_modulo(p_upper, qv),
            exprv,
            pv,
            p_upper,
            p_upper_expr_lower,
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
        return f"euclidean_modulo({self._p!r}, {self._q!r})"

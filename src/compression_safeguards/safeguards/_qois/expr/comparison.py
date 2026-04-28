from collections.abc import Callable, Mapping
from functools import partial
from typing import overload

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Callback, Context, Expr
from .constfold import ScalarFoldedConstant


class ScalarEqual(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarEqual":
        return ScalarEqual(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: comparison_to_dtype(np.equal, a, b, dtype),
            ScalarEqual,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return comparison_to_dtype(
            lambda a, b: a == b,
            self._a.eval(Xs, late_bound),
            self._b.eval(Xs, late_bound),
            Xs.dtype,
        )

    @checked_data_bounds
    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context,
        callback: Callback[Ps, Ns, F],
    ) -> None:
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant == has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        bv_nxt_av = _ensure_array(np.nextafter(bv, av))

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.add(
                np.divide(np.nan_to_num(av), Xs.dtype.type(2)),
                np.divide(np.nan_to_num(bv), Xs.dtype.type(2)),
            )
        )
        # the midpoint must be in range [av, bv), unless av == bv, since we
        #  later want to be able to step from the midpoint closer towards bv
        _maximum_zero_sign_sensitive(
            av_mid_bv, _minimum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )
        _minimum_zero_sign_sensitive(
            av_mid_bv, _maximum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the midpoint
        if not b_const:
            np.copyto(
                bv_nxt_av,
                av_mid_bv,
                where=b.eval_has_data(Xs, late_bound),
                casting="no",
            )

        # compute the next value from a towards b that can be part of b's
        #  safe interval
        # if a is const this is the literal next value
        # otherwise it is the next value from the midpoint towards b
        av_nxt_bv = _ensure_array(av, copy=True)
        if not a_const:
            np.copyto(
                av_nxt_bv,
                av_mid_bv,
                where=a.eval_has_data(Xs, late_bound),
                casting="no",
            )
        av_nxt_bv = np.nextafter(av_nxt_bv, bv)

        Xs_lower_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_upper_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )

        term_callbacks_done = [False, False]
        callback_done = [False]

        def callback_wrapper(
            Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
            Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
            *,
            j: int,
        ) -> None:
            # combine the inner data bounds
            _maximum_zero_sign_sensitive(Xs_lower_out, Xs_lower, out=Xs_lower_out)
            _minimum_zero_sign_sensitive(Xs_upper_out, Xs_upper, out=Xs_upper_out)

            term_callbacks_done[j] = True

            if all(term_callbacks_done) and not callback_done[0]:
                callback_done[0] = True

                # ensure that the bounds on Xs include Xs
                _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
                _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

                return callback(Xs_lower_out, Xs_upper_out)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a == b) = True, then
        #  - keep av and bv the same
        #  - we do *not* special-case -0.0 and +0.0 here
        # if expr_upper < 1, (a == b) = False, then
        #  - if av > bv, restrict a's lower and b's upper bound to not overlap
        #  - if av < bv, restrict a's upper and b's lower bound to not overlap
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # otherwise, (a == b) in [True, False] and all args can be anything
        if a_const:
            term_callbacks_done[0] = True
        else:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(a_lower, av, where=np.greater(expr_lower, 0), casting="no")
            np.copyto(
                a_lower,
                bv_nxt_av,
                where=(np.less(expr_upper, 1) & (av > bv)),
                casting="no",
            )

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(a_upper, av, where=np.greater(expr_lower, 0), casting="no")
            np.copyto(
                a_upper,
                bv_nxt_av,
                where=(np.less(expr_upper, 1) & (av < bv)),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(av, a_lower, out=a_lower)
            _maximum_zero_sign_sensitive(av, a_upper, out=a_upper)

            # recurse into arg a
            a.deferred_compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=0),
            )

        # cont. with b as outlined above
        if b_const:
            term_callbacks_done[1] = True
        else:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(b_lower, bv, where=np.greater(expr_lower, 0), casting="no")
            np.copyto(
                b_lower,
                av_nxt_bv,
                where=(np.less(expr_upper, 1) & (bv > av)),
                casting="no",
            )

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(b_upper, bv, where=np.greater(expr_lower, 0), casting="no")
            np.copyto(
                b_upper,
                av_nxt_bv,
                where=(np.less(expr_upper, 1) & (bv < av)),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(bv, b_lower, out=b_lower)
            _maximum_zero_sign_sensitive(bv, b_upper, out=b_upper)

            # recurse into arg b
            b.deferred_compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=1),
            )

        if all(term_callbacks_done) and not callback_done[0]:
            callback_done[0] = True

            # ensure that the bounds on Xs include Xs
            _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
            _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

            return callback(Xs_lower_out, Xs_upper_out)

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} == {self._b!r}"


class ScalarNotEqual(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarNotEqual":
        return ScalarNotEqual(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: comparison_to_dtype(np.not_equal, a, b, dtype),
            ScalarNotEqual,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return comparison_to_dtype(
            lambda a, b: a != b,
            self._a.eval(Xs, late_bound),
            self._b.eval(Xs, late_bound),
            Xs.dtype,
        )

    @checked_data_bounds
    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context,
        callback: Callback[Ps, Ns, F],
    ) -> None:
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant != has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        bv_nxt_av = _ensure_array(np.nextafter(bv, av))

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.add(
                np.divide(np.nan_to_num(av), Xs.dtype.type(2)),
                np.divide(np.nan_to_num(bv), Xs.dtype.type(2)),
            )
        )
        # the midpoint must be in range [av, bv), unless av == bv, since we
        #  later want to be able to step from the midpoint closer towards bv
        _maximum_zero_sign_sensitive(
            av_mid_bv, _minimum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )
        _minimum_zero_sign_sensitive(
            av_mid_bv, _maximum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the midpoint
        if not b_const:
            np.copyto(
                bv_nxt_av,
                av_mid_bv,
                where=b.eval_has_data(Xs, late_bound),
                casting="no",
            )

        # compute the next value from a towards b that can be part of b's
        #  safe interval
        # if a is const this is the literal next value
        # otherwise it is the next value from the midpoint towards b
        av_nxt_bv = _ensure_array(av, copy=True)
        if not a_const:
            np.copyto(
                av_nxt_bv,
                av_mid_bv,
                where=a.eval_has_data(Xs, late_bound),
                casting="no",
            )
        av_nxt_bv = np.nextafter(av_nxt_bv, bv)

        Xs_lower_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_upper_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )

        term_callbacks_done = [False, False]
        callback_done = [False]

        def callback_wrapper(
            Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
            Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
            *,
            j: int,
        ) -> None:
            # combine the inner data bounds
            _maximum_zero_sign_sensitive(Xs_lower_out, Xs_lower, out=Xs_lower_out)
            _minimum_zero_sign_sensitive(Xs_upper_out, Xs_upper, out=Xs_upper_out)

            term_callbacks_done[j] = True

            if all(term_callbacks_done) and not callback_done[0]:
                callback_done[0] = True

                # ensure that the bounds on Xs include Xs
                _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
                _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

                return callback(Xs_lower_out, Xs_upper_out)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a != b) = True, then
        #  - if av > bv, restrict a's lower and b's upper bound to not overlap
        #  - if av < bv, restrict a's upper and b's lower bound to not overlap
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # if expr_upper < 1, (a != b) = False, then
        #  - keep av and bv the same
        #  - we do *not* special-case -0.0 and +0.0 here
        # otherwise, (a != b) in [True, False] and all args can be anything
        if a_const:
            term_callbacks_done[0] = True
        else:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                a_lower,
                bv_nxt_av,
                where=(np.greater(expr_lower, 0) & (av > bv)),
                casting="no",
            )
            np.copyto(a_lower, av, where=np.less(expr_upper, 1), casting="no")

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                a_upper,
                bv_nxt_av,
                where=(np.greater(expr_lower, 0) & (av < bv)),
                casting="no",
            )
            np.copyto(a_upper, av, where=np.less(expr_upper, 1), casting="no")

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(av, a_lower, out=a_lower)
            _maximum_zero_sign_sensitive(av, a_upper, out=a_upper)

            # recurse into arg a
            a.deferred_compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=0),
            )

        # cont. with b as outlined above
        if b_const:
            term_callbacks_done[1] = True
        else:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                b_lower,
                av_nxt_bv,
                where=(np.greater(expr_lower, 0) & (bv > av)),
                casting="no",
            )
            np.copyto(b_lower, bv, where=np.less(expr_upper, 1), casting="no")

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                b_upper,
                av_nxt_bv,
                where=(np.greater(expr_lower, 0) & (bv < av)),
                casting="no",
            )
            np.copyto(b_upper, bv, where=np.less(expr_upper, 1), casting="no")

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(bv, b_lower, out=b_lower)
            _maximum_zero_sign_sensitive(bv, b_upper, out=b_upper)

            # recurse into arg b
            b.deferred_compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=1),
            )

        if all(term_callbacks_done) and not callback_done[0]:
            callback_done[0] = True

            # ensure that the bounds on Xs include Xs
            _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
            _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

            return callback(Xs_lower_out, Xs_upper_out)

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} != {self._b!r}"


class ScalarLess(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarLess":
        return ScalarLess(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: comparison_to_dtype(np.less, a, b, dtype),
            ScalarLess,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return comparison_to_dtype(
            lambda a, b: a < b,  # type: ignore
            self._a.eval(Xs, late_bound),
            self._b.eval(Xs, late_bound),
            Xs.dtype,
        )

    @checked_data_bounds
    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context,
        callback: Callback[Ps, Ns, F],
    ) -> None:
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant < has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        bv_nxt_av = _ensure_array(np.nextafter(bv, av))

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.add(
                np.divide(np.nan_to_num(av), Xs.dtype.type(2)),
                np.divide(np.nan_to_num(bv), Xs.dtype.type(2)),
            )
        )
        # the midpoint must be in range [av, bv), unless av == bv, since we
        #  later want to be able to step from the midpoint closer towards bv
        _maximum_zero_sign_sensitive(
            av_mid_bv, _minimum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )
        _minimum_zero_sign_sensitive(
            av_mid_bv, _maximum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the midpoint
        if not b_const:
            np.copyto(
                bv_nxt_av,
                av_mid_bv,
                where=b.eval_has_data(Xs, late_bound),
                casting="no",
            )

        # compute the next value from a towards b that can be part of b's
        #  safe interval
        # if a is const this is the literal next value
        # otherwise it is the next value from the midpoint towards b
        av_nxt_bv = _ensure_array(av, copy=True)
        if not a_const:
            np.copyto(
                av_nxt_bv,
                av_mid_bv,
                where=a.eval_has_data(Xs, late_bound),
                casting="no",
            )
        av_nxt_bv = np.nextafter(av_nxt_bv, bv)

        Xs_lower_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_upper_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )

        term_callbacks_done = [False, False]
        callback_done = [False]

        def callback_wrapper(
            Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
            Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
            *,
            j: int,
        ) -> None:
            # combine the inner data bounds
            _maximum_zero_sign_sensitive(Xs_lower_out, Xs_lower, out=Xs_lower_out)
            _minimum_zero_sign_sensitive(Xs_upper_out, Xs_upper, out=Xs_upper_out)

            term_callbacks_done[j] = True

            if all(term_callbacks_done) and not callback_done[0]:
                callback_done[0] = True

                # ensure that the bounds on Xs include Xs
                _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
                _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

                return callback(Xs_lower_out, Xs_upper_out)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a < b) = True, then
        #  - restrict a's upper and b's lower bound to not overlap
        # if expr_upper < 1, (a < b) = False, then
        #  - if av >= bv, restrict a's lower and b's upper bound
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # otherwise, (a < b) in [True, False] and all args can be anything
        if a_const:
            term_callbacks_done[0] = True
        else:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                a_lower,
                bv_nxt_av,  # a >= b and bv_nxt_av >= b and >= mid_b
                where=(np.less(expr_upper, 1) & (av >= bv)),
                casting="no",
            )

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                a_upper,
                bv_nxt_av,  # a < b and bv_nxt_av < b and < mid_b
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(av, a_lower, out=a_lower)
            _maximum_zero_sign_sensitive(av, a_upper, out=a_upper)

            # recurse into arg a
            a.deferred_compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=0),
            )

        # cont. with b as outlined above
        if b_const:
            term_callbacks_done[1] = True
        else:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                b_lower,
                av_nxt_bv,  # a < b and av_nxt_bv > a and > mid_a
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                b_upper,
                av_nxt_bv,  # a >= b and av_nxt_bv <= a and <= mid_a
                where=(np.less(expr_upper, 1) & (av >= bv)),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(bv, b_lower, out=b_lower)
            _maximum_zero_sign_sensitive(bv, b_upper, out=b_upper)

            # recurse into arg b
            b.deferred_compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=1),
            )

        if all(term_callbacks_done) and not callback_done[0]:
            callback_done[0] = True

            # ensure that the bounds on Xs include Xs
            _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
            _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

            return callback(Xs_lower_out, Xs_upper_out)

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} < {self._b!r}"


class ScalarGreaterEqual(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarGreaterEqual":
        return ScalarGreaterEqual(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: comparison_to_dtype(np.greater_equal, a, b, dtype),
            ScalarGreaterEqual,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return comparison_to_dtype(
            lambda a, b: a >= b,  # type: ignore
            self._a.eval(Xs, late_bound),
            self._b.eval(Xs, late_bound),
            Xs.dtype,
        )

    @checked_data_bounds
    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context,
        callback: Callback[Ps, Ns, F],
    ) -> None:
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant >= has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        bv_nxt_av = _ensure_array(np.nextafter(bv, av))

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.add(
                np.divide(np.nan_to_num(av), Xs.dtype.type(2)),
                np.divide(np.nan_to_num(bv), Xs.dtype.type(2)),
            )
        )
        # the midpoint must be in range [av, bv), unless av == bv, since we
        #  later want to be able to step from the midpoint closer towards bv
        _maximum_zero_sign_sensitive(
            av_mid_bv, _minimum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )
        _minimum_zero_sign_sensitive(
            av_mid_bv, _maximum_zero_sign_sensitive(av, bv_nxt_av), out=av_mid_bv
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the midpoint
        if not b_const:
            np.copyto(
                bv_nxt_av,
                av_mid_bv,
                where=b.eval_has_data(Xs, late_bound),
                casting="no",
            )

        # compute the next value from a towards b that can be part of b's
        #  safe interval
        # if a is const this is the literal next value
        # otherwise it is the next value from the midpoint towards b
        av_nxt_bv = _ensure_array(av, copy=True)
        if not a_const:
            np.copyto(
                av_nxt_bv,
                av_mid_bv,
                where=a.eval_has_data(Xs, late_bound),
                casting="no",
            )
        av_nxt_bv = np.nextafter(av_nxt_bv, bv)

        Xs_lower_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_upper_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )

        term_callbacks_done = [False, False]
        callback_done = [False]

        def callback_wrapper(
            Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
            Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
            *,
            j: int,
        ) -> None:
            # combine the inner data bounds
            _maximum_zero_sign_sensitive(Xs_lower_out, Xs_lower, out=Xs_lower_out)
            _minimum_zero_sign_sensitive(Xs_upper_out, Xs_upper, out=Xs_upper_out)

            term_callbacks_done[j] = True

            if all(term_callbacks_done) and not callback_done[0]:
                callback_done[0] = True

                # ensure that the bounds on Xs include Xs
                _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
                _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

                return callback(Xs_lower_out, Xs_upper_out)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a >= b) = True, then
        #  - restrict a's lower and b's upper bound
        # if expr_upper < 1, (a >= b) = False, then
        #  - if av < bv, restrict a's upper and b's lower bound to not overlap
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # otherwise, (a >= b) in [True, False] and all args can be anything
        if a_const:
            term_callbacks_done[0] = True
        else:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                a_lower,
                bv_nxt_av,  # a >= b and bv_nxt_av >= b and >= mid_b
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                a_upper,
                bv_nxt_av,  # a < b and bv_nxt_av < b and < mid_b
                where=(np.less(expr_upper, 1) & (av < bv)),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(av, a_lower, out=a_lower)
            _maximum_zero_sign_sensitive(av, a_upper, out=a_upper)

            # recurse into arg a
            a.deferred_compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=0),
            )

        # cont. with b as outlined above
        if b_const:
            term_callbacks_done[1] = True
        else:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                b_lower,
                av_nxt_bv,  # a < b and av_nxt_bv > a and > mid_a
                where=(np.less(expr_upper, 1) & (av < bv)),
                casting="no",
            )

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                b_upper,
                av_nxt_bv,  # a >= b and av_nxt_bv <= a and <= mid_a
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(bv, b_lower, out=b_lower)
            _maximum_zero_sign_sensitive(bv, b_upper, out=b_upper)

            # recurse into arg b
            b.deferred_compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=1),
            )

        if all(term_callbacks_done) and not callback_done[0]:
            callback_done[0] = True

            # ensure that the bounds on Xs include Xs
            _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
            _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

            return callback(Xs_lower_out, Xs_upper_out)

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} >= {self._b!r}"


class ScalarLessEqual(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarLessEqual":
        return ScalarLessEqual(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: comparison_to_dtype(np.less_equal, a, b, dtype),
            ScalarLessEqual,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return comparison_to_dtype(
            lambda a, b: a <= b,  # type: ignore
            self._a.eval(Xs, late_bound),
            self._b.eval(Xs, late_bound),
            Xs.dtype,
        )

    @checked_data_bounds
    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context,
        callback: Callback[Ps, Ns, F],
    ) -> None:
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant <= has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        av_nxt_bv = _ensure_array(np.nextafter(av, bv))

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.add(
                np.divide(np.nan_to_num(av), Xs.dtype.type(2)),
                np.divide(np.nan_to_num(bv), Xs.dtype.type(2)),
            )
        )
        # the midpoint must be in range (av, bv], unless av == bv, since we
        #  later want to be able to step from the midpoint closer towards av
        _maximum_zero_sign_sensitive(
            av_mid_bv, _minimum_zero_sign_sensitive(av_nxt_bv, bv), out=av_mid_bv
        )
        _minimum_zero_sign_sensitive(
            av_mid_bv, _maximum_zero_sign_sensitive(av_nxt_bv, bv), out=av_mid_bv
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the next value from the midpoint towards a
        bv_nxt_av = _ensure_array(bv, copy=True)
        if not b_const:
            np.copyto(
                bv_nxt_av,
                av_mid_bv,
                where=b.eval_has_data(Xs, late_bound),
                casting="no",
            )
        bv_nxt_av = np.nextafter(bv_nxt_av, av)

        # compute the next value from a towards b that can be part of b's
        #  safe interval
        # if a is const this is the literal next value
        # otherwise it is the midpoint
        if not a_const:
            np.copyto(
                av_nxt_bv,
                av_mid_bv,
                where=a.eval_has_data(Xs, late_bound),
                casting="no",
            )

        Xs_lower_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_upper_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )

        term_callbacks_done = [False, False]
        callback_done = [False]

        def callback_wrapper(
            Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
            Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
            *,
            j: int,
        ) -> None:
            # combine the inner data bounds
            _maximum_zero_sign_sensitive(Xs_lower_out, Xs_lower, out=Xs_lower_out)
            _minimum_zero_sign_sensitive(Xs_upper_out, Xs_upper, out=Xs_upper_out)

            term_callbacks_done[j] = True

            if all(term_callbacks_done) and not callback_done[0]:
                callback_done[0] = True

                # ensure that the bounds on Xs include Xs
                _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
                _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

                return callback(Xs_lower_out, Xs_upper_out)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a <= b) = True, then
        #  - restrict a's upper and b's lower bound
        # if expr_upper < 1, (a <= b) = False, then
        #  - if av > bv, restrict a's lower and b's upper bound to not overlap
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # otherwise, (a <= b) in [True, False] and all args can be anything
        if a_const:
            term_callbacks_done[0] = True
        else:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                a_lower,
                bv_nxt_av,  # a > b and bv_nxt_av > b and > mid_b
                where=(np.less(expr_upper, 1) & (av > bv)),
                casting="no",
            )

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                a_upper,
                bv_nxt_av,  # a <= b and bv_nxt_av <= b and <= mid_b
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(av, a_lower, out=a_lower)
            _maximum_zero_sign_sensitive(av, a_upper, out=a_upper)

            # recurse into arg a
            a.deferred_compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=0),
            )

        # cont. with b as outlined above
        if b_const:
            term_callbacks_done[1] = True
        else:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                b_lower,
                av_nxt_bv,  # a <= b and av_nxt_bv >= a and >= mid_a
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                b_upper,
                av_nxt_bv,  # a > b and av_nxt_bv < a and < mid_a
                where=(np.less(expr_upper, 1) & (av > bv)),
                casting="no",
            )

            _minimum_zero_sign_sensitive(bv, b_lower, out=b_lower)
            _maximum_zero_sign_sensitive(bv, b_upper, out=b_upper)

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            # recurse into arg b
            b.deferred_compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=1),
            )

        if all(term_callbacks_done) and not callback_done[0]:
            callback_done[0] = True

            # ensure that the bounds on Xs include Xs
            _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
            _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

            return callback(Xs_lower_out, Xs_upper_out)

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} <= {self._b!r}"


class ScalarGreater(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @property
    @override
    def extra(self) -> tuple[()]:
        return ()

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarGreater":
        return ScalarGreater(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a,
            self._b,
            dtype,
            lambda a, b: comparison_to_dtype(np.greater, a, b, dtype),
            ScalarGreater,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return comparison_to_dtype(
            lambda a, b: a > b,  # type: ignore
            self._a.eval(Xs, late_bound),
            self._b.eval(Xs, late_bound),
            Xs.dtype,
        )

    @checked_data_bounds
    @override
    def deferred_compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
        ctx: Context,
        callback: Callback[Ps, Ns, F],
    ) -> None:
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant > has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        av_nxt_bv = _ensure_array(np.nextafter(av, bv))

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.add(
                np.divide(np.nan_to_num(av), Xs.dtype.type(2)),
                np.divide(np.nan_to_num(bv), Xs.dtype.type(2)),
            )
        )
        # the midpoint must be in range (av, bv], unless av == bv, since we
        #  later want to be able to step from the midpoint closer towards av
        _maximum_zero_sign_sensitive(
            av_mid_bv, _minimum_zero_sign_sensitive(av_nxt_bv, bv), out=av_mid_bv
        )
        _minimum_zero_sign_sensitive(
            av_mid_bv, _maximum_zero_sign_sensitive(av_nxt_bv, bv), out=av_mid_bv
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the next value from the midpoint towards a
        bv_nxt_av = _ensure_array(bv, copy=True)
        if not b_const:
            np.copyto(
                bv_nxt_av,
                av_mid_bv,
                where=b.eval_has_data(Xs, late_bound),
                casting="no",
            )
        bv_nxt_av = np.nextafter(bv_nxt_av, av)

        # compute the next value from a towards b that can be part of b's
        #  safe interval
        # if a is const this is the literal next value
        # otherwise it is the midpoint
        if not a_const:
            np.copyto(
                av_nxt_bv,
                av_mid_bv,
                where=a.eval_has_data(Xs, late_bound),
                casting="no",
            )

        Xs_lower_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(-np.inf)
        )
        Xs_upper_out: np_sndarray[Ps, Ns, np.dtype[F]] = np.full(
            Xs.shape, Xs.dtype.type(np.inf)
        )

        term_callbacks_done = [False, False]
        callback_done = [False]

        def callback_wrapper(
            Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]],
            Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]],
            *,
            j: int,
        ) -> None:
            # combine the inner data bounds
            _maximum_zero_sign_sensitive(Xs_lower_out, Xs_lower, out=Xs_lower_out)
            _minimum_zero_sign_sensitive(Xs_upper_out, Xs_upper, out=Xs_upper_out)

            term_callbacks_done[j] = True

            if all(term_callbacks_done) and not callback_done[0]:
                callback_done[0] = True

                _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
                _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

                return callback(Xs_lower_out, Xs_upper_out)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a > b) = True, then
        #  - restrict a's lower and b's upper bound to not overlap
        # if expr_upper < 1, (a > b) = False, then
        #  - if av <= bv, restrict a's upper and b's lower bound
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # otherwise, (a > b) in [True, False] and all args can be anything
        if a_const:
            term_callbacks_done[0] = True
        else:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                a_lower,
                bv_nxt_av,  # a > b and bv_nxt_av > b and > mid_b
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                a_upper,
                bv_nxt_av,  # a <= b and bv_nxt_av <= b and <= mid_b
                where=(np.less(expr_upper, 1) & (av <= bv)),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(av, a_lower, out=a_lower)
            _maximum_zero_sign_sensitive(av, a_upper, out=a_upper)

            # recurse into arg a
            a.deferred_compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=0),
            )

        # cont. with b as outlined above
        if b_const:
            term_callbacks_done[1] = True
        else:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                b_lower,
                av_nxt_bv,  # a <= b and av_nxt_bv >= a and >= mid_a
                where=(np.less(expr_upper, 1) & (av <= bv)),
                casting="no",
            )

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                b_upper,
                av_nxt_bv,  # a > b and av_nxt_bv < a and < mid_a
                where=np.greater(expr_lower, 0),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            _minimum_zero_sign_sensitive(bv, b_lower, out=b_lower)
            _maximum_zero_sign_sensitive(bv, b_upper, out=b_upper)

            # recurse into arg b
            b.deferred_compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
                ctx,
                partial(callback_wrapper, j=1),
            )

        if all(term_callbacks_done) and not callback_done[0]:
            callback_done[0] = True

            _minimum_zero_sign_sensitive(Xs_lower_out, Xs, out=Xs_lower_out)
            _maximum_zero_sign_sensitive(Xs_upper_out, Xs, out=Xs_upper_out)

            return callback(Xs_lower_out, Xs_upper_out)

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} > {self._b!r}"


@overload
def comparison_to_dtype(
    classify: Callable[
        [np.ndarray[tuple[Ps], np.dtype[F]], np.ndarray[tuple[Ps], np.dtype[F]]],
        np.ndarray[tuple[Ps], np.dtype[np.bool]],
    ],
    a: np.ndarray[tuple[Ps], np.dtype[F]],
    b: np.ndarray[tuple[Ps], np.dtype[F]],
    dtype: np.dtype[F],
) -> np.ndarray[tuple[Ps], np.dtype[F]]: ...


@overload
def comparison_to_dtype(
    classify: Callable[[Fi, Fi], bool],
    a: Fi,
    b: Fi,
    dtype: np.dtype[Fi],
) -> Fi: ...


def comparison_to_dtype(comparison, a, b, dtype):
    c = comparison(a, b)

    if not isinstance(c, np.ndarray):
        return _ensure_array(c).astype(dtype, casting="safe")[()]

    return c.astype(dtype, casting="safe")

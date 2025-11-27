from collections.abc import Callable, Mapping
from typing import overload

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _nan_to_zero_inf_to_finite,
    _nextafter,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
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

    @override
    @checked_data_bounds
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
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant == has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = np.add(  # type: ignore
            np.divide(_nan_to_zero_inf_to_finite(av), Xs.dtype.type(2)),
            np.divide(_nan_to_zero_inf_to_finite(bv), Xs.dtype.type(2)),
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the midpoint
        bv_nxt_av = _ensure_array(_nextafter(bv, av))
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
        av_nxt_bv = _nextafter(av_nxt_bv, bv)

        xl: np_sndarray[Ps, Ns, np.dtype[F]]
        xu: np_sndarray[Ps, Ns, np.dtype[F]]
        Xs_lower_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None
        Xs_upper_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a == b) = True, then
        #  - keep av and bv the same
        #  - we do *not* special-case -0.0 and +0.0 here
        # if expr_upper < 1, (a == b) = False, then
        #  - if av > bv, restrict a's lower and b's upper bound to not overlap
        #  - if av < bv, restrict a's upper and b's lower bound to not overlap
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # otherwise, (a == b) in [True, False] and all args can be anything
        if not a_const:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(a_lower, av, where=np.greater_equal(expr_lower, 0), casting="no")
            np.copyto(
                a_lower,
                bv_nxt_av,
                where=(np.less_equal(expr_upper, 1) & (av > bv)),
                casting="no",
            )

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(a_upper, av, where=np.greater_equal(expr_lower, 0), casting="no")
            np.copyto(
                a_upper,
                bv_nxt_av,
                where=(np.less_equal(expr_upper, 1) & (av < bv)),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            # recurse into arg a
            Xs_lower_, Xs_upper_ = a.compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
            )

        # cont. with b as outlined above
        if not b_const:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(b_lower, bv, where=np.greater_equal(expr_lower, 0), casting="no")
            np.copyto(
                b_lower,
                av_nxt_bv,
                where=(np.less_equal(expr_upper, 1) & (bv > av)),
                casting="no",
            )

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(b_upper, bv, where=np.greater_equal(expr_lower, 0), casting="no")
            np.copyto(
                b_upper,
                av_nxt_bv,
                where=(np.less_equal(expr_upper, 1) & (bv < av)),
                casting="no",
            )

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            # recurse into arg b
            xl, xu = b.compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
            )

            # combine the inner data bounds
            if Xs_lower_ is None:
                Xs_lower_ = xl
            else:
                Xs_lower_ = _maximum_zero_sign_sensitive(Xs_lower_, xl)
            if Xs_upper_ is None:
                Xs_upper_ = xu
            else:
                Xs_upper_ = _minimum_zero_sign_sensitive(Xs_upper_, xu)

        assert Xs_lower_ is not None
        assert Xs_upper_ is not None
        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]] = Xs_lower_
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]] = Xs_upper_

        Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
        Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

        return Xs_lower, Xs_upper

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

    @override
    @checked_data_bounds
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
        a = self._a
        b = self._b

        a_const = not a.has_data
        b_const = not b.has_data
        assert not (a_const and b_const), "constant != has no data bounds"

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)

        # compute a finite midpoint for a and b
        #  - if either is NaN, we won't use this midpoint
        #  - since we use finite values, mid(-Inf, +Inf) = 0
        av_mid_bv: np.ndarray[tuple[Ps], np.dtype[F]] = np.add(  # type: ignore
            np.divide(_nan_to_zero_inf_to_finite(av), Xs.dtype.type(2)),
            np.divide(_nan_to_zero_inf_to_finite(bv), Xs.dtype.type(2)),
        )

        # compute the next value from b towards a that can be part of a's
        #  safe interval
        # if b is const this is the literal next value
        # otherwise it is the midpoint
        bv_nxt_av = _ensure_array(_nextafter(bv, av))
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
        av_nxt_bv = _nextafter(av_nxt_bv, bv)

        xl: np_sndarray[Ps, Ns, np.dtype[F]]
        xu: np_sndarray[Ps, Ns, np.dtype[F]]
        Xs_lower_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None
        Xs_upper_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, (a != b) = True, then
        #  - if av > bv, restrict a's lower and b's upper bound to not overlap
        #  - if av < bv, restrict a's upper and b's lower bound to not overlap
        #  - otherwise av or bv is NaN and any non-NaN argument can be anything
        # if expr_upper < 1, (a != b) = False, then
        #  - keep av and bv the same
        #  - we do *not* special-case -0.0 and +0.0 here
        # otherwise, (a != b) in [True, False] and all args can be anything
        if not a_const:
            a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                a_lower,
                bv_nxt_av,
                where=(np.greater_equal(expr_lower, 0) & (av > bv)),
                casting="no",
            )
            np.copyto(a_lower, av, where=np.less_equal(expr_upper, 1), casting="no")

            a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                a_upper,
                bv_nxt_av,
                where=(np.greater_equal(expr_lower, 0) & (av < bv)),
                casting="no",
            )
            np.copyto(a_upper, av, where=np.less_equal(expr_upper, 1), casting="no")

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            # recurse into arg a
            Xs_lower_, Xs_upper_ = a.compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
            )

        # cont. with b as outlined above
        if not b_const:
            b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            np.copyto(
                b_lower,
                av_nxt_bv,
                where=(np.greater_equal(expr_lower, 0) & (bv > av)),
                casting="no",
            )
            np.copyto(b_lower, bv, where=np.less_equal(expr_upper, 1), casting="no")

            b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            np.copyto(
                b_upper,
                av_nxt_bv,
                where=(np.greater_equal(expr_lower, 0) & (bv < av)),
                casting="no",
            )
            np.copyto(b_upper, bv, where=np.less_equal(expr_upper, 1), casting="no")

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            # recurse into arg b
            xl, xu = b.compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
            )

            # combine the inner data bounds
            if Xs_lower_ is None:
                Xs_lower_ = xl
            else:
                Xs_lower_ = _maximum_zero_sign_sensitive(Xs_lower_, xl)
            if Xs_upper_ is None:
                Xs_upper_ = xu
            else:
                Xs_upper_ = _minimum_zero_sign_sensitive(Xs_upper_, xu)

        assert Xs_lower_ is not None
        assert Xs_upper_ is not None
        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]] = Xs_lower_
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]] = Xs_upper_

        Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
        Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

        return Xs_lower, Xs_upper

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} != {self._b!r}"


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

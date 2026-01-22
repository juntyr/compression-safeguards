from collections.abc import Callable, Mapping
from typing import overload

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _stack,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarNot(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarNot":
        return ScalarNot(a)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: combine_to_dtype(np.logical_not, x, dtype),
            ScalarNot,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return combine_to_dtype(np.logical_not, self._a.eval(Xs, late_bound), Xs.dtype)

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
        # evaluate arg
        arg = self._a
        argv = arg.eval(Xs, late_bound)

        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, not(arg) = True, so arg must stay False, i.e zero
        # if expr_upper < 1, not(arg) = False, arg must stay True, i.e. non-zero
        #  only one of the two disjoint non-zero intervals is propagated
        # otherwise, not(arg) in [True, False] and arg can be anything
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
            Xs.shape[:1], Xs.dtype.type(-np.inf)
        )
        arg_lower[np.greater(expr_lower, 0)] = -0.0
        arg_lower[np.less(expr_upper, 1) & np.greater(argv, 0)] = smallest_subnormal

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
            Xs.shape[:1], Xs.dtype.type(np.inf)
        )
        arg_upper[np.greater(expr_lower, 0)] = +0.0
        arg_upper[np.less(expr_upper, 1) & np.less(argv, 0)] = -smallest_subnormal

        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"not({self._a!r})"


class ScalarAll(Expr[AnyExpr, AnyExpr, *tuple[AnyExpr, ...]]):
    __slots__: tuple[str, ...] = ("_a", "_b", "_cs")
    _a: AnyExpr
    _b: AnyExpr
    _cs: tuple[AnyExpr, ...]

    def __init__(self, a: AnyExpr, b: AnyExpr, *cs: AnyExpr):
        self._a = a
        self._b = b
        self._cs = cs

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr, *tuple[AnyExpr, ...]]:
        return (self._a, self._b, *self._cs)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr, *cs: AnyExpr) -> "ScalarAll":
        return ScalarAll(a, b, *cs)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        # we first individually fold each term
        fs = [abc.constant_fold(dtype) for abc in (self._a, self._b, *self._cs)]

        # any False -> False
        if any(f == 0 for f in fs):
            return dtype.type(0)

        # only keep non-const terms, which filters out all True-thy terms
        es = [f for f in fs if isinstance(f, Expr)]

        match es:
            case []:
                # all True-thy -> True
                return dtype.type(1)
            case [e]:
                # single non-const -> propagate
                return e
            case es:
                # all combinator over non-const terms
                a, b, *cs = es
                return ScalarAll(a, b, *cs)

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return reduce_combine_to_dtype(
            lambda a: np.all(a, axis=0),
            _stack(
                [self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound)]
                + [c.eval(Xs, late_bound) for c in self._cs]
            ),
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
        cs = self._cs

        a_const = not a.has_data
        b_const = not b.has_data
        cs_const = [not c.has_data for c in cs]
        assert not (a_const and b_const and all(cs_const)), (
            "constant all has no data bounds"
        )

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)
        cvs = [c.eval(Xs, late_bound) for c in cs]

        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        a_const_zero = (av == 0) & (
            True if a_const else ~a.eval_has_data(Xs, late_bound)
        )
        b_const_zero = (bv == 0) & (
            True if b_const else ~b.eval_has_data(Xs, late_bound)
        )
        cs_const_zero = [
            (cv == 0) & (True if c_const else ~c.eval_has_data(Xs, late_bound))
            for c_const, c, cv in zip(cs_const, cs, cvs)
        ]

        any_constant_zero = _ensure_array(a_const_zero, copy=True)
        any_constant_zero |= b_const_zero
        for c_const_zero in cs_const_zero:
            any_constant_zero |= c_const_zero

        xl: np_sndarray[Ps, Ns, np.dtype[F]]
        xu: np_sndarray[Ps, Ns, np.dtype[F]]
        Xs_lower_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None
        Xs_upper_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, all(*args) = True, then
        #  all args must stay True, i.e. non-zero
        # if expr_upper < 1, all(*args) = False, then
        #  - if any constant term is zero, all non-constant args can be anything
        #  - otherwise
        #    - all False (zero) args must stay False, i.e. zero
        #    - all True (non-zero) args can be anything
        # only one of the two disjoint non-zero intervals is propagated
        # otherwise, all(*args) in [True, False] and all args can be anything
        for term, tv, t_const_zero, t_const in zip(
            (a, b, *cs),
            (av, bv, *cvs),
            (a_const_zero, b_const_zero, *cs_const_zero),
            (a_const, b_const, *cs_const),
        ):
            if t_const:
                continue

            term_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            term_lower[np.greater(expr_lower, 0) & np.greater(tv, 0)] = (
                smallest_subnormal
            )
            term_lower[
                np.less(expr_upper, 1) & (tv == 0) & (~any_constant_zero | t_const_zero)
            ] = -0.0

            term_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            term_upper[np.greater(expr_lower, 0) & np.less(tv, 0)] = -smallest_subnormal
            term_upper[
                np.less(expr_upper, 1) & (tv == 0) & (~any_constant_zero | t_const_zero)
            ] = +0.0

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            # recurse into the terms
            xl, xu = term.compute_data_bounds(
                term_lower,
                term_upper,
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
        abc = ", ".join([repr(self._a), repr(self._b)] + [repr(c) for c in self._cs])
        return f"all({abc})"


class ScalarAny(Expr[AnyExpr, AnyExpr, *tuple[AnyExpr, ...]]):
    __slots__: tuple[str, ...] = ("_a", "_b", "_cs")
    _a: AnyExpr
    _b: AnyExpr
    _cs: tuple[AnyExpr, ...]

    def __init__(self, a: AnyExpr, b: AnyExpr, *cs: AnyExpr):
        self._a = a
        self._b = b
        self._cs = cs

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr, *tuple[AnyExpr, ...]]:
        return (self._a, self._b, *self._cs)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr, *cs: AnyExpr) -> "ScalarAny":
        return ScalarAny(a, b, *cs)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        # we first individually fold each term
        fs = [abc.constant_fold(dtype) for abc in (self._a, self._b, *self._cs)]

        # any True -> True
        if any((not isinstance(f, Expr) and (f != 0)) for f in fs):
            return dtype.type(1)

        # only keep non-const terms, which filters out all False terms
        es = [f for f in fs if isinstance(f, Expr)]

        match es:
            case []:
                # all False -> False
                return dtype.type(0)
            case [e]:
                # single non-const -> propagate
                return e
            case es:
                # all combinator over non-const terms
                a, b, *cs = es
                return ScalarAny(a, b, *cs)

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return reduce_combine_to_dtype(
            lambda a: np.any(a, axis=0),
            _stack(
                [self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound)]
                + [c.eval(Xs, late_bound) for c in self._cs]
            ),
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
        cs = self._cs

        a_const = not a.has_data
        b_const = not b.has_data
        cs_const = [not c.has_data for c in cs]
        assert not (a_const and b_const and all(cs_const)), (
            "constant any has no data bounds"
        )

        # evaluate args
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)
        cvs = [c.eval(Xs, late_bound) for c in cs]

        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        a_const_non_zero = (av != 0) & (
            True if a_const else ~a.eval_has_data(Xs, late_bound)
        )
        b_const_non_zero = (bv != 0) & (
            True if b_const else ~b.eval_has_data(Xs, late_bound)
        )
        cs_const_non_zero = [
            (cv != 0) & (True if c_const else ~c.eval_has_data(Xs, late_bound))
            for c_const, c, cv in zip(cs_const, cs, cvs)
        ]

        any_constant_non_zero = _ensure_array(a_const_non_zero, copy=True)
        any_constant_non_zero |= b_const_non_zero
        for c_const_zero in cs_const_non_zero:
            any_constant_non_zero |= c_const_zero

        xl: np_sndarray[Ps, Ns, np.dtype[F]]
        xu: np_sndarray[Ps, Ns, np.dtype[F]]
        Xs_lower_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None
        Xs_upper_: None | np_sndarray[Ps, Ns, np.dtype[F]] = None

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, any(*args) = True, then
        #  - if any constant term is non-zero, all non-constant args can be
        #    anything
        #  - otherwise
        #    - all False (zero) args can be anything
        #    - all True (non-zero) args must stay True, i.e. non-zero
        # only one of the two disjoint non-zero intervals is propagated
        # if expr_upper < 1, any(*args) = False, then
        #  all args must stay False, i.e. zero
        # otherwise, any(*args) in [True, False] and all args can be anything
        for term, tv, t_const_non_zero, t_const in zip(
            (a, b, *cs),
            (av, bv, *cvs),
            (a_const_non_zero, b_const_non_zero, *cs_const_non_zero),
            (a_const, b_const, *cs_const),
        ):
            if t_const:
                continue

            term_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(-np.inf)
            )
            term_lower[
                np.greater(expr_lower, 0)
                & (tv != 0)
                & (~any_constant_non_zero | t_const_non_zero)
                & np.greater(tv, 0)
            ] = smallest_subnormal
            term_lower[np.less(expr_upper, 1)] = -0.0

            term_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
                Xs.shape[:1], Xs.dtype.type(np.inf)
            )
            term_upper[
                np.greater(expr_lower, 0)
                & (tv != 0)
                & (~any_constant_non_zero | t_const_non_zero)
                & np.less(tv, 0)
            ] = -smallest_subnormal
            term_upper[np.less(expr_upper, 1)] = +0.0

            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future

            # recurse into the terms
            xl, xu = term.compute_data_bounds(
                term_lower,
                term_upper,
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
        abc = ", ".join([repr(self._a), repr(self._b)] + [repr(c) for c in self._cs])
        return f"any({abc})"


@overload
def combine_to_dtype(
    combine: Callable[
        [np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[np.bool]]
    ],
    a: np.ndarray[tuple[Ps], np.dtype[F]],
    dtype: np.dtype[F],
) -> np.ndarray[tuple[Ps], np.dtype[F]]: ...


@overload
def combine_to_dtype(
    combine: Callable[[Fi], bool],
    a: Fi,
    dtype: np.dtype[Fi],
) -> Fi: ...


def combine_to_dtype(combine, a, dtype):
    c = combine(a)

    if not isinstance(c, np.ndarray):
        return _ensure_array(c).astype(dtype, casting="safe")[()]

    return c.astype(dtype, casting="safe")


def reduce_combine_to_dtype(
    reduce_combine: Callable[
        [np.ndarray[tuple[int, Ps], np.dtype[F]]],
        np.ndarray[tuple[Ps], np.dtype[np.bool]],
    ],
    a: np.ndarray[tuple[int, Ps], np.dtype[F]],
    dtype: np.dtype[F],
):
    return reduce_combine(a).astype(dtype, casting="safe")

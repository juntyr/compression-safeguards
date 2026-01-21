from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _is_sign_negative_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarSqrt(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarSqrt":
        return ScalarSqrt(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.sqrt, ScalarSqrt
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.sqrt(self._a.eval(Xs, late_bound))

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
        # for sqrt(-0.0), we should return -0.0 as the inverse
        # this ensures that 1/sqrt(-0.0) doesn't become 1/sqrt(0.0)
        def _sqrt_inv(
            x: np.ndarray[tuple[Ps], np.dtype[F]],
        ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
            out: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
                np.square(_maximum_zero_sign_sensitive(Xs.dtype.type(0), x))
            )
            np.copyto(out, x, where=(x == 0), casting="no")
            return out

        # evaluate arg and sqrt(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.sqrt(argv)

        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # apply the inverse function to get the bounds on arg
        # sqrt(-0.0) = -0.0 and sqrt(+0.0) = +0.0
        # sqrt(...) is NaN for negative values and can then take any negative
        #  value
        # otherwise ensure that the bounds on sqrt(...) are non-negative
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _minimum_zero_sign_sensitive(argv, _sqrt_inv(expr_lower))
        )
        arg_lower[np.less(argv, 0)] = -np.inf

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _maximum_zero_sign_sensitive(argv, _sqrt_inv(expr_upper))
        )
        arg_upper[np.less(argv, 0)] = -smallest_subnormal

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in sqrt(square(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.sqrt(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.sqrt(arg_upper),
            exprv,
            argv,
            arg_upper,
            expr_lower,
            expr_upper,
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"sqrt({self._a!r})"


class ScalarSquare(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarSquare":
        return ScalarSquare(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.square, ScalarSquare
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.square(self._a.eval(Xs, late_bound))

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
        # evaluate arg and square(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.square(argv)

        # apply the inverse function to get the bounds on arg
        al = np.sqrt(_maximum_zero_sign_sensitive(expr_lower, Xs.dtype.type(0)))
        au = np.sqrt(expr_upper)

        # flip and swap the expr bounds to get the bounds on arg
        # square(...) cannot be negative, but
        #  - a > 0 and 0 < el <= eu -> al = el, au = eu
        #  - a < 0 and 0 < el <= eu -> al = -eu, au = -el
        #  - el <= 0 -> al = -eu, au = eu
        # TODO: an interval union could represent that the two sometimes-
        #       disjoint intervals in the future
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(al, copy=True)
        np.negative(
            au,
            out=arg_lower,
            where=(np.less_equal(expr_lower, 0) | _is_sign_negative_number(argv)),
        )
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(au, copy=True)
        np.negative(
            al,
            out=arg_upper,
            where=(np.greater(expr_lower, 0) & _is_sign_negative_number(argv)),
        )
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in square(sqrt(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.square(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.square(arg_upper),
            exprv,
            argv,
            arg_upper,
            expr_lower,
            expr_upper,
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"square({self._a!r})"

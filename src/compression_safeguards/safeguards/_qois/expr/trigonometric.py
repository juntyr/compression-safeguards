from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _floating_pi,
    _is_negative_zero,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarSin(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarSin":
        return ScalarSin(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.sin, ScalarSin
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.sin(self._a.eval(Xs, late_bound))

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
        # evaluate arg and sin(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.sin(argv)

        # apply the inverse function to get the bounds on arg
        # ensure that the bounds on sin(...) are in [-1, +1]
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.asin(
            _maximum_zero_sign_sensitive(Xs.dtype.type(-1), expr_lower)
        )
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.asin(
            _minimum_zero_sign_sensitive(expr_upper, Xs.dtype.type(1))
        )

        # sin(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        argv_asin = np.asin(exprv)
        arg_lower_diff = arg_lower - argv_asin
        arg_upper_diff = arg_upper - argv_asin

        # np.asin maps to [-pi/2, +pi/2] where sin is monotonically increasing
        # flip the argument difference bounds where sin is monotonically
        #  decreasing
        # we check monotonicity using the derivative sin'(x) = cos(x)
        needs_flip = np.cos(argv) < 0

        # check for the case where any finite value would work
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, -1
        ) & np.greater_equal(expr_upper, 1)

        fmax = np.finfo(Xs.dtype).max

        # sin(+-inf) = NaN, so force infinite argv to have exact bounds
        # sin(finite) in [-1, +1] so allow any finite argv if the all of
        #  [-1, +1] is allowed
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        # if expr_upper == -0.0 and arg_upper == 0, we need to guarantee that
        #  arg_upper is also -0.0
        # FIXME: how do we handle bounds right next to the peak where the
        #        expression bounds could be exceeded inside the interval?
        # TODO: the intervals can sometimes be extended if expr_lower <= -1 or
        #       expr_upper >= 1
        # TODO: since sin is periodic, an interval union could be used in the
        #       future
        arg_lower = _ensure_array(arg_lower_diff, copy=True)
        np.negative(arg_upper_diff, out=arg_lower, where=needs_flip)
        np.add(arg_lower, argv, out=arg_lower)
        arg_lower[full_domain] = -fmax
        np.copyto(arg_lower, argv, where=np.isinf(argv), casting="no")
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper = _ensure_array(arg_upper_diff, copy=True)
        np.negative(arg_lower_diff, out=arg_upper, where=needs_flip)
        np.add(arg_upper, argv, out=arg_upper)
        arg_upper[full_domain] = fmax
        np.copyto(arg_upper, argv, where=np.isinf(argv), casting="no")
        arg_upper[(arg_upper == 0) & _is_negative_zero(expr_upper)] = -0.0
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in sin(asin(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.sin(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.sin(arg_upper),
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
        return f"sin({self._a!r})"


class ScalarCos(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarCos":
        return ScalarCos(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.cos, ScalarCos
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.cos(self._a.eval(Xs, late_bound))

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
        # evaluate arg and cos(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.cos(argv)

        # apply the inverse function to get the bounds on arg
        # ensure that the bounds on cos(...) are in [-1, +1]
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.acos(
            _maximum_zero_sign_sensitive(Xs.dtype.type(-1), expr_lower)
        )
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.acos(
            _minimum_zero_sign_sensitive(expr_upper, Xs.dtype.type(1))
        )

        # cos(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        argv_acos = np.acos(exprv)
        arg_lower_diff = argv_acos - arg_lower
        arg_upper_diff = argv_acos - arg_upper

        # np.acos maps to [pi, 0] where cos is monotonically decreasing
        # flip the argument difference bounds where cos is monotonically
        #  decreasing
        # we check monotonicity using the derivative cos'(x) = -sin(x)
        needs_flip = np.sin(argv) >= 0

        # check for the case where any finite value would work
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = np.less_equal(
            expr_lower, -1
        ) & np.greater_equal(expr_upper, 1)

        fmax = np.finfo(Xs.dtype).max

        # cps(+-inf) = NaN, so force infinite argv to have exact bounds
        # cos(finite) in [-1, +1] so allow any finite argv if the all of
        #  [-1, +1] is allowed
        # FIXME: how do we handle bounds right next to the peak where the
        #        expression bounds could be exceeded inside the interval?
        # TODO: the intervals can sometimes be extended if expr_lower <= -1 or
        #       expr_upper >= 1
        # TODO: since cos is periodic, an interval union could be used in the
        #       future
        arg_lower = _ensure_array(arg_lower_diff, copy=True)
        np.negative(arg_upper_diff, out=arg_lower, where=needs_flip)
        np.add(arg_lower, argv, out=arg_lower)
        arg_lower[full_domain] = -fmax
        np.copyto(arg_lower, argv, where=np.isinf(argv), casting="no")
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper = _ensure_array(arg_upper_diff, copy=True)
        np.negative(arg_lower_diff, out=arg_upper, where=needs_flip)
        np.add(arg_upper, argv, out=arg_upper)
        arg_upper[full_domain] = fmax
        np.copyto(arg_upper, argv, where=np.isinf(argv), casting="no")
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in cos(acos(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.cos(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.cos(arg_upper),
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
        return f"cos({self._a!r})"


class ScalarTan(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarTan":
        return ScalarTan(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.tan, ScalarTan
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.tan(self._a.eval(Xs, late_bound))

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
        # evaluate arg and tan(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.tan(argv)

        # apply the inverse function to get the bounds on arg
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.atan(expr_lower)
        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.atan(expr_upper)

        # tan(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        argv_atan = np.atan(exprv)
        arg_lower_diff = arg_lower - argv_atan
        arg_upper_diff = arg_upper - argv_atan

        # check for the case where any finite value would work
        full_domain: np.ndarray[tuple[Ps], np.dtype[np.bool]] = (
            expr_lower == Xs.dtype.type(-np.inf)
        ) & (expr_upper == Xs.dtype.type(np.inf))

        fmax = np.finfo(Xs.dtype).max

        # tan(+-inf) = NaN, so force infinite argv to have exact bounds
        # tan(finite) in [-inf, +inf] so allow any finite argv if the all of
        #  [-inf, +inf] is allowed
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        # if expr_upper == -0.0 and arg_upper == 0, we need to guarantee that
        #  arg_upper is also -0.0
        # FIXME: how do we handle bounds right next to the peak where the
        #        expression bounds could be exceeded inside the interval?
        # TODO: since tan is periodic, an interval union could be used in the
        #       future
        arg_lower = _ensure_array(np.add(argv, arg_lower_diff))
        arg_lower[full_domain] = -fmax
        np.copyto(arg_lower, argv, where=np.isinf(argv), casting="no")
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper = _ensure_array(np.add(argv, arg_upper_diff))
        arg_upper[full_domain] = fmax
        np.copyto(arg_upper, argv, where=np.isinf(argv), casting="no")
        arg_upper[(arg_upper == 0) & _is_negative_zero(expr_upper)] = -0.0
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in tan(atan(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.tan(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.tan(arg_upper),
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
        return f"tan({self._a!r})"


class ScalarAsin(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarAsin":
        return ScalarAsin(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.asin, ScalarAsin
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.asin(self._a.eval(Xs, late_bound))

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
        # evaluate arg and asin(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.asin(argv)

        pi = _floating_pi(Xs.dtype)
        pi_2: F = np.divide(pi, 2)
        one_eps = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(2))

        # apply the inverse function to get the bounds on arg
        # asin(...) is NaN when abs(...) > 1 and can then take any value > 1
        # otherwise ensure that the bounds on asin(...) are in [-pi/2, +pi/2]
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.sin(_maximum_zero_sign_sensitive(-pi_2, expr_lower))
        )
        arg_lower[np.greater(argv, 1)] = one_eps
        arg_lower[np.less(argv, -1)] = -np.inf
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.sin(_minimum_zero_sign_sensitive(expr_upper, pi_2))
        )
        arg_upper[np.greater(argv, 1)] = np.inf
        arg_upper[np.less(argv, -1)] = -one_eps
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper and abs(argv) < 1
        np.copyto(
            arg_lower,
            argv,
            where=((expr_lower == expr_upper) & np.less(np.abs(argv), 1)),
            casting="no",
        )
        np.copyto(
            arg_upper,
            argv,
            where=((expr_lower == expr_upper) & np.less(np.abs(argv), 1)),
            casting="no",
        )

        # handle rounding errors in asin(sin(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.asin(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.asin(arg_upper),
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
        return f"asin({self._a!r})"


class ScalarAcos(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarAcos":
        return ScalarAcos(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.acos, ScalarAcos
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.acos(self._a.eval(Xs, late_bound))

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
        # evaluate arg and acos(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.acos(argv)

        pi = _floating_pi(Xs.dtype)
        one_eps = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(2))

        # apply the inverse function to get the bounds on arg
        # acos(...) is NaN when abs(...) > 1 and can then take any value > 1
        # otherwise ensure that the bounds on acos(...) are in [0, pi]
        # since cos is monotonically decreasing in [0, pi], the lower and upper
        #  bounds get switched
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.cos(_minimum_zero_sign_sensitive(expr_upper, pi))
        )
        arg_lower[np.greater(argv, 1)] = one_eps
        arg_lower[np.less(argv, -1)] = -np.inf
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.cos(_maximum_zero_sign_sensitive(Xs.dtype.type(0), expr_lower))
        )
        arg_upper[np.greater(argv, 1)] = np.inf
        arg_upper[np.less(argv, -1)] = -one_eps
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper and abs(argv) < 1
        np.copyto(
            arg_lower,
            argv,
            where=((expr_lower == expr_upper) & np.less(np.abs(argv), 1)),
            casting="no",
        )
        np.copyto(
            arg_upper,
            argv,
            where=((expr_lower == expr_upper) & np.less(np.abs(argv), 1)),
            casting="no",
        )

        # handle rounding errors in acos(cos(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.acos(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.acos(arg_upper),
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
        return f"acos({self._a!r})"


class ScalarAtan(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr):
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarAtan":
        return ScalarAtan(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.atan, ScalarAtan
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.atan(self._a.eval(Xs, late_bound))

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
        # evaluate arg and atan(arg)
        arg = self._a
        argv = arg.eval(Xs, late_bound)
        exprv = np.atan(argv)

        # compute pi/2 but guard against rounding error to ensure that
        #  tan(pi/2) >> 0
        atan_max: F = np.atan(Xs.dtype.type(np.inf))
        if np.tan(atan_max) < 0:
            atan_max = Xs.dtype.type(np.nextafter(atan_max, Xs.dtype.type(0)))
        assert np.tan(atan_max) > 0

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        # ensure that the bounds on atan(...) are in [-pi/2, +pi/2]
        # since tan is discontinuous at +-pi/2, we need to be extra careful
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.tan(expr_lower)
        )
        arg_lower[expr_lower < -atan_max] = -np.inf
        arg_lower[expr_lower > atan_max] = np.inf
        arg_lower = _ensure_array(_minimum_zero_sign_sensitive(argv, arg_lower))

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.tan(expr_upper)
        )
        arg_upper[expr_upper < -atan_max] = -np.inf
        arg_upper[expr_upper > atan_max] = np.inf
        arg_upper = _ensure_array(_maximum_zero_sign_sensitive(argv, arg_upper))

        # we need to force argv if expr_lower == expr_upper
        np.copyto(arg_lower, argv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(arg_upper, argv, where=(expr_lower == expr_upper), casting="no")

        # handle rounding errors in atan(tan(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.atan(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.atan(arg_upper),
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
        return f"atan({self._a!r})"

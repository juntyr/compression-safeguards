from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _floating_max,
    _is_negative_zero,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _nextafter,
    _pi,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarSin(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarSin":
        return ScalarSin(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.sin, ScalarSin
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.sin(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and sin(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.sin(argv)

        # apply the inverse function to get the bounds on arg
        # ensure that the bounds on sin(...) are in [-1, +1]
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.asin(
            _maximum_zero_sign_sensitive(X.dtype.type(-1), expr_lower)
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.asin(
            _minimum_zero_sign_sensitive(expr_upper, X.dtype.type(1))
        )

        # sin(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        argv_asin = np.asin(exprv)
        arg_lower_diff = arg_lower - argv_asin
        arg_upper_diff = arg_upper - argv_asin

        # np.asin maps to [-pi/2, +pi/2] where sin is monotonically increasing
        # flip the argument error bounds where sin is monotonically decreasing
        # we check monotonicity using the derivative sin'(x) = cos(x)
        needs_flip = np.cos(argv) < 0

        # check for the case where any finite value would work
        full_domain: np.ndarray[Ps, np.dtype[np.bool]] = np.less_equal(
            expr_lower, -1
        ) & np.greater_equal(expr_upper, 1)

        fmax = _floating_max(X.dtype)

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
        arg_lower = np.array(arg_lower_diff, copy=True)
        np.negative(arg_upper_diff, out=arg_lower, where=needs_flip)
        np.add(arg_lower, argv, out=arg_lower)
        arg_lower[full_domain] = -fmax
        np.copyto(arg_lower, argv, where=np.isinf(argv), casting="no")
        arg_lower = _minimum_zero_sign_sensitive(argv, arg_lower)

        arg_upper = np.array(arg_upper_diff, copy=True)
        np.negative(arg_lower_diff, out=arg_upper, where=needs_flip)
        np.add(arg_upper, argv, out=arg_upper)
        arg_upper[full_domain] = fmax
        np.copyto(arg_upper, argv, where=np.isinf(argv), casting="no")
        arg_upper[(arg_upper == 0) & _is_negative_zero(expr_upper)] = -0.0
        arg_upper = _maximum_zero_sign_sensitive(argv, arg_upper)

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
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"sin({self._a!r})"


class ScalarCos(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarCos":
        return ScalarCos(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.cos, ScalarCos
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.cos(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and cos(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.cos(argv)

        # apply the inverse function to get the bounds on arg
        # ensure that the bounds on cos(...) are in [-1, +1]
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.acos(
            _maximum_zero_sign_sensitive(X.dtype.type(-1), expr_lower)
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.acos(
            _minimum_zero_sign_sensitive(expr_upper, X.dtype.type(1))
        )

        # cos(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        argv_acos = np.acos(exprv)
        arg_lower_diff = argv_acos - arg_lower
        arg_upper_diff = argv_acos - arg_upper

        # np.acos maps to [pi, 0] where cos is monotonically decreasing
        # flip the argument error bounds where cos is monotonically decreasing
        # we check monotonicity using the derivative cos'(x) = -sin(x)
        needs_flip = np.sin(argv) >= 0

        # check for the case where any finite value would work
        full_domain: np.ndarray[Ps, np.dtype[np.bool]] = np.less_equal(
            expr_lower, -1
        ) & np.greater_equal(expr_upper, 1)

        fmax = _floating_max(X.dtype)

        # cps(+-inf) = NaN, so force infinite argv to have exact bounds
        # cos(finite) in [-1, +1] so allow any finite argv if the all of
        #  [-1, +1] is allowed
        # FIXME: how do we handle bounds right next to the peak where the
        #        expression bounds could be exceeded inside the interval?
        # TODO: the intervals can sometimes be extended if expr_lower <= -1 or
        #       expr_upper >= 1
        # TODO: since cos is periodic, an interval union could be used in the
        #       future
        arg_lower = np.array(arg_lower_diff, copy=True)
        np.negative(arg_upper_diff, out=arg_lower, where=needs_flip)
        np.add(arg_lower, argv, out=arg_lower)
        arg_lower[full_domain] = -fmax
        np.copyto(arg_lower, argv, where=np.isinf(argv), casting="no")
        arg_lower = _minimum_zero_sign_sensitive(argv, arg_lower)

        arg_upper = np.array(arg_upper_diff, copy=True)
        np.negative(arg_lower_diff, out=arg_upper, where=needs_flip)
        np.add(arg_upper, argv, out=arg_upper)
        arg_upper[full_domain] = fmax
        np.copyto(arg_upper, argv, where=np.isinf(argv), casting="no")
        arg_upper = _maximum_zero_sign_sensitive(argv, arg_upper)

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
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"cos({self._a!r})"


class ScalarTan(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarTan":
        return ScalarTan(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.tan, ScalarTan
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.tan(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and tan(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.tan(argv)

        # apply the inverse function to get the bounds on arg
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.atan(expr_lower)
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.atan(expr_upper)

        # tan(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        argv_atan = np.atan(exprv)
        arg_lower_diff = arg_lower - argv_atan
        arg_upper_diff = arg_upper - argv_atan

        # check for the case where any finite value would work
        full_domain: np.ndarray[Ps, np.dtype[np.bool]] = (
            expr_lower == X.dtype.type(-np.inf)
        ) & (expr_upper == X.dtype.type(np.inf))

        fmax = _floating_max(X.dtype)

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
        arg_lower = np.array(np.add(argv, arg_lower_diff), copy=None)
        arg_lower[full_domain] = -fmax
        np.copyto(arg_lower, argv, where=np.isinf(argv), casting="no")
        arg_lower = _minimum_zero_sign_sensitive(argv, arg_lower)

        arg_upper = np.array(np.add(argv, arg_upper_diff), copy=None)
        arg_upper[full_domain] = fmax
        np.copyto(arg_upper, argv, where=np.isinf(argv), casting="no")
        arg_upper[(arg_upper == 0) & _is_negative_zero(expr_upper)] = -0.0
        arg_upper = _maximum_zero_sign_sensitive(argv, arg_upper)

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
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"tan({self._a!r})"


class ScalarAsin(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarAsin":
        return ScalarAsin(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.asin, ScalarAsin
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.asin(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and asin(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.asin(argv)

        pi = _pi(X.dtype)
        one_eps = _nextafter(np.array(1, dtype=X.dtype), np.array(2, dtype=X.dtype))

        # apply the inverse function to get the bounds on arg
        # asin(...) is NaN when abs(...) > 1 and can then take any value > 1
        # otherwise ensure that the bounds on asin(...) are in [-pi/2, +pi/2]
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.sin(_maximum_zero_sign_sensitive(np.divide(-pi, 2), expr_lower)),
            copy=None,
        )
        arg_lower[np.greater(argv, 1)] = one_eps
        arg_lower[np.less(argv, -1)] = -np.inf
        arg_lower = _minimum_zero_sign_sensitive(argv, arg_lower)

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.sin(_minimum_zero_sign_sensitive(expr_upper, np.divide(pi, 2))),
            copy=None,
        )
        arg_upper[np.greater(argv, 1)] = np.inf
        arg_upper[np.less(argv, -1)] = -one_eps
        arg_upper = _maximum_zero_sign_sensitive(argv, arg_upper)

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
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"asin({self._a!r})"


class ScalarAcos(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarAcos":
        return ScalarAcos(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.acos, ScalarAcos
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.acos(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and acos(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.acos(argv)

        pi = _pi(X.dtype)
        one_eps = _nextafter(np.array(1, dtype=X.dtype), np.array(2, dtype=X.dtype))

        # apply the inverse function to get the bounds on arg
        # acos(...) is NaN when abs(...) > 1 and can then take any value > 1
        # otherwise ensure that the bounds on acos(...) are in [0, pi]
        # since cos is monotonically decreasing in [0, pi], the lower and upper
        #  bounds get switched
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.cos(_minimum_zero_sign_sensitive(expr_upper, pi)), copy=None
        )
        arg_lower[np.greater(argv, 1)] = one_eps
        arg_lower[np.less(argv, -1)] = -np.inf
        arg_lower = _minimum_zero_sign_sensitive(argv, arg_lower)

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.cos(_maximum_zero_sign_sensitive(X.dtype.type(0), expr_lower)), copy=None
        )
        arg_upper[np.greater(argv, 1)] = np.inf
        arg_upper[np.less(argv, -1)] = -one_eps
        arg_upper = _maximum_zero_sign_sensitive(argv, arg_upper)

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
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"acos({self._a!r})"


class ScalarAtan(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarAtan":
        return ScalarAtan(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.atan, ScalarAtan
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.atan(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and atan(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.atan(argv)

        # compute pi/2 but guard against rounding error to ensure that
        #  tan(pi/2) >> 0
        atan_max: F = np.atan(X.dtype.type(np.inf))
        if np.tan(atan_max) < 0:
            atan_max = X.dtype.type(
                _nextafter(np.array(atan_max), np.array(X.dtype.type(0)))
            )
        assert np.tan(atan_max) > 0

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        # ensure that the bounds on atan(...) are in [-pi/2, +pi/2]
        # since tan is discontinuous at +-pi/2, we need to be extra careful
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.tan(
            _maximum_zero_sign_sensitive(-atan_max, expr_lower)
        )
        arg_lower = np.array(
            _minimum_zero_sign_sensitive(argv, arg_lower),
            copy=None,
        )

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.tan(
            _minimum_zero_sign_sensitive(atan_max, expr_upper)
        )
        arg_upper = np.array(
            _maximum_zero_sign_sensitive(argv, arg_upper),
            copy=None,
        )

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
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"atan({self._a!r})"

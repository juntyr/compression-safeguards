from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _floating_max,
    _isinf,
    _maximum,
    _minimum,
    _nextafter,
    _pi,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds, guaranteed_data_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarSin(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarSin(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    @guaranteed_data_bounds
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
            _maximum(X.dtype.type(-1), expr_lower)
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.asin(
            _minimum(expr_upper, X.dtype.type(1))
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
        # FIXME: how do we handle bounds right next to the peak where the
        #        expression bounds could be exceeded inside the interval?
        # TODO: the intervals can sometimes be extended if expr_lower <= -1 or
        #       expr_upper >= 1
        # TODO: since sin is periodic, an interval union could be used in the
        #       future
        arg_lower = _where(
            _isinf(argv),
            argv,
            _where(
                full_domain,
                -fmax,
                _minimum(
                    argv,
                    argv
                    + _where(
                        needs_flip,
                        -arg_upper_diff,
                        arg_lower_diff,
                    ),
                ),
            ),
        )
        arg_upper = _where(
            _isinf(argv),
            argv,
            _where(
                full_domain,
                fmax,
                _maximum(
                    argv,
                    argv
                    + _where(
                        needs_flip,
                        -arg_lower_diff,
                        arg_upper_diff,
                    ),
                ),
            ),
        )

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

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


class ScalarCos(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarCos(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    @guaranteed_data_bounds
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
            _maximum(X.dtype.type(-1), expr_lower)
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.acos(
            _minimum(expr_upper, X.dtype.type(1))
        )

        # cos(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        argv_acos = np.acos(exprv)
        arg_lower_diff = arg_lower - argv_acos
        arg_upper_diff = arg_upper - argv_acos

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
        arg_lower = _where(
            _isinf(argv),
            argv,
            _where(
                full_domain,
                -fmax,
                _minimum(
                    argv,
                    argv
                    + _where(
                        needs_flip,
                        -arg_upper_diff,
                        arg_lower_diff,
                    ),
                ),
            ),
        )
        arg_upper = _where(
            _isinf(argv),
            argv,
            _where(
                full_domain,
                fmax,
                _maximum(
                    argv,
                    argv
                    + _where(
                        needs_flip,
                        -arg_lower_diff,
                        arg_upper_diff,
                    ),
                ),
            ),
        )

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

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


class ScalarTan(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarTan(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    @guaranteed_data_bounds
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
        # FIXME: how do we handle bounds right next to the peak where the
        #        expression bounds could be exceeded inside the interval?
        # TODO: since tan is periodic, an interval union could be used in the
        #       future
        arg_lower = _where(
            _isinf(argv),
            argv,
            _where(full_domain, -fmax, _minimum(argv, argv + arg_lower_diff)),
        )
        arg_upper = _where(
            _isinf(argv),
            argv,
            _where(
                full_domain,
                fmax,
                _maximum(argv, argv + arg_upper_diff),
            ),
        )

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

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


class ScalarAsin(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarAsin(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    @guaranteed_data_bounds
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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, -1),
            X.dtype.type(-np.inf),
            _where(
                np.greater(argv, 1),
                one_eps,
                _minimum(argv, np.sin(_maximum(np.divide(-pi, 2), expr_lower))),  # type: ignore
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, -1),
            -one_eps,
            _where(
                np.greater(argv, 1),
                X.dtype.type(np.inf),
                _maximum(argv, np.sin(_minimum(expr_upper, np.divide(pi, 2)))),  # type: ignore
            ),
        )

        # we need to force argv if expr_lower == expr_upper and abs(argv) < 1
        arg_lower = _where(
            (expr_lower == expr_upper) & np.less(np.abs(argv), 1), argv, arg_lower
        )
        arg_upper = _where(
            (expr_lower == expr_upper) & np.less(np.abs(argv), 1), argv, arg_upper
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


class ScalarAcos(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarAcos(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    @guaranteed_data_bounds
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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, -1),
            X.dtype.type(-np.inf),
            _where(
                np.greater(argv, 1),
                one_eps,
                _minimum(argv, np.cos(_minimum(expr_upper, pi))),  # type: ignore
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            np.less(argv, -1),
            -one_eps,
            _where(
                np.greater(argv, 1),
                X.dtype.type(np.inf),
                _maximum(argv, np.cos(_maximum(X.dtype.type(0), expr_lower))),  # type: ignore
            ),
        )

        # we need to force argv if expr_lower == expr_upper and abs(argv) < 1
        arg_lower = _where(
            (expr_lower == expr_upper) & np.less(np.abs(argv), 1), argv, arg_lower
        )
        arg_upper = _where(
            (expr_lower == expr_upper) & np.less(np.abs(argv), 1), argv, arg_upper
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


class ScalarAtan(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarAtan(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    @guaranteed_data_bounds
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

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(argv, np.tan(expr_lower))
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(argv, np.tan(expr_upper))

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

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

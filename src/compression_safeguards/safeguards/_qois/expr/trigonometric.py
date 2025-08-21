from collections.abc import Mapping
from enum import Enum, auto
from typing import Callable

import numpy as np

from ....utils._compat import _floating_max, _isinf, _nextafter, _pi, _reciprocal
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .addsub import ScalarAdd, ScalarSubtract
from .constfold import ScalarFoldedConstant
from .divmul import ScalarDivide
from .literal import Number, Pi
from .reciprocal import ScalarReciprocal
from .square import ScalarSqrt, ScalarSquare
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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.asin(np.maximum(-1, expr_lower))
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.asin(np.minimum(expr_upper, 1))

        # sin(...) is periodic, so we need to drop to difference bounds before
        #  applying the difference to argv to stay in the same period
        arg_lower_diff = arg_lower - np.asin(exprv)
        arg_upper_diff = arg_upper - np.asin(exprv)

        # np.asin maps to [-pi/2, +pi/2] where sin is monotonically increasing
        # flip the argument error bounds where sin is monotonically decreasing
        needs_flip = (np.sin(argv + arg_lower_diff) > exprv) | (
            np.sin(argv + arg_upper_diff) < exprv
        )

        # check for the case where any finite value would work
        full_domain = (expr_lower <= -1) & (expr_upper >= 1)

        fmax = _floating_max(X.dtype)

        # sin(+-inf) = NaN, so force infinite argv to have exact bounds
        # sin(finite) in [-1, +1] so allow any finite argv if the all of
        #  [-1, +1] is allowed
        # FIXME: how do we handle bounds right next to the peak where the
        #        expression bounds could be exceeded inside the interval?
        arg_lower = np.where(  # type: ignore
            _isinf(argv),
            argv,
            np.where(
                full_domain,
                -fmax,
                np.minimum(
                    argv,
                    argv
                    + np.where(
                        needs_flip,
                        -arg_upper_diff,
                        arg_lower_diff,
                    ),
                ),
            ),
        )
        arg_upper = np.where(  # type: ignore
            _isinf(argv),
            argv,
            np.where(
                full_domain,
                fmax,
                np.maximum(
                    argv,
                    argv
                    + np.where(
                        needs_flip,
                        -arg_lower_diff,
                        arg_upper_diff,
                    ),
                ),
            ),
        )
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower = np.where(arg_lower == argv, argv, arg_lower)  # type: ignore
        arg_upper = np.where(arg_upper == argv, argv, arg_upper)  # type: ignore

        # handle rounding errors in asin(sin(...)) early
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

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for sin,
        #  even though it is periodic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"sin({self._a!r})"


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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            argv < -1,
            X.dtype.type(-np.inf),
            np.where(
                argv > 1,
                one_eps,
                np.minimum(argv, np.sin(np.maximum(-pi / 2, expr_lower))),
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            argv < -1,
            -one_eps,
            np.where(
                argv > 1,
                X.dtype.type(np.inf),
                np.maximum(argv, np.sin(np.minimum(expr_upper, pi / 2))),
            ),
        )
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower = np.where(arg_lower == argv, argv, arg_lower)  # type: ignore
        arg_upper = np.where(arg_upper == argv, argv, arg_upper)  # type: ignore

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

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for asin,
        #  which is strictly monotonic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"asin({self._a!r})"


class Trigonometric(Enum):
    cos = auto()
    tan = auto()
    cot = auto()
    sec = auto()
    csc = auto()
    acos = auto()
    atan = auto()
    acot = auto()
    asec = auto()
    acsc = auto()


class ScalarTrigonometric(Expr):
    __slots__ = ("_func", "_a")
    _func: Trigonometric
    _a: Expr

    def __init__(self, func: Trigonometric, a: Expr):
        self._func = func
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
        return ScalarTrigonometric(
            self._func,
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            TRIGONOMETRIC_UFUNC[self._func],  # type: ignore
            lambda e: ScalarTrigonometric(self._func, e),
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return (TRIGONOMETRIC_UFUNC[self._func])(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # rewrite trigonometric functions with base cases for sin and asin
        return (TRIGONOMETRIC_REWRITE[self._func])(self._a).compute_data_bounds(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._func.name}({self._a!r})"


TRIGONOMETRIC_REWRITE: dict[Trigonometric, Callable[[Expr], Expr]] = {
    # derived trigonometric functions
    # cos(x) = sin(pi/2 - x)
    #        = sin(x + pi/2)
    Trigonometric.cos: lambda x: ScalarSin(
        ScalarAdd(
            x,
            ScalarDivide(
                Pi(),
                Number.TWO,
            ),
        ),
    ),
    # tan(x) = sin(x) / cos(x)
    Trigonometric.tan: lambda x: ScalarDivide(
        ScalarSin(x),
        ScalarTrigonometric(Trigonometric.cos, x),
    ),
    # cot(x) = 1 / tan(x)
    #        = cos(x) / sin(x)
    Trigonometric.cot: lambda x: ScalarDivide(
        ScalarTrigonometric(Trigonometric.cos, x),
        ScalarSin(x),
    ),
    # sec(x) = 1 / cos(x)
    Trigonometric.sec: lambda x: ScalarReciprocal(
        ScalarTrigonometric(Trigonometric.cos, x),
    ),
    # csc(x) = 1 / sin(x)
    Trigonometric.csc: lambda x: ScalarReciprocal(
        ScalarSin(x),
    ),
    # inverse trigonometric functions
    # acos(x) = pi/2 - asin(x)
    Trigonometric.acos: lambda x: ScalarSubtract(
        ScalarDivide(Pi(), Number.TWO),
        ScalarAsin(x),
    ),
    # atan(x) = asin(x / sqrt(1 + x^2))
    Trigonometric.atan: lambda x: ScalarAsin(
        ScalarDivide(
            x,
            ScalarSqrt(
                ScalarAdd(
                    Number.ONE,
                    ScalarSquare(x),
                ),
            ),
        ),
    ),
    # acot(x) = atan(1/x)
    Trigonometric.acot: lambda x: ScalarTrigonometric(
        Trigonometric.atan,
        ScalarReciprocal(x),
    ),
    # asec(x) = acos(1/x)
    Trigonometric.asec: lambda x: ScalarTrigonometric(
        Trigonometric.acos,
        ScalarReciprocal(x),
    ),
    # acsc(x) = asin(1/x)
    Trigonometric.acsc: lambda x: ScalarAsin(
        ScalarReciprocal(x),
    ),
}

TRIGONOMETRIC_UFUNC: dict[Trigonometric, Callable[[np.ndarray], np.ndarray]] = {
    Trigonometric.cos: np.cos,
    Trigonometric.tan: np.tan,
    Trigonometric.cot: lambda x: _reciprocal(np.tan(x)),
    Trigonometric.sec: lambda x: _reciprocal(np.cos(x)),
    Trigonometric.csc: lambda x: _reciprocal(np.sin(x)),
    Trigonometric.acos: np.acos,
    Trigonometric.atan: np.atan,
    Trigonometric.acot: lambda x: np.atan(_reciprocal(x)),
    Trigonometric.asec: lambda x: np.acos(_reciprocal(x)),
    Trigonometric.acsc: lambda x: np.asin(_reciprocal(x)),
}

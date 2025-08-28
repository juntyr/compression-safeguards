from collections.abc import Mapping
from enum import Enum, auto
from typing import Callable

import numpy as np

from ....utils._compat import _asinh, _maximum, _minimum, _reciprocal, _sinh, _where
from ....utils._float128 import _float128_dtype
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .abs import ScalarAbs
from .addsub import ScalarAdd, ScalarSubtract
from .constfold import ScalarFoldedConstant
from .divmul import ScalarDivide
from .literal import Number
from .reciprocal import ScalarReciprocal
from .square import ScalarSqrt, ScalarSquare
from .typing import F, Ns, Ps, PsI


class ScalarSinh(Expr):
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
        return ScalarSinh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _sinh,  # type: ignore
            ScalarSinh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _sinh(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and sinh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _sinh(argv)

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(argv, _asinh(expr_lower))
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(argv, _asinh(expr_upper))

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in sinh(asinh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _sinh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _sinh(arg_upper),
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
        # the unchecked method already handles rounding errors for sinh,
        #  which is strictly monotonic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"sinh({self._a!r})"


class ScalarAsinh(Expr):
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
        return ScalarAsinh(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _asinh,  # type: ignore
            ScalarAsinh,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _asinh(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and asinh(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _asinh(argv)

        # apply the inverse function to get the bounds on arg
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _minimum(argv, _sinh(expr_lower))
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _maximum(argv, _sinh(expr_upper))

        # we need to force argv if expr_lower == expr_upper
        arg_lower = _where(expr_lower == expr_upper, argv, arg_lower)
        arg_upper = _where(expr_lower == expr_upper, argv, arg_upper)

        # handle rounding errors in asinh(sinh(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _asinh(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _asinh(arg_upper),
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
        # the unchecked method already handles rounding errors for asinh,
        #  which is strictly monotonic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"asinh({self._a!r})"


class Hyperbolic(Enum):
    cosh = auto()
    tanh = auto()
    coth = auto()
    sech = auto()
    csch = auto()
    acosh = auto()
    atanh = auto()
    acoth = auto()
    asech = auto()
    acsch = auto()


class ScalarHyperbolic(Expr):
    __slots__ = ("_func", "_a")
    _func: Hyperbolic
    _a: Expr

    def __init__(self, func: Hyperbolic, a: Expr):
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
        return ScalarHyperbolic(
            self._func,
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        fn = (
            HYPERBOLIC_QUADDTYPE_UFUNC if dtype == _float128_dtype else HYPERBOLIC_UFUNC
        )[self._func]
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            fn,  # type: ignore
            lambda e: ScalarHyperbolic(self._func, e),
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        fn = (
            HYPERBOLIC_QUADDTYPE_UFUNC
            if Xs.dtype == _float128_dtype
            else HYPERBOLIC_UFUNC
        )[self._func]
        return fn(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # rewrite hyperbolic functions with base cases for sinh and asinh
        return (HYPERBOLIC_REWRITE[self._func])(self._a).compute_data_bounds(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._func.name}({self._a!r})"


HYPERBOLIC_REWRITE: dict[Hyperbolic, Callable[[Expr], Expr]] = {
    # basic hyperbolic functions
    # cosh(x) = sqrt(1 + square(sinh(x)))
    Hyperbolic.cosh: lambda x: ScalarSqrt(
        ScalarAdd(
            Number.ONE,
            ScalarSquare(
                ScalarSinh(x),
            ),
        ),
    ),
    # derived hyperbolic functions
    # tanh(x) = sinh(x) / cosh(x)
    Hyperbolic.tanh: lambda x: ScalarDivide(
        ScalarSinh(x),
        ScalarHyperbolic(Hyperbolic.cosh, x),
    ),
    # coth(x) = cosh(x) / sinh(x)
    Hyperbolic.coth: lambda x: ScalarDivide(
        ScalarHyperbolic(Hyperbolic.cosh, x),
        ScalarSinh(x),
    ),
    # sech(x) = 1 / cosh(x)
    Hyperbolic.sech: lambda x: ScalarReciprocal(
        ScalarHyperbolic(Hyperbolic.cosh, x),
    ),
    # csch(x) = 1 / sinh(x)
    Hyperbolic.csch: lambda x: ScalarReciprocal(
        ScalarSinh(x),
    ),
    # inverse hyperbolic functions
    # acosh(x) = abs(asinh(sqrt(square(x) - 1)))
    Hyperbolic.acosh: lambda x: ScalarAbs(
        ScalarAsinh(
            ScalarSqrt(
                ScalarSubtract(
                    ScalarSquare(x),
                    Number.ONE,
                ),
            ),
        ),
    ),
    # atanh(x) = asinh(x / sqrt(1 - square(x)))
    Hyperbolic.atanh: lambda x: ScalarAsinh(
        ScalarDivide(
            x,
            ScalarSqrt(
                ScalarSubtract(
                    Number.ONE,
                    ScalarSquare(x),
                ),
            ),
        )
    ),
    # acoth(x) = atanh(1/x)
    #          = asinh(recip(sqrt(square(x) - 1)))
    Hyperbolic.acoth: lambda x: ScalarAsinh(
        ScalarReciprocal(
            ScalarSqrt(
                ScalarSubtract(
                    ScalarSquare(x),
                    Number.ONE,
                ),
            ),
        ),
    ),
    # asech(x) = acosh(1/x)
    #          = asinh(sqrt(recip(square(x)) - 1))
    Hyperbolic.asech: lambda x: ScalarAsinh(
        ScalarSqrt(
            ScalarSubtract(
                ScalarReciprocal(
                    ScalarSquare(x),
                ),
                Number.ONE,
            ),
        ),
    ),
    # acsch(x) = asinh(1/x)
    Hyperbolic.acsch: lambda x: ScalarAsinh(
        ScalarReciprocal(x),
    ),
}


HYPERBOLIC_UFUNC: dict[Hyperbolic, Callable[[np.ndarray], np.ndarray]] = {
    Hyperbolic.cosh: np.cosh,
    Hyperbolic.tanh: np.tanh,
    Hyperbolic.coth: lambda x: _reciprocal(np.tanh(x)),
    Hyperbolic.sech: lambda x: _reciprocal(np.cosh(x)),
    Hyperbolic.csch: lambda x: _reciprocal(np.sinh(x)),
    Hyperbolic.acosh: np.acosh,
    Hyperbolic.atanh: np.atanh,
    Hyperbolic.acoth: lambda x: np.atanh(_reciprocal(x)),
    Hyperbolic.asech: lambda x: np.acosh(_reciprocal(x)),
    Hyperbolic.acsch: lambda x: np.asinh(_reciprocal(x)),
}


def propagate_negative_zero(
    x: np.ndarray[Ps, np.dtype[F]], fx: np.ndarray[Ps, np.dtype[F]]
) -> np.ndarray[Ps, np.dtype[F]]:
    return _where(fx == x, x, fx)


HYPERBOLIC_QUADDTYPE_UFUNC: dict[Hyperbolic, Callable[[np.ndarray], np.ndarray]] = {
    Hyperbolic.cosh: lambda x: (np.exp(x) + np.exp(-x)) / 2,
    Hyperbolic.tanh: lambda x: propagate_negative_zero(
        x, (np.exp(x * 2) - 1) / (np.exp(x * 2) + 1)
    ),
    Hyperbolic.coth: lambda x: _reciprocal(
        (HYPERBOLIC_QUADDTYPE_UFUNC[Hyperbolic.tanh])(x)
    ),
    Hyperbolic.sech: lambda x: _reciprocal(
        (HYPERBOLIC_QUADDTYPE_UFUNC[Hyperbolic.cosh])(x)
    ),
    Hyperbolic.csch: lambda x: _reciprocal(_sinh(x)),
    Hyperbolic.acosh: lambda x: np.log(x + np.sqrt(np.square(x) - 1)),
    Hyperbolic.atanh: lambda x: propagate_negative_zero(
        x, (np.log(1 + x) - np.log(1 - x)) / 2
    ),
    Hyperbolic.acoth: lambda x: (HYPERBOLIC_QUADDTYPE_UFUNC[Hyperbolic.atanh])(
        _reciprocal(x)
    ),
    Hyperbolic.asech: lambda x: (HYPERBOLIC_QUADDTYPE_UFUNC[Hyperbolic.acosh])(
        _reciprocal(x)
    ),
    Hyperbolic.acsch: lambda x: _asinh(_reciprocal(x)),
}

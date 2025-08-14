from collections.abc import Mapping
from enum import Enum, auto
from typing import Callable

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import (
    _float128_dtype,
    _float128_smallest_subnormal,
)
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class Logarithm(Enum):
    ln = auto()
    log2 = auto()
    log10 = auto()


class ScalarLog(Expr):
    __slots__ = ("_log", "_a")
    _log: Logarithm
    _a: Expr

    def __init__(self, log: Logarithm, a: Expr):
        self._log = log
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
        return ScalarLog(
            self._log,
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            LOGARITHM_UFUNC[self._log],  # type: ignore
            lambda e: ScalarLog(self._log, e),
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return (LOGARITHM_UFUNC[self._log])(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and log(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = (LOGARITHM_UFUNC[self._log])(argv)

        if X.dtype == _float128_dtype:
            smallest_subnormal = _float128_smallest_subnormal
        else:
            smallest_subnormal = np.finfo(X.dtype).smallest_subnormal

        # apply the inverse function to get the bounds on arg
        # log(...) is NaN for negative values, so ... can be any negative value
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            argv < 0,
            X.dtype.type(-np.inf),
            np.minimum(argv, (LOGARITHM_EXPONENTIAL_UFUNC[self._log])(expr_lower)),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            argv < 0,
            -smallest_subnormal,
            np.maximum(argv, (LOGARITHM_EXPONENTIAL_UFUNC[self._log])(expr_upper)),
        )
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower = np.where(arg_lower == argv, argv, arg_lower)  # type: ignore
        arg_upper = np.where(arg_upper == argv, argv, arg_upper)  # type: ignore

        # handle rounding errors in log(exp(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: (LOGARITHM_UFUNC[self._log])(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: (LOGARITHM_UFUNC[self._log])(arg_upper),
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
        # the unchecked method already handles rounding errors for ln / log2 /
        #  log10, which are strictly monotonic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._log.name}({self._a!r})"


LOGARITHM_UFUNC: dict[Logarithm, Callable[[np.ndarray], np.ndarray]] = {
    Logarithm.ln: np.log,
    Logarithm.log2: np.log2,
    Logarithm.log10: np.log10,
}


class Exponential(Enum):
    exp = auto()
    exp2 = auto()
    exp10 = auto()


class ScalarExp(Expr):
    __slots__ = ("_exp", "_a")
    _exp: Exponential
    _a: Expr

    def __init__(self, exp: Exponential, a: Expr):
        self._exp = exp
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
        return ScalarExp(
            self._exp,
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            EXPONENTIAL_UFUNC[self._exp],  # type: ignore
            lambda e: ScalarExp(self._exp, e),
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return (EXPONENTIAL_UFUNC[self._exp])(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and exp(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = (EXPONENTIAL_UFUNC[self._exp])(argv)

        # apply the inverse function to get the bounds on arg
        # exp(...) cannot be negative, so ensure the bounds on expr also cannot
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.minimum(
            argv, (EXPONENTIAL_LOGARITHM_UFUNC[self._exp])(np.maximum(0, expr_lower))
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.maximum(
            argv, (EXPONENTIAL_LOGARITHM_UFUNC[self._exp])(expr_upper)
        )
        # if arg_lower == argv and argv == -0.0, we need to guarantee that
        #  arg_lower is also -0.0, same for arg_upper
        arg_lower = np.where(arg_lower == argv, argv, arg_lower)  # type: ignore
        arg_upper = np.where(arg_upper == argv, argv, arg_upper)  # type: ignore

        # handle rounding errors in exp(log(...)) early
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: (EXPONENTIAL_UFUNC[self._exp])(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: (EXPONENTIAL_UFUNC[self._exp])(arg_upper),
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
        # the unchecked method already handles rounding errors for exp / exp2 /
        #  exp10, which are strictly monotonic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._exp.name}({self._a!r})"


EXPONENTIAL_UFUNC: dict[Exponential, Callable[[np.ndarray], np.ndarray]] = {
    Exponential.exp: np.exp,
    Exponential.exp2: np.exp2,
    Exponential.exp10: lambda x: np.power(10, x),
}

LOGARITHM_EXPONENTIAL_UFUNC: dict[Logarithm, Callable[[np.ndarray], np.ndarray]] = {
    Logarithm.ln: EXPONENTIAL_UFUNC[Exponential.exp],
    Logarithm.log2: EXPONENTIAL_UFUNC[Exponential.exp2],
    Logarithm.log10: EXPONENTIAL_UFUNC[Exponential.exp10],
}

EXPONENTIAL_LOGARITHM_UFUNC: dict[Exponential, Callable[[np.ndarray], np.ndarray]] = {
    Exponential.exp: LOGARITHM_UFUNC[Logarithm.ln],
    Exponential.exp2: LOGARITHM_UFUNC[Logarithm.log2],
    Exponential.exp10: LOGARITHM_UFUNC[Logarithm.log10],
}


class ScalarLogWithBase(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

    @property
    def has_data(self) -> bool:
        return self._a.has_data or self._b.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices | self._b.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarLogWithBase(
            self._a.apply_array_element_offset(axis, offset),
            self._b.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        from .divmul import ScalarDivide

        return ScalarDivide(
            ScalarLog(
                Logarithm.ln,
                self._a,
            ),
            ScalarLog(
                Logarithm.ln,
                self._b,
            ),
        ).constant_fold(dtype)

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        from .divmul import ScalarDivide

        return ScalarDivide(
            ScalarLog(
                Logarithm.ln,
                self._a,
            ),
            ScalarLog(
                Logarithm.ln,
                self._b,
            ),
        ).eval(x, Xs, late_bound)

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        from .divmul import ScalarDivide

        return ScalarDivide(
            ScalarLog(
                Logarithm.ln,
                self._a,
            ),
            ScalarLog(
                Logarithm.ln,
                self._b,
            ),
        ).compute_data_bounds_unchecked(expr_lower, expr_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"log({self._a!r}, base={self._b!r})"

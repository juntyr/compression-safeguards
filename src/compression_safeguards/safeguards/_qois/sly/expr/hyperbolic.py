from collections.abc import Mapping
from enum import Enum, auto
from typing import Callable

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import _float128_dtype
from .....utils.typing import F, S
from .abc import Expr
from .addsub import ScalarAdd, ScalarSubtract
from .constfold import FoldedScalarConst
from .divmul import ScalarDivide, ScalarMultiply
from .literal import Number
from .logexp import ScalarExp
from .neg import ScalarNegate


class Hyperbolic(Enum):
    sinh = auto()
    cosh = auto()
    tanh = auto()
    coth = auto()
    sech = auto()
    csch = auto()
    asinh = auto()
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
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        ufunc = HYPERBOLIC_EVAL.get(self._func, None)
        if (ufunc is None) or (dtype == _float128_dtype):
            return (HYPERBOLIC_REWRITE[self._func])(self._a).constant_fold(dtype)
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, ufunc)  # type: ignore

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        ufunc = HYPERBOLIC_EVAL.get(self._func, None)
        if (ufunc is None) or (X.dtype == _float128_dtype):
            return (HYPERBOLIC_REWRITE[self._func])(self._a).eval(X, late_bound)
        return ufunc(self._a.eval(X, late_bound))  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # rewrite hyperbolic functions using their exponential definition
        return (HYPERBOLIC_REWRITE[self._func])(self._a).compute_data_error_bound(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._func.name}({self._a!r})"


HYPERBOLIC_REWRITE: dict[Hyperbolic, Callable[[Expr], Expr]] = {
    # basic hyperbolic functions
    # sinh(x) = (e^x - e^(-x)) / 2
    Hyperbolic.sinh: lambda x: ScalarDivide(
        ScalarSubtract(
            ScalarExp(x),
            ScalarExp(ScalarNegate(x)),
        ),
        Number("2"),
    ),
    # cosh(x) = (e^x + e^(-x)) / 2
    Hyperbolic.cosh: lambda x: ScalarDivide(
        ScalarAdd(
            ScalarExp(x),
            ScalarExp(ScalarNegate(x)),
        ),
        Number("2"),
    ),
    # derived hyperbolic functions
    # tanh(x) = (e^(2x) + 1) / (e^(2x) - 1)
    Hyperbolic.tanh: lambda x: ScalarDivide(
        ScalarSubtract(
            ScalarExp(
                ScalarMultiply(Number("2"), x),
            ),
            Number("1"),
        ),
        ScalarAdd(
            ScalarExp(
                ScalarMultiply(Number("2"), x),
            ),
            Number("1"),
        ),
    ),
    # Hyperbolic.csch: lambda x: (2 / (ScalarExp(x) - ScalarExp(ScalarNegate(x)))),
    # Hyperbolic.sech: lambda x: (2 / (ScalarExp(x) + ScalarExp(ScalarNegate(x)))),
    # Hyperbolic.coth: lambda x: ((ScalarExp(x * 2) + 1) / (ScalarExp(x * 2) - 1)),
    # # inverse hyperbolic functions
    # Hyperbolic.asinh: lambda x: (sp.ln(x + sp.sqrt(x**2 + 1))),
    # Hyperbolic.acosh: lambda x: (sp.ln(x + sp.sqrt(x**2 - 1))),
    # Hyperbolic.atanh: lambda x: (sp.ln((1 + x) / (1 - x)) / 2),
    # Hyperbolic.acsch: lambda x: (sp.ln((1 / x) + sp.sqrt(x ** (-2) + 1))),
    # Hyperbolic.asech: lambda x: (sp.ln((1 + sp.sqrt(1 - x**2)) / x)),
    # Hyperbolic.acoth: lambda x: (sp.ln((x + 1) / (x - 1)) / 2),
}

HYPERBOLIC_EVAL: dict[Hyperbolic, Callable[[np.ndarray], np.ndarray]] = {
    Hyperbolic.sinh: np.sinh,
    Hyperbolic.cosh: np.cosh,
    Hyperbolic.tanh: np.tanh,
    Hyperbolic.asinh: np.asinh,
    Hyperbolic.acosh: np.acosh,
    Hyperbolic.atanh: np.atanh,
}

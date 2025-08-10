from collections.abc import Mapping
from enum import Enum, auto
from typing import Callable

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _float128_dtype, _reciprocal
from .abc import Expr
from .addsub import ScalarAdd, ScalarSubtract
from .constfold import ScalarFoldedConstant
from .divmul import ScalarDivide, ScalarMultiply
from .literal import Number
from .logexp import Exponential, Logarithm, ScalarExp, ScalarLog
from .neg import ScalarNegate
from .square import ScalarSqrt, ScalarSquare
from .typing import F, Ns, Ps, PsI


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
        if dtype == _float128_dtype:
            return (HYPERBOLIC_REWRITE[self._func])(self._a).constant_fold(dtype)
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            HYPERBOLIC_UFUNC[self._func],  # type: ignore
            lambda e: ScalarHyperbolic(self._func, e),
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        if Xs.dtype == _float128_dtype:
            return (HYPERBOLIC_REWRITE[self._func])(self._a).eval(x, Xs, late_bound)
        return (HYPERBOLIC_UFUNC[self._func])(self._a.eval(x, Xs, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # rewrite hyperbolic functions using their exponential definitions
        return (HYPERBOLIC_REWRITE[self._func])(self._a).compute_data_error_bound(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # rewrite hyperbolic functions using their exponential definitions
        return (HYPERBOLIC_REWRITE[self._func])(self._a).compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._func.name}({self._a!r})"


HYPERBOLIC_REWRITE: dict[Hyperbolic, Callable[[Expr], Expr]] = {
    # basic hyperbolic functions
    # sinh(x) = (e^x - e^(-x)) / 2
    Hyperbolic.sinh: lambda x: ScalarMultiply(
        ScalarSubtract(
            ScalarExp(Exponential.exp, x),
            ScalarExp(Exponential.exp, ScalarNegate(x)),
        ),
        Number.TWO,
    ),
    # cosh(x) = (e^x + e^(-x)) / 2
    Hyperbolic.cosh: lambda x: ScalarDivide(
        ScalarAdd(
            ScalarExp(Exponential.exp, x),
            ScalarExp(Exponential.exp, ScalarNegate(x)),
        ),
        Number.TWO,
    ),
    # derived hyperbolic functions
    # tanh(x) = sinh(x) / cosh(x)
    #         = (e^(2x) - 1) / (e^(2x) + 1)
    Hyperbolic.tanh: lambda x: ScalarDivide(
        ScalarSubtract(
            ScalarExp(
                Exponential.exp,
                ScalarMultiply(Number.TWO, x),
            ),
            Number.ONE,
        ),
        ScalarAdd(
            ScalarExp(
                Exponential.exp,
                ScalarMultiply(Number.TWO, x),
            ),
            Number.ONE,
        ),
    ),
    # coth(x) = cosh(x) / sinh(x)
    #         = (e^(2x) + 1) / (e^(2x) - 1)
    Hyperbolic.coth: lambda x: ScalarDivide(
        ScalarAdd(
            ScalarExp(
                Exponential.exp,
                ScalarMultiply(Number.TWO, x),
            ),
            Number.ONE,
        ),
        ScalarSubtract(
            ScalarExp(
                Exponential.exp,
                ScalarMultiply(Number.TWO, x),
            ),
            Number.ONE,
        ),
    ),
    # sech(x) = 1 / cosh(x)
    #         = 2 / (e^x + e^(-x))
    Hyperbolic.sech: lambda x: ScalarDivide(
        Number.TWO,
        ScalarAdd(
            ScalarExp(
                Exponential.exp,
                x,
            ),
            ScalarExp(
                Exponential.exp,
                ScalarNegate(x),
            ),
        ),
    ),
    # csch(x) = 1 / sinh(x)
    #         = 2 / (e^x - e^(-x))
    Hyperbolic.csch: lambda x: ScalarDivide(
        Number.TWO,
        ScalarSubtract(
            ScalarExp(
                Exponential.exp,
                x,
            ),
            ScalarExp(
                Exponential.exp,
                ScalarNegate(x),
            ),
        ),
    ),
    # inverse hyperbolic functions
    # asinh(x) = ln(x + sqrt(x^2 + 1))
    Hyperbolic.asinh: lambda x: ScalarLog(
        Logarithm.ln,
        ScalarAdd(
            x,
            ScalarSqrt(
                ScalarAdd(
                    ScalarSquare(x),
                    Number.ONE,
                )
            ),
        ),
    ),
    # acosh(x) = ln(x + sqrt(x^2 - 1))
    Hyperbolic.acosh: lambda x: ScalarLog(
        Logarithm.ln,
        ScalarAdd(
            x,
            ScalarSqrt(
                ScalarSubtract(
                    ScalarSquare(x),
                    Number.ONE,
                )
            ),
        ),
    ),
    # atanh(x) = ln( (1+x) / (1-x) ) / 2
    #          = ( ln(1+x) - ln(1-x) ) / 2
    Hyperbolic.atanh: lambda x: ScalarDivide(
        ScalarSubtract(
            ScalarLog(
                Logarithm.ln,
                ScalarAdd(
                    Number.ONE,
                    x,
                ),
            ),
            ScalarLog(
                Logarithm.ln,
                ScalarSubtract(
                    Number.ONE,
                    x,
                ),
            ),
        ),
        Number.TWO,
    ),
    # acoth(x) = ln( (x+1) / (x-1) ) / 2
    #          = ( ln(x+1) - ln(x-1) ) / 2
    Hyperbolic.acoth: lambda x: ScalarDivide(
        ScalarSubtract(
            ScalarLog(
                Logarithm.ln,
                ScalarAdd(
                    x,
                    Number.ONE,
                ),
            ),
            ScalarLog(
                Logarithm.ln,
                ScalarSubtract(
                    x,
                    Number.ONE,
                ),
            ),
        ),
        Number.TWO,
    ),
    # asech(x) = ln( 1/x + sqrt( 1/(x^2) - 1 ) )
    #          = ln( 1/x + sqrt( 1/(x^2) - (x^2)/(x^2) ) )
    #          = ln( 1/x + sqrt(1 - x^2)/x )
    #          = ln( (1 + sqrt(1 - x^2)) / x )
    #          = ln(1 + sqrt(1 - x^2)) - ln(x)
    Hyperbolic.asech: lambda x: ScalarSubtract(
        ScalarLog(
            Logarithm.ln,
            ScalarAdd(
                Number.ONE,
                ScalarSqrt(
                    ScalarSubtract(
                        Number.ONE,
                        ScalarSquare(x),
                    )
                ),
            ),
        ),
        ScalarLog(
            Logarithm.ln,
            x,
        ),
    ),
    # acsch(x) = ln( 1/x + sqrt( 1/(x^2) + 1 ) )
    #          = ln( 1/x + sqrt( 1/(x^2) + (x^2)/(x^2) ) )
    #          = ln( 1/x + sqrt(1 + x^2)/x )
    #          = ln( (1 + sqrt(1 + x^2)) / x )
    #          = ln(1 + sqrt(1 + x^2)) - ln(x)
    Hyperbolic.acsch: lambda x: ScalarSubtract(
        ScalarLog(
            Logarithm.ln,
            ScalarAdd(
                Number.ONE,
                ScalarSqrt(
                    ScalarAdd(
                        Number.ONE,
                        ScalarSquare(x),
                    )
                ),
            ),
        ),
        ScalarLog(
            Logarithm.ln,
            x,
        ),
    ),
}

HYPERBOLIC_UFUNC: dict[Hyperbolic, Callable[[np.ndarray], np.ndarray]] = {
    Hyperbolic.sinh: np.sinh,
    Hyperbolic.cosh: np.cosh,
    Hyperbolic.tanh: np.tanh,
    Hyperbolic.coth: lambda x: _reciprocal(np.tanh(x)),
    Hyperbolic.sech: lambda x: _reciprocal(np.cosh(x)),
    Hyperbolic.csch: lambda x: _reciprocal(np.sinh(x)),
    Hyperbolic.asinh: np.asinh,
    Hyperbolic.acosh: np.acosh,
    Hyperbolic.atanh: np.atanh,
    Hyperbolic.acoth: lambda x: np.atanh(_reciprocal(x)),
    Hyperbolic.asech: lambda x: np.acosh(_reciprocal(x)),
    Hyperbolic.acsch: lambda x: np.asinh(_reciprocal(x)),
}

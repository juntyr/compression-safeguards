from collections.abc import Mapping
from enum import Enum, auto
from typing import Callable

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _nan_to_zero_inf_to_finite
from ..eb import ensure_bounded_derived_error
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        zero = X.dtype.type(0)

        # evaluate arg and log(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = (LOGARITHM_UFUNC[self._log])(argv)

        # update the error bounds
        eal: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_lower == 0),
            zero,
            np.minimum(
                (LOGARITHM_EXPONENTIAL_UFUNC[self._log])(exprv + eb_expr_lower) - argv,
                0,
            ),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_upper == 0),
            zero,
            np.maximum(
                0,
                (LOGARITHM_EXPONENTIAL_UFUNC[self._log])(exprv + eb_expr_upper) - argv,
            ),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in log(exp(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: (LOGARITHM_UFUNC[self._log])(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: (LOGARITHM_UFUNC[self._log])(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return arg.compute_data_error_bound(
            eb_arg_lower,
            eb_arg_upper,
            X,
            Xs,
            late_bound,
        )

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for ln / log2,
        #  which are strictly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        zero = X.dtype.type(0)

        # evaluate arg and exp(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = (EXPONENTIAL_UFUNC[self._exp])(argv)

        # update the error bounds
        # ensure that log is not passed a negative argument
        eal: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_lower == 0),
            zero,
            np.minimum(
                (EXPONENTIAL_LOGARITHM_UFUNC[self._exp])(
                    np.maximum(0, exprv + eb_expr_lower)
                )
                - argv,
                0,
            ),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_upper == 0),
            zero,
            np.maximum(
                0,
                (EXPONENTIAL_LOGARITHM_UFUNC[self._exp])(
                    np.maximum(0, exprv + eb_expr_upper)
                )
                - argv,
            ),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in exp(log(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: (EXPONENTIAL_UFUNC[self._exp])(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: (EXPONENTIAL_UFUNC[self._exp])(argv + eau),
            exprv,
            argv,
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return arg.compute_data_error_bound(
            eb_arg_lower,
            eb_arg_upper,
            X,
            Xs,
            late_bound,
        )

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for exp / exp2,
        #  which is strictly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
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
        ).compute_data_error_bound(eb_expr_lower, eb_expr_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"log({self._a!r}, base={self._b!r})"

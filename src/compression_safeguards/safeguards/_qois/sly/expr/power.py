from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.typing import F, S
from .abc import Expr
from .constfold import FoldedScalarConst
from .divmul import ScalarMultiply
from .logexp import ScalarExp, ScalarLn


class ScalarExponentiation(Expr):
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
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.power, ScalarExponentiation
        )

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.power(self._a.eval(X, late_bound), self._b.eval(X, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # rewrite a ** b as e^(b*ln(abs(a)))
        # this is mathematically incorrect for a <= 0 but works for deriving error bounds
        return ScalarExp(
            ScalarMultiply(self._b, ScalarLn(ScalarFakeAbs(self._a)))
        ).compute_data_error_bound(eb_expr_lower, eb_expr_upper, X, late_bound)

    def __repr__(self) -> str:
        return f"{self._a!r} ** {self._b!r}"


class ScalarFakeAbs(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.abs)

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.abs(self._a.eval(X, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(X, late_bound)

        # flip the lower/upper error bound if the arg is negative
        eal = np.where(argv < 0, -eb_expr_upper, eb_expr_lower)
        eau = np.where(argv < 0, -eb_expr_lower, eb_expr_upper)

        # composition using Lemma 3 from Jiao et al.
        return arg.compute_data_error_bound(
            eal,  # type: ignore
            eau,  # type: ignore
            X,
            late_bound,
        )

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # abs cannot cause any rounding errors
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"-{self._a!r}"

from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import _nan_to_zero_inf_to_finite
from .....utils.typing import F, S
from ...eb import ensure_bounded_derived_error
from .abc import Expr
from .constfold import FoldedScalarConst


class ScalarLn(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.log)

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.log(self._a.eval(X, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        zero = X.dtype.type(0)

        # evaluate arg and ln(arg)
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.log(argv)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(np.exp(exprv + eb_expr_lower) - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(0, np.exp(exprv + eb_expr_upper) - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in ln(e^(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.log(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.log(argv + eau),
            exprv,
            argv,  # type: ignore
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return arg.compute_data_error_bound(
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
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
        # the unchecked method already handles rounding errors for ln,
        #  which is stricly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"ln({self._a!r})"


class ScalarExp(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.exp)

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.exp(self._a.eval(X, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        zero = X.dtype.type(0)

        # evaluate arg and e^arg
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.exp(argv)

        # update the error bounds
        # ensure that ln is not passed a negative argument
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(np.log(np.maximum(0, exprv + eb_expr_lower)) - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(0, np.log(np.maximum(0, exprv + eb_expr_upper)) - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in e^(ln(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.exp(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.exp(argv + eau),
            exprv,
            argv,  # type: ignore
            eau,
            eb_expr_lower,
            eb_expr_upper,
        )
        eb_arg_lower, eb_arg_upper = eal, eau

        # composition using Lemma 3 from Jiao et al.
        return arg.compute_data_error_bound(
            eb_arg_lower,  # type: ignore
            eb_arg_upper,  # type: ignore
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
        # the unchecked method already handles rounding errors for exp,
        #  which is stricly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"exp({self._a!r})"

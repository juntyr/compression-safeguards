from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import _nan_to_zero_inf_to_finite
from .....utils.typing import F, S
from ...eb import ensure_bounded_derived_error
from .abc import Expr
from .constfold import FoldedScalarConst


class ScalarSqrt(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.sqrt)

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.sqrt(self._a.eval(X, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        zero = X.dtype.type(0)

        # evaluate arg and sqrt(arg)
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.sqrt(argv)

        # update the error bounds
        # ensure that sqrt(...) = exprv + eb does not become negative
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(
                np.square(np.maximum(0, exprv + eb_expr_lower)) - argv,
                0,
            ),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(
                0,
                np.square(np.maximum(0, exprv + eb_expr_upper)) - argv,
            ),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in sqrt(square(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.sqrt(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.sqrt(argv + eau),
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
        # the unchecked method already handles rounding errors for sqrt,
        #  which is strictly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"sqrt({self._a!r})"


class ScalarSquare(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.square)

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.square(self._a.eval(X, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        zero = X.dtype.type(0)

        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.square(argv)

        argv_lower = np.sqrt(np.maximum(0, exprv + eb_expr_lower))
        argv_upper = np.sqrt(np.maximum(0, exprv + eb_expr_upper))

        # ensure that square(x) does not go below zero
        al = np.where((argv_lower == 0) | (argv < 0), -argv_upper, argv_lower)
        au = np.where((argv_lower > 0) & (argv < 0), -argv_lower, argv_upper)

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(al - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(0, au - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in square(sqrt(x)) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.square(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.square(argv + eau),
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
        # the unchecked method already handles rounding errors for square,
        #  even though it is *not* monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"square({self._a!r})"

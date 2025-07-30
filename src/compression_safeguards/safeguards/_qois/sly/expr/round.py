from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import _nan_to_zero_inf_to_finite, _nextafter
from .....utils.typing import F, S
from ...eb import ensure_bounded_derived_error
from .abc import Expr
from .constfold import FoldedScalarConst


class ScalarFloor(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.floor)  # type: ignore

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.floor(self._a.eval(X, late_bound))  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # evaluate arg and floor(arg)
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.floor(argv)

        # compute the rounded result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will round to meet the error bounds
        argv_lower = exprv_lower
        argv_upper = _nextafter(exprv_upper + 1, exprv_upper)

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in floor(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.floor(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.floor(argv + eau),
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
        # the unchecked method already handles rounding errors for floor,
        #  which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"floor({self._a!r})"


class ScalarCeil(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.ceil)  # type: ignore

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.ceil(self._a.eval(X, late_bound))  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # evaluate arg and ceil(arg)
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.ceil(argv)

        # compute the rounded result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will round to meet the error bounds
        argv_lower = _nextafter(exprv_lower - 1, exprv_lower)
        argv_upper = exprv_upper

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in ceil(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.ceil(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.ceil(argv + eau),
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
        # the unchecked method already handles rounding errors for ceil,
        #  which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"ceil({self._a!r})"


class ScalarTrunc(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.trunc)  # type: ignore

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.trunc(self._a.eval(X, late_bound))  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # evaluate arg and trunc(arg)
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.trunc(argv)

        # compute the truncated result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will truncate to meet the error bounds
        argv_lower = np.where(
            exprv_lower <= 0, _nextafter(exprv_lower - 1, exprv_lower), exprv_lower
        )
        argv_upper = np.where(
            exprv_upper >= 0, _nextafter(exprv_upper + 1, exprv_upper), exprv_upper
        )

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in trunc(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.trunc(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.trunc(argv + eau),
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
        # the unchecked method already handles rounding errors for trunc,
        #  which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"trunc({self._a!r})"


class ScalarRoundTiesEven(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, np.rint)  # type: ignore

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return np.rint(self._a.eval(X, late_bound))  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        # evaluate arg and round_ties_even(arg)
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = np.rint(argv)

        # compute the rounded result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will round to meet the error bounds
        argv_lower = exprv_lower - 0.5
        argv_upper = exprv_upper + 0.5

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal = np.minimum(argv_lower - argv, 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.maximum(0, argv_upper - argv)
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in round_ties_even(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.rint(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.rint(argv + eau),
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
        # the unchecked method already handles rounding errors for
        #  round_ties_even, which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"round_ties_even({self._a!r})"

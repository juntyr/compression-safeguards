from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import (
    _float128_dtype,
    _float128_smallest_subnormal,
    _isnan,
    _nan_to_zero_inf_to_finite,
    _sign,
)
from .....utils.typing import F, S
from ...eb import ensure_bounded_derived_error
from .abc import Expr
from .constfold import FoldedScalarConst


class ScalarSign(Expr):
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
        return FoldedScalarConst.constant_fold_unary(self._a, dtype, _sign)  # type: ignore

    def eval(
        self,
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> F | np.ndarray[tuple[int, ...], np.dtype[F]]:
        return _sign(self._a.eval(X, late_bound))  # type: ignore

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[S, np.dtype[F]],
        eb_expr_upper: np.ndarray[S, np.dtype[F]],
        X: np.ndarray[tuple[int, ...], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[int, ...], np.dtype[F]]],
    ) -> tuple[np.ndarray[S, np.dtype[F]], np.ndarray[S, np.dtype[F]]]:
        zero = X.dtype.type(0)

        # evaluate arg and sign(arg)
        arg = self._a
        argv = arg.eval(X, late_bound)
        exprv = _sign(argv)  # type: ignore

        # evaluate the lower and upper sign bounds that satisfy the error bound
        exprv_lower = np.maximum(-1, exprv + np.maximum(-2, np.ceil(eb_expr_lower)))
        exprv_upper = np.minimum(exprv + np.minimum(np.floor(eb_expr_upper), +2), +1)

        if X.dtype == _float128_dtype:
            smallest_subnormal = _float128_smallest_subnormal
        else:
            smallest_subnormal = np.finfo(X.dtype).smallest_subnormal

        # compute the lower and upper arg bounds that produce the sign bounds
        argv_lower = np.where(
            _isnan(exprv),
            exprv,
            np.where(
                exprv_lower == 0,
                zero,
                np.where(
                    exprv_lower < 0,
                    np.array(-np.inf, dtype=X.dtype),
                    smallest_subnormal,
                ),
            ),
        )
        argv_upper = np.where(
            _isnan(exprv),
            exprv,
            np.where(
                exprv_upper == 0,
                zero,
                np.where(
                    exprv_upper < 0,
                    -smallest_subnormal,
                    np.array(np.inf, dtype=X.dtype),
                ),
            ),
        )

        # update the error bounds
        eal = np.where(
            (eb_expr_lower == 0),
            zero,
            np.minimum(argv_lower - argv, 0),
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau = np.where(
            (eb_expr_upper == 0),
            zero,
            np.maximum(0, argv_upper - argv),
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in sign(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: _sign(argv + eal),
            exprv,
            argv,  # type: ignore
            eal,
            eb_expr_lower,  # type: ignore
            eb_expr_upper,  # type: ignore
        )
        eau = ensure_bounded_derived_error(
            lambda eau: _sign(argv + eau),
            exprv,
            argv,  # type: ignore
            eau,
            eb_expr_lower,  # type: ignore
            eb_expr_upper,  # type: ignore
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
        # the unchecked method already handles rounding errors for sign,
        #  which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, late_bound
        )

    def __repr__(self) -> str:
        return f"sign({self._a!r})"

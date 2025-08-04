from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import (
    _float128_dtype,
    _float128_smallest_subnormal,
    _nan_to_zero_inf_to_finite,
    _reciprocal,
)
from ..eb import ensure_bounded_derived_error
from .abc import Expr
from .constfold import FoldedScalarConst
from .typing import F, Ns, Ps, PsI


class ScalarReciprocal(Expr):
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
        return ScalarReciprocal(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_unary(
            self._a,
            dtype,
            _reciprocal,  # type: ignore
            ScalarReciprocal,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _reciprocal(self._a.eval(x, Xs, late_bound))

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        zero = X.dtype.type(0)

        # evaluate arg and reciprocal(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _reciprocal(argv)

        if X.dtype == _float128_dtype:
            smallest_subnormal = _float128_smallest_subnormal
        else:
            smallest_subnormal = np.finfo(X.dtype).smallest_subnormal

        # compute the argument bounds
        # ensure that reciprocal(...) = exprv + eb keeps the same sign
        argv_lower = _reciprocal(
            np.where(
                exprv < 0,
                np.minimum(exprv + eb_expr_upper, -smallest_subnormal),
                exprv + eb_expr_upper,
            )
        )
        argv_upper = _reciprocal(
            np.where(
                exprv < 0,
                exprv + eb_expr_lower,
                np.maximum(smallest_subnormal, exprv + eb_expr_lower),
            )
        )

        # update the error bounds
        eal: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_lower == 0) | (exprv == 0),
            zero,
            argv_lower - argv,
        )
        eal = _nan_to_zero_inf_to_finite(eal)

        eau: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_upper == 0) | (exprv == 0),
            zero,
            argv_upper - argv,
        )
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in reciprocal(reciprocal(...)) early
        eal = ensure_bounded_derived_error(
            lambda eal: _reciprocal(np.add(argv, eal)),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: _reciprocal(np.add(argv, eau)),
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
        # the unchecked method already handles rounding errors for reciprocal,
        #  which is strictly monotonic in two disjoint segments
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"reciprocal({self._a!r})"

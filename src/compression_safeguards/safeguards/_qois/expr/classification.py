from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _float128_dtype, _float128_max, _isfinite, _isinf, _isnan
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Fi, Ns, Ps, PsI


class ScalarIsFinite(Expr):
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
        return ScalarIsFinite(self._a.apply_array_element_offset(axis, offset))

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: np.astype(_isfinite(x), dtype),  # type: ignore
            ScalarIsFinite,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.astype(_isfinite(self._a.eval(x, Xs, late_bound)), Xs.dtype)  # type: ignore

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)

        fmax = _float128_max if X.dtype == _float128_dtype else np.finfo(X.dtype).max

        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_lower > 0,
            # must be finite
            -fmax,
            np.where(
                expr_upper < 1,
                # must be non-finite, i.e. stay the same
                argv,
                # can be finite or non-finite
                X.dtype.type(-np.inf),
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_lower > 0,
            # must be finite
            fmax,
            np.where(
                expr_upper < 1,
                # must be non-finite, i.e. stay the same
                argv,
                # can be finite or non-finite
                X.dtype.type(np.inf),
            ),
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
        # isfinite cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"isfinite({self._a!r})"


class ScalarIsInf(Expr):
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
        return ScalarIsInf(self._a.apply_array_element_offset(axis, offset))

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: np.astype(_isinf(x), dtype),  # type: ignore
            ScalarIsInf,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.astype(_isinf(self._a.eval(x, Xs, late_bound)), Xs.dtype)  # type: ignore

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)

        fmax = _float128_max if X.dtype == _float128_dtype else np.finfo(X.dtype).max

        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_lower > 0,
            # must be infinite, i.e. stay the same
            argv,
            np.where(
                expr_upper < 1,
                # must be non-infinite
                -fmax,
                # can be infinite or non-infinite
                X.dtype.type(-np.inf),
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_lower > 0,
            # must be infinite, i.e. stay the same
            argv,
            np.where(
                expr_upper < 1,
                # must be non-infinite
                fmax,
                # can be infinite or non-infinite
                X.dtype.type(np.inf),
            ),
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
        # isinf cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"isinf({self._a!r})"


class ScalarIsNaN(Expr):
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
        return ScalarIsNaN(self._a.apply_array_element_offset(axis, offset))

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: np.astype(_isnan(x), dtype),  # type: ignore
            ScalarIsNaN,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.astype(_isnan(self._a.eval(x, Xs, late_bound)), Xs.dtype)  # type: ignore

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)

        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_lower > 0,
            # must be NaN, i.e. stay the same
            argv,
            np.where(
                expr_upper < 1,
                # must be non-NaN
                X.dtype.type(-np.inf),
                # can be NaN or non-NaN
                X.dtype.type(-np.inf),
            ),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_lower > 0,
            # must be NaN, i.e. stay the same
            argv,
            np.where(
                expr_upper < 1,
                # must be non-NaN
                X.dtype.type(np.inf),
                # can be NaN or non-NaN
                X.dtype.type(np.inf),
            ),
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
        # isnan cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"isnan({self._a!r})"

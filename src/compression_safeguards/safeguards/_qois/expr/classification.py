from collections.abc import Callable, Mapping
from typing import overload

import numpy as np

from ....utils._compat import _astype, _floating_max
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Fi, Ns, Ps, PsI


class ScalarIsFinite(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarIsFinite":
        return ScalarIsFinite(a)

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: classify_to_dtype(np.isfinite, x, dtype),  # type: ignore
            ScalarIsFinite,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return classify_to_dtype(np.isfinite, self._a.eval(x, Xs, late_bound), Xs.dtype)

    @checked_data_bounds
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

        fmax = _floating_max(X.dtype)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, isfinite(Xs) = True, so Xs must be finite
        # if expr_upper < 1, isfinite(Xs) = False, Xs must stay non-finite
        # otherwise, isfinite(Xs) in [True, False] and Xs can be anything
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.full(X.shape, -fmax)
        arg_lower[np.less_equal(expr_lower, 0)] = -np.inf
        np.copyto(arg_lower, argv, where=np.less(expr_upper, 1), casting="no")

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.full(X.shape, fmax)
        arg_upper[np.less_equal(expr_lower, 0)] = np.inf
        np.copyto(arg_upper, argv, where=np.less(expr_upper, 1), casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"isfinite({self._a!r})"


class ScalarIsInf(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarIsInf":
        return ScalarIsInf(a)

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: classify_to_dtype(np.isinf, x, dtype),
            ScalarIsInf,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return classify_to_dtype(np.isinf, self._a.eval(x, Xs, late_bound), Xs.dtype)

    @checked_data_bounds
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

        fmax = _floating_max(X.dtype)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, isinf(Xs) = True, so Xs must stay infinite
        # if expr_upper < 0, isinf(Xs) = False, Xs must be non-infinite
        # otherwise, isinf(Xs) in [True, False] and Xs can be anything
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.full(X.shape, -fmax)
        arg_lower[np.greater_equal(expr_upper, 1)] = -np.inf
        np.copyto(arg_lower, argv, where=np.greater(expr_lower, 0), casting="no")

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.full(X.shape, fmax)
        arg_upper[np.greater_equal(expr_upper, 1)] = np.inf
        np.copyto(arg_upper, argv, where=np.greater(expr_lower, 0), casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"isinf({self._a!r})"


class ScalarIsNaN(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarIsNaN":
        return ScalarIsNaN(a)

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: classify_to_dtype(np.isnan, x, dtype),  # type: ignore
            ScalarIsNaN,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return classify_to_dtype(np.isnan, self._a.eval(x, Xs, late_bound), Xs.dtype)

    @checked_data_bounds
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

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, isnan(Xs) = True, so Xs must stay NaN
        # if expr_upper < 0, isnan(Xs) = False, Xs must be non-NaN
        # otherwise, isnan(Xs) in [True, False] and Xs can be anything
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.full(X.shape, X.dtype.type(-np.inf))
        np.copyto(arg_lower, argv, where=np.greater(expr_lower, 0), casting="no")

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.full(X.shape, X.dtype.type(np.inf))
        np.copyto(arg_upper, argv, where=np.greater(expr_lower, 0), casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"isnan({self._a!r})"


@overload
def classify_to_dtype(
    classify: Callable[
        [np.ndarray[Ps, np.dtype[F]]], np.ndarray[Ps, np.dtype[np.bool]]
    ],
    a: np.ndarray[Ps, np.dtype[F]],
    dtype: np.dtype[F],
) -> np.ndarray[Ps, np.dtype[F]]: ...


@overload
def classify_to_dtype(
    classify: Callable[[Fi], bool],
    a: Fi,
    dtype: np.dtype[Fi],
) -> Fi: ...


def classify_to_dtype(classify, a, dtype):
    c = classify(a)

    if not isinstance(c, np.ndarray):
        return _astype(np.array(c, copy=None), dtype, casting="safe")[()]

    return _astype(c, dtype, casting="safe")

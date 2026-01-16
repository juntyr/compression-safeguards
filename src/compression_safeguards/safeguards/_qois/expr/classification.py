from collections.abc import Callable, Mapping
from typing import overload

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import _ensure_array
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from ..typing import F, Fi, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant


class ScalarIsFinite(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarIsFinite":
        return ScalarIsFinite(a)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: classify_to_dtype(np.isfinite, x, dtype),
            ScalarIsFinite,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return classify_to_dtype(np.isfinite, self._a.eval(Xs, late_bound), Xs.dtype)

    @override
    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(Xs, late_bound)

        fmax = np.finfo(Xs.dtype).max

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, isfinite(arg) = True, so arg must be finite
        # if expr_upper < 1, isfinite(arg) = False, arg must stay non-finite
        # otherwise, isfinite(arg) in [True, False] and arg can be anything
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(Xs.shape[:1], -fmax)
        arg_lower[np.less_equal(expr_lower, 0)] = -np.inf
        np.copyto(arg_lower, argv, where=np.less(expr_upper, 1), casting="no")

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(Xs.shape[:1], fmax)
        arg_upper[np.less_equal(expr_lower, 0)] = np.inf
        np.copyto(arg_upper, argv, where=np.less(expr_upper, 1), casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"isfinite({self._a!r})"


class ScalarIsInf(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarIsInf":
        return ScalarIsInf(a)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: classify_to_dtype(np.isinf, x, dtype),
            ScalarIsInf,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return classify_to_dtype(np.isinf, self._a.eval(Xs, late_bound), Xs.dtype)

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(Xs, late_bound)

        fmax = np.finfo(Xs.dtype).max

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, isinf(arg) = True, so arg must stay infinite
        # if expr_upper < 0, isinf(arg) = False, arg must be non-infinite
        # otherwise, isinf(arg) in [True, False] and arg can be anything
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(Xs.shape[:1], -fmax)
        arg_lower[np.greater_equal(expr_upper, 1)] = -np.inf
        np.copyto(arg_lower, argv, where=np.greater(expr_lower, 0), casting="no")

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(Xs.shape[:1], fmax)
        arg_upper[np.greater_equal(expr_upper, 1)] = np.inf
        np.copyto(arg_upper, argv, where=np.greater(expr_lower, 0), casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"isinf({self._a!r})"


class ScalarIsNaN(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarIsNaN":
        return ScalarIsNaN(a)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            lambda x: classify_to_dtype(np.isnan, x, dtype),
            ScalarIsNaN,
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return classify_to_dtype(np.isnan, self._a.eval(Xs, late_bound), Xs.dtype)

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(Xs, late_bound)

        # by the precondition, expr_lower <= self.eval(Xs) <= expr_upper
        # if expr_lower > 0, isnan(arg) = True, so arg must stay NaN
        # if expr_upper < 0, isnan(arg) = False, arg must be non-NaN
        # otherwise, isnan(arg) in [True, False] and arg can be anything
        arg_lower: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
            Xs.shape[:1], Xs.dtype.type(-np.inf)
        )
        np.copyto(arg_lower, argv, where=np.greater(expr_lower, 0), casting="no")

        arg_upper: np.ndarray[tuple[Ps], np.dtype[F]] = np.full(
            Xs.shape[:1], Xs.dtype.type(np.inf)
        )
        np.copyto(arg_upper, argv, where=np.greater(expr_lower, 0), casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"isnan({self._a!r})"


@overload
def classify_to_dtype(
    classify: Callable[
        [np.ndarray[tuple[Ps], np.dtype[F]]], np.ndarray[tuple[Ps], np.dtype[np.bool]]
    ],
    a: np.ndarray[tuple[Ps], np.dtype[F]],
    dtype: np.dtype[F],
) -> np.ndarray[tuple[Ps], np.dtype[F]]: ...


@overload
def classify_to_dtype(
    classify: Callable[[Fi], bool],
    a: Fi,
    dtype: np.dtype[Fi],
) -> Fi: ...


def classify_to_dtype(classify, a, dtype):
    c = classify(a)

    if not isinstance(c, np.ndarray):
        return _ensure_array(c).astype(dtype, casting="safe")[()]

    return c.astype(dtype, casting="safe")

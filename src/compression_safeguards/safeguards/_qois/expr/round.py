from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _is_negative_zero,
    _is_positive_zero,
    _is_sign_negative_number,
    _is_sign_positive_number,
    _nextafter,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarFloor(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarFloor":
        return ScalarFloor(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.floor, ScalarFloor
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.floor(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        arg = self._a

        # compute the rounded result that meets the expr bounds
        # rounding uses integer steps, so e.g. floor(...) in [-3.1, 4.7] means
        #  that floor(...) in [-3, 4]
        expr_lower = np.ceil(expr_lower)
        expr_upper = np.floor(expr_upper)

        # compute the arg bound that will round to meet the expr bounds
        # floor rounds down
        # if expr_lower is -0.0, ceil(-0.0) = -0.0, only floor(-0.0) = -0.0
        # if expr_lower is +0.0, ceil(+0.0) = +0.0, there is nothing below +0.0
        #  for which floor(...) = +0.0
        # if expr_upper is -0.0, floor(-0.0) = -0.0, we need to force arg_upper
        #  to -0.0
        # if expr_upper is +0.0, floor(+0.0) = +0.0, and floor(1-eps) = +0.0
        arg_lower: np.ndarray[Ps, np.dtype[F]] = expr_lower
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(
            _nextafter(expr_upper + 1, expr_upper), copy=None
        )
        arg_upper[_is_negative_zero(expr_upper)] = -0.0

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"floor({self._a!r})"


class ScalarCeil(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarCeil":
        return ScalarCeil(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.ceil, ScalarCeil
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.ceil(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        arg = self._a

        # compute the rounded result that meets the expr bounds
        # rounding uses integer steps, so e.g. ceil(...) in [-3.1, 4.7] means
        #  that ceil(...) in [-3, 4]
        expr_lower = np.ceil(expr_lower)
        expr_upper = np.floor(expr_upper)

        # compute the arg bound that will round to meet the expr bounds
        # ceil rounds up
        # if expr_lower is -0.0, ceil(-0.0) = -0.0, and ceil(-1+eps) = -0.0
        # if expr_lower is +0.0, ceil(+0.0) = +0.0, we need to force arg_lower
        #  to +0.0
        # if expr_upper is -0.0, floor(-0.0) = -0.0, there is nothing above
        #  -0.0 for which ceil(...) = -0.0
        # if expr_upper is +0.0, floor(+0.0) = +0.0, only ceil(+0.0) = +0.0
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(
            _nextafter(expr_lower - 1, expr_lower), copy=None
        )
        arg_lower[_is_positive_zero(expr_lower)] = +0.0
        arg_upper: np.ndarray[Ps, np.dtype[F]] = expr_upper

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"ceil({self._a!r})"


class ScalarTrunc(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarTrunc":
        return ScalarTrunc(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.trunc, ScalarTrunc
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.trunc(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        arg = self._a

        # compute the rounded result that meets the expr bounds
        # rounding uses integer steps, so e.g. trunc(...) in [-3.1, 4.7] means
        #  that trunc(...) in [-3, 4]
        expr_lower = np.ceil(expr_lower)
        expr_upper = np.floor(expr_upper)

        # compute the arg bound that will round to meet the expr bounds
        # trunc rounds towards zero
        # if expr_lower is -0.0, ceil(-0.0) = -0.0, and trunc(-1+eps) = -0.0
        # if expr_lower is +0.0, ceil(+0.0) = +0.0, there is nothing below +0.0
        #  for which trunc(...) = +0.0
        # if expr_upper is -0.0, floor(-0.0) = -0.0, there is nothing above
        #  -0.0 for which trunc(...) = -0.0
        # if expr_upper is +0.0, floor(+0.0) = +0.0, and trunc(1-eps) = +0.0
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(
            _nextafter(expr_lower - 1, expr_lower), copy=None
        )
        np.copyto(
            arg_lower,
            expr_lower,
            where=_is_sign_positive_number(expr_lower),
            casting="no",
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(
            _nextafter(expr_upper + 1, expr_upper), copy=None
        )
        np.copyto(
            arg_upper,
            expr_upper,
            where=_is_sign_negative_number(expr_upper),
            casting="no",
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"trunc({self._a!r})"


class ScalarRoundTiesEven(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarRoundTiesEven":
        return ScalarRoundTiesEven(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            np.rint,
            ScalarRoundTiesEven,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.rint(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg and round_ties_even(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.rint(argv)

        # compute the rounded result that meets the expr bounds
        # rounding uses integer steps, so e.g. round_ties_even(...) in
        #  [-3.1, 4.7] means that round_ties_even(...) in [-3, 4]
        expr_lower = np.ceil(expr_lower)
        expr_upper = np.floor(expr_upper)

        # estimate the arg bound that will round to meet the expr bounds
        # round_ties_even rounds to the nearest integer, with tie breaks
        #  towards the nearest *even* integer
        # if expr_lower is -0.0, ceil(-0.0) = -0.0, and rint(-0.5) = -0.0
        # if expr_lower is +0.0, ceil(+0.0) = +0.0, we need to force arg_lower
        #  to +0.0
        # if expr_upper is -0.0, floor(-0.0) = -0.0, we need to force arg_upper
        #  to -0.0
        # if expr_upper is +0.0, floor(+0.0) = +0.0, and rint(+0.5) = +0.0
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.subtract(expr_lower, X.dtype.type(0.5)), copy=None
        )
        arg_lower[_is_positive_zero(expr_lower)] = +0.0
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.add(expr_upper, X.dtype.type(0.5)), copy=None
        )
        arg_upper[_is_negative_zero(expr_upper)] = -0.0

        # handle rounding errors in round_ties_even(...) early
        # which can occur since our +-0.5 estimate doesn't account for the even
        #  tie breaking cases
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: np.rint(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: np.rint(arg_upper),
            exprv,
            argv,
            arg_upper,
            expr_lower,
            expr_upper,
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"round_ties_even({self._a!r})"

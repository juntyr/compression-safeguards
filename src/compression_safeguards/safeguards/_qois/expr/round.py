from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _is_negative,
    _is_positive,
    _nextafter,
    _reciprocal,
    _rint,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarFloor(Expr):
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
        return ScalarFloor(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        def is_negative_zero(
            x: np.ndarray[Ps, np.dtype[F]],
        ) -> np.ndarray[Ps, np.dtype[np.bool]]:
            return (x == 0) & (_reciprocal(x) < 0)

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
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            is_negative_zero(expr_upper),
            X.dtype.type(-0.0),
            _nextafter(expr_upper + 1, expr_upper),
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
        # floor cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
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
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarCeil(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        def is_positive_zero(
            x: np.ndarray[Ps, np.dtype[F]],
        ) -> np.ndarray[Ps, np.dtype[np.bool]]:
            return (x == 0) & (_reciprocal(x) > 0)

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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            is_positive_zero(expr_lower),
            X.dtype.type(+0.0),
            _nextafter(expr_lower - 1, expr_lower),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = expr_upper

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
        # ceil cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
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
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarTrunc(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            _is_negative(expr_lower), _nextafter(expr_lower - 1, expr_lower), expr_lower
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            _is_positive(expr_upper), _nextafter(expr_upper + 1, expr_upper), expr_upper
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
        # trunc cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
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
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarRoundTiesEven(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a,
            dtype,
            _rint,  # type: ignore
            ScalarRoundTiesEven,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return _rint(self._a.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        def is_positive_zero(
            x: np.ndarray[Ps, np.dtype[F]],
        ) -> np.ndarray[Ps, np.dtype[np.bool]]:
            return (x == 0) & (_reciprocal(x) > 0)

        def is_negative_zero(
            x: np.ndarray[Ps, np.dtype[F]],
        ) -> np.ndarray[Ps, np.dtype[np.bool]]:
            return (x == 0) & (_reciprocal(x) < 0)

        # evaluate arg and round_ties_even(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _rint(argv)

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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            is_positive_zero(expr_lower),
            X.dtype.type(+0.0),
            np.subtract(expr_lower, X.dtype.type(0.5)),
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            is_negative_zero(expr_upper),
            X.dtype.type(-0.0),
            np.add(expr_upper, X.dtype.type(0.5)),
        )

        # handle rounding errors in round_ties_even(...) early
        # which can occur since our +-0.5 estimate doesn't account for the even
        #  tie breaking cases
        arg_lower = guarantee_arg_within_expr_bounds(
            lambda arg_lower: _rint(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = guarantee_arg_within_expr_bounds(
            lambda arg_upper: _rint(arg_upper),
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

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for
        #  round_ties_even, which is strictly monotonic
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"round_ties_even({self._a!r})"

from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _nan_to_zero_inf_to_finite, _nextafter, _rint
from ..eb import ensure_bounded_derived_error, ensure_bounded_expression
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # evaluate arg and floor(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
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
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.floor(argv + eau),
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
        # the unchecked method already handles rounding errors for floor,
        #  which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

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
        arg_lower = expr_lower
        arg_upper = _nextafter(expr_upper + 1, expr_upper)

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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # evaluate arg and ceil(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
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
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.ceil(argv + eau),
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
        # the unchecked method already handles rounding errors for ceil,
        #  which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

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
        arg_lower = _nextafter(expr_lower - 1, expr_lower)
        arg_upper = expr_upper

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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # evaluate arg and trunc(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = np.trunc(argv)

        # compute the truncated result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will truncate to meet the error bounds
        argv_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            exprv_lower <= 0, _nextafter(exprv_lower - 1, exprv_lower), exprv_lower
        )
        argv_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            exprv_upper >= 0, _nextafter(exprv_upper + 1, exprv_upper), exprv_upper
        )

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal: np.ndarray[Ps, np.dtype[F]] = np.minimum(np.subtract(argv_lower, argv), 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau: np.ndarray[Ps, np.dtype[F]] = np.maximum(0, np.subtract(argv_upper, argv))
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in trunc(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: np.trunc(np.add(argv, eal)),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: np.trunc(np.add(argv, eau)),
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
        # the unchecked method already handles rounding errors for trunc,
        #  which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_lower <= 0, _nextafter(expr_lower - 1, expr_lower), expr_lower
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            expr_upper >= 0, _nextafter(expr_upper + 1, expr_upper), expr_upper
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # evaluate arg and round_ties_even(arg)
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)
        exprv = _rint(argv)

        # compute the rounded result that meets the error bounds
        exprv_lower = np.trunc(exprv + eb_expr_lower)
        exprv_upper = np.trunc(exprv + eb_expr_upper)

        # compute the argv that will round to meet the error bounds
        argv_lower = exprv_lower - 0.5
        argv_upper = exprv_upper + 0.5

        # update the error bounds
        # rounding allows zero error bounds on the expression to expand into
        #  non-zero error bounds on the argument
        eal: np.ndarray[Ps, np.dtype[F]] = np.minimum(np.subtract(argv_lower, argv), 0)
        eal = _nan_to_zero_inf_to_finite(eal)

        eau: np.ndarray[Ps, np.dtype[F]] = np.maximum(0, np.subtract(argv_upper, argv))
        eau = _nan_to_zero_inf_to_finite(eau)

        # handle rounding errors in round_ties_even(...) early
        eal = ensure_bounded_derived_error(
            lambda eal: _rint(argv + eal),
            exprv,
            argv,
            eal,
            eb_expr_lower,
            eb_expr_upper,
        )
        eau = ensure_bounded_derived_error(
            lambda eau: _rint(argv + eau),
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
        # the unchecked method already handles rounding errors for
        #  round_ties_even, which is weakly monotonic
        return self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

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
        exprv = _rint(argv)

        # compute the rounded result that meets the expr bounds
        # rounding uses integer steps, so e.g. round_ties_even(...) in
        #  [-3.1, 4.7] means that round_ties_even(...) in [-3, 4]
        expr_lower = np.ceil(expr_lower)
        expr_upper = np.floor(expr_upper)

        # estimate the arg bound that will round to meet the expr bounds
        # round_ties_even rounds to the nearest integer, with tie breaks
        #  towards the nearest *even* integer
        arg_lower = np.subtract(expr_lower, 0.5)
        arg_upper = np.add(expr_upper, 0.5)

        # handle rounding errors in round_ties_even(...) early
        # which can occur since our +-0.5 estimate doesn't account for the even
        #  tie breaking cases
        # FIXME: https://github.com/numpy/numpy-user-dtypes/issues/129
        arg_lower = ensure_bounded_expression(
            lambda arg_lower: _rint(arg_lower),
            exprv,
            argv,
            arg_lower,
            expr_lower,
            expr_upper,
        )
        arg_upper = ensure_bounded_expression(
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

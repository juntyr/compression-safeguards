import operator
from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import (
    _float128_dtype,
    _float128_max,
    _isfinite,
    _isnan,
    _nan_to_zero,
)
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .abs import ScalarAbs
from .constfold import ScalarFoldedConstant
from .literal import Number
from .typing import F, Ns, Ps, PsI


class ScalarAdd(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __new__(cls, a: Expr, b: Expr):
        ab = Number.symbolic_fold_binary(a, b, operator.add)
        if ab is not None:
            return ab
        this = super(ScalarAdd, cls).__new__(cls)
        this._a = a
        this._b = b
        return this

    @property
    def has_data(self) -> bool:
        return self._a.has_data or self._b.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices | self._b.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarAdd(
            self._a.apply_array_element_offset(axis, offset),
            self._b.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.add, ScalarAdd
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.add(self._a.eval(x, Xs, late_bound), self._b.eval(x, Xs, late_bound))

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        terms = as_left_associative_sum(self)
        return compute_left_associate_sum_data_bounds(
            terms, expr_lower, expr_upper, X, Xs, late_bound
        )

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for add
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._a!r} + {self._b!r}"


class ScalarSubtract(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __new__(cls, a: Expr, b: Expr):
        ab = Number.symbolic_fold_binary(a, b, operator.sub)
        if ab is not None:
            return ab
        this = super(ScalarSubtract, cls).__new__(cls)
        this._a = a
        this._b = b
        return this

    @property
    def has_data(self) -> bool:
        return self._a.has_data or self._b.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices | self._b.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarSubtract(
            self._a.apply_array_element_offset(axis, offset),
            self._b.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.subtract, ScalarSubtract
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.subtract(
            self._a.eval(x, Xs, late_bound), self._b.eval(x, Xs, late_bound)
        )

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        terms = as_left_associative_sum(self)
        return compute_left_associate_sum_data_bounds(
            terms, expr_lower, expr_upper, X, Xs, late_bound
        )

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for subtract
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._a!r} - {self._b!r}"


def as_left_associative_sum(
    expr: ScalarAdd | ScalarSubtract,
) -> tuple[Expr, ...]:
    from .neg import ScalarNegate

    terms_rev: list[Expr] = []

    while True:
        # rewrite ( a - b ) as ( a + (-b) ), which is equivalent for
        #  floating point numbers
        terms_rev.append(
            ScalarNegate(expr._b) if isinstance(expr, ScalarSubtract) else expr._b
        )

        if isinstance(expr._a, (ScalarAdd, ScalarSubtract)):
            expr = expr._a
        else:
            terms_rev.append(expr._a)
            break

    return tuple(terms_rev[::-1])


def get_expr_left_associative_abs_factor_approximate(expr: Expr) -> None | Expr:
    from .divmul import ScalarDivide, ScalarMultiply

    if not expr.has_data:
        return None

    if not isinstance(expr, (ScalarMultiply, ScalarDivide)):
        return Number.ONE

    factor_stack: list[tuple[Expr, type[ScalarMultiply] | type[ScalarDivide]]] = []

    while True:
        factor_stack.append((expr._b, type(expr)))

        if isinstance(expr._a, (ScalarMultiply, ScalarDivide)):
            expr = expr._a
        else:
            factor_stack.append(
                (Number.ONE if expr._a.has_data else expr._a, ScalarMultiply)
            )
            break

    while len(factor_stack) > 1:
        (a, _), (b, ty) = factor_stack.pop(), factor_stack.pop()
        factor_stack.append(
            (
                ty(a, Number.ONE if b.has_data else b),
                ScalarMultiply,
            )
        )

    [(factor, _)] = factor_stack

    return ScalarAbs(factor)


def compute_left_associate_sum_data_bounds(
    left_associative_sum: tuple[Expr, ...],
    expr_lower: np.ndarray[Ps, np.dtype[F]],
    expr_upper: np.ndarray[Ps, np.dtype[F]],
    X: np.ndarray[Ps, np.dtype[F]],
    Xs: np.ndarray[Ns, np.dtype[F]],
    late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
    def _zero_add(
        a: np.ndarray[Ps, np.dtype[F]], b: np.ndarray[Ps, np.dtype[F]]
    ) -> np.ndarray[Ps, np.dtype[F]]:
        return np.where(b == 0, a, a + b)  # type: ignore

    termvs: list[np.ndarray[Ps, np.dtype[F]]] = []
    abs_factorvs: list[None | np.ndarray[Ps, np.dtype[F]]] = []

    for term in left_associative_sum:
        abs_factor = get_expr_left_associative_abs_factor_approximate(term)
        termvs.append(term.eval(X.shape, Xs, late_bound))
        abs_factorvs.append(
            None if abs_factor is None else abs_factor.eval(X.shape, Xs, late_bound)
        )

    # evaluate the total expression sum
    exprv: np.ndarray[Ps, np.dtype[F]] = np.array(sum(termvs[1:], start=termvs[0]))
    expr_lower = np.array(expr_lower)
    expr_upper = np.array(expr_upper)

    # compute the sum of absolute factors
    total_abs_factor_: None | np.ndarray[Ps, np.dtype[F]] = None
    for abs_factorv in abs_factorvs:
        if abs_factorv is None:
            continue
        if total_abs_factor_ is None:
            total_abs_factor_ = np.copy(abs_factorv)
        else:
            total_abs_factor_ += abs_factorv
    assert total_abs_factor_ is not None
    total_abs_factor: np.ndarray[Ps, np.dtype[F]] = total_abs_factor_

    # drop into expression difference bounds to divide up the bound
    # for NaN sums, we use a zero difference to ensure NaNs don't
    #  accidentally propagate into the term error bounds
    expr_lower_diff: np.ndarray[Ps, np.dtype[F]] = _nan_to_zero(expr_lower - exprv)
    expr_upper_diff: np.ndarray[Ps, np.dtype[F]] = _nan_to_zero(expr_upper - exprv)

    # if exprv is NaN, expr_[lower|upper]_diff are zero
    # if expr_[lower|upper] is infinite but exprv is finite,
    #  expr_[lower|upper]_diff is the same infinity as expr_[lower|upper]
    # if expr_[lower|upper] is infinite and exprv is infinite,
    #  (a) -inf - -inf and inf - inf are NaN, so expr_[lower|upper]_diff is
    #      zero
    #  (b) -inf - inf and inf - -inf are +-inf, so expr_[lower|upper]_diff is
    #      the same infinity as expr_[lower|upper]
    tfl: np.ndarray[Ps, np.dtype[F]] = expr_lower_diff / total_abs_factor
    tfu: np.ndarray[Ps, np.dtype[F]] = expr_upper_diff / total_abs_factor

    fmax = _float128_max if X.dtype == _float128_dtype else np.finfo(X.dtype).max

    # ensure that the bounds never contain both -inf and +inf since that would
    #  allow NaN to sneak in
    inf_clash = (tfl == -np.inf) & (tfu == np.inf)
    tfl = np.where(inf_clash, -fmax, tfl)  # type: ignore
    tfu = np.where(inf_clash, fmax, tfu)  # type: ignore

    any_nan: np.ndarray[Ps, np.dtype[np.bool]] = _isnan(termvs[0])  # type: ignore
    for termv in termvs[1:]:
        any_nan |= _isnan(termv)

    # stack the lower and upper bounds for each term factor
    # if total_abs_factor or exprv is non-finite:
    #  - non-finite values must be preserved exactly
    #  - if there is any NaN term, finite values can have any finite value
    #  - otherwise finite values are also preserved exactly, for simplicity
    # if total_abs_factor is zero, all abs_factorv are also zero and we can
    #  allow the finite terms terms to have any finite value
    # otherwise we split up the error bound for the terms by their factors
    tl_stack = np.stack(
        [
            np.where(
                ~_isfinite(total_abs_factor) | ~_isfinite(exprv),
                np.where(_isfinite(termv) & any_nan, -fmax, termv),
                np.where(
                    total_abs_factor == 0,
                    -fmax,
                    _zero_add(termv, tfl * abs_factorv),
                ),
            )
            for termv, abs_factorv in zip(termvs, abs_factorvs)
            if abs_factorv is not None
        ]
    )
    tu_stack = np.stack(
        [
            np.where(
                ~_isfinite(total_abs_factor) | ~_isfinite(exprv),
                np.where(_isfinite(termv) & any_nan, fmax, termv),
                np.where(
                    total_abs_factor == 0,
                    fmax,
                    _zero_add(termv, tfu * abs_factorv),
                ),
            )
            for termv, abs_factorv in zip(termvs, abs_factorvs)
            if abs_factorv is not None
        ]
    )

    def compute_term_sum(
        t_stack: np.ndarray[tuple[int, ...], np.dtype[F]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
        total_sum: None | np.ndarray[tuple[int, ...], np.dtype[F]] = None
        i = 0

        for termv, abs_factorv in zip(termvs, abs_factorvs):
            if total_sum is None:
                if abs_factorv is None:
                    total_sum = np.copy(termv)
                else:
                    total_sum = np.copy(t_stack[i])
            elif abs_factorv is None:
                total_sum += termv
            else:
                total_sum += t_stack[i]
            i += abs_factorv is not None

        assert total_sum is not None

        return np.broadcast_to(  # type: ignore
            np.array(total_sum).reshape((1,) + exprv.shape),
            (t_stack.shape[0],) + exprv.shape,
        )

    # handle rounding errors in the total absolute factor early
    tl_stack = guarantee_arg_within_expr_bounds(
        compute_term_sum,
        np.broadcast_to(
            exprv.reshape((1,) + exprv.shape), (tl_stack.shape[0],) + exprv.shape
        ),
        np.stack(
            [
                termv
                for termv, abs_factorv in zip(termvs, abs_factorvs)
                if abs_factorv is not None
            ]
        ),
        tl_stack,
        np.broadcast_to(
            expr_lower.reshape((1,) + exprv.shape),
            (tl_stack.shape[0],) + exprv.shape,
        ),
        np.broadcast_to(
            expr_upper.reshape((1,) + exprv.shape),
            (tl_stack.shape[0],) + exprv.shape,
        ),
    )
    tu_stack = guarantee_arg_within_expr_bounds(
        compute_term_sum,
        np.broadcast_to(
            exprv.reshape((1,) + exprv.shape), (tu_stack.shape[0],) + exprv.shape
        ),
        np.stack(
            [
                termv
                for termv, abs_factorv in zip(termvs, abs_factorvs)
                if abs_factorv is not None
            ]
        ),
        tu_stack,
        np.broadcast_to(
            expr_lower.reshape((1,) + exprv.shape),
            (tu_stack.shape[0],) + exprv.shape,
        ),
        np.broadcast_to(
            expr_upper.reshape((1,) + exprv.shape),
            (tu_stack.shape[0],) + exprv.shape,
        ),
    )

    xl: np.ndarray[Ns, np.dtype[F]]
    xu: np.ndarray[Ns, np.dtype[F]]
    Xs_lower_: None | np.ndarray[Ns, np.dtype[F]] = None
    Xs_upper_: None | np.ndarray[Ns, np.dtype[F]] = None
    i = 0
    for term, abs_factorv in zip(left_associative_sum, abs_factorvs):
        if abs_factorv is None:
            continue

        # recurse into the terms with a weighted bound
        xl, xu = term.compute_data_bounds(
            tl_stack[i],
            tu_stack[i],
            X,
            Xs,
            late_bound,
        )

        # combine the inner data bounds
        if Xs_lower_ is None:
            Xs_lower_ = xl
        else:
            Xs_lower_ = np.maximum(Xs_lower_, xl)
        if Xs_upper_ is None:
            Xs_upper_ = xu
        else:
            Xs_upper_ = np.minimum(Xs_upper_, xu)

        i += 1

    assert Xs_lower_ is not None
    assert Xs_upper_ is not None
    Xs_lower: np.ndarray[Ns, np.dtype[F]] = Xs_lower_
    Xs_upper: np.ndarray[Ns, np.dtype[F]] = Xs_upper_

    Xs_lower = np.minimum(Xs_lower, Xs)
    Xs_upper = np.maximum(Xs_upper, Xs)

    Xs_lower = np.where(Xs_lower == Xs, Xs, Xs_lower)  # type: ignore
    Xs_upper = np.where(Xs_upper == Xs, Xs, Xs_upper)  # type: ignore

    return Xs_lower, Xs_upper

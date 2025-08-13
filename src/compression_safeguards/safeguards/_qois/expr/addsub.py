import operator
from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import (
    _float128_dtype,
    _float128_max,
    _isinf,
    _isnan,
    _nan_to_zero,
    _nan_to_zero_inf_to_finite,
)
from ..eb import ensure_bounded_derived_error, ensure_bounded_expression
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        terms = SumTerm.as_left_associative_sum(self)
        return SumTerm.compute_data_error_bound(
            terms, eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for add
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
        terms = SumTerm.as_left_associative_sum(self)
        return SumTerm.compute_data_bounds(
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

    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        terms = SumTerm.as_left_associative_sum(self)
        return SumTerm.compute_data_error_bound(
            terms, eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        # the unchecked method already handles rounding errors for subtract
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
        terms = SumTerm.as_left_associative_sum(self)
        return SumTerm.compute_data_bounds(
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


class SumTerm:
    __slots__ = ("_expr", "_is_add")
    _expr: Expr
    _is_add: bool

    def __init__(self, expr: Expr, is_add: bool):
        self._expr = expr
        self._is_add = is_add

    @staticmethod
    def as_left_associative_sum(
        expr: ScalarAdd | ScalarSubtract,
    ) -> "tuple[SumTerm, ...]":
        terms_rev: list[SumTerm] = []

        while True:
            terms_rev.append(SumTerm(expr._b, isinstance(expr, ScalarAdd)))

            if isinstance(expr._a, (ScalarAdd, ScalarSubtract)):
                expr = expr._a
            else:
                terms_rev.append(SumTerm(expr._a, True))
                break

        return tuple(terms_rev[::-1])

    @staticmethod
    def get_expr_left_associative_abs_factor_approximate(expr: Expr) -> None | Expr:
        from .divmul import ScalarDivide, ScalarMultiply

        if not expr.has_data:
            return None

        if not isinstance(expr, (ScalarMultiply, ScalarDivide)):
            return Number.ONE

        factor_stack: list[tuple[Expr, bool]] = []

        while True:
            factor_stack.append((expr._b, isinstance(expr, ScalarDivide)))

            if isinstance(expr._a, (ScalarMultiply, ScalarDivide)):
                expr = expr._a
            else:
                factor_stack.append(
                    (Number.ONE if expr._a.has_data else expr._a, False)
                )
                break

        while len(factor_stack) > 1:
            (a, _), (b, is_div) = factor_stack.pop(), factor_stack.pop()
            factor_stack.append(
                (
                    (ScalarDivide if is_div else ScalarMultiply)(
                        a, Number.ONE if b.has_data else b
                    ),
                    False,
                )
            )

        [(factor, _)] = factor_stack

        return ScalarAbs(factor)

    @staticmethod
    def compute_data_error_bound(
        left_associative_sum: "tuple[SumTerm, ...]",
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        is_adds: list[bool] = []
        termvs: list[np.ndarray[Ps, np.dtype[F]]] = []
        abs_factorvs: list[None | np.ndarray[Ps, np.dtype[F]]] = []

        for term in left_associative_sum:
            abs_factor = SumTerm.get_expr_left_associative_abs_factor_approximate(
                term._expr
            )
            is_adds.append(term._is_add)
            termvs.append(term._expr.eval(X.shape, Xs, late_bound))
            abs_factorvs.append(
                None if abs_factor is None else abs_factor.eval(X.shape, Xs, late_bound)
            )

        # evaluate the total expression sum
        assert is_adds[0]
        exprv: np.ndarray[Ps, np.dtype[F]] = np.copy(termvs[0])
        for is_add, termv in zip(is_adds[1:], termvs[1:]):
            if is_add:
                exprv += termv
            else:
                exprv -= termv

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

        # if total_abs_factor is zero, then all abs_factorv are also zero
        # eb/0 = NaN is converted back to zero, so we just push down zero
        #  error bounds, which is not incorrect
        etl: np.ndarray[Ps, np.dtype[F]] = _nan_to_zero_inf_to_finite(
            eb_expr_lower / total_abs_factor
        )
        etu: np.ndarray[Ps, np.dtype[F]] = _nan_to_zero_inf_to_finite(
            eb_expr_upper / total_abs_factor
        )

        # stack the lower and upper bounds for each term factor
        etl_stack = np.stack(
            [
                etl * abs_factorv
                for abs_factorv in abs_factorvs
                if abs_factorv is not None
            ]
        )
        etu_stack = np.stack(
            [
                etu * abs_factorv
                for abs_factorv in abs_factorvs
                if abs_factorv is not None
            ]
        )

        def compute_sum(et_stack: np.ndarray) -> np.ndarray:
            total_sum = None
            i = 0
            for is_add, termv, abs_factorv in zip(is_adds, termvs, abs_factorvs):
                if total_sum is None:
                    assert is_add
                    if abs_factorv is None:
                        total_sum = np.copy(termv)
                    else:
                        total_sum = termv + et_stack[i]
                elif is_add:
                    if abs_factorv is None:
                        total_sum += termv
                    else:
                        total_sum += termv + et_stack[i]
                elif abs_factorv is None:
                    total_sum -= termv
                else:
                    total_sum -= termv - et_stack[i]
                i += abs_factorv is not None
            assert total_sum is not None
            return total_sum

        # handle rounding errors in the total absolute factor early
        etl_stack = ensure_bounded_derived_error(
            compute_sum,
            exprv,
            np.stack(
                [
                    termv
                    for termv, abs_factorv in zip(termvs, abs_factorvs)
                    if abs_factorv is not None
                ]
            ),
            etl_stack,
            eb_expr_lower,
            eb_expr_upper,
        )
        etu_stack = ensure_bounded_derived_error(
            compute_sum,
            exprv,
            np.stack(
                [
                    termv
                    for termv, abs_factorv in zip(termvs, abs_factorvs)
                    if abs_factorv is not None
                ]
            ),
            etu_stack,
            eb_expr_lower,
            eb_expr_upper,
        )

        exl: np.ndarray[Ps, np.dtype[F]]
        exu: np.ndarray[Ps, np.dtype[F]]
        eb_x_lower: None | np.ndarray[Ps, np.dtype[F]] = None
        eb_x_upper: None | np.ndarray[Ps, np.dtype[F]] = None
        i = 0
        for term, is_add, abs_factorv in zip(
            left_associative_sum, is_adds, abs_factorvs
        ):
            if abs_factorv is None:
                continue

            # recurse into the terms with a weighted error bound
            exl, exu = term._expr.compute_data_error_bound(
                # flip the lower/upper error bound if the term is subtracted
                etl_stack[i] if is_add else -etu_stack[i],
                etu_stack[i] if is_add else -etl_stack[i],
                X,
                Xs,
                late_bound,
            )

            # combine the inner error bounds
            if eb_x_lower is None:
                eb_x_lower = exl
            else:
                eb_x_lower = np.maximum(eb_x_lower, exl)
            if eb_x_upper is None:
                eb_x_upper = exu
            else:
                eb_x_upper = np.minimum(eb_x_upper, exu)

            i += 1

        assert eb_x_lower is not None
        assert eb_x_upper is not None

        return eb_x_lower, eb_x_upper

    @staticmethod
    def compute_data_bounds(
        left_associative_sum: "tuple[SumTerm, ...]",
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

        is_adds: list[bool] = []
        termvs: list[np.ndarray[Ps, np.dtype[F]]] = []
        abs_factorvs: list[None | np.ndarray[Ps, np.dtype[F]]] = []

        for term in left_associative_sum:
            abs_factor = SumTerm.get_expr_left_associative_abs_factor_approximate(
                term._expr
            )
            is_adds.append(term._is_add)
            termvs.append(term._expr.eval(X.shape, Xs, late_bound))
            abs_factorvs.append(
                None if abs_factor is None else abs_factor.eval(X.shape, Xs, late_bound)
            )

        # evaluate the total expression sum
        assert is_adds[0]
        exprv: np.ndarray[Ps, np.dtype[F]] = np.copy(termvs[0])
        for is_add, termv in zip(is_adds[1:], termvs[1:]):
            if is_add:
                exprv += termv
            else:
                exprv -= termv

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

        # if total_abs_factor is zero, all abs_factorv are also zero and we can
        #  allow terms to have any finite value
        # if total_abs_factor is Inf, we cannot do better than a zero error
        #  bound for all terms, making sure that Inf*0 != NaN
        # if total_abs_factor or exprv is NaN, we can allow terms to have any
        #  value, but propagating non-NaN error bounds works better
        # TODO: optimize infinite sums
        tld: np.ndarray[Ps, np.dtype[F]] = expr_lower_diff / total_abs_factor
        tlu: np.ndarray[Ps, np.dtype[F]] = expr_upper_diff / total_abs_factor

        fmax = _float128_max if X.dtype == _float128_dtype else np.finfo(X.dtype).max

        # stack the lower and upper bounds for each term factor
        tl_stack = np.stack(
            [
                np.where(
                    total_abs_factor == 0,
                    -fmax,
                    np.where(
                        _isinf(total_abs_factor),
                        X.dtype.type(0.0),
                        np.where(
                            _isnan(total_abs_factor) | _isnan(exprv),
                            X.dtype.type(-np.inf),
                            _zero_add((termv if is_add else -termv), tld * abs_factorv),
                        ),
                    ),
                )
                for is_add, termv, abs_factorv in zip(is_adds, termvs, abs_factorvs)
                if abs_factorv is not None
            ]
        )
        tu_stack = np.stack(
            [
                np.where(
                    total_abs_factor == 0,
                    fmax,
                    np.where(
                        _isinf(total_abs_factor),
                        X.dtype.type(0.0),
                        np.where(
                            _isnan(total_abs_factor) | _isnan(exprv),
                            X.dtype.type(np.inf),
                            _zero_add((termv if is_add else -termv), tlu * abs_factorv),
                        ),
                    ),
                )
                for is_add, termv, abs_factorv in zip(is_adds, termvs, abs_factorvs)
                if abs_factorv is not None
            ]
        )

        def compute_sum(t_stack: np.ndarray) -> np.ndarray:
            total_sum = None
            i = 0
            for is_add, termv, abs_factorv in zip(is_adds, termvs, abs_factorvs):
                if total_sum is None:
                    assert is_add
                    if abs_factorv is None:
                        total_sum = np.copy(termv)
                    else:
                        total_sum = np.copy(t_stack[i])
                elif is_add:
                    if abs_factorv is None:
                        total_sum += termv
                    else:
                        total_sum += t_stack[i]
                elif abs_factorv is None:
                    total_sum -= termv
                else:
                    # subtract already taken into account above
                    total_sum += t_stack[i]
                i += abs_factorv is not None
            assert total_sum is not None
            return total_sum

        # handle rounding errors in the total absolute factor early
        tl_stack = ensure_bounded_expression(
            compute_sum,
            exprv,
            np.stack(
                [
                    termv
                    for termv, abs_factorv in zip(termvs, abs_factorvs)
                    if abs_factorv is not None
                ]
            ),
            tl_stack,
            expr_lower,
            expr_upper,
        )
        tu_stack = ensure_bounded_expression(
            compute_sum,
            exprv,
            np.stack(
                [
                    termv
                    for termv, abs_factorv in zip(termvs, abs_factorvs)
                    if abs_factorv is not None
                ]
            ),
            tu_stack,
            expr_lower,
            expr_upper,
        )

        xl: np.ndarray[Ns, np.dtype[F]]
        xu: np.ndarray[Ns, np.dtype[F]]
        Xs_lower_: None | np.ndarray[Ns, np.dtype[F]] = None
        Xs_upper_: None | np.ndarray[Ns, np.dtype[F]] = None
        i = 0
        for term, is_add, abs_factorv in zip(
            left_associative_sum, is_adds, abs_factorvs
        ):
            if abs_factorv is None:
                continue

            # recurse into the terms with a weighted bound
            xl, xu = term._expr.compute_data_bounds(
                # flip the lower/upper error bound if the term is subtracted
                tl_stack[i] if is_add else -tu_stack[i],
                tu_stack[i] if is_add else -tl_stack[i],
                X,
                Xs,
                late_bound,
            )

            # combine the inner error bounds
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

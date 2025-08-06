import operator
from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _nan_to_zero_inf_to_finite
from ..eb import ensure_bounded_derived_error
from .abc import Expr
from .constfold import FoldedScalarConst
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
        return FoldedScalarConst.constant_fold_binary(
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
        return FoldedScalarConst.constant_fold_binary(
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
        from .abs import ScalarAbs
        from .divmul import ScalarDivide, ScalarMultiply
        from .literal import Number

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
            termvs.append(
                np.broadcast_to(term._expr.eval(X.shape, Xs, late_bound), X.shape)
            )
            abs_factorvs.append(
                None
                if abs_factor is None
                else np.broadcast_to(abs_factor.eval(X.shape, Xs, late_bound), X.shape)
            )

        # evaluate the total expression sum
        exprv = sum(termvs[1:], start=termvs[0])

        # compute the sum of absolute factors
        # unrolled loop in case a factor is a late-bound constant array
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

        # TODO: what to do about total_abs_factor == 0
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

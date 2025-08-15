from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .typing import F, Ns, Ps, PsI


class ScalarWhere(Expr):
    __slots__ = ("_condition", "_a", "_b")
    _condition: Expr
    _a: Expr
    _b: Expr

    def __init__(self, condition: Expr, a: Expr, b: Expr):
        self._condition = condition
        self._a = a
        self._b = b

    @property
    def has_data(self) -> bool:
        return self._condition.has_data or self._a.has_data or self._b.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return (
            self._condition.data_indices | self._a.data_indices | self._b.data_indices
        )

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarWhere(
            self._condition.apply_array_element_offset(axis, offset),
            self._a.apply_array_element_offset(axis, offset),
            self._b.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return (
            self._condition.late_bound_constants
            | self._a.late_bound_constants
            | self._b.late_bound_constants
        )

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_ternary(
            self._condition,
            self._a,
            self._b,
            dtype,
            np.where,  # type: ignore
            ScalarWhere,
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.where(  # type: ignore
            self._condition.eval(x, Xs, late_bound),
            self._a.eval(x, Xs, late_bound),
            self._b.eval(x, Xs, late_bound),
        )

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate the condition
        cond, a, b = self._condition, self._a, self._b
        condv = cond.eval(X.shape, Xs, late_bound)

        Xs_lower: np.ndarray[Ns, np.dtype[F]] = np.full(Xs.shape, X.dtype.type(-np.inf))
        Xs_upper: np.ndarray[Ns, np.dtype[F]] = np.full(Xs.shape, X.dtype.type(np.inf))

        if cond.has_data:
            # for simplicity, we assume that the condition must always be
            #  preserved exactly
            cl, cu = cond.compute_data_bounds(condv, condv, X, Xs, late_bound)
            Xs_lower = np.maximum(Xs_lower, cl)
            Xs_upper = np.minimum(Xs_upper, cu)

        if np.any(condv) and a.has_data:
            # pass on the data bounds to a but only use its bounds on Xs if
            #  chosen by the condition
            al, au = a.compute_data_bounds(expr_lower, expr_upper, X, Xs, late_bound)

            # combine the data bounds
            Xs_lower = np.where(  # type: ignore
                condv,
                np.maximum(Xs_lower, al),
                Xs_lower,
            )
            Xs_upper = np.where(  # type: ignore
                condv,
                np.minimum(Xs_upper, au),
                Xs_upper,
            )

        if (not np.all(condv)) and b.has_data:
            # pass on the data bounds to b but only use its bounds on Xs if
            #  chosen by the condition
            bl, bu = b.compute_data_bounds(expr_lower, expr_upper, X, Xs, late_bound)

            # combine the data bounds
            Xs_lower = np.where(  # type: ignore
                condv,
                Xs_lower,
                np.maximum(Xs_lower, bl),
            )
            Xs_upper = np.where(  # type: ignore
                condv,
                Xs_upper,
                np.minimum(Xs_upper, bu),
            )

        return Xs_lower, Xs_upper

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # where cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"where({self._condition!r}, {self._a!r}, {self._b!r})"

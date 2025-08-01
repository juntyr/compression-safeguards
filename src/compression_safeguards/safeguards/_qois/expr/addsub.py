from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _nextafter
from ..eb import ensure_bounded_derived_error
from .abc import Expr
from .constfold import FoldedScalarConst
from .neg import ScalarNegate
from .typing import F, Ns, Ps, PsI


class ScalarAdd(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

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
        a_const = not self._a.has_data
        b_const = not self._b.has_data
        assert not (a_const and b_const), "constant sum has no error bounds"

        # TODO: handle weighted sums

        if a_const or b_const:
            term, const = (self._b, self._a) if a_const else (self._a, self._b)

            # evaluate the non-constant and constant term and their sum
            termv = term.eval(X.shape, Xs, late_bound)
            constv = const.eval(X.shape, Xs, late_bound)
            # add of two terms is commutative
            exprv = np.add(termv, constv)

            # handle rounding errors in the addition
            eb_term_lower = ensure_bounded_derived_error(
                lambda etl: (termv + etl) + constv,
                exprv,
                termv,
                eb_expr_lower,
                eb_expr_lower,
                eb_expr_upper,
            )
            eb_term_upper = ensure_bounded_derived_error(
                lambda etu: (termv + etu) + constv,
                exprv,
                termv,
                eb_expr_upper,
                eb_expr_lower,
                eb_expr_upper,
            )

            # composition using Lemma 3 from Jiao et al.
            return term.compute_data_error_bound(
                eb_term_lower,
                eb_term_upper,
                X,
                Xs,
                late_bound,
            )

        # TODO: this is a temporary fix
        eb_expr_lower_half = eb_expr_lower * 0.5
        eb_expr_upper_half = eb_expr_upper * 0.5

        el: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_lower_half + eb_expr_lower_half) < eb_expr_lower,
            _nextafter(eb_expr_lower_half, 0),
            eb_expr_lower_half,
        )
        eu: np.ndarray[Ps, np.dtype[F]] = np.where(  # type: ignore
            (eb_expr_upper_half + eb_expr_upper_half) > eb_expr_upper,
            _nextafter(eb_expr_upper_half, 0),
            eb_expr_upper_half,
        )

        al, au = self._a.compute_data_error_bound(
            el,
            eu,
            X,
            Xs,
            late_bound,
        )
        bl, bu = self._b.compute_data_error_bound(
            el,
            eu,
            X,
            Xs,
            late_bound,
        )

        return np.maximum(al, bl), np.minimum(au, bu)

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

    def __init__(self, a: Expr, b: Expr):
        self._a = a
        self._b = b

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
        # TODO: implement subtract separately
        return ScalarAdd(self._a, ScalarNegate(self._b)).compute_data_error_bound(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
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

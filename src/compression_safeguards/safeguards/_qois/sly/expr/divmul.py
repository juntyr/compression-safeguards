from collections.abc import Mapping

import numpy as np

from .....utils.bindings import Parameter
from .....utils.cast import _nan_to_zero_inf_to_finite
from ...eb import ensure_bounded_derived_error
from .abc import Expr
from .addsub import ScalarAdd
from .constfold import FoldedScalarConst
from .literal import Number
from .logexp import Exponential, Logarithm, ScalarExp, ScalarLog
from .typing import F, Ns, Ps, PsI


class ScalarMultiply(Expr):
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
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.multiply, ScalarMultiply
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.multiply(
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
        a_const = not self._a.has_data
        b_const = not self._b.has_data
        assert not (a_const and b_const), "constant product has no error bounds"

        # TODO: handle larger products

        if a_const or b_const:
            term, const = (self._b, self._a) if a_const else (self._a, self._b)

            # evaluate the non-constant and constant term and their product
            termv = term.eval(X.shape, Xs, late_bound)
            constv = const.eval(X.shape, Xs, late_bound)
            # mul of two terms is commutative
            exprv = np.multiply(termv, constv)

            efl = _nan_to_zero_inf_to_finite(eb_expr_lower / np.abs(constv))
            efu = _nan_to_zero_inf_to_finite(eb_expr_upper / np.abs(constv))

            # flip the lower/upper error bound if the factor is negative
            etl: np.ndarray[Ps, np.dtype[F]] = np.where(constv < 0, -efu, efl)  # type: ignore
            etu: np.ndarray[Ps, np.dtype[F]] = np.where(constv < 0, -efl, efu)  # type: ignore

            # handle rounding errors in the multiplication
            eb_term_lower = ensure_bounded_derived_error(
                lambda etl: (termv + etl) * constv,
                exprv,
                termv,
                etl,
                eb_expr_lower,
                eb_expr_upper,
            )
            eb_term_upper = ensure_bounded_derived_error(
                lambda etu: (termv + etu) * constv,
                exprv,
                termv,
                etu,
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

        # FIXME: this is just a short-term fix
        from .power import ScalarFakeAbs

        return ScalarExp(
            Exponential.exp,
            ScalarAdd(
                ScalarLog(Logarithm.ln, ScalarFakeAbs(self._a)),
                ScalarLog(Logarithm.ln, ScalarFakeAbs(self._b)),
            ),
        ).compute_data_error_bound(
            eb_expr_lower,
            eb_expr_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"{self._a!r} * {self._b!r}"


class ScalarDivide(Expr):
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
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return FoldedScalarConst.constant_fold_binary(
            self._a, self._b, dtype, np.divide, ScalarDivide
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.divide(
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
        # TODO: implement separately
        from .power import ScalarPower

        return ScalarMultiply(
            self._a, ScalarPower(self._b, Number("-1"))
        ).compute_data_error_bound(eb_expr_lower, eb_expr_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"{self._a!r} / {self._b!r}"

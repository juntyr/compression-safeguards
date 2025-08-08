import operator
from collections.abc import Mapping
from math import gcd

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _float128_dtype, _float128_max, _nan_to_zero_inf_to_finite
from ..eb import ensure_bounded_derived_error
from .abc import Expr
from .addsub import ScalarAdd, ScalarSubtract
from .constfold import ScalarFoldedConstant
from .literal import Number
from .logexp import Exponential, Logarithm, ScalarExp, ScalarLog
from .reciprocal import ScalarReciprocal
from .typing import F, Ns, Ps, PsI


class ScalarMultiply(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __new__(cls, a: Expr, b: Expr):
        ab = Number.symbolic_fold_binary(a, b, operator.mul)
        if ab is not None:
            return ab
        this = super(ScalarMultiply, cls).__new__(cls)
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
        return ScalarMultiply(
            self._a.apply_array_element_offset(axis, offset),
            self._b.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_binary(
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

        if a_const or b_const:
            term, const = (self._b, self._a) if a_const else (self._a, self._b)

            # evaluate the non-constant and constant term and their product
            termv = term.eval(X.shape, Xs, late_bound)
            constv = const.eval(X.shape, Xs, late_bound)
            # mul of two terms is commutative
            exprv = np.multiply(termv, constv)

            fmax = (
                _float128_max if X.dtype == _float128_dtype else np.finfo(X.dtype).max
            )

            # for x*0, we can allow any finite x,
            #  which would be more easily expressed with intervals
            efl = _nan_to_zero_inf_to_finite(
                np.where(constv == 0, -fmax - termv, eb_expr_lower / np.abs(constv))
            )
            efu = _nan_to_zero_inf_to_finite(
                np.where(constv == 0, fmax - termv, eb_expr_upper / np.abs(constv))
            )

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

        return rewrite_left_associative_product_as_exp_sum_of_logs(
            self
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

    def __new__(cls, a: Expr, b: Expr):
        if isinstance(a, Number) and isinstance(b, Number):
            # symbolical constant propagation for some cases of int / int
            # division always produces a floating point number
            ai, bi = a.as_int(), b.as_int()
            if (ai is not None) and (bi is not None):
                d = gcd(ai, bi)
                if ai < 0 and bi < 0 and d > 0:
                    # symbolic reduction of -a / -b to a / b
                    d = -d
                if d != 0:
                    # symbolic reduction of (a*d) / (b*d) to a / b
                    assert (ai % d == 0) and (bi % d == 0)
                    ai //= d
                    bi //= d
                # int / 1 has an exact floating point result
                if bi == 1:
                    return Number.from_symbolic_int_as_float(ai)
                # int / -1 has an exact floating point result
                if bi == -1:
                    assert ai >= 0
                    # ensure that 0/1 = 0.0 and 0/-1 = -0.0
                    return Number.from_symbolic_int_as_float(ai, force_negative=True)
                # keep a / b after reduction
                if d != 0:
                    a = Number.from_symbolic_int(ai)
                    b = Number.from_symbolic_int(bi)
        this = super(ScalarDivide, cls).__new__(cls)
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
        return ScalarDivide(
            self._a.apply_array_element_offset(axis, offset),
            self._b.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_binary(
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
        return ScalarMultiply(
            self._a, ScalarReciprocal(self._b)
        ).compute_data_error_bound(eb_expr_lower, eb_expr_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"{self._a!r} / {self._b!r}"


def rewrite_left_associative_product_as_exp_sum_of_logs(
    expr: ScalarMultiply | ScalarDivide,
) -> Expr:
    from .power import ScalarFakeAbs

    terms_stack: list[tuple[Expr, bool]] = []

    while True:
        terms_stack.append((expr._b, isinstance(expr, ScalarMultiply)))

        if isinstance(expr._a, (ScalarMultiply, ScalarDivide)):
            expr = expr._a
        else:
            terms_stack.append((expr._a, True))
            break

    while len(terms_stack) > 1:
        (a, _), (b, is_mul) = terms_stack.pop(), terms_stack.pop()
        terms_stack.append(
            (
                (ScalarAdd if is_mul else ScalarSubtract)(
                    ScalarLog(Logarithm.ln, ScalarFakeAbs(a)),
                    ScalarLog(Logarithm.ln, ScalarFakeAbs(b)),
                ),
                True,
            )
        )

    [(sum_of_lns, _)] = terms_stack

    # rewrite a * b * ... * z as
    #  e^(ln(fake_abs(a)) + ln(fake_abs(b)) + ... + ln(fake_abs(z)))
    # this is mathematically incorrect for any negative product terms but works
    #  for deriving error bounds since fake_abs handles the error bound flips
    return ScalarExp(Exponential.exp, sum_of_lns)

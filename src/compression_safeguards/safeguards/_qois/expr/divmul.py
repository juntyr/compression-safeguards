import operator
from collections.abc import Mapping
from math import gcd

import numpy as np

from ....utils._compat import (
    _floating_max,
    _floating_smallest_subnormal,
    _is_negative,
    _isfinite,
    _isinf,
    _isnan,
    _maximum,
    _minimum,
)
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .addsub import ScalarAdd, ScalarSubtract
from .constfold import ScalarFoldedConstant
from .literal import Number
from .logexp import Exponential, Logarithm, ScalarExp, ScalarLog
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

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        a_const = not self._a.has_data
        b_const = not self._b.has_data
        assert not (a_const and b_const), "constant multiplication has no error bounds"

        # evaluate a and b and a*b
        a, b = self._a, self._b
        av = a.eval(X.shape, Xs, late_bound)
        bv = b.eval(X.shape, Xs, late_bound)
        exprv = np.multiply(av, bv)

        if a_const or b_const:
            term, termv, constv = (b, bv, av) if a_const else (a, av, bv)

            fmax = _floating_max(X.dtype)
            smallest_subnormal = _floating_smallest_subnormal(X.dtype)

            # for x*0, we can allow any finite x
            # for x*Inf, we can allow any non-zero non-NaN x with the same sign
            # for x*NaN, we can allow any x but only propagate [-inf, inf]
            #  since [-NaN, NaN] would be misunderstood as only NaN
            # if term_lower == termv and termv == -0.0, we need to guarantee
            #  that term_lower is also -0.0, same for term_upper
            term_lower: np.ndarray[Ps, np.dtype[F]] = np.array(expr_lower, copy=True)
            np.copyto(term_lower, expr_upper, where=_is_negative(constv), casting="no")
            np.divide(term_lower, constv, out=term_lower)
            term_lower[_isnan(constv)] = -np.inf
            term_lower[_isinf(constv)] = smallest_subnormal
            term_lower[_isinf(constv) & _is_negative(termv)] = -np.inf
            term_lower[constv == 0] = -fmax
            term_lower = _minimum(termv, term_lower)

            term_upper: np.ndarray[Ps, np.dtype[F]] = np.array(expr_upper, copy=True)
            np.copyto(term_upper, expr_lower, where=_is_negative(constv), casting="no")
            np.divide(term_upper, constv, out=term_upper)
            term_upper[_isnan(constv)] = np.inf
            term_upper[_isinf(constv)] = np.inf
            term_upper[_isinf(constv) & _is_negative(termv)] = -smallest_subnormal
            term_upper[constv == 0] = fmax
            term_upper = _maximum(termv, term_upper)

            # we need to force argv if expr_lower == expr_upper and constv is
            #  finite non-zero (in other cases we explicitly expand ranges)
            np.copyto(
                term_lower,
                termv,
                where=((expr_lower == expr_upper) & _isfinite(constv) & (constv != 0)),
                casting="no",
            )
            np.copyto(
                term_upper,
                termv,
                where=((expr_lower == expr_upper) & _isfinite(constv) & (constv != 0)),
                casting="no",
            )

            # handle rounding errors in multiply(divide(...)) early
            term_lower = guarantee_arg_within_expr_bounds(
                lambda term_lower: np.multiply(term_lower, constv),
                exprv,
                termv,
                term_lower,
                expr_lower,
                expr_upper,
            )
            term_upper = guarantee_arg_within_expr_bounds(
                lambda term_upper: np.multiply(term_upper, constv),
                exprv,
                termv,
                term_upper,
                expr_lower,
                expr_upper,
            )

            return term.compute_data_bounds(
                term_lower,
                term_upper,
                X,
                Xs,
                late_bound,
            )

        return compute_left_associative_product_data_bounds(
            self, exprv, expr_lower, expr_upper, X, Xs, late_bound
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

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        a_const = not self._a.has_data
        b_const = not self._b.has_data
        assert not (a_const and b_const), "constant division has no error bounds"

        # evaluate a and b and a*b
        a, b = self._a, self._b
        av = a.eval(X.shape, Xs, late_bound)
        bv = b.eval(X.shape, Xs, late_bound)
        exprv = np.divide(av, bv)

        fmax = _floating_max(X.dtype)
        smallest_subnormal = _floating_smallest_subnormal(X.dtype)

        term_lower: np.ndarray[Ps, np.dtype[F]]
        term_upper: np.ndarray[Ps, np.dtype[F]]

        if a_const:
            term, termv, constv = b, bv, av

            # ensure that the expression keeps the same sign
            np.copyto(
                expr_lower,
                _maximum(X.dtype.type(+0.0), expr_lower),
                where=~_is_negative(exprv),
                casting="no",
            )
            np.copyto(
                expr_upper,
                _minimum(X.dtype.type(-0.0), expr_upper),
                where=_is_negative(exprv),
                casting="no",
            )

            # compute the divisor bounds
            # for Inf/x, we can allow any finite x with the same sign
            # for 0/x, we can allow any non-zero non-NaN x with the same sign
            # for NaN/x, we can allow any x but only propagate [-inf, inf]
            #  since [-NaN, NaN] would be misunderstood as only NaN
            # otherwise ensure that the divisor keeps the same sign:
            #  - c < 0, t >= +0: el <= e <= eu <= -0 -> tl = -el, tu = -eu
            #  - c < 0, t <= -0: +0 <= el <= e <= eu -> tl = -el, tu = -eu
            #  - c > 0, t >= +0: +0 <= el <= e <= eu -> tl = eu, tu = el
            #  - c > 0, t <= -0: el <= e <= eu <= -0 -> tl = eu, tu = el
            # if term_lower == termv and termv == -0.0, we need to guarantee
            #  that term_lower is also -0.0, same for term_upper
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            term_lower = np.array(expr_upper, copy=True)
            np.copyto(term_lower, -expr_lower, where=_is_negative(constv), casting="no")
            np.divide(constv, term_lower, out=term_lower)
            term_lower[_isnan(constv)] = -np.inf
            term_lower[constv == 0] = smallest_subnormal
            term_lower[(constv == 0) & _is_negative(termv)] = -np.inf
            term_lower[_isinf(constv)] = +0.0
            term_lower[_isinf(constv) & _is_negative(termv)] = -fmax
            term_lower = _minimum(termv, term_lower)

            term_upper = np.array(expr_lower, copy=True)
            np.copyto(term_upper, -expr_upper, where=_is_negative(constv), casting="no")
            np.divide(constv, term_upper, out=term_upper)
            term_upper[_isnan(constv)] = np.inf
            term_upper[constv == 0] = np.inf
            term_upper[(constv == 0) & _is_negative(termv)] = -smallest_subnormal
            term_upper[_isinf(constv)] = fmax
            term_upper[_isinf(constv) & _is_negative(termv)] = -0.0
            term_upper = _maximum(termv, term_upper)

            # we need to force termv if expr_lower == expr_upper
            np.copyto(term_lower, termv, where=(expr_lower == expr_upper), casting="no")
            np.copyto(term_upper, termv, where=(expr_lower == expr_upper), casting="no")

            # handle rounding errors in divide(divide(...)) early
            term_lower = guarantee_arg_within_expr_bounds(
                lambda term_lower: np.divide(constv, term_lower),
                exprv,
                termv,
                term_lower,
                expr_lower,
                expr_upper,
            )
            term_upper = guarantee_arg_within_expr_bounds(
                lambda term_upper: np.divide(constv, term_upper),
                exprv,
                termv,
                term_upper,
                expr_lower,
                expr_upper,
            )

            return term.compute_data_bounds(
                term_lower,
                term_upper,
                X,
                Xs,
                late_bound,
            )

        if b_const:
            term, termv, constv = a, av, bv

            # for x/Inf, we can allow any finite x
            # for x/0, we can allow any non-zero non-NaN x with the same sign
            # for x/NaN, we can allow any x but only propagate [-inf, inf]
            #  since [-NaN, NaN] would be misunderstood as only NaN
            # if term_lower == termv and termv == -0.0, we need to guarantee
            #  that term_lower is also -0.0, same for term_upper
            term_lower = np.array(expr_lower, copy=True)
            np.copyto(term_lower, expr_upper, where=_is_negative(constv), casting="no")
            np.multiply(term_lower, constv, out=term_lower)
            term_lower[_isnan(constv)] = -np.inf
            term_lower[constv == 0] = smallest_subnormal
            term_lower[(constv == 0) & _is_negative(termv)] = -np.inf
            term_lower[_isinf(constv)] = -fmax
            term_lower = _minimum(termv, term_lower)

            term_upper = np.array(expr_upper, copy=True)
            np.copyto(term_upper, expr_lower, where=_is_negative(constv), casting="no")
            np.multiply(term_upper, constv, out=term_upper)
            term_upper[_isnan(constv)] = np.inf
            term_upper[constv == 0] = np.inf
            term_upper[(constv == 0) & _is_negative(termv)] = -smallest_subnormal
            term_upper[_isinf(constv)] = fmax
            term_upper = _maximum(termv, term_upper)

            # we need to force termv if expr_lower == expr_upper and constv is
            #  finite non-zero (in other cases we explicitly expand ranges)
            np.copyto(
                term_lower,
                termv,
                where=((expr_lower == expr_upper) & _isfinite(constv) & (constv != 0)),
                casting="no",
            )
            np.copyto(
                term_upper,
                termv,
                where=((expr_lower == expr_upper) & _isfinite(constv) & (constv != 0)),
                casting="no",
            )

            # handle rounding errors in divide(multiply(...)) early
            term_lower = guarantee_arg_within_expr_bounds(
                lambda term_lower: np.divide(term_lower, constv),
                exprv,
                termv,
                term_lower,
                expr_lower,
                expr_upper,
            )
            term_upper = guarantee_arg_within_expr_bounds(
                lambda term_upper: np.divide(term_upper, constv),
                exprv,
                termv,
                term_upper,
                expr_lower,
                expr_upper,
            )

            return term.compute_data_bounds(
                term_lower,
                term_upper,
                X,
                Xs,
                late_bound,
            )

        return compute_left_associative_product_data_bounds(
            self, exprv, expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"{self._a!r} / {self._b!r}"


def compute_left_associative_product_data_bounds(
    expr: ScalarMultiply | ScalarDivide,
    exprv: np.ndarray[Ps, np.dtype[F]],
    expr_lower: np.ndarray[Ps, np.dtype[F]],
    expr_upper: np.ndarray[Ps, np.dtype[F]],
    X: np.ndarray[Ps, np.dtype[F]],
    Xs: np.ndarray[Ns, np.dtype[F]],
    late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
    # inlined outer ScalarFakeAbs
    # flip the lower/upper bounds if the result is negative
    #  since our rewrite below only works with non-negative exprv
    expr_temp = np.copy(expr_upper)
    np.copyto(expr_lower, -expr_upper, where=_is_negative(exprv), casting="no")
    np.copyto(expr_upper, -expr_temp, where=_is_negative(exprv), casting="no")

    # rewrite a * b * ... * z as
    #  fake_abs(e^(ln(fake_abs(a)) + ln(fake_abs(b)) + ... + ln(fake_abs(z))))
    # this is mathematically incorrect for any negative product terms but works
    #  for deriving error bounds since fake_abs handles the error bound flips
    rewritten = rewrite_left_associative_product_as_exp_sum_of_logs(expr)
    exprv_rewritten = rewritten.eval(X.shape, Xs, late_bound)

    # ensure that the bounds at least contain the rewritten expression
    #  result
    expr_lower = _minimum(expr_lower, exprv_rewritten)
    expr_upper = _maximum(expr_upper, exprv_rewritten)

    # bail out and just use the rewritten expression result as an exact
    #  bound in case isnan was changed by the rewrite
    np.copyto(
        expr_lower,
        exprv_rewritten,
        where=(_isnan(exprv) != _isnan(exprv_rewritten)),
        casting="no",
    )
    np.copyto(
        expr_upper,
        exprv_rewritten,
        where=(_isnan(exprv) != _isnan(exprv_rewritten)),
        casting="no",
    )

    return rewritten.compute_data_bounds(
        expr_lower,
        expr_upper,
        X,
        Xs,
        late_bound,
    )


def rewrite_left_associative_product_as_exp_sum_of_logs(
    expr: ScalarMultiply | ScalarDivide,
) -> Expr:
    from .power import ScalarFakeAbs

    terms_stack: list[tuple[Expr, type[ScalarAdd] | type[ScalarSubtract]]] = []

    while True:
        terms_stack.append(
            (
                ScalarLog(Logarithm.ln, ScalarFakeAbs(expr._b)),
                ScalarAdd if isinstance(expr, ScalarMultiply) else ScalarSubtract,
            )
        )

        if isinstance(expr._a, (ScalarMultiply, ScalarDivide)):
            expr = expr._a
        else:
            terms_stack.append(
                (ScalarLog(Logarithm.ln, ScalarFakeAbs(expr._a)), ScalarAdd)
            )
            break

    while len(terms_stack) > 1:
        (a, _), (b, ty) = terms_stack.pop(), terms_stack.pop()
        terms_stack.append((ty(a, b), ScalarAdd))

    [(sum_of_lns, _)] = terms_stack

    # rewrite a * b * ... * z as
    #  fake_abs(e^(ln(fake_abs(a)) + ln(fake_abs(b)) + ... + ln(fake_abs(z))))
    # this is mathematically incorrect for any negative product terms but works
    #  for deriving error bounds since fake_abs handles the error bound flips
    return ScalarExp(Exponential.exp, sum_of_lns)

import operator
from collections.abc import Mapping
from math import gcd

import numpy as np

from ....utils._compat import (
    _broadcast_to,
    _floating_max,
    _floating_smallest_subnormal,
    _is_sign_negative_number,
    _is_sign_positive_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import guarantee_arg_within_expr_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .literal import Number
from .typing import F, Ns, Ps, PsI


class ScalarMultiply(Expr[Expr, Expr]):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __new__(cls, a: Expr, b: Expr):
        ab = Number.symbolic_fold_binary(a, b, operator.mul)
        if ab is not None:
            return ab
        this = super().__new__(cls)
        this._a = a
        this._b = b
        return this

    @property
    def args(self) -> tuple[Expr, Expr]:
        return (self._a, self._b)

    def with_args(self, a: Expr, b: Expr) -> "ScalarMultiply":
        return ScalarMultiply(a, b)

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
            np.copyto(
                term_lower,
                expr_upper,
                where=_is_sign_negative_number(constv),
                casting="no",
            )
            np.divide(term_lower, constv, out=term_lower)
            term_lower[np.isnan(constv)] = -np.inf
            term_lower[np.isinf(constv)] = smallest_subnormal
            term_lower[np.isinf(constv) & _is_sign_negative_number(termv)] = -np.inf
            np.copyto(
                term_lower, termv, where=(np.isinf(constv) & (termv == 0)), casting="no"
            )
            term_lower[constv == 0] = -fmax
            term_lower = _minimum_zero_sign_sensitive(termv, term_lower)

            term_upper: np.ndarray[Ps, np.dtype[F]] = np.array(expr_upper, copy=True)
            np.copyto(
                term_upper,
                expr_lower,
                where=_is_sign_negative_number(constv),
                casting="no",
            )
            np.divide(term_upper, constv, out=term_upper)
            term_upper[np.isnan(constv)] = np.inf
            term_upper[np.isinf(constv)] = np.inf
            term_upper[
                np.isinf(constv) & _is_sign_negative_number(termv)
            ] = -smallest_subnormal
            np.copyto(
                term_upper, termv, where=(np.isinf(constv) & (termv == 0)), casting="no"
            )
            term_upper[constv == 0] = fmax
            term_upper = _maximum_zero_sign_sensitive(termv, term_upper)

            # we need to force argv if expr_lower == expr_upper and constv is
            #  finite non-zero (in other cases we explicitly expand ranges)
            np.copyto(
                term_lower,
                termv,
                where=(
                    (expr_lower == expr_upper) & np.isfinite(constv) & (constv != 0)
                ),
                casting="no",
            )
            np.copyto(
                term_upper,
                termv,
                where=(
                    (expr_lower == expr_upper) & np.isfinite(constv) & (constv != 0)
                ),
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

        # print("expr", self, av, bv, exprv, expr_lower, expr_upper)

        expr_lower = np.array(expr_lower, copy=True)
        expr_lower[_is_sign_positive_number(exprv) & (expr_lower <= 0)] = X.dtype.type(
            +0.0
        )
        expr_upper = np.array(expr_upper, copy=True)
        expr_upper[_is_sign_negative_number(exprv) & (expr_upper >= 0)] = X.dtype.type(
            -0.0
        )
        expr_abs_lower, expr_abs_upper = (
            np.array(
                _where(_is_sign_negative_number(exprv), -expr_upper, expr_lower),
                copy=None,
            ),
            np.array(
                _where(_is_sign_negative_number(exprv), -expr_lower, expr_upper),
                copy=None,
            ),
        )

        av_abs = np.abs(av)
        bv_abs = np.abs(bv)
        exprv_abs = np.array(np.abs(exprv), copy=None)

        # print("expr_bounds", expr_lower, expr_upper, expr_abs_lower, expr_abs_upper)

        fmax = _floating_max(X.dtype)
        smallest_subnormal = _floating_smallest_subnormal(X.dtype)

        expr_abs_lower_factor: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.divide(exprv_abs, expr_abs_lower), copy=None
        )
        expr_abs_lower_factor[np.isinf(expr_abs_lower_factor)] = fmax
        np.sqrt(expr_abs_lower_factor, out=expr_abs_lower_factor)
        expr_abs_lower_factor[np.isnan(expr_abs_lower_factor)] = X.dtype.type(1)

        expr_abs_upper_factor: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.divide(
                expr_abs_upper,
                _maximum_zero_sign_sensitive(exprv_abs, smallest_subnormal),
            ),
            copy=None,
        )
        expr_abs_upper_factor[np.isinf(expr_abs_upper_factor)] = fmax
        np.sqrt(expr_abs_upper_factor, out=expr_abs_upper_factor)
        expr_abs_upper_factor[np.isnan(expr_abs_upper_factor)] = X.dtype.type(1)

        # print("expr_abs_bounds", expr_abs_lower_factor, expr_abs_upper_factor)

        a_abs_lower = np.array(np.divide(av_abs, expr_abs_lower_factor), copy=None)
        a_abs_lower[expr_abs_lower == 0] = 0
        a_abs_upper = np.array(np.multiply(av_abs, expr_abs_upper_factor), copy=None)
        # np.copyto(a_abs_upper, expr_abs_upper, where=(exprv_abs == 0), casting="no")
        a_abs_upper[np.isinf(a_abs_upper) & ~np.isinf(av_abs)] = fmax

        b_abs_lower = np.array(np.divide(bv_abs, expr_abs_lower_factor), copy=None)
        b_abs_lower[expr_abs_lower == 0] = 0
        b_abs_upper = np.array(np.multiply(bv_abs, expr_abs_upper_factor), copy=None)
        # np.copyto(b_abs_upper, expr_abs_upper, where=(exprv_abs == 0), casting="no")
        b_abs_upper[np.isinf(b_abs_upper) & ~np.isinf(bv_abs)] = fmax

        any_zero = a_abs_lower == 0
        any_zero |= b_abs_lower == 0
        any_inf = np.isinf(a_abs_upper)
        any_inf |= np.isinf(b_abs_upper)
        zero_inf_clash = any_zero & any_inf

        a_abs_lower[zero_inf_clash & (a_abs_lower == 0)] = smallest_subnormal
        a_abs_upper[zero_inf_clash & np.isinf(a_abs_upper)] = fmax

        b_abs_lower[zero_inf_clash & (b_abs_lower == 0)] = smallest_subnormal
        b_abs_upper[zero_inf_clash & np.isinf(b_abs_upper)] = fmax

        a_abs_lower = _minimum_zero_sign_sensitive(av_abs, a_abs_lower)
        a_abs_upper = _maximum_zero_sign_sensitive(av_abs, a_abs_upper)

        b_abs_lower = _minimum_zero_sign_sensitive(bv_abs, b_abs_lower)
        b_abs_upper = _maximum_zero_sign_sensitive(bv_abs, b_abs_upper)

        # print("av", av_abs, a_abs_lower, a_abs_upper)
        # print("bv", bv_abs, b_abs_lower, b_abs_upper)

        # print("lu", a_abs_lower * b_abs_lower, a_abs_upper * b_abs_upper)

        tl_abs_stack = np.stack([a_abs_lower, b_abs_lower])
        tu_abs_stack = np.stack([a_abs_upper, b_abs_upper])

        # print("tl", tl_abs_stack)
        # print("tu", tu_abs_stack)

        def compute_term_product(
            t_stack: np.ndarray[tuple[int, ...], np.dtype[F]],
        ) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
            total_product: np.ndarray[tuple[int, ...], np.dtype[F]] = np.multiply(
                t_stack[0], t_stack[1]
            )

            return _broadcast_to(
                np.array(total_product, copy=None).reshape((1,) + exprv_abs.shape),
                (t_stack.shape[0],) + exprv_abs.shape,
            )

        tl_abs_stack = guarantee_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1,) + exprv_abs.shape),
                (tl_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            np.stack([av_abs, bv_abs]),
            tl_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1,) + exprv_abs.shape),
                (tl_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1,) + exprv_abs.shape),
                (tl_abs_stack.shape[0],) + exprv_abs.shape,
            ),
        )
        tu_abs_stack = guarantee_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1,) + exprv_abs.shape),
                (tu_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            np.stack([av_abs, bv_abs]),
            tu_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1,) + exprv_abs.shape),
                (tu_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1,) + exprv_abs.shape),
                (tu_abs_stack.shape[0],) + exprv_abs.shape,
            ),
        )

        # print("tl2", tl_abs_stack)
        # print("tu2", tu_abs_stack)

        a_lower = _where(
            _is_sign_negative_number(av), -tu_abs_stack[0], tl_abs_stack[0]
        )
        a_upper = _where(
            _is_sign_negative_number(av), -tl_abs_stack[0], tu_abs_stack[0]
        )

        b_lower = _where(
            _is_sign_negative_number(bv), -tu_abs_stack[1], tl_abs_stack[1]
        )
        b_upper = _where(
            _is_sign_negative_number(bv), -tl_abs_stack[1], tu_abs_stack[1]
        )

        # print("av2", av, a_lower, a_upper)
        # print("bv2", bv, b_lower, b_upper)

        Xs_lower, Xs_upper = a.compute_data_bounds(
            a_lower,
            a_upper,
            X,
            Xs,
            late_bound,
        )

        bl, bu = b.compute_data_bounds(
            b_lower,
            b_upper,
            X,
            Xs,
            late_bound,
        )
        Xs_lower = _maximum_zero_sign_sensitive(Xs_lower, bl)
        Xs_upper = _minimum_zero_sign_sensitive(Xs_upper, bu)

        Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
        Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

        return Xs_lower, Xs_upper

    def __repr__(self) -> str:
        return f"{self._a!r} * {self._b!r}"


class ScalarDivide(Expr[Expr, Expr]):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __new__(cls, a: Expr, b: Expr):
        if isinstance(a, Number) and isinstance(b, Number):
            # symbolical constant propagation for some cases of int / int
            # division always produces a floating-point number
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
                # int / 1 has an exact floating-point result
                if bi == 1:
                    return Number.from_symbolic_int_as_float(ai)
                # int / -1 has an exact floating-point result
                if bi == -1:
                    assert ai >= 0
                    # ensure that 0/1 = 0.0 and 0/-1 = -0.0
                    return Number.from_symbolic_int_as_float(ai, force_negative=True)
                # keep a / b after reduction
                if d != 0:
                    a = Number.from_symbolic_int(ai)
                    b = Number.from_symbolic_int(bi)
        this = super().__new__(cls)
        this._a = a
        this._b = b
        return this

    @property
    def args(self) -> tuple[Expr, Expr]:
        return (self._a, self._b)

    def with_args(self, a: Expr, b: Expr) -> "ScalarDivide":
        return ScalarDivide(a, b)

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

            expr_lower, expr_upper = np.copy(expr_lower), np.copy(expr_upper)

            # ensure that the expression keeps the same sign
            np.copyto(
                expr_lower,
                _maximum_zero_sign_sensitive(X.dtype.type(+0.0), expr_lower),
                where=_is_sign_positive_number(exprv),
                casting="no",
            )
            np.copyto(
                expr_upper,
                _minimum_zero_sign_sensitive(X.dtype.type(-0.0), expr_upper),
                where=_is_sign_negative_number(exprv),
                casting="no",
            )

            # compute the divisor bounds
            # for Inf/x, we can allow any finite x with the same sign
            # for 0/x, we can allow any non-zero non-NaN x with the same sign
            # for NaN/x, we can allow any x but only propagate [-inf, inf]
            #  since [-NaN, NaN] would be misunderstood as only NaN
            # otherwise ensure that the divisor keeps the same sign:
            #  - c < 0, t >= +0: el <= e <= eu <= -0 -> tl = el, tu = eu
            #  - c < 0, t <= -0: +0 <= el <= e <= eu -> tl = el, tu = eu
            #  - c > 0, t >= +0: +0 <= el <= e <= eu -> tl = eu, tu = el
            #  - c > 0, t <= -0: el <= e <= eu <= -0 -> tl = eu, tu = el
            # if term_lower == termv and termv == -0.0, we need to guarantee
            #  that term_lower is also -0.0, same for term_upper
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            term_lower = np.array(expr_upper, copy=True)
            np.copyto(
                term_lower,
                expr_lower,
                where=_is_sign_negative_number(constv),
                casting="no",
            )
            np.divide(constv, term_lower, out=term_lower)
            term_lower[np.isnan(constv)] = -np.inf
            term_lower[constv == 0] = smallest_subnormal
            term_lower[(constv == 0) & _is_sign_negative_number(termv)] = -np.inf
            np.copyto(
                term_lower, termv, where=((constv == 0) & (termv == 0)), casting="no"
            )
            term_lower[np.isinf(constv)] = +0.0
            term_lower[np.isinf(constv) & _is_sign_negative_number(termv)] = -fmax
            term_lower = _minimum_zero_sign_sensitive(termv, term_lower)

            term_upper = np.array(expr_lower, copy=True)
            np.copyto(
                term_upper,
                expr_upper,
                where=_is_sign_negative_number(constv),
                casting="no",
            )
            np.divide(constv, term_upper, out=term_upper)
            term_upper[np.isnan(constv)] = np.inf
            term_upper[constv == 0] = np.inf
            term_upper[
                (constv == 0) & _is_sign_negative_number(termv)
            ] = -smallest_subnormal
            np.copyto(
                term_upper, termv, where=((constv == 0) & (termv == 0)), casting="no"
            )
            term_upper[np.isinf(constv)] = fmax
            term_upper[np.isinf(constv) & _is_sign_negative_number(termv)] = -0.0
            term_upper = _maximum_zero_sign_sensitive(termv, term_upper)

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
            np.copyto(
                term_lower,
                expr_upper,
                where=_is_sign_negative_number(constv),
                casting="no",
            )
            np.multiply(term_lower, constv, out=term_lower)
            term_lower[np.isnan(constv)] = -np.inf
            term_lower[constv == 0] = smallest_subnormal
            term_lower[(constv == 0) & _is_sign_negative_number(termv)] = -np.inf
            np.copyto(
                term_lower, termv, where=((constv == 0) & (termv == 0)), casting="no"
            )
            term_lower[np.isinf(constv)] = -fmax
            term_lower = _minimum_zero_sign_sensitive(termv, term_lower)

            term_upper = np.array(expr_upper, copy=True)
            np.copyto(
                term_upper,
                expr_lower,
                where=_is_sign_negative_number(constv),
                casting="no",
            )
            np.multiply(term_upper, constv, out=term_upper)
            term_upper[np.isnan(constv)] = np.inf
            term_upper[constv == 0] = np.inf
            term_upper[
                (constv == 0) & _is_sign_negative_number(termv)
            ] = -smallest_subnormal
            np.copyto(
                term_upper, termv, where=((constv == 0) & (termv == 0)), casting="no"
            )
            term_upper[np.isinf(constv)] = fmax
            term_upper = _maximum_zero_sign_sensitive(termv, term_upper)

            # we need to force termv if expr_lower == expr_upper and constv is
            #  finite non-zero (in other cases we explicitly expand ranges)
            np.copyto(
                term_lower,
                termv,
                where=(
                    (expr_lower == expr_upper) & np.isfinite(constv) & (constv != 0)
                ),
                casting="no",
            )
            np.copyto(
                term_upper,
                termv,
                where=(
                    (expr_lower == expr_upper) & np.isfinite(constv) & (constv != 0)
                ),
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

        # print("expr", self, av, bv, exprv, expr_lower, expr_upper)

        expr_lower = np.array(expr_lower, copy=True)
        expr_lower[_is_sign_positive_number(exprv) & (expr_lower <= 0)] = X.dtype.type(
            +0.0
        )
        expr_upper = np.array(expr_upper, copy=True)
        expr_upper[_is_sign_negative_number(exprv) & (expr_upper >= 0)] = X.dtype.type(
            -0.0
        )
        expr_abs_lower, expr_abs_upper = (
            np.array(
                _where(_is_sign_negative_number(exprv), -expr_upper, expr_lower),
                copy=None,
            ),
            np.array(
                _where(_is_sign_negative_number(exprv), -expr_lower, expr_upper),
                copy=None,
            ),
        )

        av_abs = np.abs(av)
        bv_abs = np.abs(bv)
        exprv_abs = np.array(np.abs(exprv), copy=None)

        # print("expr_bounds", expr_lower, expr_upper, expr_abs_lower, expr_abs_upper)

        fmax = _floating_max(X.dtype)
        smallest_subnormal = _floating_smallest_subnormal(X.dtype)

        expr_abs_lower_factor: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.divide(exprv_abs, expr_abs_lower), copy=None
        )
        expr_abs_lower_factor[np.isinf(expr_abs_lower_factor)] = fmax
        np.sqrt(expr_abs_lower_factor, out=expr_abs_lower_factor)
        expr_abs_lower_factor[np.isnan(expr_abs_lower_factor)] = X.dtype.type(1)

        expr_abs_upper_factor: np.ndarray[Ps, np.dtype[F]] = np.array(
            np.divide(
                expr_abs_upper,
                _maximum_zero_sign_sensitive(exprv_abs, smallest_subnormal),
            ),
            copy=None,
        )
        expr_abs_upper_factor[np.isinf(expr_abs_upper_factor)] = fmax
        np.sqrt(expr_abs_upper_factor, out=expr_abs_upper_factor)
        expr_abs_upper_factor[np.isnan(expr_abs_upper_factor)] = X.dtype.type(1)

        # print("expr_abs_bounds", expr_abs_lower_factor, expr_abs_upper_factor)

        a_abs_lower = np.array(np.divide(av_abs, expr_abs_lower_factor), copy=None)
        a_abs_lower[expr_abs_lower == 0] = 0
        a_abs_upper = np.array(np.multiply(av_abs, expr_abs_upper_factor), copy=None)
        # np.copyto(a_abs_upper, expr_abs_upper, where=(exprv_abs == 0), casting="no")
        a_abs_upper[np.isinf(a_abs_upper) & ~np.isinf(av_abs)] = fmax

        b_abs_lower = np.array(np.multiply(bv_abs, expr_abs_lower_factor), copy=None)
        b_abs_lower[expr_abs_lower == 0] = np.inf
        b_abs_upper = np.array(np.divide(bv_abs, expr_abs_upper_factor), copy=None)
        # np.copyto(b_abs_upper, expr_abs_upper, where=(exprv_abs == 0), casting="no")
        # b_abs_upper[np.isinf(b_abs_upper) & ~np.isinf(bv_abs)] = fmax

        both_zero = a_abs_lower == 0
        both_zero &= b_abs_lower == 0
        both_inf = np.isinf(a_abs_upper)
        both_inf &= np.isinf(b_abs_upper)
        zero_inf_clash = both_zero | both_inf

        a_abs_lower[zero_inf_clash & (a_abs_lower == 0)] = smallest_subnormal
        a_abs_upper[zero_inf_clash & np.isinf(a_abs_upper)] = fmax

        b_abs_lower[zero_inf_clash & (b_abs_lower == 0)] = smallest_subnormal
        b_abs_upper[zero_inf_clash & np.isinf(b_abs_upper)] = fmax

        a_abs_lower = _minimum_zero_sign_sensitive(av_abs, a_abs_lower)
        a_abs_upper = _maximum_zero_sign_sensitive(av_abs, a_abs_upper)

        b_abs_lower = _minimum_zero_sign_sensitive(bv_abs, b_abs_lower)
        b_abs_upper = _maximum_zero_sign_sensitive(bv_abs, b_abs_upper)

        # print("av", av_abs, a_abs_lower, a_abs_upper)
        # print("bv", bv_abs, b_abs_lower, b_abs_upper)

        # print("lu", a_abs_lower * b_abs_lower, a_abs_upper * b_abs_upper)

        tl_abs_stack = np.stack([a_abs_lower, b_abs_lower])
        tu_abs_stack = np.stack([a_abs_upper, b_abs_upper])

        # print("tl", tl_abs_stack)
        # print("tu", tu_abs_stack)

        def compute_term_product(
            t_stack: np.ndarray[tuple[int, ...], np.dtype[F]],
        ) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
            total_product: np.ndarray[tuple[int, ...], np.dtype[F]] = np.divide(
                t_stack[0], t_stack[1]
            )

            return _broadcast_to(
                np.array(total_product, copy=None).reshape((1,) + exprv_abs.shape),
                (t_stack.shape[0],) + exprv_abs.shape,
            )

        tl_abs_stack = guarantee_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1,) + exprv_abs.shape),
                (tl_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            np.stack([av_abs, bv_abs]),
            tl_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1,) + exprv_abs.shape),
                (tl_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1,) + exprv_abs.shape),
                (tl_abs_stack.shape[0],) + exprv_abs.shape,
            ),
        )
        tu_abs_stack = guarantee_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1,) + exprv_abs.shape),
                (tu_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            np.stack([av_abs, bv_abs]),
            tu_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1,) + exprv_abs.shape),
                (tu_abs_stack.shape[0],) + exprv_abs.shape,
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1,) + exprv_abs.shape),
                (tu_abs_stack.shape[0],) + exprv_abs.shape,
            ),
        )

        # print("tl2", tl_abs_stack)
        # print("tu2", tu_abs_stack)

        a_lower = _where(
            _is_sign_negative_number(av), -tu_abs_stack[0], tl_abs_stack[0]
        )
        a_upper = _where(
            _is_sign_negative_number(av), -tl_abs_stack[0], tu_abs_stack[0]
        )

        b_lower = _minimum_zero_sign_sensitive(tl_abs_stack[1], tu_abs_stack[1])
        b_upper = _maximum_zero_sign_sensitive(tl_abs_stack[1], tu_abs_stack[1])
        b_lower, b_upper = (
            _where(_is_sign_negative_number(bv), -b_upper, b_lower),
            _where(_is_sign_negative_number(bv), -b_lower, b_upper),
        )

        # print("av2", av, a_lower, a_upper)
        # print("bv2", bv, b_lower, b_upper)

        Xs_lower, Xs_upper = a.compute_data_bounds(
            a_lower,
            a_upper,
            X,
            Xs,
            late_bound,
        )

        bl, bu = b.compute_data_bounds(
            b_lower,
            b_upper,
            X,
            Xs,
            late_bound,
        )
        Xs_lower = _maximum_zero_sign_sensitive(Xs_lower, bl)
        Xs_upper = _minimum_zero_sign_sensitive(Xs_upper, bu)

        Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
        Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

        return Xs_lower, Xs_upper

    def __repr__(self) -> str:
        return f"{self._a!r} / {self._b!r}"

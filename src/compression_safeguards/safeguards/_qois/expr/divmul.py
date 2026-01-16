import operator
from collections.abc import Mapping
from math import gcd

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _broadcast_to,
    _ensure_array,
    _is_negative_zero,
    _is_positive_zero,
    _is_sign_negative_number,
    _is_sign_positive_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _stack,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import (
    checked_data_bounds,
    guarantee_arg_within_expr_bounds,
    guarantee_stacked_arg_within_expr_bounds,
)
from ..typing import F, Ns, Ps, np_sndarray
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant
from .literal import Number


class ScalarMultiply(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    def __new__(cls, a: AnyExpr, b: AnyExpr) -> "ScalarMultiply | Number":  # type: ignore[misc]
        ab = Number.symbolic_fold_binary(a, b, operator.mul)
        if ab is not None:
            return ab
        return super().__new__(cls)

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarMultiply | Number":
        return ScalarMultiply(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.multiply, ScalarMultiply
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.multiply(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        a_const = not self._a.has_data
        b_const = not self._b.has_data
        assert not (a_const and b_const), "constant multiplication has no data bounds"

        # evaluate a and b and a*b
        a, b = self._a, self._b
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)
        exprv = np.multiply(av, bv)

        if a_const or b_const:
            term, termv, constv = (b, bv, av) if a_const else (a, av, bv)

            fmax = np.finfo(Xs.dtype).max
            smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

            # for x*0, we can allow any finite x, unless the output +-0.0 sign
            #  is restricted, then we need to restrict the sign of x:
            #   | constv | expr_lower | expr_upper | term_lower | term_upper |
            #   | +0.0   | >= +0.0    | >= +0.0    | +0.0       | +fmax      |
            #   | +0.0   | <= -0.0    | <= -0.0    | -fmax      | -0.0       |
            #   | +0.0   | <= -0.0    | >= +0.0    | -fmax      | +fmax      |
            #   | -0.0   | >= +0.0    | >= +0.0    | -fmax      | -0.0       |
            #   | -0.0   | <= -0.0    | <= -0.0    | +0.0       | +fmax      |
            #   | -0.0   | <= -0.0    | >= +0.0    | -fmax      | +fmax      |
            #  - term_lower := +0.0 if
            #    signbit(constv) == signbit(expr_lower) == signbit(expr_upper)
            #  - term_upper := -0.0 if
            #    ~signbit(constv) == signbit(expr_lower) == signbit(expr_upper)
            # for x*Inf, we can allow any non-zero non-NaN x with the same sign
            # for x*NaN, we can allow any x but only propagate [-inf, inf]
            #  since [-NaN, NaN] would be misunderstood as only NaN
            # if term_lower == termv and termv == -0.0, we need to guarantee
            #  that term_lower is also -0.0, same for term_upper
            term_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
                expr_lower, copy=True
            )
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
            term_lower[
                (constv == 0)
                & (_is_positive_zero(constv) == _is_sign_positive_number(expr_lower))
                & (_is_positive_zero(constv) == _is_sign_positive_number(expr_upper))
            ] = +0.0
            term_lower = _ensure_array(_minimum_zero_sign_sensitive(termv, term_lower))

            term_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
                expr_upper, copy=True
            )
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
            term_upper[
                (constv == 0)
                & (_is_negative_zero(constv) != _is_sign_negative_number(expr_lower))
                & (_is_negative_zero(constv) != _is_sign_negative_number(expr_upper))
            ] = -0.0
            term_upper = _ensure_array(_maximum_zero_sign_sensitive(termv, term_upper))

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
                Xs,
                late_bound,
            )

        # if neither a not b is const, we simplify by not allowing the sign of
        #  a*b to change
        # we further simplify the code by working with abs(a*b) = abs(a)*abs(b)
        #  and only taking the sign of a and b into account at the end

        # limit expr_lower and expr_upper to keep the same sign
        # then compute the lower and upper bounds for the abs(expr)
        expr_lower = _ensure_array(expr_lower, copy=True)
        expr_lower[_is_sign_positive_number(exprv) & (expr_lower <= 0)] = +0.0
        expr_upper = _ensure_array(expr_upper, copy=True)
        expr_upper[_is_sign_negative_number(exprv) & (expr_upper >= 0)] = -0.0
        expr_abs_lower, expr_abs_upper = (
            _ensure_array(
                _where(_is_sign_negative_number(exprv), -expr_upper, expr_lower)
            ),
            _ensure_array(
                _where(_is_sign_negative_number(exprv), -expr_lower, expr_upper)
            ),
        )

        av_abs = np.abs(av)
        bv_abs = np.abs(bv)
        exprv_abs = _ensure_array(np.abs(exprv))

        fmax = np.finfo(Xs.dtype).max
        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # we are given l <= e <= u, which we translate into e/lf <= e <= e*uf
        # - if the factor is infinite, we limit it to fmax
        # - if the factor is NaN, e.g. from 0/0, we set the factor to 1
        # finally we split the factor geometrically in two using the sqrt
        expr_abs_lower_factor: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.divide(exprv_abs, expr_abs_lower)
        )
        expr_abs_lower_factor[np.isinf(expr_abs_lower_factor)] = fmax
        np.sqrt(expr_abs_lower_factor, out=expr_abs_lower_factor)
        expr_abs_lower_factor[np.isnan(expr_abs_lower_factor)] = 1

        expr_abs_upper_factor: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.divide(
                expr_abs_upper,
                # we avoid division by zero here
                # - if exprv_abs = 0 and expr_abs_upper = 0, factor = 0 works
                # - if exprv_abs = 0 and expr_abs_upper > 0, factor is large
                # - if exprv_abs > 0 and expr_abs_upper > 0, works as normal
                _maximum_zero_sign_sensitive(exprv_abs, smallest_subnormal),
            )
        )
        expr_abs_upper_factor[np.isinf(expr_abs_upper_factor)] = fmax
        np.sqrt(expr_abs_upper_factor, out=expr_abs_upper_factor)
        expr_abs_upper_factor[np.isnan(expr_abs_upper_factor)] = 1

        # TODO: for infinite a*b that includes finite bounds, allow them
        # TODO: we could peek at a and b if they're powers with a constant
        #       exponent to weigh the factor split
        # TODO: handle all terms of a left-associative product in one go

        # compute the bounds for abs(a) and abs(b)
        # - a lower bound of zero is always passed through
        # - we ensure that the upper bound does not overflow into inf
        a_abs_lower = _ensure_array(np.divide(av_abs, expr_abs_lower_factor))
        a_abs_lower[(expr_abs_lower == 0) & ~np.isnan(av_abs)] = 0
        a_abs_upper = _ensure_array(np.multiply(av_abs, expr_abs_upper_factor))
        a_abs_upper[np.isinf(a_abs_upper) & ~np.isinf(expr_abs_upper)] = fmax

        b_abs_lower = _ensure_array(np.divide(bv_abs, expr_abs_lower_factor))
        b_abs_lower[(expr_abs_lower == 0) & ~np.isnan(bv_abs)] = 0
        b_abs_upper = _ensure_array(np.multiply(bv_abs, expr_abs_upper_factor))
        b_abs_upper[np.isinf(b_abs_upper) & ~np.isinf(expr_abs_upper)] = fmax

        any_nan = np.isnan(av_abs)
        any_nan |= np.isnan(bv_abs)
        any_zero = a_abs_lower == 0
        any_zero |= b_abs_lower == 0
        any_inf = np.isinf(a_abs_upper)
        any_inf |= np.isinf(b_abs_upper)
        zero_inf_clash = any_zero & any_inf

        # we cannot allow 0*inf, so truncate the bounds to not include them
        # if any term is NaN, other non-NaN terms can have any value
        a_abs_lower[zero_inf_clash & (a_abs_lower == 0)] = smallest_subnormal
        a_abs_upper[zero_inf_clash & np.isinf(a_abs_upper)] = fmax
        a_abs_lower[any_nan & ~np.isnan(av_abs)] = 0
        a_abs_upper[any_nan & ~np.isnan(av_abs)] = np.inf

        b_abs_lower[zero_inf_clash & (b_abs_lower == 0)] = smallest_subnormal
        b_abs_upper[zero_inf_clash & np.isinf(b_abs_upper)] = fmax
        b_abs_lower[any_nan & ~np.isnan(bv_abs)] = 0
        b_abs_upper[any_nan & ~np.isnan(bv_abs)] = np.inf

        # ensure that the bounds on abs(a) and abs(b) include their values
        a_abs_lower = _ensure_array(_minimum_zero_sign_sensitive(av_abs, a_abs_lower))
        a_abs_upper = _ensure_array(_maximum_zero_sign_sensitive(av_abs, a_abs_upper))

        b_abs_lower = _ensure_array(_minimum_zero_sign_sensitive(bv_abs, b_abs_lower))
        b_abs_upper = _ensure_array(_maximum_zero_sign_sensitive(bv_abs, b_abs_upper))

        # stack the bounds on a and b so that we can nudge their bounds, if
        #  necessary, together
        tl_abs_stack = _stack([a_abs_lower, b_abs_lower])
        tu_abs_stack = _stack([a_abs_upper, b_abs_upper])

        def compute_term_product(
            t_stack: np.ndarray[tuple[int, Ps], np.dtype[F]],
        ) -> np.ndarray[tuple[int, Ps], np.dtype[F]]:
            total_product: np.ndarray[tuple[Ps], np.dtype[F]] = np.multiply(
                t_stack[0], t_stack[1]
            )

            return _broadcast_to(
                _ensure_array(total_product).reshape((1, *exprv_abs.shape)),
                (t_stack.shape[0], *exprv_abs.shape),
            )

        tl_abs_stack = guarantee_stacked_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1, *exprv_abs.shape)),
                (tl_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _stack([av_abs, bv_abs]),
            tl_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1, *exprv_abs.shape)),
                (tl_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1, *exprv_abs.shape)),
                (tl_abs_stack.shape[0], *exprv_abs.shape),
            ),
        )
        tu_abs_stack = guarantee_stacked_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1, *exprv_abs.shape)),
                (tu_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _stack([av_abs, bv_abs]),
            tu_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1, *exprv_abs.shape)),
                (tu_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1, *exprv_abs.shape)),
                (tu_abs_stack.shape[0], *exprv_abs.shape),
            ),
        )

        # derive the bounds on a and b based on the bounds on abs(a) and abs(b)
        # if any term is NaN, other non-NaN terms can have any value of any sign
        a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _where(_is_sign_negative_number(av), -tu_abs_stack[0], tl_abs_stack[0])
        )
        a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _where(_is_sign_negative_number(av), -tl_abs_stack[0], tu_abs_stack[0])
        )
        a_lower[any_nan & ~np.isnan(av)] = -np.inf
        a_upper[any_nan & ~np.isnan(av)] = np.inf

        b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _where(_is_sign_negative_number(bv), -tu_abs_stack[1], tl_abs_stack[1])
        )
        b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _where(_is_sign_negative_number(bv), -tl_abs_stack[1], tu_abs_stack[1])
        )
        b_lower[any_nan & ~np.isnan(bv)] = -np.inf
        b_upper[any_nan & ~np.isnan(bv)] = np.inf

        # recurse into a and b to propagate their bounds, then combine their
        #  bounds on Xs
        Xs_lower, Xs_upper = a.compute_data_bounds(
            a_lower,
            a_upper,
            Xs,
            late_bound,
        )

        bl, bu = b.compute_data_bounds(
            b_lower,
            b_upper,
            Xs,
            late_bound,
        )
        Xs_lower = _maximum_zero_sign_sensitive(Xs_lower, bl)
        Xs_upper = _minimum_zero_sign_sensitive(Xs_upper, bu)

        # ensure that the bounds on Xs include Xs
        Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
        Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

        return Xs_lower, Xs_upper

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} * {self._b!r}"


class ScalarDivide(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __new__(cls, a: AnyExpr, b: AnyExpr) -> "ScalarDivide | Number":  # type: ignore[misc]
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
        # we have to also assign inside __new__ since the symbolic constant
        #  folding check can change a and b to simplify the expression
        this = super().__new__(cls)
        this._a = a
        this._b = b
        return this

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarDivide | Number":
        return ScalarDivide(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.divide, ScalarDivide
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.divide(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        expr_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        a_const = not self._a.has_data
        b_const = not self._b.has_data
        assert not (a_const and b_const), "constant division has no data bounds"

        # evaluate a and b and a*b
        a, b = self._a, self._b
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)
        exprv = np.divide(av, bv)

        fmax = np.finfo(Xs.dtype).max
        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        term_lower: np.ndarray[tuple[Ps], np.dtype[F]]
        term_upper: np.ndarray[tuple[Ps], np.dtype[F]]

        if a_const:
            term, termv, constv = b, bv, av

            expr_lower, expr_upper = (
                _ensure_array(expr_lower, copy=True),
                _ensure_array(expr_upper, copy=True),
            )

            # ensure that the expression keeps the same sign
            expr_lower[(expr_lower <= 0) & _is_sign_positive_number(exprv)] = +0.0
            expr_upper[(expr_upper >= 0) & _is_sign_negative_number(exprv)] = -0.0

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
            term_lower = _ensure_array(expr_upper, copy=True)
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
            term_lower = _ensure_array(_minimum_zero_sign_sensitive(termv, term_lower))

            term_upper = _ensure_array(expr_lower, copy=True)
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
            term_upper = _ensure_array(_maximum_zero_sign_sensitive(termv, term_upper))

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
                Xs,
                late_bound,
            )

        if b_const:
            term, termv, constv = a, av, bv

            # for x/Inf, we can allow any finite x, unless the output +-0.0 sign
            #  is restricted, then we need to restrict the sign of x:
            #   | constv | expr_lower | expr_upper | term_lower | term_upper |
            #   | +Inf   | >= +0.0    | >= +0.0    | +0.0       | +fmax      |
            #   | +Inf   | <= -0.0    | <= -0.0    | -fmax      | -0.0       |
            #   | +Inf   | <= -0.0    | >= +0.0    | -fmax      | +fmax      |
            #   | -Inf   | >= +0.0    | >= +0.0    | -fmax      | -0.0       |
            #   | -Inf   | <= -0.0    | <= -0.0    | +0.0       | +fmax      |
            #   | -Inf   | <= -0.0    | >= +0.0    | -fmax      | +fmax      |
            #  - term_lower := +0.0 if
            #    signbit(constv) == signbit(expr_lower) == signbit(expr_upper)
            #  - term_upper := -0.0 if
            #    ~signbit(constv) == signbit(expr_lower) == signbit(expr_upper)
            # for x/0, we can allow any non-zero non-NaN x with the same sign
            # for x/NaN, we can allow any x but only propagate [-inf, inf]
            #  since [-NaN, NaN] would be misunderstood as only NaN
            # if term_lower == termv and termv == -0.0, we need to guarantee
            #  that term_lower is also -0.0, same for term_upper
            term_lower = _ensure_array(expr_lower, copy=True)
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
            term_lower[
                np.isinf(constv)
                & (
                    _is_sign_positive_number(constv)
                    == _is_sign_positive_number(expr_lower)
                )
                & (
                    _is_sign_positive_number(constv)
                    == _is_sign_positive_number(expr_upper)
                )
            ] = +0.0
            term_lower = _ensure_array(_minimum_zero_sign_sensitive(termv, term_lower))

            term_upper = _ensure_array(expr_upper, copy=True)
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
            term_upper[
                np.isinf(constv)
                & (
                    _is_sign_negative_number(constv)
                    != _is_sign_negative_number(expr_lower)
                )
                & (
                    _is_sign_negative_number(constv)
                    != _is_sign_negative_number(expr_upper)
                )
            ] = -0.0
            term_upper = _ensure_array(_maximum_zero_sign_sensitive(termv, term_upper))

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
                Xs,
                late_bound,
            )

        # if neither a not b is const, we simplify by not allowing the sign of
        #  a/b to change
        # we further simplify the code by working with abs(a/b) = abs(a)/abs(b)
        #  and only taking the sign of a and b into account at the end

        # limit expr_lower and expr_upper to keep the same sign
        # then compute the lower and upper bounds for the abs(expr)
        expr_lower = _ensure_array(expr_lower, copy=True)
        expr_lower[_is_sign_positive_number(exprv) & (expr_lower <= 0)] = +0.0
        expr_upper = _ensure_array(expr_upper, copy=True)
        expr_upper[_is_sign_negative_number(exprv) & (expr_upper >= 0)] = -0.0
        expr_abs_lower, expr_abs_upper = (
            _ensure_array(
                _where(_is_sign_negative_number(exprv), -expr_upper, expr_lower)
            ),
            _ensure_array(
                _where(_is_sign_negative_number(exprv), -expr_lower, expr_upper)
            ),
        )

        av_abs = np.abs(av)
        bv_abs = np.abs(bv)
        exprv_abs = _ensure_array(np.abs(exprv))

        fmax = np.finfo(Xs.dtype).max
        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # we are given l <= e <= u, which we translate into e/lf <= e <= e*uf
        # - if the factor is infinite, we limit it to fmax
        # - if the factor is NaN, e.g. from 0/0, we set the factor to 1
        # finally we split the factor geometrically in two using the sqrt
        expr_abs_lower_factor: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.divide(exprv_abs, expr_abs_lower)
        )
        expr_abs_lower_factor[np.isinf(expr_abs_lower_factor)] = fmax
        np.sqrt(expr_abs_lower_factor, out=expr_abs_lower_factor)
        expr_abs_lower_factor[np.isnan(expr_abs_lower_factor)] = 1

        expr_abs_upper_factor: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.divide(
                expr_abs_upper,
                # we avoid division by zero here
                _maximum_zero_sign_sensitive(exprv_abs, smallest_subnormal),
            )
        )
        expr_abs_upper_factor[np.isinf(expr_abs_upper_factor)] = fmax
        np.sqrt(expr_abs_upper_factor, out=expr_abs_upper_factor)
        expr_abs_upper_factor[np.isnan(expr_abs_upper_factor)] = 1

        # TODO: for infinite a/b that includes finite bounds, allow them
        # TODO: we could peek at a and b if they're powers with a constant
        #       exponent to weigh the factor split
        # TODO: handle all terms of a left-associative product in one go

        # compute the bounds for abs(a) and abs(b)
        # - a lower bound of zero is always passed through
        # - we ensure that the upper bound does not overflow into inf
        # - the bounds for the divisor b are flipped
        a_abs_lower = _ensure_array(np.divide(av_abs, expr_abs_lower_factor))
        a_abs_lower[(expr_abs_lower == 0) & ~np.isnan(av_abs)] = 0
        a_abs_upper = _ensure_array(np.multiply(av_abs, expr_abs_upper_factor))
        a_abs_upper[np.isinf(a_abs_upper) & ~np.isinf(expr_abs_upper)] = fmax

        b_abs_lower = _ensure_array(np.multiply(bv_abs, expr_abs_lower_factor))
        b_abs_lower[(expr_abs_lower == 0) & ~np.isnan(av_abs)] = np.inf
        b_abs_upper = _ensure_array(np.divide(bv_abs, expr_abs_upper_factor))
        b_abs_upper[(b_abs_upper == 0) & ~np.isinf(expr_abs_upper)] = smallest_subnormal

        any_nan = np.isnan(av_abs)
        any_nan |= np.isnan(bv_abs)
        both_zero = a_abs_lower == 0
        both_zero &= b_abs_lower == 0
        both_inf = np.isinf(a_abs_upper)
        both_inf &= np.isinf(b_abs_upper)
        zero_inf_clash = both_zero | both_inf

        # we cannot allow 0/0 or inf/inf, so truncate the bounds to not include
        #  them
        # if any term is NaN, other non-NaN terms can have any value
        a_abs_lower[zero_inf_clash & (a_abs_lower == 0)] = smallest_subnormal
        a_abs_upper[zero_inf_clash & np.isinf(a_abs_upper)] = fmax
        a_abs_lower[any_nan & ~np.isnan(av_abs)] = 0
        a_abs_upper[any_nan & ~np.isnan(av_abs)] = np.inf

        b_abs_lower[zero_inf_clash & (b_abs_lower == 0)] = smallest_subnormal
        b_abs_upper[zero_inf_clash & np.isinf(b_abs_upper)] = fmax
        b_abs_lower[any_nan & ~np.isnan(bv_abs)] = 0
        b_abs_upper[any_nan & ~np.isnan(bv_abs)] = np.inf

        # ensure that the bounds on abs(a) and abs(b) include their values
        a_abs_lower = _ensure_array(_minimum_zero_sign_sensitive(av_abs, a_abs_lower))
        a_abs_upper = _ensure_array(_maximum_zero_sign_sensitive(av_abs, a_abs_upper))

        b_abs_lower = _ensure_array(_minimum_zero_sign_sensitive(bv_abs, b_abs_lower))
        b_abs_upper = _ensure_array(_maximum_zero_sign_sensitive(bv_abs, b_abs_upper))

        # stack the bounds on a and b so that we can nudge their bounds, if
        #  necessary, together
        tl_abs_stack = _stack([a_abs_lower, b_abs_lower])
        tu_abs_stack = _stack([a_abs_upper, b_abs_upper])

        def compute_term_product(
            t_stack: np.ndarray[tuple[int, Ps], np.dtype[F]],
        ) -> np.ndarray[tuple[int, Ps], np.dtype[F]]:
            total_product: np.ndarray[tuple[Ps], np.dtype[F]] = np.divide(
                t_stack[0], t_stack[1]
            )

            return _broadcast_to(
                _ensure_array(total_product).reshape((1, *exprv_abs.shape)),
                (t_stack.shape[0], *exprv_abs.shape),
            )

        tl_abs_stack = guarantee_stacked_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1, *exprv_abs.shape)),
                (tl_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _stack([av_abs, bv_abs]),
            tl_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1, *exprv_abs.shape)),
                (tl_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1, *exprv_abs.shape)),
                (tl_abs_stack.shape[0], *exprv_abs.shape),
            ),
        )
        tu_abs_stack = guarantee_stacked_arg_within_expr_bounds(
            compute_term_product,
            _broadcast_to(
                exprv_abs.reshape((1, *exprv_abs.shape)),
                (tu_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _stack([av_abs, bv_abs]),
            tu_abs_stack,
            _broadcast_to(
                expr_abs_lower.reshape((1, *exprv_abs.shape)),
                (tu_abs_stack.shape[0], *exprv_abs.shape),
            ),
            _broadcast_to(
                expr_abs_upper.reshape((1, *exprv_abs.shape)),
                (tu_abs_stack.shape[0], *exprv_abs.shape),
            ),
        )

        # derive the bounds on a and b based on the bounds on abs(a) and abs(b)
        # if any term is NaN, other non-NaN terms can have any value of any sign
        # the bounds for the divisor b are flipped back here
        a_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _where(_is_sign_negative_number(av), -tu_abs_stack[0], tl_abs_stack[0])
        )
        a_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            _where(_is_sign_negative_number(av), -tl_abs_stack[0], tu_abs_stack[0])
        )
        a_lower[any_nan & ~np.isnan(av)] = -np.inf
        a_upper[any_nan & ~np.isnan(av)] = np.inf

        b_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _minimum_zero_sign_sensitive(
            tl_abs_stack[1], tu_abs_stack[1]
        )
        b_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _maximum_zero_sign_sensitive(
            tl_abs_stack[1], tu_abs_stack[1]
        )
        b_lower, b_upper = (
            _ensure_array(_where(_is_sign_negative_number(bv), -b_upper, b_lower)),
            _ensure_array(_where(_is_sign_negative_number(bv), -b_lower, b_upper)),
        )
        b_lower[any_nan & ~np.isnan(bv)] = -np.inf
        b_upper[any_nan & ~np.isnan(bv)] = np.inf

        # recurse into a and b to propagate their bounds, then combine their
        #  bounds on Xs
        Xs_lower, Xs_upper = a.compute_data_bounds(
            a_lower,
            a_upper,
            Xs,
            late_bound,
        )

        bl, bu = b.compute_data_bounds(
            b_lower,
            b_upper,
            Xs,
            late_bound,
        )
        Xs_lower = _maximum_zero_sign_sensitive(Xs_lower, bl)
        Xs_upper = _minimum_zero_sign_sensitive(Xs_upper, bu)

        # ensure that the bounds on Xs include Xs
        Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
        Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

        return Xs_lower, Xs_upper

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} / {self._b!r}"

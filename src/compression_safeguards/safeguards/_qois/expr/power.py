from collections.abc import Mapping

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


class ScalarPower(Expr[AnyExpr, AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_b")
    _a: AnyExpr
    _b: AnyExpr

    def __init__(self, a: AnyExpr, b: AnyExpr) -> None:
        self._a = a
        self._b = b

    def __new__(cls, a: AnyExpr, b: AnyExpr) -> "ScalarPower | Number":  # type: ignore[misc]
        if isinstance(a, Number) and isinstance(b, Number):
            # symbolical constant propagation for int ** int
            # where the exponent is non-negative and the result thus is an int
            ai, bi = a.as_int(), b.as_int()
            if (ai is not None) and (bi is not None):
                if bi >= 0:
                    return Number.from_symbolic_int(ai**bi)
        return super().__new__(cls)

    @property
    @override
    def args(self) -> tuple[AnyExpr, AnyExpr]:
        return (self._a, self._b)

    @override
    def with_args(self, a: AnyExpr, b: AnyExpr) -> "ScalarPower | Number":
        return ScalarPower(a, b)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.power, ScalarPower
        )

    @override
    def eval(
        self,
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        return np.power(self._a.eval(Xs, late_bound), self._b.eval(Xs, late_bound))

    @override
    @checked_data_bounds
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
        assert not (a_const and b_const), "constant power has no data bounds"

        # evaluate a, b, and power(a, b)
        a, b = self._a, self._b
        av = a.eval(Xs, late_bound)
        bv = b.eval(Xs, late_bound)
        exprv: np.ndarray[tuple[Ps], np.dtype[F]] = np.power(av, bv)

        # powers of negative numbers are just too tricky, so we just force
        #  av and bv to remain the same if av < 0 and handle av = 0 by cases
        # powers of sign-positive numbers cannot produce sign-negative numbers,
        #  so clamp expr_lower to >= +0.0 if av >= +0.0
        # TODO: when optimising for a ** int, this clamping should be disabled
        expr_lower = _ensure_array(expr_lower, copy=True)
        expr_lower[_is_sign_positive_number(av) & (expr_lower <= 0)] = +0.0

        b_lower: np.ndarray[tuple[Ps], np.dtype[F]]
        b_upper: np.ndarray[tuple[Ps], np.dtype[F]]

        if a_const:
            av_log = np.log(av)

            # apply the inverse function to get the bounds on b
            # if b_lower == bv and bv == -0.0, we need to guarantee that
            #  b_lower is also -0.0, same for b_upper
            b_lower = _ensure_array(
                _minimum_zero_sign_sensitive(bv, np.divide(np.log(expr_lower), av_log))
            )
            b_upper = _ensure_array(
                _maximum_zero_sign_sensitive(bv, np.divide(np.log(expr_upper), av_log))
            )

            # we need to force bv if expr_lower == expr_upper
            np.copyto(b_lower, bv, where=(expr_lower == expr_upper), casting="no")
            np.copyto(b_upper, bv, where=(expr_lower == expr_upper), casting="no")

            smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

            # handle 0 ** bv
            # - +-0 ** +-0 = 1 -> discontinuous, force bv = +-0
            # - +0 ** (>0) = +0 -> allow any bv > 0
            # - +0 ** (<0) = -inf -> allow any bv < 0
            # - -0 ** (>0) = +-0 -> too tricky, just keep both av and bv the same
            # - -0 ** (<0) = +-inf -> too tricky, just keep both av and bv the same
            np.copyto(b_lower, bv, where=_is_negative_zero(av), casting="no")
            np.copyto(b_upper, bv, where=_is_negative_zero(av), casting="no")
            b_lower[(av == 0) & (bv == 0)] = -0.0
            b_upper[(av == 0) & (bv == 0)] = +0.0
            b_lower[_is_positive_zero(av) & (bv > 0)] = smallest_subnormal
            b_upper[_is_positive_zero(av) & (bv > 0)] = np.inf
            b_lower[_is_positive_zero(av) & (bv < 0)] = -np.inf
            b_upper[_is_positive_zero(av) & (bv < 0)] = -smallest_subnormal

            # handle +inf ** bv
            # - +inf ** +-0 = 1 -> discontinuous, force bv = +-0
            # - +inf ** (>0) = +inf -> allow any bv > 0
            # - +inf ** (<0) = +0 -> allow any bv < 0
            b_lower[np.isinf(av) & (bv == 0)] = -0.0
            b_upper[np.isinf(av) & (bv == 0)] = +0.0
            b_lower[np.isinf(av) & (bv > 0)] = smallest_subnormal
            b_upper[np.isinf(av) & (bv > 0)] = np.inf
            b_lower[np.isinf(av) & (bv < 0)] = -np.inf
            b_upper[np.isinf(av) & (bv < 0)] = -smallest_subnormal

            # handle NaN ** bv
            # - NaN ** +-0 = 1 -> discontinuous, force bv = +-0
            # - NaN ** (<0>) = NaN -> allow any bv != 0
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            b_lower[np.isnan(av) & (bv == 0)] = -0.0
            b_upper[np.isnan(av) & (bv == 0)] = +0.0
            b_lower[np.isnan(av) & (bv > 0)] = smallest_subnormal
            b_upper[np.isnan(av) & (bv > 0)] = np.inf
            b_lower[np.isnan(av) & (bv < 0)] = -np.inf
            b_upper[np.isnan(av) & (bv < 0)] = -smallest_subnormal

            # 1 ** [-NaN, -inf, +inf, +NaN] = 1 -> allow any bv
            b_lower[(av == 1)] = -np.inf
            b_upper[(av == 1)] = np.inf

            # powers of negative numbers are just too tricky, so force bv
            np.copyto(b_lower, bv, where=(av < 0), casting="no")
            np.copyto(b_upper, bv, where=(av < 0), casting="no")

            # handle rounding errors in power(a, log(..., base=a)) early
            b_lower = guarantee_arg_within_expr_bounds(
                lambda b_lower: np.power(av, b_lower),
                exprv,
                bv,
                b_lower,
                expr_lower,
                expr_upper,
            )
            b_upper = guarantee_arg_within_expr_bounds(
                lambda b_upper: np.power(av, b_upper),
                exprv,
                bv,
                b_upper,
                expr_lower,
                expr_upper,
            )

            return b.compute_data_bounds(
                b_lower,
                b_upper,
                Xs,
                late_bound,
            )

        a_lower: np.ndarray[tuple[Ps], np.dtype[F]]
        a_upper: np.ndarray[tuple[Ps], np.dtype[F]]

        if b_const:
            # TODO: optimise bounds for a ** int

            # apply the inverse function to get the bounds on a
            # if a_lower == av and av == -0.0, we need to guarantee that
            #  a_lower is also -0.0, same for a_upper
            a_lower = _ensure_array(
                _minimum_zero_sign_sensitive(
                    av, np.power(expr_lower, np.reciprocal(bv))
                )
            )
            a_upper = _maximum_zero_sign_sensitive(
                av, np.power(expr_upper, np.reciprocal(bv))
            )

            # we need to force av if expr_lower == expr_upper
            np.copyto(a_lower, av, where=(expr_lower == expr_upper), casting="no")
            np.copyto(a_upper, av, where=(expr_lower == expr_upper), casting="no")

            # handle +-0 ** bv -> keep av the same
            #  special-cases are optimised below
            np.copyto(a_lower, av, where=(av == 0), casting="no")
            np.copyto(a_upper, av, where=(av == 0), casting="no")

            # [-NaN, -inf, +inf, +NaN] ** +-0 = 1 -> allow any av
            a_lower[(bv == 0)] = -np.inf
            a_upper[(bv == 0)] = np.inf

            one_plus_eps = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(2))
            one_minus_eps = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(0))

            # handle av ** +-inf
            # - 1 ** +-inf = 1 -> discontinuous, force av = 1
            # - (+-0<1) ** +inf = +0 -> allow any -0.0 <= av < 1
            # - (>1) ** +inf = +inf -> allow any av > 1
            # - (+-0<1) ** -inf = +inf -> allow any -0.0 <= av < 1
            # - (>1) ** -inf = +0 -> allow any av > 1
            # 1 ** +-inf = 1, so force av = 1
            a_lower[(av == 1) & np.isinf(bv)] = 1
            a_upper[(av == 1) & np.isinf(bv)] = 1
            a_lower[(av > 1) & np.isinf(bv)] = one_plus_eps
            a_upper[(av > 1) & np.isinf(bv)] = np.inf
            a_lower[(av < 1) & np.isinf(bv)] = -0.0
            a_upper[(av < 1) & np.isinf(bv)] = one_minus_eps

            # handle av ** NaN
            # - 1 ** NaN = 1 -> discontinuous, force av = 1
            # - (<1>) ** NaN = NaN -> allow any av != 1
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            a_lower[(av == 1) & np.isnan(bv)] = 1
            a_upper[(av == 1) & np.isnan(bv)] = 1
            a_lower[(av > 1) & np.isnan(bv)] = one_plus_eps
            a_upper[(av > 1) & np.isnan(bv)] = np.inf
            a_lower[(av < 1) & np.isnan(bv)] = -np.inf
            a_upper[(av < 1) & np.isnan(bv)] = one_minus_eps

            # powers of negative numbers are just too tricky, so force av
            np.copyto(a_lower, av, where=(av < 0), casting="no")
            np.copyto(a_upper, av, where=(av < 0), casting="no")

            # handle rounding errors in power(power(..., 1/b), b) early
            a_lower = guarantee_arg_within_expr_bounds(
                lambda a_lower: np.power(a_lower, bv),
                exprv,
                av,
                a_lower,
                expr_lower,
                expr_upper,
            )
            a_upper = guarantee_arg_within_expr_bounds(
                lambda a_upper: np.power(a_upper, bv),
                exprv,
                av,
                a_upper,
                expr_lower,
                expr_upper,
            )

            return a.compute_data_bounds(
                a_lower,
                a_upper,
                Xs,
                late_bound,
            )

        one_plus_eps = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(2))
        one_minus_eps = np.nextafter(Xs.dtype.type(1), Xs.dtype.type(0))

        expr_upper = _ensure_array(expr_upper, copy=True)

        # if neither a nor b is const, we need to split the data bound, and
        #  we use that, symbolically, a ** b = exp( b * ln(a) ) for a > 0
        # for av = +-0.0, we keep av the same and handle the cases for bv
        # powers of negative numbers are just too tricky, in general, e.g. can
        #  change sign or suddenly become NaN with negative exponents, so we
        #  just force av and bv to remain the same if av < 0
        # since we know how to derive data bounds for exp and ln and how to
        #  split the data bound for a multiplication, we can combine their
        #  derivations to split the data bound on a ** b into bounds on a and b
        # after the split, we check if our rewrite-based derivation works in
        #  practice by checking against the original power(a, b) function
        # in the data bound split for multiplication, the sign of the two terms
        #  ln(a) and b is kept the same to keep the sign of the product the
        #  same, meaning the sign of b and a <1 vs =1 vs >1 have to remain the
        #  same as well
        # exprv <1 vs =1 vs >1 is thus enforced to remain the same, and we can
        #  adjust the expr bounds to reflect that
        expr_lower[exprv == 1] = 1
        expr_upper[exprv == 1] = 1
        expr_lower[_is_sign_positive_number(av) & (exprv > 1) & (expr_lower <= 1)] = (
            one_plus_eps
        )
        expr_upper[_is_sign_positive_number(av) & (exprv < 1) & (expr_upper >= 1)] = (
            one_minus_eps
        )

        exprv_log = np.log(exprv)
        expr_log_lower = np.log(expr_lower)
        expr_log_upper = np.log(expr_upper)

        # we simplify the data bounds split for ln(a) * b by not allowing the
        #  sign of ln(a) * b to change
        # we further simplify the code by working with
        #  abs(ln(a) * b) = abs(ln(a)) * abs(b)
        # and only taking the sign of ln(a) and b into account after the split
        expr_log_abs_lower, expr_log_abs_upper = (
            _where(
                _is_sign_negative_number(exprv_log), -expr_log_upper, expr_log_lower
            ),
            _where(
                _is_sign_negative_number(exprv_log), -expr_log_lower, expr_log_upper
            ),
        )

        av_log = np.log(av)
        av_log_abs = np.abs(av_log)
        bv_abs = np.abs(bv)

        # we use |ln(a)| * |b| here instead of |ln(a ** b)| since the former
        #  behaves better when a**b is rounded to zero but a != 0
        # in particular, the latter would start with ln(0) = -inf, while the
        #  former can work with finite logarithm results
        exprv_log_abs = np.multiply(av_log_abs, bv_abs)

        fmax = np.finfo(Xs.dtype).max
        smallest_subnormal = np.finfo(Xs.dtype).smallest_subnormal

        # we are given l <= e <= u, which we translate into e/lf <= e <= e*uf
        # - if the factor is infinite, we limit it to fmax
        # - if the factor is NaN, e.g. from 0/0, we set the factor to 1
        # finally we split the factor geometrically in two using the sqrt
        expr_log_abs_lower_factor: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.divide(exprv_log_abs, expr_log_abs_lower)
        )
        expr_log_abs_lower_factor[np.isinf(expr_log_abs_lower_factor)] = fmax
        np.sqrt(expr_log_abs_lower_factor, out=expr_log_abs_lower_factor)
        expr_log_abs_lower_factor[np.isnan(expr_log_abs_lower_factor)] = 1

        expr_log_abs_upper_factor: np.ndarray[tuple[Ps], np.dtype[F]] = _ensure_array(
            np.divide(
                expr_log_abs_upper,
                # we avoid division by zero here
                # - if exprv_log_abs = 0 and expr_log_abs_upper = 0, factor = 0 works
                # - if exprv_log_abs = 0 and expr_log_abs_upper > 0, factor is large
                # - if exprv_log_abs > 0 and expr_log_abs_upper > 0, works as normal
                _maximum_zero_sign_sensitive(exprv_log_abs, smallest_subnormal),
            )
        )
        expr_log_abs_upper_factor[np.isinf(expr_log_abs_upper_factor)] = fmax
        np.sqrt(expr_log_abs_upper_factor, out=expr_log_abs_upper_factor)
        expr_log_abs_upper_factor[np.isnan(expr_log_abs_upper_factor)] = 1

        # TODO: for infinite ln(a) * b that includes finite bounds, allow them

        # compute the bounds for abs(ln(a)) and abs(b)
        a_log_abs_lower = np.divide(av_log_abs, expr_log_abs_lower_factor)
        a_log_abs_upper = np.multiply(av_log_abs, expr_log_abs_upper_factor)

        b_abs_lower = np.divide(bv_abs, expr_log_abs_lower_factor)
        b_abs_upper = np.multiply(bv_abs, expr_log_abs_upper_factor)

        # derive the bounds on ln(a) and b based on the bounds on abs(ln(a))
        #  and abs(b), and on a based on ln(a)
        a_log_lower: np.ndarray[tuple[Ps], np.dtype[F]] = _where(
            _is_sign_negative_number(av_log), -a_log_abs_upper, a_log_abs_lower
        )
        a_log_upper: np.ndarray[tuple[Ps], np.dtype[F]] = _where(
            _is_sign_negative_number(av_log), -a_log_abs_lower, a_log_abs_upper
        )

        # if a_lower == av and av == -0.0, we need to guarantee
        #  that a_lower is also -0.0, same for a_upper
        a_lower = _ensure_array(_minimum_zero_sign_sensitive(av, np.exp(a_log_lower)))
        a_upper = _ensure_array(_maximum_zero_sign_sensitive(av, np.exp(a_log_upper)))

        # if b_lower == bv and bv == -0.0, we need to guarantee
        #  that b_lower is also -0.0, same for b_upper
        b_lower = _where(_is_sign_negative_number(bv), -b_abs_upper, b_abs_lower)
        b_lower = _ensure_array(_minimum_zero_sign_sensitive(bv, b_lower))
        b_upper = _where(_is_sign_negative_number(bv), -b_abs_lower, b_abs_upper)
        b_upper = _ensure_array(_maximum_zero_sign_sensitive(bv, b_upper))

        # we need to force av and bv if expr_lower == expr_upper
        np.copyto(a_lower, av, where=(expr_lower == expr_upper), casting="no")
        np.copyto(a_upper, av, where=(expr_lower == expr_upper), casting="no")
        np.copyto(b_lower, bv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(b_upper, bv, where=(expr_lower == expr_upper), casting="no")

        # 1 ** [-NaN, -inf, +inf, +NaN] = 1 -> force av = 1 and allow any bv
        a_lower[av == 1] = 1
        a_upper[av == 1] = 1
        b_lower[av == 1] = -np.inf
        b_upper[av == 1] = +np.inf

        # (<1>) ** NaN = NaN -> allow any av != 1
        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future
        a_lower[(av > 1) & np.isnan(bv)] = one_plus_eps
        a_upper[(av > 1) & np.isnan(bv)] = np.inf
        a_lower[(av < 1) & np.isnan(bv)] = -np.inf
        a_upper[(av < 1) & np.isnan(bv)] = one_minus_eps

        # handle 0 ** bv
        # - +-0 ** +-0 = 1 -> discontinuous, force av = +-0 and bv = +-0
        # - +0 ** (>0) = +0 -> allow any bv > 0
        # - +0 ** (<0) = -inf -> allow any bv < 0
        # - -0 ** (>0) = +-0 -> too tricky, just keep both av and bv the same
        # - -0 ** (<0) = +-inf -> too tricky, just keep both av and bv the same
        np.copyto(a_lower, av, where=_is_negative_zero(av), casting="no")
        np.copyto(a_upper, av, where=_is_negative_zero(av), casting="no")
        np.copyto(b_lower, bv, where=_is_negative_zero(av), casting="no")
        np.copyto(b_upper, bv, where=_is_negative_zero(av), casting="no")
        a_lower[(av == 0) & (bv == 0)] = -0.0
        a_upper[(av == 0) & (bv == 0)] = +0.0
        b_lower[(av == 0) & (bv == 0)] = -0.0
        b_upper[(av == 0) & (bv == 0)] = +0.0
        b_lower[_is_positive_zero(av) & (bv > 0)] = smallest_subnormal
        b_upper[_is_positive_zero(av) & (bv > 0)] = np.inf
        b_lower[_is_positive_zero(av) & (bv < 0)] = -np.inf
        b_upper[_is_positive_zero(av) & (bv < 0)] = -smallest_subnormal

        # [-NaN, -inf, +inf, +NaN] ** +-0 = 1 -> so force bv = +-0 and allow
        #  any av
        a_lower[bv == 0] = -np.inf
        a_upper[bv == 0] = +np.inf
        b_lower[bv == 0] = -0.0
        b_upper[bv == 0] = +0.0

        # NaN ** (<0>) = NaN -> allow any av != 0
        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future
        b_lower[np.isnan(av) & (bv > 0)] = smallest_subnormal
        b_upper[np.isnan(av) & (bv > 0)] = np.inf
        b_lower[np.isnan(av) & (bv < 0)] = -np.inf
        b_upper[np.isnan(av) & (bv < 0)] = -smallest_subnormal

        # powers of negative numbers are just too tricky, so force av and bv
        np.copyto(a_lower, av, where=(av < 0), casting="no")
        np.copyto(a_upper, av, where=(av < 0), casting="no")
        np.copyto(b_lower, bv, where=(av < 0), casting="no")
        np.copyto(b_upper, bv, where=(av < 0), casting="no")

        # flip a bounds if bv < 0; flip b bounds if av < 1
        # - av < 1 -> av ** b_lower >= av ** b_upper -> flip b bounds
        # - av < 1 & bv < 0 -> a_lower ** bv >= a_upper ** bv -> flip a bounds
        # - av < 1 & bv > 0 -> a_lower ** bv <= a_upper ** bv
        # - av > 1 -> av ** b_lower <= av ** b_upper
        # - av > 1 & bv < 0 -> a_lower ** bv >= a_upper ** bv -> flip a bounds
        # - av > 1 & bv > 0 -> a_lower ** bv <= a_upper ** bv
        # so that the below nudging works with the worst case combinations
        a_lower, a_upper = (
            _where(np.less(bv, 0), a_upper, a_lower),
            _where(np.less(bv, 0), a_lower, a_upper),
        )
        b_lower, b_upper = (
            _where(np.less(av, 1), b_upper, b_lower),
            _where(np.less(av, 1), b_lower, b_upper),
        )

        # stack the bounds on a and b so that we can nudge their bounds, if
        #  necessary, together
        tl_stack = _stack([a_lower, b_lower])
        tu_stack = _stack([a_upper, b_upper])

        def compute_term_power(
            t_stack: np.ndarray[tuple[int, Ps], np.dtype[F]],
        ) -> np.ndarray[tuple[int, Ps], np.dtype[F]]:
            total_power: np.ndarray[tuple[Ps], np.dtype[F]] = np.power(
                t_stack[0], t_stack[1]
            )

            return _broadcast_to(
                _ensure_array(total_power).reshape((1, *exprv.shape)),
                (t_stack.shape[0], *exprv.shape),
            )

        exprv = _ensure_array(exprv)
        expr_lower = _ensure_array(expr_lower)
        expr_upper = _ensure_array(expr_upper)

        tl_stack = guarantee_stacked_arg_within_expr_bounds(
            compute_term_power,
            _broadcast_to(
                exprv.reshape((1, *exprv.shape)),
                (tl_stack.shape[0], *exprv.shape),
            ),
            _stack([av, bv]),
            tl_stack,
            _broadcast_to(
                expr_lower.reshape((1, *exprv.shape)),
                (tl_stack.shape[0], *exprv.shape),
            ),
            _broadcast_to(
                expr_upper.reshape((1, *exprv.shape)),
                (tl_stack.shape[0], *exprv.shape),
            ),
        )
        tu_stack = guarantee_stacked_arg_within_expr_bounds(
            compute_term_power,
            _broadcast_to(
                exprv.reshape((1, *exprv.shape)),
                (tu_stack.shape[0], *exprv.shape),
            ),
            _stack([av, bv]),
            tu_stack,
            _broadcast_to(
                expr_lower.reshape((1, *exprv.shape)),
                (tu_stack.shape[0], *exprv.shape),
            ),
            _broadcast_to(
                expr_upper.reshape((1, *exprv.shape)),
                (tu_stack.shape[0], *exprv.shape),
            ),
        )

        # extract the bounds for a and b and undo any earlier flips
        a_lower, a_upper = (
            _where(np.less(bv, 0), tu_stack[0], tl_stack[0]),
            _where(np.less(bv, 0), tl_stack[0], tu_stack[0]),
        )
        b_lower, b_upper = (
            _where(np.less(av, 1), tu_stack[1], tl_stack[1]),
            _where(np.less(av, 1), tl_stack[1], tu_stack[1]),
        )

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
        return f"{self._a!r} ** {self._b!r}"

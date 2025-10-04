from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _broadcast_to,
    _ensure_array,
    _floating_max,
    _floating_smallest_subnormal,
    _is_sign_negative_number,
    _is_sign_positive_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _nextafter,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant
from .literal import Number
from .typing import F, Ns, Ps, PsI


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
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.power(
            self._a.eval(x, Xs, late_bound), self._b.eval(x, Xs, late_bound)
        )

    @override
    @checked_data_bounds
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
        assert not (a_const and b_const), "constant power has no data bounds"

        # evaluate a, b, and power(a, b)
        a, b = self._a, self._b
        av = a.eval(X.shape, Xs, late_bound)
        bv = b.eval(X.shape, Xs, late_bound)
        exprv: np.ndarray[Ps, np.dtype[F]] = np.power(av, bv)

        # we always bail for av < 0, which is the only way to get exprv < 0,
        #  so we can clamp expr_lower to >= +0.0 if exprv >= +0.0
        expr_lower = _ensure_array(expr_lower, copy=True)
        np.copyto(
            expr_lower,
            _maximum_zero_sign_sensitive(X.dtype.type(+0.0), expr_lower),
            where=_is_sign_positive_number(exprv),
            casting="no",
        )

        b_lower: np.ndarray[Ps, np.dtype[F]]
        b_upper: np.ndarray[Ps, np.dtype[F]]

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

            smallest_subnormal = _floating_smallest_subnormal(X.dtype)

            # 0 ** 0 = 1, so force bv = 0
            np.copyto(b_lower, bv, where=((av == 0) & (bv == 0)), casting="no")
            np.copyto(b_upper, bv, where=((av == 0) & (bv == 0)), casting="no")
            # ... and also ensure that 0 ** (!=0) doesn't become 0 ** 0
            b_lower[(av == 0) & (bv > 0)] = smallest_subnormal
            b_upper[(av == 0) & (bv < 0)] = -smallest_subnormal

            # +0 ** (>0) = 0
            #   so allow all bv with the same sign (-0 is handled later)
            # +0 ** (<0) = +inf
            b_lower[(av == 0) & (bv < 0)] = -np.inf
            b_upper[(av == 0) & (bv > 0)] = np.inf

            # 1 ** [-inf, +inf] = 1
            b_lower[(av == 1)] = -np.inf
            b_upper[(av == 1)] = np.inf

            # +inf ** 0 = 1, so force bv = 0 (-inf is handled later)
            np.copyto(b_lower, bv, where=(np.isinf(av) & (bv == 0)), casting="no")
            np.copyto(b_upper, bv, where=(np.isinf(av) & (bv == 0)), casting="no")
            # ... and also ensure that +inf ** (!=0) doesn't become +inf ** 0
            b_lower[np.isinf(av) & (bv > 0)] = smallest_subnormal
            b_upper[np.isinf(av) & (bv < 0)] = -smallest_subnormal

            # +inf ** (>0) = +inf
            #   so allow all bv with the same sign (-inf is handled later)
            # +inf ** (<0) = +0
            b_lower[np.isinf(av) & (bv < 0)] = -np.inf
            b_upper[np.isinf(av) & (bv > 0)] = np.inf

            # NaN ** 0 = 1, so force bv = 0
            np.copyto(b_lower, bv, where=(np.isnan(av) & (bv == 0)), casting="no")
            np.copyto(b_upper, bv, where=(np.isnan(av) & (bv == 0)), casting="no")
            # ... and also ensure that NaN ** (!=0) doesn't become NaN ** 0
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            b_lower[np.isnan(av) & (bv > 0)] = smallest_subnormal
            b_upper[np.isnan(av) & (bv > 0)] = np.inf
            b_lower[np.isnan(av) & (bv < 0)] = -np.inf
            b_upper[np.isnan(av) & (bv < 0)] = -smallest_subnormal

            # powers of sign-negative numbers are just too tricky, so force bv
            np.copyto(b_lower, bv, where=_is_sign_negative_number(av), casting="no")
            np.copyto(b_upper, bv, where=_is_sign_negative_number(av), casting="no")

            # we need to force bv if expr_lower == expr_upper
            np.copyto(b_lower, bv, where=(expr_lower == expr_upper), casting="no")
            np.copyto(b_upper, bv, where=(expr_lower == expr_upper), casting="no")

            # print(self, "ac", av, bv, exprv, expr_lower, expr_upper, b_lower, b_upper)

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
                X,
                Xs,
                late_bound,
            )

        # TODO: support better bounds if b is a symbolic integer

        a_lower: np.ndarray[Ps, np.dtype[F]]
        a_upper: np.ndarray[Ps, np.dtype[F]]

        if b_const:
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

            smallest_subnormal = _floating_smallest_subnormal(X.dtype)

            # 0 ** 0 = 1, so force av = 0
            np.copyto(a_lower, bv, where=((av == 0) & (bv == 0)), casting="no")
            np.copyto(a_upper, bv, where=((av == 0) & (bv == 0)), casting="no")
            # ... and also ensure that (!=0) ** 0 doesn't become 0 ** 0
            a_lower[(av > 0) & (bv == 0)] = smallest_subnormal
            a_upper[(av < 0) & (bv == 0)] = -smallest_subnormal

            # (!=0) ** 0 = 1, so allow all non-zero av,
            #  simplified to allowing all av with the same sign
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            a_lower[(av < 0) & (bv == 0)] = -np.inf
            a_upper[(av > 0) & (bv == 0)] = np.inf

            one_plus_eps = _nextafter(
                np.array(1, dtype=X.dtype), np.array(2, dtype=X.dtype)
            )
            one_minus_eps = _nextafter(
                np.array(1, dtype=X.dtype), np.array(0, dtype=X.dtype)
            )

            # 1 ** +-inf = 1, so force av = 1
            a_lower[(av == 1) & np.isinf(bv)] = 1
            a_upper[(av == 1) & np.isinf(bv)] = 1
            # ... and also ensure that (!=1) ** +-inf doesn't become 1 ** +-inf
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            a_lower[(av > 1) & np.isinf(bv)] = one_plus_eps
            a_upper[(av < 1) & np.isinf(bv)] = one_minus_eps

            # (0<1) ** +inf = +0 (a < 0 is handled later)
            # (>1) ** +inf = +inf
            # (0<1) ** -inf = +inf (a < 0 is handled later)
            # (>1) ** -inf = +0
            # so allow all av with the same sign relative to 1
            a_upper[(av > 1) & np.isinf(bv)] = np.inf
            a_lower[(av < 1) & np.isinf(bv)] = +0.0

            # 1 ** NaN = 1, so force av = 1
            a_lower[(av == 1) & np.isnan(bv)] = 1
            a_upper[(av == 1) & np.isnan(bv)] = 1
            # ... and also ensure that (!=1) ** NaN doesn't become 1 ** NaN
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            a_lower[(av > 1) & np.isnan(bv)] = one_plus_eps
            a_upper[(av > 1) & np.isnan(bv)] = np.inf
            a_lower[(av < 1) & np.isnan(bv)] = -np.inf
            a_upper[(av < 1) & np.isnan(bv)] = one_minus_eps

            # powers of sign-negative numbers are just too tricky, so force av
            np.copyto(a_lower, av, where=_is_sign_negative_number(av), casting="no")
            np.copyto(a_upper, av, where=_is_sign_negative_number(av), casting="no")

            # we need to force av if expr_lower == expr_upper
            np.copyto(a_lower, av, where=(expr_lower == expr_upper), casting="no")
            np.copyto(a_upper, av, where=(expr_lower == expr_upper), casting="no")

            # print(self, "bc", av, bv, exprv, expr_lower, expr_upper, a_lower, a_upper)

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
                X,
                Xs,
                late_bound,
            )

        exprv_log = np.log(exprv)

        expr_log_lower = _ensure_array(np.log(expr_lower))
        expr_log_lower[
            _is_sign_positive_number(exprv_log) & (expr_log_lower <= 0)
        ] = +0.0
        expr_log_upper = _ensure_array(np.log(expr_upper))
        expr_log_upper[
            _is_sign_negative_number(exprv_log) & (expr_log_upper >= 0)
        ] = -0.0
        expr_log_abs_lower, expr_log_abs_upper = (
            _ensure_array(
                _where(
                    _is_sign_negative_number(exprv_log), -expr_log_upper, expr_log_lower
                )
            ),
            _ensure_array(
                _where(
                    _is_sign_negative_number(exprv_log), -expr_log_lower, expr_log_upper
                )
            ),
        )

        av_log = np.log(av)
        av_log_abs = np.abs(av_log)
        bv_abs = np.abs(bv)
        exprv_log_abs: np.ndarray[Ps, np.dtype[F]] = _ensure_array(np.abs(exprv_log))

        fmax = _floating_max(X.dtype)
        smallest_subnormal = _floating_smallest_subnormal(X.dtype)

        expr_log_abs_lower_factor: np.ndarray[Ps, np.dtype[F]] = _ensure_array(
            np.divide(exprv_log_abs, expr_log_abs_lower)
        )
        expr_log_abs_lower_factor[np.isinf(expr_log_abs_lower_factor)] = fmax
        np.sqrt(expr_log_abs_lower_factor, out=expr_log_abs_lower_factor)
        expr_log_abs_lower_factor[np.isnan(expr_log_abs_lower_factor)] = 1

        expr_log_abs_upper_factor: np.ndarray[Ps, np.dtype[F]] = _ensure_array(
            np.divide(
                expr_log_abs_upper,
                _maximum_zero_sign_sensitive(exprv_log_abs, smallest_subnormal),
            )
        )
        expr_log_abs_upper_factor[np.isinf(expr_log_abs_upper_factor)] = fmax
        np.sqrt(expr_log_abs_upper_factor, out=expr_log_abs_upper_factor)
        expr_log_abs_upper_factor[np.isnan(expr_log_abs_upper_factor)] = 1

        a_log_abs_lower = np.divide(av_log_abs, expr_log_abs_lower_factor)
        a_log_abs_upper = np.multiply(av_log_abs, expr_log_abs_upper_factor)

        b_abs_lower = np.divide(bv_abs, expr_log_abs_lower_factor)
        b_abs_upper = np.multiply(bv_abs, expr_log_abs_upper_factor)

        a_log_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            _is_sign_negative_number(av_log), -a_log_abs_upper, a_log_abs_lower
        )
        a_log_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            _is_sign_negative_number(av_log), -a_log_abs_lower, a_log_abs_upper
        )

        a_lower = _ensure_array(_minimum_zero_sign_sensitive(av, np.exp(a_log_lower)))
        a_upper = _ensure_array(_maximum_zero_sign_sensitive(av, np.exp(a_log_upper)))

        b_lower = _where(_is_sign_negative_number(bv), -b_abs_upper, b_abs_lower)
        b_lower = _ensure_array(_minimum_zero_sign_sensitive(bv, b_lower))
        b_upper = _where(_is_sign_negative_number(bv), -b_abs_lower, b_abs_upper)
        b_upper = _ensure_array(_maximum_zero_sign_sensitive(bv, b_upper))

        # we need to force av and bv if expr_lower == expr_upper
        np.copyto(a_lower, av, where=(expr_lower == expr_upper), casting="no")
        np.copyto(a_upper, av, where=(expr_lower == expr_upper), casting="no")
        np.copyto(b_lower, bv, where=(expr_lower == expr_upper), casting="no")
        np.copyto(b_upper, bv, where=(expr_lower == expr_upper), casting="no")

        # NaN ** 0 = 1, so force bv = 0 (av will stay NaN)
        np.copyto(b_lower, bv, where=(np.isnan(av) & (bv == 0)), casting="no")
        np.copyto(b_upper, bv, where=(np.isnan(av) & (bv == 0)), casting="no")
        # ... and also ensure that NaN ** (!=0) doesn't become NaN ** 0
        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future
        b_lower[np.isnan(av) & (bv > 0)] = smallest_subnormal
        b_upper[np.isnan(av) & (bv > 0)] = np.inf
        b_lower[np.isnan(av) & (bv < 0)] = -np.inf
        b_upper[np.isnan(av) & (bv < 0)] = -smallest_subnormal

        one_plus_eps = _nextafter(
            np.array(1, dtype=X.dtype), np.array(2, dtype=X.dtype)
        )
        one_minus_eps = _nextafter(
            np.array(1, dtype=X.dtype), np.array(0, dtype=X.dtype)
        )

        # 1 ** NaN = 1, so force av = 1 (bv will stay NaN)
        a_lower[(av == 1) & np.isnan(bv)] = 1
        a_upper[(av == 1) & np.isnan(bv)] = 1
        # ... and also ensure that (!=1) ** NaN doesn't become 1 ** NaN
        # TODO: an interval union could represent that the two disjoint
        #       intervals in the future
        a_lower[(av > 1) & np.isnan(bv)] = one_plus_eps
        a_upper[(av > 1) & np.isnan(bv)] = np.inf
        a_lower[(av < 1) & np.isnan(bv)] = -np.inf
        a_upper[(av < 1) & np.isnan(bv)] = one_minus_eps

        # TODO: handle special-cases, for now be overly cautious
        # powers of sign-negative numbers are just too tricky, so force av and bv
        force_same: np.ndarray[Ps, np.dtype[np.bool]] = _is_sign_negative_number(av)
        force_same |= av == 0
        force_same |= bv == 0
        force_same |= np.isinf(av)
        force_same |= np.isinf(bv)
        np.copyto(a_lower, av, where=force_same, casting="no")
        np.copyto(a_upper, av, where=force_same, casting="no")
        np.copyto(b_lower, bv, where=force_same, casting="no")
        np.copyto(b_upper, bv, where=force_same, casting="no")

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
        tl_stack = np.stack([a_lower, b_lower])
        tu_stack = np.stack([a_upper, b_upper])

        def compute_term_power(
            t_stack: np.ndarray[tuple[int, ...], np.dtype[F]],
        ) -> np.ndarray[tuple[int, ...], np.dtype[F]]:
            total_power: np.ndarray[tuple[int, ...], np.dtype[F]] = np.power(
                t_stack[0], t_stack[1]
            )

            return _broadcast_to(
                _ensure_array(total_power).reshape((1,) + exprv.shape),
                (t_stack.shape[0],) + exprv.shape,
            )

        exprv = _ensure_array(exprv)
        expr_lower = _ensure_array(expr_lower)
        expr_upper = _ensure_array(expr_upper)

        tl_stack = guarantee_arg_within_expr_bounds(
            compute_term_power,
            _broadcast_to(
                exprv.reshape((1,) + exprv.shape),
                (tl_stack.shape[0],) + exprv.shape,
            ),
            np.stack([av, bv]),
            tl_stack,
            _broadcast_to(
                expr_lower.reshape((1,) + exprv.shape),
                (tl_stack.shape[0],) + exprv.shape,
            ),
            _broadcast_to(
                expr_upper.reshape((1,) + exprv.shape),
                (tl_stack.shape[0],) + exprv.shape,
            ),
        )
        tu_stack = guarantee_arg_within_expr_bounds(
            compute_term_power,
            _broadcast_to(
                exprv.reshape((1,) + exprv.shape),
                (tu_stack.shape[0],) + exprv.shape,
            ),
            np.stack([av, bv]),
            tu_stack,
            _broadcast_to(
                expr_lower.reshape((1,) + exprv.shape),
                (tu_stack.shape[0],) + exprv.shape,
            ),
            _broadcast_to(
                expr_upper.reshape((1,) + exprv.shape),
                (tu_stack.shape[0],) + exprv.shape,
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

        # ensure that the bounds on Xs include Xs
        Xs_lower = _minimum_zero_sign_sensitive(Xs_lower, Xs)
        Xs_upper = _maximum_zero_sign_sensitive(Xs_upper, Xs)

        return Xs_lower, Xs_upper

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} ** {self._b!r}"

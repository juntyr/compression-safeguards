from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils._compat import (
    _ensure_array,
    _floating_smallest_subnormal,
    _is_sign_negative_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _nextafter,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds, guarantee_arg_within_expr_bounds
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant
from .divmul import ScalarMultiply
from .literal import Number
from .logexp import Exponential, Logarithm, ScalarExp, ScalarLog
from .typing import F, Fi, Ns, Ps, PsI


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

        if a_const:
            av_log = np.log(av)

            # apply the inverse function to get the bounds on b
            # if b_lower == bv and bv == -0.0, we need to guarantee that
            #  b_lower is also -0.0, same for b_upper
            b_lower: np.ndarray[Ps, np.dtype[F]] = _ensure_array(
                _minimum_zero_sign_sensitive(bv, np.divide(np.log(expr_lower), av_log))
            )
            b_upper: np.ndarray[Ps, np.dtype[F]] = _ensure_array(
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

        if b_const:
            # apply the inverse function to get the bounds on a
            # if a_lower == av and av == -0.0, we need to guarantee that
            #  a_lower is also -0.0, same for a_upper
            a_lower: np.ndarray[Ps, np.dtype[F]] = _ensure_array(
                _minimum_zero_sign_sensitive(
                    av, np.power(expr_lower, np.reciprocal(bv))
                )
            )
            a_upper: np.ndarray[Ps, np.dtype[F]] = _maximum_zero_sign_sensitive(
                av, np.power(expr_upper, np.reciprocal(bv))
            )

            smallest_subnormal = _floating_smallest_subnormal(X.dtype)

            # 0 ** 0 = 1, so force av = 0
            np.copyto(a_lower, bv, where=((av == 0) & (bv == 0)), casting="no")
            np.copyto(a_upper, bv, where=((av == 0) & (bv == 0)), casting="no")
            # ... and also ensure that (!=0) ** 0 doesn't become 0 ** 0
            a_lower[(av > 0) & (bv == 0)] = smallest_subnormal
            a_upper[(av < 0) & (bv == 0)] = -smallest_subnormal

            # (!=0) ** 0 = 0, so allow all non-zero av,
            #  simplified to allowing all av with the same sign
            # TODO: an interval union could represent that the two disjoint
            #       intervals in the future
            a_lower[(av < 0) & (bv == 0)] = -np.inf
            a_upper[(av > 0) & (bv == 0)] = np.inf

            # TODO: handle inf cases

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
            a_lower[(av < 1) & np.isinf(bv)] = smallest_subnormal

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

        # print(self, "og", av, bv, exprv, expr_lower, expr_upper)

        # TODO: handle a^const and const^b more efficiently

        # we need to full-force-force a and b to stay the same when
        #  (a) av is negative: powers of negative numbers are just too tricky
        #  (b) av is NaN and bv is zero: NaN ** 0 = 1 ... why???
        #      but also for NaN ** b and b != 0 ensure that b doesn't become 0
        #  (c) av is one and bv is NaN: 1 ** NaN = 1 ... why???
        #      but also for a ** NaN and a != 1 ensure that a doesn't become 1
        # for (b) and (c) it easier to force both if isnan(a) or isnan(b), the
        #  clearer rules could be applied once power no longer uses a rewrite
        force_same: np.ndarray[Ps, np.dtype[np.bool]] = _is_sign_negative_number(av)
        # force_same |= np.isnan(av) & (bv == 0)
        # force_same |= (av == 1) & np.isnan(bv)
        force_same |= np.isnan(av) | np.isnan(bv)

        # rewrite a ** b as fake_abs(e^(b*ln(fake_abs(a))))
        # this is mathematically incorrect for a <= 0 but works for deriving
        #  data bounds since fake_abs handles the data bound flips
        rewritten = ScalarExp(
            Exponential.exp,
            ScalarMultiply(
                ForceEquivalent(self._b, force_same),
                ScalarLog(
                    Logarithm.ln,
                    ScalarFakeAbs(
                        ForceEquivalent(self._a, force_same),
                    ),
                ),
            ),
        )
        exprv_rewritten = rewritten.eval(X.shape, Xs, late_bound)

        # we also need to full-force-force a and b to stay the same when
        #  (d) isnan(exprv) != isnan(exprv_rewritten): bail out
        force_same |= np.isnan(exprv) != np.isnan(exprv_rewritten)

        # inlined outer ScalarFakeAbs
        # flip the lower/upper bounds if the result is negative
        #  since our rewrite below only works with non-negative exprv
        expr_lower, expr_upper = (
            _where(_is_sign_negative_number(exprv), -expr_upper, expr_lower),
            _where(_is_sign_negative_number(exprv), -expr_lower, expr_upper),
        )

        # ensure that the bounds at least contain the rewritten expression
        #  result
        expr_lower = _ensure_array(
            _minimum_zero_sign_sensitive(expr_lower, exprv_rewritten)
        )
        expr_upper = _ensure_array(
            _maximum_zero_sign_sensitive(expr_upper, exprv_rewritten)
        )

        # enforce bounds that only contain the rewritten expression value when
        #  we also force a and b to stay the same
        np.copyto(expr_lower, exprv_rewritten, where=force_same, casting="no")
        np.copyto(expr_upper, exprv_rewritten, where=force_same, casting="no")

        # print(
        #     self,
        #     "rw",
        #     av,
        #     bv,
        #     exprv,
        #     exprv_rewritten,
        #     force_same,
        #     expr_lower,
        #     expr_upper,
        # )

        return rewritten.compute_data_bounds(expr_lower, expr_upper, X, Xs, late_bound)

    @override
    def __repr__(self) -> str:
        return f"{self._a!r} ** {self._b!r}"


class ScalarFakeAbs(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarFakeAbs":
        return ScalarFakeAbs(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.abs, ScalarFakeAbs
        )

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.abs(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)

        # flip the lower/upper bounds if the arg is negative
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _ensure_array(expr_lower, copy=True)
        np.negative(expr_upper, out=arg_lower, where=_is_sign_negative_number(argv))

        arg_upper: np.ndarray[Ps, np.dtype[F]] = _ensure_array(expr_upper, copy=True)
        np.negative(expr_lower, out=arg_upper, where=_is_sign_negative_number(argv))

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"fake_abs({self._a!r})"


class ForceEquivalent(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a", "_force")
    _a: AnyExpr
    _force: np.ndarray[tuple[int, ...], np.dtype[np.bool]]

    def __init__(
        self, a: AnyExpr, force: np.ndarray[tuple[int, ...], np.dtype[np.bool]]
    ) -> None:
        self._a = a
        self._force = force

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ForceEquivalent":
        return ForceEquivalent(a, self._force)

    @override
    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, lambda x: x, lambda a: ForceEquivalent(a, self._force)
        )

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return self._a.eval(x, Xs, late_bound)

    @checked_data_bounds
    @override
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # evaluate arg
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)

        # force the argument bounds if requested
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _ensure_array(expr_lower, copy=True)
        np.copyto(arg_lower, argv, where=self._force, casting="no")

        arg_upper: np.ndarray[Ps, np.dtype[F]] = _ensure_array(expr_upper, copy=True)
        np.copyto(arg_upper, argv, where=self._force, casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    @override
    def __repr__(self) -> str:
        return f"force_equivalent({self._a!r})"

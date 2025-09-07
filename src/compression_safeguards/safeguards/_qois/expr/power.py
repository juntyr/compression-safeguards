from collections.abc import Mapping

import numpy as np

from ....utils._compat import (
    _is_sign_negative_number,
    _maximum_zero_sign_sensitive,
    _minimum_zero_sign_sensitive,
    _where,
)
from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .divmul import ScalarMultiply
from .literal import Number
from .logexp import Exponential, Logarithm, ScalarExp, ScalarLog
from .typing import F, Fi, Ns, Ps, PsI


class ScalarPower(Expr[Expr, Expr]):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __new__(cls, a: Expr, b: Expr):
        if isinstance(a, Number) and isinstance(b, Number):
            # symbolical constant propagation for int ** int
            # where the exponent is non-negative and the result thus is an int
            ai, bi = a.as_int(), b.as_int()
            if (ai is not None) and (bi is not None):
                if bi >= 0:
                    return Number.from_symbolic_int(ai**bi)
        this = super().__new__(cls)
        this._a = a
        this._b = b
        return this

    @property
    def args(self) -> tuple[Expr, Expr]:
        return (self._a, self._b)

    def with_args(self, a: Expr, b: Expr) -> "ScalarPower":
        return ScalarPower(a, b)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_binary(
            self._a, self._b, dtype, np.power, ScalarPower
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.power(
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
        # evaluate a, b, and power(a, b)
        a, b = self._a, self._b
        av = a.eval(X.shape, Xs, late_bound)
        bv = b.eval(X.shape, Xs, late_bound)
        exprv: np.ndarray[Ps, np.dtype[F]] = np.power(av, bv)

        # TODO: handle a^const and const^b more efficiently

        # we need to full-force-force a and b to stay the same when
        #  (a) av is negative: powers of negative numbers are just too tricky
        #  (b) av is NaN and bv is zero: NaN ** 0 = 1 ... why???
        force_same: np.ndarray[Ps, np.dtype[np.bool]] = _is_sign_negative_number(av)
        force_same |= np.isnan(av) & (bv == 0)

        # rewrite a ** b as fake_abs(e^(b*ln(fake_abs(a))))
        # this is mathematically incorrect for a <= 0 but works for deriving
        #  error bounds since fake_abs handles the error bound flips
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
        #  (c) isnan(exprv) != isnan(exprv_rewritten): bail out
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
        expr_lower = _minimum_zero_sign_sensitive(expr_lower, exprv_rewritten)
        expr_upper = _maximum_zero_sign_sensitive(expr_upper, exprv_rewritten)

        # enforce bounds that only contain the rewritten expression value when
        #  we also force a and b to stay the same
        np.copyto(expr_lower, exprv_rewritten, where=force_same, casting="no")
        np.copyto(expr_upper, exprv_rewritten, where=force_same, casting="no")

        return rewritten.compute_data_bounds(expr_lower, expr_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"{self._a!r} ** {self._b!r}"


class ScalarFakeAbs(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarFakeAbs":
        return ScalarFakeAbs(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.abs, ScalarFakeAbs
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.abs(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(expr_lower, copy=True)
        np.copyto(
            arg_lower, -expr_upper, where=_is_sign_negative_number(argv), casting="no"
        )

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(expr_upper, copy=True)
        np.copyto(
            arg_upper, -expr_lower, where=_is_sign_negative_number(argv), casting="no"
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"fake_abs({self._a!r})"


class ForceEquivalent(Expr[Expr]):
    __slots__ = ("_a", "_force")
    _a: Expr
    _force: np.ndarray[tuple[int, ...], np.dtype[np.bool]]

    def __init__(self, a: Expr, force: np.ndarray[tuple[int, ...], np.dtype[np.bool]]):
        self._a = a
        self._force = force

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ForceEquivalent":
        return ForceEquivalent(a, self._force)

    def constant_fold(self, dtype: np.dtype[Fi]) -> Fi | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, lambda x: x, lambda a: ForceEquivalent(a, self._force)
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return self._a.eval(x, Xs, late_bound)

    @checked_data_bounds
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
        arg_lower: np.ndarray[Ps, np.dtype[F]] = np.array(expr_lower, copy=True)
        np.copyto(arg_lower, argv, where=self._force, casting="no")

        arg_upper: np.ndarray[Ps, np.dtype[F]] = np.array(expr_upper, copy=True)
        np.copyto(arg_upper, argv, where=self._force, casting="no")

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def __repr__(self) -> str:
        return f"force_equivalent({self._a!r})"

import operator
from collections.abc import Mapping

import numpy as np

from ....utils._compat import _reciprocal, _where
from ....utils.bindings import Parameter
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .divmul import ScalarMultiply
from .literal import Number
from .logexp import Exponential, Logarithm, ScalarExp, ScalarLog
from .typing import F, Ns, Ps, PsI


class ScalarPower(Expr):
    __slots__ = ("_a", "_b")
    _a: Expr
    _b: Expr

    def __new__(cls, a: Expr, b: Expr):
        ab = Number.symbolic_fold_binary(a, b, operator.pow)
        if ab is not None:
            return ab
        this = super(ScalarPower, cls).__new__(cls)
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
        return ScalarPower(
            self._a.apply_array_element_offset(axis, offset),
            self._b.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants | self._b.late_bound_constants

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
        def _is_negative(
            x: np.ndarray[Ps, np.dtype[F]],
        ) -> np.ndarray[Ps, np.dtype[np.bool]]:
            # check not just for x < 0 but also for x == -0.0
            return (x < 0) | (_reciprocal(x) < 0)  # type: ignore

        # evaluate a, b, and power(a, b)
        a, b = self._a, self._b
        av = a.eval(X.shape, Xs, late_bound)
        bv = b.eval(X.shape, Xs, late_bound)
        exprv: np.ndarray[Ps, np.dtype[F]] = np.power(av, bv)

        # powers of negative numbers are just too tricky since they easily
        #  become NaN, so let's enforce bounds that only contain the original
        #  expression value
        expr_lower = _where(_is_negative(av), exprv, expr_lower)
        expr_upper = _where(_is_negative(av), exprv, expr_upper)

        # inlined outer ScalarFakeAbs
        # flip the lower/upper bounds if the result is negative
        #  since our rewrite below only works with non-negative exprv
        expr_lower, expr_upper = (
            _where(_is_negative(exprv), -expr_upper, expr_lower),
            _where(_is_negative(exprv), -expr_lower, expr_upper),
        )

        # rewrite a ** b as fake_abs(e^(b*ln(fake_abs(a))))
        # this is mathematically incorrect for a <= 0 but works for deriving
        #  error bounds since fake_abs handles the error bound flips
        return ScalarExp(
            Exponential.exp,
            ScalarMultiply(self._b, ScalarLog(Logarithm.ln, ScalarFakeAbs(self._a))),
        ).compute_data_bounds(expr_lower, expr_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"{self._a!r} ** {self._b!r}"


class ScalarFakeAbs(Expr):
    __slots__ = ("_a",)
    _a: Expr

    def __init__(self, a: Expr):
        self._a = a

    @property
    def has_data(self) -> bool:
        return self._a.has_data

    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        return self._a.data_indices

    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Expr:
        return ScalarFakeAbs(
            self._a.apply_array_element_offset(axis, offset),
        )

    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        return self._a.late_bound_constants

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

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        def _is_negative(
            x: np.ndarray[Ps, np.dtype[F]],
        ) -> np.ndarray[Ps, np.dtype[np.bool]]:
            # check not just for x < 0 but also for x == -0.0
            return (x < 0) | (_reciprocal(x) < 0)  # type: ignore

        # evaluate arg
        arg = self._a
        argv = arg.eval(X.shape, Xs, late_bound)

        # flip the lower/upper bounds if the arg is negative
        arg_lower: np.ndarray[Ps, np.dtype[F]] = _where(
            _is_negative(argv),
            -expr_upper,
            expr_lower,
        )
        arg_upper: np.ndarray[Ps, np.dtype[F]] = _where(
            _is_negative(argv),
            -expr_lower,
            expr_upper,
        )

        return arg.compute_data_bounds(
            arg_lower,
            arg_upper,
            X,
            Xs,
            late_bound,
        )

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        # fake_abs cannot cause any rounding errors
        return self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return f"fake_abs({self._a!r})"

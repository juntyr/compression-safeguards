import operator
from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from .abc import Expr
from .constfold import ScalarFoldedConstant
from .literal import Number
from .typing import F, Ns, Ps, PsI


class ScalarNegate(Expr[Expr]):
    __slots__ = ("_a",)
    _a: Expr

    def __new__(cls, a: Expr):
        na = Number.symbolic_fold_unary(a, operator.neg)
        if na is not None:
            return na
        this = super(ScalarNegate, cls).__new__(cls)
        this._a = a
        return this

    @property
    def args(self) -> tuple[Expr]:
        return (self._a,)

    def with_args(self, a: Expr) -> "ScalarNegate":
        return ScalarNegate(a)

    def constant_fold(self, dtype: np.dtype[F]) -> F | Expr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.negative, ScalarNegate
        )

    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.negative(self._a.eval(x, Xs, late_bound))

    @checked_data_bounds
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        return self._a.compute_data_bounds(-expr_upper, -expr_lower, X, Xs, late_bound)

    def __repr__(self) -> str:
        return f"-{self._a!r}"

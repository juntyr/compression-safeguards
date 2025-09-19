import operator
from collections.abc import Mapping

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ....utils.bindings import Parameter
from ..bound import checked_data_bounds
from .abc import AnyExpr, Expr
from .constfold import ScalarFoldedConstant
from .literal import Number
from .typing import F, Ns, Ps, PsI


class ScalarNegate(Expr[AnyExpr]):
    __slots__: tuple[str, ...] = ("_a",)
    _a: AnyExpr

    def __init__(self, a: AnyExpr) -> None:
        self._a = a

    def __new__(cls, a: AnyExpr) -> "ScalarNegate | Number":  # type: ignore[misc]
        na = Number.symbolic_fold_unary(a, operator.neg)
        if na is not None:
            return na
        return super().__new__(cls)

    @property
    @override
    def args(self) -> tuple[AnyExpr]:
        return (self._a,)

    @override
    def with_args(self, a: AnyExpr) -> "ScalarNegate | Number":
        return ScalarNegate(a)

    @override
    def constant_fold(self, dtype: np.dtype[F]) -> F | AnyExpr:
        return ScalarFoldedConstant.constant_fold_unary(
            self._a, dtype, np.negative, ScalarNegate
        )

    @override
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        return np.negative(self._a.eval(x, Xs, late_bound))

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
        return self._a.compute_data_bounds(-expr_upper, -expr_lower, X, Xs, late_bound)

    @override
    def __repr__(self) -> str:
        return f"-{self._a!r}"

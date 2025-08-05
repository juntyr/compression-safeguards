from collections.abc import Mapping, Set

import numpy as np

from ...utils.bindings import Parameter
from ..qois import (
    PointwiseQuantityOfInterestExpression,
    StencilQuantityOfInterestExpression,
)
from .expr.abc import Expr
from .expr.array import Array
from .expr.constfold import FoldedScalarConst
from .expr.data import Data
from .expr.typing import F, Ns, Ps
from .lexer import QoILexer
from .parser import QoIParser


class PointwiseQuantityOfInterest:
    __slots__ = ("_expr", "_late_bound_constants")
    _expr: Expr
    _late_bound_constants: frozenset[Parameter]

    def __init__(self, qoi: PointwiseQuantityOfInterestExpression):
        lexer = QoILexer()
        parser = QoIParser(x=Data(index=()), X=None, I=None)

        expr = parser.parse(qoi, lexer.tokenize(qoi))
        assert isinstance(expr, Expr)
        assert not isinstance(expr, Array), (
            f"QoI expression must be a scalar but is an array expression of shape {expr.shape}"
        )
        assert expr.has_data, "QoI expression must not be constant"

        late_bound_constants = expr.late_bound_constants

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # check if the expression is well-formed and if an error bound can
            #  be computed
            _canary_data_eb = FoldedScalarConst.constant_fold_expr(
                expr, np.dtype(np.float64)
            ).compute_data_error_bound(
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                {c: np.empty(0, dtype=np.float64) for c in late_bound_constants},
            )

        self._expr = expr
        self._late_bound_constants = late_bound_constants

    @property
    def late_bound_constants(self) -> Set[Parameter]:
        return self._late_bound_constants

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def eval(
        self,
        X: np.ndarray[Ps, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ps, np.dtype[F]]],
    ) -> np.ndarray[Ps, np.dtype[F]]:
        expr = FoldedScalarConst.constant_fold_expr(self._expr, X.dtype)
        return expr.eval(X.shape, X, late_bound)

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_data_error_bound(
        self,
        eb_qoi_lower: np.ndarray[Ps, np.dtype[F]],
        eb_qoi_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ps, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        expr = FoldedScalarConst.constant_fold_expr(self._expr, X.dtype)
        return expr.compute_data_error_bound(
            eb_qoi_lower, eb_qoi_upper, X, X, late_bound
        )

    def __repr__(self) -> str:
        return repr(self._expr)


class StencilQuantityOfInterest:
    __slots__ = ("_expr", "_stencil_shape", "_stencil_I", "_late_bound_constants")
    _expr: Expr
    _stencil_shape: tuple[int, ...]
    _stencil_I: tuple[int, ...]
    _late_bound_constants: frozenset[Parameter]

    def __init__(
        self,
        qoi: StencilQuantityOfInterestExpression,
        stencil_shape: tuple[int, ...],
        stencil_I: tuple[int, ...],
    ):
        assert len(stencil_shape) == len(stencil_I)
        assert all(s > 0 for s in stencil_shape)
        assert all(i >= 0 and i < s for i, s in zip(stencil_I, stencil_shape))

        self._stencil_shape = stencil_shape
        self._stencil_I = stencil_I

        lexer = QoILexer()
        parser = QoIParser(
            x=Data(index=stencil_I), X=Array.from_data_shape(stencil_shape), I=stencil_I
        )

        expr = parser.parse(qoi, lexer.tokenize(qoi))
        assert isinstance(expr, Expr)
        assert not isinstance(expr, Array), (
            f"QoI expression must be a scalar but is an array expression of shape {expr.shape}"
        )
        assert expr.has_data, "QoI expression must not be constant"

        late_bound_constants = expr.late_bound_constants

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # check if the expression is well-formed and if an error bound can
            #  be computed
            _canary_data_eb = FoldedScalarConst.constant_fold_expr(
                expr, np.dtype(np.float64)
            ).compute_data_error_bound(
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,) + stencil_shape, dtype=np.float64),
                {
                    c: np.empty((0,) + stencil_shape, dtype=np.float64)
                    for c in late_bound_constants
                },
            )

        self._expr = expr
        self._late_bound_constants = late_bound_constants

    @property
    def late_bound_constants(self) -> Set[Parameter]:
        return self._late_bound_constants

    @property
    def data_indices(self) -> Set[tuple[int, ...]]:
        return self._expr.data_indices

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def eval(
        self,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[Ps, np.dtype[F]]:
        X_shape: Ps = Xs.shape[: -len(self._stencil_shape)]  # type: ignore
        stencil_shape = Xs.shape[-len(self._stencil_shape) :]
        assert stencil_shape == self._stencil_shape
        expr = FoldedScalarConst.constant_fold_expr(self._expr, Xs.dtype)
        return expr.eval(X_shape, Xs, late_bound)

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_data_error_bound(
        self,
        eb_qoi_lower: np.ndarray[Ps, np.dtype[F]],
        eb_qoi_upper: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        X_shape, stencil_shape = (
            Xs.shape[: -len(self._stencil_shape)],
            Xs.shape[-len(self._stencil_shape) :],
        )
        assert X_shape == eb_qoi_lower.shape
        assert stencil_shape == self._stencil_shape
        X: np.ndarray[Ps, np.dtype[F]] = Xs[(...,) + self._stencil_I]  # type: ignore
        expr = FoldedScalarConst.constant_fold_expr(self._expr, Xs.dtype)
        return expr.compute_data_error_bound(
            eb_qoi_lower, eb_qoi_upper, X, Xs, late_bound
        )

    def __repr__(self) -> str:
        return repr(self._expr)

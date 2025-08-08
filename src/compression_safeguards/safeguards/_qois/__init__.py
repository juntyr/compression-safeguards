__all__ = ["PointwiseQuantityOfInterest", "StencilQuantityOfInterest"]

from collections.abc import Mapping, Set

import numpy as np

from ...utils.bindings import Parameter
from ..qois import (
    PointwiseQuantityOfInterestExpression,
    StencilQuantityOfInterestExpression,
)
from .expr.abc import Expr
from .expr.array import Array
from .expr.constfold import ScalarFoldedConstant
from .expr.data import Data
from .expr.typing import F, Ns, Ps
from .lexer import QoILexer
from .parser import QoIParser


class PointwiseQuantityOfInterest:
    """
    Pointwise quantity of interest, which handles parsing, evaluation, and
    error bound propagation for the QoI expression.

    Parameters
    ----------
    qoi : PointwiseQuantityOfInterestExpression
        The pointwise quantity of interest in [`str`][str]ing form.
    """

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
            _canary_data_bounds = ScalarFoldedConstant.constant_fold_expr(
                expr, np.dtype(np.float64)
            ).compute_data_bounds(
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
        """
        The set of late-bound constant parameters that this QoI uses.
        """

        return self._late_bound_constants

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def eval(
        self,
        X: np.ndarray[Ps, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ps, np.dtype[F]]],
    ) -> np.ndarray[Ps, np.dtype[F]]:
        """
        Evaluate this pointwise quantity of interest on the data `X`.

        Parameters
        ----------
        X : np.ndarray[Ps, np.dtype[F]]
            The pointwise data, in floating point format.
        late_bound : Mapping[Parameter, np.ndarray[Ps, np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating point dtype as the data.

        Returns
        -------
        qoi : np.ndarray[Ps, np.dtype[F]]
            The pointwise quantity of interest values.
        """

        expr = ScalarFoldedConstant.constant_fold_expr(self._expr, X.dtype)
        return expr.eval(X.shape, X, late_bound)

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_data_bounds(
        self,
        qoi_lower: np.ndarray[Ps, np.dtype[F]],
        qoi_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ps, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        """
        Compute the lower-upper bounds on the data `X` that satisfy the
        lower-upper bounds `qoi_lower` and `qoi_lower` on the QoI.

        Parameters
        ----------
        qoi_lower : np.ndarray[Ps, np.dtype[F]]
            The pointwise lower bound on the QoI.
        qoi_upper : np.ndarray[Ps, np.dtype[F]]
            The pointwise upper bound on the QoI.
        X : np.ndarray[Ps, np.dtype[F]]
            The pointwise data, in floating point format.
        late_bound : Mapping[Parameter, np.ndarray[Ps, np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating point dtype as the data.

        Returns
        -------
        X_lower, X_upper : tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]
            The pointwise lower and upper bounds on the data `X`.
        """

        expr = ScalarFoldedConstant.constant_fold_expr(self._expr, X.dtype)
        return expr.compute_data_bounds(qoi_lower, qoi_upper, X, X, late_bound)

    def __repr__(self) -> str:
        return repr(self._expr)


class StencilQuantityOfInterest:
    """
    Stencil quantity of interest, which handles parsing, evaluation, and
    error bound propagation for the QoI expression.

    Parameters
    ----------
    qoi : StencilQuantityOfInterestExpression
        The stencil quantity of interest in [`str`][str]ing form.
    stencil_shape : tuple[int, ...]
        The shape of the stencil neighbourhood.
    stencil_I : tuple[int, ...]
        The index `I` for the centre of the stencil neighbourhood.
    """

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
            _canary_data_bounds = ScalarFoldedConstant.constant_fold_expr(
                expr, np.dtype(np.float64)
            ).compute_data_bounds(
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
        """
        The set of late-bound constant parameters that this QoI uses.
        """

        return self._late_bound_constants

    @property
    def data_indices(self) -> Set[tuple[int, ...]]:
        """
        The set of data stencil indices `X[is]` that this QoI uses.
        """

        return self._expr.data_indices

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def eval(
        self,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[Ps, np.dtype[F]]:
        """
        Evaluate this stencil quantity of interest on the stencil-extended data
        `Xs`.

        Parameters
        ----------
        Xs : np.ndarray[Ns, np.dtype[F]]
            The stencil-extended data, in floating point format, which must be
            of shape [...Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating point dtype as the stencil-extended data.

        Returns
        -------
        qoi : np.ndarray[Ps, np.dtype[F]]
            The pointwise quantity of interest values.
        """

        X_shape: Ps = Xs.shape[: -len(self._stencil_shape)]  # type: ignore
        stencil_shape = Xs.shape[-len(self._stencil_shape) :]
        assert stencil_shape == self._stencil_shape
        expr = ScalarFoldedConstant.constant_fold_expr(self._expr, Xs.dtype)
        return expr.eval(X_shape, Xs, late_bound)

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_data_bounds(
        self,
        qoi_lower: np.ndarray[Ps, np.dtype[F]],
        qoi_upper: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
        # FIXME: returned arrays should be of shape Ns
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        """
        Compute the lower-upper bounds on the stencil-extended data `Xs` that
        satisfy the lower-upper error bounds `qoi_lower` and `qoi_lower` on the
        QoI.

        Parameters
        ----------
        qoi_lower : np.ndarray[Ps, np.dtype[F]]
            The pointwise lower bound on the QoI.
        qoi_upper : np.ndarray[Ps, np.dtype[F]]
            The pointwise upper bound on the QoI.
        Xs : np.ndarray[Ps, np.dtype[F]]
            The stencil-extended data, in floating point format, which must be
            of shape [...Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating point dtype as the stencil-extended data.

        Returns
        -------
        X_lower, X_upper : tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]
            The stencil-extended lower and upper bounds on the data `X`.

            The bounds have not yet been combined across neighbouring points
            that contribute to the same QoI points.
        """

        X_shape, stencil_shape = (
            Xs.shape[: -len(self._stencil_shape)],
            Xs.shape[-len(self._stencil_shape) :],
        )
        assert X_shape == qoi_lower.shape
        assert stencil_shape == self._stencil_shape
        X: np.ndarray[Ps, np.dtype[F]] = Xs[(...,) + self._stencil_I]  # type: ignore
        expr = ScalarFoldedConstant.constant_fold_expr(self._expr, Xs.dtype)
        raise NotImplementedError
        return expr.compute_data_error_bound(qoi_lower, qoi_upper, X, Xs, late_bound)

    def __repr__(self) -> str:
        return repr(self._expr)

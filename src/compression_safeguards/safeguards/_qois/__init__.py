__all__ = ["PointwiseQuantityOfInterest", "StencilQuantityOfInterest"]

from collections.abc import Mapping, Set

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _ensure_array, _zeros
from ...utils.bindings import Parameter
from ...utils.error import ctx
from ..qois import (
    PointwiseQuantityOfInterestExpression,
    StencilQuantityOfInterestExpression,
)
from .expr.abc import AnyExpr, Expr
from .expr.array import Array
from .expr.constfold import ScalarFoldedConstant
from .expr.data import Data
from .lexer import QoILexer
from .parser import QoIParser
from .typing import F, Ns, Ps, coerce_to_flat, np_sndarray


class PointwiseQuantityOfInterest:
    """
    Pointwise quantity of interest, which handles parsing, evaluation, and
    data bound propagation for the QoI expression.

    Parameters
    ----------
    qoi : PointwiseQuantityOfInterestExpression
        The pointwise quantity of interest in [`str`][str]ing form.

    Raises
    ------
    SyntaxError
        if the `qoi` is not a valid pointwise quantity of interest expression.
    """

    __slots__: tuple[str, ...] = ("_expr", "_late_bound_constants")
    _expr: AnyExpr
    _late_bound_constants: frozenset[Parameter]

    def __init__(self, qoi: PointwiseQuantityOfInterestExpression):
        lexer = QoILexer()
        parser = QoIParser(x=Data.SCALAR, X=None, I=None)

        expr = parser.parse2(qoi, lexer.tokenize(qoi))
        assert isinstance(expr, Expr)

        if isinstance(expr, Array):
            raise (
                SyntaxError(
                    "QoI expression must evaluate to a scalar, not an array "
                    + f"expression of shape {expr.shape}",
                    ("<qoi>", None, None, None),
                )
                | ctx
            )

        if not expr.has_data:
            raise (
                SyntaxError(
                    "QoI expression must not be constant", ("<qoi>", None, None, None)
                )
                | ctx
            )

        late_bound_constants = expr.late_bound_constants

        dummy_pointwise: np.ndarray[tuple[int], np.dtype[np.float64]] = _zeros(
            (0,), np.dtype(np.float64)
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # check if the expression is well-formed and if data bounds can be
            #  computed
            _canary_expr = expr.constant_fold(np.dtype(np.float64))
            if isinstance(_canary_expr, Expr):
                _canary_data_bounds = _canary_expr.compute_data_bounds(
                    dummy_pointwise,
                    dummy_pointwise,
                    dummy_pointwise,
                    {c: dummy_pointwise for c in late_bound_constants},
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
        X: np.ndarray[tuple[Ps], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[Ps], np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        """
        Evaluate this pointwise quantity of interest on the data `X`.

        Parameters
        ----------
        X : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise data, in floating-point format.
        late_bound : Mapping[Parameter, np.ndarray[tuple[Ps], np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating-point dtype as the data.

        Returns
        -------
        qoi : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise quantity of interest values.
        """

        X = _ensure_array(X)
        expr = ScalarFoldedConstant.constant_fold_expr(self._expr, X.dtype)
        exprv = _ensure_array(expr.eval(X, late_bound))
        assert exprv.dtype == X.dtype
        assert exprv.shape == X.shape
        return exprv

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_data_bounds(
        self,
        qoi_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        qoi_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        X: np.ndarray[tuple[Ps], np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[tuple[Ps], np.dtype[F]]],
    ) -> tuple[np.ndarray[tuple[Ps], np.dtype[F]], np.ndarray[tuple[Ps], np.dtype[F]]]:
        """
        Compute the lower-upper bounds on the data `X` that satisfy the
        lower-upper bounds `qoi_lower` and `qoi_lower` on the QoI.

        Parameters
        ----------
        qoi_lower : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise lower bound on the QoI.
        qoi_upper : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise upper bound on the QoI.
        X : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise data, in floating-point format.
        late_bound : Mapping[Parameter, np.ndarray[tuple[Ps], np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating-point dtype as the data.

        Returns
        -------
        X_lower, X_upper : tuple[np.ndarray[tuple[Ps], np.dtype[F]], np.ndarray[tuple[Ps], np.dtype[F]]]
            The pointwise lower and upper bounds on the data `X`.
        """

        qoi_lower = _ensure_array(qoi_lower)
        qoi_upper = _ensure_array(qoi_upper)
        X = _ensure_array(X)
        X_lower: np.ndarray[tuple[Ps], np.dtype[F]]
        X_upper: np.ndarray[tuple[Ps], np.dtype[F]]
        expr = self._expr.constant_fold(X.dtype)
        if isinstance(expr, Expr):
            X_lower_upper: tuple[
                np_sndarray[Ps, tuple[()], np.dtype[F]],
                np_sndarray[Ps, tuple[()], np.dtype[F]],
            ] = expr.compute_data_bounds(qoi_lower, qoi_upper, X, late_bound)
            X_lower = _ensure_array(coerce_to_flat(X_lower_upper[0]))
            X_upper = _ensure_array(coerce_to_flat(X_lower_upper[1]))
        else:
            # constant fold with combinators can create top-level folded consts
            X_lower = np.full(X.shape, X.dtype.type(-np.inf))
            X_lower[np.isnan(X)] = np.nan
            X_upper = np.full(X.shape, X.dtype.type(np.inf))
            X_upper[np.isnan(X)] = np.nan
        assert X_lower.dtype == X.dtype
        assert X_upper.dtype == X.dtype
        assert X_lower.shape == X.shape
        assert X_upper.shape == X.shape
        return X_lower, X_upper

    @override
    def __repr__(self) -> str:
        return repr(self._expr)


class StencilQuantityOfInterest:
    """
    Stencil quantity of interest, which handles parsing, evaluation, and
    data bound propagation for the QoI expression.

    Parameters
    ----------
    qoi : StencilQuantityOfInterestExpression
        The stencil quantity of interest in [`str`][str]ing form.
    stencil_shape : tuple[int, ...]
        The shape of the stencil neighbourhood.
    stencil_I : tuple[int, ...]
        The index `I` for the centre of the stencil neighbourhood.

    Raises
    ------
    SyntaxError
        if the `qoi` is not a valid stencil quantity of interest expression.
    """

    __slots__: tuple[str, ...] = (
        "_expr",
        "_stencil_shape",
        "_stencil_I",
        "_late_bound_constants",
    )
    _expr: AnyExpr
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

        expr = parser.parse2(qoi, lexer.tokenize(qoi))
        assert isinstance(expr, Expr)

        if isinstance(expr, Array):
            raise (
                SyntaxError(
                    "QoI expression must evaluate to a scalar, not an array "
                    + f"expression of shape {expr.shape}",
                    ("<qoi>", None, None, None),
                )
                | ctx
            )

        if not expr.has_data:
            raise (
                SyntaxError(
                    "QoI expression must not be constant", ("<qoi>", None, None, None)
                )
                | ctx
            )

        late_bound_constants = expr.late_bound_constants

        dummy_pointwise: np.ndarray[tuple[int], np.dtype[np.float64]] = _zeros(
            (0,), np.dtype(np.float64)
        )
        dummy_stencil: np_sndarray[int, tuple[int, ...], np.dtype[np.float64]] = _zeros(
            (0, *stencil_shape), np.dtype(np.float64)
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # check if the expression is well-formed and if data bounds can be
            #  computed
            _canary_expr = expr.constant_fold(np.dtype(np.float64))
            if isinstance(_canary_expr, Expr):
                _canary_data_bounds = _canary_expr.compute_data_bounds(
                    dummy_pointwise,
                    dummy_pointwise,
                    dummy_stencil,
                    {c: dummy_stencil for c in late_bound_constants},
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
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> np.ndarray[tuple[Ps], np.dtype[F]]:
        """
        Evaluate this stencil quantity of interest on the stencil-extended data
        `Xs`.

        Parameters
        ----------
        Xs : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended data, in floating-point format, which must be
            of shape [Ps, *stencil_shape].
        late_bound : Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating-point dtype as the stencil-extended data.

        Returns
        -------
        qoi : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise quantity of interest values.
        """

        Xs = _ensure_array(Xs)
        assert Xs.shape[1:] == self._stencil_shape
        expr = ScalarFoldedConstant.constant_fold_expr(self._expr, Xs.dtype)
        exprv = _ensure_array(expr.eval(Xs, late_bound))
        assert exprv.dtype == Xs.dtype
        assert exprv.shape == Xs.shape[:1]
        return exprv

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_data_bounds(
        self,
        qoi_lower: np.ndarray[tuple[Ps], np.dtype[F]],
        qoi_upper: np.ndarray[tuple[Ps], np.dtype[F]],
        Xs: np_sndarray[Ps, Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]],
    ) -> tuple[
        np_sndarray[Ps, Ns, np.dtype[F]],
        np_sndarray[Ps, Ns, np.dtype[F]],
    ]:
        """
        Compute the lower-upper bounds on the stencil-extended data `Xs` that
        satisfy the lower-upper bounds `qoi_lower` and `qoi_lower` on the QoI.

        Parameters
        ----------
        qoi_lower : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise lower bound on the QoI.
        qoi_upper : np.ndarray[tuple[Ps], np.dtype[F]]
            The pointwise upper bound on the QoI.
        Xs : np_sndarray[Ps, Ns, np.dtype[F]]
            The stencil-extended data, in floating-point format, which must be
            of shape [Ps, *stencil_shape].
        late_bound : Mapping[Parameter, np_sndarray[Ps, Ns, np.dtype[F]]]
            The late-bound constants parameters for this QoI, with the same
            shape and floating-point dtype as the stencil-extended data.

        Returns
        -------
        Xs_lower, Xs_upper : tuple[np_sndarray[Ps, Ns, np.dtype[F]], np_sndarray[Ps, Ns, np.dtype[F]]]
            The stencil-extended lower and upper bounds on the stencil-extended
            data `Xs`.

            The bounds have not yet been combined across neighbouring points
            that contribute to the same QoI points.
        """

        qoi_lower = _ensure_array(qoi_lower)
        qoi_upper = _ensure_array(qoi_upper)
        Xs = _ensure_array(Xs)
        assert Xs.shape[:1] == qoi_lower.shape
        assert Xs.shape[1:] == self._stencil_shape
        Xs_lower: np_sndarray[Ps, Ns, np.dtype[F]]
        Xs_upper: np_sndarray[Ps, Ns, np.dtype[F]]
        expr = self._expr.constant_fold(Xs.dtype)
        if isinstance(expr, Expr):
            Xs_lower, Xs_upper = expr.compute_data_bounds(
                qoi_lower, qoi_upper, Xs, late_bound
            )
            Xs_lower = _ensure_array(Xs_lower)
            Xs_upper = _ensure_array(Xs_upper)
        else:
            # constant fold with combinators can create top-level folded consts
            Xs_lower = np.full(Xs.shape, Xs.dtype.type(-np.inf))
            Xs_lower[np.isnan(Xs)] = np.nan
            Xs_upper = np.full(Xs.shape, Xs.dtype.type(np.inf))
            Xs_upper[np.isnan(Xs)] = np.nan
        assert Xs_lower.dtype == Xs.dtype
        assert Xs_upper.dtype == Xs.dtype
        assert Xs_lower.shape == Xs.shape
        assert Xs_upper.shape == Xs.shape
        return Xs_lower, Xs_upper

    @override
    def __repr__(self) -> str:
        return repr(self._expr)

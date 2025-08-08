from abc import abstractmethod
from collections.abc import Mapping

import numpy as np

from ....utils.bindings import Parameter
from ....utils.cast import _float128_dtype, _float128_max, _isfinite
from ..eb import ensure_bounded_derived_error, ensure_bounded_expression
from .typing import F, Ns, Ps, PsI


class Expr:
    """
    Abstract base class for the quantity of interest expression abstract syntax
    tree.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def has_data(self) -> bool:
        """
        Does this expression reference the data `x` or `X[i]`?
        """

    @property
    @abstractmethod
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        """
        The set of data indices `X[is]` that this expression uses.
        """

    @abstractmethod
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> "Expr":
        """
        Apply an `offset` to the array element indices along the given `axis`.

        This method applies to data and late-bound constants.

        Parameters
        ----------
        axis : int
            The axis along which the array element indices are offset.
        offset : int
            The offset that is applied to the array element indices.

        Returns
        -------
        expr : Expr
            The modified expression.
        """

    @property
    @abstractmethod
    def late_bound_constants(self) -> frozenset[Parameter]:
        """
        The set of late-bound constant parameters that this expression uses.
        """

    @abstractmethod
    def constant_fold(self, dtype: np.dtype[F]) -> F | "Expr":
        """
        Apply scalar constant folding for the given `dtype` to this expression.

        Parameters
        ----------
        dtype : np.dtype[F]
            The dtype for which constant expressions are evaluated and folded.

        Returns
        -------
        folded : F | Expr
            The constant-folded scalar number or expression, which may have
            been simplified.
        """

    @abstractmethod
    def eval(
        self,
        x: PsI,
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> np.ndarray[PsI, np.dtype[F]]:
        """
        Evaluate this expression on the stencil-extended data `Xs`.

        Parameters
        ----------
        x : PsI
            The shape of the pointwise data.
        Xs : np.ndarray[Ns, np.dtype[F]]
            The stencil-extended data, in floating point format, which must be
            of shape [...PsI, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating point dtype as the stencil-extended data.

        Returns
        -------
        vals : np.ndarray[PsI, np.dtype[F]]
            The pointwise expression values.
        """

    @abstractmethod
    def compute_data_error_bound_unchecked(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        """
        Compute the lower-upper error bound on the stencil-extended data `Xs`
        that satisfies the lower-upper error bounds `eb_expr_lower` and
        `eb_expr_lower` on this expression.

        This method is allowed to return slightly wrongly-rounded results
        that are then corrected by
        [`compute_data_error_bound`][compression_safeguards.safeguards._qois.expr.abc.Expr.compute_data_error_bound].

        Parameters
        ----------
        eb_expr_lower : np.ndarray[Ps, np.dtype[F]]
            The pointwise lower bound on the error on this expression.
        eb_expr_upper : np.ndarray[Ps, np.dtype[F]]
            The pointwise upper bound on the error on this expression.
        X : np.ndarray[Ps, np.dtype[F]]
            The pointwise data, in floating point format, which must be
            of shape Ps.
        Xs : np.ndarray[Ps, np.dtype[F]]
            The stencil-extended data, in floating point format, which must be
            of shape [...Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating point dtype as the stencil-extended data.

        Returns
        -------
        eb_X_lower, eb_X_upper : tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]
            The pointwise lower and upper bounds on the error on the data `X`.

            The error bounds have not yet reduced across neighbouring points
            that contribute to the same expression points and thus need combine
            their error bounds.
        """

    def compute_data_error_bound(
        self,
        eb_expr_lower: np.ndarray[Ps, np.dtype[F]],
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        """
        Compute the lower-upper error bound on the stencil-extended data `Xs`
        that satisfies the lower-upper error bounds `eb_expr_lower` and
        `eb_expr_lower` on this expression.

        This method, by default, calls into
        [`compute_data_error_bound_unchecked`][compression_safeguards.safeguards._qois.expr.abc.Expr.compute_data_error_bound_unchecked]
        and then applies extensive rounding checks to ensure that the returned
        error bounds satisfy the error bounds on this expression.

        If an implementation of
        [`compute_data_error_bound_unchecked`][compression_safeguards.safeguards._qois.expr.abc.Expr.compute_data_error_bound_unchecked]
        is known to have no rounding errors, this default implementation can be
        overridden to forward its results without further rounding checks.

        Parameters
        ----------
        eb_expr_lower : np.ndarray[Ps, np.dtype[F]]
            The pointwise lower bound on the error on this expression.
        eb_expr_upper : np.ndarray[Ps, np.dtype[F]]
            The pointwise upper bound on the error on this expression.
        X : np.ndarray[Ps, np.dtype[F]]
            The pointwise data, in floating point format, which must be
            of shape Ps.
        Xs : np.ndarray[Ps, np.dtype[F]]
            The stencil-extended data, in floating point format, which must be
            of shape [...Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating point dtype as the stencil-extended data.

        Returns
        -------
        eb_X_lower, eb_X_upper : tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]
            The pointwise lower and upper bounds on the error on the data `X`.

            The error bounds have not yet reduced across neighbouring points
            that contribute to the same expression points and thus need combine
            their error bounds.
        """

        tl: np.ndarray[Ps, np.dtype[F]]
        tu: np.ndarray[Ps, np.dtype[F]]
        tl, tu = self.compute_data_error_bound_unchecked(
            eb_expr_lower, eb_expr_upper, X, Xs, late_bound
        )

        exprv = self.eval(X.shape, Xs, late_bound)

        # handle rounding errors in the lower error bound computation
        tl = ensure_bounded_derived_error(
            lambda tl: np.where(  # type: ignore
                tl == 0,
                exprv,
                self.eval(
                    X.shape,
                    Xs + tl.reshape(list(X.shape) + [1] * (Xs.ndim - X.ndim)),
                    late_bound,
                ),
            ),
            exprv,
            X,
            tl,
            eb_expr_lower,
            eb_expr_upper,
        )
        tu = ensure_bounded_derived_error(
            lambda tu: np.where(  # type: ignore
                tu == 0,
                exprv,
                self.eval(
                    X.shape,
                    Xs + tu.reshape(list(X.shape) + [1] * (Xs.ndim - X.ndim)),
                    late_bound,
                ),
            ),
            exprv,
            X,
            tu,
            eb_expr_lower,
            eb_expr_upper,
        )

        return tl, tu

    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        exprv = self.eval(X.shape, Xs, late_bound)

        eb_expr_lower: np.ndarray[Ps, np.dtype[F]] = expr_lower - exprv
        eb_expr_upper: np.ndarray[Ps, np.dtype[F]] = expr_upper - exprv

        fmax = _float128_max if X.dtype == _float128_dtype else np.finfo(X.dtype).max

        eb_expr_lower = np.where(
            ~_isfinite(eb_expr_lower) & (expr_lower < exprv), -fmax, eb_expr_lower
        )  # type: ignore
        eb_expr_upper = np.where(
            ~_isfinite(eb_expr_upper) & (expr_upper > exprv), fmax, eb_expr_upper
        )  # type: ignore

        eb_X_lower, eb_X_upper = self.compute_data_error_bound_unchecked(
            eb_expr_lower,
            eb_expr_upper,
            X,
            Xs,
            late_bound,
        )

        X_lower: np.ndarray[Ps, np.dtype[F]] = np.where(
            eb_X_lower == 0,
            X,
            X + eb_X_lower,
        )  # type: ignore
        X_upper: np.ndarray[Ps, np.dtype[F]] = np.where(
            eb_X_upper == 0,
            X,
            X + eb_X_upper,
        )  # type: ignore

        return X_lower, X_upper

    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ps, np.dtype[F]], np.ndarray[Ps, np.dtype[F]]]:
        xl: np.ndarray[Ps, np.dtype[F]]
        xu: np.ndarray[Ps, np.dtype[F]]
        xl, xu = self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

        exprv = self.eval(X.shape, Xs, late_bound)

        # handle rounding errors in the lower error bound computation
        xl = ensure_bounded_expression(
            lambda xl: self.eval(
                X.shape,
                xl.reshape(list(X.shape) + [1] * (Xs.ndim - X.ndim)),
                late_bound,
            ),  # type: ignore
            exprv,
            X,
            xl,
            expr_lower,
            expr_upper,
        )
        xu = ensure_bounded_derived_error(
            lambda xu: self.eval(
                X.shape,
                xu.reshape(list(X.shape) + [1] * (Xs.ndim - X.ndim)),
                late_bound,
            ),  # type: ignore
            exprv,
            X,
            xu,
            expr_lower,
            expr_upper,
        )

        return xl, xu

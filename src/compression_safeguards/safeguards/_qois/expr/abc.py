from abc import abstractmethod
from collections.abc import Mapping
from typing import final

import numpy as np

from ....utils._compat import _maximum, _minimum
from ....utils.bindings import Parameter
from ..bound import are_data_bounds_guaranteed, guarantee_data_within_expr_bounds
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
    def compute_data_bounds_unchecked(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        """
        Compute the lower-upper bounds on the stencil-extended data `Xs` that
        that satisfy the lower-upper bounds `expr_lower` and `expr_lower` on
        this expression.

        This method should *not* be called manually.

        This method is allowed to return slightly wrongly-rounded results
        that are then corrected by
        [`compute_data_bounds`][compression_safeguards.safeguards._qois.expr.abc.Expr.compute_data_bounds].

        If this method is known to have no rounding errors and always return
        the correct data bounds, it can be decorated with
        [`@guaranteed_data_bounds`][compression_safeguards.safeguards._qois.bound.guaranteed_data_bounds].

        Parameters
        ----------
        expr_lower : np.ndarray[Ps, np.dtype[F]]
            The pointwise lower bound on this expression.
        expr_upper : np.ndarray[Ps, np.dtype[F]]
            The pointwise upper bound on this expression.
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
        Xs_lower, Xs_upper : tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]
            The stencil-extended lower and upper bounds on the stencil-extended
            data `Xs`.

            The bounds have not yet been combined across neighbouring points
            that contribute to the same QoI points.
        """

    @final
    def compute_data_bounds(
        self,
        expr_lower: np.ndarray[Ps, np.dtype[F]],
        expr_upper: np.ndarray[Ps, np.dtype[F]],
        X: np.ndarray[Ps, np.dtype[F]],
        Xs: np.ndarray[Ns, np.dtype[F]],
        late_bound: Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]],
    ) -> tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]:
        """
        Compute the lower-upper bounds on the stencil-extended data `Xs` that
        that satisfy the lower-upper bounds `expr_lower` and `expr_lower` on
        this expression.

        This method, by default, calls into
        [`compute_data_bounds_unchecked`][compression_safeguards.safeguards._qois.expr.abc.Expr.compute_data_bounds_unchecked]
        and then applies extensive rounding checks to ensure that the returned
        bounds satisfy the bounds on this expression.

        Parameters
        ----------
        expr_lower : np.ndarray[Ps, np.dtype[F]]
            The pointwise lower bound on this expression.
        expr_upper : np.ndarray[Ps, np.dtype[F]]
            The pointwise upper bound on this expression.
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
        Xs_lower, Xs_upper : tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]
            The stencil-extended lower and upper bounds on the stencil-extended
            data `Xs`.

            The bounds have not yet been combined across neighbouring points
            that contribute to the same QoI points.
        """

        Xs_lower: np.ndarray[Ns, np.dtype[F]]
        Xs_upper: np.ndarray[Ns, np.dtype[F]]
        Xs_lower, Xs_upper = self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

        # ensure that the original data values are within the data bounds
        Xs_lower = _minimum(Xs, Xs_lower)
        Xs_upper = _maximum(Xs, Xs_upper)

        exprv = self.eval(X.shape, Xs, late_bound)

        # handle rounding errors in the lower error bound computation
        Xs_lower = guarantee_data_within_expr_bounds(
            lambda Xs_lower: self.eval(
                X.shape,
                Xs_lower,
                late_bound,
            ),
            exprv,
            Xs,
            Xs_lower,
            expr_lower,
            expr_upper,
            warn_on_bounds_exceeded=are_data_bounds_guaranteed(
                self.compute_data_bounds_unchecked
            ),
        )
        Xs_upper = guarantee_data_within_expr_bounds(
            lambda Xs_upper: self.eval(
                X.shape,
                Xs_upper,
                late_bound,
            ),
            exprv,
            Xs,
            Xs_upper,
            expr_lower,
            expr_upper,
            warn_on_bounds_exceeded=are_data_bounds_guaranteed(
                self.compute_data_bounds_unchecked
            ),
        )

        return Xs_lower, Xs_upper

    @abstractmethod
    def __repr__(self) -> str:
        pass

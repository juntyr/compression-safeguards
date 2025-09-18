from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Generic, final
from warnings import warn

import numpy as np
from typing_extensions import (
    Self,  # MSPV 3.11
    Unpack,  # MSPV 3.11
    assert_never,  # MSPV 3.11
)

from ....utils._compat import _maximum_zero_sign_sensitive, _minimum_zero_sign_sensitive
from ....utils.bindings import Parameter
from ..bound import DataBounds, data_bounds_checks, guarantee_data_within_expr_bounds
from .typing import Es, F, Ns, Ps, PsI


class Expr(ABC, Generic[Unpack[Es]]):
    """
    Abstract base class for the quantity of interest expression abstract syntax
    tree.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def args(self) -> tuple[Unpack[Es]]:
        """
        The sub-expression arguments of this expression.
        """

    @abstractmethod
    def with_args(self, *args: Unpack[Es]) -> Self:
        """
        Reconstruct this expression with different sub-expression arguments.

        Parameters
        ----------
        *args : Unpack[Es]
            The modified sub-expression arguments, derived from
            [`self.args`][compression_safeguards.safeguards._qois.expr.abc.Expr.abc].

        Returns
        -------
        expr : Self
            The modified expression.
        """

    @final
    def map_expr(self, m: "Callable[[Expr], Expr]") -> "Expr":
        """
        Recursively maps the expression mapping function `m` over this
        expression and its sub-expression arguments.

        Parameters
        ----------
        m : Callable[[Expr], Expr]
            The expression mapper, which is applied to an expression whose
            sub-expression arguments have already been mapped, i.e. the mapper
            is *not* responsible for recursion.

        Returns
        -------
        expr : Self
            The mapped expression.
        """

        args: tuple[Expr, ...] = self.args  # type: ignore
        mapped_args: tuple[Expr, ...] = tuple(a.map_expr(m) for a in args)
        mapped_self: Self = self.with_args(*mapped_args)  # type: ignore
        return m(mapped_self)

    @final
    @property
    def expr_size(self) -> int:
        """
        The size of the expression tree, counting the number of nodes.
        """

        args: tuple[Expr, ...] = self.args  # type: ignore

        return sum(a.expr_size for a in args) + 1

    @final
    @property
    def data_expr_size(self) -> int:
        """
        The size of the expression tree, counting the number of data-dependent
        nodes.
        """

        args: tuple[Expr, ...] = self.args  # type: ignore

        if args == 0:
            return int(self.has_data)

        return sum(a.expr_size for a in args if a.has_data) + 1

    @final
    @property
    def has_data(self) -> bool:
        """
        Does this expression reference the data `x` or `X[i]`?
        """

        args: tuple[Expr, ...] = self.args  # type: ignore

        return any(a.has_data for a in args)

    @final
    @property
    def data_indices(self) -> frozenset[tuple[int, ...]]:
        """
        The set of data indices `X[is]` that this expression uses.
        """

        args: tuple[Expr, ...] = self.args  # type: ignore

        match args:
            case ():
                return frozenset()
            case (a,):
                return a.data_indices
            case _:
                indices: set[tuple[int, ...]] = set()
                for a in args:
                    indices.update(a.data_indices)
                return frozenset(indices)

    @final
    def apply_array_element_offset(
        self,
        axis: int,
        offset: int,
    ) -> Self:
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

        return self.with_args(
            *(a.apply_array_element_offset(axis, offset) for a in self.args)  # type: ignore
        )

    @final
    @property
    def late_bound_constants(self) -> frozenset[Parameter]:
        """
        The set of late-bound constant parameters that this expression uses.
        """

        args: tuple[Expr, ...] = self.args  # type: ignore

        match args:
            case ():
                return frozenset()
            case (a,):
                return a.late_bound_constants
            case _:
                late_bound: set[Parameter] = set()
                for a in args:
                    late_bound.update(a.late_bound_constants)
                return frozenset(late_bound)

    # FIXME: constant_fold based on self.args and self.with_args is blocked on
    #        not being able to relate on TypeVarTuple to another, here *Expr to
    #        *F, see e.g. https://github.com/python/typing/issues/1216
    @abstractmethod
    def constant_fold(self, dtype: np.dtype[F]) -> "F | Expr":
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
            The stencil-extended data, in floating-point format, which must be
            of shape [...PsI, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating-point dtype as the stencil-extended data.

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
            The pointwise data, in floating-point format, which must be
            of shape Ps.
        Xs : np.ndarray[Ps, np.dtype[F]]
            The stencil-extended data, in floating-point format, which must be
            of shape [...Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating-point dtype as the stencil-extended data.

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
            The pointwise data, in floating-point format, which must be
            of shape Ps.
        Xs : np.ndarray[Ps, np.dtype[F]]
            The stencil-extended data, in floating-point format, which must be
            of shape [...Ps, ...stencil_shape].
        late_bound : Mapping[Parameter, np.ndarray[Ns, np.dtype[F]]]
            The late-bound constants parameters for this expression, with the
            same shape and floating-point dtype as the stencil-extended data.

        Returns
        -------
        Xs_lower, Xs_upper : tuple[np.ndarray[Ns, np.dtype[F]], np.ndarray[Ns, np.dtype[F]]]
            The stencil-extended lower and upper bounds on the stencil-extended
            data `Xs`.

            The bounds have not yet been combined across neighbouring points
            that contribute to the same QoI points.
        """

        if (
            data_bounds_checks(self.compute_data_bounds_unchecked)
            != DataBounds.infallible
        ):
            exprv: np.ndarray[Ps, np.dtype[F]] = self.eval(X.shape, Xs, late_bound)
            if not np.all((expr_lower <= exprv) | np.isnan(exprv)):
                warn("expression lower bounds are above the expression values")
            if not np.all((expr_upper >= exprv) | np.isnan(exprv)):
                warn("expression upper bounds are below the expression values")
        else:
            exprv = None  # type: ignore

        Xs_lower: np.ndarray[Ns, np.dtype[F]]
        Xs_upper: np.ndarray[Ns, np.dtype[F]]
        Xs_lower, Xs_upper = self.compute_data_bounds_unchecked(
            expr_lower, expr_upper, X, Xs, late_bound
        )

        warn_on_bounds_exceeded: bool

        match data_bounds_checked := data_bounds_checks(
            self.compute_data_bounds_unchecked
        ):
            case DataBounds.infallible:
                return Xs_lower, Xs_upper
            case DataBounds.unchecked:
                warn_on_bounds_exceeded = False
            case DataBounds.checked:
                warn_on_bounds_exceeded = True
            case _:
                assert_never(data_bounds_checked)

        # ensure that the original data values are within the data bounds
        Xs_lower = _minimum_zero_sign_sensitive(Xs, Xs_lower)
        Xs_upper = _maximum_zero_sign_sensitive(Xs, Xs_upper)

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
            warn_on_bounds_exceeded=warn_on_bounds_exceeded,
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
            warn_on_bounds_exceeded=warn_on_bounds_exceeded,
        )

        return Xs_lower, Xs_upper

    @abstractmethod
    def __repr__(self) -> str:
        pass

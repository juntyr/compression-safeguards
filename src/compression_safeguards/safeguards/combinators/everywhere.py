"""
Everywhere safeguard combinator.
"""

__all__ = ["EverywhereSafeguard"]

from collections.abc import Set
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _ones
from ...utils.bindings import Bindings, Parameter
from ...utils.error import (
    TypeCheckError,
    ctx,
)
from ...utils.intervals import Interval, IntervalUnion
from ...utils.typing import JSON, S, T
from ..pointwise.abc import PointwiseSafeguard
from ..stencil import BoundaryCondition, NeighbourhoodAxis
from ..stencil.abc import StencilSafeguard

if TYPE_CHECKING:
    from ..abc import Safeguard


class EverywhereSafeguard(StencilSafeguard):
    """
    The `EverywhereSafeguard` requires that the contained safeguard's guarantee
    is upheld everywhere.

    In other words, this combinator safeguard is only satisfied for any element
    if its contained safeguard is satisfied for all elements everywhere.

    At the moment, only pointwise and stencil safeguards and combinations
    thereof can be combined by this everywhere-combinator. The combinator is a
    stencil safeguard.

    At the moment, using this safeguard forces a chunked dataset to be
    corrected in one single chunk, though this may be relaxed in the future.

    Parameters
    ----------
    safeguard : dict[str, JSON] | PointwiseSafeguard | StencilSafeguard
        Safeguard configuration [`dict`][dict]s or already initialized
        [`PointwiseSafeguard`][....pointwise.abc.PointwiseSafeguard]
        or [`StencilSafeguard`][....stencil.abc.StencilSafeguard].

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ...
        if instantiating a safeguard raises an exception.
    """

    __slots__: tuple[str, ...] = ("_safeguard",)
    _safeguard: PointwiseSafeguard | StencilSafeguard

    kind: ClassVar[str] = "everywhere"

    def __init__(
        self,
        *,
        safeguard: dict[str, JSON] | PointwiseSafeguard | StencilSafeguard,
    ) -> None:
        from ... import SafeguardKind  # noqa: PLC0415

        with ctx.safeguard(self):
            with ctx.parameter("safeguard"):
                TypeCheckError.check_instance_or_raise(
                    safeguard, dict | PointwiseSafeguard | StencilSafeguard
                )
                safeguard_: Safeguard
                if isinstance(safeguard, dict):
                    safeguard_ = SafeguardKind.from_config(safeguard)
                else:
                    safeguard_ = safeguard
                TypeCheckError.check_instance_or_raise(
                    safeguard_, PointwiseSafeguard | StencilSafeguard
                )

        self._safeguard = safeguard_  # type: ignore

    @property
    def safeguard(self) -> PointwiseSafeguard | StencilSafeguard:
        """
        The safeguard that this everywhere combinator has been configured to
        uphold everywhere.
        """

        return self._safeguard

    @property
    @override
    def late_bound(self) -> Set[Parameter]:
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        return self._safeguard.late_bound

    @override
    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        Boundary conditions of the same kind are combined, but separate kinds
        are tracked separately.

        An empty [`dict`][dict] is returned along dimensions for which the
        stencil safeguard does not need to look at adjacent data points.

        This method also checks that the data shape is compatible with the
        contained safeguard.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the data.

        Returns
        -------
        neighbourhood_shape : tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]
            The shape of the data neighbourhood.

        Raises
        ------
        ...
            if computing the neighbourhood for the contained safeguard raises
            an exception.
        """

        neighbourhood: tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]
        if isinstance(self._safeguard, PointwiseSafeguard):
            neighbourhood = tuple(dict() for _ in data_shape)
        else:
            neighbourhood = self._safeguard.compute_check_neighbourhood_for_data_shape(
                data_shape
            )

        boundary = BoundaryCondition.valid

        # force a valid neighbourhood that spans the entire data
        for i, s in enumerate(data_shape):
            ni = neighbourhood[i]
            if boundary in ni:
                ni[boundary] = NeighbourhoodAxis(
                    max(ni[boundary].before, s), max(ni[boundary].after, s)
                )
            else:
                ni[boundary] = NeighbourhoodAxis(s, s)

        return neighbourhood

    @override
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        approximation: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check if the contained safeguard succeeds the check across all
        elemements and broadcast the result to all elements.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `approximation` is checked.
        approximation : np.ndarray[S, np.dtype[T]]
            Approximation of the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : bool
            `True` if the check succeeded.

        Raises
        ------
        ...
            if checking the contained safeguard raises an exception.
        """

        # early exit if we do not need to check anywhere
        if not np.any(where):
            return _ones(data.shape, np.dtype(np.bool))

        # if the check should succeed anywhere, it needs to succeed everywhere

        # check everywhere, no matter where
        ok = self._safeguard.check_pointwise(
            data, approximation, late_bound=late_bound, where=True
        )
        # everywhere safeguard is only ok in any point if all points are ok
        ok.fill(np.all(ok))

        # mask out results where we do not need to check
        if where is not True:
            ok[~where] = True

        return ok

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals such that the contained safeguard is satisfied
        everywhere.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the safe intervals at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of safe intervals.

        Raises
        ------
        ...
            if computing the safe intervals for the contained safeguard raises
            an exception.
        """

        # early exit if we do not need to safeguard anywhere
        if not np.any(where):
            return Interval.full_like(data).into_union()

        # if we need to safeguard anywhere, we need to safeguard everywhere

        # needs to be safe everywhere
        return self._safeguard.compute_safe_intervals(
            data, late_bound=late_bound, where=True
        )

    @override
    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the footprint of the `foot` array, e.g. for expanding data
        points into the pointwise checks that they contribute to.

        If `where` is [`True`][True] anywhere, the footprint is [`True`][True]
        everywhere.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        # early exit if there is no foot
        if not np.any(where):
            return np.zeros_like(foot)

        # everything is contributed to
        return np.ones_like(foot)

    @override
    def compute_inverse_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the inverse footprint of the `foot` array, e.g. for expanding
        pointwise check fails into the points that could have contributed to
        the failures.

        If `where` is [`True`][True] anywhere, the inverse footprint is
        [`True`][True] everywhere.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the inverse footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The inverse footprint of the `foot` array.
        """

        # early exit if there is no foot
        if not np.any(where):
            return np.zeros_like(foot)

        # everything contributes to
        return np.ones_like(foot)

    @override
    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind, safeguard=self._safeguard.get_config())

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(safeguard={self.safeguard!r})"

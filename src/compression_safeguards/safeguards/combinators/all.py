"""
Logical all (and) combinator safeguard.
"""

__all__ = ["AllSafeguards"]

from abc import ABC
from collections.abc import Collection, Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils.bindings import Bindings, Parameter
from ...utils.error import TypeCheckError, ctx
from ...utils.intervals import IntervalUnion
from ...utils.typing import JSON, S, T
from ..abc import Safeguard
from ..pointwise.abc import PointwiseSafeguard
from ..stencil import BoundaryCondition, NeighbourhoodAxis
from ..stencil.abc import StencilSafeguard


class AllSafeguards(Safeguard):
    """
    The `AllSafeguards` guarantees that, for each element, all of the combined
    safeguards' guarantees are upheld.

    At the moment, only pointwise and stencil safeguards and combinations
    thereof can be combined by this all-combinator. The combinator is a
    pointwise or a stencil safeguard, depending on the safeguards it combines.

    Parameters
    ----------
    safeguards : Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard]
        At least one safeguard configuration [`dict`][dict]s or already
        initialized [`PointwiseSafeguard`][....pointwise.abc.PointwiseSafeguard]
        or [`StencilSafeguard`][....stencil.abc.StencilSafeguard].

    Raises
    ------
    TypeCheckError
        if any parameter has the wrong type.
    ValueError
        if the `safeguards` collection is empty.
    ...
        if instantiating a safeguard raises an exception.
    """

    __slots__: tuple[str, ...] = ()

    kind: ClassVar[str] = "all"

    def __init__(
        self,
        *,
        safeguards: Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard],
    ) -> None:
        pass

    def __new__(  # type: ignore
        cls,
        *,
        safeguards: Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard],
    ) -> "_AllPointwiseSafeguards | _AllStencilSafeguards":
        from ... import SafeguardKind  # noqa: PLC0415

        with ctx.safeguardty(cls):
            with ctx.parameter("safeguards"):
                TypeCheckError.check_instance_or_raise(safeguards, Collection)

                if len(safeguards) <= 0:
                    raise (
                        ValueError("can only combine over at least one safeguard") | ctx
                    )

                safeguards_: list[PointwiseSafeguard | StencilSafeguard] = []
                safeguard: dict[str, JSON] | Safeguard
                for i, safeguard in enumerate(safeguards):
                    with ctx.index(i):
                        TypeCheckError.check_instance_or_raise(
                            safeguard, dict | PointwiseSafeguard | StencilSafeguard
                        )
                        if isinstance(safeguard, dict):
                            safeguard = SafeguardKind.from_config(safeguard)
                        TypeCheckError.check_instance_or_raise(
                            safeguard, PointwiseSafeguard | StencilSafeguard
                        )
                        safeguards_.append(safeguard)  # type: ignore

        pointwise_safeguards_: list[PointwiseSafeguard] = [
            safeguard
            for safeguard in safeguards_
            if isinstance(safeguard, PointwiseSafeguard)
        ]

        if len(pointwise_safeguards_) == len(safeguards_):
            return _AllPointwiseSafeguards(*pointwise_safeguards_)
        else:
            return _AllStencilSafeguards(*safeguards_)

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:  # type: ignore
        """
        The set of safeguards that this any combinator has been configured to
        uphold.
        """

        ...

    @property
    @override
    def late_bound(self) -> Set[Parameter]:  # type: ignore
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        ...

    @override
    def check(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> bool:
        """
        Check if, for all elements, all of the combined safeguards succeed the
        check.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
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
            if checking a safeguard raises an exception.
        """

        ...

    def check_pointwise(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check for which elements all of the combined safeguards succeed the
        check.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.

        Raises
        ------
        ...
            if checking a safeguard raises an exception.
        """

        ...

    def compute_safe_intervals(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intersection of the safe intervals of the combined
        safeguards, i.e. where all of them are safe.

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
            Intersection of safe intervals.

        Raises
        ------
        ...
            if computing the safe intervals for a safeguard raises an exception.
        """

        ...

    def compute_footprint(  # type: ignore
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the footprint of the `foot` array, e.g. for expanding data
        points into the pointwise checks that they contribute to.

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

        ...

    def compute_inverse_footprint(  # type: ignore
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

        ...

    @override
    def get_config(self) -> dict[str, JSON]:  # type: ignore
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict[str, JSON]
            Configuration of the safeguard.
        """

        ...


class _AllSafeguardsBase(ABC):
    __slots__: tuple[str, ...] = ()

    kind: ClassVar[str] = "all"

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:
        return self._safeguards  # type: ignore

    @property
    def late_bound(self) -> Set[Parameter]:
        return frozenset(b for s in self.safeguards for b in s.late_bound)

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        front, *tail = self.safeguards

        # pointwise check succeeds if all safeguards' pointwise check succeeds
        ok = front.check_pointwise(data, prediction, late_bound=late_bound, where=where)

        for safeguard in tail:
            ok &= safeguard.check_pointwise(
                data, prediction, late_bound=late_bound, where=where
            )

        return ok

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        front, *tail = self.safeguards

        # only the intersection of all safeguards' safe intervals is safe
        valid = front.compute_safe_intervals(data, late_bound=late_bound, where=where)

        for safeguard in tail:
            valid = valid.intersect(
                safeguard.compute_safe_intervals(
                    data, late_bound=late_bound, where=where
                )
            )

        return valid

    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        front, *tail = self.safeguards

        # since all safeguards can contribute to the footprint, the union of
        #  their footprints forms the combined footprint
        footprint = front.compute_footprint(foot, late_bound=late_bound, where=where)

        for safeguard in tail:
            footprint |= safeguard.compute_footprint(
                foot, late_bound=late_bound, where=where
            )

        return footprint

    def compute_inverse_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        front, *tail = self.safeguards

        # since all safeguards can contribute to the footprint, the union of
        #  their footprints forms the combined footprint
        footprint = front.compute_inverse_footprint(
            foot, late_bound=late_bound, where=where
        )

        for safeguard in tail:
            footprint |= safeguard.compute_inverse_footprint(
                foot, late_bound=late_bound, where=where
            )

        return footprint

    def get_config(self) -> dict[str, JSON]:
        return dict(
            kind=type(self).kind,
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
        )

    @override
    def __repr__(self) -> str:
        return f"{AllSafeguards.__name__}(safeguards={list(self.safeguards)!r})"


class _AllPointwiseSafeguards(_AllSafeguardsBase, PointwiseSafeguard):
    __slots__: tuple[str, ...] = ("_safeguards",)
    _safeguards: tuple[PointwiseSafeguard, ...]

    def __init__(self, *safeguards: PointwiseSafeguard) -> None:
        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard), (
                f"{safeguard!r} is not a pointwise safeguard"
            )
        self._safeguards = safeguards


class _AllStencilSafeguards(_AllSafeguardsBase, StencilSafeguard):
    __slots__: tuple[str, ...] = ("_safeguards",)
    _safeguards: tuple[PointwiseSafeguard | StencilSafeguard, ...]

    def __init__(self, *safeguards: PointwiseSafeguard | StencilSafeguard) -> None:
        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard | StencilSafeguard), (
                f"{safeguard!r} is not a pointwise or stencil safeguard"
            )
        self._safeguards = safeguards

    @override
    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]:
        neighbourhood: list[dict[BoundaryCondition, NeighbourhoodAxis]] = [
            dict() for _ in data_shape
        ]

        for safeguard in self._safeguards:
            if not isinstance(safeguard, StencilSafeguard):
                continue

            safeguard_neighbourhood = (
                safeguard.compute_check_neighbourhood_for_data_shape(data_shape)
            )

            for i, sn in enumerate(safeguard_neighbourhood):
                ni = neighbourhood[i]

                for b, s in sn.items():
                    if b in ni:
                        neighbourhood[i][b] = NeighbourhoodAxis(
                            max(ni[b].before, s.before), max(ni[b].after, s.after)
                        )
                    else:
                        neighbourhood[i][b] = s

        return tuple(neighbourhood)

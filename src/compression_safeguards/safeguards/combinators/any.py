"""
Logical any (or) combinator safeguard.
"""

__all__ = ["AnySafeguard"]

from abc import ABC
from collections.abc import Collection, Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils.bindings import Bindings, Parameter
from ...utils.error import TypeCheckError, ctx
from ...utils.intervals import Interval, IntervalUnion
from ...utils.typing import JSON, S, T
from ..abc import Safeguard
from ..pointwise.abc import PointwiseSafeguard
from ..stencil import BoundaryCondition, NeighbourhoodAxis
from ..stencil.abc import StencilSafeguard


class AnySafeguard(Safeguard):
    """
    The `AnySafeguard` guarantees that, for each element, at least one of the
    combined safeguards' guarantees is upheld.

    At the moment, only pointwise and stencil safeguards and combinations
    thereof can be combined by this any-combinator. The combinator is a
    pointwise or a stencil safeguard, depending on the safeguards it combines.

    Parameters
    ----------
    safeguards : Collection[dict[str, JSON] | PointwiseSafeguard | StencilSafeguard]
        At least one safeguard configuration [`dict`][dict]s or already
        initialized
        [`PointwiseSafeguard`][compression_safeguards.safeguards.pointwise.abc.PointwiseSafeguard]
        or
        [`StencilSafeguard`][compression_safeguards.safeguards.stencil.abc.StencilSafeguard].

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

    kind: ClassVar[str] = "any"

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
    ) -> "_AnyPointwiseSafeguard | _AnyStencilSafeguard":
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

        if all(isinstance(safeguard, PointwiseSafeguard) for safeguard in safeguards_):
            return _AnyPointwiseSafeguard(*safeguards_)  # type: ignore
        else:
            return _AnyStencilSafeguard(*safeguards_)

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
        Check if, for all elements, any of the combined safeguards succeed the
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
        Check for which elements at least one of the combined safeguards
        succeeds the check.

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
        Compute the union of the safe intervals of the combined safeguards,
        i.e. where at least one is safe.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the safe intervals at data points where the condition
            is [`True`][True].

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of safe intervals.

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
        Compute the footprint of the `foot` array, e.g. for expanding pointwise
        check fails into the points that could have contributed to the failures.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the footprint at data points where the condition is
            [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        ...

    @override
    def get_config(self) -> dict[str, JSON]:  # type: ignore
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        ...


class _AnySafeguardBase(ABC):
    __slots__: tuple[str, ...] = ()

    kind: ClassVar[str] = "any"

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

        ok = front.check_pointwise(data, prediction, late_bound=late_bound, where=where)

        for safeguard in tail:
            ok |= safeguard.check_pointwise(
                data, prediction, late_bound=late_bound, where=where
            )

        return ok

    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        footprint = np.zeros_like(foot)

        # we cannot be more clever here without knowing the data and going
        #  through the entire safe interval computation, which would defeat the
        #  entire point of this cheaper method
        for safeguard in self.safeguards:
            footprint |= safeguard.compute_footprint(
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
        return f"{AnySafeguard.__name__}(safeguards={list(self.safeguards)!r})"


class _AnyPointwiseSafeguard(_AnySafeguardBase, PointwiseSafeguard):
    __slots__: tuple[str, ...] = ("_safeguards",)
    _safeguards: tuple[PointwiseSafeguard, ...]

    def __init__(self, *safeguards: PointwiseSafeguard) -> None:
        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard), (
                f"{safeguard!r} is not a pointwise safeguard"
            )
        self._safeguards = safeguards

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        front, *tail = self.safeguards

        valid = front.compute_safe_intervals(data, late_bound=late_bound, where=where)

        for safeguard in tail:
            valid = valid.union(
                safeguard.compute_safe_intervals(
                    data, late_bound=late_bound, where=where
                )
            )

        return valid


class _AnyStencilSafeguard(_AnySafeguardBase, StencilSafeguard):
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

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        pointwise: list[PointwiseSafeguard] = []
        stencil: list[StencilSafeguard] = []
        for safeguard in self.safeguards:
            assert isinstance(safeguard, PointwiseSafeguard | StencilSafeguard)
            if isinstance(safeguard, PointwiseSafeguard):
                pointwise.append(safeguard)
            else:
                stencil.append(safeguard)

        valids = [
            safeguard.compute_safe_intervals(data, late_bound=late_bound, where=where)
            for safeguard in stencil
        ]

        # we cannot pointwise pick safe intervals from different stencil
        #  safeguards, since the stencil requirements may be violated by
        #  neighbouring picks

        has_pointwise = len(pointwise) > 0
        if has_pointwise:
            front, *tail = pointwise

            valid_pointwise = front.compute_safe_intervals(
                data, late_bound=late_bound, where=where
            )

            for safeguard in tail:
                valid_pointwise = valid_pointwise.union(
                    safeguard.compute_safe_intervals(
                        data, late_bound=late_bound, where=where
                    )
                )

            # we move the pointwise interval first to prioritise it later
            valids.insert(0, valid_pointwise)

        # simple heuristic: pick the safeguard with the largest interval
        selector = np.argmax([v.non_empty_width() for v in valids], axis=0)

        valid: IntervalUnion[T, int, int] = Interval.full_like(data).into_union()

        for i, safeguard in enumerate(stencil):
            where_i = (selector == (i + has_pointwise)) & where
            if np.any(where_i):
                valid = valid.intersect(
                    safeguard.compute_safe_intervals(
                        data, late_bound=late_bound, where=where_i
                    )
                )

        if has_pointwise:
            where_i = (selector == 0) & where
            valid = valid.intersect(
                valid_pointwise.preserve_only_where(where_i.flatten())
            )

        return valid

"""
Logical all (and) combinator safeguard.
"""

__all__ = ["AllSafeguards"]

from abc import ABC
from collections.abc import Collection

import numpy as np

from ...intervals import IntervalUnion
from ..abc import Safeguard
from ..pointwise.abc import PointwiseSafeguard, S, T
from ..stencil import NeighbourhoodAxis
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
    safeguards : Collection[dict | PointwiseSafeguard | StencilSafeguard]
        At least one safeguard configuration [`dict`][dict]s or already
        initialized
        [`PointwiseSafeguard`][numcodecs_safeguards.safeguards.pointwise.abc.PointwiseSafeguard]
        or
        [`StencilSafeguard`][numcodecs_safeguards.safeguards.stencil.abc.StencilSafeguard].
    """

    __slots__ = ()

    kind = "all"

    def __init__(
        self, *, safeguards: Collection[dict | PointwiseSafeguard | StencilSafeguard]
    ):
        pass

    def __new__(  # type: ignore
        cls, *, safeguards: Collection[dict | PointwiseSafeguard | StencilSafeguard]
    ) -> "_AllPointwiseSafeguards | _AllStencilSafeguards":
        from ... import Safeguards

        assert len(safeguards) > 1, "can only combine over at least one safeguard"

        safeguards_ = tuple(
            safeguard
            if isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard))
            else Safeguards[safeguard["kind"]].value(
                **{p: v for p, v in safeguard.items() if p != "kind"}
            )
            for safeguard in safeguards
        )

        for safeguard in safeguards_:
            assert isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard)), (
                f"{safeguard!r} is not a pointwise or stencil safeguard"
            )

        if all(isinstance(safeguard, PointwiseSafeguard) for safeguard in safeguards_):
            return _AllPointwiseSafeguards(*safeguards_)  # type: ignore
        else:
            return _AllStencilSafeguards(*safeguards_)

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:  # type: ignore
        """
        The set of safeguards that this any combinator has been configured to
        uphold.
        """

        ...

    def check_pointwise(  # type: ignore
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check for which elements all of the combined safeguards succeed the
        check.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Pointwise, `True` if the check succeeded for this element.
        """

        ...

    def compute_safe_intervals(  # type: ignore
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intersection of the safe intervals of the combined
        safeguards, i.e. where all of them are safe.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Intersection of safe intervals.
        """

        ...

    def get_config(self) -> dict:  # type: ignore
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        ...


class _AllSafeguardsBase(ABC):
    __slots__ = ()

    kind = "all"

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:
        return self._safeguards  # type: ignore

    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        front, *tail = self.safeguards

        ok = front.check_pointwise(data, decoded)

        for safeguard in tail:
            ok &= safeguard.check_pointwise(data, decoded)

        return ok

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        front, *tail = self.safeguards

        valid = front.compute_safe_intervals(data)

        for safeguard in tail:
            valid = valid.intersect(safeguard.compute_safe_intervals(data))

        return valid

    def get_config(self) -> dict:
        return dict(
            kind=type(self).kind,
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
        )

    def __repr__(self) -> str:
        return f"{AllSafeguards.__name__}(safeguards={list(self.safeguards)!r})"


class _AllPointwiseSafeguards(_AllSafeguardsBase, PointwiseSafeguard):
    __slots__ = "_safeguards"
    _safeguards: tuple[PointwiseSafeguard, ...]

    def __init__(self, *safeguards: PointwiseSafeguard):
        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard), (
                f"{safeguard!r} is not a pointwise safeguard"
            )
        self._safeguards = safeguards


class _AllStencilSafeguards(_AllSafeguardsBase, StencilSafeguard):
    __slots__ = "_safeguards"
    _safeguards: tuple[PointwiseSafeguard | StencilSafeguard, ...]

    def __init__(self, *safeguards: PointwiseSafeguard | StencilSafeguard):
        for safeguard in safeguards:
            assert isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard)), (
                f"{safeguard!r} is not a pointwise or stencil safeguard"
            )
        self._safeguards = safeguards

    def compute_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[None | NeighbourhoodAxis, ...]:
        neighbourhood: list[None | NeighbourhoodAxis] = [None] * len(data_shape)

        for safeguard in self._safeguards:
            if not isinstance(safeguard, StencilSafeguard):
                continue

            safeguard_neighbourhood = safeguard.compute_neighbourhood_for_data_shape(
                data_shape
            )

            for i, sn in enumerate(safeguard_neighbourhood):
                ni = neighbourhood[i]
                if ni is None:
                    neighbourhood[i] = sn
                elif sn is not None:
                    neighbourhood[i] = NeighbourhoodAxis(
                        max(ni.before, sn.before), max(ni.after, sn.after)
                    )

        return tuple(neighbourhood)

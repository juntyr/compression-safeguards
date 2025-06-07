"""
Logical all (and) combinator safeguard.
"""

__all__ = ["SelectSafeguard"]

from abc import ABC
from collections.abc import Collection, Set, Mapping
from typing import Any

import numpy as np

from ...utils.intervals import IntervalUnion
from ...utils.typing import S, T
from ..abc import Safeguard
from ..pointwise.abc import PointwiseSafeguard
from ..stencil import NeighbourhoodAxis
from ..stencil.abc import StencilSafeguard


class SelectSafeguard(Safeguard):
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
        [`PointwiseSafeguard`][compression_safeguards.safeguards.pointwise.abc.PointwiseSafeguard]
        or
        [`StencilSafeguard`][compression_safeguards.safeguards.stencil.abc.StencilSafeguard].
    """

    __slots__ = ()

    kind = "select"

    def __init__(
        self, *, selector: str, safeguards: Collection[dict | PointwiseSafeguard | StencilSafeguard]
    ):
        pass

    def __new__(  # type: ignore
        cls, *, selector: str, safeguards: Collection[dict | PointwiseSafeguard | StencilSafeguard]
    ) -> "_SelectPointwiseSafeguard | _SelectStencilSafeguard":
        from ... import SafeguardKind

        assert len(safeguards) > 0, "can only select over at least one safeguard"

        safeguards_ = tuple(
            safeguard
            if isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard))
            else SafeguardKind[safeguard["kind"]].value(
                **{p: v for p, v in safeguard.items() if p != "kind"}
            )
            for safeguard in safeguards
        )

        for safeguard in safeguards_:
            assert isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard)), (
                f"{safeguard!r} is not a pointwise or stencil safeguard"
            )

        if all(isinstance(safeguard, PointwiseSafeguard) for safeguard in safeguards_):
            return _SelectPointwiseSafeguard(*safeguards_)  # type: ignore
        else:
            return _SelectStencilSafeguard(*safeguards_)
    
    @property
    def selector(self) -> str:  # type: ignore
        ...

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:  # type: ignore
        """
        The set of safeguards that this any combinator has been configured to
        uphold.
        """

        ...

    @property
    def late_bound(self) -> Set[str]:  # type: ignore
        ...

    def check_pointwise(  # type: ignore
        self, data: np.ndarray[S, np.dtype[T]], decoded: np.ndarray[S, np.dtype[T]], *,
        late_bound: Mapping[str, Any],
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
        self, data: np.ndarray[S, np.dtype[T]], *,
        late_bound: Mapping[str, Any],
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


class _SelectSafeguardBase(ABC):
    __slots__ = ()

    kind = "select"

    @property
    def selector(self) -> str:
        return self._selector  # type: ignore

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:
        return self._safeguards  # type: ignore

    @property
    def late_bound(self) -> Set[str]:
        return frozenset(self.selector, **[b for s in self.safeguards for b in s.late_bound])

    def check_pointwise(
        self, data: np.ndarray[S, np.dtype[T]], decoded: np.ndarray[S, np.dtype[T]], *,
        late_bound: Mapping[str, Any],
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        assert self.selector in late_bound, f"late_bound missing binding for {self.selector}"
        selector = late_bound[self.selector]
        selector = np.broadcast_to(selector, data.shape)

        ok = np.zeros_like(data, dtype=np.bool)

        for i, safeguard in enumerate(self.safeguards):
            if not np.any(selector == i):
                continue

        front, *tail = self.safeguards

        ok = front.check_pointwise(data, decoded)

        for safeguard in tail:
            ok &= safeguard.check_pointwise(data, decoded)

        return ok

    def compute_safe_intervals(
        self, data: np.ndarray[S, np.dtype[T]], *,
        late_bound: Mapping[str, Any],
    ) -> IntervalUnion[T, int, int]:
        assert self.selector in late_bound, f"late_bound missing binding for {self.selector}"
        selector = late_bound[self.selector]

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
        return f"{SelectSafeguard.__name__}(safeguards={list(self.safeguards)!r})"


class _SelectPointwiseSafeguard(_SelectSafeguardBase, PointwiseSafeguard):
    __slots__ = ("_selector", "_safeguards",)
    _selector: str
    _safeguards: tuple[PointwiseSafeguard, ...]

    def __init__(self, selector: str, *safeguards: PointwiseSafeguard):
        assert selector.isidentifier(), "selector must be a valid indentifier"
        self._selector = selector

        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard), (
                f"{safeguard!r} is not a pointwise safeguard"
            )
        self._safeguards = safeguards


class _SelectStencilSafeguard(_SelectSafeguardBase, StencilSafeguard):
    __slots__ = ("_selector", "_safeguards",)
    _selector: str
    _safeguards: tuple[PointwiseSafeguard | StencilSafeguard, ...]

    def __init__(self, selector: str, *safeguards: PointwiseSafeguard | StencilSafeguard):
        assert selector.isidentifier(), "selector must be a valid indentifier"
        self._selector = selector

        for safeguard in safeguards:
            assert isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard)), (
                f"{safeguard!r} is not a pointwise or stencil safeguard"
            )
        self._safeguards = safeguards

    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[None | NeighbourhoodAxis, ...]:
        neighbourhood: list[None | NeighbourhoodAxis] = [None] * len(data_shape)

        for safeguard in self._safeguards:
            if not isinstance(safeguard, StencilSafeguard):
                continue

            safeguard_neighbourhood = (
                safeguard.compute_check_neighbourhood_for_data_shape(data_shape)
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

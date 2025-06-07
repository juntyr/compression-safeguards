"""
Logical selector (switch case) combinator safeguard.
"""

__all__ = ["SelectSafeguard"]

from abc import ABC
from collections.abc import Collection, Set

import numpy as np

from ...utils.binding import LateBound, Parameter
from ...utils.intervals import IntervalUnion
from ...utils.typing import S, T
from ..abc import Safeguard
from ..pointwise.abc import PointwiseSafeguard
from ..stencil import NeighbourhoodAxis
from ..stencil.abc import StencilSafeguard


class SelectSafeguard(Safeguard):
    """
    The `SelectSafeguard` guarantees that, for each element, the guarantees of
    the pointwise selected safeguard are upheld. This combinator allows
    selecting between several safeguards with per-element granularity.

    This combinator can be used to describe simple regions of interest where
    different safeguards, e.g. with different error bounds, are applied to
    different parts of the data.

    At the moment, only pointwise and stencil safeguards and combinations
    thereof can be combined by this select-combinator. The combinator is a
    pointwise or a stencil safeguard, depending on the safeguards it combines.

    Parameters
    ----------
    selector : str
        Identifier for the late-bound parameter that is used to select between
        the `safeguards`.
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
        self,
        *,
        selector: Parameter,
        safeguards: Collection[dict | PointwiseSafeguard | StencilSafeguard],
    ):
        pass

    def __new__(  # type: ignore
        cls,
        *,
        selector: Parameter,
        safeguards: Collection[dict | PointwiseSafeguard | StencilSafeguard],
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
            return _SelectPointwiseSafeguard(selector, *safeguards_)  # type: ignore
        else:
            return _SelectStencilSafeguard(selector, *safeguards_)

    @property
    def selector(self) -> str:  # type: ignore
        """
        The identifier for the late-bound selector parameter.
        """
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
        """
        The set of the identifiers of the late-bound parameters that this
        safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """
        ...

    def check_pointwise(  # type: ignore
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: LateBound,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check for which elements the selected safeguard succeed the check.

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
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: LateBound,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the safe intervals for the selected safeguard.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            The safe intervals.
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
    def selector(self) -> Parameter:
        return self._selector  # type: ignore

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:
        return self._safeguards  # type: ignore

    @property
    def late_bound(self) -> Set[Parameter]:
        return frozenset(
            [self.selector] + [b for s in self.safeguards for b in s.late_bound]
        )

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: LateBound,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        selector = late_bound.resolve_ndarray(
            self.selector, data.shape, np.dtype(np.int_)
        )

        oks = [
            safeguard.check_pointwise(data, decoded, late_bound=late_bound)
            for safeguard in self.safeguards
        ]

        return np.choose(selector, oks)  # type: ignore

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: LateBound,
    ) -> IntervalUnion[T, int, int]:
        selector = late_bound.resolve_ndarray(
            self.selector, data.shape, np.dtype(np.int_)
        )

        valids = [
            safeguard.compute_safe_intervals(data, late_bound=late_bound)
            for safeguard in self.safeguards
        ]

        umax = max(valid.u for valid in valids)

        for i, v in enumerate(valids):
            vnew = IntervalUnion.empty(data.dtype, data.size, umax)
            vnew._lower[: v.u] = v._lower
            vnew._upper[: v.u] = v._upper
            valids[i] = vnew

        valid: IntervalUnion[T, int, int] = IntervalUnion.empty(
            data.dtype, data.size, umax
        )
        valid._lower = (
            np.take_along_axis(
                np.stack([v._lower.T for v in valids], axis=0),
                selector.flatten().reshape((1, data.size, umax)),
                axis=0,
            )
            .reshape(data.size, umax)
            .T
        )
        valid._upper = (
            np.take_along_axis(
                np.stack([v._upper.T for v in valids], axis=0),
                selector.flatten().reshape((1, data.size, umax)),
                axis=0,
            )
            .reshape(data.size, umax)
            .T
        )

        return valid

    def get_config(self) -> dict:
        return dict(
            kind=type(self).kind,
            selector=self.selector,
            safeguards=[safeguard.get_config() for safeguard in self.safeguards],
        )

    def __repr__(self) -> str:
        return f"{SelectSafeguard.__name__}(selector={self.selector!r}, safeguards={list(self.safeguards)!r})"


class _SelectPointwiseSafeguard(_SelectSafeguardBase, PointwiseSafeguard):
    __slots__ = (
        "_selector",
        "_safeguards",
    )
    _selector: Parameter
    _safeguards: tuple[PointwiseSafeguard, ...]

    def __init__(self, selector: Parameter, *safeguards: PointwiseSafeguard):
        self._selector = selector

        for safeguard in safeguards:
            assert isinstance(safeguard, PointwiseSafeguard), (
                f"{safeguard!r} is not a pointwise safeguard"
            )
        self._safeguards = safeguards


class _SelectStencilSafeguard(_SelectSafeguardBase, StencilSafeguard):
    __slots__ = (
        "_selector",
        "_safeguards",
    )
    _selector: Parameter
    _safeguards: tuple[PointwiseSafeguard | StencilSafeguard, ...]

    def __init__(
        self, selector: Parameter, *safeguards: PointwiseSafeguard | StencilSafeguard
    ):
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

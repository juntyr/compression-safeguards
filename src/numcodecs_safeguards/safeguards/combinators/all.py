"""
Logical all (and) combinator safeguard.
"""

__all__ = ["AllSafeguards"]

from collections.abc import Collection

import numpy as np

from ...intervals import IntervalUnion
from ..pointwise.abc import PointwiseSafeguard, S, T
from ..stencil.abc import StencilSafeguard


class AllSafeguards:
    """
    The `AllSafeguards` guarantees that, for each element, all of the combined
    safeguards' guarantees are upheld.

    At the moment, only pointwise and stencil safeguards and combinations
    thereof can be combined by this all-combinator.

    Parameters
    ----------
    safeguards : Collection[dict | PointwiseSafeguard | StencilSafeguard]
        At least one safeguard configuration [`dict`][dict]s or already
        initialized
        [`PointwiseSafeguard`][numcodecs_safeguards.safeguards.pointwise.abc.PointwiseSafeguard]
        or
        [`StencilSafeguard`][numcodecs_safeguards.safeguards.stencil.abc.StencilSafeguard].
    """

    # __slots__ = ("_safeguards",)
    _safeguards: tuple[PointwiseSafeguard | StencilSafeguard, ...]

    kind = "all"

    def __init__(
        self, *, safeguards: Collection[dict | PointwiseSafeguard | StencilSafeguard]
    ):
        from ... import Safeguards

        assert len(safeguards) > 1, "can only combine over at least one safeguard"

        self._safeguards = tuple(
            safeguard
            if isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard))
            else Safeguards[safeguard["kind"]].value(
                **{p: v for p, v in safeguard.items() if p != "kind"}
            )
            for safeguard in safeguards
        )

        for safeguard in self._safeguards:
            assert isinstance(safeguard, (PointwiseSafeguard, StencilSafeguard)), (
                f"{safeguard!r} is not a pointwise or stencil safeguard"
            )

        if all(
            isinstance(safeguard, PointwiseSafeguard) for safeguard in self._safeguards
        ):
            self.__class__ = _AllPointwiseSafeguards
        else:
            self.__class__ = _AllStencilSafeguards

    @property
    def safeguards(self) -> tuple[PointwiseSafeguard | StencilSafeguard, ...]:
        """
        The set of safeguards that this any combinator has been configured to
        uphold.
        """

        return self._safeguards

    def check_pointwise(
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

        front, *tail = self._safeguards

        ok = front.check_pointwise(data, decoded)

        for safeguard in tail:
            ok &= safeguard.check_pointwise(data, decoded)

        return ok

    def compute_safe_intervals(
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

        front, *tail = self._safeguards

        valid = front.compute_safe_intervals(data)

        for safeguard in tail:
            valid = valid.intersect(safeguard.compute_safe_intervals(data))

        return valid

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind,
            safeguards=[safeguard.get_config() for safeguard in self._safeguards],
        )

    def __repr__(self) -> str:
        return f"{AllSafeguards.__name__}(safeguards={list(self._safeguards)!r})"


class _AllPointwiseSafeguards(AllSafeguards, PointwiseSafeguard):
    # __slots__ = ()
    pass


class _AllStencilSafeguards(AllSafeguards, StencilSafeguard):
    # __slots__ = ()
    pass

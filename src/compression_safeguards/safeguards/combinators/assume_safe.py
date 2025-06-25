"""
Always safe (logical truth) combinator safeguard.
"""

__all__ = ["AssumeAlwaysSafeguard"]

import numpy as np

from ...utils.bindings import Bindings
from ...utils.intervals import Interval, IntervalUnion
from ...utils.typing import S, T
from ..pointwise.abc import PointwiseSafeguard


class AssumeAlwaysSafeguard(PointwiseSafeguard):
    """
    The `AssumeAlwaysSafeguard` assumes that all elements always meet their guarantees
    and are thus always safe.

    This safeguards can be used with the
    [`SelectSafeguard`][compression_safeguards.safeguards.combinators.select.SelectSafeguard]
    to express regions that are *not* of interest, i.e. where no additional
    safety requirements are imposed.
    """

    __slots__ = ()

    kind = "assume_safe"

    def __init__(self):
        pass

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        All elements are safe and thus always succeed the check.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        ok : np.ndarray
            Pointwise, `True` for every element.
        """

        return np.ones_like(data, dtype=np.bool)  # type: ignore

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Since all values are always safe, the safe intervals for each element
        in the `data` array are full (contain all possible values).

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals that cover all possible values, since all are
            always safe.
        """

        return Interval.full_like(data).into_union()

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind)

"""
Abstract base class for the global safeguards.
"""

__all__ = ["GlobalSafeguard"]

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from ..abc import Safeguard
from ...intervals import IntervalUnion

T = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])
N = TypeVar("N", bound=int)


class GlobalSafeguard(Safeguard, ABC):
    """
    Global safeguard abstract base class.

    Global safeguards describe properties that are satisfied (or not) across
    the entire data.
    """

    @abstractmethod
    def compute_safe_intervals_with_priors(
        self,
        data: np.ndarray[S, T],
        priors: IntervalUnion[T, N, int],
    ) -> IntervalUnion[T, N, int]:
        """
        Compute the intervals in which the safeguard's guarantees with respect
        to the `data` are upheld.

        The returned union of intervals must not have any overlap between the
        intervals inside the union. The `data` must be contained in the union.

        Parameters
        ----------
        data : np.ndarray[S, T]
            Data for which the safe intervals should be computed.
        priors : IntervalUnion[T, N, int]
            Prior safe intervals, which can be used by this safeguard to
            compute less pessimistic safe intervals. If all-valid intervals
            are passed in, this method must return always-safe but possibly-
            pessimistic intervals.

            To ensure that the prior intervals are honoured, they should be
            intersected into the resulting intervals before returning them.

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the safeguard's guarantees are upheld.
        """

        pass

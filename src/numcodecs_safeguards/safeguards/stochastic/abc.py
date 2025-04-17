"""
Abstract base class for the stochastic safeguards.
"""

__all__ = ["StochasticSafeguard"]

from abc import ABC, abstractmethod
from typing import Any, final, TypeVar

import numpy as np

from ..abc import Safeguard
from ...intervals import IntervalUnion

T = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])


class StochasticSafeguard(Safeguard, ABC):
    """
    Stochastic safeguard abstract base class.

    Stochastic safeguards use randomness to change the stochastic properties of
    the decoded data in relation to the original data.

    Unlike other safeguards, they cannot be checked and are always applied.
    """

    @final
    def check(self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]) -> bool:
        """
        Stochastic safeguards cannot be checked and always return
        [`False`][False].

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : bool
            Always [`False`][False].
        """

        return False

    @abstractmethod
    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, Any, Any]:
        """
        Compute the intervals in which the safeguard's guarantees with respect
        to the `data` are upheld.

        The returned union of intervals must not have any overlap between the
        intervals inside the union. The `data` must be contained in the union.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the safeguard's guarantees are upheld.
        """

        pass

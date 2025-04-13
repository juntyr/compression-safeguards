"""
Abstract base class for the elementwise safeguards.
"""

__all__ = ["ElementwiseSafeguard"]

from abc import ABC, abstractmethod
from typing import Any, final, TypeVar

import numpy as np

from ..abc import Safeguard
from ...intervals import IntervalUnion

T = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])


class ElementwiseSafeguard(Safeguard, ABC):
    """
    Elementwise safeguard abstract base class.

    Elementwise safeguards describe properties that are satisfied (or not) per
    element, i.e. independent of other elements.
    """

    @final
    def check(self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]) -> bool:
        """
        Check if the `decoded` array upholds the property enforced by this
        safeguard.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """

        return bool(np.all(self.check_elementwise(data, decoded)))

    @abstractmethod
    def check_elementwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array uphold the property
        enforced by this safeguard.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : np.ndarray
            Per-element, `True` if the check succeeded for this element.
        """

        pass

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

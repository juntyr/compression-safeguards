"""
Abstract base class for the stencil safeguards.
"""

__all__ = ["StencilSafeguard"]

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np

from ...intervals import IntervalUnion
from ..abc import Safeguard

T = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])


class StencilSafeguard(Safeguard, ABC):
    """
    Stencil safeguard abstract base class.

    Stencil safeguards describe properties that are computed in a
    neighbourhood around each element, i.e. whether or not an element satisfies
    the safeguard is coupled to its neighbouring elements.
    """

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

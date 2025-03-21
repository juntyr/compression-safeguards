__all__ = ["ElementwiseSafeguard"]

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import numpy as np

from ..abc import Safeguard
from ...intervals import IntervalUnion

T = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])


class ElementwiseSafeguard(Safeguard, ABC):
    """
    Elementwise safeguard abstract base class.

    Elementwise safeguards can identitfy individual elements that violate the
    property enforced by the safeguard.
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

"""
Abstract base class for the stencil safeguards.
"""

__all__ = ["StencilSafeguard"]

from abc import ABC, abstractmethod
from typing import final

import numpy as np

from ...utils.binding import LateBound
from ...utils.intervals import IntervalUnion
from ...utils.typing import S, T
from ..abc import Safeguard
from . import NeighbourhoodAxis


class StencilSafeguard(Safeguard, ABC):
    """
    Stencil safeguard abstract base class.

    Stencil safeguards describe properties that are computed in a
    neighbourhood around each element, i.e. whether or not an element satisfies
    the safeguard is coupled to its neighbouring elements.
    """

    __slots__ = ()

    @abstractmethod
    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[None | NeighbourhoodAxis, ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        [`None`][None] is returned along dimensions for which the stencil
        safeguard does not need to look at adjacent data points.

        This method also checks that the data shape is compatible with this
        stencil safeguard.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the data.

        Returns
        -------
        neighbourhood_shape : tuple[None | NeighbourhoodAxis, ...]
            The shape of the data neighbourhood.
        """

        pass

    @final
    def check(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: LateBound,
    ) -> bool:
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

        return bool(np.all(self.check_pointwise(data, decoded, late_bound=late_bound)))

    @abstractmethod
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: LateBound,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array uphold the neighbourhood
        property enforced by this safeguard.

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

        pass

    @abstractmethod
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: LateBound,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the safeguard's guarantees with respect
        to the `data` are upheld.

        The returned union of intervals must not have any overlap between the
        intervals inside the union. The `data` must be contained in the union.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the safeguard's guarantees are upheld.
        """

        pass

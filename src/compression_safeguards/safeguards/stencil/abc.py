"""
Abstract base class for the stencil safeguards.
"""

__all__ = ["StencilSafeguard"]

from abc import ABC, abstractmethod
from typing import Literal, final

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils.bindings import Bindings
from ...utils.intervals import IntervalUnion
from ...utils.typing import S, T
from ..abc import Safeguard
from . import BoundaryCondition, NeighbourhoodAxis


class StencilSafeguard(Safeguard, ABC):
    """
    Stencil safeguard abstract base class.

    Stencil safeguards describe properties that are computed in a
    neighbourhood around each element, i.e. whether or not an element satisfies
    the safeguard is coupled to its neighbouring elements.
    """

    __slots__: tuple[str, ...] = ()

    @abstractmethod
    def compute_check_neighbourhood_for_data_shape(
        self, data_shape: tuple[int, ...]
    ) -> tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]:
        """
        Compute the shape of the data neighbourhood for data of a given shape.
        Boundary conditions of the same kind are combined, but separate kinds
        are tracked separately.

        An empty [`dict`][dict] is returned along dimensions for which the
        stencil safeguard does not need to look at adjacent data points.

        This method also checks that the data shape is compatible with this
        stencil safeguard.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            The shape of the data.

        Returns
        -------
        neighbourhood_shape : tuple[dict[BoundaryCondition, NeighbourhoodAxis], ...]
            The shape of the data neighbourhood.
        """

        pass

    @final
    @override
    def check(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> bool:
        """
        Check if the `prediction` array upholds the property enforced by this
        safeguard.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """

        return bool(
            np.all(
                self.check_pointwise(
                    data, prediction, late_bound=late_bound, where=where
                )
            )
        )

    @abstractmethod
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `prediction` array uphold the neighbourhood
        property enforced by this safeguard.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : np.ndarray[S, np.dtype[np.bool]]
            Pointwise, `True` if the check succeeded for this element.
        """

        pass

    @abstractmethod
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
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
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the safe intervals at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the safeguard's guarantees are upheld.
        """

        pass

    @abstractmethod
    def compute_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the footprint of the `foot` array, e.g. for expanding data
        points into the pointwise checks that they contribute to.

        For stencil safeguards, the footprint usually extends beyond
        `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the footprint at pointwise checks where the condition
            is [`True`][True].

            Conceptually, `where` is applied to `footprint` at the end.

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        pass

    @abstractmethod
    def compute_inverse_footprint(
        self,
        foot: np.ndarray[S, np.dtype[np.bool]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Compute the inverse footprint of the `foot` array, e.g. for expanding
        pointwise check fails into the points that could have contributed to
        the failures.

        For stencil safeguards, the inverse footprint usually extends beyond
        `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the inverse footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

            Conceptually, `where` is applied to `foot` at the start.

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The inverse footprint of the `foot` array.
        """

        pass

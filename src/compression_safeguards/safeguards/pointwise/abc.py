"""
Abstract base class for the pointwise safeguards.
"""

__all__ = ["PointwiseSafeguard"]

from abc import ABC, abstractmethod
from typing import final

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils.bindings import Bindings
from ...utils.intervals import IntervalUnion
from ...utils.typing import S, T
from ..abc import Safeguard


class PointwiseSafeguard(Safeguard, ABC):
    """
    Pointwise safeguard abstract base class.

    Pointwise safeguards describe properties that are satisfied (or not) per
    element, i.e. independent of other elements.
    """

    __slots__: tuple[str, ...] = ()

    @final
    @override
    def check(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
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

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """

        return bool(
            np.all(self.check_pointwise(data, prediction, late_bound=late_bound))
        )

    @abstractmethod
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `prediction` array uphold the property
        enforced by this safeguard.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.

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

        Returns
        -------
        intervals : IntervalUnion[T, int, int]
            Union of intervals in which the safeguard's guarantees are upheld.
        """

        pass

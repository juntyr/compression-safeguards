"""
Lossless reconstruction safeguard.
"""

__all__ = ["LosslessSafeguard"]

from collections.abc import Set
from typing import ClassVar, Literal

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ...utils._compat import _ensure_array, _logical_and
from ...utils.bindings import Bindings, Parameter
from ...utils.cast import as_bits
from ...utils.intervals import Interval, IntervalUnion
from ...utils.typing import JSON, S, T
from ..pointwise.abc import PointwiseSafeguard


class LosslessSafeguard(PointwiseSafeguard):
    """
    The `LosslessSafeguard` guarantees that all elements are reconstructed
    exactly, i.e. losslessly.

    This safeguard is an alternative to using a zero error bound safeguard.

    This safeguards can be used with the
    [`SelectSafeguard`][....combinators.select.SelectSafeguard] to express
    regions that are of extra special interest and must be kept exactly the
    same.
    """

    __slots__: tuple[str, ...] = ()

    kind: ClassVar[str] = "lossless"

    def __init__(self) -> None:
        pass

    @property
    @override
    def late_bound(self) -> Set[Parameter]:
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        return frozenset()

    @override
    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check for which elements the `data` and `prediction` are bitwise
        equivalent.

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

        ok: np.ndarray[S, np.dtype[np.bool]] = as_bits(data) == as_bits(prediction)
        ok = _ensure_array(ok)
        if where is not True:
            ok[~where] = True
        return ok

    @override
    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> IntervalUnion[T, int, int]:
        """
        The safe intervals for each element in the `data` array only contain
        this element.

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
            Singleton interval union that only contains the `data`.
        """

        wheref: Literal[True] | np.ndarray[tuple[int], np.dtype[np.bool]] = (
            True if where is True else where.flatten()
        )

        return Interval.singleton(data).preserve_only_where(wheref).into_union()

    @override
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

        The footprint is equivalent to `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the footprint at pointwise checks where the condition
            is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The footprint of the `foot` array.
        """

        return _logical_and(foot, where)

    @override
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

        The inverse footprint is equivalent to `foot & where`.

        Parameters
        ----------
        foot : np.ndarray[S, np.dtype[np.bool]]
            Array for which the inverse footprint is computed.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only compute the inverse footprint at pointwise checks where the
            condition is [`True`][True].

        Returns
        -------
        print : np.ndarray[S, np.dtype[np.bool]]
            The inverse footprint of the `foot` array.
        """

        return _logical_and(foot, where)

    @override
    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind)

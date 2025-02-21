"""
Finite difference absolute error bound safeguard.
"""

__all__ = ["FiniteDifferenceAbsoluteErrorBoundSafeguard"]

import numpy as np
from typing import Optional

from .. import ElementwiseSafeguard, _as_bits
from . import (
    FiniteDifference,
    _finite_difference_offsets,
    _finite_difference_coefficients,
)


class FiniteDifferenceAbsoluteErrorBoundSafeguard(ElementwiseSafeguard):
    """
    The `FiniteDifferenceAbsoluteErrorBoundSafeguard` guarantees that the
    elementwise absolute error of the finite-difference-approximated derivative
    is less than or equal to the provided bound `eb_abs`.

    The safeguard supports three types of
    [`FiniteDifference`][numcodecs_safeguards.safeguards.elementwise.findiff.FiniteDifference]:
    `central`, `forward`, `backward`.

    The fininite difference is computed with respect to the provided uniform
    grid spacing `dx`. If the spacing is different along different axes,
    multiple safeguards along specific axes with different spacing can be
    combined.

    If the finite difference for an element evaluates to an infinite value,
    this safeguard guarantees that the finite difference on the decoded value
    produces the exact same infinite value. For a NaN finite difference, this
    safeguard guarantees that the finite difference on the decoded value is
    also NaN, but does not guarantee that it has the same bitpattern.

    Parameters
    ----------
    type : str | FiniteDifference
        The type of finite difference.
    order : int
        The non-negative order of the derivative that is approximayed by a
        finite difference.
    accuracy : int
        The positive order of accuracy of the finite difference approximation.

        The order of accuracy must be even for a central finite difference.
    dx : float
        The uniform positive grid spacing between each point.
    eb_abs : float
        The positive absolute error bound that is enforced by this safeguard.
    axis : Optional[int]
        The axis along which the finite difference is safeguarded. The default,
        [`None`][None], is to safeguard along all axes.
    """

    __slots__ = (
        "_type",
        "_order",
        "_accuracy",
        "_dx",
        "_eb_abs",
        "_axis",
        "_eb_abs_impl",
    )
    _type: FiniteDifference
    _order: int
    _accuracy: int
    _dx: float
    _eb_abs: float
    _axis: Optional[int]
    _eb_abs_impl: float

    kind = "findiff_abs"
    _priority = 0

    def __init__(
        self,
        type: str | FiniteDifference,
        order: int,
        accuracy: int,
        dx: float,
        eb_abs: float,
        axis: Optional[int] = None,
    ):
        type = type if isinstance(type, FiniteDifference) else FiniteDifference[type]

        assert order >= 0, "order must be non-negative"
        assert accuracy > 0, "accuracy must be positive"

        if type == FiniteDifference.central:
            assert accuracy % 2 == 0, (
                "accuracy must be even for a central finite difference"
            )

        assert dx > 0.0, "dx must be positive"
        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._type = type
        self._order = order
        self._accuracy = accuracy
        self._dx = dx
        self._eb_abs = eb_abs
        self._axis = axis

        offsets = _finite_difference_offsets(type, order, accuracy)
        coefficients = _finite_difference_coefficients(order, offsets)

        self._eb_abs_impl = (
            eb_abs * np.power(dx, order) / sum(abs(c) for c in coefficients)
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check for which elements in the `decoded` array the finite differences
        satisfy the absolute error bound.

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

        return (
            (np.abs(data - decoded) <= self._eb_abs_impl)
            | (_as_bits(data) == _as_bits(decoded))
            | (np.isnan(data) == np.isnan(decoded))
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        error = decoded - data
        correction = (
            np.round(error / (self._eb_abs_impl * 2.0)) * (self._eb_abs_impl * 2.0)
        ).astype(data.dtype)
        corrected = decoded - correction

        return np.where(
            self.check_elementwise(data, corrected),
            corrected,
            data,
        )

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind,
            type=self._type.name,
            order=self._order,
            accuracy=self._accuracy,
            dx=self._dx,
            eb_abs=self._eb_abs,
            axis=self._axis,
        )

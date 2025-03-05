"""
Finite difference absolute error bound safeguard.
"""

__all__ = ["FiniteDifferenceAbsoluteErrorBoundSafeguard"]

import warnings
from fractions import Fraction
from typing import Optional

import numpy as np

from ....intervals import (
    IntervalUnion,
    Interval,
    Lower,
    Upper,
    Minimum,
    Maximum,
    _to_total_order,
    _from_total_order,
)
from .. import ElementwiseSafeguard, _as_bits
from . import (
    FiniteDifference,
    _finite_difference_offsets,
    _finite_difference_coefficients,
    _finite_difference,
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
    _offsets: tuple[int, ...]
    _coefficients: tuple[Fraction, ...]
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

        if order > 8 or accuracy > 8:
            warnings.warn(
                f"computing a finite difference of order {order} and accuracy {accuracy} will be costly",
                stacklevel=2,
            )

        self._type = type
        self._order = order
        self._accuracy = accuracy
        self._dx = dx
        self._eb_abs = eb_abs
        self._axis = axis

        self._offsets = _finite_difference_offsets(type, order, accuracy)
        self._coefficients = _finite_difference_coefficients(order, self._offsets)

        self._eb_abs_impl = (
            eb_abs * np.power(dx, order) / sum(abs(c) for c in self._coefficients)
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        axes = list(range(len(data.shape))) if self._axis is None else [self._axis]

        axes = [
            a
            for a in axes
            if ((a >= 0) and (a < len(data.shape)))
            or ((a < 0) and (a >= -len(data.shape)))
        ]
        axes = [a for a in axes if data.shape[a] >= len(self._coefficients)]

        for a in axes:
            findiff_data = _finite_difference(
                data, self._order, self._offsets, self._coefficients, self._dx, axis=a
            )
            findiff_decoded = _finite_difference(
                decoded,
                self._order,
                self._offsets,
                self._coefficients,
                self._dx,
                axis=a,
            )

            # abs(findiff_data - findiff_decoded) <= self._eb_abs, but works for
            # unsigned ints
            absolute_bound = (
                np.where(
                    findiff_data > findiff_decoded,
                    findiff_data - findiff_decoded,
                    findiff_decoded - findiff_data,
                )
                <= self._eb_abs
            )
            same_bits = _as_bits(findiff_data) == _as_bits(findiff_decoded)
            both_nan = np.isnan(findiff_data) & np.isnan(findiff_decoded)

            ok = np.where(
                np.isfinite(findiff_data),
                absolute_bound,
                np.where(
                    np.isinf(findiff_data),
                    same_bits,
                    both_nan,
                ),
            )

            if not np.all(ok):
                return False

        return True

    def compute_safe_intervals(self, data: np.ndarray) -> IntervalUnion:
        """
        Compute the intervals in which the absolute error bound is upheld with
        respect to the finite differences of the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the absolute error bound is upheld.
        """

        data = data.flatten()
        valid = Interval.empty_like(data)

        if np.issubdtype(data.dtype, np.floating):
            Lower(data) <= valid[np.isinf(data)] <= Upper(data)

        if np.issubdtype(data.dtype, np.floating):
            nan_min = np.array(
                np.array(np.inf, dtype=data.dtype).view(
                    data.dtype.str.replace("f", "u")
                )
                + 1
            ).view(data.dtype)
            nan_max = np.array(-1, dtype=data.dtype.str.replace("f", "i")).view(
                data.dtype
            )

            # any NaN with the same sign is valid
            Lower(
                np.where(
                    np.signbit(data),
                    np.copysign(nan_max, -1),
                    np.copysign(nan_min, +1),
                )
            ) <= valid[np.isnan(data)] <= Upper(
                np.where(
                    np.signbit(data),
                    np.copysign(nan_min, -1),
                    np.copysign(nan_max, +1),
                )
            )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            Lower(data - self._eb_abs_impl) <= valid[np.isfinite(data)] <= Upper(
                data + self._eb_abs_impl
            )

        if np.issubdtype(data.dtype, np.integer):
            # saturate the error bounds so that they don't wrap around
            Minimum <= valid[valid._lower > data]
            valid[valid._upper < data] <= Maximum

        # correct rounding errors in the lower and upper bound
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # we don't use abs(data - bound) here to accommodate unsigned ints
            lower_bound_outside_eb_abs = (data - valid._lower) > self._eb_abs_impl
            upper_bound_outside_eb_abs = (valid._upper - data) > self._eb_abs_impl

        valid._lower[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid._lower) + lower_bound_outside_eb_abs,
            data.dtype,
        )[np.isfinite(data)]
        valid._upper[np.isfinite(data)] = _from_total_order(
            _to_total_order(valid._upper) - upper_bound_outside_eb_abs,
            data.dtype,
        )[np.isfinite(data)]

        return valid.into_union()

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

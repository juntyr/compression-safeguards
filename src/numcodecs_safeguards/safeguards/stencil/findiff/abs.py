"""
Finite difference absolute error bound safeguard.
"""

__all__ = ["FiniteDifferenceAbsoluteErrorBoundSafeguard"]

import warnings
from fractions import Fraction

import numpy as np

from ....cast import to_float, as_bits, to_finite_float, _isnan, _isinf, _isfinite
from ....intervals import IntervalUnion
from ..abc import StencilSafeguard, S, T
from ...pointwise.abs import _compute_safe_eb_diff_interval
from . import (
    FiniteDifference,
    _finite_difference_offsets,
    _finite_difference_coefficients,
    _finite_difference,
)


class FiniteDifferenceAbsoluteErrorBoundSafeguard(StencilSafeguard):
    """
    The `FiniteDifferenceAbsoluteErrorBoundSafeguard` guarantees that the
    pointwise absolute error of the finite-difference-approximated derivative
    is less than or equal to the provided bound `eb_abs`.

    The safeguard supports three types of
    [`FiniteDifference`][numcodecs_safeguards.safeguards.stencil.findiff.FiniteDifference]:
    `central`, `forward`, `backward`.

    The fininite difference is computed with respect to the provided uniform
    grid spacing `dx`. If the spacing is different along different axes,
    multiple safeguards along specific axes with different spacing can be
    combined.

    If the finite difference for an element evaluates to an infinite value,
    this safeguard guarantees that the finite difference on the decoded value
    produces the exact same infinite value. For a NaN finite difference, this
    safeguard guarantees that the finite difference on the decoded value is
    also NaN, but does not guarantee that it has the same bit pattern.

    Parameters
    ----------
    type : str | FiniteDifference
        The type of finite difference.
    order : int
        The non-negative order of the derivative that is approximated by a
        finite difference.
    accuracy : int
        The positive order of accuracy of the finite difference approximation.

        The order of accuracy must be even for a central finite difference.
    dx : int | float
        The uniform positive grid spacing between each point.
    eb_abs : int | float
        The non-negative absolute error bound on the finite difference that is
        enforced by this safeguard.
    axis : None | int
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
    _dx: int | float
    _eb_abs: int | float
    _axis: None | int
    _offsets: tuple[int, ...]
    _coefficients: tuple[Fraction, ...]
    _eb_abs_impl: Fraction

    kind = "findiff_abs"

    def __init__(
        self,
        type: str | FiniteDifference,
        order: int,
        accuracy: int,
        dx: int | float,
        eb_abs: int | float,
        axis: None | int = None,
    ):
        type = type if isinstance(type, FiniteDifference) else FiniteDifference[type]

        assert order >= 0, "order must be non-negative"
        assert accuracy > 0, "accuracy must be positive"

        if type == FiniteDifference.central:
            assert accuracy % 2 == 0, (
                "accuracy must be even for a central finite difference"
            )

        assert dx > 0, "dx must be positive"
        assert isinstance(dx, int) or _isfinite(dx), "dx must be finite"
        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert isinstance(eb_abs, int) or _isfinite(eb_abs), "eb_abs must be finite"

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
            Fraction(eb_abs)
            * (Fraction(dx) ** order)
            / sum(abs(c) for c in self._coefficients)
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check(self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]) -> bool:
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
                to_float(data),
                self._order,
                self._offsets,
                self._coefficients,
                self._dx,
                axis=a,
            )
            findiff_decoded = _finite_difference(
                to_float(decoded),
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
            same_bits = as_bits(findiff_data, kind="V") == as_bits(
                findiff_decoded, kind="V"
            )
            both_nan = _isnan(findiff_data) & _isnan(findiff_decoded)

            ok = np.where(
                _isfinite(findiff_data),
                absolute_bound,
                np.where(
                    _isinf(findiff_data),
                    same_bits,
                    both_nan,
                ),
            )

            if not np.all(ok):
                return False

        return True

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
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

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            try:
                eb_abs_impl_float = float(self._eb_abs_impl)
            except OverflowError:
                eb_abs_impl_float = float("inf")
            eb_abs_impl: np.ndarray = to_finite_float(
                eb_abs_impl_float, data_float.dtype
            )
        assert eb_abs_impl >= 0

        return _compute_safe_eb_diff_interval(
            data, data_float, -eb_abs_impl, eb_abs_impl, equal_nan=True
        ).into_union()  # type: ignore

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

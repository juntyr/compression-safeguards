"""
Finite difference absolute error bound safeguard.
"""

__all__ = ["FiniteDifferenceAbsoluteErrorBoundSafeguard"]

import warnings
from fractions import Fraction

import numpy as np

from ....cast import _isfinite, _isinf, _isnan, as_bits, to_finite_float, to_float
from ....intervals import IntervalUnion
from ...pointwise.abs import _compute_safe_eb_diff_interval
from .. import BoundaryCondition, _pad_with_boundary
from ..abc import S, StencilSafeguard, T
from . import (
    FiniteDifference,
    _finite_difference,
    _finite_difference_coefficients,
    _finite_difference_offsets,
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
    boundary : str | BoundaryCondition
        Boundary condition for evaluating the finite difference near the data
        domain boundaries, e.g. by extending values.
    constant_boundary: None | int | float
        Optional constant value with which the data domain is extended for a
        constant boundary.
    axis : None | int
        The axis along which the absolute error of the finite difference is
        bounded. The default, [`None`][None], is to bound along all axes.
    """

    __slots__ = (
        "_type",
        "_order",
        "_accuracy",
        "_dx",
        "_eb_abs",
        "_boundary",
        "_constant_boundary",
        "_axis",
        "_offsets",
        "_coefficients",
        "_eb_abs_impl",
    )
    _type: FiniteDifference
    _order: int
    _accuracy: int
    _dx: int | float
    _eb_abs: int | float
    _boundary: BoundaryCondition
    _constant_boundary: None | int | float
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
        boundary: str | BoundaryCondition,
        constant_boundary: None | int | float = None,
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

        self._boundary = (
            boundary
            if isinstance(boundary, BoundaryCondition)
            else BoundaryCondition[boundary]
        )
        assert (self._boundary != BoundaryCondition.constant) == (
            constant_boundary is None
        ), (
            "constant_boundary must be provided if and only if the constant boundary condition is used"
        )
        self._constant_boundary = constant_boundary

        self._axis = axis

        self._offsets = _finite_difference_offsets(type, order, accuracy)
        self._coefficients = _finite_difference_coefficients(order, self._offsets)

        self._eb_abs_impl = (
            Fraction(eb_abs)
            * (Fraction(dx) ** order)
            / sum(abs(c) for c in self._coefficients)
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the absolute error
        bound on the finite difference over the `data`.

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

        axes = list(range(data.ndim)) if self._axis is None else [self._axis]

        axes = [
            a
            for a in axes
            if ((a >= 0) and (a < data.ndim)) or ((a < 0) and (a >= -data.ndim))
        ]

        if self._boundary == BoundaryCondition.valid:
            axes = [a for a in axes if data.shape[a] >= len(self._coefficients)]

        ok = np.ones_like(data, dtype=np.bool)

        for a in axes:
            if data.shape[a] == 0:
                continue

            omin, omax = (
                (min(*self._offsets), max(*self._offsets))
                if len(self._offsets) > 1
                else (self._offsets[0], self._offsets[0])
            )

            pad_before = max(0, -omin)
            pad_after = max(0, omax)

            data_boundary = _pad_with_boundary(
                data, self._boundary, pad_before, pad_after, self._constant_boundary, a
            )
            decoded_boundary = _pad_with_boundary(
                decoded,
                self._boundary,
                pad_before,
                pad_after,
                self._constant_boundary,
                a,
            )

            findiff_data = _finite_difference(
                data_boundary,
                self._order,
                self._offsets,
                self._coefficients,
                self._dx,
                axis=a,
            )
            findiff_decoded = _finite_difference(
                decoded_boundary,
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

            axis_ok = np.where(
                _isfinite(findiff_data),
                absolute_bound,
                np.where(
                    _isinf(findiff_data),
                    same_bits,
                    both_nan,
                ),
            )

            if self._boundary == BoundaryCondition.valid:
                s = tuple(
                    [slice(None)] * a
                    + [slice(pad_before, -pad_after if pad_after > 0 else None)]
                    + [slice(None)] * (data.ndim - a - 1)
                )
            else:
                s = tuple([slice(None)] * data.ndim)

            ok[s] &= axis_ok

        return ok  # type: ignore

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

        config = dict(
            kind=type(self).kind,
            type=self._type.name,
            order=self._order,
            accuracy=self._accuracy,
            dx=self._dx,
            eb_abs=self._eb_abs,
            boundary=self._boundary.name,
            constant_boundary=self._constant_boundary,
            axis=self._axis,
        )

        if self._constant_boundary is None:
            del config["constant_boundary"]

        return config

"""
Absolute error bound safeguard.
"""

__all__ = ["AbsoluteErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard, _as_bits
from ...intervals import (
    IntervalUnion,
    Interval,
    Lower,
    Upper,
    Minimum,
    Maximum,
    _to_total_order,
    _from_total_order,
)


class AbsoluteErrorBoundSafeguard(ElementwiseSafeguard):
    """
    The `AbsoluteErrorBoundSafeguard` guarantees that the elementwise absolute
    error is less than or equal to the provided bound `eb_abs`.

    In cases where the arithmetic evaluation of the error bound is not well-
    defined, e.g. for infinite or NaN values, producing the exact same
    bitpattern is defined to satisfy the error bound. If `equal_nan` is set to
    [`True`][True], decoding a NaN value to a NaN value with a different
    bitpattern also satisfies the error bound.

    Parameters
    ----------
    eb_abs : int | float
        The positive absolute error bound that is enforced by this safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_abs", "_equal_nan")
    _eb_abs: int | float
    _equal_nan: bool

    kind = "abs"
    _priority = 0

    def __init__(self, eb_abs: int | float, *, equal_nan: bool = False):
        assert eb_abs > 0.0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_abs = eb_abs
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check which elements in the `decoded` array satisfy the absolute error
        bound.

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
            (np.abs(data - decoded) <= self._eb_abs)
            | (_as_bits(data) == _as_bits(decoded))
            | (self._equal_nan and (np.isnan(data) & np.isnan(decoded)))
        )

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        error = decoded - data
        correction = (
            np.round(error / (self._eb_abs * 2.0)) * (self._eb_abs * 2.0)
        ).astype(data.dtype)
        corrected = decoded - correction

        return np.where(
            self.check_elementwise(data, corrected),
            corrected,
            data,
        )

    def _compute_intervals(self, data: np.ndarray) -> IntervalUnion:
        data = data.flatten()

        valid = Interval.empty_like(data)

        if np.issubdtype(data.dtype, np.floating):
            inf_valid, inf_data = valid[np.isinf(data)], data[np.isinf(data)]
            Lower(inf_data) <= inf_valid <= Upper(inf_data)

        if np.issubdtype(data.dtype, np.floating):
            nan_valid, nan_data = valid[np.isnan(data)], data[np.isnan(data)]

            if self._equal_nan:
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
                Lower(np.copysign(nan_min, nan_data)) <= nan_valid <= Upper(
                    np.copysign(nan_max, nan_data)
                )
            else:
                Lower(nan_data) <= nan_valid <= Upper(nan_data)

        finite_valid, finite_data = valid[np.isfinite(data)], data[np.isfinite(data)]
        Lower(finite_data - self._eb_abs) <= finite_valid <= Upper(
            finite_data + self._eb_abs
        )

        if np.issubdtype(data.dtype, np.integer):
            # saturate the error bounds so that they don't wrap around
            Minimum <= valid[valid._lower > data]
            valid[valid._upper < data] <= Maximum
        elif np.issubdtype(data.dtype, np.floating):
            # correct rounding errors in the lower and upper bound
            lower_bound_outside_eb_abs = np.abs(data - valid._lower) > self._eb_abs
            upper_bound_outside_eb_abs = np.abs(data - valid._upper) > self._eb_abs

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
            kind=type(self).kind, eb_abs=self._eb_abs, equal_nan=self._equal_nan
        )

"""
Absolute error bound safeguard.
"""

__all__ = ["AbsoluteErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard
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
        assert eb_abs > 0, "eb_abs must be positive"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_abs = eb_abs
        self._equal_nan = equal_nan

    def compute_safe_intervals(self, data: np.ndarray) -> IntervalUnion:
        """
        Compute the intervals in which the absolute error bound is upheld with
        respect to the `data`.

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
            else:
                Lower(data) <= valid[np.isnan(data)] <= Upper(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            Lower(data - self._eb_abs) <= valid[np.isfinite(data)] <= Upper(
                data + self._eb_abs
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
            lower_bound_outside_eb_abs = (data - valid._lower) > self._eb_abs
            upper_bound_outside_eb_abs = (valid._upper - data) > self._eb_abs

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

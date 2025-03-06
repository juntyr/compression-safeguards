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
    _to_total_order,
    _from_total_order,
)
from ...cast import to_float, from_float


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
        The non-negative absolute error bound that is enforced by this
        safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_abs", "_equal_nan")
    _eb_abs: int | float
    _equal_nan: bool

    kind = "abs"

    def __init__(self, eb_abs: int | float, *, equal_nan: bool = False):
        assert eb_abs >= 0, "eb_abs must be non-negative"
        assert np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_abs = eb_abs
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check if the `decoded` array satisfies the absolute error bound.

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

        # abs(data - decoded) <= self._eb_abs, but works for unsigned ints
        absolute_bound = (
            np.where(
                data > decoded,
                data - decoded,
                decoded - data,
            )
            <= self._eb_abs
        )
        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = _as_bits(data) == _as_bits(decoded)
        both_nan = self._equal_nan and (np.isnan(data) & np.isnan(decoded))

        ok = np.where(
            np.isfinite(data),
            absolute_bound,
            np.where(
                np.isinf(data),
                same_bits,
                both_nan if self._equal_nan else same_bits,
            ),
        )

        return bool(np.all(ok))

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
        data_float = to_float(data)

        valid = (
            Interval.empty_like(data)
            .preserve_inf(data)
            .preserve_nan(data, equal_nan=self._equal_nan)
        )

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs = min(
                np.array(self._eb_abs).astype(data_float.dtype),
                np.finfo(data_float.dtype).max,
            )
        assert eb_abs >= 0.0 and np.isfinite(eb_abs)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            Lower(from_float(data_float - eb_abs, data.dtype)) <= valid[
                np.isfinite(data)
            ] <= Upper(from_float(data_float + eb_abs, data.dtype))

        # correct rounding errors in the lower and upper bound
        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            # we don't use abs(data - bound) here to accommodate unsigned ints
            lower_bound_outside_eb_abs = (
                data_float - to_float(valid._lower)
            ) > self._eb_abs
            upper_bound_outside_eb_abs = (
                to_float(valid._upper) - data_float
            ) > self._eb_abs

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

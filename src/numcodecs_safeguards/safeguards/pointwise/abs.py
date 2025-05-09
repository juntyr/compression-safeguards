"""
Absolute error bound safeguard.
"""

__all__ = ["AbsoluteErrorBoundSafeguard"]

import numpy as np

from .abc import PointwiseSafeguard, S, T
from ...cast import (
    to_float,
    from_float,
    as_bits,
    to_total_order,
    from_total_order,
    to_finite_float,
    F,
)
from ...intervals import IntervalUnion, Interval, Lower, Upper


class AbsoluteErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `AbsoluteErrorBoundSafeguard` guarantees that the pointwise absolute
    error is less than or equal to the provided bound `eb_abs`.

    Infinite values are preserved with the same bit pattern. If `equal_nan` is
    set to [`True`][True], decoding a NaN value to a NaN value with a different
    bit pattern also satisfies the error bound. If `equal_nan` is set to
    [`False`][False], NaN values are also preserved with the same bit pattern.

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
        assert isinstance(eb_abs, int) or np.isfinite(eb_abs), "eb_abs must be finite"

        self._eb_abs = eb_abs
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
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
        same_bits = as_bits(data) == as_bits(decoded)
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

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
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

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_abs: np.ndarray = to_finite_float(self._eb_abs, data_float.dtype)
        assert eb_abs >= 0

        return _compute_safe_eb_diff_interval(
            data, data_float, -eb_abs, eb_abs, equal_nan=self._equal_nan
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
            kind=type(self).kind, eb_abs=self._eb_abs, equal_nan=self._equal_nan
        )


def _compute_safe_eb_diff_interval(
    data: np.ndarray[S, T],
    data_float: np.ndarray[S, F],
    eb_lower: np.ndarray[S | tuple[()], F],
    eb_upper: np.ndarray[S | tuple[()], F],
    equal_nan: bool,
) -> Interval[T, int]:
    dataf: np.ndarray[tuple[int], T] = data.flatten()
    dataf_float: np.ndarray[tuple[int], F] = data_float.flatten()
    eb_lowerf: np.ndarray[tuple[int], F] = eb_lower.flatten()
    eb_upperf: np.ndarray[tuple[int], F] = eb_upper.flatten()

    assert np.all(np.isfinite(eb_lowerf) & (eb_lowerf <= 0))
    assert np.all(np.isfinite(eb_upperf) & (eb_upperf >= 0))

    valid = (
        Interval.empty_like(dataf)
        .preserve_inf(dataf)
        .preserve_nan(dataf, equal_nan=equal_nan)
    )

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        Lower(from_float(dataf_float + eb_lowerf, dataf.dtype)) <= valid[
            np.isfinite(dataf)
        ] <= Upper(from_float(dataf_float + eb_upperf, dataf.dtype))

    # correct rounding errors in the lower and upper bound
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # we don't use abs(data - bound) here to accommodate unsigned ints
        lower_bound_outside_eb_abs = (dataf_float - to_float(valid._lower)) < eb_lowerf
        upper_bound_outside_eb_abs = (to_float(valid._upper) - dataf_float) > eb_upperf

    valid._lower[np.isfinite(dataf)] = from_total_order(
        to_total_order(valid._lower) + lower_bound_outside_eb_abs,
        dataf.dtype,
    )[np.isfinite(dataf)]
    valid._upper[np.isfinite(dataf)] = from_total_order(
        to_total_order(valid._upper) - upper_bound_outside_eb_abs,
        dataf.dtype,
    )[np.isfinite(dataf)]

    # a zero-error bound must preserve exactly, e.g. even for -0.0
    if np.any(eb_lowerf == 0):
        Lower(dataf) <= valid[np.isfinite(dataf) & (eb_lowerf == 0)]
    if np.any(eb_upperf == 0):
        valid[np.isfinite(dataf) & (eb_upperf == 0)] <= Upper(dataf)

    return valid

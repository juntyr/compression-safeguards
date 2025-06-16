import numpy as np

from ...utils.cast import (
    _isfinite,
    from_float,
    from_total_order,
    to_float,
    to_total_order,
)
from ...utils.intervals import Interval, Lower, Upper
from ...utils.typing import F, S, T


def compute_safe_eb_lower_upper_interval(
    data: np.ndarray[S, np.dtype[T]],
    data_float: np.ndarray[S, np.dtype[F]],
    eb_lower: np.ndarray[S, np.dtype[F]],
    eb_upper: np.ndarray[S, np.dtype[F]],
) -> Interval[T, int]:
    """
    Compute the safe interval for the `data` that upholds the provided
    `eb_upper` and `eb_lower` error bounds and preserves finite values and
    whether a value is NaN.

    Translate an error bound on a derived quantity of interest (QoI) into an
    error bound on the input data.

    Parameters
    ----------
    data : np.ndarray[S, np.dtype[T]]
        Data for which to compute the safe interval.
    data_float : np.ndarray[S, np.dtype[F]]
        Floating point version of the `data` array.
    eb_lower : np.ndarray[S, np.dtype[F]]
        Pointwise non-positive lower error bounds for the `data`.
    eb_upper : np.ndarray[S, np.dtype[F]]
        Pointwise non-negative upper error bounds for the `data`.

    Returns
    -------
    valid : Interval[T, int]
        Safe interval for the `data` to be within the error bounds.
    """

    dataf: np.ndarray[tuple[int], np.dtype[T]] = data.flatten()
    dataf_float: np.ndarray[tuple[int], np.dtype[F]] = data_float.flatten()
    eb_lowerf: np.ndarray[tuple[int], np.dtype[F]] = eb_lower.flatten()
    eb_upperf: np.ndarray[tuple[int], np.dtype[F]] = eb_upper.flatten()

    assert np.all(_isfinite(eb_lowerf) & (eb_lowerf <= 0))
    assert np.all(_isfinite(eb_upperf) & (eb_upperf >= 0))

    valid = (
        Interval.empty_like(dataf)
        .preserve_inf(dataf)
        .preserve_signed_nan(dataf, equal_nan=True)
    )

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        Lower(from_float(dataf_float + eb_lowerf, dataf.dtype)) <= valid[
            _isfinite(dataf)
        ] <= Upper(from_float(dataf_float + eb_upperf, dataf.dtype))

    # correct rounding errors in the lower and upper bound
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # we don't use abs(data - bound) here to accommodate unsigned ints
        lower_bound_outside_eb_abs = (to_float(valid._lower) - dataf_float) < eb_lowerf
        upper_bound_outside_eb_abs = (to_float(valid._upper) - dataf_float) > eb_upperf

    valid._lower[_isfinite(dataf)] = from_total_order(
        to_total_order(valid._lower) + lower_bound_outside_eb_abs,
        dataf.dtype,
    )[_isfinite(dataf)]
    valid._upper[_isfinite(dataf)] = from_total_order(
        to_total_order(valid._upper) - upper_bound_outside_eb_abs,
        dataf.dtype,
    )[_isfinite(dataf)]

    # a zero-error bound must preserve exactly, e.g. even for -0.0
    if np.any(eb_lowerf == 0):
        Lower(dataf) <= valid[_isfinite(dataf) & (eb_lowerf == 0)]
    if np.any(eb_upperf == 0):
        valid[_isfinite(dataf) & (eb_upperf == 0)] <= Upper(dataf)

    return valid

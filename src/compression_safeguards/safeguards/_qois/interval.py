import numpy as np

from ...utils._compat import _isfinite
from ...utils.cast import (
    from_float,
    from_total_order,
    to_float,
    to_total_order,
)
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import F, S, T


def compute_safe_data_lower_upper_interval_union(
    data: np.ndarray[S, np.dtype[T]],
    data_float_lower: np.ndarray[S, np.dtype[F]],
    data_float_upper: np.ndarray[S, np.dtype[F]],
) -> IntervalUnion[T, int, int]:
    """
    Compute the safe interval for the `data` that upholds the provided
    `data_upper` and `data_lower` bounds and preserves finite values and
    whether a value is NaN.

    Parameters
    ----------
    data : np.ndarray[S, np.dtype[T]]
        Data for which to compute the safe interval.
    data_float_lower : np.ndarray[S, np.dtype[F]]
        Pointwise lower bounds for the floating point version of `data`.
    data_float_upper : np.ndarray[S, np.dtype[F]]
        Pointwise upper bounds for the floating point version of `data`.

    Returns
    -------
    valid : Interval[T, int]
        Safe interval for the `data` to be within the bounds.
    """

    dataf: np.ndarray[tuple[int], np.dtype[T]] = data.flatten()
    dataf_float_lower: np.ndarray[tuple[int], np.dtype[F]] = data_float_lower.flatten()
    dataf_float_upper: np.ndarray[tuple[int], np.dtype[F]] = data_float_upper.flatten()

    valid = Interval.empty_like(dataf).preserve_inf(dataf).preserve_finite(dataf)

    Lower(
        np.maximum(valid._lower, from_float(dataf_float_lower, dataf.dtype))
    ) <= valid[_isfinite(dataf)] <= Upper(
        np.minimum(from_float(dataf_float_upper, dataf.dtype), valid._upper)
    )

    # correct rounding errors in the lower and upper bound
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        lower_outside_bound = to_float(valid._lower) < dataf_float_lower
        upper_outside_bound = to_float(valid._upper) > dataf_float_upper

    Lower(
        from_total_order(
            to_total_order(valid._lower) + lower_outside_bound,
            data.dtype,
        )
    ) <= valid[_isfinite(dataf)] <= Upper(
        from_total_order(
            to_total_order(valid._upper) - upper_outside_bound,
            data.dtype,
        )
    )

    return valid.preserve_any_nan(dataf, equal_nan=True)

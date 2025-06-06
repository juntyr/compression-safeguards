"""
Ratio (decimal) error bound safeguard.
"""

__all__ = ["RatioErrorBoundSafeguard"]

import numpy as np

from ...utils.cast import (
    _isfinite,
    _isinf,
    _isnan,
    as_bits,
    from_float,
    from_total_order,
    to_finite_float,
    to_float,
    to_total_order,
)
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import F, S, T
from .abc import PointwiseSafeguard


class RatioErrorBoundSafeguard(PointwiseSafeguard):
    """
    The `RatioErrorBoundSafeguard` guarantees that the ratios between the
    original and the decoded values and their inverse ratios are less than
    or equal to the provided `eb_ratio`.

    The ratio error is defined to be infinite if the signs of the data and
    decoded data do not match. Since the `eb_ratio` error bound must be
    finite, the `RatioErrorBoundSafeguard` also guarantees that the sign of
    each decoded value matches the sign of each original value and that a
    decoded value is zero if and only if it is zero in the original data.

    The ratio error bound is sometimes also known as a decimal error bound[^1]
    [^2] if the ratio is expressed as the difference in orders of magnitude. A
    decimal error bound of e.g. `2` (two orders of magnitude difference / x100
    ratio) can be expressed using `eb_ratio = 10**eb_decimal`.

    This safeguard can also be used to guarantee a relative-like error bound,
    e.g. `eb_ratio=1.02` corresponds to a 2% relative error bound.

    Infinite values are preserved with the same bit pattern. If `equal_nan` is
    set to [`True`][True], decoding a NaN value to a NaN value with a different
    bit pattern also satisfies the error bound. If `equal_nan` is set to
    [`False`][False], NaN values are also preserved with the same bit pattern.

    [^1]: Gustafson, J. L., & Yonemoto, I. T. (2017). Beating Floating Point at
        its Own Game: Posit Arithmetic. *Supercomputing Frontiers and
        Innovations*, 4(2). Available from:
        [doi:10.14529/jsfi170206](https://doi.org/10.14529/jsfi170206).

    [^2]: Klöwer, M., Düben, P. D., & Palmer, T. N. (2019). Posits as an
        alternative to floats for weather and climate models. *CoNGA'19:
        Proceedings of the Conference for Next Generation Arithmetic 2019*, 1-8.
        Available from:
        [doi:10.1145/3316279.3316281](https://doi.org/10.1145/3316279.3316281).

    Parameters
    ----------
    eb_ratio : int | float
        The >= 1 ratio error bound that is enforced by this safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_ratio", "_equal_nan")
    _eb_ratio: int | float
    _equal_nan: bool

    kind = "ratio"

    def __init__(self, eb_ratio: int | float, *, equal_nan: bool = False):
        assert eb_ratio >= 1, "eb_ratio must be a >= 1 ratio"
        assert isinstance(eb_ratio, int) or _isfinite(eb_ratio), (
            "eb_ratio must be finite"
        )

        self._eb_ratio = eb_ratio
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, np.dtype[T]], decoded: np.ndarray[S, np.dtype[T]]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the ratio error
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
            Pointwise, `True` if the check succeeded for this element.
        """

        data_float: np.ndarray = to_float(data)
        decoded_float: np.ndarray = to_float(decoded)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_ratio: np.ndarray = to_finite_float(self._eb_ratio, data_float.dtype)
        assert eb_ratio >= 1.0

        ratio_bound = (np.sign(data) == np.sign(decoded)) & (
            np.where(
                np.abs(data) > np.abs(decoded),
                data_float / decoded_float,
                decoded_float / data_float,
            )
            <= eb_ratio
        )
        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(decoded)
        both_nan = self._equal_nan and (_isnan(data) & _isnan(decoded))

        ok = np.where(
            data == 0,
            decoded == 0,
            np.where(
                _isfinite(data),
                ratio_bound,
                np.where(
                    _isinf(data),
                    same_bits,
                    both_nan if self._equal_nan else same_bits,
                ),
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, np.dtype[T]]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the ratio error bound is upheld with
        respect to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the ratio error bound is upheld.
        """

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_ratio: np.ndarray = to_finite_float(self._eb_ratio, data_float.dtype)
        assert eb_ratio >= 1.0

        return _compute_safe_eb_ratio_interval(
            data, data_float, eb_ratio, equal_nan=self._equal_nan
        ).into_union()

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
            eb_ratio=self._eb_ratio,
            equal_nan=self._equal_nan,
        )


def _compute_safe_eb_ratio_interval(
    data: np.ndarray[S, np.dtype[T]],
    data_float: np.ndarray[S, np.dtype[F]],
    eb_ratio: np.ndarray[tuple[()], np.dtype[F]],
    equal_nan: bool,
) -> Interval[T, int]:
    dataf: np.ndarray[tuple[int], np.dtype[T]] = data.flatten()
    dataf_float: np.ndarray[tuple[int], np.dtype[F]] = data_float.flatten()

    valid = (
        Interval.empty_like(dataf)
        .preserve_inf(dataf)
        .preserve_nan(dataf, equal_nan=equal_nan)
    )

    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        data_mul, data_div = (
            from_float(dataf_float * eb_ratio, dataf.dtype),
            from_float(dataf_float / eb_ratio, dataf.dtype),
        )
        Lower(np.where(dataf < 0, data_mul, data_div)) <= valid[
            _isfinite(dataf)
        ] <= Upper(np.where(dataf < 0, data_div, data_mul))

    # correct rounding errors in the lower and upper bound
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        lower_bound_outside_eb_ratio = (
            np.abs(
                np.where(
                    dataf < 0,
                    to_float(valid._lower) / dataf_float,
                    dataf_float / to_float(valid._lower),
                )
            )
            > eb_ratio
        )
        upper_bound_outside_eb_ratio = (
            np.abs(
                np.where(
                    dataf < 0,
                    dataf_float / to_float(valid._upper),
                    to_float(valid._upper) / dataf_float,
                )
            )
            > eb_ratio
        )

    valid._lower[_isfinite(dataf)] = from_total_order(
        to_total_order(valid._lower) + lower_bound_outside_eb_ratio,
        dataf.dtype,
    )[_isfinite(dataf)]
    valid._upper[_isfinite(dataf)] = from_total_order(
        to_total_order(valid._upper) - upper_bound_outside_eb_ratio,
        dataf.dtype,
    )[_isfinite(dataf)]

    # a ratio of 1 bound must preserve exactly, e.g. even for -0.0
    if np.any(eb_ratio == 1):
        Lower(dataf) <= valid[_isfinite(dataf) & (eb_ratio == 1)] <= Upper(dataf)

    return valid

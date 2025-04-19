"""
Root mean square error (RMSE) bound safeguard.
"""

__all__ = ["RootMeanSquareErrorBoundSafeguard"]

import numpy as np

from .abc import GlobalSafeguard, S, T, N
from ..elementwise.abs import _compute_safe_eb_abs_interval
from ...cast import (
    to_float,
    as_bits,
    to_finite_float,
)
from ...intervals import IntervalUnion, Interval


class RootMeanSquareErrorBoundSafeguard(GlobalSafeguard):
    __slots__ = ("_eb_rmse", "_equal_nan")
    _eb_rmse: int | float
    _equal_nan: bool

    kind = "rmse"

    def __init__(self, eb_rmse: int | float, *, equal_nan: bool = False):
        assert eb_rmse >= 0, "eb_rmse must be non-negative"
        assert isinstance(eb_rmse, int) or np.isfinite(eb_rmse), (
            "eb_rmse must be finite"
        )

        self._eb_rmse = eb_rmse
        self._equal_nan = equal_nan
        # TODO: add an ignore-non-finite option to ignore them in the sse

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check(self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]) -> bool:
        if data.size == 0:
            return True

        data_float: np.ndarray = to_float(data)
        decoded_float: np.ndarray = to_float(decoded)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_rmse: np.ndarray = to_finite_float(self._eb_rmse, data_float.dtype)
        assert eb_rmse >= 0.0

        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(decoded)
        both_nan = self._equal_nan and (np.isnan(data) & np.isnan(decoded))

        square_errors = np.where(
            np.isfinite(data),
            (data_float - decoded_float) * (data_float - decoded_float),
            np.where(
                np.isinf(data),
                np.where(same_bits, 0.0, np.inf),
                np.where(both_nan if self._equal_nan else same_bits, 0.0, np.inf),
            ),
        )

        return np.sqrt(np.mean(square_errors)) <= eb_rmse

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def compute_safe_intervals_with_priors(
        self,
        data: np.ndarray[S, T],
        priors: IntervalUnion[T, N, int],
    ) -> IntervalUnion[T, N, int]:
        dataf = data.flatten()
        dataf_float: np.ndarray = to_float(dataf)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_rmse: np.ndarray = to_finite_float(self._eb_rmse, dataf_float.dtype)
        assert eb_rmse >= 0.0

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_sse: np.ndarray = to_finite_float(
                self._eb_rmse,
                dataf_float.dtype,
                map=lambda rmse: rmse * rmse * dataf.size,
            )
        assert eb_sse >= 0.0

        lower_float: np.ndarray = to_float(priors._lower)
        upper_float: np.ndarray = to_float(priors._upper)

        square_errors_pessimistic = np.where(
            np.isfinite(dataf),
            np.maximum(
                np.amax(
                    (dataf_float - lower_float) * (dataf_float - lower_float), axis=0
                ),
                np.amax(
                    (upper_float - dataf_float) * (upper_float - dataf_float), axis=0
                ),
            ),
            # non-finite values will always end up with a zero error after the
            # safeguard has been applied
            0.0,
        )
        square_errors_pessimistic_sorted = np.sort(square_errors_pessimistic)

        num_remaining = np.arange(dataf.size)[::-1]

        square_errors_pessimistic_sorted_cumulative = np.cumsum(
            square_errors_pessimistic
        ) + (square_errors_pessimistic_sorted * num_remaining)

        index = np.searchsorted(square_errors_pessimistic_sorted_cumulative, eb_sse)

        if index == dataf.size:
            # even when RMSE is met, we still need to safeguard infinite and
            #  NaN values
            valid = (
                Interval.full_like(dataf)
                .preserve_inf(dataf)
                .preserve_nan(dataf, equal_nan=self._equal_nan)
            )
            return priors.intersect(valid.into_union())  # type: ignore

        if index == 0:
            eb_abs = eb_rmse
        else:
            # TODO: can we get a tighter bound by taking into account how many
            #       are left?
            eb_abs = np.array(np.sqrt(square_errors_pessimistic_sorted[index - 1]))

        valid = _compute_safe_eb_abs_interval(
            dataf, dataf_float, eb_abs, equal_nan=self._equal_nan
        )

        return priors.intersect(valid.into_union())  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind, eb_rmse=self._eb_rmse, equal_nan=self._equal_nan
        )

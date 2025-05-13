"""
Relative error bound safeguard.
"""

__all__ = ["RelativeErrorBoundSafeguard"]

import numpy as np

from ...cast import (
    _isfinite,
    _isinf,
    _isnan,
    _nan_to_zero,
    as_bits,
    to_finite_float,
    to_float,
)
from ...intervals import IntervalUnion, Lower, Upper
from .abc import PointwiseSafeguard, S, T
from .abs import _compute_safe_eb_diff_interval


class RelativeErrorBoundSafeguard(PointwiseSafeguard):
    r"""
    The `RelativeErrorBoundSafeguard` guarantees that the pointwise relative
    error is less than or equal to the provided bound `eb_rel`.

    The relative error bound is defined as follows:

    \[
        \left| x - \hat{x} \right| \leq \left| x \right| \cdot eb_{rel}
    \]

    Zero values are thus preserved with the same bit pattern.

    Infinite values are preserved with the same bit pattern. If `equal_nan` is
    set to [`True`][True], decoding a NaN value to a NaN value with a different
    bit pattern also satisfies the error bound. If `equal_nan` is set to
    [`False`][False], NaN values are also preserved with the same bit pattern.

    Parameters
    ----------
    eb_rel : int | float
        The non-negative relative error bound that is enforced by this
        safeguard.
    equal_nan: bool
        Whether decoding a NaN value to a NaN value with a different bit
        pattern satisfies the error bound.
    """

    __slots__ = ("_eb_rel", "_equal_nan")
    _eb_rel: int | float
    _equal_nan: bool

    kind = "rel"

    def __init__(self, eb_rel: int | float, *, equal_nan: bool = False):
        assert eb_rel >= 0, "eb_rel must be non-negative"
        assert isinstance(eb_rel, int) or _isfinite(eb_rel), "eb_rel must be finite"

        self._eb_rel = eb_rel
        self._equal_nan = equal_nan

    @np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore")
    def check_pointwise(
        self, data: np.ndarray[S, T], decoded: np.ndarray[S, T]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements in the `decoded` array satisfy the relative error
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

        data_float: np.ndarray = to_float(data)
        decoded_float: np.ndarray = to_float(decoded)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_rel_as_abs: np.ndarray = to_finite_float(
                self._eb_rel,
                data_float.dtype,
                map=lambda eb_rel: np.abs(data_float) * eb_rel,
            )
            eb_rel_as_abs = _nan_to_zero(eb_rel_as_abs)
        assert np.all((eb_rel_as_abs >= 0) & _isfinite(eb_rel_as_abs))

        relative_bound = np.abs(data_float - decoded_float) <= eb_rel_as_abs

        # bitwise equality for inf and NaNs (unless equal_nan)
        same_bits = as_bits(data) == as_bits(decoded)
        both_nan = self._equal_nan and (_isnan(data) & _isnan(decoded))

        ok = np.where(
            _isfinite(data),
            relative_bound,
            np.where(
                _isinf(data),
                same_bits,
                both_nan if self._equal_nan else same_bits,
            ),
        )

        return ok  # type: ignore

    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the relative error bound is upheld with
        respect to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the relative error bound is upheld.
        """

        data_float: np.ndarray = to_float(data)

        with np.errstate(
            divide="ignore", over="ignore", under="ignore", invalid="ignore"
        ):
            eb_rel_as_abs: np.ndarray = to_finite_float(
                self._eb_rel,
                data_float.dtype,
                map=lambda eb_rel: np.abs(data_float) * eb_rel,
            )
            eb_rel_as_abs = _nan_to_zero(eb_rel_as_abs)
        assert np.all((eb_rel_as_abs >= 0) & _isfinite(eb_rel_as_abs))

        valid = _compute_safe_eb_diff_interval(
            data, data_float, -eb_rel_as_abs, eb_rel_as_abs, equal_nan=self._equal_nan
        )

        # special case zero to handle +0.0 and -0.0
        dataf = data.flatten()
        Lower(dataf) <= valid[dataf == 0] <= Upper(dataf)

        return valid.into_union()  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(
            kind=type(self).kind, eb_rel=self._eb_rel, equal_nan=self._equal_nan
        )

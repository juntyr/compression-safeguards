"""
Absolute error bound safeguard.
"""

__all__ = ["AbsoluteErrorBoundSafeguard"]

import numpy as np

from . import ElementwiseSafeguard, _as_bits
from ...intervals import IntervalUnion


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

        # x <= valid <= x
        valid = IntervalUnion.from_singular(data)  # type: ignore

        # (x - eb_abs) <= valid <= (x + eb_abs)
        valid.lower[0, np.isfinite(data)] -= self._eb_abs
        valid.upper[0, np.isfinite(data)] += self._eb_abs

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)

            # saturate the error bounds so that they don't wrap around
            valid.lower[0, valid.lower[0] > data] = info.min
            valid.upper[0, valid.upper[0] < data] = info.max
        elif np.issubdtype(data.dtype, np.floating):
            # correct rounding errors in the lower and upper bound
            lower_bound_outside_eb_abs = np.abs(data - valid.lower[0]) > self._eb_abs
            upper_bound_outside_eb_abs = np.abs(data - valid.upper[0]) > self._eb_abs

            valid.lower[0, np.isfinite(data)] += lower_bound_outside_eb_abs[
                np.isfinite(data)
            ]
            valid.upper[0, np.isfinite(data)] -= upper_bound_outside_eb_abs[
                np.isfinite(data)
            ]

            if self._equal_nan:
                assert False, "not yet implemented for floats"

        return valid

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

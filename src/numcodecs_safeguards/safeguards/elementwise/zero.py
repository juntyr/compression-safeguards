"""
Zero-is-zero safeguard.
"""

__all__ = ["ZeroIsZeroSafeguard"]

import numpy as np

from .abc import ElementwiseSafeguard
from ...cast import as_bits
from ...intervals import IntervalUnion, Interval, Lower, Upper


class ZeroIsZeroSafeguard(ElementwiseSafeguard):
    """
    The `ZeroIsZeroSafeguard` guarantees that values that are zero in the input
    are also *exactly* zero in the decompressed output.

    This safeguard can also be used to enforce that another constant value is
    bitwise preserved, e.g. a missing value constant or a semantic "zero" value
    that is represented as a non-zero number.

    Beware that +0.0 and -0.0 are semantically equivalent in floating point but
    have different bitwise patterns. If you want to preserve both, you need to
    use two safeguards, one configured for each zero.

    Parameters
    ----------
    zero : int | float, optional
        The constant "zero" value that is preserved by this safeguard.
    """

    __slots__ = ("_zero",)
    _zero: int | float

    kind = "zero"

    def __init__(self, zero: int | float = 0):
        self._zero = zero

    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check that the elements are either
        - non-zero in the `data` array,
        - or zero in the `data` *and* the `decoded` array.

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

        zero_bits = as_bits(self._zero_like(data.dtype))

        return bool(
            np.all((as_bits(data) != zero_bits) | (as_bits(decoded) == zero_bits))
        )

    def compute_safe_intervals(self, data: np.ndarray) -> IntervalUnion:
        """
        Compute the intervals in which the zero-is-zero guarantee is upheld with
        respect to the `data`.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the zero-is-zero guarantee is upheld.
        """

        zero = self._zero_like(data.dtype)

        data = data.flatten()
        valid = Interval.full_like(data)

        # preserve zero values exactly, do not constrain other values
        Lower(zero) <= valid[as_bits(data) == as_bits(zero)] <= Upper(zero)

        return valid.into_union()

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind, zero=self._zero)

    def _zero_like(self, dtype: np.dtype) -> np.ndarray:
        zero = np.array(self._zero)
        if zero.dtype != dtype:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                zero = zero.astype(dtype)
        return zero

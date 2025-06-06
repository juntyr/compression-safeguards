"""
Zero-is-zero safeguard.
"""

__all__ = ["ZeroIsZeroSafeguard"]

import numpy as np

from ...utils.cast import as_bits
from ...utils.intervals import Interval, IntervalUnion, Lower, Upper
from ...utils.typing import S, T
from .abc import PointwiseSafeguard


class ZeroIsZeroSafeguard(PointwiseSafeguard):
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

    def check_pointwise(
        self, data: np.ndarray[S, np.dtype[T]], decoded: np.ndarray[S, np.dtype[T]]
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check which elements are either
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
        ok : np.ndarray
            Pointwise, `True` if the check succeeded for this element.
        """

        zero_bits = as_bits(self._zero_like(data.dtype))

        return (as_bits(data) != zero_bits) | (as_bits(decoded) == zero_bits)

    def compute_safe_intervals(
        self, data: np.ndarray[S, np.dtype[T]]
    ) -> IntervalUnion[T, int, int]:
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

        dataf = data.flatten()
        valid = Interval.full_like(dataf)

        # preserve zero values exactly, do not constrain other values
        Lower(zero) <= valid[as_bits(dataf) == as_bits(zero)] <= Upper(zero)

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

    def _zero_like(self, dtype: np.dtype[T]) -> np.ndarray[tuple[()], np.dtype[T]]:
        zero = np.array(self._zero)
        if zero.dtype != dtype:
            with np.errstate(
                divide="ignore", over="ignore", under="ignore", invalid="ignore"
            ):
                zero = zero.astype(dtype)
        return zero  # type: ignore

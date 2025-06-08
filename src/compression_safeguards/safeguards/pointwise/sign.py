"""
Sign-preserving safeguard.
"""

__all__ = ["SignPreservingSafeguard"]

import numpy as np

from ...utils.binding import Bindings
from ...utils.intervals import Interval, IntervalUnion, Lower, Maximum, Minimum, Upper
from ...utils.typing import S, T
from .abc import PointwiseSafeguard


class SignPreservingSafeguard(PointwiseSafeguard):
    r"""
    The `SignPreservingSafeguard` guarantees that values have the same sign
    (-1, 0, +1) in the decompressed output as they have in the input data.

    The sign for NaNs is derived from their sign bit, e.g. sign(-NaN) = -1.

    This safeguard should be combined with e.g. an error bound, as it by itself
    accepts *any* value with the same sign.
    """

    __slots__ = ()

    kind = "sign"

    def __init__(self):
        pass

    def check_pointwise(
        self,
        data: np.ndarray[S, np.dtype[T]],
        decoded: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> np.ndarray[S, np.dtype[np.bool]]:
        """
        Check for which elements in the `decoded` array the signs match the
        signs of the `data` array elements'.

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

        return self._sign(data) == self._sign(decoded)

    def compute_safe_intervals(
        self,
        data: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
    ) -> IntervalUnion[T, int, int]:
        """
        Compute the intervals in which the `data`'s sign is preserved.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the `data`'s sign is preserved.
        """

        # tiny: the smallest-in-magnitude non-zero value
        tiny, neg_tiny = self._tiny_like(data)

        dataf = data.flatten()
        valid = Interval.empty_like(dataf)

        sign = self._sign(dataf)

        # preserve zero-sign values exactly
        Lower(dataf) <= valid[sign == 0] <= Upper(dataf)
        Minimum <= valid[sign == -1] <= Upper(neg_tiny)
        Lower(tiny) <= valid[sign == +1] <= Maximum

        return valid.into_union()

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind)

    def _sign(self, x: np.ndarray[S, np.dtype[T]]) -> np.ndarray[S, np.dtype[np.int_]]:
        zero = np.array(0, dtype=x.dtype)

        # if >0: (true) * (1 - 0*2) = 1
        # if =0: (false) * (1 - 0*2) = 0
        # if <0: (true) * (1 - 1*2) = -1
        return (x != zero) * (1 - np.signbit(x) * 2)

    def _tiny_like(
        self, a: np.ndarray[S, np.dtype[T]]
    ) -> tuple[np.ndarray[tuple[()], np.dtype[T]], np.ndarray[tuple[()], np.dtype[T]]]:
        if np.issubdtype(a.dtype, np.integer):
            tiny = np.array(1, dtype=a.dtype)
            neg_tiny = np.array(-tiny)
        else:
            info = np.finfo(a.dtype)  # type: ignore
            tiny = np.array(info.smallest_subnormal)
            neg_tiny = np.array(-info.smallest_subnormal)

        return tiny, neg_tiny  # type: ignore

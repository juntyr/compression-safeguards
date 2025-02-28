"""
Sign-preserving safeguard.
"""

__all__ = ["SignPreservingSafeguard"]

import numpy as np

from . import ElementwiseSafeguard
from ...intervals import IntervalUnion, Interval, Lower, Upper, Minimum, Maximum


class SignPreservingSafeguard(ElementwiseSafeguard):
    r"""
    The `SignPreservingSafeguard` guarantees that values have the same sign
    (-1, 0, +1) in the decompressed output as they have in the input data.

    The sign for NaNs is derived from their sign bit, e.g. sign(-NaN) = -1.
    """

    __slots__ = ()

    kind = "sign"
    _priority = 1

    def __init__(self):
        pass

    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check which elements have matching signs in the `data` and the
        `decoded` array.

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

        return self._sign(data) == self._sign(decoded)

    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        return np.where(
            self.check_elementwise(data, decoded),
            decoded,
            data,
        )

    def _compute_intervals(self, data: np.ndarray) -> IntervalUnion:
        sign = self._sign(data).flatten()

        zero = np.zeros((), dtype=data.dtype)

        if np.issubdtype(data.dtype, np.integer):
            tiny = np.ones((), dtype=data.dtype)
            neg_tiny = np.array(-tiny)
        else:
            info = np.finfo(data.dtype)
            tiny = np.array(info.smallest_subnormal)
            neg_tiny = np.array(-info.smallest_subnormal)

        valid = Interval.empty_like(data)

        Lower(zero) <= valid[sign == 0] <= Upper(zero)
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

    def _sign(self, x: np.ndarray) -> np.ndarray:
        zero = np.zeros((), dtype=x.dtype)

        # if >0: (true) * (1 - 0*2) = 1
        # if =0: (false) * (1 - 0*2) = 0
        # if <0: (true) * (1 - 1*2) = -1
        return (x != zero) * (1 - np.signbit(x) * 2)

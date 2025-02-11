"""
Zero-is-zero safeguard.
"""

__all__ = ["ZeroIsZeroSafeguard"]

import numpy as np

from . import ElementwiseSafeguard, _as_bits


class ZeroIsZeroSafeguard(ElementwiseSafeguard):
    """
    The `ZeroIsZeroSafeguard` guarantees that values that are zero in the input
    are also *exactly* zero in the decompressed output.

    This safeguard can also be used to enforce that another constant value is
    bitwise preserved, e.g. a missing value constant or a semantic "zero" value
    that is represented as a non-zero number.

    Parameters
    ----------
    zero : int | float, optional
        The constant "zero" value that is preserved by this safeguard.
    """

    __slots__ = ("_zero",)
    _zero: int | float

    kind = "zero"
    _priority = -1

    def __init__(self, zero: int | float = 0):
        self._zero = zero

    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
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
            Per-element, `True` if the check succeeded for this element.
        """

        zero = np.array(self._zero)
        if zero.dtype != data.dtype:
            zero = zero.astype(data.dtype)
        zero = _as_bits(zero)

        return (_as_bits(data) != zero) | (_as_bits(decoded) == zero)

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

    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        return dict(kind=type(self).kind, zero=self._zero)

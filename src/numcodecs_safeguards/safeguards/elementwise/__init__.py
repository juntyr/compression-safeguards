"""
Implementations for the provided elementwise safeguards.
"""

__all__ = ["ElementwiseSafeguard"]

from abc import ABC, abstractmethod
from typing import final, Any, TypeVar

import numcodecs.compat
import numpy as np
from numcodecs.abc import Codec

from .. import Safeguard
from ...intervals import IntervalUnion

T = TypeVar("T", bound=np.dtype)
S = TypeVar("S", bound=tuple[int, ...])


class ElementwiseSafeguard(Safeguard, ABC):
    """
    Elementwise safeguard abstract base class.

    Elementwise safeguards can identitfy individual elements that violate the
    property enforced by the safeguard.
    """

    @abstractmethod
    def compute_safe_intervals(
        self, data: np.ndarray[S, T]
    ) -> IntervalUnion[T, Any, Any]:
        """
        Compute the intervals in which the safeguard's guarantees with respect
        to the `data` are upheld.

        The returned union of intervals must not have any overlap between the
        intervals inside the union. The `data` must be contained in the union.

        Parameters
        ----------
        data : np.ndarray
            Data for which the safe intervals should be computed.

        Returns
        -------
        intervals : IntervalUnion
            Union of intervals in which the safeguard's guarantees are upheld.
        """

        pass

    @final
    @staticmethod
    def _encode_correction(
        decoded: np.ndarray[S, T], corrected: np.ndarray[S, T], lossless: Codec
    ) -> bytes:
        """
        Encode the combined correction from one or more elementwise safeguards
        to [`bytes`][bytes].

        Parameters
        ----------
        decoded : np.ndarray
            Decoded data.
        corrected : np.ndarray
            Corrected decoded data.
        lossless : Codec
            Lossless codec to compress the correction with.

        Returns
        -------
        correction : bytes
            Encoded correction for the `decoded` array.
        """

        decoded_bits = _buffer_as_bits(decoded)
        corrected_bits = _buffer_as_bits(corrected, like=decoded)

        correction_bits = decoded_bits - corrected_bits

        correction_bytes = lossless.encode(correction_bits)

        return numcodecs.compat.ensure_bytes(correction_bytes)

    @final
    @staticmethod
    def _apply_correction(
        decoded: np.ndarray[S, T], correction: bytes, lossless: Codec
    ) -> np.ndarray[S, T]:
        """
        Apply the encoded `correction` to the `decoded` array.

        Parameters
        ----------
        decoded : np.ndarray
            Decoded data.
        correction : bytes
            Encoded correction for the `decoded` array.
        lossless : Codec
            Lossless codec to decompress the correction with.

        Returns
        -------
        corrected : np.ndarray
            Corrected decoded data.
        """

        decoded_bits = _buffer_as_bits(decoded)

        correction_bits = (
            numcodecs.compat.ensure_ndarray(lossless.decode(correction))
            .view(decoded_bits.dtype)
            .reshape(decoded_bits.shape)
        )

        return (
            (decoded_bits - correction_bits).view(decoded.dtype).reshape(decoded.shape)
        )


def _buffer_as_bits(a: np.ndarray, *, like: None | np.ndarray = None) -> np.ndarray:
    """
    Reinterpret the array `a` to an array of equal-sized uints (bits).

    Parameters
    ----------
    a : np.ndarray
        Input array.
    like : None | np.ndarray
        Array whose `dtype` should be used to derive the uint type.
        If [`None`][None], `a` is used to derive the uint type.

    Returns
    -------
    bits : np.ndarray
        Binary representation of the input data.
    """
    return np.frombuffer(
        a,
        dtype=np.dtype(
            (a if like is None else like).dtype.str.replace("f", "u").replace("i", "u")
        ),
    )

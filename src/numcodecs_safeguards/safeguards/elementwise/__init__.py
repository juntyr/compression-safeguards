"""
Implementations for the provided elementwise safeguards.
"""

__all__ = ["ElementwiseSafeguard"]

from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

import numcodecs.compat
import numpy as np
import varint

from numcodecs.abc import Codec

from .. import Safeguard
from ...intervals import IntervalUnion


class ElementwiseSafeguard(Safeguard, ABC):
    """
    Elementwise safeguard abstract base class.

    Elementwise safeguards can identitfy individual elements that violate the
    property enforced by the safeguard.
    """

    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check if the `decoded` array upholds the property enforced by this
        safeguard.

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

        return bool(np.all(self.check_elementwise(data, decoded)))

    @abstractmethod
    def check_elementwise(self, data: np.ndarray, decoded: np.ndarray) -> np.ndarray:
        """
        Check which elements in the `decoded` array uphold the property
        enforced by this safeguard.

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

        # TODO: implement non-abstract based on the intervals
        pass

    @abstractmethod
    def _compute_correction(
        self,
        data: np.ndarray,
        decoded: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the correction for the `decoded` array to uphold the property
        enforced by this safeguard.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        corrected : np.ndarray
            Corrected decoded data.

            If the `decoded` array already upholds the property, it can be
            returned. It is always valid to return elements of the `data`.
        """

        # TODO: remove
        pass

    # TODO: @abstractmethod
    def _compute_intervals(self, data: np.ndarray) -> IntervalUnion:
        raise NotImplementedError("todo")

    @staticmethod
    def _encode_correction(
        decoded: np.ndarray, correction: np.ndarray, lossless: Codec
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

        decoded_bits = _as_bits(decoded)
        corrected_bits = _as_bits(correction, like=decoded)

        correction_bits = decoded_bits - corrected_bits

        correction_bytes = _runlength_encode(correction_bits)
        correction_bytes = lossless.encode(correction_bytes)

        return numcodecs.compat.ensure_bytes(correction_bytes)

    @staticmethod
    def _apply_correction(
        decoded: np.ndarray, correction: bytes, lossless: Codec
    ) -> np.ndarray:
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

        decoded_bits = _as_bits(decoded)

        correction = lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)

        correction_bits = _runlength_decode(correction, like=decoded_bits)

        return (decoded_bits - correction_bits).view(decoded.dtype)


def _as_bits(a: np.ndarray, *, like: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reinterpret the array `a` to an array of equal-sized uints (bits).

    Parameters
    ----------
    a : np.ndarray
        Input array.
    like : Optional[np.ndarray]
        Optional array whose `dtype` should be used to derive the uint type.

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


def _runlength_encode(a: np.ndarray) -> bytes:
    """
    Encode the array `a` using run-length encoding.

    Currently, only zero-runs are RL-encoded and non-zero values are stored
    verbatim in non-zero runs.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    rle : bytes
        Run-length encoded bytes.
    """

    a = a.flatten()
    zeros = a == 0

    # run-length encoding of the "is-a-zero" mask
    starts = np.r_[0, np.flatnonzero(zeros[1:] != zeros[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(a)])

    # store all non-zero values and the first zero of each zero-run
    indices = np.r_[0, np.flatnonzero((~zeros[1:]) | (zeros[1:] != zeros[:-1])) + 1]
    values = a[indices]

    encoded = [varint.encode(length) for length in lengths]
    encoded.append(values.tobytes())

    return b"".join(encoded)


def _runlength_decode(b: bytes, *, like: np.ndarray) -> np.ndarray:
    """
    Decode the bytes `b` using run-length encoding.

    Currently, only zero-runs are RL-encoded and non-zero values are stored
    verbatim in non-zero runs.

    Parameters
    ----------
    rle : bytes
        Run-length encoded bytes.
    like : Optional[np.ndarray]
        Optional array whose `dtype` and shape determine the output's.

    Returns
    -------
    decoded : np.ndarray
        Run-length decoded array.
    """

    lengths = []
    total_length = 0

    b_io = BytesIO(b)

    while total_length < like.size:
        length = varint.decode_stream(b_io)
        assert length > 0
        total_length += length
        lengths.append(length)

    assert total_length >= 0

    decoded = np.zeros(like.size, dtype=like.dtype)

    if total_length == 0:
        return decoded.reshape(like.shape)

    values = np.frombuffer(b, dtype=like.dtype, offset=b_io.tell())

    id, iv = 0, 0
    for length in lengths:
        if values[iv] == 0:
            iv += 1
        else:
            decoded[id : id + length] = values[iv : iv + length]
            iv += length
        id += length

    return decoded.reshape(like.shape)

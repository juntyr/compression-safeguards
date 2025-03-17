"""
Implementations for the provided elementwise safeguards.
"""

__all__ = ["ElementwiseSafeguard"]

from abc import ABC, abstractmethod
from io import BytesIO
from typing import final, Any, TypeVar

import numcodecs.compat
import numpy as np
import varint
from dahuffman import HuffmanCodec
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

        correction_bytes = _runlength_encode(correction_bits)
        correction_bytes = lossless.encode(correction_bytes)

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

        correction = lossless.decode(correction)
        correction = numcodecs.compat.ensure_bytes(correction)

        correction_bits = _runlength_decode(correction, like=decoded_bits)

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

    # FIXME: use a different encoding
    import pickle

    a = a.flatten()

    huffman = HuffmanCodec.from_data(a)
    encoded = huffman.encode(a)

    a_delta = a.copy()
    a_delta[1:] = np.diff(a_delta)

    huffman_delta = HuffmanCodec.from_data(a_delta)
    encoded_delta = huffman_delta.encode(a_delta)

    marker, huffman, encoded = (
        (bytes([0]), huffman, encoded)
        if len(encoded) <= len(encoded_delta)
        else (bytes([1]), huffman_delta, encoded_delta)
    )

    table = pickle.dumps(huffman.get_code_table())

    return marker + varint.encode(len(table)) + table + encoded

    # zeros = a == 0

    # # run-length encoding of the "is-a-zero" mask
    # starts = np.r_[0, np.flatnonzero(zeros[1:] != zeros[:-1]) + 1]
    # lengths = np.diff(np.r_[starts, len(a)])

    # # store all non-zero values and the first zero of each zero-run
    # indices = np.r_[0, np.flatnonzero((~zeros[1:]) | (zeros[1:] != zeros[:-1])) + 1]
    # values = a[indices]

    # encoded = [varint.encode(length) for length in lengths]
    # encoded.append(values.tobytes())

    # return b"".join(encoded)


def _runlength_decode(b: bytes, *, like: np.ndarray) -> np.ndarray:
    """
    Decode the bytes `b` using run-length encoding.

    Currently, only zero-runs are RL-encoded and non-zero values are stored
    verbatim in non-zero runs.

    Parameters
    ----------
    rle : bytes
        Run-length encoded bytes.
    like : np.ndarray
        Array whose `dtype` and shape determine the output's.

    Returns
    -------
    decoded : np.ndarray
        Run-length decoded array.
    """

    # FIXME: use a different encoding
    import pickle

    marker, b = b[0], b[1:]

    b_io = BytesIO(b)

    table_len = varint.decode_stream(b_io)
    table = pickle.loads(b[b_io.tell() : b_io.tell() + table_len])
    huffman = HuffmanCodec(table)

    decoded = np.array(huffman.decode(b[b_io.tell() + table_len :]))

    if marker != 0:
        decoded = np.cumsum(decoded, dtype=decoded.dtype)

    return decoded.reshape(like.shape)

    # lengths = []
    # total_length = 0

    # b_io = BytesIO(b)

    # while total_length < like.size:
    #     length = varint.decode_stream(b_io)
    #     assert length > 0
    #     total_length += length
    #     lengths.append(length)

    # assert total_length >= 0

    # decoded = np.zeros(like.size, dtype=like.dtype)

    # if total_length == 0:
    #     return decoded.reshape(like.shape)

    # values = np.frombuffer(b, dtype=like.dtype, offset=b_io.tell())

    # id, iv = 0, 0
    # for length in lengths:
    #     if values[iv] == 0:
    #         iv += 1
    #     else:
    #         decoded[id : id + length] = values[iv : iv + length]
    #         iv += length
    #     id += length

    # return decoded.reshape(like.shape)

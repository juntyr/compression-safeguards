"""
Helper classes for lossless encoding for the codec with safeguards.
"""

__all__ = ["Lossless"]

from dataclasses import dataclass, field
from functools import reduce
from io import BytesIO
from sys import byteorder
from typing import Any, TypeVar

import numcodecs
import numcodecs.compat
import numpy as np
import varint
from compression_safeguards.utils.typing import JSON
from numcodecs.abc import Codec
from numcodecs_combinators.best import PickBestCodec
from numcodecs_combinators.framed import FramedCodecStack
from numcodecs_combinators.stack import CodecStack
from numcodecs_delta import BinaryDeltaCodec
from numcodecs_shuffle import TypedByteShuffleCodec
from numcodecs_tokenize import TokenizeCodec
from typing_extensions import Buffer  # MSPV 3.12


def _default_lossless_for_safeguards() -> Codec:
    return PickBestCodec(
        CodecStack(
            TokenizeCodec(),
            PackZeroCodec(),
            TypedByteShuffleCodec(),
            FramedCodecStack(numcodecs.zstd.Zstd(level=3)),
        ),
        CodecStack(
            BinaryDeltaCodec(),
            TokenizeCodec(),
            PackZeroCodec(),
            TypedByteShuffleCodec(),
            FramedCodecStack(numcodecs.zstd.Zstd(level=3)),
        ),
    )


@dataclass(kw_only=True)
class Lossless:
    """
    Configuration for the lossless encoding used by the
    [`SafeguardsCodec`][...SafeguardsCodec] to encode the
    wrapped codec's encoded data and any safeguards-computed corrections.
    """

    for_codec: None | dict[str, JSON] | Codec = None
    """
    Lossless codec (configuration) that is applied to wrapped codec's encoding.

    By default, no further lossless compression is applied.
    """

    for_safeguards: dict[str, JSON] | Codec = field(
        default_factory=_default_lossless_for_safeguards,
    )
    """
    Lossless codec (configuration) that is applied to the safeguard-computed
    corrections.

    The default is considered an implementation detail.
    """


S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """


class PackZeroCodec(Codec):
    codec_id = "pack-zero"

    def encode(self, buf: Buffer) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape
        a = _as_bits(a.flatten())

        is_zero = a == 0
        packed_is_zero = np.packbits(is_zero, axis=None, bitorder="big")
        a_non_zero = np.extract(~is_zero, a)

        # message: dtype shape table encoded
        message = []

        message.append(varint.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(varint.encode(len(shape)))
        for s in shape:
            message.append(varint.encode(s))

        message.append(packed_is_zero.tobytes())

        # insert padding to align with itemsize
        message.append(
            b"\0" * (dtype.itemsize - (sum(len(m) for m in message) % dtype.itemsize))
        )

        # ensure that the table keys are encoded in little endian binary
        a_byteorder = a.dtype.byteorder
        a_byteorder = (
            a_byteorder
            if a_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )
        if a_byteorder != "<":
            a_non_zero = a_non_zero.byteswap()
        message.append(a_non_zero.tobytes())

        message = b"".join(message)
        return np.frombuffer(
            message, dtype=a_non_zero.dtype, count=len(message) // dtype.itemsize
        )

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))

        shape = tuple(
            varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
        )
        size = reduce(lambda a, b: a * b, shape, 1)

        packed_is_zero = np.frombuffer(
            b_io.read((size + 7) // 8),
            dtype=np.uint8,
            count=(size + 7) // 8,
        )
        is_zero = np.unpackbits(
            packed_is_zero, axis=None, count=size, bitorder="big"
        ).astype(np.bool)
        num_non_zero = is_zero.size - np.sum(is_zero)

        # remove padding to align with itemsize
        b_io.read(dtype.itemsize - (b_io.tell() % dtype.itemsize))

        compressed = np.frombuffer(
            b_io.read(num_non_zero * dtype.itemsize),
            dtype=_dtype_bits(dtype).newbyteorder("<"),
            count=num_non_zero,
        )
        dtype_bits_byteorder = _dtype_bits(dtype).byteorder
        dtype_bits_byteorder = (
            dtype_bits_byteorder
            if dtype_bits_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )
        if dtype_bits_byteorder != "<":
            compressed = compressed.byteswap()

        decoded = np.zeros(size, compressed.dtype)
        np.place(decoded, ~is_zero, compressed)
        decoded = decoded.view(dtype).reshape(shape)

        return numcodecs.compat.ndarray_copy(decoded, out)


def _as_bits(a: np.ndarray[S, np.dtype[Any]], /) -> np.ndarray[S, np.dtype[Any]]:
    """
    Reinterprets the array `a` to its binary (unsigned integer) representation.

    Parameters
    ----------
    a : np.ndarray[S, np.dtype[Any]]
        The array to reinterpret as binary.

    Returns
    -------
    binary : np.ndarray[S, np.dtype[Any]]
        The binary representation of the array `a`.
    """

    return a.view(_dtype_bits(a.dtype))  # type: ignore


def _dtype_bits(dtype: np.dtype) -> np.dtype:
    """
    Converts the `dtype` to its binary (unsigned integer) representation.

    Parameters
    ----------
    dtype : np.dtype
        The dtype to convert.

    Returns
    -------
    binary : np.dtype
        The binary dtype with equivalent size and alignment but unsigned
        integer kind.
    """

    return np.dtype(dtype.str.replace("f", "u").replace("i", "u"))

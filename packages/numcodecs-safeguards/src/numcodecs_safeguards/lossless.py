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

        # FIXME: MPSV 3.11 numpy 2.3: sorted=True
        unique, counts = np.unique(a, return_counts=True)
        argsort = np.argsort(-counts, stable=True)  # sort with decreasing order
        unique = unique[argsort]
        counts = counts[argsort]

        bitsavings = counts * (dtype.itemsize * 8 - 1)
        bitcosts = a.size - np.cumsum(counts) + dtype.itemsize * 8

        factor = 2

        num_bitmaps = np.argmax((bitcosts * factor) >= bitsavings)

        # message: dtype shape is-zero [padding] non-zero
        message = []

        message.append(varint.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(varint.encode(len(shape)))
        for s in shape:
            message.append(varint.encode(s))

        message.append(varint.encode(num_bitmaps))

        a_byteorder = a.dtype.byteorder
        a_byteorder = (
            a_byteorder
            if a_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )

        for u in unique[:num_bitmaps]:
            ule = u
            if a_byteorder != "<":
                ule = ule.byteswap()
            message.append(ule.tobytes())

            is_u = a == u
            packed_is_u = np.packbits(is_u, axis=None, bitorder="big")
            a = np.extract(~is_u, a)

            message.append(packed_is_u.tobytes())

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
            a = a.byteswap()
        message.append(a.tobytes())

        encoded_bytes = b"".join(message)

        encoded: np.ndarray[tuple[int], np.dtype[np.unsignedinteger]] = np.frombuffer(
            encoded_bytes,
            dtype=a.dtype,
            count=len(encoded_bytes) // dtype.itemsize,
        )

        return encoded  # type: ignore

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)
        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))

        shape = tuple(
            varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
        )
        size = reduce(lambda a, b: a * b, shape, 1)

        num_bitmaps = varint.decode_stream(b_io)

        indices = np.arange(size)

        dtype_bits_byteorder = _dtype_bits(dtype).byteorder
        dtype_bits_byteorder = (
            dtype_bits_byteorder
            if dtype_bits_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )

        decoded = np.zeros(size, _dtype_bits(dtype))

        for i in range(num_bitmaps):
            u = np.frombuffer(
                b_io.read(dtype.itemsize),
                dtype=_dtype_bits(dtype).newbyteorder("<"),
                count=1,
            )
            if dtype_bits_byteorder != "<":
                u = u.byteswap()

            packed_is_u = np.frombuffer(
                b_io.read((indices.size + 7) // 8),
                dtype=np.uint8,
                count=(indices.size + 7) // 8,
            )
            is_u = np.unpackbits(
                packed_is_u, axis=None, count=indices.size, bitorder="big"
            ).astype(np.bool)

            decoded[indices[is_u]] = u
            indices = np.extract(~is_u, indices)

        # remove padding to align with itemsize
        b_io.read(dtype.itemsize - (b_io.tell() % dtype.itemsize))

        leftover: np.ndarray = np.frombuffer(
            b_io.read(indices.size * dtype.itemsize),
            dtype=_dtype_bits(dtype).newbyteorder("<"),
            count=indices.size,
        )
        dtype_bits_byteorder = _dtype_bits(dtype).byteorder
        dtype_bits_byteorder = (
            dtype_bits_byteorder
            if dtype_bits_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )
        if dtype_bits_byteorder != "<":
            leftover = leftover.byteswap()

        decoded[indices] = leftover

        decoded = decoded.view(dtype).reshape(shape)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


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

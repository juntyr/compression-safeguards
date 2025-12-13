"""
Helper classes for lossless encoding for the codec with safeguards.
"""

__all__ = ["Lossless"]

from dataclasses import dataclass, field
from functools import reduce

import numcodecs
from compression_safeguards.utils.typing import JSON
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray, ndarray_copy
from numcodecs_combinators.best import PickBestCodec
from numcodecs_combinators.stack import CodecStack
from numcodecs_delta import BinaryDeltaCodec
from numcodecs_huffman import HuffmanCodec


def _default_lossless_for_safeguards() -> Codec:
    # return PickBestCodec(
    #     CodecStack(HuffmanCodec(), numcodecs.zstd.Zstd(level=3)),
    #     CodecStack(RemapCodec(), Shuffle(), numcodecs.zstd.Zstd(level=3)),
    # )
    return CodecStack(
        RemapCodec(), Shuffle(), numcodecs.zstd.Zstd(level=3)
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


class Shuffle(Codec):
    codec_id = "shuffle"

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)

        self.itemsize = buf.dtype.itemsize

        return numcodecs.Shuffle(elementsize=self.itemsize).encode(buf)

    def decode(self, buf, out=None):
        # FIXME: hack
        return numcodecs.Shuffle(elementsize=self.itemsize).decode(buf, out=out)


from io import BytesIO
from sys import byteorder
from typing import Any, TypeVar

import numcodecs.compat
import numpy as np
import varint
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12

S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """


class RemapCodec(Codec):
    codec_id = "remap"

    def encode(self, buf: Buffer) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape
        a = _as_bits(a.flatten())

        # FIXME: MPSV 3.11 numpy 2.3: sorted=True
        unique, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        argsort = np.argsort(-counts, stable=True)  # sort with decreasing order
        argsortinv = np.argsort(argsort, stable=True)

        # message: dtype shape table encoded
        message = []

        message.append(varint.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(varint.encode(len(shape)))
        for s in shape:
            message.append(varint.encode(s))

        message.append(varint.encode(unique.size))

        # insert padding to align with itemsize
        message.append(
            b"\0" * (dtype.itemsize - (sum(len(m) for m in message) % dtype.itemsize))
        )

        # ensure that the table keys are encoded in little endian binary
        table_keys_array = unique[argsort]
        table_keys_byteorder = table_keys_array.dtype.byteorder
        table_keys_byteorder = (
            table_keys_byteorder
            if table_keys_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )
        if table_keys_byteorder != "<":
            table_keys_array = table_keys_array.byteswap()
        message.append(table_keys_array.tobytes())

        encoded = argsortinv[inverse].astype(a.dtype)
        if table_keys_byteorder != "<":
            encoded = encoded.byteswap()
        message.append(encoded.tobytes())
        # message.append(numcodecs.Shuffle(dtype.itemsize).encode(encoded).tobytes())

        message = b"".join(message)
        return np.frombuffer(
            message, dtype=table_keys_array.dtype, count=len(message) // dtype.itemsize
        )

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))

        shape = tuple(
            varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
        )

        table_len = varint.decode_stream(b_io)

        # remove padding to align with itemsize
        b_io.read(dtype.itemsize - (b_io.tell() % dtype.itemsize))

        # decode the table keys from little endian binary
        # change them back to dtype_bits byte order
        table_keys = np.frombuffer(
            b_io.read(table_len * dtype.itemsize),
            dtype=_dtype_bits(dtype).newbyteorder("<"),
            count=table_len,
        )
        dtype_bits_byteorder = _dtype_bits(dtype).byteorder
        dtype_bits_byteorder = (
            dtype_bits_byteorder
            if dtype_bits_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )
        if dtype_bits_byteorder != "<":
            table_keys = table_keys.byteswap()

        # encoded = np.empty(shape, dtype=_dtype_bits(dtype).newbyteorder("<"))
        # numcodecs.Shuffle(dtype.itemsize).decode(b_io.read(), out=encoded)
        encoded = np.frombuffer(
            b_io.read(),
            dtype=_dtype_bits(dtype).newbyteorder("<"),
            count=np.prod(shape, dtype=np.uintp),
        )
        if dtype_bits_byteorder != "<":
            encoded = encoded.byteswap()

        decoded = table_keys[encoded].view(dtype).reshape(shape)

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

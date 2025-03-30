"""
Helper classes for lossless encoding for the codec and quantizer with safeguards.
"""

__all__ = ["Lossless"]

from dataclasses import dataclass, field
from io import BytesIO
from typing_extensions import Buffer  # MSPV 3.12

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from dahuffman import HuffmanCodec as DaHuffmanCodec
from dahuffman.huffmancodec import _EOF
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .cast import as_bits


class HuffmanCodec(Codec):
    __slots__ = ()

    codec_id: str = "safeguards.lossless.huffman"  # type: ignore

    def encode(self, buf: Buffer) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape
        a = as_bits(a.flatten())

        huffman = DaHuffmanCodec.from_data(a)
        encoded = huffman.encode(a)

        # message: dtype shape table encoded
        message = []

        message.append(varint.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(varint.encode(len(shape)))
        for s in shape:
            message.append(varint.encode(s))

        table = huffman.get_code_table()
        table_no_eof = [
            (k, e) for k, e in huffman.get_code_table().items() if k != _EOF
        ]
        message.append(varint.encode(len(table_no_eof)))
        message.append(np.array([k for k, _ in table_no_eof]).tobytes())
        for k, (bitsize, value) in table_no_eof:
            message.append(varint.encode(bitsize))
            message.append(varint.encode(value))
        bitsize, value = table[_EOF]
        message.append(varint.encode(bitsize))
        message.append(varint.encode(value))

        message.append(encoded)

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))

        shape = tuple(
            varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
        )

        table_len = varint.decode_stream(b_io)
        table_keys = np.frombuffer(
            b_io.read(table_len * dtype.itemsize), dtype=dtype, count=table_len
        )
        table = dict()
        for k in table_keys:
            table[k] = (varint.decode_stream(b_io), varint.decode_stream(b_io))
        table[_EOF] = (varint.decode_stream(b_io), varint.decode_stream(b_io))
        huffman = DaHuffmanCodec(table)

        decoded = np.array(huffman.decode(b_io.read())).reshape(shape)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(HuffmanCodec)


class DeltaHuffmanCodec(HuffmanCodec):
    __slots__ = ()

    codec_id: str = "safeguards.lossless.delta_huffman"  # type: ignore

    def encode(self, buf: Buffer) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape
        a = as_bits(a.flatten())

        a_delta = a.copy()
        a_delta[1:] = np.diff(a_delta)

        encoded: bytes = super().encode(a.view(dtype).reshape(shape))
        encoded_delta: bytes = super().encode(a_delta.view(dtype).reshape(shape))

        if len(encoded_delta) < len(encoded):
            return bytes([1]) + encoded_delta

        return bytes([0]) + encoded

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        marker, b = b[0], b[1:]

        decoded = numcodecs.compat.ensure_ndarray(super().decode(b, out=out))

        if marker != 0:
            shape, dtype = decoded.shape, decoded.dtype
            decoded = as_bits(decoded).flatten()
            decoded = np.cumsum(decoded, dtype=decoded.dtype)
            decoded = decoded.view(dtype).reshape(shape)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(DeltaHuffmanCodec)


def _default_lossless_for_safeguards():
    return CodecStack(
        DeltaHuffmanCodec(),
        numcodecs.zstd.Zstd(level=3),
    )


@dataclass
class Lossless:
    """
    Configuration for the lossless encoding used by the
    [`SafeguardsCodec`][numcodecs_safeguards.codec.SafeguardsCodec] to encode
    the wrapped codec's encoded data and any safeguards-quantized corrections.
    """

    for_codec: None | dict | Codec = None
    """
    Lossless codec (configuration) that is applied to wrapped codec's encoding.
    
    By default, no further lossless encoding is applied.
    """

    for_safeguards: dict | Codec = field(
        default_factory=_default_lossless_for_safeguards,
    )
    """
    Lossless codec (configuration) that is applied to the safeguard-quantized
    corrections.
    
    The default is considered an implementation detail.
    """

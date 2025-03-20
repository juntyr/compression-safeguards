from dataclasses import dataclass, field
from io import BytesIO
from typing_extensions import Buffer  # MSPV 3.12

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from dahuffman import HuffmanCodec
from dahuffman.huffmancodec import _EOF
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .intervals import _as_bits

_LOSSLESS_VERSION: str = "0.1.x"


class DeltaHuffmanCodec(Codec):
    __slots__ = ()

    codec_id: str = "safeguards.lossless.delta_huffman"  # type: ignore

    def encode(self, buf: Buffer) -> Buffer:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape
        a = _as_bits(a.flatten())

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

        # message: marker dtype shape table encoded
        message = [marker]

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

        marker, b = b[0], b[1:]

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
        huffman = HuffmanCodec(table)

        decoded = np.array(huffman.decode(b_io.read()))

        if marker != 0:
            decoded = _as_bits(decoded)
            decoded = np.cumsum(decoded, dtype=decoded.dtype)
            decoded = decoded.view(dtype)

        return decoded.reshape(shape)  # type: ignore


numcodecs.registry.register_codec(DeltaHuffmanCodec)


@dataclass
class Lossless:
    for_codec: None | dict | Codec = None
    for_safeguards: dict | Codec = field(
        default_factory=lambda: CodecStack(
            DeltaHuffmanCodec(),
            numcodecs.zstd.Zstd(level=3),
        )
    )

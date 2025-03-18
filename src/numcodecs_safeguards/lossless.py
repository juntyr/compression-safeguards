import pickle
from dataclasses import dataclass, field
from io import BytesIO
from typing_extensions import Buffer  # MSPV 3.12

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from dahuffman import HuffmanCodec
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

_LOSSLESS_VERSION: str = "0.1.x"


class DeltaHuffmanCodec(Codec):
    __slots__ = ()

    codec_id: str = "safeguards.lossless.delta_huffman"  # type: ignore

    def encode(self, buf: Buffer) -> Buffer:
        a = numcodecs.compat.ensure_ndarray(buf).flatten()

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

        # FIXME: use safe format
        table = pickle.dumps(huffman.get_code_table())

        return marker + varint.encode(len(table)) + table + encoded

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        marker, b = b[0], b[1:]

        b_io = BytesIO(b)

        table_len = varint.decode_stream(b_io)
        # FIXME: use safe format
        table = pickle.loads(b[b_io.tell() : b_io.tell() + table_len])
        huffman = HuffmanCodec(table)

        decoded = np.array(huffman.decode(b[b_io.tell() + table_len :]))

        if marker != 0:
            decoded = np.cumsum(decoded, dtype=decoded.dtype)

        return decoded  # type: ignore


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

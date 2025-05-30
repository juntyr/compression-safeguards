"""
Helper classes for encoding n-dimensional arrays as bytes.
"""

__all__ = ["BytesCodec"]

from io import BytesIO

import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12


class BytesCodec(Codec):
    __slots__ = ()

    codec_id: str = "safeguards.bytes"  # type: ignore

    def encode(self, buf: Buffer) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        # message: dtype shape bytes
        message = []

        message.append(varint.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(varint.encode(len(shape)))
        for s in shape:
            message.append(varint.encode(s))

        # FIXME: what about endianness
        message.append(a.tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))

        shape = tuple(
            varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
        )

        # FIXME: what about endianness
        decoded = np.frombuffer(
            b_io.read(np.prod(shape) * dtype.itemsize),
            dtype=dtype,
            count=np.prod(shape),
        )

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore


numcodecs.registry.register_codec(BytesCodec)

"""
Helper classes for lossless encoding for the codec with safeguards.
"""

__all__ = ["Lossless"]

from dataclasses import dataclass, field

import numcodecs.compat
import numcodecs.registry
import numpy as np
from compression_safeguards.cast import as_bits
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack
from numcodecs_huffman import HuffmanCodec
from typing_extensions import Buffer  # MSPV 3.12


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
    the wrapped codec's encoded data and any safeguards-computed corrections.
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
    Lossless codec (configuration) that is applied to the safeguard-computed
    corrections.
    
    The default is considered an implementation detail.
    """

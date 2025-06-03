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
from numcodecs_combinators.best import PickBestCodec
from numcodecs_combinators.stack import CodecStack
from numcodecs_huffman import HuffmanCodec
from typing_extensions import Buffer  # MSPV 3.12


class BinaryDeltaCodec(Codec):
    __slots__ = ()

    codec_id: str = "safeguards.lossless.binary_delta"  # type: ignore

    def encode(self, buf: Buffer) -> Buffer:
        a = numcodecs.compat.ensure_ndarray(buf)
        a_bits = as_bits(a.flatten())

        a_bits_delta = a_bits.copy()
        a_bits_delta[1:] = np.diff(a_bits)

        a_delta = a_bits_delta.view(a.dtype).reshape(a.shape)

        return a_delta

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        a_delta = numcodecs.compat.ensure_ndarray(buf)
        a_bits_delta = as_bits(a_delta.flatten())

        a_bits = np.cumsum(a_bits_delta, dtype=a_bits_delta.dtype)

        a = a_bits.view(a_delta.dtype, a_delta.shape)

        return numcodecs.compat.ndarray_copy(a, out)  # type: ignore


numcodecs.registry.register_codec(BinaryDeltaCodec)


def _default_lossless_for_safeguards():
    return CodecStack(
        PickBestCodec(
            HuffmanCodec(),
            CodecStack(BinaryDeltaCodec(), HuffmanCodec()),
        ),
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

"""
Helper classes for lossless encoding for the codec with safeguards.
"""

__all__ = ["Lossless"]

from dataclasses import dataclass, field

import numcodecs
from compression_safeguards.utils.typing import JSON
from numcodecs.abc import Codec
from numcodecs_bitmap_index import BitmapIndexCodec
from numcodecs_combinators.best import PickBestCodec
from numcodecs_combinators.framed import FramedCodecStack
from numcodecs_combinators.stack import CodecStack
from numcodecs_delta import BinaryDeltaCodec
from numcodecs_shuffle import TypedByteShuffleCodec
from numcodecs_tokenize import TokenizeCodec


def _default_lossless_for_safeguards() -> Codec:
    return PickBestCodec(
        CodecStack(
            TokenizeCodec(),
            BitmapIndexCodec(max_bitmaps=1),
            TypedByteShuffleCodec(),
            FramedCodecStack(numcodecs.zstd.Zstd(level=3)),
        ),
        CodecStack(
            BinaryDeltaCodec(),
            TokenizeCodec(),
            BitmapIndexCodec(max_bitmaps=1),
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

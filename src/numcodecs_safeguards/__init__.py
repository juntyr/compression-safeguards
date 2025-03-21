"""
# Fearless lossy compression with `numcodecs-safeguards`

Lossy compression can be scary as valuable information may be lost.

This package provides the
[`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] adapter and several
[`Safeguards`][numcodecs_safeguards.Safeguards] that can be applied to *any*
existing (lossy) compressor to *guarantee* that certain properties about the
compression error are upheld.

Note that the wrapped compressor is treated as a blackbox and the decompressed
data is postprocessed to re-establish the properties, if necessary.

By using the [`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] adapter,
badly behaving lossy compressors become safe to use, at the cost of potentially
less efficient compression, and lossy compression can be applied without fear.
"""

__all__ = ["Safeguards", "SafeguardsCodec", "SafeguardsQuantizer"]

from .codec import SafeguardsCodec
from .safeguards import Safeguards
from .quantizer import SafeguardsQuantizer

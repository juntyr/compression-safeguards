"""
# Fearless lossy compression with `numcodecs-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

By using safeguards to ensure your safety requirements, lossy compression can
be applied safely and *without fear*.

## Safeguards for users of lossy compression

This package provides the
[`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] adapter that can be
wrapped around *any* existing (lossy) compressor to *guarantee* that certain
properties about the data are upheld.

Note that the wrapped compressor is treated as a blackbox and the decompressed
data is postprocessed to re-establish the properties, if necessary.

By using the [`SafeguardsCodec`][numcodecs_safeguards.SafeguardsCodec] adapter,
badly behaving lossy compressors become safe to use, at the cost of potentially
less efficient compression, and lossy compression can be applied *without
fear*.
"""

__all__ = ["SafeguardsCodec"]

from .codec import SafeguardsCodec

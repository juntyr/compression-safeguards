"""
# Fearless lossy compression with `numcodecs-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

This package provides several [`Safeguards`][numcodecs_safeguards.Safeguards]
to express *your* requirements for lossy compression to be safe to use and to
*guarantee* that they are upheld by lossy compression.

By using safeguards to ensure your safety requirements, lossy compression can
be applied safely and *without fear*.

## (a) Safeguards for users of lossy compression

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

## (b) Safeguards for developers of lossy compressors

Safeguards can fill the role of a quantizer, which is part of many (predictive)
(error-bounded) compressors. If you currently use e.g. a linear quantizer module
in your compressor to provide an absolute error bound, you could replace it with
the [`SafeguardsQuantizer`][numcodecs_safeguards.quantizer.SafeguardsQuantizer],
which provides a larger selection of safeguards that your compressor can then
guarantee.
"""

__all__ = ["Safeguards", "SafeguardsCodec"]

from .codec import SafeguardsCodec
from .safeguards import Safeguards

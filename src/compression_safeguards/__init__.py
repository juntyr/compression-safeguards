"""
# Fearless lossy compression with `compression-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

This package provides several [`Safeguards`][compression_safeguards.Safeguards]
to express *your* requirements for lossy compression to be safe to use and to
*guarantee* that they are upheld by lossy compression.

The [`SafeguardsCollection`][compression_safeguards.SafeguardsCollection] can
be used to compute and apply the required correction to lossy-compressed data
so that it satisfies the safeguards' safety guarantees.

By using safeguards to ensure your safety requirements, lossy compression can
be applied safely and *without fear*.
"""

__all__ = ["Safeguards", "SafeguardsCollection"]

from .collection import SafeguardsCollection
from .safeguards import Safeguards

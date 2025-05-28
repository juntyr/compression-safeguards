"""
# Fearless lossy compression with `compression-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

This package provides several [`Safeguards`][compression_safeguards.Safeguards]
to express *your* requirements for lossy compression to be safe to use and to
*guarantee* that they are upheld by lossy compression.

By using safeguards to ensure your safety requirements, lossy compression can
be applied safely and *without fear*.
"""

__all__ = ["Safeguards"]

from .safeguards import Safeguards

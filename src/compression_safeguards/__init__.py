"""
# Fearless lossy compression with `compression-safeguards`

Lossy compression can be *scary* as valuable information or features of the
data may be lost.

This package provides several
[`SafeguardKind`][compression_safeguards.SafeguardKind]s to express *your*
requirements for lossy compression to be safe to use and to *guarantee* that
they are upheld by lossy compression.

The [`Safeguards`][compression_safeguards.Safeguards] can be used to compute
and apply the required correction to lossy-compressed data so that it satisfies
the safety guarantees of a set of
[`Safeguard`][compression_safeguards.safeguards.abc.Safeguard]s.

By using safeguards to ensure your safety requirements, lossy compression can
be applied safely and *without fear*.
"""

__all__ = ["Safeguards", "SafeguardKind"]

from .api import Safeguards
from .safeguards import SafeguardKind
